from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest

import exo.worker.runner.llm_inference.batch_generator as batch_generator_module
from exo.shared.types.chunks import PrefillProgressChunk
from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.events import ChunkGenerated
from exo.shared.types.tasks import TaskId, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    MimoMtpFastpathParams,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.mlx.generator.batch_generate import ExoBatchGenerator
from exo.worker.runner.llm_inference.batch_generator import (
    BatchGenerator,
    SequentialGenerator,
)


class _CancelReceiver:
    def collect(self) -> list[TaskId]:
        return []


class _EventSender:
    def __init__(self) -> None:
        self.events: list[object] = []

    def send(self, event: object) -> None:
        self.events.append(event)


class _FakeBatchEngine:
    def __init__(self) -> None:
        self.submitted_params: list[TextGenerationTaskParams] = []

    @property
    def has_work(self) -> bool:
        return False

    def submit(
        self,
        *,
        task_params: TextGenerationTaskParams,
        prompt: str,
        on_prefill_progress: Callable[[int, int], None],
        distributed_prompt_progress_callback: object,
        on_generation_token: object,
    ) -> int:
        on_prefill_progress(1, 3)
        self.submitted_params.append(task_params)
        return 7

    def cancel(self, _uids: list[int]) -> None:
        pass

    def close(self) -> None:
        pass


def _task(
    *,
    model: str = "kernelpool/MiMo-V2.5-Pro-6bit",
    mtp: MimoMtpFastpathParams | None = None,
    bench: bool = True,
) -> TextGeneration:
    return TextGeneration(
        task_id=TaskId("12345678-1234-1234-1234-123456789abc"),
        command_id=CommandId("cmd-1"),
        instance_id=InstanceId("instance-1"),
        task_params=TextGenerationTaskParams(
            model=ModelId(model),
            input=[InputMessage(role="user", content="hello")],
            max_output_tokens=4,
            temperature=0.0,
            bench=bench,
            mimo_mtp_fastpath_params=mtp,
        ),
    )


def _sequential_generator(
    event_sender: _EventSender | None = None,
) -> SequentialGenerator:
    return SequentialGenerator(
        model=MagicMock(),
        tokenizer=MagicMock(),
        group=None,
        kv_prefix_cache=None,
        tool_parser=None,
        model_id=ModelId("kernelpool/MiMo-V2.5-Pro-6bit"),
        device_rank=0,
        cancel_receiver=_CancelReceiver(),  # type: ignore[arg-type]
        event_sender=event_sender or _EventSender(),  # type: ignore[arg-type]
    )


def _batch_generator(
    fake_engine: _FakeBatchEngine,
    event_sender: _EventSender | None = None,
) -> BatchGenerator:
    generator = BatchGenerator(
        model=MagicMock(),
        tokenizer=MagicMock(),
        group=None,
        kv_prefix_cache=None,
        tool_parser=None,
        model_id=ModelId("kernelpool/MiMo-V2.5-Pro-6bit"),
        device_rank=0,
        cancel_receiver=_CancelReceiver(),  # type: ignore[arg-type]
        event_sender=event_sender or _EventSender(),  # type: ignore[arg-type]
    )
    generator._mlx_gen = cast(ExoBatchGenerator, cast(object, fake_engine))  # pyright: ignore[reportPrivateUsage]
    return generator


def _noop_check_for_debug_prompts(_params: TextGenerationTaskParams) -> None:
    return None


def _fake_apply_chat_template(
    _tokenizer: object,
    _task_params: TextGenerationTaskParams,
) -> str:
    return "prompt"


def _patch_common_generation_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        batch_generator_module,
        "_check_for_debug_prompts",
        _noop_check_for_debug_prompts,
    )
    monkeypatch.setattr(
        batch_generator_module, "apply_chat_template", _fake_apply_chat_template
    )


def test_sequential_default_request_uses_existing_ar_generator_without_mtp_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_common_generation_dependencies(monkeypatch)
    captured: dict[str, object] = {}
    event_sender = _EventSender()

    def reject_mtp_guard_for_default_request(
        _task_params: TextGenerationTaskParams,
        **_kwargs: object,
    ) -> object:
        raise AssertionError("default AR request must not enter MTP guard evaluation")

    def fake_mlx_generate(**kwargs: object) -> Generator[GenerationResponse]:
        captured.update(kwargs)
        on_prefill_progress = cast(
            Callable[[int, int], None], kwargs["on_prefill_progress"]
        )
        on_prefill_progress(2, 5)
        yield GenerationResponse(text="ok", token=1, finish_reason="stop", usage=None)

    monkeypatch.setattr(
        batch_generator_module,
        "evaluate_mimo_mtp_worker_fastpath",
        reject_mtp_guard_for_default_request,
    )
    monkeypatch.setattr(batch_generator_module, "mlx_generate", fake_mlx_generate)
    task = _task(mtp=None)

    result = next(_sequential_generator(event_sender)._build_generator(task))  # pyright: ignore[reportPrivateUsage]

    assert result.text == "ok"
    assert captured["task"] is task.task_params
    captured_task = cast(TextGenerationTaskParams, captured["task"])
    assert captured_task.mimo_mtp_fastpath is None
    assert len(event_sender.events) == 1
    event = event_sender.events[0]
    assert isinstance(event, ChunkGenerated)
    assert event.command_id == task.command_id
    assert isinstance(event.chunk, PrefillProgressChunk)
    assert event.chunk.processed_tokens == 2
    assert event.chunk.total_tokens == 5


def test_sequential_fail_open_mtp_fallback_strips_mtp_before_ar_generator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_common_generation_dependencies(monkeypatch)
    captured: dict[str, object] = {}

    def fake_mlx_generate(**kwargs: object) -> Generator[GenerationResponse]:
        captured.update(kwargs)
        yield GenerationResponse(
            text="fallback", token=1, finish_reason="stop", usage=None
        )

    monkeypatch.setattr(batch_generator_module, "mlx_generate", fake_mlx_generate)
    task = _task(
        mtp=MimoMtpFastpathParams(
            enabled=True,
            depth=2,
            sidecar_path=str(tmp_path / "missing-model_mtp.safetensors"),
            fail_closed=False,
        )
    )

    result = next(_sequential_generator()._build_generator(task))  # pyright: ignore[reportPrivateUsage]

    assert result.text == "fallback"
    assert captured["task"] is not task.task_params
    captured_task = cast(TextGenerationTaskParams, captured["task"])
    assert captured_task.mimo_mtp_fastpath is None


def test_sequential_fail_closed_missing_sidecar_rejection_does_not_call_ar_generator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_common_generation_dependencies(monkeypatch)

    def fail_if_called(**_kwargs: object) -> Generator[GenerationResponse]:
        raise AssertionError(
            "fail-closed missing-sidecar MTP request must not dispatch AR generation"
        )
        yield GenerationResponse(
            text="unreachable", token=0, finish_reason="stop", usage=None
        )

    monkeypatch.setattr(batch_generator_module, "mlx_generate", fail_if_called)
    missing_sidecar = tmp_path / "missing-model_mtp.safetensors"
    task = _task(
        mtp=MimoMtpFastpathParams(
            enabled=True,
            depth=1,
            sidecar_path=str(missing_sidecar),
            fail_closed=True,
        ),
    )

    result = next(_sequential_generator()._build_generator(task))  # pyright: ignore[reportPrivateUsage]

    assert result.finish_reason == "error"
    assert "missing_sidecar" in result.text
    assert "reject_reason=missing_sidecar" in result.text


def test_sequential_fail_closed_mtp_rejection_does_not_call_ar_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_common_generation_dependencies(monkeypatch)

    def fail_if_called(**_kwargs: object) -> Generator[GenerationResponse]:
        raise AssertionError("fail-closed MTP request must not dispatch AR generation")
        yield GenerationResponse(
            text="unreachable", token=0, finish_reason="stop", usage=None
        )

    monkeypatch.setattr(batch_generator_module, "mlx_generate", fail_if_called)
    task = _task(
        model="other/model",
        mtp=MimoMtpFastpathParams(enabled=True, depth=1, fail_closed=True),
    )

    result = next(_sequential_generator()._build_generator(task))  # pyright: ignore[reportPrivateUsage]

    assert result.finish_reason == "error"
    assert "unsupported_model" in result.text


def test_batch_default_request_submits_existing_ar_task_params_without_mtp_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_common_generation_dependencies(monkeypatch)
    fake_engine = _FakeBatchEngine()
    event_sender = _EventSender()

    def reject_mtp_guard_for_default_request(
        _task_params: TextGenerationTaskParams,
        **_kwargs: object,
    ) -> object:
        raise AssertionError("default AR request must not enter MTP guard evaluation")

    monkeypatch.setattr(
        batch_generator_module,
        "evaluate_mimo_mtp_worker_fastpath",
        reject_mtp_guard_for_default_request,
    )
    task = _task(mtp=None)

    uid = _batch_generator(fake_engine, event_sender)._start_task(task)  # pyright: ignore[reportPrivateUsage]

    assert uid == 7
    assert fake_engine.submitted_params == [task.task_params]
    assert fake_engine.submitted_params[0].mimo_mtp_fastpath is None
    assert len(event_sender.events) == 1
    event = event_sender.events[0]
    assert isinstance(event, ChunkGenerated)
    assert event.command_id == task.command_id
    assert isinstance(event.chunk, PrefillProgressChunk)
    assert event.chunk.processed_tokens == 1
    assert event.chunk.total_tokens == 3


def test_batch_fail_open_mtp_fallback_strips_mtp_before_ar_submit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_common_generation_dependencies(monkeypatch)
    fake_engine = _FakeBatchEngine()
    task = _task(
        mtp=MimoMtpFastpathParams(
            enabled=True,
            depth=1,
            sidecar_path=str(tmp_path / "missing-model_mtp.safetensors"),
            fail_closed=False,
        )
    )

    uid = _batch_generator(fake_engine)._start_task(task)  # pyright: ignore[reportPrivateUsage]

    assert uid == 7
    assert len(fake_engine.submitted_params) == 1
    assert fake_engine.submitted_params[0] is not task.task_params
    assert fake_engine.submitted_params[0].mimo_mtp_fastpath is None
