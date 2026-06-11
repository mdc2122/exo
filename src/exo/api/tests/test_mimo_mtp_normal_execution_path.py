# pyright: reportUnknownLambdaType=false, reportUnknownArgumentType=false
from __future__ import annotations

import json
import sys
import types
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from loguru import logger

from exo.api.adapters import chat_completions
from exo.api.types import ChatCompletionMessage, ChatCompletionRequest
from exo.shared.models.model_cards import MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.commands import TextGeneration
from exo.shared.types.common import CommandId
from exo.shared.types.state import State
from exo.shared.types.worker.instances import MlxJacclInstance

if TYPE_CHECKING:
    from exo.api.main import API


def _install_bootstrap_logger_stub() -> None:
    """Avoid importing MLX patches while this API-boundary test imports API."""
    bootstrap_stub = types.ModuleType("exo.worker.runner.bootstrap")
    cast(Any, bootstrap_stub).logger = logger
    sys.modules.setdefault("exo.worker.runner.bootstrap", bootstrap_stub)


def _api_with_empty_topology() -> "API":
    _install_bootstrap_logger_stub()
    from exo.api.main import API

    api = object.__new__(API)
    api.state = State()
    return api


def _mark_mimo_mtp_sidecar_ready(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    ready_sidecar_probe = type(
        "ReadySidecarProbe",
        (),
        {"status": "ready", "path": path},
    )()
    monkeypatch.setattr(
        chat_completions,
        "probe_mimo_mtp_sidecar",
        lambda _path: ready_sidecar_probe,
    )


async def test_normal_chat_completion_without_enabled_mtp_uses_non_mtp_text_generation_path() -> (
    None
):
    api = _api_with_empty_topology()
    preview_response = await api.get_placement_previews(
        MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID
    )
    preview = next(
        preview
        for preview in preview_response.previews
        if preview.instance is not None and preview.sharding.value == "Tensor"
    )
    assert preview.instance is not None
    api.state = api.state.model_copy(
        update={"instances": {preview.instance.instance_id: preview.instance}}
    )

    async def stream(_command_id: CommandId) -> AsyncGenerator[TokenChunk, None]:
        yield TokenChunk(
            model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
            text="ok",
            token_id=1,
            usage=None,
            finish_reason="stop",
            stats=None,
        )

    sent_commands: list[object] = []

    async def fake_send(command: object) -> None:
        sent_commands.append(command)

    object.__setattr__(api, "_token_chunk_stream", stream)
    object.__setattr__(api, "_send", fake_send)

    response = await api.chat_completions(
        ChatCompletionRequest(
            model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
            messages=[
                ChatCompletionMessage(role="user", content="Summarize MiMo Pro.")
            ],
            max_tokens=1,
        )
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 200
    response_body = (
        response.body.tobytes()
        if isinstance(response.body, memoryview)
        else response.body
    )
    assert json.loads(response_body)["choices"][0]["message"]["content"] == "ok"
    assert len(sent_commands) == 1
    command = sent_commands[0]
    assert isinstance(command, TextGeneration)
    assert command.task_params.bench is False
    assert command.task_params.stream is False
    assert command.task_params.input[0].content == "Summarize MiMo Pro."
    assert command.task_params.mimo_mtp_fastpath is None
    task_param_payload = command.task_params.model_dump()
    assert task_param_payload.get("mimo_mtp_fastpath") is None
    assert "mimo_mtp_depth" not in task_param_payload
    assert "mimo_mtp_sidecar_path" not in task_param_payload
    assert "mimo_mtp_fail_closed" not in task_param_payload


async def test_normal_chat_mtp_fail_open_dispatches_ar_without_mtp_intent() -> None:
    api = _api_with_empty_topology()
    preview_response = await api.get_placement_previews(
        MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID
    )
    preview = next(
        preview
        for preview in preview_response.previews
        if preview.instance is not None and preview.sharding.value == "Tensor"
    )
    assert preview.instance is not None
    api.state = api.state.model_copy(
        update={"instances": {preview.instance.instance_id: preview.instance}}
    )

    async def stream(_command_id: CommandId) -> AsyncGenerator[TokenChunk, None]:
        yield TokenChunk(
            model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
            text="ok",
            token_id=1,
            usage=None,
            finish_reason="stop",
            stats=None,
        )

    sent_commands: list[object] = []

    async def fake_send(command: object) -> None:
        sent_commands.append(command)

    object.__setattr__(api, "_token_chunk_stream", stream)
    object.__setattr__(api, "_send", fake_send)

    response = await api.chat_completions(
        ChatCompletionRequest(
            model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
            messages=[ChatCompletionMessage(role="user", content="Hello")],
            max_tokens=1,
            mimo_mtp_fastpath=True,
            mimo_mtp_depth=1,
            mimo_mtp_fail_closed=False,
        )
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 200
    assert len(sent_commands) == 1
    command = sent_commands[0]
    assert isinstance(command, TextGeneration)
    assert command.task_params.mimo_mtp_fastpath is None
    task_param_payload = command.task_params.model_dump()
    assert "mimo_mtp_fastpath" not in task_param_payload


async def test_experimental_mtp_request_fields_are_rejected_by_default_when_native_runtime_guard_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("EXO_MIMO_MTP_NATIVE_RUNTIME", raising=False)
    api = _api_with_empty_topology()
    preview_response = await api.get_placement_previews(
        MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID
    )
    preview = next(
        preview
        for preview in preview_response.previews
        if preview.instance is not None and preview.sharding.value == "Tensor"
    )
    assert preview.instance is not None
    api.state = api.state.model_copy(
        update={"instances": {preview.instance.instance_id: preview.instance}}
    )

    async def fail_send(_: object) -> None:
        raise AssertionError(
            "default-disabled MTP guard must reject before dispatching generation"
        )

    object.__setattr__(api, "_send", fail_send)
    sidecar_path = tmp_path / "model_mtp.safetensors"
    sidecar_path.write_bytes(b"contract test sidecar placeholder")
    _mark_mimo_mtp_sidecar_ready(monkeypatch, sidecar_path)

    with pytest.raises(HTTPException) as exc_info:
        await api.chat_completions(
            ChatCompletionRequest.model_validate(
                {
                    "model": str(MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID),
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 1,
                    "mimo_mtp_fastpath": True,
                    "mimo_mtp_depth": 1,
                    "mimo_mtp_sidecar_path": str(sidecar_path),
                }
            )
        )

    assert exc_info.value.status_code == 400
    raw_detail = cast(object, exc_info.value.detail)
    assert isinstance(raw_detail, dict)
    detail = cast(dict[str, object], raw_detail)
    assert detail["error"] == "mimo_mtp_execution_backend_unwired"
    assert detail["mtp_enabled"] is False
    assert detail["requested_mtp_depth"] == 1
    assert detail["mtp_disable_reason"] == "mimo_mtp_execution_backend_unwired"
    assert "EXO_MIMO_MTP_NATIVE_RUNTIME" in str(detail["message"])


async def test_normal_chat_mtp_fail_closed_rejects_before_send_when_backend_unwired(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api = _api_with_empty_topology()
    preview_response = await api.get_placement_previews(
        MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID
    )
    preview = next(
        preview
        for preview in preview_response.previews
        if preview.instance is not None and preview.sharding.value == "Tensor"
    )
    assert preview.instance is not None
    api.state = api.state.model_copy(
        update={"instances": {preview.instance.instance_id: preview.instance}}
    )

    async def fail_send(_: object) -> None:
        raise AssertionError("MTP fail-closed guard must reject before _send")

    object.__setattr__(api, "_send", fail_send)
    sidecar_path = tmp_path / "model_mtp.safetensors"
    sidecar_path.write_bytes(b"not-a-real-sidecar; backend is intentionally unwired")
    _mark_mimo_mtp_sidecar_ready(monkeypatch, sidecar_path)

    try:
        await api.chat_completions(
            ChatCompletionRequest(
                model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
                messages=[ChatCompletionMessage(role="user", content="Hello")],
                max_tokens=1,
                mimo_mtp_fastpath=True,
                mimo_mtp_depth=1,
                mimo_mtp_sidecar_path=str(sidecar_path),
            )
        )
    except HTTPException as exc:
        assert exc.status_code == 400
        raw_detail = cast(object, exc.detail)
        assert isinstance(raw_detail, dict)
        detail = cast(dict[str, object], raw_detail)
        assert detail["error"] == "mimo_mtp_execution_backend_unwired"
        assert detail["accepted_execution_path"] == "unwired"
        assert detail["mtp_enabled"] is False
        assert detail["mtp_disable_reason"] == "mimo_mtp_execution_backend_unwired"
        assert detail["mtp_execution_state"] == "unwired_execution"
        assert detail.get("mode") != "mtp"
        assert detail.get("mtp_execution_state") != "successful_mtp"
    else:
        raise AssertionError("expected unwired MTP fail-closed HTTPException")


async def test_mtp_request_on_incompatible_backend_fails_closed_before_send(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("EXO_MIMO_MTP_NATIVE_RUNTIME", "1")
    api = _api_with_empty_topology()
    preview_response = await api.get_placement_previews(
        MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID
    )
    preview = next(
        preview
        for preview in preview_response.previews
        if preview.instance is not None and preview.sharding.value == "Tensor"
    )
    assert preview.instance is not None
    incompatible_instance = MlxJacclInstance(
        instance_id=preview.instance.instance_id,
        shard_assignments=preview.instance.shard_assignments,
        jaccl_devices=[],
        jaccl_coordinators={},
    )
    api.state = api.state.model_copy(
        update={"instances": {incompatible_instance.instance_id: incompatible_instance}}
    )

    async def fail_send(_: object) -> None:
        raise AssertionError(
            "incompatible MTP backend must reject before dispatching generation"
        )

    object.__setattr__(api, "_send", fail_send)
    sidecar_path = tmp_path / "model_mtp.safetensors"
    sidecar_path.write_bytes(b"runtime guard enabled; backend is incompatible")
    _mark_mimo_mtp_sidecar_ready(monkeypatch, sidecar_path)

    with pytest.raises(HTTPException) as exc_info:
        await api.chat_completions(
            ChatCompletionRequest(
                model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
                messages=[ChatCompletionMessage(role="user", content="Hello")],
                max_tokens=1,
                mimo_mtp_fastpath=True,
                mimo_mtp_depth=1,
                mimo_mtp_sidecar_path=str(sidecar_path),
                mimo_mtp_fail_closed=True,
            )
        )

    assert exc_info.value.status_code == 400
    raw_detail = cast(object, exc_info.value.detail)
    assert isinstance(raw_detail, dict)
    detail = cast(dict[str, object], raw_detail)
    assert detail["error"] == "mimo_mtp_backend_incompatible"
    assert detail["accepted_execution_path"] == "rejected"
    assert detail["mtp_enabled"] is False
    assert detail["requested_mtp_depth"] == 1
    assert detail["mtp_depth"] is None
    assert detail["mtp_disable_reason"] == "unsupported_backend"
    assert detail["mtp_execution_state"] == "unwired_execution"
    assert detail["backend_source"] == "mlx_jaccl"
    assert "mlx_ring" in str(detail["supported_backends"])
    assert "backend mlx_jaccl" in str(detail["message"])


async def test_normal_chat_mtp_runtime_guard_allows_worker_dispatch_with_mtp_intent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("EXO_MIMO_MTP_NATIVE_RUNTIME", "1")
    api = _api_with_empty_topology()
    preview_response = await api.get_placement_previews(
        MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID
    )
    preview = next(
        preview
        for preview in preview_response.previews
        if preview.instance is not None and preview.sharding.value == "Tensor"
    )
    assert preview.instance is not None
    api.state = api.state.model_copy(
        update={"instances": {preview.instance.instance_id: preview.instance}}
    )

    async def stream(_command_id: CommandId) -> AsyncGenerator[TokenChunk, None]:
        yield TokenChunk(
            model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
            text="ok",
            token_id=1,
            usage=None,
            finish_reason="stop",
            stats=None,
        )

    sent_commands: list[object] = []

    async def fake_send(command: object) -> None:
        sent_commands.append(command)

    object.__setattr__(api, "_token_chunk_stream", stream)
    object.__setattr__(api, "_send", fake_send)
    sidecar_path = tmp_path / "model_mtp.safetensors"
    sidecar_path.write_bytes(b"runtime guard enabled; worker validates real sidecar")
    _mark_mimo_mtp_sidecar_ready(monkeypatch, sidecar_path)

    response = await api.chat_completions(
        ChatCompletionRequest(
            model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
            messages=[ChatCompletionMessage(role="user", content="Hello")],
            max_tokens=1,
            mimo_mtp_fastpath=True,
            mimo_mtp_depth=1,
            mimo_mtp_sidecar_path=str(sidecar_path),
            mimo_mtp_fail_closed=True,
        )
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 200
    assert len(sent_commands) == 1
    command = sent_commands[0]
    assert isinstance(command, TextGeneration)
    mtp_params = command.task_params.mimo_mtp_fastpath
    assert mtp_params is not None
    assert mtp_params.enabled is True
    assert mtp_params.depth == 1
    assert mtp_params.sidecar_path == str(sidecar_path)
    assert mtp_params.fail_closed is True
