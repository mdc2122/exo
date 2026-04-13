from unittest.mock import MagicMock

import pytest

from exo.shared.models.model_cards import ModelId
from exo.worker.runner.llm_inference import runner as runner_module
from exo.worker.runner.llm_inference.runner import Builder


class _FakeGroup:
    def rank(self) -> int:
        return 0


def _builder_with_group(*, group: object | None) -> Builder:
    tokenizer = MagicMock()
    tokenizer.has_tool_calling = False
    tokenizer.tool_call_start = None
    tokenizer.tool_call_end = None
    tokenizer.tool_parser = None

    return Builder(
        model_id=ModelId("mlx-community/GLM-5.1-8b-crit-6b-exp"),
        event_sender=MagicMock(),
        cancel_receiver=MagicMock(),
        inference_model=MagicMock(),
        tokenizer=tokenizer,
        group=group,  # type: ignore[arg-type]
    )


def test_builder_uses_sequential_generator_for_clustered_turboquant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXO_NO_BATCH", raising=False)
    monkeypatch.setattr(runner_module, "TURBOQUANT_KV_BITS", 4)

    builder = _builder_with_group(group=_FakeGroup())

    generator = builder.build()

    assert isinstance(generator, runner_module.SequentialGenerator)


def test_builder_keeps_batch_generator_for_single_node_turboquant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXO_NO_BATCH", raising=False)
    monkeypatch.setattr(runner_module, "TURBOQUANT_KV_BITS", 4)

    builder = _builder_with_group(group=None)

    generator = builder.build()

    assert isinstance(generator, runner_module.BatchGenerator)
