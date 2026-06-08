import os

import pytest
from pytest import MonkeyPatch

from exo.shared.types.common import ModelId
from exo.worker.engines.mlx.generator import generate as generate_module

pytestmark = pytest.mark.skipif(
    os.environ.get("EXO_TEST_TURBOQUANT") != "1",
    reason="TurboQuant optional accelerator tests require EXO_TEST_TURBOQUANT=1",
)


def test_warmup_inference_skips_generation_when_turboquant_active(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(generate_module, "TURBOQUANT_KV_BITS", 4)

    def fake_apply_chat_template(**_: object) -> str:
        return "warmup"

    monkeypatch.setattr(
        generate_module, "apply_chat_template", fake_apply_chat_template
    )

    barrier_calls: list[object | None] = []

    def fake_barrier(group: object | None) -> None:
        barrier_calls.append(group)

    monkeypatch.setattr(
        generate_module,
        "mx_barrier",
        fake_barrier,
    )

    mlx_generate_calls = {"count": 0}

    def fail_if_called(**_: object) -> object:
        mlx_generate_calls["count"] += 1
        raise AssertionError("mlx_generate should be skipped during TurboQuant warmup")

    monkeypatch.setattr(generate_module, "mlx_generate", fail_if_called)

    check_for_cancel_every = generate_module.warmup_inference(
        model=object(),  # type: ignore[arg-type]
        tokenizer=object(),  # type: ignore[arg-type]
        group=None,
        model_id=ModelId("mlx-community/GLM-5.1-8b-crit-6b-exp"),
    )

    assert check_for_cancel_every == 50
    assert barrier_calls == [None, None]
    assert mlx_generate_calls["count"] == 0
