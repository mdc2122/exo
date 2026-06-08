from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pytest
from pydantic import ValidationError

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import (
    InputMessage,
    MimoMtpFastpathParams,
    TextGenerationTaskParams,
)
from exo.worker.engines.mlx.mimo_mtp_fast.sidecar_contract import (
    MIMO_MTP_LAYER_COUNT,
    MIMO_MTP_REQUIRED_SUFFIXES,
    official_mimo_mtp_key,
)
from exo.worker.engines.mlx.mimo_mtp_fast.worker_fastpath import (
    MimoMtpWorkerFastpathCache,
    MimoMtpWorkerFastpathDecision,
    evaluate_mimo_mtp_worker_fastpath,
)


class _SaveFileFn(Protocol):
    def __call__(
        self,
        tensor_dict: dict[str, np.ndarray],
        filename: str,
        metadata: dict[str, str] | None = None,
    ) -> None: ...


_safetensors_numpy = import_module("safetensors.numpy")
_save_file = cast(_SaveFileFn, _safetensors_numpy.save_file)


def _tiny_tensor_for_suffix(layer_index: int, suffix: str) -> np.ndarray:
    base_value = float(layer_index + 1)
    if suffix.endswith(".weight_scale_inv"):
        return np.full((1, 1), base_value, dtype=np.float32)
    if suffix.endswith(".weight") and suffix not in {
        "enorm.weight",
        "hnorm.weight",
        "final_layernorm.weight",
        "input_layernorm.weight",
        "pre_mlp_layernorm.weight",
    }:
        return np.full((2, 2), base_value, dtype=np.float32)
    return np.full((2,), base_value, dtype=np.float32)


def _write_synthetic_official_sidecar(path: Path) -> None:
    tensors: dict[str, np.ndarray] = {}
    for layer_index in range(MIMO_MTP_LAYER_COUNT):
        for suffix in MIMO_MTP_REQUIRED_SUFFIXES:
            tensors[official_mimo_mtp_key(layer_index, suffix)] = (
                _tiny_tensor_for_suffix(layer_index, suffix)
            )
    _save_file(tensors, str(path))


def _task_params(
    *,
    model: str = "kernelpool/MiMo-V2.5-Pro-6bit",
    mtp: MimoMtpFastpathParams | None = None,
    bench: bool = True,
) -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=ModelId(model),
        input=[InputMessage(role="user", content="hello")],
        max_output_tokens=8,
        temperature=0.0,
        bench=bench,
        mimo_mtp_fastpath_params=mtp,
    )


def test_default_request_keeps_worker_on_autoregressive_path() -> None:
    decision = evaluate_mimo_mtp_worker_fastpath(_task_params(mtp=None))

    assert decision == MimoMtpWorkerFastpathDecision(
        should_use_mtp=False,
        accepted_execution_path="ar",
        mtp_enabled=False,
        requested_depth=None,
        mtp_depth=None,
        sidecar_status=None,
        disable_reason=None,
        reject_reason=None,
        fallback_reason=None,
        telemetry={
            "accepted_execution_path": "ar",
            "mtp_enabled": False,
        },
    )


def test_non_mimo_explicit_mtp_fails_closed_before_ar_dispatch(tmp_path: Path) -> None:
    sidecar_path = tmp_path / "model_mtp.safetensors"
    _write_synthetic_official_sidecar(sidecar_path)

    decision = evaluate_mimo_mtp_worker_fastpath(
        _task_params(
            model="other/model",
            mtp=MimoMtpFastpathParams(
                enabled=True,
                depth=1,
                sidecar_path=str(sidecar_path),
                fail_closed=True,
            ),
        )
    )

    assert decision.should_use_mtp is False
    assert decision.accepted_execution_path == "rejected"
    assert decision.mtp_enabled is False
    assert decision.disable_reason == "unsupported_model"
    assert decision.error_message is not None
    assert "unsupported_model" in decision.error_message
    assert decision.telemetry["accepted_execution_path"] == "rejected"
    assert decision.telemetry["mtp_disable_reason"] == "unsupported_model"


def test_explicit_mimo_mtp_fail_closed_missing_sidecar_reports_reject_reason(
    tmp_path: Path,
) -> None:
    missing_sidecar = tmp_path / "model_mtp.safetensors"

    decision = evaluate_mimo_mtp_worker_fastpath(
        _task_params(
            mtp=MimoMtpFastpathParams(
                enabled=True,
                depth=2,
                sidecar_path=str(missing_sidecar),
                fail_closed=True,
            )
        )
    )

    assert decision.should_use_mtp is False
    assert decision.accepted_execution_path == "rejected"
    assert decision.mtp_enabled is False
    assert decision.requested_depth == 2
    assert decision.mtp_depth is None
    assert decision.sidecar_status == "missing"
    assert decision.disable_reason == "missing_sidecar"
    assert decision.reject_reason == "missing_sidecar"
    assert decision.fallback_reason is None
    assert decision.error_message is not None
    assert "reject_reason=missing_sidecar" in decision.error_message
    assert decision.telemetry["accepted_execution_path"] == "rejected"
    assert decision.telemetry["mtp_enabled"] is False
    assert decision.telemetry["requested_mtp_depth"] == 2
    assert decision.telemetry["mtp_sidecar_status"] == "missing"
    assert decision.telemetry["mtp_disable_reason"] == "missing_sidecar"
    assert decision.telemetry["mtp_reject_reason"] == "missing_sidecar"


def test_explicit_mimo_mtp_fail_open_missing_sidecar_reports_ar_fallback(
    tmp_path: Path,
) -> None:
    missing_sidecar = tmp_path / "model_mtp.safetensors"

    decision = evaluate_mimo_mtp_worker_fastpath(
        _task_params(
            mtp=MimoMtpFastpathParams(
                enabled=True,
                depth=2,
                sidecar_path=str(missing_sidecar),
                fail_closed=False,
            )
        )
    )

    assert decision.should_use_mtp is False
    assert decision.accepted_execution_path == "ar"
    assert decision.mtp_enabled is False
    assert decision.requested_depth == 2
    assert decision.mtp_depth is None
    assert decision.sidecar_status == "missing"
    assert decision.disable_reason == "missing_sidecar"
    assert decision.fallback_reason == "fail_open_missing_sidecar"
    assert decision.error_message is None
    assert decision.telemetry["accepted_execution_path"] == "ar"
    assert decision.telemetry["mtp_enabled"] is False
    assert decision.telemetry["requested_mtp_depth"] == 2
    assert decision.telemetry["mtp_sidecar_status"] == "missing"
    assert decision.telemetry["mtp_disable_reason"] == "missing_sidecar"
    assert decision.telemetry["mtp_fallback_reason"] == "fail_open_missing_sidecar"


def test_ready_sidecar_with_wired_execution_routes_to_native_mtp_seam_and_loads_once(
    tmp_path: Path,
) -> None:
    sidecar_path = tmp_path / "model_mtp.safetensors"
    _write_synthetic_official_sidecar(sidecar_path)
    cache = MimoMtpWorkerFastpathCache()

    first = evaluate_mimo_mtp_worker_fastpath(
        _task_params(
            mtp=MimoMtpFastpathParams(
                enabled=True,
                depth=3,
                sidecar_path=str(sidecar_path),
                fail_closed=True,
            )
        ),
        cache=cache,
        native_runtime_enabled=True,
        execution_path_wired=True,
    )
    second = evaluate_mimo_mtp_worker_fastpath(
        _task_params(
            mtp=MimoMtpFastpathParams(
                enabled=True,
                depth=1,
                sidecar_path=str(sidecar_path),
                fail_closed=True,
            )
        ),
        cache=cache,
        native_runtime_enabled=True,
        execution_path_wired=True,
    )

    assert first.should_use_mtp is True
    assert first.accepted_execution_path == "mimo_mtp_fastpath"
    assert first.mtp_enabled is True
    assert first.mtp_depth == 3
    assert first.sidecar_status == "ready"
    assert first.telemetry["accepted_execution_path"] == "mimo_mtp_fastpath"
    assert first.telemetry["mtp_enabled"] is True
    assert first.telemetry["mtp_depth"] == 3
    assert first.telemetry["mtp_sidecar_status"] == "ready"
    assert first.sidecar is not None
    assert second.should_use_mtp is True
    assert second.mtp_depth == 1
    assert second.sidecar is first.sidecar
    assert cache.load_count_for_path(sidecar_path) == 1


def test_ready_sidecar_with_unwired_execution_fails_closed_before_enabling_mtp(
    tmp_path: Path,
) -> None:
    sidecar_path = tmp_path / "model_mtp.safetensors"
    _write_synthetic_official_sidecar(sidecar_path)
    cache = MimoMtpWorkerFastpathCache()

    decision = evaluate_mimo_mtp_worker_fastpath(
        _task_params(
            mtp=MimoMtpFastpathParams(
                enabled=True,
                depth=2,
                sidecar_path=str(sidecar_path),
                fail_closed=True,
            )
        ),
        cache=cache,
        native_runtime_enabled=True,
    )

    assert decision.should_use_mtp is False
    assert decision.accepted_execution_path == "rejected"
    assert decision.mtp_enabled is False
    assert decision.requested_depth == 2
    assert decision.mtp_depth is None
    assert decision.sidecar_status == "ready"
    assert decision.disable_reason == "mimo_mtp_distributed_generator_unwired"
    assert decision.reject_reason == "mimo_mtp_distributed_generator_unwired"
    assert decision.fallback_reason is None
    assert decision.error_message is not None
    assert (
        "reject_reason=mimo_mtp_distributed_generator_unwired" in decision.error_message
    )
    assert decision.telemetry["accepted_execution_path"] == "rejected"
    assert decision.telemetry["mtp_enabled"] is False
    assert decision.telemetry["requested_mtp_depth"] == 2
    assert decision.telemetry["mtp_sidecar_status"] == "ready"
    assert (
        decision.telemetry["mtp_disable_reason"]
        == "mimo_mtp_distributed_generator_unwired"
    )
    assert (
        decision.telemetry["mtp_reject_reason"]
        == "mimo_mtp_distributed_generator_unwired"
    )
    assert decision.sidecar is None
    assert cache.load_count_for_path(sidecar_path) == 0


def test_ready_sidecar_with_unwired_execution_can_fail_open_to_ar_without_mtp(
    tmp_path: Path,
) -> None:
    sidecar_path = tmp_path / "model_mtp.safetensors"
    _write_synthetic_official_sidecar(sidecar_path)
    cache = MimoMtpWorkerFastpathCache()

    decision = evaluate_mimo_mtp_worker_fastpath(
        _task_params(
            mtp=MimoMtpFastpathParams(
                enabled=True,
                depth=2,
                sidecar_path=str(sidecar_path),
                fail_closed=False,
            )
        ),
        cache=cache,
        native_runtime_enabled=True,
    )

    assert decision.should_use_mtp is False
    assert decision.accepted_execution_path == "ar"
    assert decision.mtp_enabled is False
    assert decision.requested_depth == 2
    assert decision.mtp_depth is None
    assert decision.sidecar_status == "ready"
    assert decision.disable_reason == "mimo_mtp_distributed_generator_unwired"
    assert decision.reject_reason is None
    assert decision.fallback_reason == "fail_open_distributed_generator_unwired"
    assert decision.error_message is None
    assert decision.telemetry["accepted_execution_path"] == "ar"
    assert decision.telemetry["mtp_enabled"] is False
    assert decision.telemetry["requested_mtp_depth"] == 2
    assert decision.telemetry["mtp_sidecar_status"] == "ready"
    assert (
        decision.telemetry["mtp_disable_reason"]
        == "mimo_mtp_distributed_generator_unwired"
    )
    assert (
        decision.telemetry["mtp_fallback_reason"]
        == "fail_open_distributed_generator_unwired"
    )
    assert decision.sidecar is None
    assert cache.load_count_for_path(sidecar_path) == 0


def test_ready_sidecar_without_native_runtime_fails_closed_not_ar_as_mtp(
    tmp_path: Path,
) -> None:
    sidecar_path = tmp_path / "model_mtp.safetensors"
    _write_synthetic_official_sidecar(sidecar_path)

    decision = evaluate_mimo_mtp_worker_fastpath(
        _task_params(
            mtp=MimoMtpFastpathParams(
                enabled=True,
                depth=1,
                sidecar_path=str(sidecar_path),
                fail_closed=True,
            )
        ),
        cache=MimoMtpWorkerFastpathCache(),
        native_runtime_enabled=False,
    )

    assert decision.should_use_mtp is False
    assert decision.accepted_execution_path == "rejected"
    assert decision.mtp_enabled is False
    assert decision.sidecar_status == "ready"
    assert decision.disable_reason == "mimo_mtp_native_runtime_disabled"
    assert decision.error_message is not None
    assert "mimo_mtp_native_runtime_disabled" in decision.error_message
    assert decision.telemetry["accepted_execution_path"] == "rejected"
    assert decision.telemetry["mtp_enabled"] is False
    assert decision.telemetry["mtp_sidecar_status"] == "ready"
    assert (
        decision.telemetry["mtp_disable_reason"] == "mimo_mtp_native_runtime_disabled"
    )
