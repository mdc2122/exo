from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pytest

from exo.shared.types.common import ModelId
from exo.shared.types.mimo_mtp_classifier import classify_mimo_mtp_request
from exo.shared.types.mimo_mtp_errors import (
    MimoMtpFailClosedError,
    MimoMtpUnsupportedDepthError,
)
from exo.shared.types.text_generation import (
    InputMessage,
    MimoMtpFastpathParams,
    TextGenerationTaskParams,
)
from exo.shared.types.validate_mtp_intent import (
    MTPDisabledReason,
    MTPValidatedIntent,
    validate_mtp_intent,
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
    if suffix.endswith((".weight.scales", ".weight.biases")):
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
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
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


# ---------------------------------------------------------------------------
# Helper: construct MTP params with unsupported depth (bypass Pydantic)
# ---------------------------------------------------------------------------

def _mtp_params_with_unsupported_depth(
    *,
    depth: int,
    sidecar_path: str | None = "/models/model_mtp.safetensors",
    fail_closed: bool = True,
) -> MimoMtpFastpathParams:
    """Construct MimoMtpFastpathParams with an unsupported depth value.

    Pydantic validation at construction rejects unsupported depths, so we
    use model_construct to bypass the validator. This simulates the scenario
    where data arrives through deserialization or another bypass path.
    """
    return MimoMtpFastpathParams.model_construct(
        enabled=True,
        depth=depth,
        sidecar_path=sidecar_path,
        fail_closed=fail_closed,
    )


# ---------------------------------------------------------------------------
# Sub-AC 2b: unsupported-depth fail-closed condition
# ---------------------------------------------------------------------------


class TestUnsupportedDepthFailClosedWorkerFastpath:
    """When MTP is requested with a depth that exceeds the maximum supported
    depth (1, 2, 3), the worker fastpath must reject the request with
    disable_reason=unsupported_depth and never return MTPValidatedIntent.

    These tests verify Sub-AC 2b: the unsupported-depth fail-closed condition.
    """

    @pytest.mark.parametrize("unsupported_depth", [0, 4, 5, -1, 10, 100])
    def test_unsupported_depth_fail_closed_returns_rejected_decision(
        self,
        unsupported_depth: int,
    ) -> None:
        """Exceeding max supported depth returns a rejected decision, not MTP."""
        mtp = _mtp_params_with_unsupported_depth(
            depth=unsupported_depth, fail_closed=True
        )
        decision = evaluate_mimo_mtp_worker_fastpath(
            _task_params(mtp=mtp),
        )

        assert decision.should_use_mtp is False
        assert decision.accepted_execution_path == "rejected"
        assert decision.mtp_enabled is False
        assert decision.mtp_depth is None
        assert decision.disable_reason == "unsupported_depth"
        assert decision.reject_reason == "unsupported_depth"
        assert decision.fallback_reason is None
        assert decision.error_message is not None
        assert "unsupported_depth" in decision.error_message
        assert f"depth={unsupported_depth}" in decision.error_message

    @pytest.mark.parametrize("unsupported_depth", [0, 4, 5, -1, 10])
    def test_unsupported_depth_fail_closed_raises_typed_error(
        self,
        unsupported_depth: int,
    ) -> None:
        """Exceeding max supported depth with fail_closed=True raises
        MimoMtpUnsupportedDepthError via raise_fail_closed_error()."""
        mtp = _mtp_params_with_unsupported_depth(
            depth=unsupported_depth, fail_closed=True
        )
        decision = evaluate_mimo_mtp_worker_fastpath(
            _task_params(mtp=mtp),
        )

        assert decision.accepted_execution_path == "rejected"
        with pytest.raises(MimoMtpUnsupportedDepthError) as exc_info:
            decision.raise_fail_closed_error()

        error = exc_info.value
        assert isinstance(error, MimoMtpUnsupportedDepthError)
        assert isinstance(error, MimoMtpFailClosedError)
        assert error.mtp_execution_state == "unsupported_depth"
        assert error.disable_reason == "unsupported_depth"
        assert error.error_code == "mimo_mtp_unsupported_depth"
        assert error.requested_depth == unsupported_depth
        assert error.model_id == "kernelpool/MiMo-V2.5-Pro-6bit"
        assert error.supported_depths == (1, 2, 3)

    @pytest.mark.parametrize("unsupported_depth", [0, 4, 5, -1, 10])
    def test_unsupported_depth_fail_closed_typed_error_detail(
        self,
        unsupported_depth: int,
    ) -> None:
        """MimoMtpUnsupportedDepthError.detail() includes supported_depths."""
        mtp = _mtp_params_with_unsupported_depth(
            depth=unsupported_depth, fail_closed=True
        )
        decision = evaluate_mimo_mtp_worker_fastpath(
            _task_params(mtp=mtp),
        )

        assert decision.accepted_execution_path == "rejected"
        with pytest.raises(MimoMtpUnsupportedDepthError) as exc_info:
            decision.raise_fail_closed_error()

        detail = exc_info.value.detail()
        assert detail["mtp_execution_state"] == "unsupported_depth"
        assert detail["mtp_disable_reason"] == "unsupported_depth"
        assert detail["mtp_enabled"] is False
        assert detail["mtp_depth"] is None
        assert detail["accepted_execution_path"] == "rejected"
        assert detail["supported_depths"] == (1, 2, 3)
        assert detail["requested_mtp_depth"] == unsupported_depth

    @pytest.mark.parametrize("unsupported_depth", [0, 4, 5, -1, 10, 100])
    def test_unsupported_depth_fail_closed_never_returns_mtp_validated_intent(
        self,
        unsupported_depth: int,
    ) -> None:
        """Exceeding max supported depth must NEVER produce MTPValidatedIntent
        through the classifier → validate_mtp_intent pipeline."""
        mtp = _mtp_params_with_unsupported_depth(
            depth=unsupported_depth, fail_closed=True
        )
        params = _task_params(mtp=mtp)

        # Classifier must classify as unsupported_depth
        classification = classify_mimo_mtp_request(params)
        assert classification.label == "unsupported_depth"

        # validate_mtp_intent must return MTPDisabledReason, NEVER MTPValidatedIntent
        result = validate_mtp_intent(classification)
        assert isinstance(result, MTPDisabledReason), (
            f"unsupported_depth={unsupported_depth} must not produce "
            f"MTPValidatedIntent; got {type(result).__name__}"
        )
        assert not isinstance(result, MTPValidatedIntent), (
            f"unsupported_depth={unsupported_depth} MUST NOT produce "
            f"MTPValidatedIntent under any condition"
        )
        assert result.disable_reason == "unsupported_depth"
        assert result.is_validated_intent is False
        assert result.is_disabled_reason is True

    def test_unsupported_depth_fail_open_falls_back_to_ar(self) -> None:
        """Exceeding max supported depth with fail_closed=False falls back to AR."""
        mtp = _mtp_params_with_unsupported_depth(
            depth=4, fail_closed=False
        )
        decision = evaluate_mimo_mtp_worker_fastpath(
            _task_params(mtp=mtp),
        )

        assert decision.should_use_mtp is False
        assert decision.accepted_execution_path == "ar"
        assert decision.mtp_enabled is False
        assert decision.mtp_depth is None
        assert decision.disable_reason == "unsupported_depth"
        assert decision.reject_reason is None
        assert decision.fallback_reason == "fail_open_unsupported_depth"
        assert decision.error_message is None
        assert decision.telemetry["accepted_execution_path"] == "ar"
        assert decision.telemetry["mtp_enabled"] is False
        assert decision.telemetry["mtp_disable_reason"] == "unsupported_depth"
        assert decision.telemetry["mtp_fallback_reason"] == "fail_open_unsupported_depth"

    def test_unsupported_depth_fail_open_never_returns_mtp_validated_intent(self) -> None:
        """Even fail-open unsupported depth must never produce MTPValidatedIntent."""
        mtp = _mtp_params_with_unsupported_depth(
            depth=4, fail_closed=False
        )
        params = _task_params(mtp=mtp)

        classification = classify_mimo_mtp_request(params)
        assert classification.label == "unsupported_depth"

        result = validate_mtp_intent(classification)
        assert isinstance(result, MTPDisabledReason)
        assert not isinstance(result, MTPValidatedIntent)

    def test_unsupported_depth_none_depth_is_not_rejected(self) -> None:
        """None depth (default) must NOT be classified as unsupported_depth."""
        mtp = MimoMtpFastpathParams(
            enabled=True,
            depth=None,
            sidecar_path="/models/model_mtp.safetensors",
            fail_closed=True,
        )
        params = _task_params(mtp=mtp)

        classification = classify_mimo_mtp_request(params)
        assert classification.label != "unsupported_depth"

    @pytest.mark.parametrize("supported_depth", [1, 2, 3])
    def test_supported_depths_are_not_rejected_as_unsupported(
        self,
        supported_depth: int,
    ) -> None:
        """Each supported depth must NOT trigger the unsupported_depth guard."""
        mtp = MimoMtpFastpathParams(
            enabled=True,
            depth=supported_depth,
            sidecar_path="/models/model_mtp.safetensors",
            fail_closed=True,
        )
        params = _task_params(mtp=mtp)

        classification = classify_mimo_mtp_request(params)
        assert classification.label != "unsupported_depth"

        result = validate_mtp_intent(classification)
        assert isinstance(result, MTPValidatedIntent), (
            f"supported depth={supported_depth} should produce MTPValidatedIntent"
        )

    def test_unsupported_depth_with_sidecar_path(self) -> None:
        """Unsupported depth is caught even when a valid sidecar path is provided.
        The depth guard fires before the sidecar check."""
        mtp = _mtp_params_with_unsupported_depth(
            depth=4,
            sidecar_path="/models/model_mtp.safetensors",
            fail_closed=True,
        )
        decision = evaluate_mimo_mtp_worker_fastpath(
            _task_params(mtp=mtp),
        )

        assert decision.should_use_mtp is False
        assert decision.accepted_execution_path == "rejected"
        assert decision.disable_reason == "unsupported_depth"
        # The unsupported_depth guard fires before the sidecar check,
        # so sidecar_status is None (not probed yet)
        assert decision.sidecar_status is None

    def test_unsupported_depth_with_valid_sidecar_file_on_disk(
        self,
        tmp_path: Path,
    ) -> None:
        """Unsupported depth is caught even when a valid sidecar file exists on disk."""
        sidecar_path = tmp_path / "model_mtp.safetensors"
        _write_synthetic_official_sidecar(sidecar_path)

        mtp = _mtp_params_with_unsupported_depth(
            depth=99,
            sidecar_path=str(sidecar_path),
            fail_closed=True,
        )
        decision = evaluate_mimo_mtp_worker_fastpath(
            _task_params(mtp=mtp),
        )

        assert decision.should_use_mtp is False
        assert decision.accepted_execution_path == "rejected"
        assert decision.disable_reason == "unsupported_depth"
        assert decision.mtp_enabled is False

    def test_unsupported_depth_telemetry_records_reject_reason(self) -> None:
        """Telemetry for unsupported depth includes the correct reject reason."""
        mtp = _mtp_params_with_unsupported_depth(
            depth=4, fail_closed=True
        )
        decision = evaluate_mimo_mtp_worker_fastpath(
            _task_params(mtp=mtp),
        )

        assert decision.telemetry["accepted_execution_path"] == "rejected"
        assert decision.telemetry["mtp_enabled"] is False
        assert decision.telemetry["mtp_disable_reason"] == "unsupported_depth"
        assert decision.telemetry["mtp_reject_reason"] == "unsupported_depth"
        assert decision.telemetry["requested_mtp_depth"] == 4


class TestUnsupportedDepthErrorProperties:
    """MimoMtpUnsupportedDepthError has correct typed properties and
    inheritance hierarchy."""

    def test_is_subclass_of_mimo_mtp_fail_closed_error(self) -> None:
        assert issubclass(MimoMtpUnsupportedDepthError, MimoMtpFailClosedError)

    def test_is_subclass_of_exception(self) -> None:
        assert issubclass(MimoMtpUnsupportedDepthError, Exception)

    def test_direct_construction(self) -> None:
        error = MimoMtpUnsupportedDepthError(
            "test error",
            model_id="test-model",
            requested_depth=4,
        )
        assert error.mtp_execution_state == "unsupported_depth"
        assert error.disable_reason == "unsupported_depth"
        assert error.error_code == "mimo_mtp_unsupported_depth"
        assert error.requested_depth == 4
        assert error.model_id == "test-model"
        assert error.supported_depths == (1, 2, 3)

    def test_custom_supported_depths(self) -> None:
        """supported_depths can be overridden for testing."""
        error = MimoMtpUnsupportedDepthError(
            "test error",
            model_id="test-model",
            requested_depth=4,
            supported_depths=(1, 2),
        )
        assert error.supported_depths == (1, 2)

    def test_detail_includes_supported_depths(self) -> None:
        error = MimoMtpUnsupportedDepthError(
            "test error",
            model_id="test-model",
            requested_depth=4,
        )
        detail = error.detail()
        assert "supported_depths" in detail
        assert detail["supported_depths"] == (1, 2, 3)
        assert detail["mtp_execution_state"] == "unsupported_depth"
        assert detail["mtp_enabled"] is False
        assert detail["accepted_execution_path"] == "rejected"
