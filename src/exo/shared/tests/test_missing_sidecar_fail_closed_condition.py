"""Sub-AC 2d: Missing-sidecar-with-fail-closed-true condition.

Comprehensive cross-layer tests proving that when MTP is requested with
fail_closed=True but the sidecar is missing or not provided, every layer
returns the correct typed error and never silently claims MTP execution.

Layers tested:
  1. Classifier (pure function): classify_mimo_mtp_request
  2. Intent validator (pure function): validate_mtp_intent
  3. Worker fastpath evaluator (pure function): evaluate_mimo_mtp_worker_fastpath
  4. Typed error hierarchy: MimoMtpMissingSidecarError + raise_fail_closed_error
  5. API boundary: validate_mimo_mtp_fastpath_eligibility
  6. Worker generator routing: SequentialGenerator and BatchGenerator

Key invariant: No layer, under any configuration of fail_closed=True or
fail_closed=False, may ever produce a result that:
  - claims mtp_enabled=True
  - claims accepted_execution_path="mimo_mtp_fastpath"
  - claims mtp_execution_state="successful_mtp"
  - claims mode="mtp"
  - omits a disable_reason containing "missing_sidecar"

This is the **fail-closed** contract: missing-sidecar MTP requests must
produce a typed error (fail-closed) or AR fallback with telemetry
(fail-open), but must NEVER silently claim MTP execution.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from exo.shared.models.model_cards import (
    MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
    MIMO_V25_PRO_MODEL_IDS,
)
from exo.shared.types.commands import CommandId
from exo.shared.types.common import ModelId
from exo.shared.types.mimo_mtp_classifier import (
    classify_mimo_mtp_request,
)
from exo.shared.types.mimo_mtp_errors import (
    MimoMtpFailClosedError,
    MimoMtpMissingSidecarError,
    raise_mimo_mtp_fail_closed_error,
)
from exo.shared.types.tasks import TaskId
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
from exo.worker.engines.mlx.mimo_mtp_fast.worker_fastpath import (
    MimoMtpWorkerFastpathCache,
    evaluate_mimo_mtp_worker_fastpath,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MIMO_MODEL = MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID
_ALL_MIMO_MODELS: list[ModelId] = sorted(MIMO_V25_PRO_MODEL_IDS)


def _ar_task_params(
    model: ModelId = _MIMO_MODEL,
) -> TextGenerationTaskParams:
    """Default AR request: no MTP intent."""
    return TextGenerationTaskParams(
        model=model,
        input=[InputMessage(role="user", content="Hello")],
        max_output_tokens=4,
    )


def _mtp_task_params(
    *,
    model: ModelId = _MIMO_MODEL,
    depth: int | None = 1,
    sidecar_path: str | None = None,
    fail_closed: bool = True,
    enabled: bool = True,
) -> TextGenerationTaskParams:
    """MTP-intent request with missing sidecar by default.

    sidecar_path defaults to None so that the missing-sidecar condition
    is the primary test scenario. Callers that want a present sidecar
    should pass sidecar_path explicitly.
    """
    return TextGenerationTaskParams(
        model=model,
        input=[InputMessage(role="user", content="Hello")],
        max_output_tokens=4,
        mimo_mtp_fastpath_params=MimoMtpFastpathParams(
            enabled=enabled,
            depth=depth,
            sidecar_path=sidecar_path,
            fail_closed=fail_closed,
        ),
    )


def _mtp_task_params_with_missing_sidecar_file(
    tmp_path: Path,
    *,
    model: ModelId = _MIMO_MODEL,
    depth: int | None = 1,
    fail_closed: bool = True,
) -> TextGenerationTaskParams:
    """MTP-intent request pointing to a sidecar path that does not exist on disk.

    The sidecar_path is set (not None), but the file is not created,
    so the sidecar probe will detect status="missing".
    """
    missing_sidecar = tmp_path / "missing-model_mtp.safetensors"
    return TextGenerationTaskParams(
        model=model,
        input=[InputMessage(role="user", content="Hello")],
        max_output_tokens=4,
        mimo_mtp_fastpath_params=MimoMtpFastpathParams(
            enabled=True,
            depth=depth,
            sidecar_path=str(missing_sidecar),
            fail_closed=fail_closed,
        ),
    )


# Strings that would indicate MTP was claimed/successful if present
_MTP_CLAIM_STRINGS: frozenset[str] = frozenset(
    {
        "successful_mtp",
        "mimo_mtp_fastpath",  # as accepted_execution_path value
    }
)


def _assert_no_silent_mtp_claim(
    *,
    mtp_enabled: bool | None,
    accepted_execution_path: str | None = None,
    mtp_execution_state: str | None = None,
    mode: str | None = None,
    context: str = "",
) -> None:
    """Assert that no field silently claims MTP was successful.

    This is the core invariant checker for Sub-AC 2d.
    """
    assert mtp_enabled is not True, (
        f"{context}: mtp_enabled must never be True for missing-sidecar "
        f"requests, got mtp_enabled={mtp_enabled}"
    )
    if accepted_execution_path is not None:
        assert accepted_execution_path != "mimo_mtp_fastpath", (
            f"{context}: accepted_execution_path must never be "
            f"'mimo_mtp_fastpath' for missing-sidecar requests, "
            f"got {accepted_execution_path}"
        )
    if mtp_execution_state is not None:
        assert mtp_execution_state != "successful_mtp", (
            f"{context}: mtp_execution_state must never be "
            f"'successful_mtp' for missing-sidecar requests, "
            f"got {mtp_execution_state}"
        )
    if mode is not None:
        assert mode != "mtp", (
            f"{context}: mode must never be 'mtp' for missing-sidecar "
            f"requests, got mode={mode}"
        )


# ===========================================================================
# Layer 1: Classifier — pure function classify_mimo_mtp_request
# ===========================================================================


class TestClassifierMissingSidecarFailClosed:
    """classify_mimo_mtp_request must return missing_sidecar for MiMo models
    requesting MTP without a sidecar path, regardless of other field values."""

    def test_mimo_model_with_no_sidecar_path_returns_missing_sidecar(self) -> None:
        params = _mtp_task_params(sidecar_path=None)
        result = classify_mimo_mtp_request(params)
        assert result.label == "missing_sidecar"
        assert result.discriminant == "unsupported"
        assert result.disable_reason == "missing_sidecar"

    @pytest.mark.parametrize("mimo_model", _ALL_MIMO_MODELS)
    def test_missing_sidecar_for_each_mimo_model_id(
        self, mimo_model: ModelId
    ) -> None:
        """Every MiMo V2.5 Pro variant without sidecar classifies as missing_sidecar."""
        params = _mtp_task_params(model=mimo_model, sidecar_path=None)
        result = classify_mimo_mtp_request(params)
        assert result.label == "missing_sidecar"
        assert result.disable_reason == "missing_sidecar"

    def test_classifier_never_claims_mtp_enabled_for_missing_sidecar(self) -> None:
        params = _mtp_task_params(sidecar_path=None)
        result = classify_mimo_mtp_request(params)
        _assert_no_silent_mtp_claim(
            mtp_enabled=result.mtp_enabled,
            context="classifier for missing-sidecar",
        )
        assert result.mtp_enabled is False

    def test_classifier_fail_closed_true_for_missing_sidecar(self) -> None:
        params = _mtp_task_params(sidecar_path=None, fail_closed=True)
        result = classify_mimo_mtp_request(params)
        assert result.label == "missing_sidecar"
        assert result.fail_closed is True

    def test_classifier_fail_closed_false_for_missing_sidecar(self) -> None:
        params = _mtp_task_params(sidecar_path=None, fail_closed=False)
        result = classify_mimo_mtp_request(params)
        assert result.label == "missing_sidecar"
        assert result.fail_closed is False
        # Even fail-open intent is still classified as missing_sidecar
        # The classifier does not silently allow MTP
        assert result.mtp_enabled is False

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_missing_sidecar_for_each_supported_depth(
        self, depth: int
    ) -> None:
        params = _mtp_task_params(depth=depth, sidecar_path=None)
        result = classify_mimo_mtp_request(params)
        assert result.label == "missing_sidecar"
        assert result.requested_depth == depth

    def test_none_depth_with_missing_sidecar(self) -> None:
        """None depth is treated as potentially compatible, but missing
        sidecar still blocks MTP."""
        params = _mtp_task_params(depth=None, sidecar_path=None)
        result = classify_mimo_mtp_request(params)
        assert result.label == "missing_sidecar"
        assert result.requested_depth is None

    def test_missing_sidecar_is_unsupported(self) -> None:
        params = _mtp_task_params(sidecar_path=None)
        result = classify_mimo_mtp_request(params)
        assert result.is_unsupported is True
        assert result.is_compatible is False
        assert result.is_unwired is False


# ===========================================================================
# Layer 2: Intent validator — pure function validate_mtp_intent
# ===========================================================================


class TestIntentValidatorMissingSidecarFailClosed:
    """validate_mtp_intent must return MTPDisabledReason with
    disable_reason='missing_sidecar' for MiMo models requesting MTP
    without a sidecar path."""

    def test_missing_sidecar_returns_disabled_reason(self) -> None:
        params = _mtp_task_params(sidecar_path=None)
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)
        assert isinstance(result, MTPDisabledReason)
        assert result.disable_reason == "missing_sidecar"
        assert result.classification_label == "missing_sidecar"

    def test_missing_sidecar_never_produces_validated_intent(self) -> None:
        params = _mtp_task_params(sidecar_path=None)
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)
        assert not isinstance(result, MTPValidatedIntent), (
            "missing-sidecar request must never produce MTPValidatedIntent"
        )

    def test_missing_sidecar_mtp_enabled_is_false(self) -> None:
        params = _mtp_task_params(sidecar_path=None)
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)
        assert isinstance(result, MTPDisabledReason)
        _assert_no_silent_mtp_claim(
            mtp_enabled=result.mtp_enabled,
            context="validator for missing-sidecar",
        )
        assert result.mtp_enabled is False

    def test_missing_sidecar_fail_closed_true(self) -> None:
        params = _mtp_task_params(sidecar_path=None, fail_closed=True)
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)
        assert isinstance(result, MTPDisabledReason)
        assert result.fail_closed is True

    def test_missing_sidecar_fail_closed_false(self) -> None:
        params = _mtp_task_params(sidecar_path=None, fail_closed=False)
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)
        assert isinstance(result, MTPDisabledReason)
        assert result.fail_closed is False
        # Still disabled, even with fail_open
        assert result.mtp_enabled is False
        assert result.disable_reason == "missing_sidecar"

    def test_missing_sidecar_sidecar_path_provided_is_false(self) -> None:
        params = _mtp_task_params(sidecar_path=None)
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)
        assert isinstance(result, MTPDisabledReason)
        assert result.sidecar_path_provided is False

    @pytest.mark.parametrize("mimo_model", _ALL_MIMO_MODELS)
    def test_missing_sidecar_for_each_mimo_model_id(
        self, mimo_model: ModelId
    ) -> None:
        """Every MiMo V2.5 Pro variant without sidecar produces MTPDisabledReason."""
        params = _mtp_task_params(model=mimo_model, sidecar_path=None)
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)
        assert isinstance(result, MTPDisabledReason)
        assert result.disable_reason == "missing_sidecar"


# ===========================================================================
# Layer 3: Worker fastpath evaluator — evaluate_mimo_mtp_worker_fastpath
# ===========================================================================


class TestWorkerFastpathMissingSidecarFailClosed:
    """evaluate_mimo_mtp_worker_fastpath must reject missing-sidecar MTP
    requests with typed error when fail_closed=True."""

    def test_fail_closed_no_sidecar_path_returns_rejected_decision(self) -> None:
        """When sidecar_path is None and fail_closed=True, returns rejected."""
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params(sidecar_path=None, fail_closed=True),
        )
        assert decision.should_use_mtp is False
        assert decision.accepted_execution_path == "rejected"
        assert decision.mtp_enabled is False
        assert decision.disable_reason == "missing_sidecar"
        assert decision.reject_reason == "missing_sidecar"
        assert decision.fallback_reason is None

    def test_fail_closed_no_sidecar_path_never_claims_mtp(self) -> None:
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params(sidecar_path=None, fail_closed=True),
        )
        _assert_no_silent_mtp_claim(
            mtp_enabled=decision.mtp_enabled,
            accepted_execution_path=decision.accepted_execution_path,
            context="worker-fastpath fail_closed no sidecar_path",
        )
        assert decision.mtp_depth is None
        assert decision.sidecar is None

    def test_fail_closed_no_sidecar_path_error_message(self) -> None:
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params(sidecar_path=None, fail_closed=True),
        )
        assert decision.error_message is not None
        assert "missing_sidecar" in decision.error_message
        assert "reject_reason=missing_sidecar" in decision.error_message

    def test_fail_closed_no_sidecar_path_telemetry(self) -> None:
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params(sidecar_path=None, fail_closed=True, depth=2),
        )
        assert decision.telemetry["accepted_execution_path"] == "rejected"
        assert decision.telemetry["mtp_enabled"] is False
        assert decision.telemetry["requested_mtp_depth"] == 2
        assert decision.telemetry["mtp_sidecar_status"] == "missing"
        assert decision.telemetry["mtp_disable_reason"] == "missing_sidecar"
        assert decision.telemetry["mtp_reject_reason"] == "missing_sidecar"

    def test_fail_closed_missing_sidecar_file_returns_rejected_decision(
        self, tmp_path: Path
    ) -> None:
        """When sidecar file doesn't exist on disk and fail_closed=True, returns rejected."""
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params_with_missing_sidecar_file(
                tmp_path, fail_closed=True
            ),
        )
        assert decision.should_use_mtp is False
        assert decision.accepted_execution_path == "rejected"
        assert decision.mtp_enabled is False
        assert decision.disable_reason == "missing_sidecar"
        assert decision.sidecar_status == "missing"
        assert decision.reject_reason == "missing_sidecar"

    def test_fail_closed_missing_sidecar_file_never_claims_mtp(
        self, tmp_path: Path
    ) -> None:
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params_with_missing_sidecar_file(
                tmp_path, fail_closed=True
            ),
        )
        _assert_no_silent_mtp_claim(
            mtp_enabled=decision.mtp_enabled,
            accepted_execution_path=decision.accepted_execution_path,
            context="worker-fastpath fail_closed missing sidecar file",
        )

    def test_fail_closed_missing_sidecar_file_error_message(
        self, tmp_path: Path
    ) -> None:
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params_with_missing_sidecar_file(
                tmp_path, fail_closed=True, depth=3
            ),
        )
        assert decision.error_message is not None
        assert "missing_sidecar" in decision.error_message
        assert "reject_reason=missing_sidecar" in decision.error_message

    def test_fail_closed_missing_sidecar_file_telemetry(
        self, tmp_path: Path
    ) -> None:
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params_with_missing_sidecar_file(
                tmp_path, fail_closed=True, depth=3
            ),
        )
        assert decision.telemetry["accepted_execution_path"] == "rejected"
        assert decision.telemetry["mtp_enabled"] is False
        assert decision.telemetry["mtp_sidecar_status"] == "missing"
        assert decision.telemetry["mtp_disable_reason"] == "missing_sidecar"
        assert decision.telemetry["mtp_reject_reason"] == "missing_sidecar"
        assert decision.telemetry["requested_mtp_depth"] == 3

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_fail_closed_missing_sidecar_for_each_depth(
        self, depth: int
    ) -> None:
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params(sidecar_path=None, depth=depth, fail_closed=True),
        )
        assert decision.accepted_execution_path == "rejected"
        assert decision.requested_depth == depth
        assert decision.disable_reason == "missing_sidecar"

    def test_fail_open_no_sidecar_path_returns_ar_fallback(self) -> None:
        """When sidecar_path is None and fail_closed=False, returns AR fallback."""
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params(sidecar_path=None, fail_closed=False),
        )
        assert decision.should_use_mtp is False
        assert decision.accepted_execution_path == "ar"
        assert decision.mtp_enabled is False
        assert decision.disable_reason == "missing_sidecar"
        assert decision.reject_reason is None
        assert decision.fallback_reason == "fail_open_missing_sidecar"
        assert decision.error_message is None

    def test_fail_open_missing_sidecar_file_returns_ar_fallback(
        self, tmp_path: Path
    ) -> None:
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params_with_missing_sidecar_file(
                tmp_path, fail_closed=False
            ),
        )
        assert decision.should_use_mtp is False
        assert decision.accepted_execution_path == "ar"
        assert decision.mtp_enabled is False
        assert decision.disable_reason == "missing_sidecar"
        assert decision.fallback_reason == "fail_open_missing_sidecar"

    def test_fail_open_missing_sidecar_never_claims_mtp(self) -> None:
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params(sidecar_path=None, fail_closed=False),
        )
        _assert_no_silent_mtp_claim(
            mtp_enabled=decision.mtp_enabled,
            accepted_execution_path=decision.accepted_execution_path,
            context="worker-fastpath fail_open missing sidecar",
        )

    def test_fail_open_missing_sidecar_telemetry(self) -> None:
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params(sidecar_path=None, fail_closed=False, depth=2),
        )
        assert decision.telemetry["accepted_execution_path"] == "ar"
        assert decision.telemetry["mtp_enabled"] is False
        assert decision.telemetry["requested_mtp_depth"] == 2
        assert decision.telemetry["mtp_sidecar_status"] == "missing"
        assert decision.telemetry["mtp_disable_reason"] == "missing_sidecar"
        assert decision.telemetry["mtp_fallback_reason"] == "fail_open_missing_sidecar"

    def test_default_ar_request_returns_ar_not_rejection(self) -> None:
        """A MiMo model with no MTP intent returns normal AR, not rejection."""
        decision = evaluate_mimo_mtp_worker_fastpath(
            _ar_task_params(),
        )
        assert decision.should_use_mtp is False
        assert decision.accepted_execution_path == "ar"
        assert decision.mtp_enabled is False
        assert decision.disable_reason is None
        assert decision.error_message is None


# ===========================================================================
# Layer 4: Typed error hierarchy — MimoMtpMissingSidecarError
# ===========================================================================


class TestMimoMtpMissingSidecarErrorProperties:
    """MimoMtpMissingSidecarError has correct typed properties and
    inheritance hierarchy."""

    def test_is_subclass_of_mimo_mtp_fail_closed_error(self) -> None:
        assert issubclass(MimoMtpMissingSidecarError, MimoMtpFailClosedError)

    def test_is_subclass_of_exception(self) -> None:
        assert issubclass(MimoMtpMissingSidecarError, Exception)

    def test_direct_construction_with_sidecar_path(self) -> None:
        error = MimoMtpMissingSidecarError(
            "test error",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            requested_depth=1,
            sidecar_path="/models/model_mtp.safetensors",
        )
        assert error.mtp_execution_state == "missing_sidecar"
        assert error.disable_reason == "missing_sidecar"
        assert error.error_code == "mimo_mtp_missing_sidecar"
        assert error.requested_depth == 1
        assert error.model_id == "kernelpool/MiMo-V2.5-Pro-6bit"
        assert error.sidecar_path == "/models/model_mtp.safetensors"
        assert error.sidecar_status == "missing"

    def test_direct_construction_without_sidecar_path(self) -> None:
        """MimoMtpMissingSidecarError can be constructed when sidecar_path is None
        (the primary missing-sidecar scenario)."""
        error = MimoMtpMissingSidecarError(
            "test error",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            requested_depth=2,
            sidecar_path=None,
        )
        assert error.mtp_execution_state == "missing_sidecar"
        assert error.sidecar_path is None
        assert error.sidecar_status == "missing"

    def test_detail_includes_all_required_fields(self) -> None:
        error = MimoMtpMissingSidecarError(
            "test error",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            requested_depth=1,
            sidecar_path="/models/model_mtp.safetensors",
        )
        detail = error.detail()
        assert detail["mtp_execution_state"] == "missing_sidecar"
        assert detail["mtp_disable_reason"] == "missing_sidecar"
        assert detail["mtp_enabled"] is False
        assert detail["mtp_depth"] is None
        assert detail["accepted_execution_path"] == "rejected"
        assert detail["requested_mtp_depth"] == 1
        assert detail["mtp_sidecar_status"] == "missing"
        assert detail["sidecar_path"] == "/models/model_mtp.safetensors"

    def test_detail_with_none_sidecar_path(self) -> None:
        error = MimoMtpMissingSidecarError(
            "test error",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            requested_depth=1,
            sidecar_path=None,
        )
        detail = error.detail()
        assert detail["sidecar_path"] is None
        assert detail["mtp_sidecar_status"] == "missing"

    def test_detail_never_claims_successful_mtp(self) -> None:
        error = MimoMtpMissingSidecarError(
            "test error",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            requested_depth=1,
        )
        detail = error.detail()
        _assert_no_silent_mtp_claim(
            mtp_enabled=cast(bool | None, detail.get("mtp_enabled")),
            accepted_execution_path=cast(
                str | None, detail.get("accepted_execution_path")
            ),
            mtp_execution_state=cast(
                str | None, detail.get("mtp_execution_state")
            ),
            context="MimoMtpMissingSidecarError.detail()",
        )


class TestRaiseFailClosedErrorForMissingSidecar:
    """MimoMtpWorkerFastpathDecision.raise_fail_closed_error() must raise
    MimoMtpMissingSidecarError when disable_reason='missing_sidecar'."""

    def test_raise_fail_closed_error_raises_missing_sidecar_error_no_path(
        self,
    ) -> None:
        """When sidecar_path is None and fail_closed=True, raise_fail_closed_error
        raises MimoMtpMissingSidecarError."""
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params(sidecar_path=None, fail_closed=True),
        )
        assert decision.accepted_execution_path == "rejected"
        assert decision.disable_reason == "missing_sidecar"

        with pytest.raises(MimoMtpMissingSidecarError) as exc_info:
            decision.raise_fail_closed_error()

        error = exc_info.value
        assert isinstance(error, MimoMtpMissingSidecarError)
        assert isinstance(error, MimoMtpFailClosedError)
        assert error.mtp_execution_state == "missing_sidecar"
        assert error.disable_reason == "missing_sidecar"
        assert error.error_code == "mimo_mtp_missing_sidecar"
        assert error.sidecar_status == "missing"

    def test_raise_fail_closed_error_raises_missing_sidecar_error_with_path(
        self, tmp_path: Path
    ) -> None:
        """When sidecar file is missing on disk and fail_closed=True,
        raise_fail_closed_error raises MimoMtpMissingSidecarError."""
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params_with_missing_sidecar_file(
                tmp_path, fail_closed=True
            ),
        )
        assert decision.accepted_execution_path == "rejected"
        assert decision.disable_reason == "missing_sidecar"

        with pytest.raises(MimoMtpMissingSidecarError) as exc_info:
            decision.raise_fail_closed_error()

        error = exc_info.value
        assert isinstance(error, MimoMtpMissingSidecarError)
        assert error.mtp_execution_state == "missing_sidecar"
        assert error.model_id == str(_MIMO_MODEL)

    def test_raise_fail_closed_error_detail_includes_telemetry(self) -> None:
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params(sidecar_path=None, fail_closed=True, depth=2),
        )
        with pytest.raises(MimoMtpMissingSidecarError) as exc_info:
            decision.raise_fail_closed_error()

        detail = exc_info.value.detail()
        assert detail["mtp_execution_state"] == "missing_sidecar"
        assert detail["mtp_enabled"] is False
        assert detail["mtp_depth"] is None
        assert detail["accepted_execution_path"] == "rejected"
        assert detail["requested_mtp_depth"] == 2
        assert detail["mtp_sidecar_status"] == "missing"
        assert detail["mtp_disable_reason"] == "missing_sidecar"

    def test_raise_fail_closed_error_on_non_rejected_raises_value_error(
        self,
    ) -> None:
        """Calling raise_fail_closed_error on a non-rejected decision raises ValueError."""
        decision = evaluate_mimo_mtp_worker_fastpath(
            _ar_task_params(),
        )
        assert decision.accepted_execution_path == "ar"
        with pytest.raises(ValueError, match="non-rejected decision"):
            decision.raise_fail_closed_error()

    def test_raise_fail_closed_error_on_fail_open_rejected_raises_value_error(
        self,
    ) -> None:
        """Calling raise_fail_closed_error on a fail-open AR-fallback decision
        (not rejected) raises ValueError."""
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params(sidecar_path=None, fail_closed=False),
        )
        assert decision.accepted_execution_path == "ar"
        with pytest.raises(ValueError, match="non-rejected decision"):
            decision.raise_fail_closed_error()


class TestRaiseMimoMtpFailClosedErrorFactory:
    """raise_mimo_mtp_fail_closed_error(disable_reason='missing_sidecar')
    must raise MimoMtpMissingSidecarError with correct properties."""

    def test_factory_raises_missing_sidecar_error(self) -> None:
        with pytest.raises(MimoMtpMissingSidecarError) as exc_info:
            raise_mimo_mtp_fail_closed_error(
                disable_reason="missing_sidecar",
                model_id="kernelpool/MiMo-V2.5-Pro-6bit",
                requested_depth=1,
                sidecar_path=None,
            )
        error = exc_info.value
        assert isinstance(error, MimoMtpMissingSidecarError)
        assert error.mtp_execution_state == "missing_sidecar"
        assert error.model_id == "kernelpool/MiMo-V2.5-Pro-6bit"
        assert error.requested_depth == 1
        assert error.sidecar_path is None
        assert error.sidecar_status == "missing"

    def test_factory_with_sidecar_path(self) -> None:
        with pytest.raises(MimoMtpMissingSidecarError) as exc_info:
            raise_mimo_mtp_fail_closed_error(
                disable_reason="missing_sidecar",
                model_id="kernelpool/MiMo-V2.5-Pro-6bit",
                requested_depth=2,
                sidecar_path="/models/missing_mtp.safetensors",
            )
        error = exc_info.value
        assert error.sidecar_path == "/models/missing_mtp.safetensors"

    def test_factory_unknown_reason_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown MTP fail-closed"):
            raise_mimo_mtp_fail_closed_error(
                disable_reason="totally_unknown_reason",
                model_id="test",
                requested_depth=1,
            )

    def test_factory_error_is_catchable_as_base_class(self) -> None:
        """MimoMtpMissingSidecarError is catchable as MimoMtpFailClosedError."""
        with pytest.raises(MimoMtpFailClosedError):
            raise_mimo_mtp_fail_closed_error(
                disable_reason="missing_sidecar",
                model_id="kernelpool/MiMo-V2.5-Pro-6bit",
                requested_depth=1,
            )


# ===========================================================================
# Layer 5: API boundary — validate_mimo_mtp_fastpath_eligibility
# ===========================================================================


class TestApiBoundaryMissingSidecarFailClosed:
    """validate_mimo_mtp_fastpath_eligibility must raise HTTPException(400)
    with typed error detail for missing-sidecar requests with fail_closed=True."""

    @pytest.mark.asyncio
    async def test_fail_closed_missing_sidecar_path_raises_400(
        self,
    ) -> None:
        """When mimo_mtp_sidecar_path is None and fail_closed=True,
        the API boundary raises HTTP 400 with missing_sidecar detail."""
        from exo.api.adapters import chat_completions
        from exo.api.types import ChatCompletionMessage, ChatCompletionRequest

        request = ChatCompletionRequest(
            model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
            messages=[ChatCompletionMessage(role="user", content="Hello")],
            mimo_mtp_fastpath=True,
            mimo_mtp_depth=1,
            mimo_mtp_sidecar_path=None,
            mimo_mtp_fail_closed=True,
        )
        with pytest.raises(HTTPException) as exc_info:
            await chat_completions.chat_request_to_text_generation(request)
        assert exc_info.value.status_code == 400
        raw_detail = cast(object, exc_info.value.detail)
        assert isinstance(raw_detail, dict)
        detail = cast(Mapping[str, object], raw_detail)
        assert detail["error"] == "mimo_mtp_sidecar_missing"
        assert detail["mtp_enabled"] is False
        assert detail["accepted_execution_path"] == "rejected"
        assert detail["mtp_disable_reason"] == "missing_sidecar"
        assert detail["mtp_execution_state"] == "missing_sidecar"
        assert detail["mtp_sidecar_status"] == "missing"
        _assert_no_silent_mtp_claim(
            mtp_enabled=detail.get("mtp_enabled"),
            accepted_execution_path=cast(
                str | None, detail.get("accepted_execution_path")
            ),
            mtp_execution_state=cast(
                str | None, detail.get("mtp_execution_state")
            ),
            context="API boundary for missing-sidecar fail-closed",
        )

    @pytest.mark.asyncio
    async def test_fail_closed_missing_sidecar_file_raises_400(
        self, tmp_path: Path
    ) -> None:
        """When the sidecar file is missing on disk and fail_closed=True,
        the API boundary raises HTTP 400 with missing_sidecar detail."""
        from exo.api.adapters import chat_completions
        from exo.api.types import ChatCompletionMessage, ChatCompletionRequest

        missing_sidecar = tmp_path / "missing-model_mtp.safetensors"
        request = ChatCompletionRequest(
            model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
            messages=[ChatCompletionMessage(role="user", content="Hello")],
            mimo_mtp_fastpath=True,
            mimo_mtp_depth=2,
            mimo_mtp_sidecar_path=str(missing_sidecar),
            mimo_mtp_fail_closed=True,
        )
        with pytest.raises(HTTPException) as exc_info:
            await chat_completions.chat_request_to_text_generation(request)
        assert exc_info.value.status_code == 400
        raw_detail = cast(object, exc_info.value.detail)
        assert isinstance(raw_detail, dict)
        detail = cast(Mapping[str, object], raw_detail)
        assert detail["error"] == "mimo_mtp_sidecar_missing"
        assert detail["mtp_enabled"] is False
        assert detail["mtp_disable_reason"] == "missing_sidecar"
        assert detail["mtp_execution_state"] == "missing_sidecar"
        assert detail["mtp_sidecar_status"] == "missing"
        assert str(missing_sidecar) in str(detail["message"])

    @pytest.mark.asyncio
    async def test_fail_open_missing_sidecar_passes_api_validation(
        self,
    ) -> None:
        """When fail_closed=False, the API boundary's sidecar guard is
        skipped (fail-open requests pass without sidecar validation).
        The fail-open fallback to AR happens at the worker/generator layer."""
        from exo.api.types import ChatCompletionMessage, ChatCompletionRequest

        request = ChatCompletionRequest(
            model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
            messages=[ChatCompletionMessage(role="user", content="Hello")],
            mimo_mtp_fastpath=True,
            mimo_mtp_depth=1,
            mimo_mtp_sidecar_path=None,
            mimo_mtp_fail_closed=False,
        )
        from exo.api.adapters.chat_completions import (
            validate_mimo_mtp_fastpath_eligibility,
        )
        # Should not raise — fail-open skips sidecar validation at API boundary
        validate_mimo_mtp_fastpath_eligibility(request)

    @pytest.mark.asyncio
    async def test_mimo_model_without_mtp_intent_passes_api_validation(
        self,
    ) -> None:
        """A MiMo model with no MTP intent should pass API validation
        without raising an HTTPException."""
        from exo.api.adapters.chat_completions import (
            validate_mimo_mtp_fastpath_eligibility,
        )
        from exo.api.types import ChatCompletionMessage, ChatCompletionRequest

        request = ChatCompletionRequest(
            model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
            messages=[ChatCompletionMessage(role="user", content="Hello")],
        )
        # Should not raise
        validate_mimo_mtp_fastpath_eligibility(request)


# ===========================================================================
# Layer 6: Worker generator routing — SequentialGenerator / BatchGenerator
# ===========================================================================


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
        on_prefill_progress: object,
        distributed_prompt_progress_callback: object,
        on_generation_token: object,
    ) -> int:
        self.submitted_params.append(task_params)
        return 7

    def cancel(self, _uids: list[int]) -> None:
        pass

    def close(self) -> None:
        pass


class TestWorkerGeneratorRoutingMissingSidecarFailClosed:
    """The worker generator routing must reject missing-sidecar MTP requests
    fail-closed and never dispatch them as MTP generation."""

    def test_sequential_fail_closed_missing_sidecar_no_path_rejection_does_not_call_ar_generator(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When MTP is requested with no sidecar path and fail_closed=True,
        the sequential generator must reject without calling the AR generator."""
        import exo.worker.runner.llm_inference.batch_generator as bg_module
        from exo.shared.types.tasks import TextGeneration
        from exo.shared.types.worker.instances import InstanceId

        def _noop_check(_params: TextGenerationTaskParams) -> None:
            return None

        def _fake_chat_template(
            _tokenizer: object,
            _params: TextGenerationTaskParams,
        ) -> str:
            return "prompt"

        monkeypatch.setattr(
            bg_module, "_check_for_debug_prompts", _noop_check
        )
        monkeypatch.setattr(
            bg_module, "apply_chat_template", _fake_chat_template
        )

        def fail_if_called(**_kwargs: object) -> object:
            raise AssertionError(
                "fail-closed missing-sidecar MTP request must not "
                "dispatch AR generation"
            )

        monkeypatch.setattr(bg_module, "mlx_generate", fail_if_called)

        task = TextGeneration(
            task_id=TaskId("12345678-1234-1234-1234-123456789abc"),
            command_id=CommandId("cmd-1"),
            instance_id=InstanceId("instance-1"),
            task_params=_mtp_task_params(
                sidecar_path=None, fail_closed=True
            ),
        )

        cancel_receiver = _CancelReceiver()
        event_sender = _EventSender()
        seq_gen = bg_module.SequentialGenerator(
            model=MagicMock(),
            tokenizer=MagicMock(),
            group=None,
            kv_prefix_cache=None,
            tool_parser=None,
            model_id=_MIMO_MODEL,
            device_rank=0,
            cancel_receiver=cancel_receiver,  # type: ignore[arg-type]
            event_sender=event_sender,  # type: ignore[arg-type]
        )
        result = next(seq_gen._build_generator(task))  # pyright: ignore[reportPrivateUsage]
        assert result.finish_reason == "error"
        assert "missing_sidecar" in result.text

    def test_sequential_fail_closed_missing_sidecar_file_rejection_does_not_call_ar_generator(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """When MTP is requested with a missing sidecar file and fail_closed=True,
        the sequential generator must reject without calling the AR generator."""
        import exo.worker.runner.llm_inference.batch_generator as bg_module
        from exo.shared.types.tasks import TextGeneration
        from exo.shared.types.worker.instances import InstanceId

        def _noop_check(_params: TextGenerationTaskParams) -> None:
            return None

        def _fake_chat_template(
            _tokenizer: object,
            _params: TextGenerationTaskParams,
        ) -> str:
            return "prompt"

        monkeypatch.setattr(
            bg_module, "_check_for_debug_prompts", _noop_check
        )
        monkeypatch.setattr(
            bg_module, "apply_chat_template", _fake_chat_template
        )

        def fail_if_called(**_kwargs: object) -> object:
            raise AssertionError(
                "fail-closed missing-sidecar MTP request must not "
                "dispatch AR generation"
            )

        monkeypatch.setattr(bg_module, "mlx_generate", fail_if_called)

        task = TextGeneration(
            task_id=TaskId("12345678-1234-1234-1234-123456789abc"),
            command_id=CommandId("cmd-1"),
            instance_id=InstanceId("instance-1"),
            task_params=_mtp_task_params_with_missing_sidecar_file(
                tmp_path, fail_closed=True
            ),
        )

        cancel_receiver = _CancelReceiver()
        event_sender = _EventSender()
        seq_gen = bg_module.SequentialGenerator(
            model=MagicMock(),
            tokenizer=MagicMock(),
            group=None,
            kv_prefix_cache=None,
            tool_parser=None,
            model_id=_MIMO_MODEL,
            device_rank=0,
            cancel_receiver=cancel_receiver,  # type: ignore[arg-type]
            event_sender=event_sender,  # type: ignore[arg-type]
        )
        result = next(seq_gen._build_generator(task))  # pyright: ignore[reportPrivateUsage]
        assert result.finish_reason == "error"
        assert "missing_sidecar" in result.text

    def test_sequential_fail_open_missing_sidecar_falls_back_to_ar(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When MTP is requested with no sidecar path and fail_closed=False,
        the sequential generator strips MTP intent and dispatches AR generation."""
        import exo.worker.runner.llm_inference.batch_generator as bg_module
        from exo.shared.types.tasks import TextGeneration
        from exo.shared.types.worker.instances import InstanceId
        from exo.shared.types.worker.runner_response import GenerationResponse

        def _noop_check(_params: TextGenerationTaskParams) -> None:
            return None

        def _fake_chat_template(
            _tokenizer: object,
            _params: TextGenerationTaskParams,
        ) -> str:
            return "prompt"

        monkeypatch.setattr(
            bg_module, "_check_for_debug_prompts", _noop_check
        )
        monkeypatch.setattr(
            bg_module, "apply_chat_template", _fake_chat_template
        )

        captured_task_params: list[TextGenerationTaskParams] = []

        def fake_mlx_generate(**kwargs: object) -> object:
            captured_task_params.append(
                cast(TextGenerationTaskParams, kwargs["task"])
            )

            def gen() -> object:
                yield GenerationResponse(
                    text="fallback-ar",
                    token=1,
                    finish_reason="stop",
                    usage=None,
                )

            return gen()

        monkeypatch.setattr(bg_module, "mlx_generate", fake_mlx_generate)

        task = TextGeneration(
            task_id=TaskId("12345678-1234-1234-1234-123456789abc"),
            command_id=CommandId("cmd-1"),
            instance_id=InstanceId("instance-1"),
            task_params=_mtp_task_params(
                sidecar_path=None, fail_closed=False
            ),
        )

        cancel_receiver = _CancelReceiver()
        event_sender = _EventSender()
        seq_gen = bg_module.SequentialGenerator(
            model=MagicMock(),
            tokenizer=MagicMock(),
            group=None,
            kv_prefix_cache=None,
            tool_parser=None,
            model_id=_MIMO_MODEL,
            device_rank=0,
            cancel_receiver=cancel_receiver,  # type: ignore[arg-type]
            event_sender=event_sender,  # type: ignore[arg-type]
        )
        result = next(seq_gen._build_generator(task))  # pyright: ignore[reportPrivateUsage]
        assert result.text == "fallback-ar"
        assert result.finish_reason == "stop"
        # Verify MTP intent was stripped — AR generation saw no MTP params
        assert len(captured_task_params) == 1
        assert captured_task_params[0].mimo_mtp_fastpath is None

    def test_batch_fail_closed_missing_sidecar_raises_runtime_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When MTP is requested with no sidecar path and fail_closed=True,
        the batch generator must raise RuntimeError before submitting to the engine."""
        import exo.worker.runner.llm_inference.batch_generator as bg_module
        from exo.shared.types.tasks import TextGeneration
        from exo.shared.types.worker.instances import InstanceId

        def _noop_check(_params: TextGenerationTaskParams) -> None:
            return None

        def _fake_chat_template(
            _tokenizer: object,
            _params: TextGenerationTaskParams,
        ) -> str:
            return "prompt"

        monkeypatch.setattr(
            bg_module, "_check_for_debug_prompts", _noop_check
        )
        monkeypatch.setattr(
            bg_module, "apply_chat_template", _fake_chat_template
        )

        task = TextGeneration(
            task_id=TaskId("12345678-1234-1234-1234-123456789abc"),
            command_id=CommandId("cmd-1"),
            instance_id=InstanceId("instance-1"),
            task_params=_mtp_task_params(
                sidecar_path=None, fail_closed=True
            ),
        )

        cancel_receiver = _CancelReceiver()
        event_sender = _EventSender()
        fake_engine = _FakeBatchEngine()
        batch_gen = bg_module.BatchGenerator(
            model=MagicMock(),
            tokenizer=MagicMock(),
            group=None,
            kv_prefix_cache=None,
            tool_parser=None,
            model_id=_MIMO_MODEL,
            device_rank=0,
            cancel_receiver=cancel_receiver,  # type: ignore[arg-type]
            event_sender=event_sender,  # type: ignore[arg-type]
        )
        batch_gen._mlx_gen = cast(Any, fake_engine)  # pyright: ignore[reportPrivateUsage]

        with pytest.raises(RuntimeError) as exc_info:
            batch_gen._start_task(task)  # pyright: ignore[reportPrivateUsage]
        assert "missing_sidecar" in str(exc_info.value)
        # Verify nothing was submitted to the engine
        assert len(fake_engine.submitted_params) == 0

    def test_batch_fail_open_missing_sidecar_strips_mtp_and_submits_ar(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When MTP is requested with no sidecar path and fail_closed=False,
        the batch generator strips MTP and submits AR generation."""
        import exo.worker.runner.llm_inference.batch_generator as bg_module
        from exo.shared.types.tasks import TextGeneration
        from exo.shared.types.worker.instances import InstanceId

        def _noop_check(_params: TextGenerationTaskParams) -> None:
            return None

        def _fake_chat_template(
            _tokenizer: object,
            _params: TextGenerationTaskParams,
        ) -> str:
            return "prompt"

        monkeypatch.setattr(
            bg_module, "_check_for_debug_prompts", _noop_check
        )
        monkeypatch.setattr(
            bg_module, "apply_chat_template", _fake_chat_template
        )

        task = TextGeneration(
            task_id=TaskId("12345678-1234-1234-1234-123456789abc"),
            command_id=CommandId("cmd-1"),
            instance_id=InstanceId("instance-1"),
            task_params=_mtp_task_params(
                sidecar_path=None, fail_closed=False
            ),
        )

        cancel_receiver = _CancelReceiver()
        event_sender = _EventSender()
        fake_engine = _FakeBatchEngine()
        batch_gen = bg_module.BatchGenerator(
            model=MagicMock(),
            tokenizer=MagicMock(),
            group=None,
            kv_prefix_cache=None,
            tool_parser=None,
            model_id=_MIMO_MODEL,
            device_rank=0,
            cancel_receiver=cancel_receiver,  # type: ignore[arg-type]
            event_sender=event_sender,  # type: ignore[arg-type]
        )
        batch_gen._mlx_gen = cast(Any, fake_engine)  # pyright: ignore[reportPrivateUsage]
        batch_gen._start_task(task)  # pyright: ignore[reportPrivateUsage]

        assert len(fake_engine.submitted_params) == 1
        assert fake_engine.submitted_params[0].mimo_mtp_fastpath is None


# ===========================================================================
# Cross-layer invariant: no layer ever claims MTP for missing-sidecar
# ===========================================================================


class TestCrossLayerNoSilentMtpClaim:
    """Invariant: across all layers, no missing-sidecar MTP request ever
    produces a result that claims successful MTP execution."""

    def test_classifier_invariant(self) -> None:
        params = _mtp_task_params(sidecar_path=None)
        result = classify_mimo_mtp_request(params)
        _assert_no_silent_mtp_claim(
            mtp_enabled=result.mtp_enabled,
            context="classifier invariant for missing-sidecar",
        )
        assert result.label != "compatible_mtp"
        assert result.label != "successful_mtp"

    def test_validator_invariant(self) -> None:
        params = _mtp_task_params(sidecar_path=None)
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)
        assert isinstance(result, MTPDisabledReason)
        _assert_no_silent_mtp_claim(
            mtp_enabled=result.mtp_enabled,
            context="validator invariant for missing-sidecar",
        )

    def test_worker_fastpath_fail_closed_invariant(self) -> None:
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params(sidecar_path=None, fail_closed=True),
        )
        _assert_no_silent_mtp_claim(
            mtp_enabled=decision.mtp_enabled,
            accepted_execution_path=decision.accepted_execution_path,
            context="worker-fastpath fail-closed invariant for missing-sidecar",
        )
        assert decision.should_use_mtp is False
        assert decision.sidecar is None
        assert decision.mtp_depth is None

    def test_worker_fastpath_fail_open_invariant(self) -> None:
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params(sidecar_path=None, fail_closed=False),
        )
        _assert_no_silent_mtp_claim(
            mtp_enabled=decision.mtp_enabled,
            accepted_execution_path=decision.accepted_execution_path,
            context="worker-fastpath fail-open invariant for missing-sidecar",
        )
        assert decision.should_use_mtp is False
        assert decision.sidecar is None

    def test_worker_fastpath_missing_file_fail_closed_invariant(
        self, tmp_path: Path
    ) -> None:
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params_with_missing_sidecar_file(
                tmp_path, fail_closed=True
            ),
        )
        _assert_no_silent_mtp_claim(
            mtp_enabled=decision.mtp_enabled,
            accepted_execution_path=decision.accepted_execution_path,
            context="worker-fastpath fail-closed missing-file invariant",
        )

    def test_worker_fastpath_missing_file_fail_open_invariant(
        self, tmp_path: Path
    ) -> None:
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params_with_missing_sidecar_file(
                tmp_path, fail_closed=False
            ),
        )
        _assert_no_silent_mtp_claim(
            mtp_enabled=decision.mtp_enabled,
            accepted_execution_path=decision.accepted_execution_path,
            context="worker-fastpath fail-open missing-file invariant",
        )

    def test_error_detail_invariant(self) -> None:
        """The error detail dict never claims successful MTP."""
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params(sidecar_path=None, fail_closed=True),
        )
        with pytest.raises(MimoMtpMissingSidecarError) as exc_info:
            decision.raise_fail_closed_error()
        detail = exc_info.value.detail()
        _assert_no_silent_mtp_claim(
            mtp_enabled=cast(bool | None, detail.get("mtp_enabled")),
            accepted_execution_path=cast(
                str | None, detail.get("accepted_execution_path")
            ),
            mtp_execution_state=cast(
                str | None, detail.get("mtp_execution_state")
            ),
            context="error detail invariant for missing-sidecar",
        )

    def test_telemetry_no_successful_mtp_state(self) -> None:
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params(sidecar_path=None, fail_closed=True),
        )
        for key, value in decision.telemetry.items():
            if isinstance(value, str):
                assert value != "successful_mtp", (
                    f"telemetry[{key}] for missing-sidecar must not be "
                    f"'successful_mtp': {value}"
                )


# ===========================================================================
# Positive control: MiMo models with sidecar provided are NOT missing-sidecar
# ===========================================================================


class TestMimoModelWithSidecarNotMissingSidecar:
    """Positive control: MiMo models with a sidecar path provided are NOT
    classified as missing_sidecar, proving the guard is sidecar-specific."""

    def test_mimo_model_with_sidecar_classifies_as_unwired(self) -> None:
        params = _mtp_task_params(
            sidecar_path="/models/model_mtp.safetensors"
        )
        result = classify_mimo_mtp_request(params)
        assert result.label == "unwired_execution"
        assert result.label != "missing_sidecar"

    def test_mimo_model_with_sidecar_validates_as_validated_intent(self) -> None:
        params = _mtp_task_params(
            sidecar_path="/models/model_mtp.safetensors"
        )
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)
        assert isinstance(result, MTPValidatedIntent)

    def test_mimo_model_with_sidecar_worker_fastpath_not_missing_sidecar(
        self, tmp_path: Path
    ) -> None:
        """With a real sidecar file, the worker fastpath does not trigger
        missing_sidecar — it proceeds to subsequent guard checks."""
        from exo.worker.engines.mlx.tests.test_mimo_mtp_fast_worker_fastpath import (
            _write_synthetic_official_sidecar,
        )

        sidecar_path = tmp_path / "model_mtp.safetensors"
        _write_synthetic_official_sidecar(sidecar_path)
        params = TextGenerationTaskParams(
            model=_MIMO_MODEL,
            input=[InputMessage(role="user", content="Hello")],
            max_output_tokens=4,
            mimo_mtp_fastpath_params=MimoMtpFastpathParams(
                enabled=True,
                depth=1,
                sidecar_path=str(sidecar_path),
                fail_closed=True,
            ),
        )
        decision = evaluate_mimo_mtp_worker_fastpath(
            params,
            cache=MimoMtpWorkerFastpathCache(),
        )
        assert decision.disable_reason != "missing_sidecar"
        # It will fail on a subsequent guard (unwired execution, runtime, etc.)
        # but NOT on missing_sidecar


# ===========================================================================
# Default AR preservation: MiMo models without MTP intent are normal AR
# ===========================================================================


class TestMimoDefaultArPreserved:
    """MiMo models without MTP intent must produce normal AR behavior
    (no error, no rejection, no MTP telemetry)."""

    def test_mimo_ar_classifier_returns_disabled_default(self) -> None:
        params = _ar_task_params()
        result = classify_mimo_mtp_request(params)
        assert result.label == "disabled_default"
        assert result.mtp_enabled is False

    def test_mimo_ar_validator_returns_disabled_reason_mtp_not_requested(
        self,
    ) -> None:
        params = _ar_task_params()
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)
        assert isinstance(result, MTPDisabledReason)
        assert result.disable_reason == "mtp_not_requested"
        # This is "no MTP intent", not an error condition
        assert result.classification_label == "disabled_default"

    def test_mimo_ar_worker_fastpath_returns_ar_decision(self) -> None:
        decision = evaluate_mimo_mtp_worker_fastpath(
            _ar_task_params(),
        )
        assert decision.accepted_execution_path == "ar"
        assert decision.should_use_mtp is False
        assert decision.mtp_enabled is False
        assert decision.disable_reason is None
        assert decision.error_message is None


# ===========================================================================
# Fail-open exception boundary: fail-open without opt-in still protects
# ===========================================================================


class TestFailOpenExceptionBoundary:
    """The fail-open exception boundary for missing-sidecar: fail-open is
    only permissible for explicitly documented experimental modes with user
    opt-in. Default and production paths must always be fail-closed.

    These tests verify:
    1. Default fail_closed=True produces rejection (fail-closed)
    2. Explicit fail_closed=False with opt-in produces AR fallback (fail-open)
    3. The boundary between fail-closed and fail-open is clear and typed
    """

    def test_default_fail_closed_produces_rejection(self) -> None:
        """Default fail_closed=True must produce a rejected decision,
        never an AR fallback."""
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params(sidecar_path=None, fail_closed=True),
        )
        assert decision.accepted_execution_path == "rejected"
        assert decision.error_message is not None
        assert decision.fallback_reason is None
        # The typed error must be raisable
        with pytest.raises(MimoMtpMissingSidecarError):
            decision.raise_fail_closed_error()

    def test_explicit_fail_open_produces_ar_fallback(self) -> None:
        """Explicit fail_closed=False produces AR fallback with telemetry,
        not a rejection."""
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params(sidecar_path=None, fail_closed=False),
        )
        assert decision.accepted_execution_path == "ar"
        assert decision.error_message is None
        assert decision.fallback_reason == "fail_open_missing_sidecar"
        # The typed error must NOT be raisable (not a rejected decision)
        with pytest.raises(ValueError, match="non-rejected decision"):
            decision.raise_fail_closed_error()

    def test_fail_closed_rejection_is_typed_as_missing_sidecar_error(
        self,
    ) -> None:
        """The fail-closed rejection produces MimoMtpMissingSidecarError,
        not the generic MimoMtpFailClosedError base class."""
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params(sidecar_path=None, fail_closed=True),
        )
        with pytest.raises(MimoMtpMissingSidecarError) as exc_info:
            decision.raise_fail_closed_error()
        # Must be the specific subclass, not just the base
        assert type(exc_info.value) is MimoMtpMissingSidecarError
        assert isinstance(exc_info.value, MimoMtpFailClosedError)

    def test_fail_open_telemetry_records_disable_and_fallback_reasons(
        self,
    ) -> None:
        """Fail-open decision records both disable_reason and fallback_reason
        for diagnosability."""
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params(sidecar_path=None, fail_closed=False, depth=2),
        )
        assert decision.disable_reason == "missing_sidecar"
        assert decision.fallback_reason == "fail_open_missing_sidecar"
        assert decision.telemetry["mtp_disable_reason"] == "missing_sidecar"
        assert (
            decision.telemetry["mtp_fallback_reason"]
            == "fail_open_missing_sidecar"
        )

    def test_fail_closed_telemetry_records_disable_and_reject_reasons(
        self,
    ) -> None:
        """Fail-closed decision records both disable_reason and reject_reason
        for diagnosability."""
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params(sidecar_path=None, fail_closed=True, depth=2),
        )
        assert decision.disable_reason == "missing_sidecar"
        assert decision.reject_reason == "missing_sidecar"
        assert decision.telemetry["mtp_disable_reason"] == "missing_sidecar"
        assert decision.telemetry["mtp_reject_reason"] == "missing_sidecar"
        assert decision.fallback_reason is None


# ===========================================================================
# BenchmarkRow compatibility: missing-sidecar state is valid in telemetry
# ===========================================================================


class TestBenchmarkRowMissingSidecarState:
    """Missing-sidecar telemetry values are valid BenchmarkRow fields."""

    def test_missing_sidecar_is_valid_mtp_execution_state(self) -> None:
        """'missing_sidecar' is a valid MtpExecutionState value for benchmark rows."""
        from exo.shared.types.benchmark_telemetry import BenchmarkRow

        row = BenchmarkRow(
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            mode="ar",
            mtp_enabled=False,
            mtp_execution_state="missing_sidecar",
            mtp_disable_reason="missing_sidecar",
            mtp_sidecar_status="missing",
        )
        assert row.mtp_execution_state == "missing_sidecar"
        assert row.mtp_enabled is False
        assert row.mode == "ar"
        assert row.mtp_disable_reason == "missing_sidecar"

    def test_fail_closed_error_is_valid_mtp_execution_state(self) -> None:
        """'fail_closed_error' is a valid MtpExecutionState for benchmark rows
        when the error is not further classified."""
        from exo.shared.types.benchmark_telemetry import BenchmarkRow

        row = BenchmarkRow(
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            mode="ar",
            mtp_enabled=False,
            mtp_execution_state="fail_closed_error",
            mtp_disable_reason="missing_sidecar",
        )
        assert row.mtp_execution_state == "fail_closed_error"
