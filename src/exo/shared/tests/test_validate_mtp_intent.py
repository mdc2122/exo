"""Unit tests for validate_mtp_intent (Sub-AC 2a).

Verifies:
- A valid MTP-intent request (unwired_execution or compatible_mtp classification)
  returns MTPValidatedIntent.
- All disabled/unsupported classifications return MTPDisabledReason.
- Discriminated-union result_type field enables pattern-matching.
- MTPValidatedIntent is frozen, strict, extra-forbid.
- MTPDisabledReason is frozen, strict, extra-forbid, with disable_reason always populated.
- The pure function is deterministic: same input → same output.
- Round-trip serialization for both result types.
- MTPValidatedIntent properties: is_validated_intent=True, is_disabled_reason=False.
- MTPDisabledReason properties: is_validated_intent=False, is_disabled_reason=True.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from pydantic import ValidationError

from exo.shared.models.model_cards import (
    MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
    MIMO_V25_PRO_MODEL_IDS,
)
from exo.shared.types.common import ModelId
from exo.shared.types.mimo_mtp_classifier import classify_mimo_mtp_request
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NON_MIMO_MODEL = ModelId("mlx-community/Llama-3.3-70B-Instruct-4bit")


def _ar_task_params(
    model: ModelId = MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
) -> TextGenerationTaskParams:
    """Default AR request: no MTP intent."""
    return TextGenerationTaskParams(
        model=model,
        input=[InputMessage(role="user", content="Hello")],
        max_output_tokens=4,
    )


def _mtp_task_params(
    *,
    model: ModelId = MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
    depth: int | None = 1,
    sidecar_path: str | None = "/models/model_mtp.safetensors",
    fail_closed: bool = True,
    enabled: bool = True,
) -> TextGenerationTaskParams:
    """MTP-intent request with configurable fields."""
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


# ---------------------------------------------------------------------------
# Core AC test: valid MTP-intent request returns MTPValidatedIntent
# ---------------------------------------------------------------------------


class TestValidMtpIntentReturnsValidatedIntent:
    """A valid MTP-intent request must return MTPValidatedIntent.

    This is the primary acceptance criterion test for Sub-AC 2a.
    A valid MTP-intent request is one where:
    - MTP is explicitly enabled
    - The model is a MiMo V2.5 Pro variant
    - The depth is supported (1, 2, 3) or None for default
    - A sidecar path is provided
    - The static classifier returns unwired_execution (because it cannot
      verify runtime wiring, but the intent is valid)
    """

    def test_valid_mtp_intent_returns_validated_intent(self) -> None:
        """A valid MTP-intent request returns MTPValidatedIntent."""
        params = _mtp_task_params(
            depth=1,
            sidecar_path="/models/model_mtp.safetensors",
            fail_closed=True,
        )
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)

        assert isinstance(result, MTPValidatedIntent)
        assert result.result_type == "validated_intent"
        assert result.is_validated_intent is True
        assert result.is_disabled_reason is False

    def test_valid_mtp_intent_carries_model_id(self) -> None:
        params = _mtp_task_params()
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)

        assert isinstance(result, MTPValidatedIntent)
        assert result.model_id == str(MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID)

    def test_valid_mtp_intent_carries_depth(self) -> None:
        params = _mtp_task_params(depth=2)
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)

        assert isinstance(result, MTPValidatedIntent)
        assert result.requested_depth == 2

    def test_valid_mtp_intent_carries_none_depth(self) -> None:
        """None depth (default) is a valid MTP intent."""
        params = _mtp_task_params(depth=None)
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)

        assert isinstance(result, MTPValidatedIntent)
        assert result.requested_depth is None

    def test_valid_mtp_intent_sidecar_path_provided(self) -> None:
        params = _mtp_task_params()
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)

        assert isinstance(result, MTPValidatedIntent)
        assert result.sidecar_path_provided is True

    def test_valid_mtp_intent_fail_closed_true(self) -> None:
        params = _mtp_task_params(fail_closed=True)
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)

        assert isinstance(result, MTPValidatedIntent)
        assert result.fail_closed is True

    def test_valid_mtp_intent_fail_closed_false(self) -> None:
        """Even fail_open intent is a valid MTP intent (for experimental mode)."""
        params = _mtp_task_params(fail_closed=False)
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)

        assert isinstance(result, MTPValidatedIntent)
        assert result.fail_closed is False

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_all_supported_depths_produce_validated_intent(
        self, depth: int
    ) -> None:
        """Each supported depth produces MTPValidatedIntent."""
        params = _mtp_task_params(depth=depth)
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)

        assert isinstance(result, MTPValidatedIntent)
        assert result.requested_depth == depth

    def test_valid_mtp_intent_classification_label_is_unwired(self) -> None:
        """The static classifier produces unwired_execution for valid MTP intent."""
        params = _mtp_task_params()
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)

        assert isinstance(result, MTPValidatedIntent)
        assert result.classification_label == "unwired_execution"

    def test_valid_mtp_intent_for_each_mimo_model_id(self) -> None:
        """Every MiMo V2.5 Pro variant with MTP intent produces MTPValidatedIntent."""
        for model_id in MIMO_V25_PRO_MODEL_IDS:
            params = _mtp_task_params(model=model_id)
            classification = classify_mimo_mtp_request(params)
            result = validate_mtp_intent(classification)

            assert isinstance(result, MTPValidatedIntent), (
                f"model {model_id} with MTP intent should produce MTPValidatedIntent"
            )
            assert result.model_id == str(model_id)


# ---------------------------------------------------------------------------
# disabled_default → MTPDisabledReason
# ---------------------------------------------------------------------------


class TestDisabledDefaultReturnsDisabledReason:
    """No MTP intent produces MTPDisabledReason with mtp_not_requested."""

    def test_no_mtp_params_returns_disabled_reason(self) -> None:
        params = _ar_task_params()
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)

        assert isinstance(result, MTPDisabledReason)
        assert result.result_type == "disabled_reason"
        assert result.disable_reason == "mtp_not_requested"
        assert result.is_validated_intent is False
        assert result.is_disabled_reason is True

    def test_explicit_none_mtp_returns_disabled_reason(self) -> None:
        params = TextGenerationTaskParams(
            model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
            input=[InputMessage(role="user", content="Hello")],
            max_output_tokens=4,
            mimo_mtp_fastpath_params=None,
        )
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)

        assert isinstance(result, MTPDisabledReason)
        assert result.disable_reason == "mtp_not_requested"

    def test_disabled_enabled_flag_returns_disabled_reason(self) -> None:
        params = _mtp_task_params(enabled=False)
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)

        assert isinstance(result, MTPDisabledReason)
        assert result.disable_reason == "mtp_not_requested"

    def test_disabled_default_mtp_enabled_is_false(self) -> None:
        params = _ar_task_params()
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)

        assert isinstance(result, MTPDisabledReason)
        assert result.mtp_enabled is False

    def test_disabled_default_classification_label(self) -> None:
        params = _ar_task_params()
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)

        assert isinstance(result, MTPDisabledReason)
        assert result.classification_label == "disabled_default"


# ---------------------------------------------------------------------------
# unsupported_model → MTPDisabledReason
# ---------------------------------------------------------------------------


class TestUnsupportedModelReturnsDisabledReason:
    """MTP requested for non-MiMo model produces MTPDisabledReason."""

    def test_non_mimo_model_returns_disabled_reason(self) -> None:
        params = _mtp_task_params(model=_NON_MIMO_MODEL)
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)

        assert isinstance(result, MTPDisabledReason)
        assert result.disable_reason == "unsupported_model"
        assert result.classification_label == "unsupported_model"

    def test_non_mimo_model_mtp_enabled_is_false(self) -> None:
        params = _mtp_task_params(model=_NON_MIMO_MODEL)
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)

        assert isinstance(result, MTPDisabledReason)
        assert result.mtp_enabled is False


# ---------------------------------------------------------------------------
# unsupported_depth → MTPDisabledReason
# ---------------------------------------------------------------------------


class TestUnsupportedDepthReturnsDisabledReason:
    """MTP requested with unsupported depth produces MTPDisabledReason."""

    def test_unsupported_depth_returns_disabled_reason(self) -> None:
        """Construct a classification with unsupported depth (bypassing Pydantic validator)."""
        base_params = _mtp_task_params(depth=1)
        raw_mtp = base_params.mimo_mtp_fastpath
        assert raw_mtp is not None
        mtp_with_bad_depth = raw_mtp.model_construct(
            enabled=True,
            depth=4,
            sidecar_path="/models/model_mtp.safetensors",
            fail_closed=True,
        )
        params = base_params.model_copy(
            update={"mimo_mtp_fastpath_params": mtp_with_bad_depth}
        )
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)

        assert isinstance(result, MTPDisabledReason)
        assert result.disable_reason == "unsupported_depth"
        assert result.classification_label == "unsupported_depth"


# ---------------------------------------------------------------------------
# missing_sidecar → MTPDisabledReason
# ---------------------------------------------------------------------------


class TestMissingSidecarReturnsDisabledReason:
    """MTP requested without sidecar path produces MTPDisabledReason."""

    def test_missing_sidecar_returns_disabled_reason(self) -> None:
        params = _mtp_task_params(sidecar_path=None)
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)

        assert isinstance(result, MTPDisabledReason)
        assert result.disable_reason == "missing_sidecar"
        assert result.classification_label == "missing_sidecar"

    def test_missing_sidecar_sidecar_path_provided_is_false(self) -> None:
        params = _mtp_task_params(sidecar_path=None)
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)

        assert isinstance(result, MTPDisabledReason)
        assert result.sidecar_path_provided is False


# ---------------------------------------------------------------------------
# Result type model constraints
# ---------------------------------------------------------------------------


class TestMTPValidatedIntentModel:
    """MTPValidatedIntent is frozen, strict, extra-forbid."""

    def _sample_validated_intent(self) -> MTPValidatedIntent:
        params = _mtp_task_params()
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)
        assert isinstance(result, MTPValidatedIntent)
        return result

    def test_frozen_rejects_mutation(self) -> None:
        result = self._sample_validated_intent()
        with pytest.raises(ValidationError):
            result.model_id = "mutated"

    def test_strict_rejects_coerced_types(self) -> None:
        with pytest.raises(ValidationError):
            MTPValidatedIntent(
                model_id="test-model",
                requested_depth=1,
                sidecar_path_provided="true",  # type: ignore[arg-type]
                fail_closed=True,
                classification_label="unwired_execution",
            )

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MTPValidatedIntent(
                model_id="test-model",
                requested_depth=1,
                sidecar_path_provided=True,
                fail_closed=True,
                classification_label="unwired_execution",
                unexpected_field=True,  # type: ignore[call-arg]
            )

    def test_result_type_is_validated_intent(self) -> None:
        result = self._sample_validated_intent()
        assert result.result_type == "validated_intent"

    def test_round_trip_serialization(self) -> None:
        result = self._sample_validated_intent()
        dumped = result.model_dump()
        restored = MTPValidatedIntent.model_validate(dumped)
        assert restored == result

    def test_json_round_trip_serialization(self) -> None:
        result = self._sample_validated_intent()
        json_str = result.model_dump_json()
        parsed = cast(dict[str, Any], json.loads(json_str))
        restored = MTPValidatedIntent.model_validate(parsed)
        assert restored == result


class TestMTPDisabledReasonModel:
    """MTPDisabledReason is frozen, strict, extra-forbid, disable_reason always populated."""

    def _sample_disabled_reason(self) -> MTPDisabledReason:
        params = _ar_task_params()
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)
        assert isinstance(result, MTPDisabledReason)
        return result

    def test_frozen_rejects_mutation(self) -> None:
        result = self._sample_disabled_reason()
        with pytest.raises(ValidationError):
            result.disable_reason = "mutated"

    def test_strict_rejects_coerced_types(self) -> None:
        with pytest.raises(ValidationError):
            MTPDisabledReason(
                model_id="test-model",
                mtp_enabled="false",  # type: ignore[arg-type]
                requested_depth=None,
                sidecar_path_provided=False,
                fail_closed=None,
                disable_reason="test",
                classification_label="disabled_default",
            )

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MTPDisabledReason(
                model_id="test-model",
                mtp_enabled=False,
                requested_depth=None,
                sidecar_path_provided=False,
                fail_closed=None,
                disable_reason="test",
                classification_label="disabled_default",
                unexpected_field=True,  # type: ignore[call-arg]
            )

    def test_disable_reason_always_populated(self) -> None:
        """Every MTPDisabledReason must have a non-empty disable_reason."""
        result = self._sample_disabled_reason()
        assert result.disable_reason is not None
        assert result.disable_reason != ""

    def test_result_type_is_disabled_reason(self) -> None:
        result = self._sample_disabled_reason()
        assert result.result_type == "disabled_reason"

    def test_round_trip_serialization(self) -> None:
        result = self._sample_disabled_reason()
        dumped = result.model_dump()
        restored = MTPDisabledReason.model_validate(dumped)
        assert restored == result

    def test_json_round_trip_serialization(self) -> None:
        result = self._sample_disabled_reason()
        json_str = result.model_dump_json()
        parsed = cast(dict[str, Any], json.loads(json_str))
        restored = MTPDisabledReason.model_validate(parsed)
        assert restored == result


# ---------------------------------------------------------------------------
# Discriminated-union dispatch: result_type enables pattern-matching
# ---------------------------------------------------------------------------


class TestDiscriminatedUnionDispatch:
    """The result_type discriminant field enables pattern-matching."""

    def test_validated_intent_dispatches_on_result_type(self) -> None:
        params = _mtp_task_params()
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)

        match result.result_type:
            case "validated_intent":
                assert isinstance(result, MTPValidatedIntent)
            case "disabled_reason":
                pytest.fail("Expected validated_intent, got disabled_reason")

    def test_disabled_reason_dispatches_on_result_type(self) -> None:
        params = _ar_task_params()
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)

        match result.result_type:
            case "disabled_reason":
                assert isinstance(result, MTPDisabledReason)
            case "validated_intent":
                pytest.fail("Expected disabled_reason, got validated_intent")

    def test_isinstance_check_works_for_validated_intent(self) -> None:
        params = _mtp_task_params()
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)

        if isinstance(result, MTPValidatedIntent):
            assert result.fail_closed is not None
        else:
            pytest.fail("Expected MTPValidatedIntent")

    def test_isinstance_check_works_for_disabled_reason(self) -> None:
        params = _ar_task_params()
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)

        if isinstance(result, MTPDisabledReason):
            assert result.disable_reason is not None
        else:
            pytest.fail("Expected MTPDisabledReason")


# ---------------------------------------------------------------------------
# Determinism: same input → same output
# ---------------------------------------------------------------------------


class TestDeterminism:
    """validate_mtp_intent is a pure function: same input → same output."""

    def test_deterministic_for_validated_intent(self) -> None:
        params = _mtp_task_params()
        classification = classify_mimo_mtp_request(params)
        a = validate_mtp_intent(classification)
        b = validate_mtp_intent(classification)
        assert a == b

    def test_deterministic_for_disabled_reason(self) -> None:
        params = _ar_task_params()
        classification = classify_mimo_mtp_request(params)
        a = validate_mtp_intent(classification)
        b = validate_mtp_intent(classification)
        assert a == b

    def test_deterministic_for_unsupported_model(self) -> None:
        params = _mtp_task_params(model=_NON_MIMO_MODEL)
        classification = classify_mimo_mtp_request(params)
        a = validate_mtp_intent(classification)
        b = validate_mtp_intent(classification)
        assert a == b


# ---------------------------------------------------------------------------
# Invariants across all classification paths
# ---------------------------------------------------------------------------


class TestValidationInvariants:
    """Invariants that hold across all validation paths."""

    def test_result_type_is_never_none(self) -> None:
        """Every result has a non-None result_type."""
        for params in [
            _ar_task_params(),
            _mtp_task_params(),
            _mtp_task_params(model=_NON_MIMO_MODEL),
            _mtp_task_params(sidecar_path=None),
        ]:
            classification = classify_mimo_mtp_request(params)
            result = validate_mtp_intent(classification)
            assert result.result_type in ("validated_intent", "disabled_reason")

    def test_model_id_always_matches_classification(self) -> None:
        """The result's model_id always matches the classification's model_id."""
        for params in [
            _ar_task_params(),
            _mtp_task_params(),
            _mtp_task_params(model=_NON_MIMO_MODEL),
        ]:
            classification = classify_mimo_mtp_request(params)
            result = validate_mtp_intent(classification)
            assert result.model_id == classification.model_id

    def test_classification_label_always_preserved(self) -> None:
        """The classification_label on the result matches the original classification."""
        for params in [
            _ar_task_params(),
            _mtp_task_params(),
            _mtp_task_params(model=_NON_MIMO_MODEL),
            _mtp_task_params(sidecar_path=None),
        ]:
            classification = classify_mimo_mtp_request(params)
            result = validate_mtp_intent(classification)
            assert result.classification_label == classification.label

    def test_no_result_is_both_validated_and_disabled(self) -> None:
        """A result cannot be both validated intent and disabled reason."""
        for params in [
            _ar_task_params(),
            _mtp_task_params(),
            _mtp_task_params(model=_NON_MIMO_MODEL),
        ]:
            classification = classify_mimo_mtp_request(params)
            result = validate_mtp_intent(classification)
            assert result.is_validated_intent != result.is_disabled_reason
