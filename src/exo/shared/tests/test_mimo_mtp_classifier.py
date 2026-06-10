"""Unit tests for the typed MiMo MTP request classifier (Sub-AC 1).

Verifies:
- Each classification path returns the correct label and discriminant.
- Edge cases for non-MiMo requests.
- Discriminated result properties (is_compatible, is_unsupported, is_unwired).
- Result immutability (frozen), strict mode, and extra-field rejection.
- disabled_default: no MTP intent (params absent or enabled=False).
- unsupported_model: MTP requested for a non-MiMo model.
- unsupported_depth: MTP requested with an out-of-range depth.
- missing_sidecar: MTP requested for MiMo model without sidecar path.
- unwired_execution: all static fields present but runtime not verified.
- All supported depths (1, 2, 3) are classified as unwired_execution
  (statically compatible) when sidecar path is provided.
- None depth is treated as potentially compatible.
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
from exo.shared.types.mimo_mtp_classifier import (
    DISCRIMINANT_BY_LABEL,
    MimoMtpClassificationLabel,
    MimoMtpRequestClassification,
    classify_mimo_mtp_request,
)
from exo.shared.types.text_generation import (
    SUPPORTED_MIMO_MTP_FASTPATH_DEPTHS,
    InputMessage,
    MimoMtpFastpathParams,
    TextGenerationTaskParams,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NON_MIMO_MODEL = ModelId("mlx-community/Llama-3.3-70B-Instruct-4bit")

_NON_MIMO_MODEL_STR = "mlx-community/Llama-3.3-70B-Instruct-4bit"


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
# disabled_default: no MTP intent
# ---------------------------------------------------------------------------

class TestDisabledDefault:
    """When no MTP intent is present, the classifier returns disabled_default."""

    def test_no_mtp_params_returns_disabled_default(self) -> None:
        params = _ar_task_params()
        result = classify_mimo_mtp_request(params)
        assert result.label == "disabled_default"
        assert result.discriminant == "unsupported"
        assert result.mtp_enabled is False
        assert result.requested_depth is None
        assert result.sidecar_path_provided is False
        assert result.fail_closed is None
        assert result.disable_reason == "mtp_not_requested"

    def test_explicit_none_mtp_params_returns_disabled_default(self) -> None:
        params = TextGenerationTaskParams(
            model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
            input=[InputMessage(role="user", content="Hello")],
            max_output_tokens=4,
            mimo_mtp_fastpath_params=None,
        )
        result = classify_mimo_mtp_request(params)
        assert result.label == "disabled_default"
        assert result.discriminant == "unsupported"

    def test_disabled_enabled_flag_returns_disabled_default(self) -> None:
        params = _mtp_task_params(enabled=False)
        result = classify_mimo_mtp_request(params)
        assert result.label == "disabled_default"
        assert result.discriminant == "unsupported"
        assert result.mtp_enabled is False

    def test_disabled_default_for_non_mimo_model(self) -> None:
        """AR request for a non-MiMo model is still disabled_default (no MTP intent)."""
        params = _ar_task_params(model=_NON_MIMO_MODEL)
        result = classify_mimo_mtp_request(params)
        assert result.label == "disabled_default"
        assert result.model_id == _NON_MIMO_MODEL_STR

    def test_disabled_default_is_unsupported(self) -> None:
        params = _ar_task_params()
        result = classify_mimo_mtp_request(params)
        assert result.is_unsupported is True
        assert result.is_compatible is False
        assert result.is_unwired is False


# ---------------------------------------------------------------------------
# unsupported_model: MTP requested for non-MiMo model
# ---------------------------------------------------------------------------

class TestUnsupportedModel:
    """When MTP is requested for a non-MiMo model, the classifier returns unsupported_model."""

    def test_non_mimo_model_with_mtp_intent(self) -> None:
        params = _mtp_task_params(model=_NON_MIMO_MODEL)
        result = classify_mimo_mtp_request(params)
        assert result.label == "unsupported_model"
        assert result.discriminant == "unsupported"
        assert result.mtp_enabled is False
        assert result.model_id == _NON_MIMO_MODEL_STR
        assert result.disable_reason == "unsupported_model"

    def test_non_mimo_model_mtp_fail_closed(self) -> None:
        params = _mtp_task_params(model=_NON_MIMO_MODEL, fail_closed=True)
        result = classify_mimo_mtp_request(params)
        assert result.label == "unsupported_model"
        assert result.fail_closed is True

    def test_non_mimo_model_mtp_fail_open(self) -> None:
        params = _mtp_task_params(model=_NON_MIMO_MODEL, fail_closed=False)
        result = classify_mimo_mtp_request(params)
        assert result.label == "unsupported_model"
        assert result.fail_closed is False

    @pytest.mark.parametrize(
        "model_id_str",
        [
            "mlx-community/Llama-3.3-70B-Instruct-4bit",
            "Qwen/Qwen3-30B-A3B",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "some-org/some-model",
        ],
    )
    def test_various_non_mimo_models_are_unsupported(self, model_id_str: str) -> None:
        params = _mtp_task_params(model=ModelId(model_id_str))
        result = classify_mimo_mtp_request(params)
        assert result.label == "unsupported_model"
        assert result.model_id == model_id_str

    def test_unsupported_model_is_unsupported(self) -> None:
        params = _mtp_task_params(model=_NON_MIMO_MODEL)
        result = classify_mimo_mtp_request(params)
        assert result.is_unsupported is True
        assert result.is_compatible is False
        assert result.is_unwired is False

    def test_non_mimo_model_reports_requested_depth(self) -> None:
        params = _mtp_task_params(model=_NON_MIMO_MODEL, depth=3)
        result = classify_mimo_mtp_request(params)
        assert result.requested_depth == 3


# ---------------------------------------------------------------------------
# unsupported_depth: MTP requested with unsupported depth
# ---------------------------------------------------------------------------

class TestUnsupportedDepth:
    """When MTP is requested for a MiMo model with an unsupported depth,
    the classifier returns unsupported_depth."""

    @pytest.mark.parametrize("unsupported_depth", [0, 4, 5, -1, 10, 100])
    def test_unsupported_depth_values_are_classified(
        self, unsupported_depth: int
    ) -> None:
        # Note: MimoMtpFastpathParams validates depth at construction time
        # and rejects unsupported values. We test the classifier's logic
        # directly by constructing params that bypass the validator, or we
        # test that the classifier handles the case if it ever receives
        # such a depth. Since Pydantic prevents construction with invalid
        # depths, we use model_construct to bypass validation for this test.
        base_params = _mtp_task_params(depth=1)
        # Bypass validation to inject an unsupported depth
        raw_mtp = base_params.mimo_mtp_fastpath
        assert raw_mtp is not None
        mtp_with_bad_depth = raw_mtp.model_construct(
            enabled=True,
            depth=unsupported_depth,
            sidecar_path="/models/model_mtp.safetensors",
            fail_closed=True,
        )
        params = base_params.model_copy(
            update={"mimo_mtp_fastpath_params": mtp_with_bad_depth}
        )
        result = classify_mimo_mtp_request(params)
        assert result.label == "unsupported_depth"
        assert result.discriminant == "unsupported"
        assert result.mtp_enabled is False
        assert result.requested_depth == unsupported_depth
        assert result.disable_reason == "unsupported_depth"
        assert result.supported_depths == tuple(sorted(SUPPORTED_MIMO_MTP_FASTPATH_DEPTHS))

    def test_unsupported_depth_reports_supported_depths(self) -> None:
        base_params = _mtp_task_params(depth=1)
        raw_mtp = base_params.mimo_mtp_fastpath
        assert raw_mtp is not None
        mtp_with_bad_depth = raw_mtp.model_construct(
            enabled=True, depth=4,
            sidecar_path="/models/model_mtp.safetensors", fail_closed=True,
        )
        params = base_params.model_copy(
            update={"mimo_mtp_fastpath_params": mtp_with_bad_depth}
        )
        result = classify_mimo_mtp_request(params)
        assert result.supported_depths == (1, 2, 3)

    def test_unsupported_depth_is_unsupported(self) -> None:
        base_params = _mtp_task_params(depth=1)
        raw_mtp = base_params.mimo_mtp_fastpath
        assert raw_mtp is not None
        mtp_with_bad_depth = raw_mtp.model_construct(
            enabled=True, depth=4,
            sidecar_path="/models/model_mtp.safetensors", fail_closed=True,
        )
        params = base_params.model_copy(
            update={"mimo_mtp_fastpath_params": mtp_with_bad_depth}
        )
        result = classify_mimo_mtp_request(params)
        assert result.is_unsupported is True
        assert result.is_compatible is False
        assert result.is_unwired is False


# ---------------------------------------------------------------------------
# missing_sidecar: MTP requested for MiMo model without sidecar path
# ---------------------------------------------------------------------------

class TestMissingSidecar:
    """When MTP is requested for a MiMo model with a supported depth but no
    sidecar path, the classifier returns missing_sidecar."""

    def test_no_sidecar_path_with_mtp_intent(self) -> None:
        params = _mtp_task_params(sidecar_path=None)
        result = classify_mimo_mtp_request(params)
        assert result.label == "missing_sidecar"
        assert result.discriminant == "unsupported"
        assert result.mtp_enabled is False
        assert result.sidecar_path_provided is False
        assert result.disable_reason == "missing_sidecar"

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_missing_sidecar_for_each_supported_depth(
        self, depth: int
    ) -> None:
        params = _mtp_task_params(depth=depth, sidecar_path=None)
        result = classify_mimo_mtp_request(params)
        assert result.label == "missing_sidecar"
        assert result.requested_depth == depth

    def test_missing_sidecar_preserves_fail_closed_flag(self) -> None:
        params = _mtp_task_params(sidecar_path=None, fail_closed=True)
        result = classify_mimo_mtp_request(params)
        assert result.fail_closed is True

    def test_missing_sidecar_preserves_fail_open_flag(self) -> None:
        params = _mtp_task_params(sidecar_path=None, fail_closed=False)
        result = classify_mimo_mtp_request(params)
        assert result.fail_closed is False

    def test_missing_sidecar_is_unsupported(self) -> None:
        params = _mtp_task_params(sidecar_path=None)
        result = classify_mimo_mtp_request(params)
        assert result.is_unsupported is True
        assert result.is_compatible is False
        assert result.is_unwired is False

    def test_none_depth_with_missing_sidecar(self) -> None:
        """None depth is treated as potentially compatible, but missing
        sidecar still blocks MTP."""
        params = _mtp_task_params(depth=None, sidecar_path=None)
        result = classify_mimo_mtp_request(params)
        assert result.label == "missing_sidecar"
        assert result.requested_depth is None


# ---------------------------------------------------------------------------
# unwired_execution: all static fields present, runtime not verified
# ---------------------------------------------------------------------------

class TestUnwiredExecution:
    """When all static MTP fields are present and valid, the classifier
    returns unwired_execution because the runtime guard and execution
    wiring are environment-dependent and cannot be verified by a pure
    function."""

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_supported_depths_with_sidecar_path_are_unwired(
        self, depth: int
    ) -> None:
        params = _mtp_task_params(depth=depth)
        result = classify_mimo_mtp_request(params)
        assert result.label == "unwired_execution"
        assert result.discriminant == "unwired"
        assert result.mtp_enabled is False
        assert result.requested_depth == depth
        assert result.sidecar_path_provided is True
        assert result.disable_reason == "mimo_mtp_execution_backend_unwired"

    def test_none_depth_with_sidecar_path_is_unwired(self) -> None:
        """None depth is treated as potentially compatible; with sidecar
        it becomes unwired_execution."""
        params = _mtp_task_params(depth=None)
        result = classify_mimo_mtp_request(params)
        assert result.label == "unwired_execution"
        assert result.requested_depth is None

    def test_unwired_execution_preserves_fail_closed(self) -> None:
        params = _mtp_task_params(fail_closed=True)
        result = classify_mimo_mtp_request(params)
        assert result.fail_closed is True

    def test_unwired_execution_preserves_fail_open(self) -> None:
        params = _mtp_task_params(fail_closed=False)
        result = classify_mimo_mtp_request(params)
        assert result.fail_closed is False

    def test_unwired_execution_is_unwired(self) -> None:
        params = _mtp_task_params()
        result = classify_mimo_mtp_request(params)
        assert result.is_unwired is True
        assert result.is_compatible is False
        assert result.is_unsupported is False

    def test_unwired_execution_for_each_mimo_model_id(self) -> None:
        """Every MiMo V2.5 Pro model variant should classify as unwired
        when MTP fields are present."""
        for model_id in MIMO_V25_PRO_MODEL_IDS:
            params = _mtp_task_params(model=model_id)
            result = classify_mimo_mtp_request(params)
            assert result.label == "unwired_execution", (
                f"model {model_id} with MTP intent should be unwired_execution"
            )


# ---------------------------------------------------------------------------
# compatible_mtp: no static classifier path produces this label
# ---------------------------------------------------------------------------

class TestCompatibleMtpNotProducedByStaticClassifier:
    """The pure static classifier never returns compatible_mtp because it
    cannot verify the runtime guard or execution wiring.  The
    compatible_mtp label is reserved for when the dispatch/worker layer
    confirms runtime readiness.  This test documents that design
    decision."""

    def test_static_classifier_never_returns_compatible_mtp(self) -> None:
        """Even with all fields filled, the pure classifier returns
        unwired_execution, not compatible_mtp."""
        params = _mtp_task_params(
            depth=3,
            sidecar_path="/models/model_mtp.safetensors",
            fail_closed=True,
        )
        result = classify_mimo_mtp_request(params)
        assert result.label == "unwired_execution"
        assert result.label != "compatible_mtp"

    def test_discriminant_mapping_exists_for_compatible_mtp(self) -> None:
        """The discriminant mapping does include compatible_mtp for when
        the dispatch layer upgrades unwired_execution to compatible_mtp
        after verifying runtime readiness."""
        assert DISCRIMINANT_BY_LABEL["compatible_mtp"] == "compatible_mtp"


# ---------------------------------------------------------------------------
# Discriminant consistency: label always maps to the correct discriminant
# ---------------------------------------------------------------------------

class TestDiscriminantConsistency:
    """Every classification label must map to the correct three-way
    discriminant: compatible_mtp | unsupported | unwired."""

    @pytest.mark.parametrize(
        ("label", "expected_discriminant"),
        [
            ("disabled_default", "unsupported"),
            ("unsupported_model", "unsupported"),
            ("unsupported_depth", "unsupported"),
            ("missing_sidecar", "unsupported"),
            ("unwired_execution", "unwired"),
            ("compatible_mtp", "compatible_mtp"),
        ],
    )
    def test_label_discriminant_mapping(
        self, label: MimoMtpClassificationLabel, expected_discriminant: str
    ) -> None:
        discriminant = DISCRIMINANT_BY_LABEL[label]
        assert discriminant == expected_discriminant


# ---------------------------------------------------------------------------
# Classification result model constraints
# ---------------------------------------------------------------------------

class TestClassificationResultModel:
    """MimoMtpRequestClassification is frozen, strict, and extra-forbid."""

    def _sample_classification(self) -> MimoMtpRequestClassification:
        return MimoMtpRequestClassification(
            label="disabled_default",
            discriminant="unsupported",
            model_id="test-model",
            mtp_enabled=False,
            requested_depth=None,
            sidecar_path_provided=False,
            fail_closed=None,
            disable_reason="mtp_not_requested",
        )

    def test_frozen_rejects_mutation(self) -> None:
        result = self._sample_classification()
        with pytest.raises(ValidationError):
            result.label = "unsupported_model"  # type: ignore[misc]

    def test_strict_rejects_coerced_types(self) -> None:
        with pytest.raises(ValidationError):
            MimoMtpRequestClassification(
                label="disabled_default",
                discriminant="unsupported",
                model_id="test-model",
                mtp_enabled="false",  # type: ignore[arg-type]
                requested_depth=None,
                sidecar_path_provided=False,
                fail_closed=None,
            )

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MimoMtpRequestClassification(
                label="disabled_default",
                discriminant="unsupported",
                model_id="test-model",
                mtp_enabled=False,
                requested_depth=None,
                sidecar_path_provided=False,
                fail_closed=None,
                unexpected_field=True,  # type: ignore[call-arg]
            )

    def test_round_trip_serialization(self) -> None:
        result = self._sample_classification()
        dumped = result.model_dump()
        restored = MimoMtpRequestClassification.model_validate(dumped)
        assert restored == result

    def test_json_round_trip_serialization(self) -> None:
        result = self._sample_classification()
        json_str = result.model_dump_json()
        parsed = cast(dict[str, Any], json.loads(json_str))
        restored = MimoMtpRequestClassification.model_validate(parsed)
        assert restored == result

    def test_invalid_label_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MimoMtpRequestClassification(
                label="invalid_label",  # type: ignore[arg-type]
                discriminant="unsupported",
                model_id="test-model",
                mtp_enabled=False,
                requested_depth=None,
                sidecar_path_provided=False,
                fail_closed=None,
            )

    def test_invalid_discriminant_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MimoMtpRequestClassification(
                label="disabled_default",
                discriminant="invalid",  # type: ignore[arg-type]
                model_id="test-model",
                mtp_enabled=False,
                requested_depth=None,
                sidecar_path_provided=False,
                fail_closed=None,
            )


# ---------------------------------------------------------------------------
# Property-based invariants
# ---------------------------------------------------------------------------

class TestClassificationInvariants:
    """Invariants that hold across all classification paths."""

    def test_model_id_always_matches_task_params_model(self) -> None:
        for model_id in MIMO_V25_PRO_MODEL_IDS:
            result = classify_mimo_mtp_request(_ar_task_params(model=model_id))
            assert result.model_id == str(model_id)

    def test_mtp_enabled_is_false_for_all_static_classifications(self) -> None:
        """The static classifier never sets mtp_enabled=True because
        runtime readiness is not verified."""
        for model_id in list(MIMO_V25_PRO_MODEL_IDS) + [_NON_MIMO_MODEL]:
            for params in [
                _ar_task_params(model=model_id),
                _mtp_task_params(model=model_id),
                _mtp_task_params(model=model_id, sidecar_path=None),
            ]:
                result = classify_mimo_mtp_request(params)
                assert result.mtp_enabled is False, (
                    f"static classifier must not set mtp_enabled=True; "
                    f"got {result.label} for model {model_id}"
                )

    def test_disable_reason_is_never_none_when_label_is_not_disabled_default(
        self,
    ) -> None:
        """All non-default classifications must carry a disable_reason."""
        # unsupported_model
        result = classify_mimo_mtp_request(
            _mtp_task_params(model=_NON_MIMO_MODEL)
        )
        assert result.disable_reason is not None

        # missing_sidecar
        result = classify_mimo_mtp_request(
            _mtp_task_params(sidecar_path=None)
        )
        assert result.disable_reason is not None

        # unwired_execution
        result = classify_mimo_mtp_request(_mtp_task_params())
        assert result.disable_reason is not None

    def test_sidecar_path_provided_matches_params(self) -> None:
        """sidecar_path_provided reflects whether the request included a sidecar path."""
        result_with = classify_mimo_mtp_request(
            _mtp_task_params(sidecar_path="/models/mtp.safetensors")
        )
        assert result_with.sidecar_path_provided is True

        result_without = classify_mimo_mtp_request(
            _mtp_task_params(sidecar_path=None)
        )
        assert result_without.sidecar_path_provided is False

    def test_fail_closed_reflects_request_intent(self) -> None:
        """fail_closed on the result reflects the request's fail_closed flag."""
        result_closed = classify_mimo_mtp_request(
            _mtp_task_params(fail_closed=True)
        )
        assert result_closed.fail_closed is True

        result_open = classify_mimo_mtp_request(
            _mtp_task_params(fail_closed=False)
        )
        assert result_open.fail_closed is False


# ---------------------------------------------------------------------------
# Edge cases: non-MiMo requests
# ---------------------------------------------------------------------------

class TestNonMimoEdgeCases:
    """Edge cases for non-MiMo models requesting MTP."""

    def test_non_mimo_with_depth_1_is_still_unsupported(self) -> None:
        params = _mtp_task_params(model=_NON_MIMO_MODEL, depth=1)
        result = classify_mimo_mtp_request(params)
        assert result.label == "unsupported_model"

    def test_non_mimo_with_sidecar_is_still_unsupported(self) -> None:
        params = _mtp_task_params(
            model=_NON_MIMO_MODEL, sidecar_path="/models/sidecar.safetensors"
        )
        result = classify_mimo_mtp_request(params)
        assert result.label == "unsupported_model"

    def test_non_mimo_with_fail_open_is_still_unsupported(self) -> None:
        params = _mtp_task_params(model=_NON_MIMO_MODEL, fail_closed=False)
        result = classify_mimo_mtp_request(params)
        assert result.label == "unsupported_model"

    def test_non_mimo_ar_request_is_disabled_default(self) -> None:
        """Non-MiMo AR request (no MTP intent) → disabled_default, not unsupported_model."""
        params = _ar_task_params(model=_NON_MIMO_MODEL)
        result = classify_mimo_mtp_request(params)
        assert result.label == "disabled_default"

    def test_non_mimo_model_id_preserved_in_result(self) -> None:
        params = _mtp_task_params(model=_NON_MIMO_MODEL)
        result = classify_mimo_mtp_request(params)
        assert result.model_id == _NON_MIMO_MODEL_STR


# ---------------------------------------------------------------------------
# Classification ordering: model check before depth check
# ---------------------------------------------------------------------------

class TestClassificationOrdering:
    """The classifier evaluates conditions in order: disabled_default →
    unsupported_model → unsupported_depth → missing_sidecar →
    unwired_execution.  This ordering means that a non-MiMo model with
    an unsupported depth should classify as unsupported_model (not
    unsupported_depth)."""

    def test_non_mimo_with_unsupported_depth_classifies_as_unsupported_model(
        self,
    ) -> None:
        """Model check comes before depth check."""
        base_params = _mtp_task_params(model=_NON_MIMO_MODEL, depth=1)
        raw_mtp = base_params.mimo_mtp_fastpath
        assert raw_mtp is not None
        mtp_with_bad_depth = raw_mtp.model_construct(
            enabled=True, depth=99,
            sidecar_path="/models/model_mtp.safetensors", fail_closed=True,
        )
        params = base_params.model_copy(
            update={"mimo_mtp_fastpath_params": mtp_with_bad_depth}
        )
        result = classify_mimo_mtp_request(params)
        assert result.label == "unsupported_model"
        assert result.label != "unsupported_depth"

    def test_mimo_with_unsupported_depth_classifies_as_unsupported_depth(
        self,
    ) -> None:
        """Depth check applies when model is MiMo."""
        base_params = _mtp_task_params(depth=1)
        raw_mtp = base_params.mimo_mtp_fastpath
        assert raw_mtp is not None
        mtp_with_bad_depth = raw_mtp.model_construct(
            enabled=True, depth=99,
            sidecar_path="/models/model_mtp.safetensors", fail_closed=True,
        )
        params = base_params.model_copy(
            update={"mimo_mtp_fastpath_params": mtp_with_bad_depth}
        )
        result = classify_mimo_mtp_request(params)
        assert result.label == "unsupported_depth"


# ---------------------------------------------------------------------------
# Determinism: same input → same output
# ---------------------------------------------------------------------------

class TestDeterminism:
    """The classifier is a pure function: same input always yields same output."""

    def test_deterministic_for_ar_request(self) -> None:
        params = _ar_task_params()
        a = classify_mimo_mtp_request(params)
        b = classify_mimo_mtp_request(params)
        assert a == b

    def test_deterministic_for_mtp_request(self) -> None:
        params = _mtp_task_params()
        a = classify_mimo_mtp_request(params)
        b = classify_mimo_mtp_request(params)
        assert a == b

    def test_deterministic_for_unsupported_model(self) -> None:
        params = _mtp_task_params(model=_NON_MIMO_MODEL)
        a = classify_mimo_mtp_request(params)
        b = classify_mimo_mtp_request(params)
        assert a == b
