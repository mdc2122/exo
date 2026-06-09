"""Pure validate_mtp_intent function with typed discriminated-union results.

This module provides :func:`validate_mtp_intent`, a pure function that takes a
:class:`~exo.shared.types.mimo_mtp_classifier.MimoMtpRequestClassification` and
returns a discriminated-union result: either :class:`MTPValidatedIntent` on
success (when the classification indicates a valid MTP-intent request) or
:class:`MTPDisabledReason` on error (when MTP is disabled or unsupported).

Design decisions:
- This is a **pure function** — no side effects, no HTTP exceptions, no I/O.
- It builds on the classifier's static result and produces a stricter
  discriminated union that distinguishes "validated MTP intent" from
  "disabled with a reason".
- The discriminant field (``result_type``) enables pattern-matching by callers
  without isinstance checks, aligned with the Seed's ``mtp_execution_state``
  ontology.
- MTPValidatedIntent is only produced for classifications that indicate the
  request has passed static validation (label in ``unwired_execution`` or
  ``compatible_mtp`` — i.e., the request is a valid MTP-intent candidate that
  may proceed to runtime verification).
- All other classification labels produce MTPDisabledReason with a specific
  disable reason aligned with the ontology's ``mtp_execution_state`` enum.
"""

from __future__ import annotations

from typing import Final, Literal, final

from pydantic import BaseModel, ConfigDict

from exo.shared.types.mimo_mtp_classifier import (
    MimoMtpClassificationLabel,
    MimoMtpRequestClassification,
)

# ---------------------------------------------------------------------------
# Discriminated-union result types
# ---------------------------------------------------------------------------

MtpIntentResultType = Literal["validated_intent", "disabled_reason"]

# Labels that represent a valid MTP-intent request (statically validated,
# may proceed to runtime verification). These are the only labels that
# produce MTPValidatedIntent.
_MTP_INTENT_VALID_LABELS: frozenset[MimoMtpClassificationLabel] = frozenset(
    {"unwired_execution", "compatible_mtp"}
)

# Map from classification label to MTP execution state for disabled reasons
_LABEL_TO_DISABLE_REASON: dict[MimoMtpClassificationLabel, str] = {
    "disabled_default": "mtp_not_requested",
    "unsupported_model": "unsupported_model",
    "unsupported_depth": "unsupported_depth",
    "missing_sidecar": "missing_sidecar",
}


@final
class MTPValidatedIntent(BaseModel):
    """Success result: the request carries valid MTP intent.

    This means the request has passed static validation — the model is a
    supported MiMo V2.5 Pro variant, the depth is supported (or None for
    default), and a sidecar path is provided. The request may proceed to
    runtime verification (sidecar probe, execution wiring check).

    This result type is frozen and strict per project conventions.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    result_type: Literal["validated_intent"] = "validated_intent"
    model_id: str
    requested_depth: int | None
    sidecar_path_provided: bool
    fail_closed: bool
    classification_label: MimoMtpClassificationLabel

    @property
    def is_validated_intent(self) -> bool:
        return True

    @property
    def is_disabled_reason(self) -> bool:
        return False


@final
class MTPDisabledReason(BaseModel):
    """Error result: MTP is disabled or unsupported for this request.

    This means the request did not pass static validation — either no MTP
    intent was expressed, the model is unsupported, the depth is invalid,
    or the sidecar path is missing.

    The ``disable_reason`` field is aligned with the Seed's
    ``mtp_execution_state`` ontology and is always populated for
    MTPDisabledReason results.

    This result type is frozen and strict per project conventions.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    result_type: Literal["disabled_reason"] = "disabled_reason"
    model_id: str
    mtp_enabled: bool
    requested_depth: int | None
    sidecar_path_provided: bool
    fail_closed: bool | None
    disable_reason: str
    classification_label: MimoMtpClassificationLabel

    @property
    def is_validated_intent(self) -> bool:
        return False

    @property
    def is_disabled_reason(self) -> bool:
        return True


# The discriminated union type used by callers for pattern-matching.
MtpIntentValidationResult = MTPValidatedIntent | MTPDisabledReason

# Result discriminant for the union — matches the result_type field.
MTP_INTENT_RESULT_DISCRIMINANT_FIELD: Final[str] = "result_type"


# ---------------------------------------------------------------------------
# Pure validation function
# ---------------------------------------------------------------------------

def validate_mtp_intent(
    classification: MimoMtpRequestClassification,
) -> MtpIntentValidationResult:
    """Validate MTP intent from a classification result.

    This is a **pure function** that inspects only the classification result
    and returns a typed discriminated-union:

    - :class:`MTPValidatedIntent` when the classification indicates a valid
      MTP-intent request (label is ``unwired_execution`` or ``compatible_mtp``).
      The request has passed static validation and may proceed to runtime
      verification.
    - :class:`MTPDisabledReason` when the classification indicates MTP is
      disabled or unsupported. The ``disable_reason`` field is always populated
      with a value aligned with the ``mtp_execution_state`` ontology.

    Parameters
    ----------
    classification:
        The classification result from
        :func:`~exo.shared.types.mimo_mtp_classifier.classify_mimo_mtp_request`.

    Returns
    -------
    MtpIntentValidationResult
        Discriminated-union result with ``result_type`` field for dispatch.

    Examples
    --------
    >>> from exo.shared.types.mimo_mtp_classifier import classify_mimo_mtp_request
    >>> from exo.shared.types.text_generation import TextGenerationTaskParams, MimoMtpFastpathParams, InputMessage
    >>> params = TextGenerationTaskParams(
    ...     model=ModelId("kernelpool/MiMo-V2.5-Pro-6bit"),
    ...     input=[InputMessage(role="user", content="Hello")],
    ...     max_output_tokens=4,
    ...     mimo_mtp_fastpath_params=MimoMtpFastpathParams(enabled=True, depth=1, sidecar_path="/models/mtp.safetensors", fail_closed=True),
    ... )
    >>> classification = classify_mimo_mtp_request(params)
    >>> result = validate_mtp_intent(classification)
    >>> result.result_type
    'validated_intent'
    """
    label = classification.label

    # Valid MTP-intent labels → MTPValidatedIntent
    if label in _MTP_INTENT_VALID_LABELS:
        # fail_closed is guaranteed non-None for valid intent labels
        # because the classifier only produces unwired_execution/compatible_mtp
        # when MimoMtpFastpathParams was present with enabled=True, which
        # requires fail_closed to be set (it defaults to True).
        fail_closed_value = classification.fail_closed
        if fail_closed_value is None:
            # Defensive: if somehow fail_closed is None for a valid intent,
            # this is a contract violation. Default to True (fail-closed)
            # as the safe default per the Seed's requirements.
            fail_closed_value = True

        return MTPValidatedIntent(
            model_id=classification.model_id,
            requested_depth=classification.requested_depth,
            sidecar_path_provided=classification.sidecar_path_provided,
            fail_closed=fail_closed_value,
            classification_label=label,
        )

    # All other labels → MTPDisabledReason
    disable_reason = _LABEL_TO_DISABLE_REASON.get(
        label, classification.disable_reason or "unknown"
    )

    return MTPDisabledReason(
        model_id=classification.model_id,
        mtp_enabled=classification.mtp_enabled,
        requested_depth=classification.requested_depth,
        sidecar_path_provided=classification.sidecar_path_provided,
        fail_closed=classification.fail_closed,
        disable_reason=disable_reason,
        classification_label=label,
    )
