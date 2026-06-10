"""Typed MiMo MTP request classifier for guarded fail-closed request propagation.

This module provides a pure function :func:`classify_mimo_mtp_request` that
examines a :class:`~exo.shared.types.text_generation.TextGenerationTaskParams`
and returns a discriminated classification result indicating whether the request
is a compatible MiMo MTP candidate, unsupported, or unwired.

The classifier is a **pure function** — no side effects, no HTTP exceptions,
no sidecar probing.  It inspects only the typed fields already present on the
task params.  This makes the decision logic independently testable and
reusable across the API boundary, dispatch layer, and worker fastpath without
duplicating imperative guard logic.

Classification results align with the Seed's ``mtp_execution_state`` ontology:

- ``disabled_default`` — no MTP intent (``mimo_mtp_fastpath_params`` is None
  or ``enabled`` is False)
- ``compatible_mtp`` — model supports the MTP fastpath (MiMo V2.5 Pro or
  GLM 5.1), depth is supported (1–3), sidecar path is provided,
  fail-closed semantics are explicit
- ``unsupported_model`` — MTP requested for a model without MTP support
- ``unsupported_depth`` — MTP requested with an unsupported depth value
- ``missing_sidecar`` — MTP requested without a sidecar path
- ``unwired_execution`` — MTP intent is present but runtime guard /
  execution wiring status is not verified (caller must check separately)
"""

from __future__ import annotations

from typing import Final, Literal, final

from pydantic import BaseModel, ConfigDict

from exo.shared.models.model_cards import MTP_FASTPATH_MODEL_IDS
from exo.shared.types.text_generation import (
    SUPPORTED_MIMO_MTP_FASTPATH_DEPTHS,
    MimoMtpFastpathParams,
    TextGenerationTaskParams,
)

# ---------------------------------------------------------------------------
# Discriminated classification labels
# ---------------------------------------------------------------------------

MimoMtpClassificationLabel = Literal[
    "disabled_default",
    "compatible_mtp",
    "unsupported_model",
    "unsupported_depth",
    "missing_sidecar",
    "unwired_execution",
]

# Grouping labels matching the AC's three-way discriminant
_MIMO_MTP_COMPATIBLE_LABELS: frozenset[MimoMtpClassificationLabel] = frozenset(
    {"compatible_mtp"}
)
_MIMO_MTP_UNSUPPORTED_LABELS: frozenset[MimoMtpClassificationLabel] = frozenset(
    {"disabled_default", "unsupported_model", "unsupported_depth", "missing_sidecar"}
)
_MIMO_MTP_UNWIRED_LABELS: frozenset[MimoMtpClassificationLabel] = frozenset(
    {"unwired_execution"}
)

# The discriminant grouping used by callers who need the three-way split
MimoMtpRequestDiscriminant = Literal["compatible_mtp", "unsupported", "unwired"]

DISCRIMINANT_BY_LABEL: dict[MimoMtpClassificationLabel, MimoMtpRequestDiscriminant] = {
    **{label: "compatible_mtp" for label in _MIMO_MTP_COMPATIBLE_LABELS},
    **{label: "unsupported" for label in _MIMO_MTP_UNSUPPORTED_LABELS},
    **{label: "unwired" for label in _MIMO_MTP_UNWIRED_LABELS},
}

_SUPPORTED_DEPTH_TEXT: Final[str] = ",".join(
    str(d) for d in sorted(SUPPORTED_MIMO_MTP_FASTPATH_DEPTHS)
)


# ---------------------------------------------------------------------------
# Classification result — immutable, strict, frozen
# ---------------------------------------------------------------------------

@final
class MimoMtpRequestClassification(BaseModel):
    """Immutable discriminated classification result for a MiMo MTP request.

    ``label`` is the fine-grained classification aligned with
    ``MtpExecutionState``.  ``discriminant`` is the three-way coarse
    grouping required by the AC: ``compatible_mtp | unsupported | unwired``.

    The model is frozen and strict per project conventions; it cannot be
    silently mutated or constructed with coerced types.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    label: MimoMtpClassificationLabel
    discriminant: MimoMtpRequestDiscriminant
    model_id: str
    mtp_enabled: bool
    requested_depth: int | None
    sidecar_path_provided: bool
    fail_closed: bool | None
    disable_reason: str | None = None
    supported_depths: tuple[int, ...] | None = None

    @property
    def is_compatible(self) -> bool:
        return self.discriminant == "compatible_mtp"

    @property
    def is_unsupported(self) -> bool:
        return self.discriminant == "unsupported"

    @property
    def is_unwired(self) -> bool:
        return self.discriminant == "unwired"


# ---------------------------------------------------------------------------
# Pure classifier function
# ---------------------------------------------------------------------------

def classify_mimo_mtp_request(
    task_params: TextGenerationTaskParams,
) -> MimoMtpRequestClassification:
    """Classify a text generation request's MiMo MTP eligibility.

    This is a **pure function**: it inspects only the typed fields on
    ``task_params`` and returns a classification without side effects,
    HTTP exceptions, or I/O.

    Classification logic (evaluated in order):

    1.  **disabled_default** — ``mimo_mtp_fastpath_params`` is ``None`` or
        ``enabled`` is ``False``.  Normal AR generation; no MTP intent.
    2.  **unsupported_model** — MTP is requested but the model is not in
        ``MTP_FASTPATH_MODEL_IDS`` (MiMo V2.5 Pro variants or GLM 5.1).
    3.  **unsupported_depth** — MTP is requested for a supported model but
        the depth is not in ``SUPPORTED_MIMO_MTP_FASTPATH_DEPTHS`` (1, 2, 3).
        ``None`` depth is treated as potentially compatible (the worker
        resolves it to the default depth).
    4.  **missing_sidecar** — MTP is requested for a supported model with a
        supported depth but no ``sidecar_path`` is provided.
    5.  **unwired_execution** — MTP is requested, the model is supported,
        depth is supported, and sidecar path is provided, but the classifier cannot
        verify the runtime guard or execution wiring status (those are
        environment-dependent and checked at the dispatch/worker layer).
        The caller must separately verify the runtime guard and execution
        wiring before dispatching.
    6.  **compatible_mtp** — All static fields are present and valid.  The
        request *appears* compatible, but actual MTP execution still
        depends on the runtime guard, execution wiring, and sidecar probe
        at the dispatch/worker layer.  This label maps to the
        ``compatible_attempted`` / ``enabled_intent`` execution states in
        the ontology.

    Returns
    -------
    MimoMtpRequestClassification
        Immutable classification result with label, discriminant, and
        telemetry fields.
    """
    mtp_params: MimoMtpFastpathParams | None = task_params.mimo_mtp_fastpath
    model_id = str(task_params.model)

    # 1. No MTP intent → disabled_default
    if mtp_params is None or not mtp_params.enabled:
        return MimoMtpRequestClassification(
            label="disabled_default",
            discriminant="unsupported",
            model_id=model_id,
            mtp_enabled=False,
            requested_depth=None,
            sidecar_path_provided=False,
            fail_closed=None,
            disable_reason="mtp_not_requested",
        )

    requested_depth = mtp_params.depth
    sidecar_path_provided = mtp_params.sidecar_path is not None
    fail_closed = mtp_params.fail_closed

    # 2. Model without MTP fastpath support → unsupported_model
    if task_params.model not in MTP_FASTPATH_MODEL_IDS:
        return MimoMtpRequestClassification(
            label="unsupported_model",
            discriminant="unsupported",
            model_id=model_id,
            mtp_enabled=False,
            requested_depth=requested_depth,
            sidecar_path_provided=sidecar_path_provided,
            fail_closed=fail_closed,
            disable_reason="unsupported_model",
        )

    # 3. Unsupported depth → unsupported_depth
    #    None depth is treated as potentially compatible (worker resolves
    #    default), but explicit unsupported values are rejected.
    if (
        requested_depth is not None
        and requested_depth not in SUPPORTED_MIMO_MTP_FASTPATH_DEPTHS
    ):
        return MimoMtpRequestClassification(
            label="unsupported_depth",
            discriminant="unsupported",
            model_id=model_id,
            mtp_enabled=False,
            requested_depth=requested_depth,
            sidecar_path_provided=sidecar_path_provided,
            fail_closed=fail_closed,
            disable_reason="unsupported_depth",
            supported_depths=tuple(sorted(SUPPORTED_MIMO_MTP_FASTPATH_DEPTHS)),
        )

    # 4. Missing sidecar path → missing_sidecar
    if mtp_params.sidecar_path is None:
        return MimoMtpRequestClassification(
            label="missing_sidecar",
            discriminant="unsupported",
            model_id=model_id,
            mtp_enabled=False,
            requested_depth=requested_depth,
            sidecar_path_provided=False,
            fail_closed=fail_closed,
            disable_reason="missing_sidecar",
        )

    # 5. All static fields present → unwired_execution
    #    The classifier cannot verify runtime guard or execution wiring;
    #    those are environment-dependent checks at the dispatch/worker
    #    layer.  This result tells the caller: "statically compatible,
    #    but you must verify runtime readiness separately."
    return MimoMtpRequestClassification(
        label="unwired_execution",
        discriminant="unwired",
        model_id=model_id,
        mtp_enabled=False,
        requested_depth=requested_depth,
        sidecar_path_provided=True,
        fail_closed=fail_closed,
        disable_reason="mimo_mtp_execution_backend_unwired",
    )
