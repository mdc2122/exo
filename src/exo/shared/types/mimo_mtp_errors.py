"""Typed fail-closed error hierarchy for MiMo MTP request propagation.

These exceptions are raised when an MTP request encounters a condition
that prevents MTP execution and the request is configured as fail-closed
(``mimo_mtp_fail_closed=True``). Each error class carries a stable
``mtp_execution_state`` label aligned with the Seed's ``mtp_execution_state``
ontology and a structured ``detail`` dict with telemetry-grade fields.

Fail-closed means the request is **rejected** — it does NOT silently fall
back to AR generation. The caller must handle the error explicitly.

For fail-open requests (``mimo_mtp_fail_closed=False``), these errors are
NOT raised; instead, the worker falls back to AR generation with telemetry
recording the disable reason.
"""

from __future__ import annotations

from typing import Final, final

from exo.shared.types.benchmark_telemetry import MtpExecutionState
from exo.shared.types.text_generation import (
    SUPPORTED_MIMO_MTP_FASTPATH_DEPTHS,
)

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class MimoMtpFailClosedError(Exception):
    """Base class for all MiMo MTP fail-closed errors.

    Raised when an MTP request is configured as fail-closed and encounters
    a condition that prevents MTP execution. Subclasses carry a stable
    ``mtp_execution_state`` label and a structured ``detail`` dict.

    This base class is not intended to be raised directly; use one of the
    concrete subclasses instead. Catching ``MimoMtpFailClosedError`` will
    catch all MTP fail-closed errors, which is useful for generic handling
    at the API boundary.
    """

    _MTP_EXECUTION_STATE: MtpExecutionState = "fail_closed_error"

    def __init__(
        self,
        message: str,
        *,
        mtp_execution_state: MtpExecutionState | None = None,
        model_id: str,
        requested_depth: int | None,
        disable_reason: str,
        sidecar_status: str | None = None,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.mtp_execution_state: MtpExecutionState = (
            mtp_execution_state or self._MTP_EXECUTION_STATE
        )
        self.model_id = model_id
        self.requested_depth = requested_depth
        self.disable_reason = disable_reason
        self.sidecar_status = sidecar_status
        self.error_code = error_code

    def detail(self) -> dict[str, object]:
        """Structured telemetry-grade detail dict for the error."""
        return {
            "error": self.error_code or self.disable_reason,
            "message": str(self),
            "mtp_enabled": False,
            "mtp_execution_state": self.mtp_execution_state,
            "model_id": self.model_id,
            "requested_mtp_depth": self.requested_depth,
            "mtp_depth": None,
            "mtp_sidecar_status": self.sidecar_status,
            "mtp_disable_reason": self.disable_reason,
            "accepted_execution_path": "rejected",
        }


# ---------------------------------------------------------------------------
# Concrete error: unwired_execution
# ---------------------------------------------------------------------------

_MIMO_MTP_UNWIRED_EXECUTION_REASON: Final[str] = (
    "mimo_mtp_distributed_generator_unwired"
)


@final
class MimoMtpUnwiredExecutionError(MimoMtpFailClosedError):
    """Raised when MTP intent is present, sidecar and native runtime are
    ready, but the distributed generator execution path is not wired.

    This is a fail-closed condition: the request must be rejected rather
    than silently falling back to AR, because the user explicitly requested
    MTP with ``fail_closed=True`` and the execution backend cannot honour
    that intent.

    ``mtp_execution_state`` is ``unwired_execution`` (not the generic
    ``fail_closed_error``) so that telemetry and the classifier can
    distinguish this specific condition from other failures.
    """

    _MTP_EXECUTION_STATE: MtpExecutionState = "unwired_execution"

    def __init__(
        self,
        message: str,
        *,
        model_id: str,
        requested_depth: int | None,
        sidecar_status: str | None = None,
        native_runtime_enabled: bool = True,
    ) -> None:
        super().__init__(
            message,
            mtp_execution_state="unwired_execution",
            model_id=model_id,
            requested_depth=requested_depth,
            disable_reason=_MIMO_MTP_UNWIRED_EXECUTION_REASON,
            sidecar_status=sidecar_status,
            error_code="mimo_mtp_unwired_execution",
        )
        self.native_runtime_enabled = native_runtime_enabled

    def detail(self) -> dict[str, object]:
        d = super().detail()
        d["native_runtime_enabled"] = self.native_runtime_enabled
        return d


# ---------------------------------------------------------------------------
# Concrete error: incompatible_backend
# ---------------------------------------------------------------------------

_MIMO_MTP_INCOMPATIBLE_BACKEND_REASON: Final[str] = (
    "mimo_mtp_incompatible_backend"
)


@final
class MimoMtpIncompatibleBackendError(MimoMtpFailClosedError):
    """Raised when MTP intent is present but the inference backend is
    fundamentally incompatible with MiMo MTP execution.

    This covers conditions where the backend cannot support MTP at all,
    such as:
    - A non-MLX inference backend that does not implement the MTP
      speculative loop seam.
    - A hardware/driver configuration that is incompatible with the MTP
      sidecar's tensor operations (e.g., wrong device type).

    This is a fail-closed condition: the request must be rejected rather
    than silently falling back to AR, because the user explicitly requested
    MTP with ``fail_closed=True`` and the backend cannot honour that intent
    even in principle.

    ``mtp_execution_state`` is ``unwired_execution`` (per the ontology,
    this state covers cases where the execution wiring is missing or the
    backend is incompatible) with ``disable_reason=incompatible_backend``
    to distinguish it from the unwired-generator case.
    """

    _MTP_EXECUTION_STATE: MtpExecutionState = "unwired_execution"

    def __init__(
        self,
        message: str,
        *,
        model_id: str,
        requested_depth: int | None,
        backend_type: str,
        sidecar_status: str | None = None,
    ) -> None:
        super().__init__(
            message,
            mtp_execution_state="unwired_execution",
            model_id=model_id,
            requested_depth=requested_depth,
            disable_reason=_MIMO_MTP_INCOMPATIBLE_BACKEND_REASON,
            sidecar_status=sidecar_status,
            error_code="mimo_mtp_incompatible_backend",
        )
        self.backend_type = backend_type

    def detail(self) -> dict[str, object]:
        d = super().detail()
        d["backend_type"] = self.backend_type
        return d


# ---------------------------------------------------------------------------
# Concrete error: unsupported_depth
# ---------------------------------------------------------------------------

_MAX_SUPPORTED_MIMO_MTP_DEPTH: Final[int] = max(SUPPORTED_MIMO_MTP_FASTPATH_DEPTHS)
_SUPPORTED_MIMO_MTP_DEPTHS: Final[tuple[int, ...]] = SUPPORTED_MIMO_MTP_FASTPATH_DEPTHS


@final
class MimoMtpUnsupportedDepthError(MimoMtpFailClosedError):
    """Raised when MTP is requested with a depth value that exceeds the
    maximum supported depth for MiMo V2.5 Pro MTP, and the request is
    configured as fail-closed.

    This is a **fail-closed condition**: the request must be rejected rather
    than silently falling back to AR, because the user explicitly requested
    MTP with ``fail_closed=True`` and the depth is outside the supported range.

    ``mtp_execution_state`` is ``unsupported_depth`` per the Seed ontology.

    This error is the typed equivalent of the Pydantic ``ValueError`` raised
    by ``MimoMtpFastpathParams._validate_supported_depth``, but is raised at
    the worker fastpath evaluation layer when an unsupported depth bypasses
    Pydantic validation (e.g., via ``model_construct``).
    """

    _MTP_EXECUTION_STATE: MtpExecutionState = "unsupported_depth"

    def __init__(
        self,
        message: str,
        *,
        model_id: str,
        requested_depth: int | None,
        supported_depths: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__(
            message,
            mtp_execution_state="unsupported_depth",
            model_id=model_id,
            requested_depth=requested_depth,
            disable_reason="unsupported_depth",
            error_code="mimo_mtp_unsupported_depth",
        )
        self.supported_depths: tuple[int, ...] = (
            supported_depths if supported_depths is not None
            else _SUPPORTED_MIMO_MTP_DEPTHS
        )

    def detail(self) -> dict[str, object]:
        d = super().detail()
        d["supported_depths"] = self.supported_depths
        return d


# ---------------------------------------------------------------------------
# Concrete error: unsupported_model
# ---------------------------------------------------------------------------


@final
class MimoMtpUnsupportedModelError(MimoMtpFailClosedError):
    """Raised when MTP is requested for a model that is not a MiMo V2.5 Pro
    variant and the request is configured as fail-closed.

    ``mtp_execution_state`` is ``unsupported_model``.
    """

    _MTP_EXECUTION_STATE: MtpExecutionState = "unsupported_model"

    def __init__(
        self,
        message: str,
        *,
        model_id: str,
        requested_depth: int | None,
    ) -> None:
        super().__init__(
            message,
            mtp_execution_state="unsupported_model",
            model_id=model_id,
            requested_depth=requested_depth,
            disable_reason="unsupported_model",
            error_code="mimo_mtp_unsupported_model",
        )


# ---------------------------------------------------------------------------
# Concrete error: missing_sidecar
# ---------------------------------------------------------------------------


@final
class MimoMtpMissingSidecarError(MimoMtpFailClosedError):
    """Raised when MTP is requested but the sidecar path is missing or the
    sidecar file does not exist, and the request is fail-closed.

    ``mtp_execution_state`` is ``missing_sidecar``.
    """

    _MTP_EXECUTION_STATE: MtpExecutionState = "missing_sidecar"

    def __init__(
        self,
        message: str,
        *,
        model_id: str,
        requested_depth: int | None,
        sidecar_path: str | None = None,
    ) -> None:
        super().__init__(
            message,
            mtp_execution_state="missing_sidecar",
            model_id=model_id,
            requested_depth=requested_depth,
            disable_reason="missing_sidecar",
            sidecar_status="missing",
            error_code="mimo_mtp_missing_sidecar",
        )
        self.sidecar_path = sidecar_path

    def detail(self) -> dict[str, object]:
        d = super().detail()
        d["sidecar_path"] = self.sidecar_path
        return d


# ---------------------------------------------------------------------------
# Concrete error: invalid_sidecar
# ---------------------------------------------------------------------------


@final
class MimoMtpInvalidSidecarError(MimoMtpFailClosedError):
    """Raised when MTP is requested but the sidecar file is present but
    invalid (e.g., missing required tensors, wrong dtype), and the request
    is fail-closed.

    ``mtp_execution_state`` is ``missing_sidecar`` (per the ontology,
    an invalid sidecar is functionally equivalent to missing for MTP
    eligibility), with ``disable_reason=invalid_sidecar`` to distinguish
    from a truly absent file.
    """

    _MTP_EXECUTION_STATE: MtpExecutionState = "missing_sidecar"

    def __init__(
        self,
        message: str,
        *,
        model_id: str,
        requested_depth: int | None,
        sidecar_path: str,
        sidecar_error: str | None = None,
    ) -> None:
        super().__init__(
            message,
            mtp_execution_state="missing_sidecar",
            model_id=model_id,
            requested_depth=requested_depth,
            disable_reason="invalid_sidecar",
            sidecar_status="invalid",
            error_code="mimo_mtp_invalid_sidecar",
        )
        self.sidecar_path = sidecar_path
        self.sidecar_error = sidecar_error

    def detail(self) -> dict[str, object]:
        d = super().detail()
        d["sidecar_path"] = self.sidecar_path
        d["sidecar_error"] = self.sidecar_error
        return d


# ---------------------------------------------------------------------------
# Concrete error: native_runtime_disabled
# ---------------------------------------------------------------------------


@final
class MimoMtpNativeRuntimeDisabledError(MimoMtpFailClosedError):
    """Raised when MTP is requested with a ready sidecar but the native
    MLX MTP runtime is not enabled, and the request is fail-closed.

    ``mtp_execution_state`` is ``fail_closed_error`` (the sidecar is ready
    but the runtime environment variable is not set).
    """

    _MTP_EXECUTION_STATE: MtpExecutionState = "fail_closed_error"

    def __init__(
        self,
        message: str,
        *,
        model_id: str,
        requested_depth: int | None,
        sidecar_status: str | None = None,
    ) -> None:
        super().__init__(
            message,
            mtp_execution_state="fail_closed_error",
            model_id=model_id,
            requested_depth=requested_depth,
            disable_reason="mimo_mtp_native_runtime_disabled",
            sidecar_status=sidecar_status,
            error_code="mimo_mtp_native_runtime_disabled",
        )


# ---------------------------------------------------------------------------
# Discriminated union of all MTP fail-closed error types
# ---------------------------------------------------------------------------

MimoMtpFailClosedErrorType = (
    MimoMtpUnwiredExecutionError
    | MimoMtpIncompatibleBackendError
    | MimoMtpUnsupportedModelError
    | MimoMtpUnsupportedDepthError
    | MimoMtpMissingSidecarError
    | MimoMtpInvalidSidecarError
    | MimoMtpNativeRuntimeDisabledError
)

# Mapping from disable_reason to the appropriate error class
_MIMO_MTP_FAIL_CLOSED_ERROR_BY_REASON: dict[
    str, type[MimoMtpFailClosedError]
] = {
    "unsupported_model": MimoMtpUnsupportedModelError,
    "unsupported_depth": MimoMtpUnsupportedDepthError,
    "missing_sidecar": MimoMtpMissingSidecarError,
    "invalid_sidecar": MimoMtpInvalidSidecarError,
    "mimo_mtp_native_runtime_disabled": MimoMtpNativeRuntimeDisabledError,
    "mimo_mtp_distributed_generator_unwired": MimoMtpUnwiredExecutionError,
    "mimo_mtp_incompatible_backend": MimoMtpIncompatibleBackendError,
}


def raise_mimo_mtp_fail_closed_error(
    *,
    disable_reason: str,
    model_id: str,
    requested_depth: int | None,
    sidecar_status: str | None = None,
    backend_type: str | None = None,
    native_runtime_enabled: bool | None = None,
    sidecar_path: str | None = None,
    sidecar_error: str | None = None,
) -> None:
    """Raise the correct typed MTP fail-closed error for a given disable_reason.

    This factory function maps a ``disable_reason`` string from the worker
    fastpath decision to the appropriate typed error class, so that callers
    don't need to repeat the mapping logic.

    Raises the appropriate ``MimoMtpFailClosedError`` subclass.
    Raises ``ValueError`` if the ``disable_reason`` is not recognized.
    """
    error_cls = _MIMO_MTP_FAIL_CLOSED_ERROR_BY_REASON.get(disable_reason)
    if error_cls is None:
        raise ValueError(
            f"Unknown MTP fail-closed disable_reason: {disable_reason!r}"
        )

    message = (
        f"MiMo MTP fastpath rejected before execution: "
        f"model={model_id} disable_reason={disable_reason}"
    )

    if error_cls is MimoMtpUnwiredExecutionError:
        assert native_runtime_enabled is not None
        raise MimoMtpUnwiredExecutionError(
            message,
            model_id=model_id,
            requested_depth=requested_depth,
            sidecar_status=sidecar_status,
            native_runtime_enabled=native_runtime_enabled,
        )

    if error_cls is MimoMtpIncompatibleBackendError:
        assert backend_type is not None
        raise MimoMtpIncompatibleBackendError(
            message,
            model_id=model_id,
            requested_depth=requested_depth,
            backend_type=backend_type,
            sidecar_status=sidecar_status,
        )

    if error_cls is MimoMtpMissingSidecarError:
        raise MimoMtpMissingSidecarError(
            message,
            model_id=model_id,
            requested_depth=requested_depth,
            sidecar_path=sidecar_path,
        )

    if error_cls is MimoMtpInvalidSidecarError:
        assert sidecar_path is not None
        raise MimoMtpInvalidSidecarError(
            message,
            model_id=model_id,
            requested_depth=requested_depth,
            sidecar_path=sidecar_path,
            sidecar_error=sidecar_error,
        )

    if error_cls is MimoMtpNativeRuntimeDisabledError:
        raise MimoMtpNativeRuntimeDisabledError(
            message,
            model_id=model_id,
            requested_depth=requested_depth,
            sidecar_status=sidecar_status,
        )

    if error_cls is MimoMtpUnsupportedDepthError:
        raise MimoMtpUnsupportedDepthError(
            message,
            model_id=model_id,
            requested_depth=requested_depth,
        )

    if error_cls is MimoMtpUnsupportedModelError:
        raise MimoMtpUnsupportedModelError(
            message,
            model_id=model_id,
            requested_depth=requested_depth,
        )

    # Should be unreachable if the mapping is complete
    raise MimoMtpFailClosedError(
        message,
        model_id=model_id,
        requested_depth=requested_depth,
        disable_reason=disable_reason,
    )
