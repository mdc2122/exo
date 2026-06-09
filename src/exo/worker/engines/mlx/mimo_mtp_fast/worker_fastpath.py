from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, final

from exo.shared.models.model_cards import MIMO_V25_PRO_MODEL_IDS
from exo.shared.types.mimo_mtp_errors import raise_mimo_mtp_fail_closed_error
from exo.shared.types.text_generation import (
    SUPPORTED_MIMO_MTP_FASTPATH_DEPTHS,
    MimoMtpFastpathParams,
    TextGenerationTaskParams,
)
from exo.worker.engines.mlx.mimo_mtp_fast.sidecar_contract import (
    MimoMtpSidecarStatus,
    probe_mimo_mtp_sidecar,
)
from exo.worker.engines.mlx.mimo_mtp_fast.sidecar_loader import (
    MimoMtpSidecarLoadError,
    MimoMtpSidecarTensors,
    load_mimo_mtp_sidecar_tensors,
)

MIMO_MTP_WORKER_NATIVE_RUNTIME_ENV = "EXO_MIMO_MTP_NATIVE_RUNTIME"

_FALSEY_ENV_VALUES = frozenset({"", "0", "false", "off", "no"})

SUPPORTED_MIMO_MTP_BACKEND_TYPES: frozenset[str] = frozenset({"mlx"})

MimoMtpAcceptedExecutionPath = Literal["ar", "mimo_mtp_fastpath", "rejected"]


@final
@dataclass(frozen=True, slots=True)
class MimoMtpWorkerFastpathDecision:
    should_use_mtp: bool
    accepted_execution_path: MimoMtpAcceptedExecutionPath
    mtp_enabled: bool
    model_id: str
    requested_depth: int | None
    mtp_depth: int | None
    sidecar_status: MimoMtpSidecarStatus | None
    disable_reason: str | None
    reject_reason: str | None
    fallback_reason: str | None
    telemetry: Mapping[str, object]
    error_message: str | None = None
    sidecar: MimoMtpSidecarTensors | None = None
    backend_type: str | None = None
    native_runtime_enabled: bool | None = None

    def raise_fail_closed_error(self) -> None:
        """Raise the correct typed MTP fail-closed error for this decision.

        This method maps the decision's ``disable_reason`` to the
        appropriate typed error subclass from ``mimo_mtp_errors`` and
        raises it. It is only valid to call this method when
        ``accepted_execution_path == "rejected"`` and the request was
        configured as fail-closed.

        Raises ``MimoMtpFailClosedError`` (a concrete subclass).
        Raises ``ValueError`` if the decision is not in a rejected state
        or if required fields are missing.
        """
        if self.accepted_execution_path != "rejected":
            raise ValueError(
                f"raise_fail_closed_error called on non-rejected decision: "
                f"accepted_execution_path={self.accepted_execution_path}"
            )
        if self.disable_reason is None:
            raise ValueError(
                "raise_fail_closed_error called on rejected decision "
                "with no disable_reason"
            )
        raise_mimo_mtp_fail_closed_error(
            disable_reason=self.disable_reason,
            model_id=self.model_id,
            requested_depth=self.requested_depth,
            sidecar_status=self.sidecar_status,
            backend_type=self.backend_type,
            native_runtime_enabled=self.native_runtime_enabled,
        )


@final
class MimoMtpWorkerFastpathCache:
    """Caches validated MTP sidecar tensors for a runner lifetime.

    The sidecar is loaded at most once per expanded sidecar path while the
    worker runner process is alive. Generation cycles must reuse this cache;
    they must not rebuild or reload the sidecar stack per token or per MTP
    cycle.
    """

    def __init__(self) -> None:
        self._sidecars: dict[Path, MimoMtpSidecarTensors] = {}
        self._load_counts: dict[Path, int] = {}

    def load_sidecar(self, path: str | Path) -> MimoMtpSidecarTensors:
        sidecar_path = Path(path).expanduser()
        if sidecar_path not in self._sidecars:
            self._sidecars[sidecar_path] = load_mimo_mtp_sidecar_tensors(
                sidecar_path
            )
            self._load_counts[sidecar_path] = self._load_counts.get(sidecar_path, 0) + 1
        return self._sidecars[sidecar_path]

    def load_count_for_path(self, path: str | Path) -> int:
        return self._load_counts.get(Path(path).expanduser(), 0)


def worker_native_mimo_mtp_runtime_enabled() -> bool:
    raw_value = os.environ.get(MIMO_MTP_WORKER_NATIVE_RUNTIME_ENV)
    return raw_value is not None and raw_value.strip().lower() not in _FALSEY_ENV_VALUES


def _telemetry(
    *,
    accepted_execution_path: MimoMtpAcceptedExecutionPath,
    mtp_enabled: bool,
    requested_depth: int | None = None,
    mtp_depth: int | None = None,
    sidecar_status: MimoMtpSidecarStatus | None = None,
    disable_reason: str | None = None,
    reject_reason: str | None = None,
    fallback_reason: str | None = None,
) -> dict[str, object]:
    values: dict[str, object] = {
        "accepted_execution_path": accepted_execution_path,
        "mtp_enabled": mtp_enabled,
    }
    if requested_depth is not None:
        values["requested_mtp_depth"] = requested_depth
    if mtp_depth is not None:
        values["mtp_depth"] = mtp_depth
    if sidecar_status is not None:
        values["mtp_sidecar_status"] = sidecar_status
    if disable_reason is not None:
        values["mtp_disable_reason"] = disable_reason
    if reject_reason is not None:
        values["mtp_reject_reason"] = reject_reason
    if fallback_reason is not None:
        values["mtp_fallback_reason"] = fallback_reason
    return values


def _ar_decision(*, model_id: str = "") -> MimoMtpWorkerFastpathDecision:
    return MimoMtpWorkerFastpathDecision(
        should_use_mtp=False,
        accepted_execution_path="ar",
        mtp_enabled=False,
        model_id=model_id,
        requested_depth=None,
        mtp_depth=None,
        sidecar_status=None,
        disable_reason=None,
        reject_reason=None,
        fallback_reason=None,
        telemetry=_telemetry(accepted_execution_path="ar", mtp_enabled=False),
    )


def _rejected_decision(
    *,
    model_id: str,
    requested_depth: int | None,
    sidecar_status: MimoMtpSidecarStatus | None,
    disable_reason: str,
    error_message: str,
    backend_type: str | None = None,
    native_runtime_enabled: bool | None = None,
) -> MimoMtpWorkerFastpathDecision:
    reject_reason = disable_reason
    if f"reject_reason={reject_reason}" not in error_message:
        error_message = f"{error_message} reject_reason={reject_reason}"
    return MimoMtpWorkerFastpathDecision(
        should_use_mtp=False,
        accepted_execution_path="rejected",
        mtp_enabled=False,
        model_id=model_id,
        requested_depth=requested_depth,
        mtp_depth=None,
        sidecar_status=sidecar_status,
        disable_reason=disable_reason,
        reject_reason=disable_reason,
        fallback_reason=None,
        error_message=error_message,
        telemetry=_telemetry(
            accepted_execution_path="rejected",
            mtp_enabled=False,
            requested_depth=requested_depth,
            sidecar_status=sidecar_status,
            disable_reason=disable_reason,
            reject_reason=disable_reason,
        ),
        backend_type=backend_type,
        native_runtime_enabled=native_runtime_enabled,
    )


def _fallback_ar_decision(
    *,
    model_id: str,
    requested_depth: int | None,
    sidecar_status: MimoMtpSidecarStatus | None,
    disable_reason: str,
    fallback_reason: str,
) -> MimoMtpWorkerFastpathDecision:
    return MimoMtpWorkerFastpathDecision(
        should_use_mtp=False,
        accepted_execution_path="ar",
        mtp_enabled=False,
        model_id=model_id,
        requested_depth=requested_depth,
        mtp_depth=None,
        sidecar_status=sidecar_status,
        disable_reason=disable_reason,
        reject_reason=None,
        fallback_reason=fallback_reason,
        telemetry=_telemetry(
            accepted_execution_path="ar",
            mtp_enabled=False,
            requested_depth=requested_depth,
            sidecar_status=sidecar_status,
            disable_reason=disable_reason,
            fallback_reason=fallback_reason,
        ),
    )


def _requested_depth(params: MimoMtpFastpathParams) -> int:
    return params.depth if params.depth is not None else 1


def _fail_or_fallback(
    *,
    model_id: str,
    params: MimoMtpFastpathParams,
    requested_depth: int | None,
    sidecar_status: MimoMtpSidecarStatus | None,
    disable_reason: str,
    fallback_reason: str,
    error_message: str,
    backend_type: str | None = None,
    native_runtime_enabled: bool | None = None,
) -> MimoMtpWorkerFastpathDecision:
    if params.fail_closed:
        return _rejected_decision(
            model_id=model_id,
            requested_depth=requested_depth,
            sidecar_status=sidecar_status,
            disable_reason=disable_reason,
            error_message=error_message,
            backend_type=backend_type,
            native_runtime_enabled=native_runtime_enabled,
        )
    return _fallback_ar_decision(
        model_id=model_id,
        requested_depth=requested_depth,
        sidecar_status=sidecar_status,
        disable_reason=disable_reason,
        fallback_reason=fallback_reason,
    )


def evaluate_mimo_mtp_worker_fastpath(
    task_params: TextGenerationTaskParams,
    *,
    cache: MimoMtpWorkerFastpathCache | None = None,
    native_runtime_enabled: bool | None = None,
    execution_path_wired: bool = False,
    backend_type: str = "mlx",
) -> MimoMtpWorkerFastpathDecision:
    """Evaluate whether a text generation request should use MiMo MTP fastpath.

    This is a pure decision function (no side effects beyond optional sidecar
    probing). It checks the request's MTP intent against a series of
    fail-closed guards and returns a decision indicating whether MTP should
    be used, whether it was rejected, or whether the request should fall back
    to AR generation.

    Guard order (each guard is evaluated independently; the first failing
    guard short-circuits):

    1. **disabled_default** — No MTP intent (params absent or enabled=False).
    2. **unsupported_model** — MTP requested for a non-MiMo model.
    3. **unsupported_depth** — MTP requested with depth outside (1, 2, 3).
    4. **missing_sidecar** — No sidecar_path provided.
    5. **invalid_sidecar** — Sidecar file missing or invalid on disk.
    6. **mimo_mtp_incompatible_backend** — Backend type is not ``"mlx"``.
    7. **mimo_mtp_native_runtime_disabled** — Native runtime env var not set.
    8. **mimo_mtp_distributed_generator_unwired** — Execution path not wired.
    9. **invalid_sidecar** (load error) — Sidecar load failed at cache level.

    Parameters
    ----------
    task_params:
        The text generation task parameters with optional MTP intent.
    cache:
        Optional sidecar cache for reuse across calls.
    native_runtime_enabled:
        Override for the native runtime environment variable check.
        If None, reads from ``EXO_MIMO_MTP_NATIVE_RUNTIME``.
    execution_path_wired:
        Whether the distributed generator execution path is wired.
        Defaults to False (unwired) for safety.
    backend_type:
        The inference backend type. Defaults to ``"mlx"``. If not
        in ``SUPPORTED_MIMO_MTP_BACKEND_TYPES``, the incompatible
        backend guard triggers.
    """
    model_id = str(task_params.model)
    mtp_params = task_params.mimo_mtp_fastpath

    # 1. No MTP intent → disabled_default
    if mtp_params is None or not mtp_params.enabled:
        return _ar_decision(model_id=model_id)

    requested_depth = _requested_depth(mtp_params)

    # 2. Non-MiMo model → unsupported_model
    if task_params.model not in MIMO_V25_PRO_MODEL_IDS:
        return _fail_or_fallback(
            model_id=model_id,
            params=mtp_params,
            requested_depth=requested_depth,
            sidecar_status=None,
            disable_reason="unsupported_model",
            fallback_reason="fail_open_unsupported_model",
            error_message=(
                "MiMo MTP fastpath rejected before AR dispatch: "
                f"model={task_params.model} disable_reason=unsupported_model"
            ),
        )

    # 3. Unsupported depth → unsupported_depth
    # None depth is treated as compatible (worker resolves to default depth=1),
    # but explicit values outside SUPPORTED_MIMO_MTP_FASTPATH_DEPTHS are
    # rejected. This guard is defensive: Pydantic validates depth at
    # construction, but model_construct can bypass validation.
    if (
        mtp_params.depth is not None
        and mtp_params.depth not in SUPPORTED_MIMO_MTP_FASTPATH_DEPTHS
    ):
        supported_depths_text = ",".join(
            str(d) for d in sorted(SUPPORTED_MIMO_MTP_FASTPATH_DEPTHS)
        )
        return _fail_or_fallback(
            model_id=model_id,
            params=mtp_params,
            requested_depth=requested_depth,
            sidecar_status=None,
            disable_reason="unsupported_depth",
            fallback_reason="fail_open_unsupported_depth",
            error_message=(
                "MiMo MTP fastpath rejected before AR dispatch: "
                f"depth={mtp_params.depth} is not in supported depths "
                f"[{supported_depths_text}]; "
                "disable_reason=unsupported_depth"
            ),
        )

    # 4. Missing sidecar path → missing_sidecar
    if mtp_params.sidecar_path is None:
        return _fail_or_fallback(
            model_id=model_id,
            params=mtp_params,
            requested_depth=requested_depth,
            sidecar_status="missing",
            disable_reason="missing_sidecar",
            fallback_reason="fail_open_missing_sidecar",
            error_message=(
                "MiMo MTP fastpath rejected before AR dispatch: "
                "sidecar_path is required disable_reason=missing_sidecar"
            ),
        )

    # 5. Invalid/missing sidecar file → missing_sidecar / invalid_sidecar
    probe = probe_mimo_mtp_sidecar(mtp_params.sidecar_path)
    if not probe.ready:
        return _fail_or_fallback(
            model_id=model_id,
            params=mtp_params,
            requested_depth=requested_depth,
            sidecar_status=probe.status,
            disable_reason=(
                "missing_sidecar"
                if probe.status == "missing"
                else "invalid_sidecar"
            ),
            fallback_reason=(
                "fail_open_missing_sidecar"
                if probe.status == "missing"
                else "fail_open_invalid_sidecar"
            ),
            error_message=(
                "MiMo MTP fastpath rejected before AR dispatch: "
                f"sidecar_status={probe.status} disable_reason="
                f"{'missing_sidecar' if probe.status == 'missing' else 'invalid_sidecar'} "
                f"error={probe.error}"
            ),
        )

    # 6. Incompatible backend → mimo_mtp_incompatible_backend
    if backend_type not in SUPPORTED_MIMO_MTP_BACKEND_TYPES:
        return _fail_or_fallback(
            model_id=model_id,
            params=mtp_params,
            requested_depth=requested_depth,
            sidecar_status="ready",
            disable_reason="mimo_mtp_incompatible_backend",
            fallback_reason="fail_open_incompatible_backend",
            error_message=(
                "MiMo MTP fastpath rejected before execution: "
                f"backend_type={backend_type} is not compatible with MTP; "
                f"supported_backends={','.join(sorted(SUPPORTED_MIMO_MTP_BACKEND_TYPES))} "
                "disable_reason=mimo_mtp_incompatible_backend"
            ),
            backend_type=backend_type,
        )

    # 7. Native runtime disabled → mimo_mtp_native_runtime_disabled
    runtime_enabled = (
        worker_native_mimo_mtp_runtime_enabled()
        if native_runtime_enabled is None
        else native_runtime_enabled
    )
    if not runtime_enabled:
        return _fail_or_fallback(
            model_id=model_id,
            params=mtp_params,
            requested_depth=requested_depth,
            sidecar_status="ready",
            disable_reason="mimo_mtp_native_runtime_disabled",
            fallback_reason="fail_open_native_runtime_disabled",
            error_message=(
                "MiMo MTP fastpath rejected before AR dispatch: sidecar is ready "
                "but native worker runtime is disabled; "
                "disable_reason=mimo_mtp_native_runtime_disabled"
            ),
            native_runtime_enabled=runtime_enabled,
        )

    # 8. Execution path unwired → mimo_mtp_distributed_generator_unwired
    if not execution_path_wired:
        return _fail_or_fallback(
            model_id=model_id,
            params=mtp_params,
            requested_depth=requested_depth,
            sidecar_status="ready",
            disable_reason="mimo_mtp_distributed_generator_unwired",
            fallback_reason="fail_open_distributed_generator_unwired",
            error_message=(
                "MiMo MTP fastpath rejected before execution: sidecar is ready "
                "and native worker runtime is enabled, but distributed generator "
                "execution is not wired; "
                "disable_reason=mimo_mtp_distributed_generator_unwired"
            ),
            native_runtime_enabled=runtime_enabled,
        )

    # 9. Load sidecar via cache → invalid_sidecar if load fails
    sidecar_cache = cache if cache is not None else MimoMtpWorkerFastpathCache()
    try:
        sidecar = sidecar_cache.load_sidecar(mtp_params.sidecar_path)
    except MimoMtpSidecarLoadError as exc:
        return _fail_or_fallback(
            model_id=model_id,
            params=mtp_params,
            requested_depth=requested_depth,
            sidecar_status="invalid",
            disable_reason="invalid_sidecar",
            fallback_reason="fail_open_invalid_sidecar",
            error_message=(
                "MiMo MTP fastpath rejected before AR dispatch: "
                f"disable_reason=invalid_sidecar error={exc}"
            ),
        )

    return MimoMtpWorkerFastpathDecision(
        should_use_mtp=True,
        accepted_execution_path="mimo_mtp_fastpath",
        mtp_enabled=True,
        model_id=model_id,
        requested_depth=requested_depth,
        mtp_depth=requested_depth,
        sidecar_status="ready",
        disable_reason=None,
        reject_reason=None,
        fallback_reason=None,
        telemetry=_telemetry(
            accepted_execution_path="mimo_mtp_fastpath",
            mtp_enabled=True,
            requested_depth=requested_depth,
            mtp_depth=requested_depth,
            sidecar_status="ready",
        ),
        sidecar=sidecar,
        backend_type=backend_type,
        native_runtime_enabled=runtime_enabled,
    )
