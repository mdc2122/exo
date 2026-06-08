from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, final

from exo.shared.models.model_cards import MIMO_V25_PRO_MODEL_IDS
from exo.shared.types.text_generation import (
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
MimoMtpAcceptedExecutionPath = Literal["ar", "mimo_mtp_fastpath", "rejected"]


@final
@dataclass(frozen=True, slots=True)
class MimoMtpWorkerFastpathDecision:
    should_use_mtp: bool
    accepted_execution_path: MimoMtpAcceptedExecutionPath
    mtp_enabled: bool
    requested_depth: int | None
    mtp_depth: int | None
    sidecar_status: MimoMtpSidecarStatus | None
    disable_reason: str | None
    reject_reason: str | None
    fallback_reason: str | None
    telemetry: Mapping[str, object]
    error_message: str | None = None
    sidecar: MimoMtpSidecarTensors | None = None


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
            self._sidecars[sidecar_path] = load_mimo_mtp_sidecar_tensors(sidecar_path)
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


def _ar_decision() -> MimoMtpWorkerFastpathDecision:
    return MimoMtpWorkerFastpathDecision(
        should_use_mtp=False,
        accepted_execution_path="ar",
        mtp_enabled=False,
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
    requested_depth: int | None,
    sidecar_status: MimoMtpSidecarStatus | None,
    disable_reason: str,
    error_message: str,
) -> MimoMtpWorkerFastpathDecision:
    reject_reason = disable_reason
    if f"reject_reason={reject_reason}" not in error_message:
        error_message = f"{error_message} reject_reason={reject_reason}"
    return MimoMtpWorkerFastpathDecision(
        should_use_mtp=False,
        accepted_execution_path="rejected",
        mtp_enabled=False,
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
    )


def _fallback_ar_decision(
    *,
    requested_depth: int | None,
    sidecar_status: MimoMtpSidecarStatus | None,
    disable_reason: str,
    fallback_reason: str,
) -> MimoMtpWorkerFastpathDecision:
    return MimoMtpWorkerFastpathDecision(
        should_use_mtp=False,
        accepted_execution_path="ar",
        mtp_enabled=False,
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
    params: MimoMtpFastpathParams,
    requested_depth: int | None,
    sidecar_status: MimoMtpSidecarStatus | None,
    disable_reason: str,
    fallback_reason: str,
    error_message: str,
) -> MimoMtpWorkerFastpathDecision:
    if params.fail_closed:
        return _rejected_decision(
            requested_depth=requested_depth,
            sidecar_status=sidecar_status,
            disable_reason=disable_reason,
            error_message=error_message,
        )
    return _fallback_ar_decision(
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
) -> MimoMtpWorkerFastpathDecision:
    mtp_params = task_params.mimo_mtp_fastpath
    if mtp_params is None or not mtp_params.enabled:
        return _ar_decision()

    requested_depth = _requested_depth(mtp_params)
    if task_params.model not in MIMO_V25_PRO_MODEL_IDS:
        return _fail_or_fallback(
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

    if mtp_params.sidecar_path is None:
        return _fail_or_fallback(
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

    probe = probe_mimo_mtp_sidecar(mtp_params.sidecar_path)
    if not probe.ready:
        return _fail_or_fallback(
            params=mtp_params,
            requested_depth=requested_depth,
            sidecar_status=probe.status,
            disable_reason=(
                "missing_sidecar" if probe.status == "missing" else "invalid_sidecar"
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

    runtime_enabled = (
        worker_native_mimo_mtp_runtime_enabled()
        if native_runtime_enabled is None
        else native_runtime_enabled
    )
    if not runtime_enabled:
        return _fail_or_fallback(
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
        )

    if not execution_path_wired:
        return _fail_or_fallback(
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
        )

    sidecar_cache = cache if cache is not None else MimoMtpWorkerFastpathCache()
    try:
        sidecar = sidecar_cache.load_sidecar(mtp_params.sidecar_path)
    except MimoMtpSidecarLoadError as exc:
        return _fail_or_fallback(
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
    )
