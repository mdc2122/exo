from __future__ import annotations

import json
import os
import resource
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Protocol

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.worker.engines.mlx.mimo_mtp_fast.one_cycle import (
    MimoMtpCycleTiming,
    MimoMtpOneCycleResult,
)
from exo.worker.engines.mlx.mimo_mtp_fast.sidecar_contract import (
    probe_mimo_mtp_sidecar,
)
from exo.worker.engines.mlx.mimo_mtp_fast.speculative_loop import (
    MimoMtpCycleTrace,
    stream_mimo_mtp_fast,
)


class MimoMtpBenchmarkMode(StrEnum):
    AR = "ar"
    D1 = "d1"
    D2 = "d2"
    D3 = "d3"
    AUTO = "auto"


@dataclass(frozen=True, slots=True)
class BenchmarkRunResult:
    mode: MimoMtpBenchmarkMode
    generated_tokens: int
    decode_seconds: float
    attempted_depth_counts: dict[int, int]
    accepted_depth_counts: dict[int, int]
    timing_totals: MimoMtpCycleTiming = MimoMtpCycleTiming()


@dataclass(frozen=True, slots=True)
class ArBenchmarkRequest:
    model: object
    tokenizer: object
    model_id: str
    prompt: str
    max_tokens: int


@dataclass(frozen=True, slots=True)
class MtpBenchmarkRequest:
    mode: MimoMtpBenchmarkMode
    token_history: tuple[int, ...]
    max_tokens: int
    requested_depth: int


class _GeneratorResponse(Protocol):
    pass


PromptBuilder = Callable[[object, TextGenerationTaskParams], str]
GenerationFn = Callable[..., Iterable[_GeneratorResponse]]
OneCycleFn = Callable[[tuple[int, ...], int], MimoMtpOneCycleResult]
TimerFn = Callable[[], float]
BenchmarkRunner = Callable[[MimoMtpBenchmarkMode], BenchmarkRunResult]
JsonRow = dict[str, Any]
_ZERO_CYCLE_TIMING = MimoMtpCycleTiming()


def _process_current_rss_bytes() -> int | None:
    try:
        raw_rss_kib = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    if not raw_rss_kib:
        return None
    try:
        return int(raw_rss_kib) * 1024
    except ValueError:
        return None


def _linux_system_available_bytes() -> int | None:
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.exists():
        return None
    try:
        lines = meminfo_path.read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) < 2:
                return None
            try:
                return int(parts[1]) * 1024
            except ValueError:
                return None
    return None


def memory_diagnostic_snapshot() -> dict[str, int | None]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "process_current_rss_bytes": _process_current_rss_bytes(),
        "process_max_rss_platform_units": int(usage.ru_maxrss),
        "system_available_bytes": _linux_system_available_bytes(),
    }


def build_benchmark_stage_row(
    *,
    stage: str,
    status: Literal["started", "completed", "failed"],
    elapsed_seconds: float | None = None,
    details: Mapping[str, object] | None = None,
) -> JsonRow:
    row: JsonRow = {
        "kind": "benchmark_stage",
        "stage": stage,
        "status": status,
        "memory": memory_diagnostic_snapshot(),
    }
    if elapsed_seconds is not None:
        row["elapsed_seconds"] = round(elapsed_seconds, 6)
    if details is not None:
        row["details"] = dict(details)
    return row


class ArBenchmarkFn(Protocol):
    def __call__(self, request: ArBenchmarkRequest) -> BenchmarkRunResult: ...


class MtpBenchmarkFn(Protocol):
    def __call__(
        self, request: MtpBenchmarkRequest, *, one_cycle: OneCycleFn
    ) -> BenchmarkRunResult: ...


def render_json_line(row: JsonRow) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"))


def parse_benchmark_modes(raw_modes: str) -> tuple[MimoMtpBenchmarkMode, ...]:
    modes: list[MimoMtpBenchmarkMode] = []
    for raw_mode in raw_modes.split(","):
        mode_text = raw_mode.strip().lower()
        if not mode_text:
            continue
        try:
            modes.append(MimoMtpBenchmarkMode(mode_text))
        except ValueError as exc:
            supported = ",".join(mode.value for mode in MimoMtpBenchmarkMode)
            raise ValueError(
                f"Unsupported MiMo MTP benchmark mode {mode_text!r}; supported modes: {supported}"
            ) from exc
    if not modes:
        raise ValueError("At least one MiMo MTP benchmark mode is required")
    return tuple(modes)


def _next_step_for_contract_probe(*, ready: bool) -> str:
    if ready:
        return "run ar,d1,d2,d3,auto benchmark with the validated sidecar"
    return "provide a valid official-layout MiMo model_mtp.safetensors sidecar"


def build_contract_probe_row(
    *, sidecar_path: str | Path, model_path: str | Path | None
) -> JsonRow:
    probe = probe_mimo_mtp_sidecar(sidecar_path)
    return {
        "kind": "contract_probe",
        "ready": probe.ready,
        "model_path": None
        if model_path is None
        else str(Path(model_path).expanduser()),
        "sidecar_path": str(probe.path),
        "sidecar_status": probe.status,
        "sidecar_layer_count": probe.layer_count,
        "missing_key_count": len(probe.missing_keys),
        "error": probe.error,
        "next_step": _next_step_for_contract_probe(ready=probe.ready),
    }


def _stringify_depth_counts(counts: dict[int, int]) -> dict[str, int]:
    return {str(depth): count for depth, count in sorted(counts.items())}


def _decode_tok_s(*, generated_tokens: int, decode_seconds: float) -> float:
    if decode_seconds <= 0.0:
        return 0.0
    return generated_tokens / decode_seconds


def acceptance_rate(
    *, attempted_depth_counts: dict[int, int], accepted_depth_counts: dict[int, int]
) -> float:
    attempted = sum(depth * count for depth, count in attempted_depth_counts.items())
    if attempted <= 0:
        return 0.0
    accepted = sum(depth * count for depth, count in accepted_depth_counts.items())
    return accepted / attempted


def _timing_breakdown_row(timing: MimoMtpCycleTiming) -> dict[str, float]:
    return {
        "proposal": round(timing.proposal_seconds, 6),
        "verification": round(timing.verification_seconds, 6),
        "acceptance": round(timing.acceptance_seconds, 6),
        "fallback": round(timing.fallback_seconds, 6),
    }


def _add_timing(
    left: MimoMtpCycleTiming, right: MimoMtpCycleTiming
) -> MimoMtpCycleTiming:
    return MimoMtpCycleTiming(
        proposal_seconds=left.proposal_seconds + right.proposal_seconds,
        verification_seconds=left.verification_seconds + right.verification_seconds,
        acceptance_seconds=left.acceptance_seconds + right.acceptance_seconds,
        fallback_seconds=left.fallback_seconds + right.fallback_seconds,
    )


def _next_step_for_metric(
    *, mode: MimoMtpBenchmarkMode, decode_tok_s: float, ar_baseline_tok_s: float | None
) -> str:
    if mode == MimoMtpBenchmarkMode.AR:
        return "use AR row as baseline for MTP mode comparison"
    if ar_baseline_tok_s is None:
        return (
            f"MTP mode {mode.value} needs same-model/same-hardware AR baseline "
            "before any speedup or production claim"
        )
    if decode_tok_s > ar_baseline_tok_s:
        return (
            f"MTP mode {mode.value} has benchmark evidence; "
            "next gate is guarded integration review"
        )
    return (
        f"MTP mode {mode.value} does not beat AR; "
        "inspect proposal/verify hot path before integration"
    )


def build_metric_row(
    *,
    mode: MimoMtpBenchmarkMode,
    generated_tokens: int,
    decode_seconds: float,
    attempted_depth_counts: dict[int, int],
    accepted_depth_counts: dict[int, int],
    ar_baseline_tok_s: float | None,
    timing_totals: MimoMtpCycleTiming = _ZERO_CYCLE_TIMING,
) -> JsonRow:
    decode_tok_s = _decode_tok_s(
        generated_tokens=generated_tokens, decode_seconds=decode_seconds
    )
    rounded_tok_s = round(decode_tok_s, 4)
    return {
        "kind": "benchmark_metric",
        "mode": mode.value,
        "generated_tokens": generated_tokens,
        "decode_seconds": round(decode_seconds, 6),
        "decode_tok_s": rounded_tok_s,
        "attempted_depth_counts": _stringify_depth_counts(attempted_depth_counts),
        "accepted_depth_counts": _stringify_depth_counts(accepted_depth_counts),
        "acceptance_rate": acceptance_rate(
            attempted_depth_counts=attempted_depth_counts,
            accepted_depth_counts=accepted_depth_counts,
        ),
        "timing_breakdown_seconds": _timing_breakdown_row(timing_totals),
        "ar_baseline_tok_s": None
        if ar_baseline_tok_s is None
        else round(ar_baseline_tok_s, 4),
        "next_step": _next_step_for_metric(
            mode=mode,
            decode_tok_s=rounded_tok_s,
            ar_baseline_tok_s=ar_baseline_tok_s,
        ),
    }


def _tok_s_for_result(result: BenchmarkRunResult) -> float:
    return round(
        _decode_tok_s(
            generated_tokens=result.generated_tokens,
            decode_seconds=result.decode_seconds,
        ),
        4,
    )


def run_benchmark_modes(
    *, modes: tuple[MimoMtpBenchmarkMode, ...], runner: BenchmarkRunner
) -> list[JsonRow]:
    rows: list[JsonRow] = []
    ar_baseline_tok_s: float | None = None
    for mode in modes:
        result = runner(mode)
        row = build_metric_row(
            mode=result.mode,
            generated_tokens=result.generated_tokens,
            decode_seconds=result.decode_seconds,
            attempted_depth_counts=result.attempted_depth_counts,
            accepted_depth_counts=result.accepted_depth_counts,
            ar_baseline_tok_s=ar_baseline_tok_s,
            timing_totals=result.timing_totals,
        )
        rows.append(row)
        if mode == MimoMtpBenchmarkMode.AR:
            ar_baseline_tok_s = _tok_s_for_result(result)
    return rows


def build_ar_task_params(request: ArBenchmarkRequest) -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=ModelId(request.model_id),
        input=[InputMessage(role="user", content=request.prompt)],
        max_output_tokens=request.max_tokens,
        temperature=0.0,
        bench=True,
    )


def run_ar_benchmark(
    request: ArBenchmarkRequest,
    *,
    prompt_builder: PromptBuilder,
    generate: GenerationFn,
    timer: TimerFn = time.perf_counter,
    elapsed_seconds_override: float | None = None,
) -> BenchmarkRunResult:
    task_params = build_ar_task_params(request)
    templated_prompt = prompt_builder(request.tokenizer, task_params)
    start = timer()
    generated_tokens = 0
    for _response in generate(
        model=request.model,
        tokenizer=request.tokenizer,
        task=task_params,
        prompt=templated_prompt,
        kv_prefix_cache=None,
        group=None,
    ):
        generated_tokens += 1
    elapsed_seconds = (
        elapsed_seconds_override
        if elapsed_seconds_override is not None
        else max(0.0, timer() - start)
    )
    return BenchmarkRunResult(
        mode=MimoMtpBenchmarkMode.AR,
        generated_tokens=generated_tokens,
        decode_seconds=elapsed_seconds,
        attempted_depth_counts={},
        accepted_depth_counts={},
    )


def requested_depth_for_mode(mode: MimoMtpBenchmarkMode) -> int:
    match mode:
        case MimoMtpBenchmarkMode.D1:
            return 1
        case MimoMtpBenchmarkMode.D2:
            return 2
        case MimoMtpBenchmarkMode.D3 | MimoMtpBenchmarkMode.AUTO:
            return 3
        case MimoMtpBenchmarkMode.AR:
            return 0


def _increment_count(counts: dict[int, int], depth: int) -> None:
    counts[depth] = counts.get(depth, 0) + 1


def run_mtp_benchmark(
    request: MtpBenchmarkRequest,
    *,
    one_cycle: OneCycleFn,
    timer: TimerFn = time.perf_counter,
    elapsed_seconds_override: float | None = None,
) -> BenchmarkRunResult:
    start = timer()
    generated_tokens = 0
    attempted_depth_counts: dict[int, int] = {}
    accepted_depth_counts: dict[int, int] = {}
    timing_totals = MimoMtpCycleTiming()

    def collect_trace(trace: MimoMtpCycleTrace) -> None:
        nonlocal timing_totals
        timing_totals = _add_timing(timing_totals, trace.timing)

    for event in stream_mimo_mtp_fast(
        token_history=request.token_history,
        max_tokens=request.max_tokens,
        requested_depth=request.requested_depth,
        one_cycle=one_cycle,
        trace_collector=collect_trace,
    ):
        generated_tokens += 1
        _increment_count(attempted_depth_counts, event.attempted_depth)
        _increment_count(accepted_depth_counts, event.accepted_depth)
    elapsed_seconds = (
        elapsed_seconds_override
        if elapsed_seconds_override is not None
        else max(0.0, timer() - start)
    )
    return BenchmarkRunResult(
        mode=request.mode,
        generated_tokens=generated_tokens,
        decode_seconds=elapsed_seconds,
        attempted_depth_counts=attempted_depth_counts,
        accepted_depth_counts=accepted_depth_counts,
        timing_totals=timing_totals,
    )


def build_benchmark_runner(
    *,
    model: object,
    tokenizer: object,
    model_id: str,
    prompt: str,
    prompt_token_history: tuple[int, ...],
    max_tokens: int,
    one_cycle: OneCycleFn,
    ar_benchmark: ArBenchmarkFn,
    mtp_benchmark: MtpBenchmarkFn,
) -> BenchmarkRunner:
    def runner(mode: MimoMtpBenchmarkMode) -> BenchmarkRunResult:
        if mode == MimoMtpBenchmarkMode.AR:
            return ar_benchmark(
                ArBenchmarkRequest(
                    model=model,
                    tokenizer=tokenizer,
                    model_id=model_id,
                    prompt=prompt,
                    max_tokens=max_tokens,
                )
            )
        return mtp_benchmark(
            MtpBenchmarkRequest(
                mode=mode,
                token_history=prompt_token_history,
                max_tokens=max_tokens,
                requested_depth=requested_depth_for_mode(mode),
            ),
            one_cycle=one_cycle,
        )

    return runner
