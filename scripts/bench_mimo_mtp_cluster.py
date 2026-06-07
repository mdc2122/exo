#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Final, Literal, Protocol, cast

from exo.worker.engines.mlx.mimo_mtp_fast.benchmark import render_json_line

JsonObject = dict[str, object]
MtpThroughputThreshold = Literal[
    "below_30_tok_s", "at_least_30_tok_s", "at_least_40_tok_s"
]
HttpJsonGet = Callable[[str, float], tuple[int, JsonObject]]
HttpJsonPost = Callable[[str, JsonObject, float], tuple[int, JsonObject]]
AR_BASELINE_MAX_TOKENS: Final[tuple[int, int]] = (16, 64)
BENCHMARK_MATRIX_MODES: Final[tuple[str, str, str, str]] = (
    "ar",
    "mtp-d1",
    "mtp-d2",
    "mtp-d3",
)
MTP_TARGET_TOKENS_PER_SECOND: Final[float] = 30.0
MTP_PREFERRED_TOKENS_PER_SECOND: Final[float] = 40.0


class _HttpResponse(Protocol):
    status: int

    def read(self) -> bytes: ...

    def __enter__(self) -> _HttpResponse: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class ClusterBenchmarkArgs:
    api_base: str
    model_id: str
    prompt: str
    max_tokens: int
    repeats: int
    mode_label: str
    timeout_seconds: float
    payload_extra_json: str | None
    list_models: bool
    matrix_max_tokens: tuple[int, ...] = AR_BASELINE_MAX_TOKENS
    matrix_commands: bool = False
    mimo_mtp_sidecar_path: str | None = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark MiMo MTP work through an already-running exo cluster API "
            "instead of materializing the full model in this process."
        )
    )
    parser.add_argument("--api-base", default="http://127.0.0.1:52415")
    parser.add_argument("--model-id", default="kernelpool/MiMo-V2.5-Pro-6bit")
    parser.add_argument(
        "--prompt", default="Write a Python function that parses JSON lines."
    )
    parser.add_argument("--max-tokens", type=int, action="append", default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--mode-label",
        default="ar",
        help=(
            "Label for the row being collected, for example ar or mtp-d1. This "
            "script does not itself enable MTP; use --payload-extra-json only "
            "after a guarded cluster MTP request flag exists."
        ),
    )
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--payload-extra-json",
        default=None,
        help=(
            "Optional JSON object merged into the /bench/chat/completions payload "
            "for guarded experimental cluster flags."
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Emit a /v1/models probe row before benchmark requests.",
    )
    parser.add_argument(
        "--matrix-commands",
        action="store_true",
        help=(
            "Emit runnable same-cluster AR/MTP benchmark matrix commands as JSONL "
            "without contacting the cluster."
        ),
    )
    parser.add_argument(
        "--mimo-mtp-sidecar-path",
        default=None,
        help=(
            "Optional official model_mtp.safetensors path to include in emitted "
            "guarded MTP matrix payloads."
        ),
    )
    return parser


def _parse_args(argv: list[str] | None = None) -> ClusterBenchmarkArgs:
    namespace = _parser().parse_args(argv)
    raw_max_tokens = cast(list[int] | None, namespace.max_tokens)
    matrix_max_tokens = (
        AR_BASELINE_MAX_TOKENS if raw_max_tokens is None else tuple(raw_max_tokens)
    )
    max_tokens = (
        matrix_max_tokens[-1] if matrix_max_tokens else AR_BASELINE_MAX_TOKENS[0]
    )
    return ClusterBenchmarkArgs(
        api_base=cast(str, namespace.api_base).rstrip("/"),
        model_id=cast(str, namespace.model_id),
        prompt=cast(str, namespace.prompt),
        max_tokens=max_tokens,
        matrix_max_tokens=matrix_max_tokens,
        repeats=cast(int, namespace.repeats),
        mode_label=cast(str, namespace.mode_label),
        timeout_seconds=cast(float, namespace.timeout_seconds),
        payload_extra_json=cast(str | None, namespace.payload_extra_json),
        list_models=cast(bool, namespace.list_models),
        matrix_commands=cast(bool, namespace.matrix_commands),
        mimo_mtp_sidecar_path=cast(str | None, namespace.mimo_mtp_sidecar_path),
    )


def _decode_json_object(raw_body: str) -> JsonObject:
    decoded = cast(object, json.loads(raw_body))
    if not isinstance(decoded, dict):
        raise ValueError("expected JSON object response")
    return cast(JsonObject, decoded)


def _http_json_get(url: str, timeout_seconds: float) -> tuple[int, JsonObject]:
    request = urllib.request.Request(url, method="GET")
    try:
        raw_response = cast(
            _HttpResponse, urllib.request.urlopen(request, timeout=timeout_seconds)
        )
        with raw_response as response:
            body = response.read().decode("utf-8", errors="replace")
            return int(response.status), _decode_json_object(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return int(exc.code), _decode_json_object(body)


def _http_json_post(
    url: str, payload: JsonObject, timeout_seconds: float
) -> tuple[int, JsonObject]:
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        raw_response = cast(
            _HttpResponse, urllib.request.urlopen(request, timeout=timeout_seconds)
        )
        with raw_response as response:
            body = response.read().decode("utf-8", errors="replace")
            return int(response.status), _decode_json_object(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return int(exc.code), _decode_json_object(body)


def _payload_extra(raw_payload_extra_json: str | None) -> JsonObject:
    if raw_payload_extra_json is None:
        return {}
    decoded = cast(object, json.loads(raw_payload_extra_json))
    if not isinstance(decoded, dict):
        raise ValueError("--payload-extra-json must decode to a JSON object")
    return cast(JsonObject, decoded)


def render_cluster_benchmark_command(
    *,
    api_base: str,
    model_id: str,
    max_tokens: int,
    repeats: int,
    mode_label: str,
    list_models: bool,
    payload_extra_json: str | None = None,
) -> str:
    """Render a runnable cluster benchmark command for evidence capture."""
    command_parts = [
        "uv",
        "run",
        "python3",
        "scripts/bench_mimo_mtp_cluster.py",
        "--api-base",
        shlex.quote(api_base.rstrip("/")),
        "--model-id",
        shlex.quote(model_id),
        "--max-tokens",
        str(max_tokens),
        "--repeats",
        str(repeats),
        "--mode-label",
        shlex.quote(mode_label),
    ]
    if list_models:
        command_parts.append("--list-models")
    if payload_extra_json is not None:
        command_parts.extend(["--payload-extra-json", shlex.quote(payload_extra_json)])
    return " ".join(command_parts)


def render_ar_baseline_commands(*, api_base: str, model_id: str) -> list[str]:
    """Render exact runnable same-cluster AR baseline commands for rollout evidence."""
    return [
        render_cluster_benchmark_command(
            api_base=api_base,
            model_id=model_id,
            max_tokens=max_tokens,
            repeats=1,
            mode_label="ar",
            list_models=True,
        )
        for max_tokens in AR_BASELINE_MAX_TOKENS
    ]


def _matrix_payload_extra_json(
    *, mode_label: str, mimo_mtp_sidecar_path: str | None
) -> str | None:
    if mode_label == "ar":
        return None
    try:
        depth = int(mode_label.removeprefix("mtp-d"))
    except ValueError as exc:
        raise ValueError(
            "unsupported MiMo MTP matrix mode " + repr(mode_label)
        ) from exc
    payload: JsonObject = {
        "mimo_mtp_fastpath": True,
        "mimo_mtp_depth": depth,
        "mimo_mtp_fail_closed": True,
    }
    if mimo_mtp_sidecar_path is not None:
        payload["mimo_mtp_sidecar_path"] = mimo_mtp_sidecar_path
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def render_benchmark_matrix_command_rows(
    *,
    api_base: str,
    model_id: str,
    max_tokens_values: Iterable[int],
    repeats: int,
    mimo_mtp_sidecar_path: str | None,
) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for max_tokens in max_tokens_values:
        for mode_label in BENCHMARK_MATRIX_MODES:
            payload_extra_json = _matrix_payload_extra_json(
                mode_label=mode_label,
                mimo_mtp_sidecar_path=mimo_mtp_sidecar_path,
            )
            rows.append(
                {
                    "kind": "cluster_benchmark_command",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "mode": mode_label,
                    "model": model_id,
                    "max_tokens": max_tokens,
                    "repeats": repeats,
                    "payload_extra_json": payload_extra_json,
                    "command": render_cluster_benchmark_command(
                        api_base=api_base,
                        model_id=model_id,
                        max_tokens=max_tokens,
                        repeats=repeats,
                        mode_label=mode_label,
                        list_models=mode_label == "ar",
                        payload_extra_json=payload_extra_json,
                    ),
                    "next_step": (
                        "collect this same-cluster AR baseline row"
                        if mode_label == "ar"
                        else "collect this guarded same-cluster MTP row only after MTP execution is wired"
                    ),
                }
            )
    return rows


def build_cluster_chat_payload(
    *,
    model_id: str,
    prompt: str,
    max_tokens: int,
    payload_extra: Mapping[str, object],
) -> JsonObject:
    payload: JsonObject = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0.0,
    }
    payload.update(payload_extra)
    return payload


def _generation_stats(response: JsonObject) -> JsonObject | None:
    raw_stats = response.get("generation_stats")
    if isinstance(raw_stats, dict):
        return cast(JsonObject, raw_stats)
    return None


def _accepted_execution_path(
    *,
    mode_label: str,
    response: Mapping[str, object],
    stats: Mapping[str, object] | None,
) -> str:
    for source in (stats, response):
        if source is None:
            continue
        raw_path = source.get("accepted_execution_path")
        if isinstance(raw_path, str) and raw_path:
            return raw_path
    if mode_label == "ar":
        return "ar"
    return "unknown"


def classify_mtp_throughput_threshold(
    generation_tps: float,
) -> MtpThroughputThreshold:
    """Classify guarded MTP throughput against Slice 5 target bands."""
    if generation_tps >= MTP_PREFERRED_TOKENS_PER_SECOND:
        return "at_least_40_tok_s"
    if generation_tps >= MTP_TARGET_TOKENS_PER_SECOND:
        return "at_least_30_tok_s"
    return "below_30_tok_s"


def build_mtp_speedup_budget(generation_tps: object) -> JsonObject | None:
    """Build MTP threshold budget telemetry from a computed tok/s metric."""
    if isinstance(generation_tps, bool) or not isinstance(generation_tps, int | float):
        return None
    measured_tps = float(generation_tps)
    return {
        "generation_tps": measured_tps,
        "threshold_30_tok_s_met": measured_tps >= MTP_TARGET_TOKENS_PER_SECOND,
        "threshold_40_tok_s_met": measured_tps >= MTP_PREFERRED_TOKENS_PER_SECOND,
        "tok_s_gap_to_30": round(
            max(0.0, MTP_TARGET_TOKENS_PER_SECOND - measured_tps), 6
        ),
        "tok_s_gap_to_40": round(
            max(0.0, MTP_PREFERRED_TOKENS_PER_SECOND - measured_tps), 6
        ),
    }


def _model_path_classification(response: Mapping[str, object]) -> JsonObject:
    raw_execution_path = response.get("execution_path")
    if not isinstance(raw_execution_path, dict):
        return {
            "path": "unknown",
            "is_exo_cluster_tensor_parallel": False,
            "source": "response.execution_path",
            "sharding": None,
            "world_size": None,
            "model_path": None,
            "disable_reason": "response did not include execution_path telemetry",
        }
    execution_path = cast(Mapping[str, object], raw_execution_path)
    sharding = execution_path.get("sharding")
    world_size = execution_path.get("world_size")
    model_path = execution_path.get("model_path")
    is_tensor_parallel_cluster = (
        sharding == "Tensor" and isinstance(world_size, int) and world_size > 1
    )
    return {
        "path": "exo_cluster_tensor_parallel"
        if is_tensor_parallel_cluster
        else "single_studio_full_model_load",
        "is_exo_cluster_tensor_parallel": is_tensor_parallel_cluster,
        "source": "response.execution_path",
        "sharding": sharding,
        "world_size": world_size,
        "model_path": model_path,
        "disable_reason": None
        if is_tensor_parallel_cluster
        else "response execution path is not tensor-parallel across multiple cluster nodes",
    }


def build_cluster_metric_row(
    *,
    api_base: str,
    model_id: str,
    mode_label: str,
    repeat_index: int,
    elapsed_seconds: float,
    status_code: int,
    response: JsonObject,
    payload_extra_keys: list[str],
) -> JsonObject:
    stats = _generation_stats(response)
    generation_tps = None if stats is None else stats.get("generation_tps")
    generation_tokens = None if stats is None else stats.get("generation_tokens")
    prompt_tps = None if stats is None else stats.get("prompt_tps")
    accepted_execution_path = _accepted_execution_path(
        mode_label=mode_label,
        response=response,
        stats=stats,
    )
    model_path_classification = _model_path_classification(response)
    row: JsonObject = {
        "kind": "cluster_benchmark_metric",
        "cluster_path": "exo_api_bench_chat_completions",
        "live_execution_path": model_path_classification["path"],
        "accepted_execution_path": accepted_execution_path,
        "model_path_classification": model_path_classification,
        "api_base": api_base,
        "mode": mode_label,
        "model": model_id,
        "repeat_index": repeat_index,
        "http_status": status_code,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "generation_tps": generation_tps,
        "generation_tokens": generation_tokens,
        "prompt_tps": prompt_tps,
        "generation_stats": stats,
        "power_usage": response.get("power_usage"),
        "payload_extra_keys": payload_extra_keys,
        "next_step": (
            "use this as the distributed AR baseline"
            if mode_label == "ar"
            else "compare only against same-cluster AR rows and confirm the response used the guarded MTP path"
        ),
    }
    if mode_label != "ar":
        mtp_speedup_budget = build_mtp_speedup_budget(generation_tps)
        if mtp_speedup_budget is not None:
            row["mtp_speedup_budget"] = mtp_speedup_budget
            row["mtp_throughput_threshold"] = classify_mtp_throughput_threshold(
                cast(float, mtp_speedup_budget["generation_tps"])
            )
    return row


def _render_cluster_benchmark_rerun_command(args: ClusterBenchmarkArgs) -> str:
    command = [
        "uv",
        "run",
        "python3",
        "scripts/bench_mimo_mtp_cluster.py",
        "--api-base",
        args.api_base,
        "--model-id",
        args.model_id,
        "--max-tokens",
        str(args.max_tokens),
        "--repeats",
        str(args.repeats),
        "--mode-label",
        args.mode_label,
    ]
    if args.list_models:
        command.append("--list-models")
    if args.payload_extra_json is not None:
        command.extend(["--payload-extra-json", args.payload_extra_json])
    return " ".join(shlex.quote(part) for part in command)


def _cluster_error_row(
    *,
    api_base: str,
    mode_label: str,
    repeat_index: int | None,
    error: str,
    stage: str,
    success_row_count: int,
    rerun_command: str,
) -> JsonObject:
    return {
        "kind": "cluster_benchmark_error",
        "cluster_path": "exo_api_bench_chat_completions",
        "api_base": api_base,
        "mode": mode_label,
        "repeat_index": repeat_index,
        "stage": stage,
        "error": error,
        "status": "blocked_with_command",
        "success_row_count": success_row_count,
        "rerun_command": rerun_command,
        "next_step": "start or point to an exo cluster API that can serve /bench/chat/completions, then rerun the command in rerun_command",
    }


def _models_probe_row(
    *,
    api_base: str,
    status_code: int,
    response: JsonObject,
) -> JsonObject:
    data: object = response.get("data")
    model_ids: list[str] = []
    if isinstance(data, list):
        for raw_item in cast(list[object], data):
            item = raw_item
            if isinstance(item, dict):
                mapped_item = cast(Mapping[str, object], item)
                model_id = mapped_item.get("id")
                if isinstance(model_id, str):
                    model_ids.append(model_id)
    return {
        "kind": "cluster_models_probe",
        "api_base": api_base,
        "http_status": status_code,
        "model_ids": model_ids,
        "model_count": len(model_ids),
    }


def _float_metric(row: Mapping[str, object], key: str) -> float | None:
    raw_value = row.get(key)
    if isinstance(raw_value, bool):
        return None
    if isinstance(raw_value, int | float):
        return float(raw_value)
    return None


def _nested_float_metric(
    row: Mapping[str, object], *, parent_key: str, child_key: str
) -> float | None:
    raw_parent = row.get(parent_key)
    if not isinstance(raw_parent, dict):
        return None
    return _float_metric(cast(Mapping[str, object], raw_parent), child_key)


def _median_float(values: Iterable[float]) -> float | None:
    sorted_values = sorted(values)
    value_count = len(sorted_values)
    if value_count == 0:
        return None
    midpoint = value_count // 2
    if value_count % 2 == 1:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0


def _positive_generation_tps(row: Mapping[str, object]) -> float | None:
    generation_tps = _float_metric(row, "generation_tps")
    if generation_tps is None or generation_tps <= 0.0:
        return None
    return generation_tps


def _is_cluster_metric_row(row: Mapping[str, object]) -> bool:
    return (
        row.get("kind") == "cluster_benchmark_metric"
        and row.get("cluster_path") == "exo_api_bench_chat_completions"
    )


def _is_live_mtp_metric_row(row: Mapping[str, object]) -> bool:
    return (
        row.get("mode") != "ar"
        and row.get("accepted_execution_path") == "mimo_mtp_fastpath"
    )


def _cluster_generation_tps_values(
    rows: Iterable[Mapping[str, object]], *, mode_kind: Literal["ar", "mtp"]
) -> list[float]:
    values: list[float] = []
    for row in rows:
        if not _is_cluster_metric_row(row):
            continue
        if mode_kind == "ar":
            if row.get("mode") != "ar":
                continue
        elif not _is_live_mtp_metric_row(row):
            continue
        generation_tps = _positive_generation_tps(row)
        if generation_tps is not None:
            values.append(generation_tps)
    return values


def calculate_mtp_speedup_budget(rows: Iterable[Mapping[str, object]]) -> JsonObject:
    """Compute same-cluster AR-vs-MTP budget telemetry without fabricating rows."""
    row_list = list(rows)
    ar_generation_tps_values = _cluster_generation_tps_values(row_list, mode_kind="ar")
    mtp_generation_tps_values = _cluster_generation_tps_values(
        row_list, mode_kind="mtp"
    )
    ar_median_tps = _median_float(ar_generation_tps_values)
    mtp_median_tps = _median_float(mtp_generation_tps_values)

    if ar_median_tps is not None and mtp_median_tps is not None:
        status = "same_cluster_budget_ready"
        next_step = (
            "use this budget only with same-cluster AR-vs-MTP evidence rows; "
            "Slice 5 still requires a real MTP win without fallback concerns"
        )
    elif ar_median_tps is not None:
        status = "blocked_missing_mtp_rows"
        next_step = "collect same-cluster guarded MTP rows before making speedup or target claims"
    elif mtp_median_tps is not None:
        status = "blocked_missing_ar_rows"
        next_step = "collect same-cluster AR baseline rows before making speedup or target claims"
    else:
        status = "blocked_incomplete_telemetry"
        next_step = (
            "rerun same-cluster AR and guarded MTP benchmarks until generation_tps "
            "is present on both row sets"
        )

    return {
        "kind": "mimo_mtp_speedup_budget",
        "ar_row_count": len(ar_generation_tps_values),
        "mtp_row_count": len(mtp_generation_tps_values),
        "ar_median_tok_s": None if ar_median_tps is None else round(ar_median_tps, 6),
        "ar_ms_per_token": None
        if ar_median_tps is None
        else round(1000.0 / ar_median_tps, 6),
        "mtp_median_tok_s": None
        if mtp_median_tps is None
        else round(mtp_median_tps, 6),
        "mtp_target_gap_to_30_tok_s": None
        if mtp_median_tps is None
        else round(max(0.0, MTP_TARGET_TOKENS_PER_SECOND - mtp_median_tps), 6),
        "mtp_target_gap_to_40_tok_s": None
        if mtp_median_tps is None
        else round(max(0.0, MTP_PREFERRED_TOKENS_PER_SECOND - mtp_median_tps), 6),
        "mtp_vs_ar_speedup_ratio": None
        if ar_median_tps is None or mtp_median_tps is None
        else round(mtp_median_tps / ar_median_tps, 6),
        "status": status,
        "next_step": next_step,
    }


def _ms_per_token(generation_tps: float) -> float | None:
    if generation_tps <= 0.0:
        return None
    return round(1000.0 / generation_tps, 4)


def _is_same_cluster_ar_vs_mtp(
    *, ar_row: Mapping[str, object], mtp_row: Mapping[str, object]
) -> bool:
    return (
        ar_row.get("kind") == "cluster_benchmark_metric"
        and mtp_row.get("kind") == "cluster_benchmark_metric"
        and ar_row.get("cluster_path") == "exo_api_bench_chat_completions"
        and mtp_row.get("cluster_path") == "exo_api_bench_chat_completions"
        and ar_row.get("mode") == "ar"
        and mtp_row.get("mode") != "ar"
        and ar_row.get("model") == mtp_row.get("model")
    )


def _mtp_bottlenecks(
    *, mtp_beats_ar: bool, mtp_generation_tps: float, mtp_row: Mapping[str, object]
) -> list[str]:
    bottlenecks: list[str] = []
    if mtp_beats_ar:
        bottlenecks.append("mtp_beats_ar")
    if mtp_generation_tps >= 30.0:
        bottlenecks.append("mtp_reaches_30")
    if mtp_generation_tps >= 40.0:
        bottlenecks.append("mtp_reaches_40")
    if mtp_beats_ar:
        return bottlenecks

    acceptance_rate = _nested_float_metric(
        mtp_row, parent_key="generation_stats", child_key="acceptance_rate"
    )
    if acceptance_rate is None:
        acceptance_rate = _float_metric(mtp_row, "acceptance_rate")
    if acceptance_rate is not None and acceptance_rate < 0.5:
        bottlenecks.append("acceptance_rate_low")
    if not bottlenecks:
        bottlenecks.append("verifier_too_slow")
    return bottlenecks


def classify_mtp_vs_ar_baseline(
    *, ar_row: Mapping[str, object], mtp_row: Mapping[str, object]
) -> JsonObject:
    """Classify one same-cluster AR baseline row against one guarded MTP row."""
    ar_generation_tps = _float_metric(ar_row, "generation_tps")
    mtp_generation_tps = _float_metric(mtp_row, "generation_tps")
    same_cluster_comparison = _is_same_cluster_ar_vs_mtp(ar_row=ar_row, mtp_row=mtp_row)
    if (
        ar_generation_tps is None
        or ar_generation_tps <= 0.0
        or mtp_generation_tps is None
        or mtp_generation_tps <= 0.0
        or not same_cluster_comparison
    ):
        mtp_telemetry_absent = mtp_generation_tps is None or mtp_generation_tps <= 0.0
        return {
            "kind": "mtp_vs_ar_baseline_classification",
            "classification": "ambiguous",
            "bottlenecks": ["absent_mtp_telemetry"] if mtp_telemetry_absent else [],
            "ar_generation_tps": ar_generation_tps,
            "ar_ms_per_token": None
            if ar_generation_tps is None
            else _ms_per_token(ar_generation_tps),
            "mtp_generation_tps": mtp_generation_tps,
            "mtp_ms_per_token": None
            if mtp_generation_tps is None
            else _ms_per_token(mtp_generation_tps),
            "speedup_ratio": None,
            "target_gap_tps_30": None
            if mtp_generation_tps is None
            else round(max(0.0, 30.0 - mtp_generation_tps), 4),
            "target_gap_tps_40": None
            if mtp_generation_tps is None
            else round(max(0.0, 40.0 - mtp_generation_tps), 4),
            "same_cluster_comparison": same_cluster_comparison,
            "reason": "MTP row lacks live fastpath generation_tps telemetry; cannot classify pass or fail"
            if mtp_telemetry_absent
            else "classification requires positive same-cluster AR and MTP generation_tps rows",
            "next_step": "collect same-cluster guarded MTP rows with accepted_execution_path=mimo_mtp_fastpath and generation_tps before making speedup or target claims"
            if mtp_telemetry_absent
            else "collect same-model AR and guarded MTP rows from /bench/chat/completions before making speedup claims",
        }

    mtp_beats_ar = mtp_generation_tps > ar_generation_tps
    return {
        "kind": "mtp_vs_ar_baseline_classification",
        "classification": "pass" if mtp_beats_ar else "fail",
        "bottlenecks": _mtp_bottlenecks(
            mtp_beats_ar=mtp_beats_ar,
            mtp_generation_tps=mtp_generation_tps,
            mtp_row=mtp_row,
        ),
        "ar_generation_tps": round(ar_generation_tps, 4),
        "ar_ms_per_token": _ms_per_token(ar_generation_tps),
        "mtp_generation_tps": round(mtp_generation_tps, 4),
        "mtp_ms_per_token": _ms_per_token(mtp_generation_tps),
        "speedup_ratio": round(mtp_generation_tps / ar_generation_tps, 4),
        "target_gap_tps_30": round(max(0.0, 30.0 - mtp_generation_tps), 4),
        "target_gap_tps_40": round(max(0.0, 40.0 - mtp_generation_tps), 4),
        "same_cluster_comparison": same_cluster_comparison,
        "reason": "MTP row beats same-cluster AR baseline"
        if mtp_beats_ar
        else "MTP row does not beat same-cluster AR baseline",
        "next_step": "eligible for measured bottleneck analysis and guarded Slice 5 review only if median rows repeat this win"
        if mtp_beats_ar
        else "do not claim MTP speedup; inspect acceptance/proposal/verification/fallback telemetry before further optimization",
    }


def validate_rollout_baseline_row(row: Mapping[str, object]) -> JsonObject:
    """Validate whether a benchmark row is eligible as a MiMo rollout baseline."""
    required_source = "exo cluster /bench/chat/completions AR row"
    kind = row.get("kind")
    mode = row.get("mode")
    cluster_path = row.get("cluster_path")
    if (
        kind == "cluster_benchmark_metric"
        and cluster_path == "exo_api_bench_chat_completions"
        and mode == "ar"
    ):
        classification = row.get("model_path_classification")
        if isinstance(classification, dict):
            mapped_classification = cast(Mapping[str, object], classification)
            if mapped_classification.get("path") == "single_studio_full_model_load":
                return {
                    "kind": "rollout_baseline_validation",
                    "allowed": False,
                    "baseline_source": "single_studio_full_model_load",
                    "reason": "single-Studio full-model loading is not an allowed MiMo rollout baseline",
                    "required_source": required_source,
                    "next_step": "collect an AR row with scripts/bench_mimo_mtp_cluster.py against a running tensor-parallel exo cluster API",
                }
        return {
            "kind": "rollout_baseline_validation",
            "allowed": True,
            "baseline_source": "exo_cluster_bench_chat_completions",
            "reason": "cluster AR row is eligible as a MiMo rollout baseline",
            "required_source": required_source,
            "next_step": "compare only with same-cluster guarded MTP rows",
        }
    if kind == "benchmark_metric":
        return {
            "kind": "rollout_baseline_validation",
            "allowed": False,
            "baseline_source": "single_studio_full_model_load",
            "reason": "single-Studio full-model benchmark rows are diagnostic only and cannot be used as MiMo rollout baselines",
            "required_source": required_source,
            "next_step": "collect an AR row with scripts/bench_mimo_mtp_cluster.py against a running exo cluster API",
        }
    return {
        "kind": "rollout_baseline_validation",
        "allowed": False,
        "baseline_source": "not_exo_cluster_bench_chat_completions",
        "reason": "rollout baselines must be AR metric rows from the exo cluster benchmark endpoint",
        "required_source": required_source,
        "next_step": "collect an AR row with scripts/bench_mimo_mtp_cluster.py against a running exo cluster API",
    }


def run_cluster_benchmark(
    *,
    args: ClusterBenchmarkArgs,
    http_post: HttpJsonPost,
    http_get: HttpJsonGet,
    timer: Callable[[], float] = time.perf_counter,
) -> list[JsonObject]:
    rows: list[JsonObject] = []
    rerun_command = _render_cluster_benchmark_rerun_command(args)
    try:
        payload_extra = _payload_extra(args.payload_extra_json)
    except (json.JSONDecodeError, ValueError) as exc:
        return [
            _cluster_error_row(
                api_base=args.api_base,
                mode_label=args.mode_label,
                repeat_index=None,
                stage="payload_validation",
                error=str(exc),
                success_row_count=0,
                rerun_command=rerun_command,
            )
        ]

    if args.list_models:
        try:
            status_code, response = http_get(
                f"{args.api_base}/v1/models", args.timeout_seconds
            )
            rows.append(
                _models_probe_row(
                    api_base=args.api_base,
                    status_code=status_code,
                    response=response,
                )
            )
        except (OSError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            rows.append(
                _cluster_error_row(
                    api_base=args.api_base,
                    mode_label=args.mode_label,
                    repeat_index=None,
                    stage="models_probe",
                    error=str(exc),
                    success_row_count=sum(
                        1 for row in rows if row["kind"] == "cluster_benchmark_metric"
                    ),
                    rerun_command=rerun_command,
                )
            )
            return rows

    payload = build_cluster_chat_payload(
        model_id=args.model_id,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        payload_extra=payload_extra,
    )
    payload_extra_keys = sorted(payload_extra.keys())
    for repeat_index in range(args.repeats):
        request_payload = dict(payload)
        request_payload["benchmark_mode_label"] = args.mode_label
        request_payload["benchmark_repeat_index"] = repeat_index
        start = timer()
        try:
            status_code, response = http_post(
                f"{args.api_base}/bench/chat/completions",
                request_payload,
                args.timeout_seconds,
            )
        except (OSError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            rows.append(
                _cluster_error_row(
                    api_base=args.api_base,
                    mode_label=args.mode_label,
                    repeat_index=repeat_index,
                    stage="bench_chat_completions",
                    error=str(exc),
                    success_row_count=sum(
                        1 for row in rows if row["kind"] == "cluster_benchmark_metric"
                    ),
                    rerun_command=rerun_command,
                )
            )
            continue
        rows.append(
            build_cluster_metric_row(
                api_base=args.api_base,
                model_id=args.model_id,
                mode_label=args.mode_label,
                repeat_index=repeat_index,
                elapsed_seconds=max(0.0, timer() - start),
                status_code=status_code,
                response=response,
                payload_extra_keys=payload_extra_keys,
            )
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.matrix_commands:
        rows = render_benchmark_matrix_command_rows(
            api_base=args.api_base,
            model_id=args.model_id,
            max_tokens_values=args.matrix_max_tokens,
            repeats=args.repeats,
            mimo_mtp_sidecar_path=args.mimo_mtp_sidecar_path,
        )
    else:
        rows = run_cluster_benchmark(
            args=args, http_post=_http_json_post, http_get=_http_json_get
        )
    for row in rows:
        print(render_json_line(row), flush=True)
    return 1 if any(row["kind"] == "cluster_benchmark_error" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
