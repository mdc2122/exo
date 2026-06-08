#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Final, Literal, Protocol, cast

from exo.worker.engines.mlx.mimo_mtp_fast.benchmark import render_json_line

try:
    from scripts import mimo_mtp_bottleneck_classifier as mtp_bottleneck_classifier
except ModuleNotFoundError:
    import mimo_mtp_bottleneck_classifier as mtp_bottleneck_classifier

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
CANONICAL_BENCHMARK_ROW_SCHEMA_VERSION: Final[int] = 1
CANONICAL_BENCHMARK_ROW_STATUSES: Final[tuple[str, ...]] = (
    "blocked",
    "fail_closed",
    "fail_open",
    "invalid",
    "live",
)
CANONICAL_MTP_EXECUTION_STATES: Final[tuple[str, ...]] = (
    "blocked_unavailable",
    "compatible_attempted",
    "disabled_default",
    "enabled_intent",
    "fail_closed_error",
    "fail_open_fallback",
    "missing_sidecar",
    "runtime_error",
    "successful_mtp",
    "unsupported_depth",
    "unsupported_model",
    "unwired_execution",
)
CANONICAL_BENCHMARK_ROW_REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "evidence_kind",
    "row_status",
    "benchmark_session_id",
    "endpoint",
    "model_id",
    "temperature",
    "max_tokens",
    "mode",
    "repeat_index",
    "generation_tps",
    "generation_tokens",
    "prompt_tps",
    "power_usage",
    "payload_extra",
    "mtp_enabled",
    "mtp_depth",
    "mtp_execution_state",
    "mtp_disable_reason",
    "telemetry_completeness",
)
CANONICAL_LIVE_TIMING_THROUGHPUT_FIELDS: Final[tuple[str, ...]] = (
    "generation_tps",
    "generation_tokens",
    "prompt_tps",
)
CANONICAL_SUCCESSFUL_MTP_TELEMETRY_FIELDS: Final[tuple[str, ...]] = (
    "accepted_execution_path=mimo_mtp_fastpath",
    "accepted_depth_counts",
    "acceptance_rate",
    "attempted_depth_counts",
    "fallback_count",
    "mtp_sidecar_status",
    "timing_breakdown_seconds",
)
CANONICAL_FAIL_CLOSED_MTP_TELEMETRY_FIELDS: Final[tuple[str, ...]] = (
    "mtp_disable_reason",
    "mtp_sidecar_status",
)
_CANONICAL_BENCHMARK_ROW_REQUIRED_ONE_OF: Final[tuple[tuple[str, ...], ...]] = (
    ("api_url", "cluster_id"),
    ("prompt_id", "prompt_hash"),
)
_BENCH_RESPONSE_KNOWN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "accepted_depth_counts",
        "accepted_execution_path",
        "accepted_tokens",
        "acceptance_rate",
        "attempted_depth_counts",
        "attempted_tokens",
        "choices",
        "created",
        "execution_path",
        "fallback_count",
        "generation_stats",
        "generation_tokens",
        "generation_tps",
        "id",
        "mimo_mtp_accepted_depth_counts",
        "mimo_mtp_accepted_tokens",
        "mimo_mtp_attempted_depth_counts",
        "mimo_mtp_attempted_tokens",
        "mimo_mtp_fallback_count",
        "mimo_mtp_timing_breakdown_seconds",
        "model",
        "mtp_depth",
        "mtp_disable_reason",
        "mtp_enabled",
        "mtp_execution_state",
        "mtp_fallback_reason",
        "mtp_sidecar_status",
        "object",
        "power_usage",
        "prompt_tps",
        "requested_mtp_depth",
        "service_tier",
        "timing_breakdown_seconds",
        "usage",
    }
)

_CANONICAL_CLUSTER_METRIC_ROW_KEYS: Final[frozenset[str]] = frozenset(
    {
        "accepted_depth_counts",
        "accepted_execution_path",
        "acceptance_rate",
        "api_base",
        "attempted_depth_counts",
        "cluster_path",
        "elapsed_seconds",
        "fallback_count",
        "generation_stats",
        "generation_tokens",
        "generation_tps",
        "http_status",
        "kind",
        "schema_version",
        "evidence_kind",
        "row_status",
        "benchmark_session_id",
        "api_url",
        "cluster_id",
        "endpoint",
        "prompt_id",
        "prompt_hash",
        "temperature",
        "telemetry_completeness",
        "mtp_execution_state",
        "live_execution_path",
        "live_model_path_validation",
        "max_tokens",
        "mode",
        "mode_label",
        "model",
        "model_id",
        "model_path_classification",
        "mtp_depth",
        "mtp_disable_reason",
        "mtp_enabled",
        "mtp_fallback_kind",
        "mtp_fallback_mode",
        "mtp_fallback_reason",
        "mtp_sidecar_status",
        "mtp_speedup_budget",
        "mtp_throughput_threshold",
        "next_step",
        "payload_extra",
        "payload_extra_canonical_collision_keys",
        "payload_extra_keys",
        "power_usage",
        "prompt_tps",
        "repeat_index",
        "requested_max_tokens",
        "requested_mtp_depth",
        "timing_breakdown_seconds",
    }
)


def extract_response_payload_extra(
    response: Mapping[str, object],
) -> JsonObject:
    """Extract unrecognized extra keys from an API response into payload_extra.

    Maps a response payload's extra keys (those not consumed by the canonical
    field extraction in ``build_cluster_metric_row`` and its helpers) into the
    telemetry row's ``payload_extra`` dict. Known keys that are explicitly
    mapped into benchmark row fields are excluded; all unrecognized keys are
    preserved so that custom metadata, version stamps, and future extension
    fields survive row parsing and are accessible on the parsed object.

    This function is pure: it does not mutate its inputs and produces a new
    dict containing only the unrecognized key-value pairs.
    """
    return {
        key: value
        for key, value in response.items()
        if key not in _BENCH_RESPONSE_KNOWN_KEYS
    }


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
    output_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class ClusterAvailabilityPreflightResult:
    available: bool
    api_base: str
    probe_url: str
    stage: str
    status_code: int | None
    response: JsonObject | None
    error: str | None


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the canonical live cluster benchmark CLI parser without side effects."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark MiMo MTP work through an already-running exo cluster API "
            "instead of materializing the full model in this process. "
            "Supported same-cluster AR baseline harness; default mode-label is ar."
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for persisted AR baseline JSONL rows; available "
            "cluster AR metric rows are saved per requested max_tokens budget."
        ),
    )
    return parser


def _parse_args(argv: list[str] | None = None) -> ClusterBenchmarkArgs:
    namespace = build_cli_parser().parse_args(argv)
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
        output_dir=cast(Path | None, namespace.output_dir),
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
    prompt: str | None = None,
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
    ]
    if prompt is not None:
        command_parts.extend(["--prompt", shlex.quote(prompt)])
    command_parts.extend(
        [
            "--max-tokens",
            str(max_tokens),
            "--repeats",
            str(repeats),
            "--mode-label",
            shlex.quote(mode_label),
        ]
    )
    if list_models:
        command_parts.append("--list-models")
    if payload_extra_json is not None:
        command_parts.extend(["--payload-extra-json", shlex.quote(payload_extra_json)])
    return " ".join(command_parts)


def build_blocked_benchmark_row_from_context(
    *,
    api_base: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    repeats: int,
    mode_label: str,
    payload_extra: Mapping[str, object] | None,
    output_path: str | Path,
    blocker_reason: str,
    benchmark_session_id: str,
    environment_assumptions: Mapping[str, object],
    live_runner: Callable[[], object] | None = None,
) -> JsonObject:
    """Build blocked benchmark evidence without contacting the live cluster.

    ``live_runner`` is accepted only as a regression guard for callers/tests that
    need to prove blocked evidence construction is pure; it is intentionally not
    invoked here.
    """
    del live_runner
    normalized_api_base = api_base.rstrip("/")
    normalized_payload_extra: JsonObject = (
        {} if payload_extra is None else dict(payload_extra)
    )
    payload_extra_json = (
        None
        if not normalized_payload_extra
        else json.dumps(normalized_payload_extra, sort_keys=True, separators=(",", ":"))
    )
    command = render_cluster_benchmark_command(
        api_base=normalized_api_base,
        model_id=model_id,
        prompt=prompt,
        max_tokens=max_tokens,
        repeats=repeats,
        mode_label=mode_label,
        list_models=mode_label == "ar",
        payload_extra_json=payload_extra_json,
    )
    output_path_string = str(output_path)
    command = f"{command} --output-dir {shlex.quote(str(Path(output_path).parent))}"

    mtp_enabled = bool(normalized_payload_extra.get("mimo_mtp_fastpath"))
    raw_mtp_depth = normalized_payload_extra.get("mimo_mtp_depth")
    mtp_depth = (
        raw_mtp_depth
        if isinstance(raw_mtp_depth, int) and not isinstance(raw_mtp_depth, bool)
        else None
    )
    if mode_label == "ar":
        mtp_enabled = False
        mtp_depth = None

    return {
        "kind": "cluster_benchmark_blocked_row",
        "schema_version": "mimo_mtp_benchmark_row.v1",
        "evidence_kind": "blocked_evidence",
        "row_status": "blocked_unavailable",
        "benchmark_session_id": benchmark_session_id,
        "cluster_path": "exo_api_bench_chat_completions",
        "api_base": normalized_api_base,
        "endpoint": "/bench/chat/completions",
        "model": model_id,
        "model_id": model_id,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "mode": mode_label,
        "repeat_index": None,
        "generation_tps": None,
        "generation_tokens": None,
        "prompt_tps": None,
        "power_usage": None,
        "payload_extra": normalized_payload_extra,
        "payload_extra_keys": sorted(normalized_payload_extra.keys()),
        "mtp_enabled": mtp_enabled,
        "mtp_depth": mtp_depth,
        "mtp_execution_state": "blocked_unavailable",
        "mtp_disable_reason": blocker_reason,
        "telemetry_completeness": "blocked_no_live_execution",
        "commands": [{"command_kind": "rerun_benchmark", "command": command}],
        "environment_assumptions": dict(environment_assumptions),
        "expected_output_path": output_path_string,
        "blocker_reason": blocker_reason,
        "no_performance_result_claimed": True,
        "next_step": "run the rerun_benchmark command against an available same-cluster exo API; do not make throughput, speedup, or Slice 5 eligibility claims from this blocked row",
    }


def build_exact_ar_baseline_command_row(
    *, api_base: str, model_id: str, max_tokens: int
) -> JsonObject:
    """Build a benchmark evidence command row for one exact AR token budget."""
    normalized_api_base = api_base.rstrip("/")
    return {
        "kind": "cluster_benchmark_command",
        "cluster_path": "exo_api_bench_chat_completions",
        "benchmark_endpoint": "/bench/chat/completions",
        "api_base": normalized_api_base,
        "mode": "ar",
        "model": model_id,
        "model_id": model_id,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "repeats": 1,
        "mtp_enabled": False,
        "payload_extra_json": None,
        "command": render_cluster_benchmark_command(
            api_base=normalized_api_base,
            model_id=model_id,
            max_tokens=max_tokens,
            repeats=1,
            mode_label="ar",
            list_models=True,
        ),
        "next_step": "run this same-cluster AR command before making MTP speedup or Slice 5 claims",
    }


def render_ar_baseline_commands(*, api_base: str, model_id: str) -> list[str]:
    """Render exact runnable same-cluster AR baseline commands for rollout evidence."""
    return [
        str(
            build_exact_ar_baseline_command_row(
                api_base=api_base,
                model_id=model_id,
                max_tokens=max_tokens,
            )["command"]
        )
        for max_tokens in AR_BASELINE_MAX_TOKENS
    ]


def build_ar_baseline_blocked_status(
    *, api_base: str, model_id: str, reason: str
) -> JsonObject:
    """Record required AR baseline commands when live cluster collection is blocked."""
    commands: list[JsonObject] = []
    for max_tokens, command in zip(
        AR_BASELINE_MAX_TOKENS,
        render_ar_baseline_commands(api_base=api_base, model_id=model_id),
        strict=True,
    ):
        commands.append({"max_tokens": max_tokens, "command": command})
    return {
        "kind": "ar_baseline_collection_status",
        "status": "blocked_with_command",
        "cluster_path": "exo_api_bench_chat_completions",
        "benchmark_endpoint": "/bench/chat/completions",
        "api_base": api_base.rstrip("/"),
        "model": model_id,
        "model_id": model_id,
        "mode": "ar",
        "reason": reason,
        "required_max_tokens": list(AR_BASELINE_MAX_TOKENS),
        "commands": commands,
        **_blocked_non_fabrication_fields(rerun_field="the exact commands"),
        "next_step": "start or point to an exo cluster API, then run each command in commands to collect same-cluster AR baseline rows before MTP speedup or Slice 5 claims",
    }


def render_ar_baseline_command_documentation(*, api_base: str, model_id: str) -> str:
    """Render the documented exact commands for same-cluster AR baselines."""
    command_blocks = "\n\n".join(
        f"```bash\n{command}\n```"
        for command in render_ar_baseline_commands(api_base=api_base, model_id=model_id)
    )
    return (
        "## Exact AR baseline commands\n\n"
        "Collect same-cluster AR baseline rows through `/bench/chat/completions` "
        "before making any MTP speedup or Slice 5 claims. These commands keep "
        "MTP disabled by using AR mode and no experimental payload flag.\n\n"
        + command_blocks
    )


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
    prompt: str,
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
                    "benchmark_endpoint": "/bench/chat/completions",
                    "api_base": api_base.rstrip("/"),
                    "mode": mode_label,
                    "model": model_id,
                    "prompt": prompt,
                    "temperature": 0.0,
                    "max_tokens": max_tokens,
                    "repeats": repeats,
                    "payload_extra_json": payload_extra_json,
                    "mtp_enabled": mode_label != "ar",
                    "mtp_depth": None
                    if mode_label == "ar"
                    else int(mode_label.removeprefix("mtp-d")),
                    "mtp_execution_state": "disabled_default"
                    if mode_label == "ar"
                    else "enabled_intent",
                    "mtp_disable_reason": "default_ar_no_mtp_payload"
                    if mode_label == "ar"
                    else None,
                    "telemetry_completeness": "command_only_no_live_execution",
                    "command": render_cluster_benchmark_command(
                        api_base=api_base,
                        model_id=model_id,
                        prompt=prompt,
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
    top_level_stats = {
        key: response[key]
        for key in ("prompt_tps", "generation_tps", "generation_tokens")
        if key in response
    }
    if top_level_stats:
        return top_level_stats
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


def _first_present_value(
    *,
    response: Mapping[str, object],
    stats: Mapping[str, object] | None,
    keys: tuple[str, ...],
) -> object | None:
    for source in (response, stats):
        if source is None:
            continue
        for key in keys:
            if key in source:
                return source[key]
    return None


def _optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _optional_string(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _optional_mapping(value: object) -> JsonObject | None:
    if isinstance(value, dict):
        return dict(cast(Mapping[str, object], value))
    return None


def _fallback_mode(fallback_reason: str | None) -> str | None:
    if fallback_reason is None:
        return None
    if "fail_open" in fallback_reason:
        return "fail_open"
    if "fail_closed" in fallback_reason:
        return "fail_closed"
    return None


def _fallback_kind(
    *, disable_reason: str | None, fallback_reason: str | None
) -> str | None:
    reason_text = " ".join(
        reason for reason in (disable_reason, fallback_reason) if reason is not None
    )
    if not reason_text:
        return None
    if "missing_sidecar" in reason_text:
        return "missing_sidecar"
    if "unsupported" in reason_text:
        return "unsupported"
    if "low_acceptance" in reason_text:
        return "low_acceptance"
    if "runtime_error" in reason_text:
        return "runtime_error"
    if "fail_open" in reason_text:
        return "fail_open"
    return None


def _guarded_mtp_accepted_execution_path(
    *, mode_label: str, accepted_execution_path: str, telemetry: Mapping[str, object]
) -> str:
    """Prevent disabled or failed MTP telemetry from claiming live MTP execution."""
    if mode_label == "ar" or accepted_execution_path != "mimo_mtp_fastpath":
        return accepted_execution_path
    if telemetry.get("mtp_enabled") is False:
        fallback_reason = _optional_string(telemetry.get("mtp_fallback_reason"))
        if fallback_reason is not None:
            return "ar"
        return "rejected"
    if (
        telemetry.get("mtp_depth") is None
        and telemetry.get("mtp_disable_reason") is not None
    ):
        return "rejected"
    return accepted_execution_path


def _acceptance_rate_from_telemetry(
    *,
    response: Mapping[str, object],
    stats: Mapping[str, object] | None,
) -> float | None:
    explicit_rate = _optional_float(
        _first_present_value(response=response, stats=stats, keys=("acceptance_rate",))
    )
    if explicit_rate is not None:
        return explicit_rate
    attempted_tokens = _optional_float(
        _first_present_value(
            response=response,
            stats=stats,
            keys=("mimo_mtp_attempted_tokens", "attempted_tokens"),
        )
    )
    accepted_tokens = _optional_float(
        _first_present_value(
            response=response,
            stats=stats,
            keys=("mimo_mtp_accepted_tokens", "accepted_tokens"),
        )
    )
    if attempted_tokens is None or accepted_tokens is None or attempted_tokens <= 0.0:
        return None
    return accepted_tokens / attempted_tokens


def _mtp_telemetry_fields(
    *,
    mode_label: str,
    response: Mapping[str, object],
    stats: Mapping[str, object] | None,
) -> JsonObject:
    mtp_enabled = _optional_bool(
        _first_present_value(response=response, stats=stats, keys=("mtp_enabled",))
    )
    if mtp_enabled is None and mode_label == "ar":
        mtp_enabled = False

    mtp_depth = _optional_int(
        _first_present_value(response=response, stats=stats, keys=("mtp_depth",))
    )
    requested_mtp_depth = _optional_int(
        _first_present_value(
            response=response, stats=stats, keys=("requested_mtp_depth",)
        )
    )
    mtp_sidecar_status = _optional_string(
        _first_present_value(
            response=response, stats=stats, keys=("mtp_sidecar_status",)
        )
    )
    mtp_disable_reason = _optional_string(
        _first_present_value(
            response=response, stats=stats, keys=("mtp_disable_reason",)
        )
    )
    mtp_fallback_reason = _optional_string(
        _first_present_value(
            response=response, stats=stats, keys=("mtp_fallback_reason",)
        )
    )
    attempted_depth_counts = _optional_mapping(
        _first_present_value(
            response=response,
            stats=stats,
            keys=("attempted_depth_counts", "mimo_mtp_attempted_depth_counts"),
        )
    )
    accepted_depth_counts = _optional_mapping(
        _first_present_value(
            response=response,
            stats=stats,
            keys=("accepted_depth_counts", "mimo_mtp_accepted_depth_counts"),
        )
    )
    fallback_count = _optional_int(
        _first_present_value(
            response=response,
            stats=stats,
            keys=("fallback_count", "mimo_mtp_fallback_count"),
        )
    )
    timing_breakdown = _optional_mapping(
        _first_present_value(
            response=response,
            stats=stats,
            keys=(
                "timing_breakdown_seconds",
                "mimo_mtp_timing_breakdown_seconds",
            ),
        )
    )
    acceptance_rate = _acceptance_rate_from_telemetry(response=response, stats=stats)

    telemetry: JsonObject = {}
    if mtp_enabled is not None:
        telemetry["mtp_enabled"] = mtp_enabled
    if requested_mtp_depth is not None:
        telemetry["requested_mtp_depth"] = requested_mtp_depth
    if mtp_depth is not None or mode_label == "ar":
        telemetry["mtp_depth"] = mtp_depth
    if mtp_sidecar_status is not None:
        telemetry["mtp_sidecar_status"] = mtp_sidecar_status
    if mtp_disable_reason is not None:
        telemetry["mtp_disable_reason"] = mtp_disable_reason
    if mtp_fallback_reason is not None:
        telemetry["mtp_fallback_reason"] = mtp_fallback_reason
    if attempted_depth_counts is not None:
        telemetry["attempted_depth_counts"] = attempted_depth_counts
    if accepted_depth_counts is not None:
        telemetry["accepted_depth_counts"] = accepted_depth_counts
    if acceptance_rate is not None:
        telemetry["acceptance_rate"] = acceptance_rate
    if fallback_count is not None:
        telemetry["fallback_count"] = fallback_count
    if timing_breakdown is not None:
        telemetry["timing_breakdown_seconds"] = timing_breakdown

    fallback_mode = _fallback_mode(mtp_fallback_reason)
    if fallback_mode is not None:
        telemetry["mtp_fallback_mode"] = fallback_mode
    fallback_kind = _fallback_kind(
        disable_reason=mtp_disable_reason, fallback_reason=mtp_fallback_reason
    )
    if fallback_kind is not None:
        telemetry["mtp_fallback_kind"] = fallback_kind
    return telemetry


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


def _sequence_of_mappings(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    raw_items = cast(list[object], value)
    mappings: list[Mapping[str, object]] = []
    for item in raw_items:
        if isinstance(item, dict):
            mappings.append(cast(Mapping[str, object], item))
    return mappings


def _proves_multi_participant_tensor_shards(
    *, execution_path: Mapping[str, object], requested_model_id: str | None
) -> bool:
    cluster_nodes = _sequence_of_mappings(execution_path.get("cluster_nodes"))
    tensor_shards = _sequence_of_mappings(execution_path.get("tensor_parallel_shards"))
    if len(cluster_nodes) < 2 or len(tensor_shards) < 2:
        return False
    cluster_node_ids = {
        str(node["node_id"])
        for node in cluster_nodes
        if isinstance(node.get("node_id"), str) and node.get("node_id")
    }
    tensor_node_ids = {
        str(shard["node_id"])
        for shard in tensor_shards
        if isinstance(shard.get("node_id"), str) and shard.get("node_id")
    }
    tensor_device_ranks: set[int] = set()
    for shard in tensor_shards:
        device_rank = shard.get("device_rank")
        if isinstance(device_rank, int) and not isinstance(device_rank, bool):
            tensor_device_ranks.add(device_rank)
    tensor_model_ids = {
        str(shard["model_id"])
        for shard in tensor_shards
        if isinstance(shard.get("model_id"), str) and shard.get("model_id")
    }
    if requested_model_id is not None and tensor_model_ids != {requested_model_id}:
        return False
    return (
        len(cluster_node_ids) > 1
        and len(tensor_node_ids) > 1
        and tensor_node_ids.issubset(cluster_node_ids)
        and len(tensor_device_ranks) > 1
        and len(tensor_shards) == len(cluster_nodes)
        and bool(tensor_model_ids)
    )


def validate_live_model_path(
    response: Mapping[str, object], requested_model_id: str | None = None
) -> JsonObject:
    """Validate and record the live model path reported by the cluster API."""
    raw_execution_path = response.get("execution_path")
    if not isinstance(raw_execution_path, dict):
        return {
            "kind": "live_model_path_validation",
            "allowed": False,
            "live_model_path": "unknown",
            "is_exo_cluster_tensor_parallel": False,
            "source": "response.execution_path",
            "sharding": None,
            "world_size": None,
            "model_path": None,
            "instance_id": None,
            "reason": "response did not include execution_path telemetry",
            "disable_reason": "response did not include execution_path telemetry",
            "next_step": "collect an AR row from a running tensor-parallel exo cluster API before using it as rollout evidence",
        }
    execution_path = cast(Mapping[str, object], raw_execution_path)
    raw_live_model_path = execution_path.get("path")
    live_model_path_from_response = (
        raw_live_model_path if isinstance(raw_live_model_path, str) else None
    )
    sharding = execution_path.get("sharding")
    world_size = execution_path.get("world_size")
    model_path = execution_path.get("model_path")
    instance_id = execution_path.get("instance_id")
    next_step = "collect an AR row with scripts/bench_mimo_mtp_cluster.py against a running tensor-parallel exo cluster API"
    if live_model_path_from_response == "single_studio_full_model_load":
        return {
            "kind": "live_model_path_validation",
            "allowed": False,
            "live_model_path": live_model_path_from_response,
            "is_exo_cluster_tensor_parallel": False,
            "source": "response.execution_path",
            "sharding": sharding,
            "world_size": world_size,
            "model_path": model_path,
            "instance_id": instance_id,
            "reason": "single-Studio full-model loading is not valid tensor-parallel evidence",
            "disable_reason": "response execution path self-reported single-Studio full-model loading",
            "next_step": next_step,
        }
    if (
        live_model_path_from_response is not None
        and live_model_path_from_response != "exo_cluster_tensor_parallel"
        and "full_model" in live_model_path_from_response
    ):
        return {
            "kind": "live_model_path_validation",
            "allowed": False,
            "live_model_path": live_model_path_from_response,
            "is_exo_cluster_tensor_parallel": False,
            "source": "response.execution_path",
            "sharding": sharding,
            "world_size": world_size,
            "model_path": model_path,
            "instance_id": instance_id,
            "reason": "non-cluster full-model loading is not valid tensor-parallel evidence",
            "disable_reason": "response execution path is a non-cluster full-model path",
            "next_step": next_step,
        }
    has_tensor_world_size = (
        sharding == "Tensor"
        and isinstance(world_size, int)
        and not isinstance(world_size, bool)
        and world_size > 1
    )
    has_multi_participant_tensor_evidence = _proves_multi_participant_tensor_shards(
        execution_path=execution_path, requested_model_id=requested_model_id
    )
    is_tensor_parallel_cluster = (
        has_tensor_world_size and has_multi_participant_tensor_evidence
    )
    live_model_path = (
        "exo_cluster_tensor_parallel"
        if is_tensor_parallel_cluster
        else "single_studio_full_model_load"
    )
    disable_reason = None
    reason = "exo cluster tensor parallelization is the live model path"
    next_step = "use this row as eligible same-cluster rollout evidence only with matching guarded MTP rows"
    if not is_tensor_parallel_cluster:
        disable_reason = "response execution path is not tensor-parallel across multiple cluster nodes"
        reason = (
            "single-Studio full-model loading is not an allowed MiMo rollout baseline"
        )
        if has_tensor_world_size:
            disable_reason = "response execution path does not prove multiple cluster participants host tensor shards for the requested model"
            reason = "tensor-parallel execution path lacks multi-participant tensor shard evidence"
        next_step = "collect an AR row with scripts/bench_mimo_mtp_cluster.py against a running tensor-parallel exo cluster API"
    return {
        "kind": "live_model_path_validation",
        "allowed": is_tensor_parallel_cluster,
        "live_model_path": live_model_path,
        "is_exo_cluster_tensor_parallel": is_tensor_parallel_cluster,
        "source": "response.execution_path",
        "sharding": sharding,
        "world_size": world_size,
        "model_path": model_path,
        "instance_id": instance_id,
        "reason": reason,
        "disable_reason": disable_reason,
        "next_step": next_step,
    }


def _model_path_classification(
    response: Mapping[str, object], requested_model_id: str | None = None
) -> JsonObject:
    validation = validate_live_model_path(
        response, requested_model_id=requested_model_id
    )
    return {
        "path": validation["live_model_path"],
        "is_exo_cluster_tensor_parallel": validation["is_exo_cluster_tensor_parallel"],
        "source": validation["source"],
        "sharding": validation["sharding"],
        "world_size": validation["world_size"],
        "model_path": validation["model_path"],
        "disable_reason": validation["disable_reason"],
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
    payload_extra_keys: list[str] | None = None,
    payload_extra: Mapping[str, object] | None = None,
    requested_max_tokens: int | None = None,
    benchmark_session_id: str | None = None,
    prompt_hash: str | None = None,
    prompt_id: str | None = None,
) -> JsonObject:
    normalized_payload_extra: JsonObject = (
        {} if payload_extra is None else dict(payload_extra)
    )
    response_extra = extract_response_payload_extra(response)
    for response_extra_key, response_extra_value in response_extra.items():
        normalized_payload_extra.setdefault(response_extra_key, response_extra_value)
    normalized_payload_extra_keys = (
        sorted(normalized_payload_extra.keys())
        if payload_extra_keys is None
        else sorted(payload_extra_keys)
    )
    payload_extra_canonical_collision_keys = sorted(
        key
        for key in normalized_payload_extra_keys
        if key in _CANONICAL_CLUSTER_METRIC_ROW_KEYS
    )
    stats = _generation_stats(response)
    generation_tps = None if stats is None else stats.get("generation_tps")
    generation_tokens = None if stats is None else stats.get("generation_tokens")
    prompt_tps = None if stats is None else stats.get("prompt_tps")
    raw_accepted_execution_path = _accepted_execution_path(
        mode_label=mode_label,
        response=response,
        stats=stats,
    )
    mtp_telemetry = _mtp_telemetry_fields(
        mode_label=mode_label,
        response=response,
        stats=stats,
    )
    accepted_execution_path = _guarded_mtp_accepted_execution_path(
        mode_label=mode_label,
        accepted_execution_path=raw_accepted_execution_path,
        telemetry=mtp_telemetry,
    )
    live_model_path_validation = validate_live_model_path(
        response, requested_model_id=model_id
    )
    model_path_classification = _model_path_classification(
        response, requested_model_id=model_id
    )
    row: JsonObject = {
        "kind": "cluster_benchmark_metric",
        "schema_version": CANONICAL_BENCHMARK_ROW_SCHEMA_VERSION,
        "evidence_kind": "benchmark_row",
        "row_status": "live",
        "benchmark_session_id": benchmark_session_id or "manual-build-unspecified",
        "api_url": api_base,
        "endpoint": "/bench/chat/completions",
        "prompt_hash": prompt_hash or "unrecorded",
        "temperature": 0.0,
        "cluster_path": "exo_api_bench_chat_completions",
        "live_execution_path": model_path_classification["path"],
        "accepted_execution_path": accepted_execution_path,
        "live_model_path_validation": live_model_path_validation,
        "model_path_classification": model_path_classification,
        "api_base": api_base,
        "mode": mode_label,
        "mode_label": mode_label,
        "model": model_id,
        "model_id": model_id,
        "repeat_index": repeat_index,
        "http_status": status_code,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "generation_tps": generation_tps,
        "generation_tokens": generation_tokens,
        "prompt_tps": prompt_tps,
        "generation_stats": stats,
        "power_usage": response.get("power_usage"),
        "payload_extra_keys": normalized_payload_extra_keys,
        "payload_extra": normalized_payload_extra,
        "payload_extra_canonical_collision_keys": payload_extra_canonical_collision_keys,
        "mtp_execution_state": _canonical_mtp_execution_state(
            mode_label=mode_label,
            accepted_execution_path=accepted_execution_path,
            response=response,
            stats=stats,
        ),
        "telemetry_completeness": "complete" if mode_label == "ar" else "partial",
        "next_step": (
            "use this as the distributed AR baseline"
            if mode_label == "ar"
            else "compare only against same-cluster AR rows and confirm the response used the guarded MTP path"
        ),
    }
    row.update(mtp_telemetry)
    if prompt_id is not None:
        row["prompt_id"] = prompt_id
    row.setdefault("mtp_enabled", False if mode_label == "ar" else None)
    row.setdefault("mtp_depth", None)
    row.setdefault("mtp_disable_reason", None)
    if requested_max_tokens is not None:
        row["max_tokens"] = requested_max_tokens
        row["requested_max_tokens"] = requested_max_tokens
    if mode_label != "ar" and accepted_execution_path == "mimo_mtp_fastpath":
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
        "--prompt",
        args.prompt,
    ]
    requested_max_tokens_values = (
        args.matrix_max_tokens if args.matrix_max_tokens else (args.max_tokens,)
    )
    for requested_max_tokens in requested_max_tokens_values:
        command.extend(["--max-tokens", str(requested_max_tokens)])
    command.extend(
        [
            "--repeats",
            str(args.repeats),
            "--mode-label",
            args.mode_label,
        ]
    )
    if args.list_models:
        command.append("--list-models")
    if args.payload_extra_json is not None:
        command.extend(["--payload-extra-json", args.payload_extra_json])
    if args.output_dir is not None:
        command.extend(["--output-dir", str(args.output_dir)])
    return " ".join(shlex.quote(part) for part in command)


def _blocked_non_fabrication_fields(*, rerun_field: str) -> JsonObject:
    return {
        "live_metrics_status": "unavailable",
        "non_fabrication_statement": (
            "No live benchmark metrics are claimed in this blocked row; run "
            + rerun_field
            + " before using performance, speedup, or Slice 5 evidence."
        ),
    }


def _ar_baseline_blocker(rerun_command: str) -> JsonObject:
    remediation_message = (
        "AR baseline collection is blocked because the exo cluster API is unavailable; "
        "start exo or set --api-base to a running cluster, then rerun: " + rerun_command
    )
    return {
        "kind": "ar_baseline_blocker",
        "status": "blocked_with_command",
        **_blocked_non_fabrication_fields(rerun_field="remediation_command"),
        "remediation_command": rerun_command,
        "remediation_message": remediation_message,
    }


def _blocked_evidence_contract(
    *,
    api_base: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    mode_label: str,
    repeat_index: int | None,
    error: str,
    stage: str,
    rerun_command: str,
    payload_extra: Mapping[str, object] | None = None,
    row_status: str = "blocked",
    mtp_execution_state: str = "blocked_unavailable",
    mtp_disable_reason: str = "cluster_unavailable",
    telemetry_completeness: str = "blocked_no_live_telemetry",
    non_fabrication_statement: str = "No performance result is claimed because the target was unavailable before benchmark execution.",
) -> JsonObject:
    normalized_api_base = api_base.rstrip("/")
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    normalized_payload_extra: JsonObject = (
        {} if payload_extra is None else dict(payload_extra)
    )
    raw_mtp_depth = normalized_payload_extra.get("mimo_mtp_depth")
    mtp_depth = (
        raw_mtp_depth
        if isinstance(raw_mtp_depth, int) and not isinstance(raw_mtp_depth, bool)
        else None
    )
    mtp_enabled = bool(normalized_payload_extra.get("mimo_mtp_fastpath"))
    if mode_label == "ar":
        mtp_enabled = False
        mtp_depth = None
    return {
        "schema_version": "mimo_mtp_benchmark_evidence.v1",
        "evidence_kind": "blocked_evidence",
        "row_status": row_status,
        "benchmark_session_id": f"blocked:{normalized_api_base}:/bench/chat/completions:{model_id}:{mode_label}:{max_tokens}",
        "api_url": normalized_api_base,
        "cluster_id": None,
        "endpoint": "/bench/chat/completions",
        "model_id": model_id,
        "prompt_id": None,
        "prompt_hash": prompt_hash,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "mode": mode_label,
        "repeat_index": repeat_index,
        "generation_tps": None,
        "generation_tokens": None,
        "prompt_tps": None,
        "power_usage": None,
        "payload_extra": normalized_payload_extra,
        "mtp_enabled": mtp_enabled,
        "mtp_depth": mtp_depth,
        "mtp_execution_state": mtp_execution_state,
        "mtp_disable_reason": mtp_disable_reason,
        "telemetry_completeness": telemetry_completeness,
        "runnable_command": rerun_command,
        "expected_output_path": None,
        "blocker_reason": error,
        "blocked_stage": stage,
        "non_fabrication_statement": non_fabrication_statement,
    }


def _cluster_error_row(
    *,
    api_base: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    mode_label: str,
    repeat_index: int | None,
    error: str,
    stage: str,
    success_row_count: int,
    rerun_command: str,
    blocked_reason_kind: str | None = None,
    benchmark_execution_attempted: bool | None = None,
) -> JsonObject:
    row: JsonObject = {
        "kind": "cluster_benchmark_error",
        "cluster_path": "exo_api_bench_chat_completions",
        "api_base": api_base,
        "mode": mode_label,
        "repeat_index": repeat_index,
        "stage": stage,
        "error": error,
        "status": "blocked_with_command",
        "success_row_count": success_row_count,
        **_blocked_non_fabrication_fields(rerun_field="rerun_command"),
        "blocked_evidence": _blocked_evidence_contract(
            api_base=api_base,
            model_id=model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            mode_label=mode_label,
            repeat_index=repeat_index,
            error=error,
            stage=stage,
            rerun_command=rerun_command,
        ),
        "rerun_command": rerun_command,
        "next_step": "start or point to an exo cluster API that can serve /bench/chat/completions, then rerun the command in rerun_command",
    }
    if blocked_reason_kind is not None:
        row["blocked_reason_kind"] = blocked_reason_kind
    if benchmark_execution_attempted is not None:
        row["benchmark_execution_attempted"] = benchmark_execution_attempted
        row["benchmark_endpoint"] = "/bench/chat/completions"
    if mode_label == "ar":
        row["ar_baseline_blocker"] = _ar_baseline_blocker(rerun_command)
    return row


MTP_GUARD_VALIDATION_ERROR: Final[str] = (
    "MTP benchmark mode requires guarded payload fields: "
    "mimo_mtp_fastpath=true, mimo_mtp_fail_closed=true, and "
    "mimo_mtp_depth matching --mode-label depth"
)


def _requested_mtp_depth_from_mode(mode_label: str) -> int | None:
    if not mode_label.startswith("mtp-d"):
        return None
    try:
        return int(mode_label.removeprefix("mtp-d"))
    except ValueError:
        return None


def _is_guarded_mtp_payload(
    *, mode_label: str, payload_extra: Mapping[str, object]
) -> bool:
    requested_depth = _requested_mtp_depth_from_mode(mode_label)
    return (
        requested_depth is not None
        and payload_extra.get("mimo_mtp_fastpath") is True
        and payload_extra.get("mimo_mtp_fail_closed") is True
        and payload_extra.get("mimo_mtp_depth") == requested_depth
    )


def _mtp_guard_error_row(
    *, args: ClusterBenchmarkArgs, payload_extra: Mapping[str, object]
) -> JsonObject:
    rerun_command = _render_cluster_benchmark_rerun_command(args)
    return {
        "kind": "cluster_benchmark_error",
        "cluster_path": "exo_api_bench_chat_completions",
        "api_base": args.api_base,
        "mode": args.mode_label,
        "repeat_index": None,
        "stage": "mtp_guard_validation",
        "error": MTP_GUARD_VALIDATION_ERROR,
        "status": "fail_closed",
        "success_row_count": 0,
        **_blocked_non_fabrication_fields(rerun_field="rerun_command"),
        "blocked_evidence": _blocked_evidence_contract(
            api_base=args.api_base,
            model_id=args.model_id,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            mode_label=args.mode_label,
            repeat_index=None,
            error=MTP_GUARD_VALIDATION_ERROR,
            stage="mtp_guard_validation",
            rerun_command=rerun_command,
            payload_extra=payload_extra,
            row_status="fail_closed",
            mtp_execution_state="fail_closed_error",
            mtp_disable_reason=MTP_GUARD_VALIDATION_ERROR,
            telemetry_completeness="fail_closed_no_live_execution",
            non_fabrication_statement=(
                "No performance result is claimed because guarded MTP request "
                "validation failed before live benchmark execution."
            ),
        ),
        "rerun_command": rerun_command,
        "next_step": (
            "rerun with guarded MTP payload fields or use --mode-label ar for "
            "the default AR benchmark path"
        ),
        "blocked_reason_kind": "unguarded_mtp_benchmark_request",
        "benchmark_execution_attempted": False,
        "benchmark_endpoint": "/bench/chat/completions",
    }


def _http_error_message(*, status_code: int, response: Mapping[str, object]) -> str:
    status_suffix = ""
    raw_status = response.get("status")
    if isinstance(raw_status, str) and raw_status:
        status_suffix = f" {raw_status}"

    raw_detail = response.get("detail")
    if isinstance(raw_detail, str) and raw_detail:
        return f"HTTP {status_code}{status_suffix}: {raw_detail}"

    raw_error = response.get("error")
    if isinstance(raw_error, str) and raw_error:
        return f"HTTP {status_code}{status_suffix}: {raw_error}"
    if isinstance(raw_error, dict):
        raw_message = cast(Mapping[str, object], raw_error).get("message")
        if isinstance(raw_message, str) and raw_message:
            return f"HTTP {status_code}{status_suffix}: {raw_message}"

    return f"HTTP {status_code}{status_suffix}: benchmark request did not complete successfully"


def preflight_cluster_for_ar_baseline(
    *,
    api_base: str,
    timeout_seconds: float,
    http_get: HttpJsonGet,
    list_models: bool,
) -> ClusterAvailabilityPreflightResult:
    """Determine whether the exo cluster API is available before AR collection."""
    normalized_api_base = api_base.rstrip("/")
    probe_url = f"{normalized_api_base}/v1/models"
    stage = "models_probe" if list_models else "cluster_availability_probe"
    try:
        status_code, response = http_get(probe_url, timeout_seconds)
    except (OSError, urllib.error.URLError, TimeoutError, ValueError) as exc:
        return ClusterAvailabilityPreflightResult(
            available=False,
            api_base=normalized_api_base,
            probe_url=probe_url,
            stage=stage,
            status_code=None,
            response=None,
            error=str(exc),
        )
    if status_code < 200 or status_code >= 300:
        return ClusterAvailabilityPreflightResult(
            available=False,
            api_base=normalized_api_base,
            probe_url=probe_url,
            stage=stage,
            status_code=status_code,
            response=response,
            error=_http_error_message(status_code=status_code, response=response),
        )
    return ClusterAvailabilityPreflightResult(
        available=True,
        api_base=normalized_api_base,
        probe_url=probe_url,
        stage=stage,
        status_code=status_code,
        response=response,
        error=None,
    )


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


def _is_mtp_labeled_metric_row(row: Mapping[str, object]) -> bool:
    return _is_cluster_metric_row(row) and row.get("mode") != "ar"


def _comparison_value(row: Mapping[str, object], *keys: str) -> object | None:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def _has_canonical_comparison_metadata(row: Mapping[str, object]) -> bool:
    return any(
        key in row
        for key in (
            "api_url",
            "cluster_id",
            "endpoint",
            "prompt_id",
            "prompt_hash",
            "temperature",
            "max_tokens",
            "benchmark_session_id",
        )
    )


def _is_same_cluster_budget_pair(
    *, ar_row: Mapping[str, object], mtp_row: Mapping[str, object]
) -> bool:
    if not _has_canonical_comparison_metadata(
        ar_row
    ) and not _has_canonical_comparison_metadata(mtp_row):
        return True

    ar_cluster = _comparison_value(ar_row, "api_url", "cluster_id")
    mtp_cluster = _comparison_value(mtp_row, "api_url", "cluster_id")
    ar_prompt = _comparison_value(ar_row, "prompt_id", "prompt_hash")
    mtp_prompt = _comparison_value(mtp_row, "prompt_id", "prompt_hash")
    required_pairs = (
        (ar_cluster, mtp_cluster),
        (
            _comparison_value(ar_row, "model_id", "model"),
            _comparison_value(mtp_row, "model_id", "model"),
        ),
        (_comparison_value(ar_row, "endpoint"), _comparison_value(mtp_row, "endpoint")),
        (ar_prompt, mtp_prompt),
        (
            _comparison_value(ar_row, "temperature"),
            _comparison_value(mtp_row, "temperature"),
        ),
        (
            _comparison_value(ar_row, "max_tokens", "requested_max_tokens"),
            _comparison_value(mtp_row, "max_tokens", "requested_max_tokens"),
        ),
        (
            _comparison_value(ar_row, "benchmark_session_id"),
            _comparison_value(mtp_row, "benchmark_session_id"),
        ),
    )
    return all(left is not None and left == right for left, right in required_pairs)


def _live_mtp_metric_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    return [
        row
        for row in rows
        if _is_cluster_metric_row(row) and _is_live_mtp_metric_row(row)
    ]


def _comparable_live_mtp_rows(
    rows: Sequence[Mapping[str, object]], ar_rows: Sequence[Mapping[str, object]]
) -> list[Mapping[str, object]]:
    live_mtp_rows = _live_mtp_metric_rows(rows)
    if not ar_rows:
        return live_mtp_rows
    return [
        mtp_row
        for mtp_row in live_mtp_rows
        if any(
            _is_same_cluster_budget_pair(ar_row=ar_row, mtp_row=mtp_row)
            for ar_row in ar_rows
        )
    ]


def _has_incomparable_live_mtp_rows(
    *, rows: Sequence[Mapping[str, object]], ar_rows: Sequence[Mapping[str, object]]
) -> bool:
    return (
        bool(ar_rows)
        and bool(_live_mtp_metric_rows(rows))
        and not bool(_comparable_live_mtp_rows(rows, ar_rows))
    )


def _comparison_values_equal(left: object | None, right: object | None) -> bool:
    return left is not None and right is not None and left == right


def _cluster_identity(row: Mapping[str, object]) -> object | None:
    return _comparison_value(row, "api_url", "api_base", "cluster_id")


def _prompt_identity(row: Mapping[str, object]) -> object | None:
    return _comparison_value(row, "prompt_id", "prompt_hash")


def _benchmark_window_identity(row: Mapping[str, object]) -> object | None:
    return _comparison_value(row, "benchmark_session_id", "run_window_id")


def _append_comparison_reason(
    *,
    matched_fields: list[str],
    blocked_reasons: list[str],
    ambiguous_reasons: list[str],
    field_name: str,
    missing_reason: str,
    mismatch_reason: str,
    left: object | None,
    right: object | None,
) -> None:
    if _comparison_values_equal(left, right):
        matched_fields.append(field_name)
    elif left is None or right is None:
        blocked_reasons.append(missing_reason)
    else:
        ambiguous_reasons.append(mismatch_reason)


def classify_same_cluster_comparability(
    *, ar_row: Mapping[str, object], mtp_row: Mapping[str, object]
) -> JsonObject:
    """Classify whether AR and MTP rows are valid same-cluster evidence.

    Comparable rows must share the cluster identity and benchmark contract fields
    needed for honest AR-vs-MTP claims. Missing cluster identity is blocked
    because no same-cluster claim can be evaluated; mismatched populated fields
    are ambiguous because they may represent different clusters, prompts, run
    windows, schemas, or topology.
    """
    matched_fields: list[str] = []
    blocked_reasons: list[str] = []
    ambiguous_reasons: list[str] = []

    if (
        ar_row.get("kind") != "cluster_benchmark_metric"
        or mtp_row.get("kind") != "cluster_benchmark_metric"
    ):
        blocked_reasons.append("missing_cluster_benchmark_metric_rows")
    if (
        ar_row.get("cluster_path") != "exo_api_bench_chat_completions"
        or mtp_row.get("cluster_path") != "exo_api_bench_chat_completions"
    ):
        blocked_reasons.append("missing_bench_chat_completions_cluster_path")
    if ar_row.get("mode") != "ar":
        blocked_reasons.append("missing_ar_row")
    if mtp_row.get("mode") == "ar":
        blocked_reasons.append("missing_mtp_row")

    comparison_contract = (
        (
            "cluster_identity",
            "missing_cluster_identity",
            "cluster_identity_mismatch",
            _cluster_identity(ar_row),
            _cluster_identity(mtp_row),
        ),
        (
            "endpoint",
            "missing_endpoint",
            "endpoint_mismatch",
            _comparison_value(ar_row, "endpoint"),
            _comparison_value(mtp_row, "endpoint"),
        ),
        (
            "model_id",
            "missing_model_id",
            "model_id_mismatch",
            _comparison_value(ar_row, "model_id", "model"),
            _comparison_value(mtp_row, "model_id", "model"),
        ),
        (
            "prompt",
            "missing_prompt_identity",
            "prompt_mismatch",
            _prompt_identity(ar_row),
            _prompt_identity(mtp_row),
        ),
        (
            "temperature",
            "missing_temperature",
            "temperature_mismatch",
            _comparison_value(ar_row, "temperature"),
            _comparison_value(mtp_row, "temperature"),
        ),
        (
            "max_tokens_bucket",
            "missing_max_tokens_bucket",
            "max_tokens_bucket_mismatch",
            _comparison_value(ar_row, "max_tokens", "requested_max_tokens"),
            _comparison_value(mtp_row, "max_tokens", "requested_max_tokens"),
        ),
        (
            "request_schema",
            "missing_request_schema",
            "request_schema_mismatch",
            _comparison_value(ar_row, "request_schema"),
            _comparison_value(mtp_row, "request_schema"),
        ),
        (
            "benchmark_session_or_run_window",
            "missing_benchmark_session_or_run_window",
            "benchmark_session_or_run_window_mismatch",
            _benchmark_window_identity(ar_row),
            _benchmark_window_identity(mtp_row),
        ),
        (
            "repeat_policy",
            "missing_repeat_policy",
            "repeat_policy_mismatch",
            _comparison_value(ar_row, "repeat_policy"),
            _comparison_value(mtp_row, "repeat_policy"),
        ),
        (
            "node_topology",
            "missing_node_topology",
            "node_topology_mismatch",
            _comparison_value(ar_row, "node_topology", "cluster_topology"),
            _comparison_value(mtp_row, "node_topology", "cluster_topology"),
        ),
    )
    for (
        field_name,
        missing_reason,
        mismatch_reason,
        left,
        right,
    ) in comparison_contract:
        _append_comparison_reason(
            matched_fields=matched_fields,
            blocked_reasons=blocked_reasons,
            ambiguous_reasons=ambiguous_reasons,
            field_name=field_name,
            missing_reason=missing_reason,
            mismatch_reason=mismatch_reason,
            left=left,
            right=right,
        )

    if blocked_reasons:
        classification = "blocked"
        next_step = (
            "rerun AR and guarded MTP rows with api_base or cluster_id before making speedup claims"
            if blocked_reasons == ["missing_cluster_identity"]
            else "rerun AR and guarded MTP rows with complete same-cluster comparison metadata before making speedup claims"
        )
    elif ambiguous_reasons:
        classification = "ambiguous"
        next_step = "collect AR and guarded MTP rows from the same cluster, prompt, schema, run window, and topology before making speedup claims"
    else:
        classification = "comparable"
        next_step = "rows are comparable for AR-vs-MTP budget analysis"

    return {
        "kind": "same_cluster_comparability_classification",
        "classification": classification,
        "same_cluster_comparison": classification == "comparable",
        "blocked_reasons": blocked_reasons,
        "ambiguous_reasons": ambiguous_reasons,
        "matched_fields": matched_fields,
        "next_step": next_step,
    }


def _cluster_generation_tps_values(
    rows: Iterable[Mapping[str, object]], *, mode_kind: Literal["ar", "mtp"]
) -> list[float]:
    row_list = list(rows)
    ar_rows = [
        row
        for row in row_list
        if _is_cluster_metric_row(row) and row.get("mode") == "ar"
    ]
    comparable_rows: Sequence[Mapping[str, object]]
    if mode_kind == "ar":
        comparable_rows = ar_rows
    else:
        comparable_rows = _comparable_live_mtp_rows(row_list, ar_rows)

    values: list[float] = []
    for row in comparable_rows:
        generation_tps = _positive_generation_tps(row)
        if generation_tps is not None:
            values.append(generation_tps)
    return values


def _cluster_ar_ms_per_token_values(
    rows: Iterable[Mapping[str, object]],
) -> list[float]:
    values: list[float] = []
    for row in rows:
        if not _is_cluster_metric_row(row) or row.get("mode") != "ar":
            continue
        generation_tps = _positive_generation_tps(row)
        if generation_tps is not None:
            values.append(1000.0 / generation_tps)
    return values


def _matrix_group_key(row: Mapping[str, object]) -> tuple[str, int | None, int | None]:
    raw_mode = row.get("mode")
    mode = raw_mode if isinstance(raw_mode, str) else "unknown"
    raw_depth = row.get("mtp_depth")
    mtp_depth = (
        raw_depth
        if isinstance(raw_depth, int) and not isinstance(raw_depth, bool)
        else None
    )
    raw_max_tokens = row.get("max_tokens")
    max_tokens = (
        raw_max_tokens
        if isinstance(raw_max_tokens, int) and not isinstance(raw_max_tokens, bool)
        else None
    )
    return mode, mtp_depth, max_tokens


def _rounded_optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)


def summarize_benchmark_matrix_rows(rows: Iterable[Mapping[str, object]]) -> JsonObject:
    """Summarize captured same-cluster matrix rows without inventing metrics."""
    groups: dict[tuple[str, int | None, int | None], list[Mapping[str, object]]] = {}
    blocked_commands: list[str] = []
    blocked_row_count = 0

    for row in rows:
        if (
            row.get("kind") == "cluster_benchmark_error"
            and row.get("status") == "blocked_with_command"
        ):
            blocked_row_count += 1
            raw_command = row.get("rerun_command")
            if isinstance(raw_command, str) and raw_command:
                blocked_commands.append(raw_command)
            continue
        if not _is_cluster_metric_row(row):
            continue
        group_key = _matrix_group_key(row)
        groups.setdefault(group_key, []).append(row)

    group_summaries: list[JsonObject] = []
    metric_row_count = 0
    for (mode, mtp_depth, max_tokens), group_rows in groups.items():
        metric_row_count += len(group_rows)
        generation_tps_values = [
            generation_tps
            for group_row in group_rows
            if (generation_tps := _positive_generation_tps(group_row)) is not None
        ]
        acceptance_rate_values = [
            acceptance_rate
            for group_row in group_rows
            if (acceptance_rate := _float_metric(group_row, "acceptance_rate"))
            is not None
        ]
        accepted_mtp_row_count = sum(
            1
            for group_row in group_rows
            if group_row.get("accepted_execution_path") == "mimo_mtp_fastpath"
        )
        group_summaries.append(
            {
                "mode": mode,
                "mtp_depth": mtp_depth,
                "max_tokens": max_tokens,
                "row_count": len(group_rows),
                "median_tok_s": _rounded_optional_float(
                    _median_float(generation_tps_values)
                ),
                "best_tok_s": _rounded_optional_float(
                    max(generation_tps_values) if generation_tps_values else None
                ),
                "accepted_mtp_row_count": accepted_mtp_row_count,
                "acceptance_rate_median": _rounded_optional_float(
                    _median_float(acceptance_rate_values)
                ),
                "acceptance_rate_best": _rounded_optional_float(
                    max(acceptance_rate_values) if acceptance_rate_values else None
                ),
            }
        )

    if blocked_row_count > 0 and metric_row_count > 0:
        status = "blocked_partial_matrix"
        next_step = (
            "rerun blocked commands before making complete AR-vs-MTP performance claims"
        )
    elif blocked_row_count > 0:
        status = "blocked_no_metric_rows"
        next_step = "run the blocked commands against an available guarded cluster before making performance claims"
    elif metric_row_count > 0:
        status = "matrix_summary_ready"
        next_step = "compare same-cluster median/best rows; claim speedup only if live guarded MTP beats AR"
    else:
        status = "blocked_empty_matrix"
        next_step = "collect same-cluster AR and guarded MTP benchmark rows before making performance claims"

    return {
        "kind": "cluster_benchmark_matrix_summary",
        "cluster_path": "exo_api_bench_chat_completions",
        "status": status,
        "metric_row_count": metric_row_count,
        "blocked_row_count": blocked_row_count,
        "groups": group_summaries,
        "blocked_commands": blocked_commands,
        "next_step": next_step,
    }


def calculate_mtp_speedup_budget(rows: Iterable[Mapping[str, object]]) -> JsonObject:
    """Compute same-cluster AR-vs-MTP budget telemetry without fabricating rows."""
    row_list = list(rows)
    ar_generation_tps_values = _cluster_generation_tps_values(row_list, mode_kind="ar")
    ar_ms_per_token_values = _cluster_ar_ms_per_token_values(row_list)
    mtp_generation_tps_values = _cluster_generation_tps_values(
        row_list, mode_kind="mtp"
    )
    mtp_labeled_row_count = sum(
        1 for row in row_list if _is_mtp_labeled_metric_row(row)
    )
    ar_median_tps = _median_float(ar_generation_tps_values)
    mtp_median_tps = _median_float(mtp_generation_tps_values)
    has_incomparable_live_mtp_rows = _has_incomparable_live_mtp_rows(
        rows=row_list,
        ar_rows=[
            row
            for row in row_list
            if _is_cluster_metric_row(row) and row.get("mode") == "ar"
        ],
    )
    if ar_median_tps is not None and mtp_median_tps is not None:
        status = "same_cluster_budget_ready"
        next_step = (
            "use this budget only with same-cluster AR-vs-MTP evidence rows; "
            "Slice 5 still requires a real MTP win without fallback concerns"
        )
    elif ar_median_tps is not None:
        if has_incomparable_live_mtp_rows:
            status = "ambiguous_non_comparable_cluster_rows"
            next_step = "collect AR and guarded MTP rows with matching same-cluster comparison metadata before making speedup or target claims"
        elif mtp_labeled_row_count > 0:
            status = "blocked_missing_live_mtp_rows"
            next_step = "MTP-labeled rows did not execute the guarded fastpath; collect accepted same-cluster MTP rows before making speedup or target claims"
        else:
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
        if not ar_ms_per_token_values
        else round(cast(float, _median_float(ar_ms_per_token_values)), 6),
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


def _row_or_generation_stats_has_key(row: Mapping[str, object], key: str) -> bool:
    if key in row and row[key] is not None:
        return True
    generation_stats = row.get("generation_stats")
    return (
        isinstance(generation_stats, dict)
        and key in generation_stats
        and generation_stats[key] is not None
    )


def _has_benchmark_grade_mtp_telemetry(mtp_row: Mapping[str, object]) -> bool:
    """Return True only when an MTP speedup row has pass-grade telemetry."""
    generation_stats = mtp_row.get("generation_stats")
    stats = (
        cast(Mapping[str, object], generation_stats)
        if isinstance(generation_stats, dict)
        else None
    )
    accepted_execution_path = _optional_string(
        _first_present_value(
            response=mtp_row,
            stats=stats,
            keys=("accepted_execution_path",),
        )
    )
    if accepted_execution_path != "mimo_mtp_fastpath":
        return False
    if mtp_row.get("telemetry_completeness") != "complete":
        return False
    if (
        _optional_bool(
            _first_present_value(response=mtp_row, stats=stats, keys=("mtp_enabled",))
        )
        is not True
    ):
        return False
    if _first_present_value(response=mtp_row, stats=stats, keys=("mtp_depth",)) is None:
        return False
    if (
        _optional_string(
            _first_present_value(
                response=mtp_row,
                stats=stats,
                keys=("mtp_execution_state",),
            )
        )
        != "successful_mtp"
    ):
        return False
    if (
        _optional_string(
            _first_present_value(
                response=mtp_row,
                stats=stats,
                keys=("mtp_sidecar_status",),
            )
        )
        is None
    ):
        return False
    for key in (
        "acceptance_rate",
        "fallback_count",
        "timing_breakdown_seconds",
    ):
        if not _row_or_generation_stats_has_key(mtp_row, key):
            return False
    has_attempted_depth_counts = _row_or_generation_stats_has_key(
        mtp_row, "attempted_depth_counts"
    ) or _row_or_generation_stats_has_key(mtp_row, "mimo_mtp_attempted_depth_counts")
    has_accepted_depth_counts = _row_or_generation_stats_has_key(
        mtp_row, "accepted_depth_counts"
    ) or _row_or_generation_stats_has_key(mtp_row, "mimo_mtp_accepted_depth_counts")
    return has_attempted_depth_counts and has_accepted_depth_counts


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
    bottlenecks.extend(
        mtp_bottleneck_classifier.classify_bottlenecks(
            mtp_row,
            thresholds=mtp_bottleneck_classifier.BottleneckThresholds(
                slow_proposal_tokens_per_second=12.0,
                slow_verifier_tokens_per_second=25.0,
            ),
        )
    )
    return bottlenecks


def classify_mtp_vs_ar_baseline(
    *, ar_row: Mapping[str, object], mtp_row: Mapping[str, object]
) -> JsonObject:
    """Classify one same-cluster AR baseline row against one guarded MTP row."""
    ar_generation_tps = _float_metric(ar_row, "generation_tps")
    mtp_generation_tps = _float_metric(mtp_row, "generation_tps")
    same_cluster_comparison = _is_same_cluster_ar_vs_mtp(ar_row=ar_row, mtp_row=mtp_row)
    raw_accepted_execution_path = mtp_row.get("accepted_execution_path")
    has_positive_mtp_generation_tps = (
        mtp_generation_tps is not None and mtp_generation_tps > 0.0
    )
    has_explicit_non_live_execution_path = (
        has_positive_mtp_generation_tps
        and isinstance(raw_accepted_execution_path, str)
        and raw_accepted_execution_path != ""
        and raw_accepted_execution_path != "mimo_mtp_fastpath"
    )
    if same_cluster_comparison and has_explicit_non_live_execution_path:
        return {
            "kind": "mtp_vs_ar_baseline_classification",
            "classification": "ambiguous",
            "bottlenecks": ["absent_live_mtp_fastpath"],
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
            "reason": "MTP-labeled row did not execute accepted_execution_path=mimo_mtp_fastpath; cannot classify pass or fail",
            "next_step": "collect same-cluster guarded MTP rows with accepted_execution_path=mimo_mtp_fastpath and generation_tps before making speedup or target claims",
        }

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
    if mtp_beats_ar and not _has_benchmark_grade_mtp_telemetry(mtp_row):
        return {
            "kind": "mtp_vs_ar_baseline_classification",
            "classification": "ambiguous",
            "bottlenecks": ["incomplete_mtp_telemetry"],
            "ar_generation_tps": round(ar_generation_tps, 4),
            "ar_ms_per_token": _ms_per_token(ar_generation_tps),
            "mtp_generation_tps": round(mtp_generation_tps, 4),
            "mtp_ms_per_token": _ms_per_token(mtp_generation_tps),
            "speedup_ratio": None,
            "target_gap_tps_30": round(max(0.0, 30.0 - mtp_generation_tps), 4),
            "target_gap_tps_40": round(max(0.0, 40.0 - mtp_generation_tps), 4),
            "same_cluster_comparison": same_cluster_comparison,
            "reason": "MTP row lacks benchmark-grade fastpath telemetry; cannot classify pass",
            "next_step": "rerun guarded MTP benchmark until telemetry_completeness=complete and depth, acceptance, fallback, sidecar, and timing telemetry are present",
        }

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


def _prompt_hash(prompt: str) -> str:
    return "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _benchmark_session_id(args: ClusterBenchmarkArgs) -> str:
    parts = (
        args.api_base.rstrip("/"),
        args.model_id,
        args.mode_label,
        str(args.repeats),
        ",".join(str(value) for value in _requested_max_tokens_values(args)),
    )
    return (
        "cluster-bench-"
        + hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    )


def _canonical_mtp_execution_state(
    *,
    mode_label: str,
    accepted_execution_path: str,
    response: Mapping[str, object],
    stats: Mapping[str, object] | None,
) -> str:
    mtp_enabled = _optional_bool(
        _first_present_value(response=response, stats=stats, keys=("mtp_enabled",))
    )
    mtp_disable_reason = _optional_string(
        _first_present_value(
            response=response, stats=stats, keys=("mtp_disable_reason",)
        )
    )
    mtp_fallback_reason = _optional_string(
        _first_present_value(
            response=response, stats=stats, keys=("mtp_fallback_reason",)
        )
    )
    explicit_state = _optional_string(
        _first_present_value(
            response=response, stats=stats, keys=("mtp_execution_state",)
        )
    )
    if mode_label == "ar":
        return "disabled_default"
    if mtp_fallback_reason is not None:
        return "fail_open_fallback"
    if mtp_disable_reason is not None:
        if mtp_disable_reason == "missing_sidecar":
            return "missing_sidecar"
        if mtp_disable_reason == "unsupported_model":
            return "unsupported_model"
        if "unwired" in mtp_disable_reason:
            return "unwired_execution"
        if "runtime_error" in mtp_disable_reason:
            return "runtime_error"
        if mtp_enabled is False:
            return "fail_closed_error"
    if mtp_enabled is False:
        return "fail_closed_error"
    if explicit_state is not None and explicit_state != "successful_mtp":
        return explicit_state
    if accepted_execution_path == "mimo_mtp_fastpath" and mtp_enabled is not False:
        return "successful_mtp"
    if explicit_state is not None:
        return explicit_state
    return "unwired_execution"


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _cluster_identity_fields_present(row: Mapping[str, object]) -> list[str]:
    return [
        field
        for field in ("api_url", "cluster_id", "prompt_id", "prompt_hash")
        if field in row and row[field] is not None
    ]


def _missing_successful_mtp_telemetry_fields(row: Mapping[str, object]) -> list[str]:
    missing: list[str] = []
    if row.get("accepted_execution_path") != "mimo_mtp_fastpath":
        missing.append("accepted_execution_path=mimo_mtp_fastpath")
    for field in CANONICAL_SUCCESSFUL_MTP_TELEMETRY_FIELDS[1:]:
        if field not in row or row.get(field) is None:
            missing.append(field)
    return missing


def _missing_fail_closed_mtp_telemetry_fields(row: Mapping[str, object]) -> list[str]:
    return [
        field
        for field in CANONICAL_FAIL_CLOSED_MTP_TELEMETRY_FIELDS
        if field not in row or row.get(field) is None
    ]


def _benchmark_telemetry_required_fields_missing(
    row: Mapping[str, object],
) -> list[str]:
    mtp_execution_state = row.get("mtp_execution_state")
    if mtp_execution_state == "successful_mtp":
        return _missing_successful_mtp_telemetry_fields(row)
    if mtp_execution_state in {
        "fail_closed_error",
        "missing_sidecar",
        "unsupported_depth",
        "unsupported_model",
        "unwired_execution",
    }:
        return _missing_fail_closed_mtp_telemetry_fields(row)
    return []


def _row_classification_from_canonical_fields(row: Mapping[str, object]) -> str:
    validation = validate_canonical_benchmark_row(row)
    if not validation["valid"]:
        return "invalid"
    mode = row.get("mode")
    row_status = row.get("row_status")
    mtp_execution_state = row.get("mtp_execution_state")
    if mode == "ar" and mtp_execution_state == "disabled_default":
        return "ar_live" if row_status == "live" else "ar_" + str(row_status)
    if mtp_execution_state == "successful_mtp":
        return "mtp_live_successful"
    if mtp_execution_state == "fail_open_fallback":
        return "mtp_fail_open"
    if mtp_execution_state in {
        "fail_closed_error",
        "missing_sidecar",
        "unsupported_depth",
        "unsupported_model",
        "unwired_execution",
    }:
        return "mtp_fail_closed"
    if mtp_execution_state == "blocked_unavailable":
        return "mtp_blocked" if str(mode).startswith("mtp") else "blocked"
    return "mtp_attempted" if str(mode).startswith("mtp") else "unknown"


def validate_canonical_benchmark_row(row: Mapping[str, object]) -> JsonObject:
    missing_required_fields = [
        field for field in CANONICAL_BENCHMARK_ROW_REQUIRED_FIELDS if field not in row
    ]
    required_one_of_missing = [
        list(group)
        for group in _CANONICAL_BENCHMARK_ROW_REQUIRED_ONE_OF
        if not any(field in row for field in group)
    ]
    invalid_fields: list[JsonObject] = []
    schema_version = row.get("schema_version")
    if (
        "schema_version" in row
        and schema_version != CANONICAL_BENCHMARK_ROW_SCHEMA_VERSION
    ):
        invalid_fields.append(
            {
                "field": "schema_version",
                "reason": f"must equal {CANONICAL_BENCHMARK_ROW_SCHEMA_VERSION}",
                "value": schema_version,
            }
        )
    row_status = row.get("row_status")
    if "row_status" in row and row_status not in CANONICAL_BENCHMARK_ROW_STATUSES:
        invalid_fields.append(
            {
                "field": "row_status",
                "reason": "must be one of: "
                + ", ".join(CANONICAL_BENCHMARK_ROW_STATUSES),
                "value": row_status,
            }
        )
    mtp_execution_state = row.get("mtp_execution_state")
    if (
        "mtp_execution_state" in row
        and mtp_execution_state not in CANONICAL_MTP_EXECUTION_STATES
    ):
        invalid_fields.append(
            {
                "field": "mtp_execution_state",
                "reason": "must be one of: "
                + ", ".join(CANONICAL_MTP_EXECUTION_STATES),
                "value": mtp_execution_state,
            }
        )
    if row.get("row_status") == "live":
        for field in CANONICAL_LIVE_TIMING_THROUGHPUT_FIELDS:
            if field in row and not _is_number(row[field]):
                invalid_fields.append(
                    {
                        "field": field,
                        "reason": "live rows require numeric timing/throughput telemetry",
                        "value": row[field],
                    }
                )
    mode = row.get("mode")
    if mode == "ar":
        if row.get("mtp_enabled") is not False:
            invalid_fields.append(
                {
                    "field": "mtp_enabled",
                    "reason": "AR rows must keep MTP disabled",
                    "value": row.get("mtp_enabled"),
                }
            )
        if mtp_execution_state != "disabled_default":
            invalid_fields.append(
                {
                    "field": "mtp_execution_state",
                    "reason": "AR rows must use disabled_default MTP execution state",
                    "value": mtp_execution_state,
                }
            )
    telemetry_required_fields_missing = _benchmark_telemetry_required_fields_missing(
        row
    )
    if mtp_execution_state == "successful_mtp" and telemetry_required_fields_missing:
        invalid_fields.append(
            {
                "field": "mtp_execution_state",
                "reason": "successful_mtp rows require fastpath attempt telemetry and benchmark-grade MTP telemetry fields",
                "value": mtp_execution_state,
            }
        )
    return {
        "kind": "canonical_benchmark_row_validation",
        "valid": not missing_required_fields
        and not required_one_of_missing
        and not invalid_fields,
        "schema_version": schema_version,
        "row_status": row_status,
        "missing_required_fields": missing_required_fields,
        "invalid_fields": invalid_fields,
        "required_one_of_missing": required_one_of_missing,
    }


def classify_canonical_benchmark_row(row: Mapping[str, object]) -> JsonObject:
    validation = validate_canonical_benchmark_row(row)
    return {
        "kind": "canonical_benchmark_row_classification",
        "valid": validation["valid"],
        "row_classification": _row_classification_from_canonical_fields(row),
        "mode": row.get("mode"),
        "mtp_execution_state": row.get("mtp_execution_state"),
        "telemetry_required_fields_missing": _benchmark_telemetry_required_fields_missing(
            row
        ),
        "cluster_identity_fields_present": _cluster_identity_fields_present(row),
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
            if (
                mapped_classification.get("path") == "exo_cluster_tensor_parallel"
                and mapped_classification.get("is_exo_cluster_tensor_parallel") is True
            ):
                return {
                    "kind": "rollout_baseline_validation",
                    "allowed": True,
                    "baseline_source": "exo_cluster_bench_chat_completions",
                    "reason": "cluster AR row is eligible as a MiMo rollout baseline",
                    "required_source": required_source,
                    "next_step": "compare only with same-cluster guarded MTP rows",
                }
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
            "allowed": False,
            "baseline_source": "unknown_model_path",
            "reason": "cluster AR row is missing tensor-parallel execution-path evidence",
            "required_source": required_source,
            "next_step": "collect an AR row whose response.execution_path reports Tensor sharding with world_size > 1",
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


def _requested_max_tokens_values(args: ClusterBenchmarkArgs) -> tuple[int, ...]:
    if args.matrix_max_tokens:
        return args.matrix_max_tokens
    return (args.max_tokens,)


def _persist_ar_baseline_metric_row(
    *, output_dir: Path | None, max_tokens: int, row: JsonObject
) -> None:
    if output_dir is None or row.get("mode") != "ar":
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"ar-baseline-max-tokens-{max_tokens}.jsonl"
    with output_file.open("a", encoding="utf-8") as file:
        file.write(render_json_line(row))
        file.write("\n")


def collect_ar_baseline_rows(
    *,
    api_base: str,
    model_id: str,
    prompt: str,
    repeats: int,
    timeout_seconds: float,
    output_dir: Path | None,
    http_get: HttpJsonGet,
    http_post: HttpJsonPost,
    timer: Callable[[], float] = time.perf_counter,
) -> list[JsonObject]:
    """Collect and persist required same-cluster AR baseline rows.

    This unit intentionally keeps the request on the default AR path by omitting
    experimental MTP payload extras and forcing the required 16/64 token budgets.
    If the cluster is unavailable, the delegated benchmark runner returns a
    blocked row with an exact rerun command rather than fabricating metrics.
    """
    rows = run_cluster_benchmark(
        args=ClusterBenchmarkArgs(
            api_base=api_base.rstrip("/"),
            model_id=model_id,
            prompt=prompt,
            max_tokens=AR_BASELINE_MAX_TOKENS[-1],
            matrix_max_tokens=AR_BASELINE_MAX_TOKENS,
            repeats=repeats,
            mode_label="ar",
            timeout_seconds=timeout_seconds,
            payload_extra_json=None,
            list_models=True,
            matrix_commands=False,
            mimo_mtp_sidecar_path=None,
            output_dir=output_dir,
        ),
        http_get=http_get,
        http_post=http_post,
        timer=timer,
    )
    metric_rows = [row for row in rows if row.get("kind") == "cluster_benchmark_metric"]
    error_rows = [row for row in rows if row.get("kind") == "cluster_benchmark_error"]
    if not metric_rows and error_rows:
        raw_error = error_rows[0].get("error")
        error = raw_error if isinstance(raw_error, str) else "cluster unavailable"
        return [
            build_ar_baseline_blocked_status(
                api_base=api_base,
                model_id=model_id,
                reason="cluster availability preflight failed: " + error,
            )
        ]
    return rows


def run_cluster_benchmark(
    *,
    args: ClusterBenchmarkArgs,
    http_post: HttpJsonPost,
    http_get: HttpJsonGet,
    timer: Callable[[], float] = time.perf_counter,
) -> list[JsonObject]:
    rows: list[JsonObject] = []
    rerun_command = _render_cluster_benchmark_rerun_command(args)
    api_base = args.api_base.rstrip("/")
    try:
        payload_extra = _payload_extra(args.payload_extra_json)
    except (json.JSONDecodeError, ValueError) as exc:
        return [
            _cluster_error_row(
                api_base=args.api_base,
                model_id=args.model_id,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                mode_label=args.mode_label,
                repeat_index=None,
                stage="payload_validation",
                error=str(exc),
                success_row_count=0,
                rerun_command=rerun_command,
            )
        ]
    if args.mode_label != "ar" and not _is_guarded_mtp_payload(
        mode_label=args.mode_label, payload_extra=payload_extra
    ):
        return [_mtp_guard_error_row(args=args, payload_extra=payload_extra)]

    preflight = preflight_cluster_for_ar_baseline(
        api_base=args.api_base,
        timeout_seconds=args.timeout_seconds,
        http_get=http_get,
        list_models=args.list_models,
    )
    if not preflight.available:
        rows.append(
            _cluster_error_row(
                api_base=args.api_base,
                model_id=args.model_id,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                mode_label=args.mode_label,
                repeat_index=None,
                stage=preflight.stage,
                error=preflight.error or "cluster availability preflight failed",
                success_row_count=sum(
                    1 for row in rows if row["kind"] == "cluster_benchmark_metric"
                ),
                rerun_command=rerun_command,
                blocked_reason_kind="unavailable_cluster_api_target",
                benchmark_execution_attempted=False,
            )
        )
        return rows
    if (
        args.list_models
        and preflight.status_code is not None
        and preflight.response is not None
    ):
        rows.append(
            _models_probe_row(
                api_base=args.api_base,
                status_code=preflight.status_code,
                response=preflight.response,
            )
        )

    payload_extra_keys = sorted(payload_extra.keys())
    for requested_max_tokens in _requested_max_tokens_values(args):
        payload = build_cluster_chat_payload(
            model_id=args.model_id,
            prompt=args.prompt,
            max_tokens=requested_max_tokens,
            payload_extra=payload_extra,
        )
        for repeat_index in range(args.repeats):
            request_payload = dict(payload)
            request_payload["benchmark_mode_label"] = args.mode_label
            request_payload["benchmark_repeat_index"] = repeat_index
            start = timer()
            try:
                status_code, response = http_post(
                    f"{api_base}/bench/chat/completions",
                    request_payload,
                    args.timeout_seconds,
                )
            except (OSError, urllib.error.URLError, TimeoutError, ValueError) as exc:
                rows.append(
                    _cluster_error_row(
                        api_base=args.api_base,
                        model_id=args.model_id,
                        prompt=args.prompt,
                        max_tokens=requested_max_tokens,
                        mode_label=args.mode_label,
                        repeat_index=repeat_index,
                        stage="bench_chat_completions",
                        error=str(exc),
                        success_row_count=sum(
                            1
                            for row in rows
                            if row["kind"] == "cluster_benchmark_metric"
                        ),
                        rerun_command=rerun_command,
                    )
                )
                continue
            if status_code < 200 or status_code >= 300:
                rows.append(
                    _cluster_error_row(
                        api_base=args.api_base,
                        model_id=args.model_id,
                        prompt=args.prompt,
                        max_tokens=requested_max_tokens,
                        mode_label=args.mode_label,
                        repeat_index=repeat_index,
                        stage="bench_chat_completions",
                        error=_http_error_message(
                            status_code=status_code, response=response
                        ),
                        success_row_count=sum(
                            1
                            for row in rows
                            if row["kind"] == "cluster_benchmark_metric"
                        ),
                        rerun_command=rerun_command,
                    )
                )
                continue
            metric_row = build_cluster_metric_row(
                api_base=args.api_base,
                model_id=args.model_id,
                mode_label=args.mode_label,
                repeat_index=repeat_index,
                elapsed_seconds=max(0.0, timer() - start),
                status_code=status_code,
                response=response,
                payload_extra_keys=payload_extra_keys,
                payload_extra=payload_extra,
                requested_max_tokens=requested_max_tokens,
                benchmark_session_id=_benchmark_session_id(args),
                prompt_hash=_prompt_hash(args.prompt),
            )
            rows.append(metric_row)
            _persist_ar_baseline_metric_row(
                output_dir=args.output_dir,
                max_tokens=requested_max_tokens,
                row=metric_row,
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.matrix_commands:
        rows = render_benchmark_matrix_command_rows(
            api_base=args.api_base,
            model_id=args.model_id,
            prompt=args.prompt,
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
