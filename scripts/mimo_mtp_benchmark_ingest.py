from __future__ import annotations

import json
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

try:
    from scripts import mimo_mtp_bottleneck_classifier
except ModuleNotFoundError:
    import mimo_mtp_bottleneck_classifier

JsonObject = dict[str, object]
TelemetryMode = Literal["ar", "mtp"]
ParsedTelemetryDatasetKind = Literal["ar_only", "ar_plus_mtp"]
BottleneckClassification = mimo_mtp_bottleneck_classifier.BottleneckLabel
BlockedRowStatus = Literal["blocked", "blocked_with_command", "blocked_unavailable"]
MTPExecutionState = Literal[
    "disabled_default",
    "enabled_intent",
    "compatible_attempted",
    "successful_mtp",
    "fail_closed_error",
    "fail_open_fallback",
    "unsupported_model",
    "unsupported_depth",
    "missing_sidecar",
    "unwired_execution",
    "runtime_error",
    "blocked_unavailable",
]
MTP_TARGET_TOKENS_PER_SECOND: Final[float] = 30.0
MTP_PREFERRED_TOKENS_PER_SECOND: Final[float] = 40.0
_ALLOWED_MTP_EXECUTION_STATES: Final[frozenset[str]] = frozenset(
    {
        "disabled_default",
        "enabled_intent",
        "compatible_attempted",
        "successful_mtp",
        "fail_closed_error",
        "fail_open_fallback",
        "unsupported_model",
        "unsupported_depth",
        "missing_sidecar",
        "unwired_execution",
        "runtime_error",
        "blocked_unavailable",
    }
)
_CANONICAL_CLUSTER_METRIC_ROW_KEYS: Final[frozenset[str]] = frozenset(
    {
        "accepted_execution_path",
        "acceptance_rate",
        "api_base",
        "cluster_path",
        "elapsed_seconds",
        "generation_stats",
        "generation_tokens",
        "generation_tps",
        "http_status",
        "kind",
        "live_execution_path",
        "max_tokens",
        "mode",
        "mode_label",
        "model",
        "model_id",
        "payload_extra",
        "payload_extra_canonical_collision_keys",
        "payload_extra_keys",
        "power_usage",
        "prompt_tps",
        "repeat_index",
        "requested_max_tokens",
    }
)


@dataclass(frozen=True, slots=True)
class BottleneckClassifierConfig:
    low_acceptance_rate_threshold: float = 0.30
    slow_proposal_tokens_per_second: float | None = None
    slow_verifier_tokens_per_second: float | None = None
    high_fallback_rate_threshold: float | None = None
    aggressive_rollout_depth_threshold: int | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkIngestionError:
    line_number: int
    reason: str
    raw_line: str


@dataclass(frozen=True, slots=True)
class IngestedBlockedBenchmarkRow:
    line_number: int
    evidence_kind: str
    row_status: str
    mtp_disable_reason: str
    commands: tuple[str, ...]
    environment_assumptions: tuple[str, ...]
    output_paths: tuple[str, ...]
    non_fabrication_statement: str
    raw: JsonObject


@dataclass(frozen=True, slots=True)
class IngestedBenchmarkRow:
    line_number: int
    telemetry_mode: TelemetryMode
    is_live_mtp: bool
    kind: str
    cluster_path: str
    mode: str
    mode_label: str
    model: str
    model_id: str
    repeat_index: int
    generation_tps: float
    generation_tokens: int
    accepted_execution_path: str | None
    live_execution_path: str | None
    optional_prompt_tps: float | None
    optional_power_usage: object | None
    optional_acceptance_rate: float | None
    bottleneck_classifications: list[BottleneckClassification]
    payload_extra_keys: tuple[str, ...]
    payload_extra: JsonObject
    payload_extra_canonical_collision_keys: tuple[str, ...]
    raw: JsonObject


@dataclass(frozen=True, slots=True)
class ParsedTelemetryDataset:
    kind: ParsedTelemetryDatasetKind
    ar_rows: tuple[IngestedBenchmarkRow, ...]
    mtp_rows: tuple[IngestedBenchmarkRow, ...]
    live_mtp_rows: tuple[IngestedBenchmarkRow, ...]
    errors: tuple[BenchmarkIngestionError, ...]
    summary: JsonObject
    speedup_budget: JsonObject

    @property
    def has_live_mtp_rows(self) -> bool:
        return len(self.live_mtp_rows) > 0


@dataclass(frozen=True, slots=True)
class BenchmarkIngestionResult:
    rows: list[IngestedBenchmarkRow]
    blocked_rows: list[IngestedBlockedBenchmarkRow]
    errors: list[BenchmarkIngestionError]
    summary: JsonObject

    @property
    def ar_rows(self) -> list[IngestedBenchmarkRow]:
        return [row for row in self.rows if row.telemetry_mode == "ar"]

    @property
    def mtp_rows(self) -> list[IngestedBenchmarkRow]:
        return [row for row in self.rows if row.telemetry_mode == "mtp"]

    @property
    def live_mtp_rows(self) -> list[IngestedBenchmarkRow]:
        return [row for row in self.rows if row.is_live_mtp]


def ingest_benchmark_jsonl_file(
    path: str | Path,
    *,
    classifier_config: BottleneckClassifierConfig | None = None,
) -> BenchmarkIngestionResult:
    return ingest_benchmark_jsonl(
        Path(path).read_text(encoding="utf-8"),
        classifier_config=classifier_config,
    )


def ingest_analyzer_benchmark_jsonl_file(path: str | Path) -> BenchmarkIngestionResult:
    return ingest_analyzer_benchmark_jsonl(Path(path).read_text(encoding="utf-8"))


def ingest_analyzer_benchmark_jsonl(content: str) -> BenchmarkIngestionResult:
    """Parse canonical benchmark JSONL into analyzer rows only.

    This ingestion path validates line-oriented JSONL and canonical row contracts,
    then normalizes rows into IngestedBenchmarkRow objects for analyzers. It does
    not calculate speedup budgets, same-cluster sufficiency, or Slice 5 decisions.
    """
    rows: list[IngestedBenchmarkRow] = []
    errors: list[BenchmarkIngestionError] = []
    for index, line in enumerate(content.splitlines(), start=1):
        stripped_line = line.strip()
        if not stripped_line:
            continue
        try:
            decoded = cast(object, json.loads(stripped_line))
        except json.JSONDecodeError:
            errors.append(_error(index, line, "line is not valid JSON"))
            continue
        if not isinstance(decoded, dict):
            errors.append(_error(index, line, "line must decode to a JSON object"))
            continue
        row = cast(JsonObject, decoded)
        if row.get("evidence_kind") == "blocked_benchmark_row":
            blocked = _parse_blocked_benchmark_row(
                line_number=index,
                row=row,
                raw_line=line,
            )
            if isinstance(blocked, BenchmarkIngestionError):
                errors.append(blocked)
            continue
        parsed = _parse_analyzer_benchmark_row(
            line_number=index,
            row=row,
            raw_line=line,
        )
        if isinstance(parsed, BenchmarkIngestionError):
            errors.append(parsed)
        else:
            rows.append(parsed)
    return BenchmarkIngestionResult(
        rows=rows,
        blocked_rows=[],
        errors=errors,
        summary={
            "valid_row_count": len(rows),
            "error_count": len(errors),
            "blocked_row_count": 0,
        },
    )


def ingest_benchmark_jsonl(
    content: str,
    *,
    classifier_config: BottleneckClassifierConfig | None = None,
) -> BenchmarkIngestionResult:
    rows: list[IngestedBenchmarkRow] = []
    blocked_rows: list[IngestedBlockedBenchmarkRow] = []
    errors: list[BenchmarkIngestionError] = []
    for index, line in enumerate(content.splitlines(), start=1):
        stripped_line = line.strip()
        if not stripped_line:
            continue
        try:
            decoded = cast(object, json.loads(stripped_line))
        except json.JSONDecodeError:
            errors.append(
                BenchmarkIngestionError(
                    line_number=index,
                    reason="line is not valid JSON",
                    raw_line=line,
                )
            )
            continue
        if not isinstance(decoded, dict):
            errors.append(
                BenchmarkIngestionError(
                    line_number=index,
                    reason="line must decode to a JSON object",
                    raw_line=line,
                )
            )
            continue
        row = cast(JsonObject, decoded)
        evidence_kind = row.get("evidence_kind")
        if evidence_kind == "blocked_benchmark_row":
            blocked = _parse_blocked_benchmark_row(
                line_number=index,
                row=row,
                raw_line=line,
            )
            if isinstance(blocked, BenchmarkIngestionError):
                errors.append(blocked)
            else:
                blocked_rows.append(blocked)
            continue
        if evidence_kind == "benchmark_row":
            parsed = _parse_analyzer_benchmark_row(
                line_number=index,
                row=row,
                raw_line=line,
            )
        else:
            parsed = _parse_benchmark_row(
                line_number=index,
                row=row,
                raw_line=line,
                classifier_config=classifier_config,
            )
        if isinstance(parsed, BenchmarkIngestionError):
            errors.append(parsed)
        else:
            rows.append(parsed)
    return BenchmarkIngestionResult(
        rows=rows,
        blocked_rows=blocked_rows,
        errors=errors,
        summary=_build_summary(
            rows=rows,
            blocked_rows=blocked_rows,
            error_count=len(errors),
            include_bottleneck_classifications=classifier_config is not None,
        ),
    )


def parse_benchmark_telemetry_dataset_file(
    path: str | Path,
    *,
    classifier_config: BottleneckClassifierConfig | None = None,
) -> ParsedTelemetryDataset:
    return parse_benchmark_telemetry_dataset_jsonl(
        Path(path).read_text(encoding="utf-8"),
        classifier_config=classifier_config,
    )


def parse_benchmark_telemetry_dataset_jsonl(
    content: str,
    *,
    classifier_config: BottleneckClassifierConfig | None = None,
) -> ParsedTelemetryDataset:
    ingestion_result = ingest_benchmark_jsonl(
        content,
        classifier_config=classifier_config,
    )
    ar_rows = tuple(ingestion_result.ar_rows)
    live_mtp_rows = tuple(ingestion_result.live_mtp_rows)
    return ParsedTelemetryDataset(
        kind="ar_plus_mtp" if live_mtp_rows else "ar_only",
        ar_rows=ar_rows,
        mtp_rows=live_mtp_rows,
        live_mtp_rows=live_mtp_rows,
        errors=tuple(ingestion_result.errors),
        summary=ingestion_result.summary,
        speedup_budget=calculate_parsed_mtp_speedup_budget(ingestion_result.rows),
    )


def _parsed_comparison_value(row: IngestedBenchmarkRow, key: str) -> object | None:
    value = row.raw.get(key)
    if value is not None:
        return value
    if key == "repeat_policy":
        return "observed_repeat_count=2"
    return None


def _parsed_comparison_key(rows: list[IngestedBenchmarkRow]) -> JsonObject:
    first = rows[0] if rows else None
    return {
        "api_url": None
        if first is None
        else _parsed_comparison_value(first, "api_url"),
        "cluster_id": None
        if first is None
        else _parsed_comparison_value(first, "cluster_id"),
        "endpoint": None
        if first is None
        else _parsed_comparison_value(first, "endpoint"),
        "model_id": None if first is None else first.model_id,
        "prompt_id": None
        if first is None
        else _parsed_comparison_value(first, "prompt_id"),
        "prompt_hash": None
        if first is None
        else _parsed_comparison_value(first, "prompt_hash"),
        "temperature": None
        if first is None
        else _parsed_comparison_value(first, "temperature"),
        "max_tokens": None
        if first is None
        else _parsed_comparison_value(first, "max_tokens"),
        "benchmark_session_id": None
        if first is None
        else _parsed_comparison_value(first, "benchmark_session_id"),
        "repeat_policy": "observed_repeat_count="
        + str(
            max(
                sum(1 for row in rows if row.telemetry_mode == "ar"),
                sum(1 for row in rows if row.telemetry_mode == "mtp"),
            )
        )
        if rows
        else None,
        "node_topology": None
        if first is None
        else _parsed_comparison_value(first, "node_topology"),
    }


def _parsed_comparability(
    rows: list[IngestedBenchmarkRow],
) -> tuple[str, JsonObject | None, list[str]]:
    ar_rows = [row for row in rows if row.telemetry_mode == "ar"]
    mtp_rows = [row for row in rows if row.is_live_mtp]
    if not ar_rows or not mtp_rows:
        return "insufficient_data", None, []

    blockers: list[str] = []
    ar_cluster_identities = _field_values(ar_rows, "api_url") | _field_values(
        ar_rows, "cluster_id"
    )
    mtp_cluster_identities = _field_values(mtp_rows, "api_url") | _field_values(
        mtp_rows, "cluster_id"
    )
    if not ar_cluster_identities or not mtp_cluster_identities:
        blockers.append("api_url_or_cluster_id")
    elif ar_cluster_identities != mtp_cluster_identities:
        if _field_values(ar_rows, "api_url") != _field_values(mtp_rows, "api_url"):
            blockers.append("api_url")
        if _field_values(ar_rows, "cluster_id") != _field_values(
            mtp_rows, "cluster_id"
        ):
            blockers.append("cluster_id")

    ar_prompt_identities = _field_values(ar_rows, "prompt_id") | _field_values(
        ar_rows, "prompt_hash"
    )
    mtp_prompt_identities = _field_values(mtp_rows, "prompt_id") | _field_values(
        mtp_rows, "prompt_hash"
    )
    if not ar_prompt_identities or not mtp_prompt_identities:
        blockers.append("prompt_id_or_prompt_hash")
    elif ar_prompt_identities != mtp_prompt_identities:
        if _field_values(ar_rows, "prompt_id") != _field_values(mtp_rows, "prompt_id"):
            blockers.append("prompt_id")
        if _field_values(ar_rows, "prompt_hash") != _field_values(
            mtp_rows, "prompt_hash"
        ):
            blockers.append("prompt_hash")

    for field in (
        "endpoint",
        "model_id",
        "temperature",
        "max_tokens",
        "benchmark_session_id",
    ):
        ar_values = _field_values(ar_rows, field)
        mtp_values = _field_values(mtp_rows, field)
        if not ar_values or not mtp_values or ar_values != mtp_values:
            blockers.append(field)

    ar_node_topology = _field_values(ar_rows, "node_topology")
    mtp_node_topology = _field_values(mtp_rows, "node_topology")
    if ar_node_topology != mtp_node_topology:
        blockers.append("node_topology")

    if blockers:
        return "ambiguous", None, sorted(set(blockers))
    return "same_cluster_comparable", _parsed_comparison_key(ar_rows + mtp_rows), []


def _field_values(rows: list[IngestedBenchmarkRow], field: str) -> set[object]:
    return {
        value
        for row in rows
        if (value := _parsed_comparison_value(row, field)) is not None
    }


def calculate_parsed_mtp_speedup_budget(
    rows: Iterable[IngestedBenchmarkRow],
) -> JsonObject:
    """Compute same-cluster AR-vs-MTP budget metrics from parsed telemetry rows."""
    row_list = list(rows)
    ar_generation_tps_values = [
        row.generation_tps for row in row_list if row.telemetry_mode == "ar"
    ]
    mtp_generation_tps_values = [
        row.generation_tps for row in row_list if row.is_live_mtp
    ]
    ar_median_tps = _median_float(ar_generation_tps_values)
    mtp_median_tps = _median_float(mtp_generation_tps_values)
    comparability_status, comparison_key, comparability_blockers = (
        _parsed_comparability(row_list)
    )

    if ar_median_tps is not None and mtp_median_tps is not None:
        if comparability_status == "same_cluster_comparable":
            status = "same_cluster_budget_ready"
            next_step = (
                "use this parsed budget only with same-cluster AR-vs-MTP evidence rows; "
                "Slice 5 still requires a real MTP win without fallback concerns"
            )
        else:
            status = "ambiguous_non_comparable_rows"
            next_step = "non-comparable AR/MTP rows are insufficient; no MTP speedup claim is allowed until same-cluster rows match"
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
            "is present on both parsed row sets"
        )

    speedup_ratio = None
    if (
        status == "same_cluster_budget_ready"
        and ar_median_tps is not None
        and mtp_median_tps is not None
    ):
        speedup_ratio = round(mtp_median_tps / ar_median_tps, 6)

    report: JsonObject = {
        "kind": "mimo_mtp_parsed_speedup_budget",
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
        "mtp_vs_ar_speedup_ratio": speedup_ratio,
        "status": status,
        "comparability_status": comparability_status,
        "comparability_blockers": comparability_blockers,
        "next_step": next_step,
    }
    report["same_cluster_comparison_key"] = comparison_key
    return report


def _parse_analyzer_benchmark_row(
    *,
    line_number: int,
    row: JsonObject,
    raw_line: str,
) -> IngestedBenchmarkRow | BenchmarkIngestionError:
    evidence_kind = _required_string(row, "evidence_kind")
    if evidence_kind != "benchmark_row":
        return _error(line_number, raw_line, "evidence_kind must be benchmark_row")
    if _required_string(row, "schema_version") is None:
        return _error(line_number, raw_line, "schema_version must be a string")
    if _required_string(row, "row_status") is None:
        return _error(line_number, raw_line, "row_status must be a string")
    for key in ("benchmark_session_id", "endpoint", "model_id"):
        if _required_string(row, key) is None:
            return _error(line_number, raw_line, key + " must be a string")
    if (
        _required_string(row, "api_url") is None
        and _required_string(row, "cluster_id") is None
    ):
        return _error(line_number, raw_line, "api_url or cluster_id must be a string")
    if (
        _required_string(row, "prompt_id") is None
        and _required_string(row, "prompt_hash") is None
    ):
        return _error(
            line_number, raw_line, "prompt_id or prompt_hash must be a string"
        )
    mode = _required_string(row, "mode")
    if mode is None:
        return _error(line_number, raw_line, "mode must be a string")
    repeat_index = row.get("repeat_index")
    if isinstance(repeat_index, bool) or not isinstance(repeat_index, int):
        return _error(line_number, raw_line, "repeat_index must be an integer")
    generation_tps = _required_metric_float(row, "generation_tps")
    if generation_tps is None:
        return _error(line_number, raw_line, "generation_tps must be a number")
    generation_tokens = _required_metric_int(row, "generation_tokens")
    if generation_tokens is None:
        return _error(line_number, raw_line, "generation_tokens must be an integer")
    prompt_tps = _optional_metric_float(row, "prompt_tps")
    if isinstance(prompt_tps, str):
        return _error(line_number, raw_line, "prompt_tps must be a number when present")
    mtp_execution_state = _required_string(row, "mtp_execution_state")
    if mtp_execution_state not in _ALLOWED_MTP_EXECUTION_STATES:
        return _error(
            line_number,
            raw_line,
            "mtp_execution_state must be a recognized MTP execution state",
        )
    mtp_enabled = row.get("mtp_enabled")
    if not isinstance(mtp_enabled, bool):
        return _error(line_number, raw_line, "mtp_enabled must be a boolean")
    max_tokens = _required_metric_int(row, "max_tokens")
    if max_tokens is None:
        return _error(line_number, raw_line, "max_tokens must be an integer")
    temperature = _required_metric_float(row, "temperature")
    if temperature is None:
        return _error(line_number, raw_line, "temperature must be a number")
    payload_extra = _payload_extra(row.get("payload_extra"))
    if payload_extra is None:
        return _error(
            line_number, raw_line, "payload_extra must be an object when present"
        )
    raw = dict(row)
    raw["model"] = cast(str, row["model_id"])
    raw["model_id"] = cast(str, row["model_id"])
    raw["mode"] = mode
    raw["mode_label"] = mode
    raw["temperature"] = temperature
    raw["max_tokens"] = max_tokens
    raw["generation_tps"] = generation_tps
    raw["generation_tokens"] = generation_tokens
    raw["payload_extra"] = payload_extra
    if prompt_tps is not None:
        raw["prompt_tps"] = prompt_tps
    power_usage = _normalized_power_usage(row.get("power_usage"))
    if power_usage is not None:
        raw["power_usage"] = power_usage
    accepted_execution_path = _analyzer_accepted_execution_path(
        mode=mode,
        mtp_execution_state=mtp_execution_state,
    )
    telemetry_mode = _telemetry_mode(mode)
    return IngestedBenchmarkRow(
        line_number=line_number,
        telemetry_mode=telemetry_mode,
        is_live_mtp=_is_live_mtp(
            telemetry_mode=telemetry_mode,
            accepted_execution_path=accepted_execution_path,
        ),
        kind="cluster_benchmark_metric",
        cluster_path="exo_api_bench_chat_completions",
        mode=mode,
        mode_label=mode,
        model=cast(str, row["model_id"]),
        model_id=cast(str, row["model_id"]),
        repeat_index=repeat_index,
        generation_tps=generation_tps,
        generation_tokens=generation_tokens,
        accepted_execution_path=accepted_execution_path,
        live_execution_path=None,
        optional_prompt_tps=prompt_tps,
        optional_power_usage=power_usage,
        optional_acceptance_rate=_acceptance_rate(row),
        bottleneck_classifications=[],
        payload_extra_keys=tuple(sorted(payload_extra)),
        payload_extra=payload_extra,
        payload_extra_canonical_collision_keys=_payload_extra_canonical_collision_keys(
            payload_extra
        ),
        raw=raw,
    )


def _analyzer_accepted_execution_path(*, mode: str, mtp_execution_state: str) -> str:
    if mode == "ar":
        return "ar"
    if mtp_execution_state == "successful_mtp":
        return "mimo_mtp_fastpath"
    if mtp_execution_state == "fail_open_fallback":
        return "ar_fallback"
    return mtp_execution_state


def _parse_blocked_benchmark_row(
    *,
    line_number: int,
    row: JsonObject,
    raw_line: str,
) -> IngestedBlockedBenchmarkRow | BenchmarkIngestionError:
    row_status = _required_string(row, "row_status")
    if row_status is None:
        return _error(
            line_number, raw_line, "row_status must be a string for blocked rows"
        )
    mtp_disable_reason = _required_string(row, "mtp_disable_reason")
    if mtp_disable_reason is None:
        return _error(
            line_number,
            raw_line,
            "mtp_disable_reason must be a string for blocked rows",
        )
    commands = _required_non_empty_string_list(row.get("commands"))
    if commands is None:
        return _error(
            line_number,
            raw_line,
            "commands must be a non-empty string list for blocked rows",
        )
    environment_assumptions = _required_non_empty_string_list(
        row.get("environment_assumptions")
    )
    if environment_assumptions is None:
        return _error(
            line_number,
            raw_line,
            "environment_assumptions must be a non-empty string list for blocked rows",
        )
    output_paths = _required_non_empty_string_list(row.get("output_paths"))
    if output_paths is None:
        return _error(
            line_number,
            raw_line,
            "output_paths must be a non-empty string list for blocked rows",
        )
    non_fabrication_statement = _required_string(row, "non_fabrication_statement")
    if non_fabrication_statement is None:
        return _error(
            line_number,
            raw_line,
            "non_fabrication_statement must be a string for blocked rows",
        )
    if "no performance result is claimed" not in non_fabrication_statement.lower():
        return _error(
            line_number,
            raw_line,
            "non_fabrication_statement must explicitly state that no performance result is claimed",
        )
    return IngestedBlockedBenchmarkRow(
        line_number=line_number,
        evidence_kind="blocked_benchmark_row",
        row_status=row_status,
        mtp_disable_reason=mtp_disable_reason,
        commands=commands,
        environment_assumptions=environment_assumptions,
        output_paths=output_paths,
        non_fabrication_statement=non_fabrication_statement,
        raw=dict(row),
    )


def _parse_benchmark_row(
    *,
    line_number: int,
    row: JsonObject,
    raw_line: str,
    classifier_config: BottleneckClassifierConfig | None,
) -> IngestedBenchmarkRow | BenchmarkIngestionError:
    kind = _required_string(row, "kind")
    if kind is None:
        return _error(line_number, raw_line, "kind must be a string")
    cluster_path = _required_string(row, "cluster_path")
    if cluster_path is None:
        return _error(line_number, raw_line, "cluster_path must be a string")
    if kind != "cluster_benchmark_metric":
        return _error(line_number, raw_line, "kind must be cluster_benchmark_metric")
    if cluster_path != "exo_api_bench_chat_completions":
        return _error(
            line_number,
            raw_line,
            "cluster_path must be exo_api_bench_chat_completions",
        )
    mode = _canonical_metadata_string(
        line_number=line_number,
        row=row,
        raw_line=raw_line,
        canonical_key="mode_label",
        legacy_key="mode",
    )
    if isinstance(mode, BenchmarkIngestionError):
        return mode
    model = _canonical_metadata_string(
        line_number=line_number,
        row=row,
        raw_line=raw_line,
        canonical_key="model_id",
        legacy_key="model",
    )
    if isinstance(model, BenchmarkIngestionError):
        return model
    repeat_index = row.get("repeat_index")
    if isinstance(repeat_index, bool) or not isinstance(repeat_index, int):
        return _error(line_number, raw_line, "repeat_index must be an integer")
    metric_source = _metric_source(row)
    generation_tps = _required_metric_float(metric_source, "generation_tps")
    if generation_tps is None:
        return _error(line_number, raw_line, "generation_tps must be a number")
    generation_tokens = _required_metric_int(metric_source, "generation_tokens")
    if generation_tokens is None:
        return _error(line_number, raw_line, "generation_tokens must be an integer")
    prompt_tps = _optional_metric_float(metric_source, "prompt_tps")
    if isinstance(prompt_tps, str):
        return _error(line_number, raw_line, "prompt_tps must be a number when present")
    power_usage = _normalized_power_usage(row.get("power_usage"))
    payload_extra_keys = _payload_extra_keys(row.get("payload_extra_keys"))
    if payload_extra_keys is None:
        return _error(
            line_number,
            raw_line,
            "payload_extra_keys must be a string list when present",
        )
    telemetry_mode = _telemetry_mode(mode)
    accepted_execution_path = _optional_string(row, "accepted_execution_path")
    live_execution_path = _optional_string(row, "live_execution_path")
    optional_acceptance_rate = _acceptance_rate(row)
    payload_extra = _payload_extra(row.get("payload_extra"))
    if payload_extra is None:
        return _error(
            line_number, raw_line, "payload_extra must be an object when present"
        )
    payload_extra_canonical_collision_keys = _payload_extra_canonical_collision_keys(
        payload_extra
    )
    return IngestedBenchmarkRow(
        line_number=line_number,
        telemetry_mode=telemetry_mode,
        is_live_mtp=_is_live_mtp(
            telemetry_mode=telemetry_mode,
            accepted_execution_path=accepted_execution_path,
        ),
        kind=kind,
        cluster_path=cluster_path,
        mode=mode,
        mode_label=mode,
        model=model,
        model_id=model,
        repeat_index=repeat_index,
        generation_tps=generation_tps,
        generation_tokens=generation_tokens,
        accepted_execution_path=accepted_execution_path,
        live_execution_path=live_execution_path,
        optional_prompt_tps=prompt_tps,
        optional_power_usage=power_usage,
        optional_acceptance_rate=optional_acceptance_rate,
        bottleneck_classifications=_classify_bottlenecks(
            row=row,
            classifier_config=classifier_config,
        ),
        payload_extra_keys=payload_extra_keys,
        payload_extra=payload_extra,
        payload_extra_canonical_collision_keys=payload_extra_canonical_collision_keys,
        raw=_normalized_raw_row(
            row=row,
            payload_extra_keys=payload_extra_keys,
            mode=mode,
            model=model,
            generation_tps=generation_tps,
            generation_tokens=generation_tokens,
            prompt_tps=prompt_tps,
            power_usage=power_usage,
            payload_extra=payload_extra,
        ),
    )


def _normalized_raw_row(
    *,
    row: Mapping[str, object],
    payload_extra_keys: tuple[str, ...],
    mode: str,
    model: str,
    generation_tps: float,
    generation_tokens: int,
    prompt_tps: float | None,
    power_usage: object | None,
    payload_extra: JsonObject,
) -> JsonObject:
    normalized = dict(row)
    normalized.setdefault("payload_extra_keys", list(payload_extra_keys))
    normalized["mode"] = mode
    normalized["mode_label"] = mode
    normalized["model"] = model
    normalized["model_id"] = model
    normalized["generation_tps"] = generation_tps
    normalized["generation_tokens"] = generation_tokens
    if prompt_tps is not None:
        normalized["prompt_tps"] = prompt_tps
    elif "prompt_tps" in normalized:
        del normalized["prompt_tps"]
    if power_usage is not None:
        normalized["power_usage"] = power_usage
    elif "power_usage" in normalized:
        del normalized["power_usage"]
    if payload_extra:
        normalized["payload_extra"] = payload_extra
    return normalized


def _canonical_metadata_string(
    *,
    line_number: int,
    row: Mapping[str, object],
    raw_line: str,
    canonical_key: str,
    legacy_key: str,
) -> str | BenchmarkIngestionError:
    canonical_present = canonical_key in row
    legacy_present = legacy_key in row
    canonical_value = row.get(canonical_key)
    legacy_value = row.get(legacy_key)
    if canonical_present and not isinstance(canonical_value, str):
        return _error(line_number, raw_line, canonical_key + " must be a string")
    if legacy_present and not isinstance(legacy_value, str):
        return _error(line_number, raw_line, legacy_key + " must be a string")
    if not canonical_present and not legacy_present:
        return _error(line_number, raw_line, canonical_key + " must be a string")
    if (
        canonical_present
        and legacy_present
        and cast(str, canonical_value) != cast(str, legacy_value)
    ):
        return _error(
            line_number,
            raw_line,
            legacy_key + " and " + canonical_key + " must match when both are present",
        )
    if canonical_present:
        return cast(str, canonical_value)
    return cast(str, legacy_value)


def _metric_source(row: Mapping[str, object]) -> Mapping[str, object]:
    generation_stats = row.get("generation_stats")
    nested_stats: Mapping[str, object]
    if isinstance(generation_stats, dict):
        nested_stats = cast(Mapping[str, object], generation_stats)
    else:
        nested_stats = {}
    merged: dict[str, object] = dict(nested_stats)
    for key in ("generation_tps", "generation_tokens", "prompt_tps"):
        if key in row:
            merged[key] = row[key]
    return merged


def _coerced_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _required_metric_float(row: Mapping[str, object], key: str) -> float | None:
    return _coerced_float(row.get(key))


def _optional_metric_float(row: Mapping[str, object], key: str) -> float | str | None:
    if key not in row or row[key] is None:
        return None
    value = _coerced_float(row[key])
    if value is None:
        return "invalid"
    return value


def _required_metric_int(row: Mapping[str, object], key: str) -> int | None:
    value = row.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        return int(parsed) if parsed.is_integer() else None
    return None


def _normalized_power_usage(value: object) -> object | None:
    if value is None:
        return None
    return _coerce_numeric_strings(value)


def _coerce_numeric_strings(value: object) -> object:
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    if isinstance(value, list):
        return [_coerce_numeric_strings(item) for item in cast(list[object], value)]
    if isinstance(value, dict):
        return {
            str(key): _coerce_numeric_strings(item)
            for key, item in cast(Mapping[str, object], value).items()
        }
    return value


def _payload_extra(value: object) -> JsonObject | None:
    if value is None:
        return {}
    if not isinstance(value, dict):
        return None
    return dict(cast(Mapping[str, object], value))


_CANONICAL_PAYLOAD_EXTRA_COLLISION_KEYS: Final[set[str]] = {
    "kind",
    "cluster_path",
    "mode",
    "mode_label",
    "model",
    "model_id",
    "repeat_index",
    "generation_tps",
    "generation_tokens",
    "prompt_tps",
    "power_usage",
}


def _payload_extra_canonical_collision_keys(
    payload_extra: Mapping[str, object],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            key
            for key in payload_extra
            if key in _CANONICAL_PAYLOAD_EXTRA_COLLISION_KEYS
        )
    )


def _error(line_number: int, raw_line: str, reason: str) -> BenchmarkIngestionError:
    return BenchmarkIngestionError(
        line_number=line_number,
        reason=reason,
        raw_line=raw_line,
    )


def _required_string(row: Mapping[str, object], key: str) -> str | None:
    value = row.get(key)
    if isinstance(value, str):
        return value
    return None


def _optional_string(row: Mapping[str, object], key: str) -> str | None:
    value = row.get(key)
    if isinstance(value, str):
        return value
    return None


def _payload_extra_keys(value: object) -> tuple[str, ...] | None:
    if value is None:
        return ()
    if not isinstance(value, list):
        return None
    keys: list[str] = []
    for item in cast(list[object], value):
        if not isinstance(item, str):
            return None
        keys.append(item)
    return tuple(keys)


def _required_non_empty_string_list(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, list) or not value:
        return None
    items: list[str] = []
    for item in cast(list[object], value):
        if not isinstance(item, str) or not item:
            return None
        items.append(item)
    return tuple(items)


def _telemetry_mode(mode: str) -> TelemetryMode:
    if mode == "ar":
        return "ar"
    return "mtp"


def _is_live_mtp(
    *, telemetry_mode: TelemetryMode, accepted_execution_path: str | None
) -> bool:
    return telemetry_mode == "mtp" and accepted_execution_path == "mimo_mtp_fastpath"


def _acceptance_rate(row: Mapping[str, object]) -> float | None:
    explicit_acceptance_rate = _optional_number(row, "acceptance_rate")
    if explicit_acceptance_rate is not None:
        return explicit_acceptance_rate

    generation_stats = row.get("generation_stats")
    if not isinstance(generation_stats, dict):
        return None
    mapped_generation_stats = cast(Mapping[str, object], generation_stats)
    nested_acceptance_rate = _optional_number(
        mapped_generation_stats, "acceptance_rate"
    )
    if nested_acceptance_rate is not None:
        return nested_acceptance_rate
    attempted_tokens = _optional_number(
        mapped_generation_stats, "mimo_mtp_attempted_tokens"
    )
    accepted_tokens = _optional_number(
        mapped_generation_stats, "mimo_mtp_accepted_tokens"
    )
    if attempted_tokens is None or accepted_tokens is None or attempted_tokens <= 0.0:
        return None
    return accepted_tokens / attempted_tokens


def _optional_number(row: Mapping[str, object], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _classify_bottlenecks(
    *,
    row: Mapping[str, object],
    classifier_config: BottleneckClassifierConfig | None,
) -> list[BottleneckClassification]:
    if classifier_config is None:
        return []
    return mimo_mtp_bottleneck_classifier.classify_bottlenecks(
        row,
        thresholds=mimo_mtp_bottleneck_classifier.BottleneckThresholds(
            low_acceptance_rate=classifier_config.low_acceptance_rate_threshold,
            slow_proposal_tokens_per_second=classifier_config.slow_proposal_tokens_per_second,
            slow_verifier_tokens_per_second=classifier_config.slow_verifier_tokens_per_second,
            high_fallback_rate=classifier_config.high_fallback_rate_threshold,
            aggressive_rollout_depth=classifier_config.aggressive_rollout_depth_threshold,
        ),
    )


def _build_summary(
    *,
    rows: list[IngestedBenchmarkRow],
    blocked_rows: list[IngestedBlockedBenchmarkRow],
    error_count: int,
    include_bottleneck_classifications: bool,
) -> JsonObject:
    ar_rows = [row for row in rows if row.telemetry_mode == "ar"]
    mtp_rows = [row for row in rows if row.telemetry_mode == "mtp"]
    live_mtp_rows = [row for row in rows if row.is_live_mtp]
    generation_tps_values = [row.generation_tps for row in rows]
    summary: JsonObject = {
        "valid_row_count": len(rows),
        "error_count": error_count,
        "ar_row_count": len(ar_rows),
        "mtp_row_count": len(mtp_rows),
        "live_mtp_row_count": len(live_mtp_rows),
        "max_generation_tps": max(generation_tps_values)
        if generation_tps_values
        else None,
        "median_ar_generation_tps": _median_generation_tps(ar_rows),
        "median_live_mtp_generation_tps": _median_generation_tps(live_mtp_rows),
        "status": _comparison_status(
            mtp_rows=mtp_rows,
            live_mtp_rows=live_mtp_rows,
            blocked_rows=blocked_rows,
        ),
    }
    if blocked_rows:
        summary["blocked_row_count"] = len(blocked_rows)
    if include_bottleneck_classifications:
        summary["bottleneck_classifications"] = _unique_classifications(rows)
    return summary


def _comparison_status(
    *,
    mtp_rows: list[IngestedBenchmarkRow],
    live_mtp_rows: list[IngestedBenchmarkRow],
    blocked_rows: list[IngestedBlockedBenchmarkRow],
) -> str:
    if live_mtp_rows:
        return "ready_for_same_cluster_comparison"
    if mtp_rows:
        return "blocked_no_live_mtp_rows"
    if blocked_rows:
        return "blocked_no_metric_rows"
    return "blocked_no_mtp_rows"


def _unique_classifications(
    rows: list[IngestedBenchmarkRow],
) -> list[BottleneckClassification]:
    classifications: list[BottleneckClassification] = []
    for row in rows:
        for classification in row.bottleneck_classifications:
            if classification not in classifications:
                classifications.append(classification)
    return classifications


def _median_generation_tps(rows: list[IngestedBenchmarkRow]) -> float | None:
    if not rows:
        return None
    return float(statistics.median(row.generation_tps for row in rows))


def _median_float(values: Iterable[float]) -> float | None:
    value_list = list(values)
    if not value_list:
        return None
    return float(statistics.median(value_list))
