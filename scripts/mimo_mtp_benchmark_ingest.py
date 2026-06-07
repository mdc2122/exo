from __future__ import annotations

import json
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

JsonObject = dict[str, object]
TelemetryMode = Literal["ar", "mtp"]
BottleneckClassification = Literal["acceptance_rate_low"]


@dataclass(frozen=True, slots=True)
class BottleneckClassifierConfig:
    low_acceptance_rate_threshold: float = 0.30


@dataclass(frozen=True, slots=True)
class BenchmarkIngestionError:
    line_number: int
    reason: str
    raw_line: str


@dataclass(frozen=True, slots=True)
class IngestedBenchmarkRow:
    line_number: int
    telemetry_mode: TelemetryMode
    is_live_mtp: bool
    kind: str
    cluster_path: str
    mode: str
    model: str
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
    raw: JsonObject


@dataclass(frozen=True, slots=True)
class BenchmarkIngestionResult:
    rows: list[IngestedBenchmarkRow]
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


def ingest_benchmark_jsonl(
    content: str,
    *,
    classifier_config: BottleneckClassifierConfig | None = None,
) -> BenchmarkIngestionResult:
    rows: list[IngestedBenchmarkRow] = []
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
        parsed = _parse_benchmark_row(
            line_number=index,
            row=cast(JsonObject, decoded),
            raw_line=line,
            classifier_config=classifier_config,
        )
        if isinstance(parsed, BenchmarkIngestionError):
            errors.append(parsed)
        else:
            rows.append(parsed)
    return BenchmarkIngestionResult(
        rows=rows,
        errors=errors,
        summary=_build_summary(
            rows=rows,
            error_count=len(errors),
            include_bottleneck_classifications=classifier_config is not None,
        ),
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
    mode = _required_string(row, "mode")
    if mode is None:
        return _error(line_number, raw_line, "mode must be a string")
    model = _required_string(row, "model")
    if model is None:
        return _error(line_number, raw_line, "model must be a string")
    repeat_index = row.get("repeat_index")
    if not isinstance(repeat_index, int):
        return _error(line_number, raw_line, "repeat_index must be an integer")
    generation_tps = _required_float(row, "generation_tps")
    if generation_tps is None:
        return _error(line_number, raw_line, "generation_tps must be a number")
    generation_tokens = row.get("generation_tokens")
    if not isinstance(generation_tokens, int):
        return _error(line_number, raw_line, "generation_tokens must be an integer")
    prompt_tps = _optional_float(row, "prompt_tps")
    if isinstance(prompt_tps, str):
        return _error(line_number, raw_line, "prompt_tps must be a number when present")
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
        model=model,
        repeat_index=repeat_index,
        generation_tps=generation_tps,
        generation_tokens=generation_tokens,
        accepted_execution_path=accepted_execution_path,
        live_execution_path=live_execution_path,
        optional_prompt_tps=prompt_tps,
        optional_power_usage=row.get("power_usage"),
        optional_acceptance_rate=optional_acceptance_rate,
        bottleneck_classifications=_classify_bottlenecks(
            optional_acceptance_rate=optional_acceptance_rate,
            classifier_config=classifier_config,
        ),
        payload_extra_keys=payload_extra_keys,
        raw=dict(row),
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


def _required_float(row: Mapping[str, object], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _optional_float(row: Mapping[str, object], key: str) -> float | str | None:
    if key not in row or row[key] is None:
        return None
    value = row[key]
    if isinstance(value, bool):
        return "invalid"
    if isinstance(value, int | float):
        return float(value)
    return "invalid"


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
    optional_acceptance_rate: float | None,
    classifier_config: BottleneckClassifierConfig | None,
) -> list[BottleneckClassification]:
    if classifier_config is None or optional_acceptance_rate is None:
        return []
    if optional_acceptance_rate < classifier_config.low_acceptance_rate_threshold:
        return ["acceptance_rate_low"]
    return []


def _build_summary(
    *,
    rows: list[IngestedBenchmarkRow],
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
        "status": "ready_for_same_cluster_comparison"
        if live_mtp_rows
        else "blocked_no_live_mtp_rows",
    }
    if include_bottleneck_classifications:
        summary["bottleneck_classifications"] = _unique_classifications(rows)
    return summary


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
