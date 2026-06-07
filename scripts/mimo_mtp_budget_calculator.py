#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Final, cast

try:
    from scripts import bench_mimo_mtp_cluster as cluster_bench
    from scripts import mimo_mtp_benchmark_ingest as benchmark_ingest
except ModuleNotFoundError:
    import bench_mimo_mtp_cluster as cluster_bench
    import mimo_mtp_benchmark_ingest as benchmark_ingest

JsonObject = dict[str, object]
LOW_ACCEPTANCE_RATE_THRESHOLD: Final[float] = 0.30


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute MiMo MTP budget metrics from AR/MTP JSONL benchmark rows."
    )
    parser.add_argument(
        "jsonl_paths",
        nargs="+",
        type=Path,
        help="One or more JSONL files emitted by scripts/bench_mimo_mtp_cluster.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path where the stable JSON report should be written.",
    )
    return parser


def _ingestion_summary(
    *,
    rows: Sequence[benchmark_ingest.IngestedBenchmarkRow],
    errors: Sequence[benchmark_ingest.BenchmarkIngestionError],
) -> JsonObject:
    return {
        "valid_row_count": len(rows),
        "error_count": len(errors),
        "ar_row_count": sum(1 for row in rows if row.telemetry_mode == "ar"),
        "mtp_row_count": sum(1 for row in rows if row.telemetry_mode == "mtp"),
        "live_mtp_row_count": sum(1 for row in rows if row.is_live_mtp),
    }


def _float_field(row: Mapping[str, object], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _budget_status(budget: Mapping[str, object]) -> str:
    status = budget.get("status")
    if isinstance(status, str):
        return status
    return "blocked_incomplete_telemetry"


def _next_step(budget: Mapping[str, object]) -> str:
    next_step = budget.get("next_step")
    if isinstance(next_step, str):
        return next_step
    return "collect same-cluster AR and guarded MTP rows before making speedup claims"


def _speedup_claim_allowed(budget: Mapping[str, object]) -> bool:
    speedup_ratio = _float_field(budget, "mtp_vs_ar_speedup_ratio")
    return _budget_status(budget) == "same_cluster_budget_ready" and (
        speedup_ratio is not None and speedup_ratio > 1.0
    )


def _slice_5_gate(budget: Mapping[str, object]) -> str:
    status = _budget_status(budget)
    if status != "same_cluster_budget_ready":
        return status
    if _speedup_claim_allowed(budget):
        return "eligible_for_guarded_review"
    return "blocked_mtp_not_faster_than_ar"


def _classifications(budget: Mapping[str, object]) -> JsonObject:
    mtp_median_tok_s = _float_field(budget, "mtp_median_tok_s")
    speedup_claim_allowed = _speedup_claim_allowed(budget)
    return {
        "same_cluster_evidence": _budget_status(budget),
        "mtp_throughput_threshold": None
        if mtp_median_tok_s is None
        else cluster_bench.classify_mtp_throughput_threshold(mtp_median_tok_s),
        "slice_5_gate": _slice_5_gate(budget),
        "speedup_claim_allowed": speedup_claim_allowed,
        "target_30_tok_s_claim_allowed": speedup_claim_allowed
        and mtp_median_tok_s is not None
        and mtp_median_tok_s >= cluster_bench.MTP_TARGET_TOKENS_PER_SECOND,
        "target_40_tok_s_claim_allowed": speedup_claim_allowed
        and mtp_median_tok_s is not None
        and mtp_median_tok_s >= cluster_bench.MTP_PREFERRED_TOKENS_PER_SECOND,
    }


def _bottlenecks(
    *,
    rows: Iterable[benchmark_ingest.IngestedBenchmarkRow],
    budget: Mapping[str, object],
) -> list[str]:
    labels: list[str] = []
    for row in rows:
        for label in row.bottleneck_classifications:
            if label not in labels:
                labels.append(label)

    status = _budget_status(budget)
    if status == "blocked_missing_mtp_rows":
        labels.append("missing_live_mtp_rows")
    elif status == "blocked_missing_ar_rows":
        labels.append("missing_ar_baseline_rows")
    elif status == "blocked_incomplete_telemetry":
        labels.append("incomplete_generation_tps_telemetry")
    elif status == "same_cluster_budget_ready" and not _speedup_claim_allowed(budget):
        labels.append("mtp_not_faster_than_ar")
    return labels


def build_budget_report(jsonl_paths: Sequence[str | Path]) -> JsonObject:
    rows: list[benchmark_ingest.IngestedBenchmarkRow] = []
    errors: list[benchmark_ingest.BenchmarkIngestionError] = []
    source_paths = [Path(path) for path in jsonl_paths]
    for path in source_paths:
        result = benchmark_ingest.ingest_benchmark_jsonl_file(
            path,
            classifier_config=benchmark_ingest.BottleneckClassifierConfig(
                low_acceptance_rate_threshold=LOW_ACCEPTANCE_RATE_THRESHOLD
            ),
        )
        rows.extend(result.rows)
        errors.extend(result.errors)

    budget = cluster_bench.calculate_mtp_speedup_budget(row.raw for row in rows)
    return {
        "kind": "mimo_mtp_budget_report",
        "schema_version": 1,
        "source_files": [str(path) for path in source_paths],
        "ingestion": _ingestion_summary(rows=rows, errors=errors),
        "budget": budget,
        "classifications": _classifications(budget),
        "bottlenecks": _bottlenecks(rows=rows, budget=budget),
        "next_step": _next_step(budget),
    }


def render_report(report: Mapping[str, object]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    namespace = _parser().parse_args(argv)
    jsonl_paths = cast(list[Path], namespace.jsonl_paths)
    output_path = cast(Path | None, namespace.output)
    report = build_budget_report(jsonl_paths)
    rendered_report = render_report(report)
    if output_path is None:
        print(rendered_report, end="")
    else:
        output_path.write_text(rendered_report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
