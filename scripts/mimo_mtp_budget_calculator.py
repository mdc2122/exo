#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Final, Literal, cast

try:
    from scripts import bench_mimo_mtp_cluster as cluster_bench
    from scripts import mimo_mtp_benchmark_ingest as benchmark_ingest
    from scripts import mimo_mtp_bottleneck_classifier as mtp_bottleneck_classifier
except ModuleNotFoundError:
    import bench_mimo_mtp_cluster as cluster_bench
    import mimo_mtp_benchmark_ingest as benchmark_ingest
    import mimo_mtp_bottleneck_classifier as mtp_bottleneck_classifier

JsonObject = dict[str, object]
OptimizationFamily = Literal[
    "proposal",
    "verifier",
    "acceptance_semantics",
    "fallback",
    "depth_policy",
    "slice_5_review",
]
LOW_ACCEPTANCE_RATE_THRESHOLD: Final[float] = 0.30
SLOW_PROPOSAL_TOKENS_PER_SECOND: Final[float] = 12.0
SLOW_VERIFIER_TOKENS_PER_SECOND: Final[float] = 25.0
HIGH_FALLBACK_RATE_THRESHOLD: Final[float] = 0.25
SLICE_5_MINIMUM_SPEEDUP_RATIO: Final[float] = 1.15
AGGRESSIVE_DEPTH_THRESHOLD: Final[int] = 4
OPTIMIZATION_FAMILY_BY_BOTTLENECK: Final[dict[str, OptimizationFamily]] = {
    "proposal_too_slow": "proposal",
    "verifier_too_slow": "verifier",
    "acceptance_rate_low": "acceptance_semantics",
    "fallback_too_high": "fallback",
    "depth_too_aggressive": "depth_policy",
    "mtp_not_faster_than_ar": "depth_policy",
}
BOTTLENECK_PRIORITY: Final[tuple[str, ...]] = (
    "fallback_too_high",
    "acceptance_rate_low",
    "verifier_too_slow",
    "proposal_too_slow",
    "depth_too_aggressive",
    "mtp_not_faster_than_ar",
)
EVIDENCE_COLLECTION_BLOCKERS: Final[set[str]] = {
    "missing_live_mtp_rows",
    "missing_ar_baseline_rows",
    "incomplete_generation_tps_telemetry",
    "insufficient_bottleneck_telemetry",
}


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


def _evidence_sufficiency_counts(
    *,
    rows: Sequence[benchmark_ingest.IngestedBenchmarkRow],
    errors: Sequence[benchmark_ingest.BenchmarkIngestionError],
    blocked_rows: Sequence[benchmark_ingest.IngestedBlockedBenchmarkRow],
) -> JsonObject:
    return {
        "valid_row_count": len(rows),
        "error_count": len(errors),
        "blocked_row_count": len(blocked_rows),
        "ar_row_count": sum(1 for row in rows if row.telemetry_mode == "ar"),
        "mtp_row_count": sum(1 for row in rows if row.telemetry_mode == "mtp"),
        "live_mtp_row_count": sum(1 for row in rows if row.is_live_mtp),
    }


def _evidence_error_reason(
    errors: Sequence[benchmark_ingest.BenchmarkIngestionError],
) -> tuple[str, str]:
    malformed_reasons = {
        "line is not valid JSON",
        "line must decode to a JSON object",
    }
    if any(error.reason in malformed_reasons for error in errors):
        return (
            "malformed_benchmark_evidence",
            "fix malformed benchmark JSONL before evaluating AR-vs-MTP sufficiency",
        )
    return (
        "missing_required_benchmark_fields",
        "repair benchmark rows to the canonical required field contract before evaluating sufficiency",
    )


def _evidence_sufficiency(
    *,
    rows: Sequence[benchmark_ingest.IngestedBenchmarkRow],
    errors: Sequence[benchmark_ingest.BenchmarkIngestionError],
    blocked_rows: Sequence[benchmark_ingest.IngestedBlockedBenchmarkRow],
    budget: Mapping[str, object],
) -> JsonObject:
    budget_status = _budget_status(budget)
    counts = _evidence_sufficiency_counts(
        rows=rows, errors=errors, blocked_rows=blocked_rows
    )
    performance_claim_allowed = _speedup_claim_allowed(budget)

    if errors:
        reason, next_step = _evidence_error_reason(errors)
        return {
            "status": "insufficient_data",
            "reason": reason,
            "budget_status": budget_status,
            **counts,
            "performance_claim_allowed": False,
            "next_step": next_step,
        }

    if blocked_rows:
        return {
            "status": "blocked",
            "reason": "blocked_benchmark_evidence",
            "budget_status": budget_status,
            **counts,
            "performance_claim_allowed": False,
            "next_step": "run the recorded blocked benchmark command against an available guarded cluster before evaluating sufficiency",
        }

    if budget_status == "blocked_missing_mtp_rows":
        return {
            "status": "insufficient_data",
            "reason": "absent_mtp_benchmark_evidence",
            "budget_status": budget_status,
            **counts,
            "performance_claim_allowed": False,
            "next_step": _next_step(budget),
        }

    if budget_status == "blocked_missing_live_mtp_rows":
        return {
            "status": "insufficient_data",
            "reason": "absent_live_mtp_benchmark_evidence",
            "budget_status": budget_status,
            **counts,
            "performance_claim_allowed": False,
            "next_step": _next_step(budget),
        }

    if budget_status == "blocked_missing_ar_rows":
        return {
            "status": "insufficient_data",
            "reason": "absent_ar_benchmark_evidence",
            "budget_status": budget_status,
            **counts,
            "performance_claim_allowed": False,
            "next_step": _next_step(budget),
        }

    if budget_status == "blocked_incomplete_telemetry":
        return {
            "status": "insufficient_data",
            "reason": "incomplete_benchmark_telemetry",
            "budget_status": budget_status,
            **counts,
            "performance_claim_allowed": False,
            "next_step": _next_step(budget),
        }

    if budget_status == "ambiguous_non_comparable_cluster_rows":
        return {
            "status": "insufficient_data",
            "reason": "non_comparable_cluster_evidence",
            "budget_status": budget_status,
            **counts,
            "performance_claim_allowed": False,
            "next_step": _next_step(budget),
        }

    return {
        "status": "sufficient" if performance_claim_allowed else "insufficient_data",
        "reason": "same_cluster_ar_mtp_evidence_ready"
        if performance_claim_allowed
        else "same_cluster_mtp_not_faster_than_ar",
        "budget_status": budget_status,
        **counts,
        "performance_claim_allowed": performance_claim_allowed,
        "next_step": _next_step(budget),
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


def _speedup_ratio(budget: Mapping[str, object]) -> float | None:
    return _float_field(budget, "mtp_vs_ar_speedup_ratio")


def _speedup_claim_allowed(budget: Mapping[str, object]) -> bool:
    speedup_ratio = _speedup_ratio(budget)
    return _budget_status(budget) == "same_cluster_budget_ready" and (
        speedup_ratio is not None and speedup_ratio > 1.0
    )


def _budget_classification(budget: Mapping[str, object]) -> str:
    status = _budget_status(budget)
    if status != "same_cluster_budget_ready":
        return "ambiguous"
    speedup_ratio = _speedup_ratio(budget)
    if speedup_ratio is None:
        return "ambiguous"
    if speedup_ratio > 1.0:
        return "pass"
    return "fail"


def _slice_5_gate(
    budget: Mapping[str, object], *, bottlenecks: Sequence[str] = ()
) -> str:
    status = _budget_status(budget)
    if status != "same_cluster_budget_ready":
        return status
    if not _speedup_claim_allowed(budget):
        return "blocked_mtp_not_faster_than_ar"
    if "insufficient_bottleneck_telemetry" in bottlenecks:
        return "blocked_insufficient_bottleneck_telemetry"
    if "fallback_too_high" in bottlenecks:
        return "blocked_fallback_or_correctness_concern"

    speedup_ratio = _speedup_ratio(budget)
    if speedup_ratio is None:
        return "blocked_incomplete_telemetry"
    if speedup_ratio < SLICE_5_MINIMUM_SPEEDUP_RATIO:
        return "blocked_margin_below_15_percent"
    return "eligible_for_guarded_review"


def _slice_5_review_allowed(
    budget: Mapping[str, object], *, bottlenecks: Sequence[str] = ()
) -> bool:
    return (
        _slice_5_gate(budget, bottlenecks=bottlenecks) == "eligible_for_guarded_review"
    )


def _refusal_reasons(budget: Mapping[str, object]) -> list[str]:
    status = _budget_status(budget)
    if status == "blocked_missing_mtp_rows":
        return ["ar_only_evidence_no_guarded_live_mtp_rows"]
    if status == "blocked_missing_live_mtp_rows":
        return ["mtp_labeled_rows_without_guarded_live_fastpath"]
    if status == "blocked_missing_ar_rows":
        return ["mtp_evidence_without_same_cluster_ar_baseline"]
    if status == "ambiguous_non_comparable_cluster_rows":
        return ["non_comparable_cluster_evidence"]
    if status == "blocked_incomplete_telemetry":
        return ["incomplete_generation_tps_telemetry"]
    if status == "ambiguous_non_comparable_cluster_rows":
        return ["non_comparable_cluster_evidence"]
    return []


def _claim_suppression_reason(
    *,
    errors: Sequence[benchmark_ingest.BenchmarkIngestionError],
    blocked_rows: Sequence[benchmark_ingest.IngestedBlockedBenchmarkRow],
) -> str | None:
    if errors:
        reason, _next_step = _evidence_error_reason(errors)
        return reason
    if blocked_rows:
        return "blocked_benchmark_evidence"
    return None


def _classifications(
    budget: Mapping[str, object],
    *,
    bottlenecks: Sequence[str] = (),
    errors: Sequence[benchmark_ingest.BenchmarkIngestionError] = (),
    blocked_rows: Sequence[benchmark_ingest.IngestedBlockedBenchmarkRow] = (),
) -> JsonObject:
    mtp_median_tok_s = _float_field(budget, "mtp_median_tok_s")
    suppression_reason = _claim_suppression_reason(
        errors=errors, blocked_rows=blocked_rows
    )
    if suppression_reason is not None:
        return {
            "same_cluster_evidence": suppression_reason,
            "budget_classification": "ambiguous",
            "mtp_throughput_threshold": None,
            "slice_5_gate": suppression_reason
            if suppression_reason.startswith("blocked_")
            else f"blocked_{suppression_reason}",
            "slice_5_review_allowed": False,
            "slice_5_minimum_speedup_ratio": SLICE_5_MINIMUM_SPEEDUP_RATIO,
            "speedup_claim_allowed": False,
            "target_30_tok_s_claim_allowed": False,
            "target_40_tok_s_claim_allowed": False,
            "refusal_reasons": [suppression_reason],
        }

    speedup_claim_allowed = _speedup_claim_allowed(budget)
    classifications: JsonObject = {
        "same_cluster_evidence": _budget_status(budget),
        "budget_classification": _budget_classification(budget),
        "mtp_throughput_threshold": None
        if mtp_median_tok_s is None
        else cluster_bench.classify_mtp_throughput_threshold(mtp_median_tok_s),
        "slice_5_gate": _slice_5_gate(budget, bottlenecks=bottlenecks),
        "slice_5_review_allowed": _slice_5_review_allowed(
            budget, bottlenecks=bottlenecks
        ),
        "slice_5_minimum_speedup_ratio": SLICE_5_MINIMUM_SPEEDUP_RATIO,
        "speedup_claim_allowed": speedup_claim_allowed,
        "target_30_tok_s_claim_allowed": speedup_claim_allowed
        and mtp_median_tok_s is not None
        and mtp_median_tok_s >= cluster_bench.MTP_TARGET_TOKENS_PER_SECOND,
        "target_40_tok_s_claim_allowed": speedup_claim_allowed
        and mtp_median_tok_s is not None
        and mtp_median_tok_s >= cluster_bench.MTP_PREFERRED_TOKENS_PER_SECOND,
    }
    refusal_reasons = _refusal_reasons(budget)
    if refusal_reasons:
        classifications["refusal_reasons"] = refusal_reasons
    return classifications


def _add_label(labels: list[str], label: str) -> None:
    if label not in labels:
        labels.append(label)


def _mapping_field(row: Mapping[str, object], key: str) -> Mapping[str, object] | None:
    value = row.get(key)
    if isinstance(value, dict):
        return cast(Mapping[str, object], value)
    return None


def _acceptance_rate(row: Mapping[str, object]) -> float | None:
    explicit_rate = _float_field(row, "acceptance_rate")
    if explicit_rate is not None:
        return explicit_rate
    generation_stats = _mapping_field(row, "generation_stats")
    if generation_stats is None:
        return None
    nested_rate = _float_field(generation_stats, "acceptance_rate")
    if nested_rate is not None:
        return nested_rate
    accepted_tokens = _float_field(generation_stats, "mimo_mtp_accepted_tokens")
    attempted_tokens = _float_field(generation_stats, "mimo_mtp_attempted_tokens")
    if accepted_tokens is None or attempted_tokens is None or attempted_tokens <= 0.0:
        return None
    return accepted_tokens / attempted_tokens


def _fallback_rate(row: Mapping[str, object]) -> float | None:
    explicit_rate = _float_field(row, "fallback_rate")
    if explicit_rate is not None:
        return explicit_rate
    generation_stats = _mapping_field(row, "generation_stats")
    if generation_stats is None:
        return None
    nested_rate = _float_field(generation_stats, "fallback_rate")
    if nested_rate is not None:
        return nested_rate
    fallback_count = _float_field(generation_stats, "mimo_mtp_fallback_count")
    attempted_tokens = _float_field(generation_stats, "mimo_mtp_attempted_tokens")
    if fallback_count is None or attempted_tokens is None or attempted_tokens <= 0.0:
        return None
    return fallback_count / attempted_tokens


def _has_benchmark_grade_mtp_bottleneck_telemetry(
    row: benchmark_ingest.IngestedBenchmarkRow,
) -> bool:
    return (
        _acceptance_rate(row.raw) is not None
        and _fallback_rate(row.raw) is not None
        and _measured_depth(row) is not None
    )


def _rollout_depth(row: Mapping[str, object]) -> int | None:
    for key in ("mimo_mtp_depth", "rollout_depth"):
        value = _float_field(row, key)
        if value is not None:
            return int(value)
    generation_stats = _mapping_field(row, "generation_stats")
    if generation_stats is not None:
        for key in ("mimo_mtp_depth", "rollout_depth"):
            value = _float_field(generation_stats, key)
            if value is not None:
                return int(value)
    return None


def _depth_from_mode_label(mode: str) -> int | None:
    prefix = "mtp-d"
    if not mode.startswith(prefix):
        return None
    raw_depth = mode.removeprefix(prefix)
    if not raw_depth.isdigit():
        return None
    return int(raw_depth)


def _measured_depth(row: benchmark_ingest.IngestedBenchmarkRow) -> int | None:
    rollout_depth = _rollout_depth(row.raw)
    if rollout_depth is not None:
        return rollout_depth
    return _depth_from_mode_label(row.mode)


def _append_row_bottlenecks(labels: list[str], row: Mapping[str, object]) -> None:
    for label in mtp_bottleneck_classifier.classify_bottlenecks(
        row,
        thresholds=mtp_bottleneck_classifier.BottleneckThresholds(
            slow_proposal_tokens_per_second=SLOW_PROPOSAL_TOKENS_PER_SECOND,
            slow_verifier_tokens_per_second=SLOW_VERIFIER_TOKENS_PER_SECOND,
        ),
    ):
        _add_label(labels, label)

    fallback_rate = _fallback_rate(row)
    if fallback_rate is not None and fallback_rate >= HIGH_FALLBACK_RATE_THRESHOLD:
        _add_label(labels, "fallback_too_high")

    rollout_depth = _rollout_depth(row)
    if rollout_depth is not None and rollout_depth >= AGGRESSIVE_DEPTH_THRESHOLD:
        _add_label(labels, "depth_too_aggressive")


def _append_budget_bottlenecks(labels: list[str], budget: Mapping[str, object]) -> None:
    for label in mtp_bottleneck_classifier.classify_budget_bottlenecks(budget):
        _add_label(labels, label)


def _bottlenecks(
    *,
    rows: Iterable[benchmark_ingest.IngestedBenchmarkRow],
    budget: Mapping[str, object],
) -> list[str]:
    labels: list[str] = []
    live_mtp_row_seen = False
    benchmark_grade_bottleneck_telemetry_seen = False
    for row in rows:
        for label in row.bottleneck_classifications:
            _add_label(labels, label)
        if row.is_live_mtp:
            live_mtp_row_seen = True
            if _has_benchmark_grade_mtp_bottleneck_telemetry(row):
                benchmark_grade_bottleneck_telemetry_seen = True
            _append_row_bottlenecks(labels, row.raw)
    _append_budget_bottlenecks(labels, budget)

    status = _budget_status(budget)
    if status in ("blocked_missing_mtp_rows", "blocked_missing_live_mtp_rows"):
        labels.append("missing_live_mtp_rows")
    elif status == "blocked_missing_ar_rows":
        labels.append("missing_ar_baseline_rows")
    elif status == "blocked_incomplete_telemetry":
        labels.append("incomplete_generation_tps_telemetry")
    elif status == "same_cluster_budget_ready" and not _speedup_claim_allowed(budget):
        labels.append("mtp_not_faster_than_ar")
    elif (
        status == "same_cluster_budget_ready"
        and live_mtp_row_seen
        and not benchmark_grade_bottleneck_telemetry_seen
    ):
        labels.append("insufficient_bottleneck_telemetry")
    return labels


def _depth_policy(rows: Sequence[benchmark_ingest.IngestedBenchmarkRow]) -> JsonObject:
    depth_tps: dict[int, list[float]] = {}
    for row in rows:
        if not row.is_live_mtp:
            continue
        depth = _measured_depth(row)
        if depth is None:
            continue
        depth_tps.setdefault(depth, []).append(row.generation_tps)

    measured_depths = sorted(depth_tps)
    if not measured_depths:
        return {
            "policy": "blocked_missing_measured_depth_rows",
            "recommended_depth": None,
            "measured_depths": [],
            "d3_assumed_best": False,
        }

    median_by_depth = {
        str(depth): round(statistics.median(depth_tps[depth]), 6)
        for depth in measured_depths
    }
    recommended_depth = max(
        measured_depths,
        key=lambda depth: (statistics.median(depth_tps[depth]), -depth),
    )
    return {
        "policy": "auto_depth_from_measured_rows",
        "recommended_depth": recommended_depth,
        "measured_depths": measured_depths,
        "median_generation_tps_by_depth": median_by_depth,
        "d3_assumed_best": False,
    }


def _candidate_optimization_families(
    bottlenecks: Sequence[str],
) -> list[OptimizationFamily]:
    families: list[OptimizationFamily] = []
    for bottleneck in BOTTLENECK_PRIORITY:
        if bottleneck not in bottlenecks:
            continue
        family = OPTIMIZATION_FAMILY_BY_BOTTLENECK[bottleneck]
        if family not in families:
            families.append(family)
    return families


def _optimization_loop(
    *,
    rows: Sequence[benchmark_ingest.IngestedBenchmarkRow],
    budget: Mapping[str, object],
    bottlenecks: Sequence[str],
) -> JsonObject:
    depth_policy = _depth_policy(rows)
    if "insufficient_bottleneck_telemetry" in bottlenecks:
        return {
            "recommendation_status": "insufficient_data",
            "next_optimization_family": None,
            "candidate_families": [],
            "all_measured_candidate_families": [],
            "blind_optimization_allowed": False,
            "blocking_reason": (
                "collect benchmark-grade MTP bottleneck telemetry before choosing an "
                "optimization family or Slice 5 review"
            ),
            "depth_policy": depth_policy,
        }

    candidate_families = _candidate_optimization_families(bottlenecks)
    if candidate_families:
        recommended_family = candidate_families[0]
        return {
            "recommendation_status": "measured_bottleneck_ready",
            "next_optimization_family": recommended_family,
            "candidate_families": [recommended_family],
            "all_measured_candidate_families": candidate_families,
            "blind_optimization_allowed": False,
            "depth_policy": depth_policy,
        }

    if _slice_5_gate(budget, bottlenecks=bottlenecks) == "eligible_for_guarded_review":
        return {
            "recommendation_status": "measured_bottleneck_ready",
            "next_optimization_family": "slice_5_review",
            "candidate_families": ["slice_5_review"],
            "all_measured_candidate_families": [],
            "blind_optimization_allowed": False,
            "depth_policy": depth_policy,
        }

    evidence_blockers = [
        bottleneck
        for bottleneck in bottlenecks
        if bottleneck in EVIDENCE_COLLECTION_BLOCKERS
    ]
    blocking_reason = (
        "collect same-cluster AR and guarded live MTP rows before optimizing"
    )
    if evidence_blockers:
        blocking_reason = (
            f"{blocking_reason}; {evidence_blockers[0]} is an evidence collection "
            "blocker, not an optimization family"
        )
    return {
        "recommendation_status": "blocked_no_measured_bottleneck",
        "next_optimization_family": None,
        "candidate_families": [],
        "all_measured_candidate_families": [],
        "blind_optimization_allowed": False,
        "blocking_reason": blocking_reason,
        "depth_policy": depth_policy,
    }


def _refine_budget_status_from_ingested_rows(
    *,
    rows: Sequence[benchmark_ingest.IngestedBenchmarkRow],
    budget: Mapping[str, object],
) -> JsonObject:
    refined_budget = dict(budget)
    has_mtp_labeled_rows = any(row.telemetry_mode == "mtp" for row in rows)
    has_live_mtp_rows = any(row.is_live_mtp for row in rows)
    if (
        _budget_status(budget) == "blocked_missing_mtp_rows"
        and has_mtp_labeled_rows
        and not has_live_mtp_rows
    ):
        refined_budget["status"] = "blocked_missing_live_mtp_rows"
        refined_budget["next_step"] = (
            "collect guarded MTP rows with accepted_execution_path=mimo_mtp_fastpath "
            "before making speedup or target claims"
        )
    return refined_budget


def build_budget_report(jsonl_paths: Sequence[str | Path]) -> JsonObject:
    rows: list[benchmark_ingest.IngestedBenchmarkRow] = []
    blocked_rows: list[benchmark_ingest.IngestedBlockedBenchmarkRow] = []
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
        blocked_rows.extend(result.blocked_rows)
        errors.extend(result.errors)

    budget = _refine_budget_status_from_ingested_rows(
        rows=rows,
        budget=cluster_bench.calculate_mtp_speedup_budget(row.raw for row in rows),
    )
    bottlenecks = _bottlenecks(rows=rows, budget=budget)
    return {
        "kind": "mimo_mtp_budget_report",
        "schema_version": 1,
        "source_files": [str(path) for path in source_paths],
        "ingestion": _ingestion_summary(rows=rows, errors=errors),
        "evidence_sufficiency": _evidence_sufficiency(
            rows=rows,
            errors=errors,
            blocked_rows=blocked_rows,
            budget=budget,
        ),
        "budget": budget,
        "classifications": _classifications(
            budget,
            bottlenecks=bottlenecks,
            errors=errors,
            blocked_rows=blocked_rows,
        ),
        "bottlenecks": bottlenecks,
        "optimization_loop": _optimization_loop(
            rows=rows,
            budget=budget,
            bottlenecks=bottlenecks,
        ),
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
