#!/usr/bin/env python3
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, cast

BottleneckLabel = Literal[
    "acceptance_rate_low",
    "proposal_too_slow",
    "verifier_too_slow",
    "fallback_too_high",
    "depth_too_aggressive",
]
BudgetBottleneckLabel = Literal[
    "mtp_beats_ar",
    "mtp_reaches_30",
    "mtp_reaches_40",
]
TerminalDecisionLabel = Literal[
    "pass",
    "fail",
    "ambiguous",
    "blocked",
    "insufficient_data",
]
JsonMapping = Mapping[str, object]
DEFAULT_SLOW_VERIFIER_TIME_SHARE: Final[float] = 0.50
DEFAULT_SLOW_VERIFIER_TOKENS_PER_SECOND: Final[float | None] = None
DEFAULT_SLOW_PROPOSAL_TOKENS_PER_SECOND: Final[float | None] = None
DEFAULT_LOW_ACCEPTANCE_RATE: Final[float | None] = None
DEFAULT_HIGH_FALLBACK_RATE: Final[float | None] = None
DEFAULT_AGGRESSIVE_ROLLOUT_DEPTH: Final[int | None] = None
SAME_CLUSTER_BUDGET_READY_STATUS: Final[str] = "same_cluster_budget_ready"
MTP_TARGET_TOKENS_PER_SECOND: Final[float] = 30.0
MTP_PREFERRED_TOKENS_PER_SECOND: Final[float] = 40.0


@dataclass(frozen=True, slots=True)
class BottleneckThresholds:
    slow_verifier_time_share: float = DEFAULT_SLOW_VERIFIER_TIME_SHARE
    slow_verifier_tokens_per_second: float | None = (
        DEFAULT_SLOW_VERIFIER_TOKENS_PER_SECOND
    )
    slow_proposal_tokens_per_second: float | None = (
        DEFAULT_SLOW_PROPOSAL_TOKENS_PER_SECOND
    )
    low_acceptance_rate: float | None = DEFAULT_LOW_ACCEPTANCE_RATE
    high_fallback_rate: float | None = DEFAULT_HIGH_FALLBACK_RATE
    aggressive_rollout_depth: int | None = DEFAULT_AGGRESSIVE_ROLLOUT_DEPTH


DEFAULT_BOTTLENECK_THRESHOLDS: Final[BottleneckThresholds] = BottleneckThresholds()


def _json_mapping(value: object) -> JsonMapping | None:
    if not isinstance(value, dict):
        return None
    return cast(JsonMapping, value)


def _float_value(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _generation_stats(row: JsonMapping) -> JsonMapping | None:
    return _json_mapping(row.get("generation_stats"))


def _timing_breakdown(row: JsonMapping) -> JsonMapping | None:
    direct_timing = _json_mapping(row.get("timing_breakdown_seconds"))
    if direct_timing is not None:
        return direct_timing
    generation_stats = _generation_stats(row)
    if generation_stats is None:
        return None
    return _json_mapping(generation_stats.get("timing_breakdown_seconds"))


def _generated_tokens(row: JsonMapping) -> float | None:
    for key in ("generated_tokens", "generation_tokens"):
        value = _float_value(row.get(key))
        if value is not None:
            return value
    generation_stats = _generation_stats(row)
    if generation_stats is None:
        return None
    return _float_value(generation_stats.get("generation_tokens"))


def _verifier_time_share(timing_breakdown: JsonMapping) -> float | None:
    timing_seconds: list[float] = []
    for key in ("proposal", "verification", "acceptance", "fallback"):
        seconds = _float_value(timing_breakdown.get(key))
        if seconds is None or seconds < 0.0:
            return None
        timing_seconds.append(seconds)
    verification_seconds = timing_seconds[1]
    if verification_seconds <= 0.0:
        return None
    total_seconds = sum(timing_seconds)
    if total_seconds <= 0.0:
        return None
    return verification_seconds / total_seconds


def _explicit_rate(row: JsonMapping, key: str) -> float | None:
    value = _float_value(row.get(key))
    if value is not None:
        return value
    generation_stats = _generation_stats(row)
    if generation_stats is None:
        return None
    return _float_value(generation_stats.get(key))


def _verifier_tokens_per_second(
    row: JsonMapping, timing_breakdown: JsonMapping | None
) -> float | None:
    explicit_value = _explicit_rate(row, "verification_tps")
    if explicit_value is not None:
        return explicit_value
    if timing_breakdown is None:
        return None
    generated_tokens = _generated_tokens(row)
    verification_seconds = _float_value(timing_breakdown.get("verification"))
    if (
        generated_tokens is None
        or generated_tokens <= 0.0
        or verification_seconds is None
        or verification_seconds <= 0.0
    ):
        return None
    return generated_tokens / verification_seconds


def _proposal_tokens(row: JsonMapping) -> float | None:
    for key in ("proposal_tokens", "mimo_mtp_proposed_tokens"):
        value = _float_value(row.get(key))
        if value is not None:
            return value
    generation_stats = _generation_stats(row)
    if generation_stats is None:
        return None
    for key in (
        "mimo_mtp_proposed_tokens",
        "proposal_tokens",
        "mimo_mtp_attempted_tokens",
    ):
        value = _float_value(generation_stats.get(key))
        if value is not None:
            return value
    return None


def _proposal_tokens_per_second(
    row: JsonMapping, timing_breakdown: JsonMapping | None
) -> float | None:
    explicit_value = _explicit_rate(row, "proposal_tps")
    if explicit_value is not None:
        return explicit_value
    if timing_breakdown is None:
        return None
    proposed_tokens = _proposal_tokens(row)
    proposal_seconds = _float_value(timing_breakdown.get("proposal"))
    if (
        proposed_tokens is None
        or proposed_tokens <= 0.0
        or proposal_seconds is None
        or proposal_seconds <= 0.0
    ):
        return None
    return proposed_tokens / proposal_seconds


def _numeric_generation_stat(row: JsonMapping, key: str) -> float | None:
    value = _float_value(row.get(key))
    if value is not None:
        return value
    generation_stats = _generation_stats(row)
    if generation_stats is None:
        return None
    return _float_value(generation_stats.get(key))


def _acceptance_rate(row: JsonMapping) -> float | None:
    explicit_acceptance_rate = _numeric_generation_stat(row, "acceptance_rate")
    if explicit_acceptance_rate is not None:
        return explicit_acceptance_rate
    attempted_tokens = _numeric_generation_stat(row, "mimo_mtp_attempted_tokens")
    accepted_tokens = _numeric_generation_stat(row, "mimo_mtp_accepted_tokens")
    if attempted_tokens is None or accepted_tokens is None or attempted_tokens <= 0.0:
        return None
    return accepted_tokens / attempted_tokens


def _fallback_rate(row: JsonMapping) -> float | None:
    explicit_fallback_rate = _numeric_generation_stat(row, "fallback_rate")
    if explicit_fallback_rate is not None:
        return explicit_fallback_rate
    attempted_tokens = _numeric_generation_stat(row, "mimo_mtp_attempted_tokens")
    fallback_count = _numeric_generation_stat(row, "mimo_mtp_fallback_count")
    if attempted_tokens is None or fallback_count is None or attempted_tokens <= 0.0:
        return None
    return fallback_count / attempted_tokens


def _rollout_depth(row: JsonMapping) -> int | None:
    for key in ("mimo_mtp_depth", "rollout_depth"):
        value = _numeric_generation_stat(row, key)
        if value is not None:
            return int(value)
    return None


def _is_acceptance_rate_low(
    row: JsonMapping, *, thresholds: BottleneckThresholds
) -> bool:
    if thresholds.low_acceptance_rate is None:
        return False
    acceptance_rate = _acceptance_rate(row)
    return (
        acceptance_rate is not None
        and acceptance_rate <= thresholds.low_acceptance_rate
    )


def _is_proposal_too_slow(
    row: JsonMapping, *, thresholds: BottleneckThresholds
) -> bool:
    if thresholds.slow_proposal_tokens_per_second is None:
        return False
    proposal_tokens_per_second = _proposal_tokens_per_second(
        row, _timing_breakdown(row)
    )
    return (
        proposal_tokens_per_second is not None
        and proposal_tokens_per_second <= thresholds.slow_proposal_tokens_per_second
    )


def _is_verifier_too_slow(
    row: JsonMapping, *, thresholds: BottleneckThresholds
) -> bool:
    timing_breakdown = _timing_breakdown(row)
    time_share = (
        None if timing_breakdown is None else _verifier_time_share(timing_breakdown)
    )
    if time_share is not None and time_share >= thresholds.slow_verifier_time_share:
        return True
    if thresholds.slow_verifier_tokens_per_second is None:
        return False
    verifier_tokens_per_second = _verifier_tokens_per_second(row, timing_breakdown)
    return (
        verifier_tokens_per_second is not None
        and verifier_tokens_per_second <= thresholds.slow_verifier_tokens_per_second
    )


def _is_fallback_too_high(
    row: JsonMapping, *, thresholds: BottleneckThresholds
) -> bool:
    if thresholds.high_fallback_rate is None:
        return False
    fallback_rate = _fallback_rate(row)
    return fallback_rate is not None and fallback_rate >= thresholds.high_fallback_rate


def _is_depth_too_aggressive(
    row: JsonMapping, *, thresholds: BottleneckThresholds
) -> bool:
    if thresholds.aggressive_rollout_depth is None:
        return False
    rollout_depth = _rollout_depth(row)
    return (
        rollout_depth is not None
        and rollout_depth >= thresholds.aggressive_rollout_depth
    )


def classify_bottlenecks(
    row: JsonMapping,
    *,
    thresholds: BottleneckThresholds = DEFAULT_BOTTLENECK_THRESHOLDS,
) -> list[BottleneckLabel]:
    bottlenecks: list[BottleneckLabel] = []
    if _is_acceptance_rate_low(row, thresholds=thresholds):
        bottlenecks.append("acceptance_rate_low")
    if _is_proposal_too_slow(row, thresholds=thresholds):
        bottlenecks.append("proposal_too_slow")
    if _is_verifier_too_slow(row, thresholds=thresholds):
        bottlenecks.append("verifier_too_slow")
    if _is_fallback_too_high(row, thresholds=thresholds):
        bottlenecks.append("fallback_too_high")
    if _is_depth_too_aggressive(row, thresholds=thresholds):
        bottlenecks.append("depth_too_aggressive")
    return bottlenecks


def classify_terminal_decision(evidence: JsonMapping) -> TerminalDecisionLabel:
    """Map a validated same-cluster metrics/evidence summary to one terminal label.

    This classifier is intentionally fail-safe: unavailable benchmark evidence is
    blocked, comparability violations are ambiguous, and missing validated metrics
    are insufficient_data. Only comparable, fully validated same-cluster speedup
    metrics can pass or fail.
    """
    if evidence.get("input_validated") is not True:
        return "insufficient_data"

    status = evidence.get("status")
    if isinstance(status, str) and status.startswith("blocked_"):
        return "blocked"

    if evidence.get("same_cluster_comparable") is not True:
        return "ambiguous"

    if status != SAME_CLUSTER_BUDGET_READY_STATUS:
        return "insufficient_data"

    speedup_ratio = _float_value(evidence.get("mtp_vs_ar_speedup_ratio"))
    if speedup_ratio is None:
        return "insufficient_data"
    if speedup_ratio > 1.0:
        return "pass"
    return "fail"


def _positive_count(value: object) -> bool:
    numeric_value = _float_value(value)
    return numeric_value is not None and numeric_value > 0.0


def _has_required_ar_mtp_budget_telemetry(budget: JsonMapping) -> bool:
    return (
        _positive_count(budget.get("ar_row_count"))
        and _positive_count(budget.get("mtp_row_count"))
        and _float_value(budget.get("ar_median_tok_s")) is not None
        and _float_value(budget.get("mtp_median_tok_s")) is not None
        and _float_value(budget.get("mtp_vs_ar_speedup_ratio")) is not None
    )


def classify_budget_bottlenecks(budget: JsonMapping) -> list[BudgetBottleneckLabel]:
    if budget.get("status") != SAME_CLUSTER_BUDGET_READY_STATUS:
        return []
    if not _has_required_ar_mtp_budget_telemetry(budget):
        return []

    bottlenecks: list[BudgetBottleneckLabel] = []
    speedup_ratio = _float_value(budget.get("mtp_vs_ar_speedup_ratio"))
    if speedup_ratio is not None and speedup_ratio > 1.0:
        bottlenecks.append("mtp_beats_ar")

    mtp_median_tok_s = _float_value(budget.get("mtp_median_tok_s"))
    if mtp_median_tok_s is None:
        return bottlenecks
    if mtp_median_tok_s >= MTP_TARGET_TOKENS_PER_SECOND:
        bottlenecks.append("mtp_reaches_30")
    if mtp_median_tok_s >= MTP_PREFERRED_TOKENS_PER_SECOND:
        bottlenecks.append("mtp_reaches_40")
    return bottlenecks
