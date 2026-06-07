#!/usr/bin/env python3
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, cast

BottleneckLabel = Literal["proposal_too_slow", "verifier_too_slow"]
JsonMapping = Mapping[str, object]
DEFAULT_SLOW_VERIFIER_TIME_SHARE: Final[float] = 0.50
DEFAULT_SLOW_VERIFIER_TOKENS_PER_SECOND: Final[float | None] = None
DEFAULT_SLOW_PROPOSAL_TOKENS_PER_SECOND: Final[float | None] = None


@dataclass(frozen=True, slots=True)
class BottleneckThresholds:
    slow_verifier_time_share: float = DEFAULT_SLOW_VERIFIER_TIME_SHARE
    slow_verifier_tokens_per_second: float | None = (
        DEFAULT_SLOW_VERIFIER_TOKENS_PER_SECOND
    )
    slow_proposal_tokens_per_second: float | None = (
        DEFAULT_SLOW_PROPOSAL_TOKENS_PER_SECOND
    )


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
    verification_seconds = _float_value(timing_breakdown.get("verification"))
    if verification_seconds is None or verification_seconds <= 0.0:
        return None
    total_seconds = 0.0
    for key in ("proposal", "verification", "acceptance", "fallback"):
        seconds = _float_value(timing_breakdown.get(key))
        if seconds is not None and seconds > 0.0:
            total_seconds += seconds
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


def classify_bottlenecks(
    row: JsonMapping,
    *,
    thresholds: BottleneckThresholds = DEFAULT_BOTTLENECK_THRESHOLDS,
) -> list[BottleneckLabel]:
    bottlenecks: list[BottleneckLabel] = []
    if _is_proposal_too_slow(row, thresholds=thresholds):
        bottlenecks.append("proposal_too_slow")
    if _is_verifier_too_slow(row, thresholds=thresholds):
        bottlenecks.append("verifier_too_slow")
    return bottlenecks
