from __future__ import annotations

from scripts.mimo_mtp_bottleneck_classifier import (
    BottleneckThresholds,
    classify_bottlenecks,
    classify_budget_bottlenecks,
    classify_terminal_decision,
)


def test_classify_bottlenecks_emits_verifier_too_slow_when_verifier_share_crosses_threshold() -> (
    None
):
    bottlenecks = classify_bottlenecks(
        {
            "kind": "benchmark_metric",
            "mode": "d2",
            "generated_tokens": 8,
            "decode_seconds": 1.0,
            "timing_breakdown_seconds": {
                "proposal": 0.10,
                "verification": 0.62,
                "acceptance": 0.02,
                "fallback": 0.0,
            },
        },
        thresholds=BottleneckThresholds(slow_verifier_time_share=0.50),
    )

    assert bottlenecks == ["verifier_too_slow"]


def test_classify_bottlenecks_suppresses_verifier_too_slow_when_verifier_telemetry_is_missing() -> (
    None
):
    bottlenecks = classify_bottlenecks(
        {
            "kind": "benchmark_metric",
            "mode": "d2",
            "generated_tokens": 8,
            "decode_seconds": 1.0,
            "timing_breakdown_seconds": {
                "proposal": 0.10,
                "acceptance": 0.02,
                "fallback": 0.0,
            },
        },
        thresholds=BottleneckThresholds(slow_verifier_time_share=0.50),
    )

    assert bottlenecks == []


def test_classify_bottlenecks_emits_proposal_too_slow_when_proposal_throughput_crosses_threshold() -> (
    None
):
    bottlenecks = classify_bottlenecks(
        {
            "kind": "benchmark_metric",
            "mode": "d2",
            "generated_tokens": 8,
            "decode_seconds": 1.0,
            "timing_breakdown_seconds": {
                "proposal": 0.75,
                "verification": 0.10,
                "acceptance": 0.02,
                "fallback": 0.0,
            },
            "generation_stats": {
                "mimo_mtp_proposed_tokens": 8,
            },
        },
        thresholds=BottleneckThresholds(slow_proposal_tokens_per_second=12.0),
    )

    assert bottlenecks == ["proposal_too_slow"]


def test_classify_bottlenecks_suppresses_proposal_too_slow_when_proposal_telemetry_is_missing() -> (
    None
):
    bottlenecks = classify_bottlenecks(
        {
            "kind": "benchmark_metric",
            "mode": "d2",
            "generated_tokens": 8,
            "decode_seconds": 1.0,
            "timing_breakdown_seconds": {
                "verification": 0.10,
                "acceptance": 0.02,
                "fallback": 0.0,
            },
            "generation_stats": {
                "mimo_mtp_proposed_tokens": 8,
            },
        },
        thresholds=BottleneckThresholds(
            slow_verifier_time_share=1.0,
            slow_proposal_tokens_per_second=12.0,
        ),
    )

    assert bottlenecks == []


def test_classify_bottlenecks_emits_acceptance_rate_low_when_rate_crosses_threshold() -> (
    None
):
    bottlenecks = classify_bottlenecks(
        {
            "kind": "benchmark_metric",
            "mode": "mtp-d2",
            "generation_stats": {
                "mimo_mtp_attempted_tokens": 10,
                "mimo_mtp_accepted_tokens": 2,
            },
        },
        thresholds=BottleneckThresholds(low_acceptance_rate=0.30),
    )

    assert bottlenecks == ["acceptance_rate_low"]


def test_classify_bottlenecks_suppresses_acceptance_rate_low_without_acceptance_telemetry() -> (
    None
):
    bottlenecks = classify_bottlenecks(
        {
            "kind": "benchmark_metric",
            "mode": "mtp-d2",
            "generation_stats": {
                "mimo_mtp_proposed_tokens": 10,
            },
        },
        thresholds=BottleneckThresholds(low_acceptance_rate=0.30),
    )

    assert bottlenecks == []


def test_classify_bottlenecks_emits_fallback_too_high_when_rate_crosses_threshold() -> (
    None
):
    bottlenecks = classify_bottlenecks(
        {
            "kind": "benchmark_metric",
            "mode": "mtp-d2",
            "generation_stats": {
                "mimo_mtp_attempted_tokens": 10,
                "mimo_mtp_fallback_count": 3,
            },
        },
        thresholds=BottleneckThresholds(high_fallback_rate=0.25),
    )

    assert bottlenecks == ["fallback_too_high"]


def test_classify_bottlenecks_suppresses_fallback_too_high_without_fallback_telemetry() -> (
    None
):
    bottlenecks = classify_bottlenecks(
        {
            "kind": "benchmark_metric",
            "mode": "mtp-d2",
            "generation_stats": {
                "mimo_mtp_attempted_tokens": 10,
            },
        },
        thresholds=BottleneckThresholds(high_fallback_rate=0.25),
    )

    assert bottlenecks == []


def test_classify_bottlenecks_emits_depth_too_aggressive_when_depth_crosses_threshold() -> (
    None
):
    bottlenecks = classify_bottlenecks(
        {
            "kind": "benchmark_metric",
            "mode": "mtp-d4",
            "generation_stats": {
                "mimo_mtp_depth": 4,
            },
        },
        thresholds=BottleneckThresholds(aggressive_rollout_depth=4),
    )

    assert bottlenecks == ["depth_too_aggressive"]


def test_classify_bottlenecks_suppresses_depth_too_aggressive_without_depth_telemetry() -> (
    None
):
    bottlenecks = classify_bottlenecks(
        {
            "kind": "benchmark_metric",
            "mode": "mtp-d4",
        },
        thresholds=BottleneckThresholds(aggressive_rollout_depth=4),
    )

    assert bottlenecks == []


def test_classify_bottlenecks_suppresses_verifier_share_when_timing_breakdown_is_incomplete() -> (
    None
):
    bottlenecks = classify_bottlenecks(
        {
            "kind": "benchmark_metric",
            "mode": "mtp-d2",
            "generated_tokens": 8,
            "timing_breakdown_seconds": {
                "verification": 0.62,
            },
        },
        thresholds=BottleneckThresholds(slow_verifier_time_share=0.50),
    )

    assert bottlenecks == []


def test_classify_terminal_decision_maps_valid_comparable_speedup_to_pass() -> None:
    decision = classify_terminal_decision(
        {
            "input_validated": True,
            "same_cluster_comparable": True,
            "status": "same_cluster_budget_ready",
            "mtp_vs_ar_speedup_ratio": 1.15,
        }
    )

    assert decision == "pass"


def test_classify_terminal_decision_maps_valid_comparable_slowdown_to_fail() -> None:
    decision = classify_terminal_decision(
        {
            "input_validated": True,
            "same_cluster_comparable": True,
            "status": "same_cluster_budget_ready",
            "mtp_vs_ar_speedup_ratio": 1.0,
        }
    )

    assert decision == "fail"


def test_classify_terminal_decision_maps_comparability_violation_to_ambiguous() -> None:
    decision = classify_terminal_decision(
        {
            "input_validated": True,
            "same_cluster_comparable": False,
            "status": "same_cluster_budget_ready",
            "mtp_vs_ar_speedup_ratio": 1.50,
        }
    )

    assert decision == "ambiguous"


def test_classify_terminal_decision_maps_blocked_evidence_status_to_blocked() -> None:
    decision = classify_terminal_decision(
        {
            "input_validated": True,
            "same_cluster_comparable": True,
            "status": "blocked_missing_mtp_rows",
            "mtp_vs_ar_speedup_ratio": None,
        }
    )

    assert decision == "blocked"


def test_classify_terminal_decision_maps_missing_metrics_to_insufficient_data() -> None:
    decision = classify_terminal_decision(
        {
            "input_validated": True,
            "same_cluster_comparable": True,
            "status": "same_cluster_budget_ready",
            "mtp_vs_ar_speedup_ratio": None,
        }
    )

    assert decision == "insufficient_data"


def test_classify_terminal_decision_rejects_unvalidated_input_as_insufficient_data() -> (
    None
):
    decision = classify_terminal_decision(
        {
            "input_validated": False,
            "same_cluster_comparable": True,
            "status": "same_cluster_budget_ready",
            "mtp_vs_ar_speedup_ratio": 1.50,
        }
    )

    assert decision == "insufficient_data"


def test_classify_budget_bottlenecks_emits_mtp_beats_ar_when_same_cluster_speedup_exists() -> (
    None
):
    bottlenecks = classify_budget_bottlenecks(
        {
            "status": "same_cluster_budget_ready",
            "ar_row_count": 2,
            "mtp_row_count": 2,
            "ar_median_tok_s": 27.619048,
            "mtp_median_tok_s": 29.0,
            "mtp_vs_ar_speedup_ratio": 1.05,
        }
    )

    assert bottlenecks == ["mtp_beats_ar"]


def test_classify_budget_bottlenecks_emits_mtp_reaches_30_when_same_cluster_mtp_crosses_target() -> (
    None
):
    bottlenecks = classify_budget_bottlenecks(
        {
            "status": "same_cluster_budget_ready",
            "ar_row_count": 2,
            "mtp_row_count": 2,
            "ar_median_tok_s": 31.578947,
            "mtp_median_tok_s": 30.0,
            "mtp_vs_ar_speedup_ratio": 0.95,
        }
    )

    assert bottlenecks == ["mtp_reaches_30"]


def test_classify_budget_bottlenecks_emits_mtp_reaches_40_when_same_cluster_mtp_crosses_preferred_target() -> (
    None
):
    bottlenecks = classify_budget_bottlenecks(
        {
            "status": "same_cluster_budget_ready",
            "ar_row_count": 2,
            "mtp_row_count": 2,
            "ar_median_tok_s": 42.105263,
            "mtp_median_tok_s": 40.0,
            "mtp_vs_ar_speedup_ratio": 0.95,
        }
    )

    assert bottlenecks == ["mtp_reaches_30", "mtp_reaches_40"]


def test_classify_budget_bottlenecks_suppresses_throughput_labels_without_same_cluster_budget() -> (
    None
):
    bottlenecks = classify_budget_bottlenecks(
        {
            "status": "blocked_missing_ar_rows",
            "mtp_vs_ar_speedup_ratio": 1.50,
            "mtp_median_tok_s": 45.0,
        }
    )

    assert bottlenecks == []


def test_classify_budget_bottlenecks_suppresses_outcome_labels_without_ar_telemetry() -> (
    None
):
    bottlenecks = classify_budget_bottlenecks(
        {
            "status": "same_cluster_budget_ready",
            "ar_row_count": 0,
            "mtp_row_count": 2,
            "ar_median_tok_s": None,
            "mtp_median_tok_s": 45.0,
            "mtp_vs_ar_speedup_ratio": 1.50,
        }
    )

    assert bottlenecks == []


def test_classify_budget_bottlenecks_suppresses_outcome_labels_without_mtp_telemetry() -> (
    None
):
    bottlenecks = classify_budget_bottlenecks(
        {
            "status": "same_cluster_budget_ready",
            "ar_row_count": 2,
            "mtp_row_count": 0,
            "ar_median_tok_s": 20.0,
            "mtp_median_tok_s": None,
            "mtp_vs_ar_speedup_ratio": 1.50,
        }
    )

    assert bottlenecks == []
