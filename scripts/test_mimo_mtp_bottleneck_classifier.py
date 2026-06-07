from __future__ import annotations

from scripts.mimo_mtp_bottleneck_classifier import (
    BottleneckThresholds,
    classify_bottlenecks,
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
