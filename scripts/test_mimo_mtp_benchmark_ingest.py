from __future__ import annotations

import json
from pathlib import Path

from scripts import mimo_mtp_benchmark_ingest as ingest


def _jsonl(rows: list[dict[str, object]]) -> str:
    return "\n".join(json.dumps(row, sort_keys=True) for row in rows)


def test_ingest_valid_ar_and_live_mtp_rows_distinguishes_modes() -> None:
    result = ingest.ingest_benchmark_jsonl(
        _jsonl(
            [
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "live_execution_path": "exo_cluster_tensor_parallel",
                    "accepted_execution_path": "ar",
                    "mode": "ar",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 0,
                    "http_status": 200,
                    "generation_tps": 22.5,
                    "generation_tokens": 16,
                    "prompt_tps": 88.0,
                    "power_usage": {"elapsed_seconds": 1.0, "nodes": []},
                    "payload_extra_keys": [],
                },
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "live_execution_path": "exo_cluster_tensor_parallel",
                    "accepted_execution_path": "mimo_mtp_fastpath",
                    "mode": "mtp-d2",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 0,
                    "http_status": 200,
                    "generation_tps": 35.25,
                    "generation_tokens": 16,
                    "prompt_tps": 90.0,
                    "power_usage": {"elapsed_seconds": 1.0, "nodes": []},
                    "payload_extra_keys": ["mimo_mtp_depth", "mimo_mtp_fastpath"],
                    "generation_stats": {
                        "mimo_mtp_attempted_tokens": 12,
                        "mimo_mtp_accepted_tokens": 7,
                        "mimo_mtp_fallback_count": 0,
                    },
                },
            ]
        )
    )

    assert result.errors == []
    assert result.rows[0].telemetry_mode == "ar"
    assert result.rows[0].is_live_mtp is False
    assert result.rows[0].optional_prompt_tps == 88.0
    assert result.rows[0].optional_power_usage == {"elapsed_seconds": 1.0, "nodes": []}
    assert result.rows[1].telemetry_mode == "mtp"
    assert result.rows[1].is_live_mtp is True
    assert result.rows[1].accepted_execution_path == "mimo_mtp_fastpath"
    assert result.live_mtp_rows == [result.rows[1]]
    assert result.summary == {
        "valid_row_count": 2,
        "error_count": 0,
        "ar_row_count": 1,
        "mtp_row_count": 1,
        "live_mtp_row_count": 1,
        "max_generation_tps": 35.25,
        "median_ar_generation_tps": 22.5,
        "median_live_mtp_generation_tps": 35.25,
        "status": "ready_for_same_cluster_comparison",
    }


def test_ingest_reports_malformed_rows_without_dropping_other_valid_rows() -> None:
    result = ingest.ingest_benchmark_jsonl(
        "\n".join(
            [
                "not json",
                json.dumps(
                    {
                        "kind": "cluster_benchmark_metric",
                        "cluster_path": "exo_api_bench_chat_completions",
                        "mode": "ar",
                        "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                        "repeat_index": "zero",
                        "generation_tps": 22.5,
                        "generation_tokens": 16,
                    }
                ),
                json.dumps(
                    {
                        "kind": "cluster_benchmark_metric",
                        "cluster_path": "exo_api_bench_chat_completions",
                        "mode": "ar",
                        "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                        "repeat_index": 1,
                        "generation_tps": 23.0,
                        "generation_tokens": 16,
                    }
                ),
            ]
        )
    )

    assert [row.repeat_index for row in result.rows] == [1]
    assert [(error.line_number, error.reason) for error in result.errors] == [
        (1, "line is not valid JSON"),
        (2, "repeat_index must be an integer"),
    ]
    assert result.summary["valid_row_count"] == 1
    assert result.summary["error_count"] == 2


def test_ingest_tolerates_missing_optional_fields_and_no_live_mtp_rows() -> None:
    result = ingest.ingest_benchmark_jsonl(
        _jsonl(
            [
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "mode": "ar",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 0,
                    "generation_tps": 21.0,
                    "generation_tokens": 16,
                },
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "accepted_execution_path": "unknown",
                    "mode": "mtp-d1",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 0,
                    "generation_tps": 20.0,
                    "generation_tokens": 16,
                    "payload_extra_keys": ["mimo_mtp_fastpath"],
                },
            ]
        )
    )

    assert result.errors == []
    assert [row.telemetry_mode for row in result.rows] == ["ar", "mtp"]
    assert [row.is_live_mtp for row in result.rows] == [False, False]
    assert result.rows[0].optional_prompt_tps is None
    assert result.rows[0].optional_power_usage is None
    assert result.live_mtp_rows == []
    assert result.summary["live_mtp_row_count"] == 0
    assert result.summary["median_live_mtp_generation_tps"] is None
    assert result.summary["status"] == "blocked_no_live_mtp_rows"


def test_bottleneck_classifier_emits_acceptance_rate_low_below_configured_threshold() -> (
    None
):
    result = ingest.ingest_benchmark_jsonl(
        _jsonl(
            [
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "live_execution_path": "exo_cluster_tensor_parallel",
                    "accepted_execution_path": "mimo_mtp_fastpath",
                    "mode": "mtp-d2",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 0,
                    "generation_tps": 24.0,
                    "generation_tokens": 16,
                    "generation_stats": {
                        "mimo_mtp_attempted_tokens": 20,
                        "mimo_mtp_accepted_tokens": 4,
                    },
                }
            ]
        ),
        classifier_config=ingest.BottleneckClassifierConfig(
            low_acceptance_rate_threshold=0.30
        ),
    )

    assert result.errors == []
    assert result.rows[0].optional_acceptance_rate == 0.20
    assert result.rows[0].bottleneck_classifications == ["acceptance_rate_low"]
    assert result.summary["bottleneck_classifications"] == ["acceptance_rate_low"]


def test_bottleneck_classifier_accepts_explicit_acceptance_rate_telemetry() -> None:
    result = ingest.ingest_benchmark_jsonl(
        _jsonl(
            [
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "live_execution_path": "exo_cluster_tensor_parallel",
                    "accepted_execution_path": "mimo_mtp_fastpath",
                    "mode": "mtp-d1",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 0,
                    "generation_tps": 24.0,
                    "generation_tokens": 16,
                    "acceptance_rate": 0.25,
                }
            ]
        ),
        classifier_config=ingest.BottleneckClassifierConfig(
            low_acceptance_rate_threshold=0.30
        ),
    )

    assert result.errors == []
    assert result.rows[0].optional_acceptance_rate == 0.25
    assert result.rows[0].bottleneck_classifications == ["acceptance_rate_low"]


def test_bottleneck_classifier_suppresses_acceptance_rate_low_without_acceptance_telemetry() -> (
    None
):
    result = ingest.ingest_benchmark_jsonl(
        _jsonl(
            [
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "live_execution_path": "exo_cluster_tensor_parallel",
                    "accepted_execution_path": "mimo_mtp_fastpath",
                    "mode": "mtp-d2",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 0,
                    "generation_tps": 24.0,
                    "generation_tokens": 16,
                    "generation_stats": {
                        "mimo_mtp_fallback_count": 0,
                    },
                }
            ]
        ),
        classifier_config=ingest.BottleneckClassifierConfig(
            low_acceptance_rate_threshold=0.30
        ),
    )

    assert result.errors == []
    assert result.rows[0].optional_acceptance_rate is None
    assert "acceptance_rate_low" not in result.rows[0].bottleneck_classifications
    assert result.summary["bottleneck_classifications"] == []


def test_ingest_benchmark_jsonl_file_reads_jsonl_path(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "rows.jsonl"
    jsonl_path.write_text(
        _jsonl(
            [
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "mode": "ar",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 0,
                    "generation_tps": 21.0,
                    "generation_tokens": 16,
                }
            ]
        )
    )

    result = ingest.ingest_benchmark_jsonl_file(jsonl_path)

    assert len(result.rows) == 1
    assert result.rows[0].telemetry_mode == "ar"
