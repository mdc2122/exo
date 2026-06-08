from __future__ import annotations

import json
from pathlib import Path

from scripts import mimo_mtp_benchmark_ingest as ingest


def _jsonl(rows: list[dict[str, object]]) -> str:
    return "\n".join(json.dumps(row, sort_keys=True) for row in rows)


def test_ingest_ar_baseline_row_parses_explicit_canonical_metadata() -> None:
    result = ingest.ingest_benchmark_jsonl(
        _jsonl(
            [
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "mode_label": "ar",
                    "model_id": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 2,
                    "generation_tps": 23.5,
                    "generation_tokens": 16,
                }
            ]
        )
    )

    assert result.errors == []
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.model_id == "kernelpool/MiMo-V2.5-Pro-6bit"
    assert row.model == "kernelpool/MiMo-V2.5-Pro-6bit"
    assert row.mode_label == "ar"
    assert row.mode == "ar"
    assert row.repeat_index == 2
    assert row.raw["model_id"] == "kernelpool/MiMo-V2.5-Pro-6bit"
    assert row.raw["model"] == "kernelpool/MiMo-V2.5-Pro-6bit"
    assert row.raw["mode_label"] == "ar"
    assert row.raw["mode"] == "ar"


def test_ingest_ar_baseline_row_defaults_canonical_metadata_from_legacy_fields() -> (
    None
):
    result = ingest.ingest_benchmark_jsonl(
        _jsonl(
            [
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "mode": "ar",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 2,
                    "generation_tps": 23.5,
                    "generation_tokens": 16,
                }
            ]
        )
    )

    assert result.errors == []
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.model_id == "kernelpool/MiMo-V2.5-Pro-6bit"
    assert row.model == "kernelpool/MiMo-V2.5-Pro-6bit"
    assert row.mode_label == "ar"
    assert row.mode == "ar"
    assert row.repeat_index == 2
    assert row.raw["model_id"] == "kernelpool/MiMo-V2.5-Pro-6bit"
    assert row.raw["model"] == "kernelpool/MiMo-V2.5-Pro-6bit"
    assert row.raw["mode_label"] == "ar"
    assert row.raw["mode"] == "ar"


def test_ingest_ar_baseline_row_reports_invalid_or_conflicting_metadata() -> None:
    result = ingest.ingest_benchmark_jsonl(
        "\n".join(
            [
                json.dumps(
                    {
                        "kind": "cluster_benchmark_metric",
                        "cluster_path": "exo_api_bench_chat_completions",
                        "mode_label": 123,
                        "model_id": "kernelpool/MiMo-V2.5-Pro-6bit",
                        "repeat_index": 0,
                        "generation_tps": 23.5,
                        "generation_tokens": 16,
                    }
                ),
                json.dumps(
                    {
                        "kind": "cluster_benchmark_metric",
                        "cluster_path": "exo_api_bench_chat_completions",
                        "mode_label": "ar",
                        "model_id": ["kernelpool/MiMo-V2.5-Pro-6bit"],
                        "repeat_index": 0,
                        "generation_tps": 23.5,
                        "generation_tokens": 16,
                    }
                ),
                json.dumps(
                    {
                        "kind": "cluster_benchmark_metric",
                        "cluster_path": "exo_api_bench_chat_completions",
                        "mode_label": "ar",
                        "model_id": "kernelpool/MiMo-V2.5-Pro-6bit",
                        "repeat_index": True,
                        "generation_tps": 23.5,
                        "generation_tokens": 16,
                    }
                ),
                json.dumps(
                    {
                        "kind": "cluster_benchmark_metric",
                        "cluster_path": "exo_api_bench_chat_completions",
                        "mode": "ar",
                        "mode_label": "mtp-d1",
                        "model_id": "kernelpool/MiMo-V2.5-Pro-6bit",
                        "repeat_index": 0,
                        "generation_tps": 23.5,
                        "generation_tokens": 16,
                    }
                ),
                json.dumps(
                    {
                        "kind": "cluster_benchmark_metric",
                        "cluster_path": "exo_api_bench_chat_completions",
                        "mode_label": "ar",
                        "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                        "model_id": "different/MiMo-V2.5-Pro-6bit",
                        "repeat_index": 0,
                        "generation_tps": 23.5,
                        "generation_tokens": 16,
                    }
                ),
                json.dumps(
                    {
                        "kind": "cluster_benchmark_metric",
                        "cluster_path": "exo_api_bench_chat_completions",
                        "mode_label": "ar",
                        "model_id": "kernelpool/MiMo-V2.5-Pro-6bit",
                        "repeat_index": 1,
                        "generation_tps": 24.0,
                        "generation_tokens": 16,
                    }
                ),
            ]
        )
    )

    assert [row.repeat_index for row in result.rows] == [1]
    assert [(error.line_number, error.reason) for error in result.errors] == [
        (1, "mode_label must be a string"),
        (2, "model_id must be a string"),
        (3, "repeat_index must be an integer"),
        (4, "mode and mode_label must match when both are present"),
        (5, "model and model_id must match when both are present"),
    ]


def test_ingest_ar_baseline_normalized_raw_preserves_payload_extra_keys() -> None:
    result = ingest.ingest_benchmark_jsonl(
        _jsonl(
            [
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "mode": "ar",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 3,
                    "generation_tps": 24.5,
                    "generation_tokens": 64,
                    "prompt_tps": 91.0,
                    "payload_extra_keys": ["diagnostic_flag"],
                },
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "mode": "ar",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 4,
                    "generation_tps": 25.5,
                    "generation_tokens": 64,
                    "prompt_tps": 92.0,
                },
            ]
        )
    )

    assert result.errors == []
    explicit_extra_row = result.rows[0]
    assert explicit_extra_row.payload_extra_keys == ("diagnostic_flag",)
    assert explicit_extra_row.raw["payload_extra_keys"] == ["diagnostic_flag"]
    assert explicit_extra_row.raw["model"] == "kernelpool/MiMo-V2.5-Pro-6bit"
    assert explicit_extra_row.raw["mode"] == "ar"
    assert explicit_extra_row.raw["repeat_index"] == 3
    assert explicit_extra_row.raw["generation_tps"] == 24.5
    assert explicit_extra_row.raw["generation_tokens"] == 64
    assert explicit_extra_row.raw["prompt_tps"] == 91.0

    default_extra_row = result.rows[1]
    assert default_extra_row.payload_extra_keys == ()
    assert default_extra_row.raw["payload_extra_keys"] == []
    assert default_extra_row.raw["model"] == "kernelpool/MiMo-V2.5-Pro-6bit"
    assert default_extra_row.raw["mode"] == "ar"
    assert default_extra_row.raw["repeat_index"] == 4
    assert default_extra_row.raw["generation_tps"] == 25.5
    assert default_extra_row.raw["generation_tokens"] == 64
    assert default_extra_row.raw["prompt_tps"] == 92.0


def test_ingest_ar_baseline_payload_extra_preserves_values() -> None:
    result = ingest.ingest_benchmark_jsonl(
        _jsonl(
            [
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "mode": "ar",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 0,
                    "generation_tps": 24.5,
                    "generation_tokens": 64,
                    "payload_extra_keys": ["diagnostic_flag", "run_tag"],
                    "payload_extra": {
                        "diagnostic_flag": True,
                        "run_tag": "same-cluster-ar-baseline",
                    },
                }
            ]
        )
    )

    assert result.errors == []
    row = result.rows[0]
    assert row.payload_extra == {
        "diagnostic_flag": True,
        "run_tag": "same-cluster-ar-baseline",
    }
    assert row.raw["payload_extra"] == {
        "diagnostic_flag": True,
        "run_tag": "same-cluster-ar-baseline",
    }


def test_ingest_ar_baseline_payload_extra_canonical_collisions_do_not_override_row_fields() -> (
    None
):
    result = ingest.ingest_benchmark_jsonl(
        _jsonl(
            [
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "mode": "ar",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 0,
                    "generation_tps": 24.5,
                    "generation_tokens": 64,
                    "payload_extra_keys": ["mode", "generation_tps", "run_tag"],
                    "payload_extra": {
                        "mode": "mtp-d999",
                        "generation_tps": 999.0,
                        "run_tag": "collision-probe",
                    },
                }
            ]
        )
    )

    assert result.errors == []
    row = result.rows[0]
    assert row.mode == "ar"
    assert row.generation_tps == 24.5
    assert row.payload_extra == {
        "mode": "mtp-d999",
        "generation_tps": 999.0,
        "run_tag": "collision-probe",
    }
    assert row.payload_extra_canonical_collision_keys == ("generation_tps", "mode")


def test_ingest_payload_extra_unrecognized_keys_survive_row_parsing() -> None:
    """Payload extra keys like custom_metadata and model_version survive
    ingestion and are accessible on the parsed IngestedBenchmarkRow."""
    result = ingest.ingest_benchmark_jsonl(
        _jsonl(
            [
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "mode": "ar",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 0,
                    "generation_tps": 24.5,
                    "generation_tokens": 64,
                    "payload_extra_keys": [
                        "custom_metadata",
                        "model_version",
                        "run_tag",
                    ],
                    "payload_extra": {
                        "custom_metadata": {"run_group": "nightly"},
                        "model_version": "6bit-v3",
                        "run_tag": "same-cluster-ar-baseline",
                    },
                }
            ]
        )
    )
    assert result.errors == []
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.payload_extra == {
        "custom_metadata": {"run_group": "nightly"},
        "model_version": "6bit-v3",
        "run_tag": "same-cluster-ar-baseline",
    }
    assert row.payload_extra_keys == (
        "custom_metadata",
        "model_version",
        "run_tag",
    )
    assert row.raw["payload_extra"] == {
        "custom_metadata": {"run_group": "nightly"},
        "model_version": "6bit-v3",
        "run_tag": "same-cluster-ar-baseline",
    }


def test_ingest_analyzer_payload_extra_unrecognized_keys_survive_row_parsing() -> (
    None
):
    """Unrecognized payload extra keys also survive the analyzer ingestion path
    which uses the canonical benchmark_row evidence_kind schema."""
    result = ingest.ingest_analyzer_benchmark_jsonl(
        _jsonl(
            [
                {
                    "evidence_kind": "benchmark_row",
                    "schema_version": "mimo_mtp_benchmark_row.v1",
                    "row_status": "live",
                    "benchmark_session_id": "session-20260607",
                    "api_url": "http://127.0.0.1:52415",
                    "endpoint": "/bench/chat/completions",
                    "model_id": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "prompt_id": "standard-ar-baseline",
                    "mode": "ar",
                    "repeat_index": 0,
                    "temperature": 0.0,
                    "max_tokens": 64,
                    "generation_tps": 24.5,
                    "generation_tokens": 64,
                    "mtp_enabled": False,
                    "mtp_depth": None,
                    "mtp_execution_state": "disabled_default",
                    "mtp_disable_reason": None,
                    "payload_extra": {
                        "custom_metadata": {"run_group": "nightly"},
                        "model_version": "6bit-v3",
                    },
                }
            ]
        )
    )
    assert result.errors == []
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.payload_extra == {
        "custom_metadata": {"run_group": "nightly"},
        "model_version": "6bit-v3",
    }
    assert row.payload_extra_keys == (
        "custom_metadata",
        "model_version",
    )


def test_ingest_blocked_benchmark_evidence_requires_non_fabricated_command_contract() -> (
    None
):
    result = ingest.ingest_benchmark_jsonl(
        _jsonl(
            [
                {
                    "schema_version": "mimo_mtp_benchmark_evidence/v1",
                    "evidence_kind": "blocked_benchmark_row",
                    "row_status": "blocked_with_command",
                    "benchmark_session_id": "session-20260607",
                    "api_url": "http://127.0.0.1:52415",
                    "endpoint": "/bench/chat/completions",
                    "model_id": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "mode": "mtp-d2",
                    "mtp_enabled": True,
                    "mtp_depth": 2,
                    "mtp_execution_state": "blocked_unavailable",
                    "mtp_disable_reason": "cluster_api_unavailable",
                    "commands": [
                        "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 16 --mode-label mtp-d2"
                    ],
                    "environment_assumptions": [
                        "exo cluster API is expected at http://127.0.0.1:52415",
                        "MiMo V2.5 Pro MTP sidecar path is configured before rerun",
                    ],
                    "output_paths": [
                        ".goose-ultrawork/evidence/sub-ac-2.1-blocked-mtp.jsonl"
                    ],
                    "non_fabrication_statement": "No performance result is claimed; this row records a blocked rerun command only.",
                    "blocker_reason": "cluster API unavailable in this execution environment",
                }
            ]
        )
    )

    assert result.errors == []
    assert result.rows == []
    assert len(result.blocked_rows) == 1
    blocked = result.blocked_rows[0]
    assert blocked.row_status == "blocked_with_command"
    assert blocked.mtp_disable_reason == "cluster_api_unavailable"
    assert blocked.commands == (
        "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 16 --mode-label mtp-d2",
    )
    assert blocked.environment_assumptions == (
        "exo cluster API is expected at http://127.0.0.1:52415",
        "MiMo V2.5 Pro MTP sidecar path is configured before rerun",
    )
    assert blocked.output_paths == (
        ".goose-ultrawork/evidence/sub-ac-2.1-blocked-mtp.jsonl",
    )
    assert blocked.non_fabrication_statement.startswith(
        "No performance result is claimed"
    )
    assert result.summary["blocked_row_count"] == 1
    assert result.summary["status"] == "blocked_no_metric_rows"


def test_ingest_rejects_blocked_benchmark_evidence_with_missing_required_contract_fields() -> (
    None
):
    result = ingest.ingest_benchmark_jsonl(
        "\n".join(
            [
                json.dumps(
                    {
                        "evidence_kind": "blocked_benchmark_row",
                        "row_status": "blocked_with_command",
                        "mtp_disable_reason": "cluster_api_unavailable",
                        "commands": [
                            "uv run python3 scripts/bench_mimo_mtp_cluster.py"
                        ],
                        "environment_assumptions": ["cluster API unavailable"],
                        "output_paths": [".goose-ultrawork/evidence/blocked.jsonl"],
                    }
                ),
                json.dumps(
                    {
                        "evidence_kind": "blocked_benchmark_row",
                        "row_status": "blocked_with_command",
                        "mtp_disable_reason": "cluster_api_unavailable",
                        "commands": [],
                        "environment_assumptions": ["cluster API unavailable"],
                        "output_paths": [".goose-ultrawork/evidence/blocked.jsonl"],
                        "non_fabrication_statement": "No performance result is claimed.",
                    }
                ),
                json.dumps(
                    {
                        "evidence_kind": "blocked_benchmark_row",
                        "row_status": "blocked_with_command",
                        "mtp_disable_reason": "cluster_api_unavailable",
                        "commands": [
                            "uv run python3 scripts/bench_mimo_mtp_cluster.py"
                        ],
                        "environment_assumptions": [],
                        "output_paths": [".goose-ultrawork/evidence/blocked.jsonl"],
                        "non_fabrication_statement": "No performance result is claimed.",
                    }
                ),
                json.dumps(
                    {
                        "evidence_kind": "blocked_benchmark_row",
                        "row_status": "blocked_with_command",
                        "mtp_disable_reason": "cluster_api_unavailable",
                        "commands": [
                            "uv run python3 scripts/bench_mimo_mtp_cluster.py"
                        ],
                        "environment_assumptions": ["cluster API unavailable"],
                        "output_paths": [],
                        "non_fabrication_statement": "No performance result is claimed.",
                    }
                ),
                json.dumps(
                    {
                        "evidence_kind": "blocked_benchmark_row",
                        "row_status": "blocked_with_command",
                        "mtp_disable_reason": "cluster_api_unavailable",
                        "commands": [
                            "uv run python3 scripts/bench_mimo_mtp_cluster.py"
                        ],
                        "environment_assumptions": ["cluster API unavailable"],
                        "output_paths": [".goose-ultrawork/evidence/blocked.jsonl"],
                        "non_fabrication_statement": "rerun later",
                    }
                ),
            ]
        )
    )

    assert result.blocked_rows == []
    assert [(error.line_number, error.reason) for error in result.errors] == [
        (1, "non_fabrication_statement must be a string for blocked rows"),
        (2, "commands must be a non-empty string list for blocked rows"),
        (3, "environment_assumptions must be a non-empty string list for blocked rows"),
        (4, "output_paths must be a non-empty string list for blocked rows"),
        (
            5,
            "non_fabrication_statement must explicitly state that no performance result is claimed",
        ),
    ]


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


def test_ingest_ar_only_rows_are_valid_when_mtp_rows_are_missing() -> None:
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
                    "mode": "ar",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 1,
                    "generation_tps": 23.0,
                    "generation_tokens": 16,
                },
            ]
        )
    )

    assert result.errors == []
    assert [row.telemetry_mode for row in result.rows] == ["ar", "ar"]
    assert result.ar_rows == result.rows
    assert result.mtp_rows == []
    assert result.live_mtp_rows == []
    assert result.summary["ar_row_count"] == 2
    assert result.summary["mtp_row_count"] == 0
    assert result.summary["live_mtp_row_count"] == 0
    assert result.summary["median_ar_generation_tps"] == 22.0
    assert result.summary["median_live_mtp_generation_tps"] is None
    assert result.summary["status"] == "blocked_no_mtp_rows"


def test_ingest_ar_baseline_normalizes_nested_generation_stats_metrics() -> None:
    result = ingest.ingest_benchmark_jsonl(
        _jsonl(
            [
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "mode": "ar",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 0,
                    "generation_stats": {
                        "generation_tps": 27.25,
                        "generation_tokens": 16,
                        "prompt_tps": 91.5,
                    },
                    "power_usage": {
                        "elapsed_seconds": 1.0,
                        "nodes": [],
                        "total_avg_sys_power_watts": 42.0,
                        "total_energy_joules": 42.0,
                    },
                }
            ]
        )
    )

    assert result.errors == []
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.generation_tps == 27.25
    assert row.generation_tokens == 16
    assert row.optional_prompt_tps == 91.5
    assert row.optional_power_usage == {
        "elapsed_seconds": 1.0,
        "nodes": [],
        "total_avg_sys_power_watts": 42.0,
        "total_energy_joules": 42.0,
    }
    assert row.raw["generation_tps"] == 27.25
    assert row.raw["generation_tokens"] == 16
    assert row.raw["prompt_tps"] == 91.5
    assert row.raw["power_usage"] == row.optional_power_usage


def test_ingest_ar_baseline_metric_normalization_tolerates_missing_optional_metrics() -> (
    None
):
    result = ingest.ingest_benchmark_jsonl(
        _jsonl(
            [
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "mode": "ar",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 0,
                    "generation_stats": {
                        "generation_tps": 28.0,
                        "generation_tokens": 16,
                    },
                }
            ]
        )
    )

    assert result.errors == []
    row = result.rows[0]
    assert row.generation_tps == 28.0
    assert row.generation_tokens == 16
    assert row.optional_prompt_tps is None
    assert row.optional_power_usage is None
    assert row.raw["generation_tps"] == 28.0
    assert row.raw["generation_tokens"] == 16
    assert "prompt_tps" not in row.raw
    assert "power_usage" not in row.raw


def test_ingest_ar_baseline_metric_normalization_coerces_numeric_strings() -> None:
    result = ingest.ingest_benchmark_jsonl(
        _jsonl(
            [
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "mode": "ar",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 0,
                    "generation_stats": {
                        "generation_tps": "29.75",
                        "generation_tokens": "16",
                        "prompt_tps": "92.25",
                    },
                    "power_usage": {
                        "elapsed_seconds": "1.5",
                        "nodes": [],
                        "total_avg_sys_power_watts": "44.0",
                        "total_energy_joules": "66.0",
                    },
                }
            ]
        )
    )

    assert result.errors == []
    row = result.rows[0]
    assert row.generation_tps == 29.75
    assert row.generation_tokens == 16
    assert row.optional_prompt_tps == 92.25
    assert row.optional_power_usage == {
        "elapsed_seconds": 1.5,
        "nodes": [],
        "total_avg_sys_power_watts": 44.0,
        "total_energy_joules": 66.0,
    }
    assert row.raw["generation_tps"] == 29.75
    assert row.raw["generation_tokens"] == 16
    assert row.raw["prompt_tps"] == 92.25
    assert row.raw["power_usage"] == row.optional_power_usage


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


def test_parsed_budget_metrics_compute_complete_ar_and_mtp_telemetry() -> None:
    result = ingest.ingest_benchmark_jsonl(
        _jsonl(
            [
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "mode": "ar",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 0,
                    "generation_tps": 20.0,
                    "generation_tokens": 16,
                },
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "mode": "ar",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 1,
                    "generation_tps": 22.0,
                    "generation_tokens": 16,
                },
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "accepted_execution_path": "mimo_mtp_fastpath",
                    "mode": "mtp-d2",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 0,
                    "generation_tps": 34.0,
                    "generation_tokens": 16,
                },
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "accepted_execution_path": "mimo_mtp_fastpath",
                    "mode": "mtp-d2",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 1,
                    "generation_tps": 36.0,
                    "generation_tokens": 16,
                },
            ]
        )
    )

    budget = ingest.calculate_parsed_mtp_speedup_budget(result.rows)

    assert budget["kind"] == "mimo_mtp_parsed_speedup_budget"
    assert budget["ar_row_count"] == 2
    assert budget["mtp_row_count"] == 2
    assert budget["ar_median_tok_s"] == 21.0
    assert budget["ar_ms_per_token"] == 47.619048
    assert budget["mtp_median_tok_s"] == 35.0
    assert budget["mtp_target_gap_to_30_tok_s"] == 0.0
    assert budget["mtp_target_gap_to_40_tok_s"] == 5.0
    assert budget["mtp_vs_ar_speedup_ratio"] is None
    assert budget["status"] == "ambiguous_non_comparable_rows"
    assert budget["comparability_status"] == "ambiguous"
    assert set(budget["comparability_blockers"]) == {
        "api_url_or_cluster_id",
        "benchmark_session_id",
        "endpoint",
        "max_tokens",
        "prompt_id_or_prompt_hash",
        "temperature",
    }


def test_parsed_budget_metrics_keep_mtp_fields_null_when_mtp_telemetry_absent() -> None:
    result = ingest.ingest_benchmark_jsonl(
        _jsonl(
            [
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "mode": "ar",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 0,
                    "generation_tps": 20.0,
                    "generation_tokens": 16,
                },
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "mode": "ar",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 1,
                    "generation_tps": 22.0,
                    "generation_tokens": 16,
                },
            ]
        )
    )

    budget = ingest.calculate_parsed_mtp_speedup_budget(result.rows)

    assert budget["kind"] == "mimo_mtp_parsed_speedup_budget"
    assert budget["ar_row_count"] == 2
    assert budget["mtp_row_count"] == 0
    assert budget["ar_median_tok_s"] == 21.0
    assert budget["ar_ms_per_token"] == 47.619048
    assert budget["mtp_median_tok_s"] is None
    assert budget["mtp_target_gap_to_30_tok_s"] is None
    assert budget["mtp_target_gap_to_40_tok_s"] is None
    assert budget["mtp_vs_ar_speedup_ratio"] is None
    assert budget["status"] == "blocked_missing_mtp_rows"
    assert budget["comparability_status"] == "insufficient_data"
    assert budget["same_cluster_comparison_key"] is None
    assert (
        budget["next_step"]
        == "collect same-cluster guarded MTP rows before making speedup or target claims"
    )


def _canonical_speedup_row(
    *,
    mode: str,
    repeat_index: int,
    generation_tps: float,
    api_url: str = "http://127.0.0.1:52415",
    benchmark_session_id: str = "session-20260607",
    endpoint: str = "/bench/chat/completions",
    model_id: str = "kernelpool/MiMo-V2.5-Pro-6bit",
    prompt_hash: str = "sha256:prompt-a",
    temperature: float = 0.0,
    max_tokens: int = 64,
    mtp_execution_state: str | None = None,
) -> dict[str, object]:
    execution_state = mtp_execution_state or (
        "successful_mtp" if mode == "mtp" else "disabled_default"
    )
    return {
        "schema_version": "mimo_mtp_benchmark_evidence/v1",
        "evidence_kind": "benchmark_row",
        "row_status": "ok",
        "benchmark_session_id": benchmark_session_id,
        "api_url": api_url,
        "endpoint": endpoint,
        "model_id": model_id,
        "prompt_hash": prompt_hash,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "mode": mode,
        "repeat_index": repeat_index,
        "generation_tps": generation_tps,
        "generation_tokens": max_tokens,
        "prompt_tps": 90.0,
        "power_usage": None,
        "payload_extra": {"repeat_policy": "n=2"},
        "mtp_enabled": mode == "mtp",
        "mtp_depth": 2 if mode == "mtp" else None,
        "mtp_execution_state": execution_state,
        "mtp_disable_reason": "" if mode == "mtp" else "default_ar_request",
        "telemetry_completeness": "complete",
    }


def test_parsed_budget_computes_speedup_only_for_same_cluster_canonical_rows() -> None:
    result = ingest.ingest_analyzer_benchmark_jsonl(
        _jsonl(
            [
                _canonical_speedup_row(mode="ar", repeat_index=0, generation_tps=20.0),
                _canonical_speedup_row(mode="ar", repeat_index=1, generation_tps=22.0),
                _canonical_speedup_row(mode="mtp", repeat_index=0, generation_tps=34.0),
                _canonical_speedup_row(mode="mtp", repeat_index=1, generation_tps=36.0),
            ]
        )
    )

    budget = ingest.calculate_parsed_mtp_speedup_budget(result.rows)

    assert result.errors == []
    assert budget["status"] == "same_cluster_budget_ready"
    assert budget["comparability_status"] == "same_cluster_comparable"
    assert budget["same_cluster_comparison_key"] == {
        "api_url": "http://127.0.0.1:52415",
        "cluster_id": None,
        "endpoint": "/bench/chat/completions",
        "model_id": "kernelpool/MiMo-V2.5-Pro-6bit",
        "prompt_id": None,
        "prompt_hash": "sha256:prompt-a",
        "temperature": 0.0,
        "max_tokens": 64,
        "benchmark_session_id": "session-20260607",
        "repeat_policy": "observed_repeat_count=2",
        "node_topology": None,
    }
    assert budget["ar_median_tok_s"] == 21.0
    assert budget["mtp_median_tok_s"] == 35.0
    assert budget["mtp_vs_ar_speedup_ratio"] == 1.666667


def test_parsed_budget_makes_no_speedup_claim_for_ambiguous_non_comparable_rows() -> (
    None
):
    result = ingest.ingest_analyzer_benchmark_jsonl(
        _jsonl(
            [
                _canonical_speedup_row(mode="ar", repeat_index=0, generation_tps=20.0),
                _canonical_speedup_row(
                    mode="mtp",
                    repeat_index=0,
                    generation_tps=36.0,
                    api_url="http://127.0.0.1:52416",
                ),
            ]
        )
    )

    budget = ingest.calculate_parsed_mtp_speedup_budget(result.rows)

    assert result.errors == []
    assert budget["status"] == "ambiguous_non_comparable_rows"
    assert budget["comparability_status"] == "ambiguous"
    assert budget["mtp_vs_ar_speedup_ratio"] is None
    assert "api_url" in budget["comparability_blockers"]
    assert "no MTP speedup claim" in budget["next_step"]


def test_parsed_budget_makes_no_speedup_claim_for_blocked_evidence_inputs() -> None:
    dataset = ingest.parse_benchmark_telemetry_dataset_jsonl(
        _jsonl(
            [
                _canonical_speedup_row(mode="ar", repeat_index=0, generation_tps=20.0),
                {
                    "evidence_kind": "blocked_benchmark_row",
                    "row_status": "blocked_unavailable",
                    "mtp_disable_reason": "cluster API unavailable",
                    "commands": [
                        "python3 scripts/bench_mimo_mtp_cluster.py --api-url http://127.0.0.1:52415 --mode mtp"
                    ],
                    "environment_assumptions": [
                        "same cluster URL expected when available"
                    ],
                    "output_paths": [".goose-ultrawork/evidence/sub-ac-2.4-mtp.jsonl"],
                    "non_fabrication_statement": "No performance result is claimed; blocked evidence only.",
                },
            ]
        )
    )

    assert dataset.errors == ()
    assert dataset.speedup_budget["status"] == "blocked_missing_mtp_rows"
    assert dataset.speedup_budget["comparability_status"] == "insufficient_data"
    assert dataset.speedup_budget["mtp_vs_ar_speedup_ratio"] is None


def test_parsed_telemetry_dataset_exposes_ar_only_rows_when_live_mtp_is_absent() -> (
    None
):
    dataset = ingest.parse_benchmark_telemetry_dataset_jsonl(
        _jsonl(
            [
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "mode": "ar",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 0,
                    "generation_tps": 20.0,
                    "generation_tokens": 16,
                },
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "mode": "ar",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 1,
                    "generation_tps": 22.0,
                    "generation_tokens": 16,
                },
            ]
        )
    )

    assert dataset.kind == "ar_only"
    assert [row.repeat_index for row in dataset.ar_rows] == [0, 1]
    assert dataset.mtp_rows == ()
    assert dataset.live_mtp_rows == ()
    assert dataset.has_live_mtp_rows is False
    assert dataset.speedup_budget["status"] == "blocked_missing_mtp_rows"
    assert dataset.summary["status"] == "blocked_no_mtp_rows"


def test_parsed_telemetry_dataset_exposes_ar_plus_mtp_rows_when_live_mtp_exists() -> (
    None
):
    dataset = ingest.parse_benchmark_telemetry_dataset_jsonl(
        _jsonl(
            [
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "mode": "ar",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 0,
                    "generation_tps": 20.0,
                    "generation_tokens": 16,
                },
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "mode": "ar",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 1,
                    "generation_tps": 22.0,
                    "generation_tokens": 16,
                },
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "accepted_execution_path": "mimo_mtp_fastpath",
                    "mode": "mtp-d2",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 0,
                    "generation_tps": 34.0,
                    "generation_tokens": 16,
                },
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "accepted_execution_path": "mimo_mtp_fastpath",
                    "mode": "mtp-d2",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 1,
                    "generation_tps": 36.0,
                    "generation_tokens": 16,
                },
            ]
        )
    )

    assert dataset.kind == "ar_plus_mtp"
    assert len(dataset.ar_rows) == 2
    assert [row.generation_tps for row in dataset.live_mtp_rows] == [34.0, 36.0]
    assert dataset.mtp_rows == dataset.live_mtp_rows
    assert dataset.has_live_mtp_rows is True
    assert dataset.speedup_budget["status"] == "ambiguous_non_comparable_rows"
    assert dataset.speedup_budget["comparability_status"] == "ambiguous"
    assert dataset.speedup_budget["mtp_vs_ar_speedup_ratio"] is None


def test_analyzer_jsonl_ingestion_parses_canonical_rows_and_reports_line_errors() -> (
    None
):
    result = ingest.ingest_analyzer_benchmark_jsonl(
        "\n".join(
            [
                json.dumps(
                    {
                        "schema_version": "mimo_mtp_benchmark_evidence/v1",
                        "evidence_kind": "benchmark_row",
                        "row_status": "ok",
                        "benchmark_session_id": "session-20260607",
                        "api_url": "http://127.0.0.1:52415",
                        "endpoint": "/bench/chat/completions",
                        "model_id": "kernelpool/MiMo-V2.5-Pro-6bit",
                        "prompt_hash": "sha256:abc123",
                        "temperature": 0.0,
                        "max_tokens": 64,
                        "mode": "ar",
                        "repeat_index": 0,
                        "generation_tps": 22.5,
                        "generation_tokens": 64,
                        "prompt_tps": 88.0,
                        "power_usage": {"elapsed_seconds": "1.25"},
                        "payload_extra": {"run_tag": "ar-baseline"},
                        "mtp_enabled": False,
                        "mtp_depth": None,
                        "mtp_execution_state": "disabled_default",
                        "mtp_disable_reason": "default_ar_request",
                        "telemetry_completeness": "complete",
                    }
                ),
                "not json",
                json.dumps(["not", "an", "object"]),
                json.dumps(
                    {
                        "schema_version": "mimo_mtp_benchmark_evidence/v1",
                        "evidence_kind": "benchmark_row",
                        "row_status": "ok",
                        "benchmark_session_id": "session-20260607",
                        "api_url": "http://127.0.0.1:52415",
                        "endpoint": "/bench/chat/completions",
                        "model_id": "kernelpool/MiMo-V2.5-Pro-6bit",
                        "prompt_hash": "sha256:abc123",
                        "temperature": 0.0,
                        "max_tokens": 64,
                        "mode": "mtp",
                        "repeat_index": 1,
                        "generation_tps": 30.0,
                        "generation_tokens": 64,
                        "mtp_enabled": True,
                        "mtp_depth": 2,
                        "mtp_execution_state": "not_a_state",
                        "telemetry_completeness": "partial",
                    }
                ),
                json.dumps(
                    {
                        "schema_version": "mimo_mtp_benchmark_evidence/v1",
                        "evidence_kind": "benchmark_row",
                        "row_status": "ok",
                        "benchmark_session_id": "session-20260607",
                        "api_url": "http://127.0.0.1:52415",
                        "endpoint": "/bench/chat/completions",
                        "model_id": "kernelpool/MiMo-V2.5-Pro-6bit",
                        "prompt_hash": "sha256:def456",
                        "temperature": 0.0,
                        "max_tokens": 64,
                        "mode": "mtp",
                        "repeat_index": 2,
                        "generation_tps": 35.0,
                        "generation_tokens": 64,
                        "prompt_tps": 90.0,
                        "payload_extra": {"mimo_mtp_fastpath": True},
                        "mtp_enabled": True,
                        "mtp_depth": 2,
                        "mtp_execution_state": "successful_mtp",
                        "telemetry_completeness": "complete",
                    }
                ),
            ]
        )
    )

    assert [row.line_number for row in result.rows] == [1, 5]
    assert [(error.line_number, error.reason) for error in result.errors] == [
        (2, "line is not valid JSON"),
        (3, "line must decode to a JSON object"),
        (4, "mtp_execution_state must be a recognized MTP execution state"),
    ]
    ar_row = result.rows[0]
    assert ar_row.telemetry_mode == "ar"
    assert ar_row.is_live_mtp is False
    assert ar_row.mode == "ar"
    assert ar_row.mode_label == "ar"
    assert ar_row.model_id == "kernelpool/MiMo-V2.5-Pro-6bit"
    assert ar_row.cluster_path == "exo_api_bench_chat_completions"
    assert ar_row.optional_prompt_tps == 88.0
    assert ar_row.optional_power_usage == {"elapsed_seconds": 1.25}
    assert ar_row.raw["schema_version"] == "mimo_mtp_benchmark_evidence/v1"
    assert ar_row.raw["evidence_kind"] == "benchmark_row"
    assert ar_row.raw["endpoint"] == "/bench/chat/completions"
    assert ar_row.raw["mtp_enabled"] is False
    assert ar_row.raw["mtp_execution_state"] == "disabled_default"

    mtp_row = result.rows[1]
    assert mtp_row.telemetry_mode == "mtp"
    assert mtp_row.is_live_mtp is True
    assert mtp_row.accepted_execution_path == "mimo_mtp_fastpath"
    assert mtp_row.raw["mode"] == "mtp"
    assert mtp_row.raw["mode_label"] == "mtp"
    assert mtp_row.raw["mtp_depth"] == 2


def test_analyzer_jsonl_ingestion_does_not_make_evidence_sufficiency_decisions() -> (
    None
):
    result = ingest.ingest_analyzer_benchmark_jsonl(
        _jsonl(
            [
                {
                    "schema_version": "mimo_mtp_benchmark_evidence/v1",
                    "evidence_kind": "benchmark_row",
                    "row_status": "ok",
                    "benchmark_session_id": "session-20260607",
                    "api_url": "http://127.0.0.1:52415",
                    "endpoint": "/bench/chat/completions",
                    "model_id": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "prompt_hash": "sha256:abc123",
                    "temperature": 0.0,
                    "max_tokens": 64,
                    "mode": "ar",
                    "repeat_index": 0,
                    "generation_tps": 22.5,
                    "generation_tokens": 64,
                    "mtp_enabled": False,
                    "mtp_execution_state": "disabled_default",
                    "telemetry_completeness": "complete",
                }
            ]
        )
    )

    assert result.errors == []
    assert len(result.rows) == 1
    assert result.summary == {
        "valid_row_count": 1,
        "error_count": 0,
        "blocked_row_count": 0,
    }
    assert not hasattr(result, "speedup_budget")
    assert "status" not in result.summary


def test_bottleneck_classifier_emits_all_performance_regression_labels_only_with_required_telemetry() -> None:
    result = ingest.ingest_benchmark_jsonl(
        _jsonl(
            [
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "live_execution_path": "exo_cluster_tensor_parallel",
                    "accepted_execution_path": "mimo_mtp_fastpath",
                    "mode": "mtp-d4",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 0,
                    "generation_tps": 24.0,
                    "generation_tokens": 16,
                    "timing_breakdown_seconds": {
                        "proposal": 2.0,
                        "verification": 4.0,
                        "acceptance": 0.1,
                        "fallback": 0.1,
                    },
                    "generation_stats": {
                        "mimo_mtp_attempted_tokens": 20,
                        "mimo_mtp_accepted_tokens": 4,
                        "mimo_mtp_proposed_tokens": 8,
                        "mimo_mtp_fallback_count": 6,
                        "mimo_mtp_depth": 4,
                    },
                },
                {
                    "kind": "cluster_benchmark_metric",
                    "cluster_path": "exo_api_bench_chat_completions",
                    "live_execution_path": "exo_cluster_tensor_parallel",
                    "accepted_execution_path": "mimo_mtp_fastpath",
                    "mode": "mtp-d4",
                    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                    "repeat_index": 1,
                    "generation_tps": 24.0,
                    "generation_tokens": 16,
                },
            ]
        ),
        classifier_config=ingest.BottleneckClassifierConfig(
            low_acceptance_rate_threshold=0.30,
            slow_proposal_tokens_per_second=5.0,
            slow_verifier_tokens_per_second=5.0,
            high_fallback_rate_threshold=0.25,
            aggressive_rollout_depth_threshold=4,
        ),
    )

    assert result.errors == []
    assert result.rows[0].bottleneck_classifications == [
        "acceptance_rate_low",
        "proposal_too_slow",
        "verifier_too_slow",
        "fallback_too_high",
        "depth_too_aggressive",
    ]
    assert result.rows[1].bottleneck_classifications == []
    assert result.summary["bottleneck_classifications"] == [
        "acceptance_rate_low",
        "proposal_too_slow",
        "verifier_too_slow",
        "fallback_too_high",
        "depth_too_aggressive",
    ]
