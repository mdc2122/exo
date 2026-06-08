from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

from scripts import mimo_mtp_budget_calculator as calculator

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "scripts" / "fixtures"


def test_calculator_reports_ar_only_budget_without_fabricating_mtp_claims() -> None:
    fixture_path = FIXTURE_DIR / "mimo_mtp_budget_ar_only.jsonl"

    report = calculator.build_budget_report([fixture_path])

    assert report["kind"] == "mimo_mtp_budget_report"
    assert report["schema_version"] == 1
    assert report["source_files"] == [str(fixture_path)]
    assert report["ingestion"] == {
        "valid_row_count": 2,
        "error_count": 0,
        "ar_row_count": 2,
        "mtp_row_count": 0,
        "live_mtp_row_count": 0,
    }
    assert report["evidence_sufficiency"] == {
        "status": "insufficient_data",
        "reason": "absent_mtp_benchmark_evidence",
        "budget_status": "blocked_missing_mtp_rows",
        "valid_row_count": 2,
        "error_count": 0,
        "blocked_row_count": 0,
        "ar_row_count": 2,
        "mtp_row_count": 0,
        "live_mtp_row_count": 0,
        "performance_claim_allowed": False,
        "next_step": "collect same-cluster guarded MTP rows before making speedup or target claims",
    }
    assert report["budget"] == {
        "kind": "mimo_mtp_speedup_budget",
        "ar_row_count": 2,
        "mtp_row_count": 0,
        "ar_median_tok_s": 21.0,
        "ar_ms_per_token": 47.727273,
        "mtp_median_tok_s": None,
        "mtp_target_gap_to_30_tok_s": None,
        "mtp_target_gap_to_40_tok_s": None,
        "mtp_vs_ar_speedup_ratio": None,
        "status": "blocked_missing_mtp_rows",
        "next_step": "collect same-cluster guarded MTP rows before making speedup or target claims",
    }
    assert report["classifications"] == {
        "same_cluster_evidence": "blocked_missing_mtp_rows",
        "budget_classification": "ambiguous",
        "mtp_throughput_threshold": None,
        "slice_5_gate": "blocked_missing_mtp_rows",
        "slice_5_review_allowed": False,
        "slice_5_minimum_speedup_ratio": 1.15,
        "speedup_claim_allowed": False,
        "target_30_tok_s_claim_allowed": False,
        "target_40_tok_s_claim_allowed": False,
        "refusal_reasons": ["ar_only_evidence_no_guarded_live_mtp_rows"],
    }
    assert report["bottlenecks"] == ["missing_live_mtp_rows"]
    assert (
        cast(dict[str, object], report["optimization_loop"])["next_optimization_family"]
        is None
    )
    assert (
        report["next_step"]
        == "collect same-cluster guarded MTP rows before making speedup or target claims"
    )


def test_calculator_marks_mtp_labeled_rows_without_live_fastpath_as_ambiguous(
    tmp_path: Path,
) -> None:
    fixture_path = tmp_path / "ar-plus-non-live-mtp.jsonl"
    fixture_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "kind": "cluster_benchmark_metric",
                        "cluster_path": "exo_api_bench_chat_completions",
                        "live_execution_path": "exo_cluster_tensor_parallel",
                        "accepted_execution_path": "ar",
                        "mode": "ar",
                        "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                        "repeat_index": 0,
                        "http_status": 200,
                        "generation_tps": 20.0,
                        "generation_tokens": 16,
                        "payload_extra_keys": [],
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "kind": "cluster_benchmark_metric",
                        "cluster_path": "exo_api_bench_chat_completions",
                        "live_execution_path": "exo_cluster_tensor_parallel",
                        "accepted_execution_path": "ar_fallback",
                        "mode": "mtp-d1",
                        "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                        "repeat_index": 0,
                        "http_status": 200,
                        "generation_tps": 100.0,
                        "generation_tokens": 16,
                        "payload_extra_keys": ["mimo_mtp_fastpath"],
                    },
                    sort_keys=True,
                ),
            ]
        ),
        encoding="utf-8",
    )

    report = calculator.build_budget_report([fixture_path])

    assert report["ingestion"] == {
        "valid_row_count": 2,
        "error_count": 0,
        "ar_row_count": 1,
        "mtp_row_count": 1,
        "live_mtp_row_count": 0,
    }
    assert cast(dict[str, object], report["classifications"]) == {
        "same_cluster_evidence": "blocked_missing_live_mtp_rows",
        "budget_classification": "ambiguous",
        "mtp_throughput_threshold": None,
        "slice_5_gate": "blocked_missing_live_mtp_rows",
        "slice_5_review_allowed": False,
        "slice_5_minimum_speedup_ratio": 1.15,
        "speedup_claim_allowed": False,
        "target_30_tok_s_claim_allowed": False,
        "target_40_tok_s_claim_allowed": False,
        "refusal_reasons": ["mtp_labeled_rows_without_guarded_live_fastpath"],
    }
    assert report["bottlenecks"] == ["missing_live_mtp_rows"]


def test_calculator_uses_only_valid_comparable_live_mtp_rows_for_median_and_target_gaps(
    tmp_path: Path,
) -> None:
    fixture_path = tmp_path / "canonical-comparable-mtp-budget.jsonl"
    common_metadata = {
        "kind": "cluster_benchmark_metric",
        "cluster_path": "exo_api_bench_chat_completions",
        "live_execution_path": "exo_cluster_tensor_parallel",
        "api_url": "http://cluster-a.example",
        "benchmark_session_id": "session-a",
        "endpoint": "/bench/chat/completions",
        "model_id": "kernelpool/MiMo-V2.5-Pro-6bit",
        "prompt_hash": "prompt-a",
        "temperature": 0.0,
        "max_tokens": 64,
        "http_status": 200,
        "generation_tokens": 64,
    }
    rows = [
        {
            **common_metadata,
            "accepted_execution_path": "ar",
            "mode": "ar",
            "repeat_index": 0,
            "generation_tps": 20.0,
        },
        {
            **common_metadata,
            "accepted_execution_path": "mimo_mtp_fastpath",
            "mode": "mtp-d2",
            "repeat_index": 0,
            "generation_tps": 28.0,
        },
        {
            **common_metadata,
            "accepted_execution_path": "mimo_mtp_fastpath",
            "mode": "mtp-d2",
            "repeat_index": 1,
            "generation_tps": 32.0,
        },
        {
            **common_metadata,
            "api_url": "http://other-cluster.example",
            "accepted_execution_path": "mimo_mtp_fastpath",
            "mode": "mtp-d2",
            "repeat_index": 2,
            "generation_tps": 100.0,
        },
    ]
    fixture_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows),
        encoding="utf-8",
    )

    report = calculator.build_budget_report([fixture_path])

    assert cast(dict[str, object], report["budget"]) == {
        "kind": "mimo_mtp_speedup_budget",
        "ar_row_count": 1,
        "mtp_row_count": 2,
        "ar_median_tok_s": 20.0,
        "ar_ms_per_token": 50.0,
        "mtp_median_tok_s": 30.0,
        "mtp_target_gap_to_30_tok_s": 0.0,
        "mtp_target_gap_to_40_tok_s": 10.0,
        "mtp_vs_ar_speedup_ratio": 1.5,
        "status": "same_cluster_budget_ready",
        "next_step": "use this budget only with same-cluster AR-vs-MTP evidence rows; Slice 5 still requires a real MTP win without fallback concerns",
    }


def test_calculator_marks_live_mtp_rows_from_incomparable_cluster_as_ambiguous(
    tmp_path: Path,
) -> None:
    fixture_path = tmp_path / "incomparable-live-mtp.jsonl"
    common_metadata = {
        "kind": "cluster_benchmark_metric",
        "cluster_path": "exo_api_bench_chat_completions",
        "live_execution_path": "exo_cluster_tensor_parallel",
        "benchmark_session_id": "session-a",
        "endpoint": "/bench/chat/completions",
        "model_id": "kernelpool/MiMo-V2.5-Pro-6bit",
        "prompt_hash": "prompt-a",
        "temperature": 0.0,
        "max_tokens": 64,
        "http_status": 200,
        "generation_tokens": 64,
    }
    rows = [
        {
            **common_metadata,
            "api_url": "http://cluster-a.example",
            "accepted_execution_path": "ar",
            "mode": "ar",
            "repeat_index": 0,
            "generation_tps": 20.0,
        },
        {
            **common_metadata,
            "api_url": "http://cluster-b.example",
            "accepted_execution_path": "mimo_mtp_fastpath",
            "mode": "mtp-d2",
            "repeat_index": 0,
            "generation_tps": 100.0,
        },
    ]
    fixture_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows),
        encoding="utf-8",
    )

    report = calculator.build_budget_report([fixture_path])

    assert cast(dict[str, object], report["budget"]) == {
        "kind": "mimo_mtp_speedup_budget",
        "ar_row_count": 1,
        "mtp_row_count": 0,
        "ar_median_tok_s": 20.0,
        "ar_ms_per_token": 50.0,
        "mtp_median_tok_s": None,
        "mtp_target_gap_to_30_tok_s": None,
        "mtp_target_gap_to_40_tok_s": None,
        "mtp_vs_ar_speedup_ratio": None,
        "status": "ambiguous_non_comparable_cluster_rows",
        "next_step": "collect AR and guarded MTP rows with matching same-cluster comparison metadata before making speedup or target claims",
    }
    assert cast(dict[str, object], report["classifications"]) == {
        "same_cluster_evidence": "ambiguous_non_comparable_cluster_rows",
        "budget_classification": "ambiguous",
        "mtp_throughput_threshold": None,
        "slice_5_gate": "ambiguous_non_comparable_cluster_rows",
        "slice_5_review_allowed": False,
        "slice_5_minimum_speedup_ratio": 1.15,
        "speedup_claim_allowed": False,
        "target_30_tok_s_claim_allowed": False,
        "target_40_tok_s_claim_allowed": False,
        "refusal_reasons": ["non_comparable_cluster_evidence"],
    }
    assert report["evidence_sufficiency"] == {
        "status": "insufficient_data",
        "reason": "non_comparable_cluster_evidence",
        "budget_status": "ambiguous_non_comparable_cluster_rows",
        "valid_row_count": 2,
        "error_count": 0,
        "blocked_row_count": 0,
        "ar_row_count": 1,
        "mtp_row_count": 1,
        "live_mtp_row_count": 1,
        "performance_claim_allowed": False,
        "next_step": "collect AR and guarded MTP rows with matching same-cluster comparison metadata before making speedup or target claims",
    }


def test_calculator_reports_median_ar_ms_per_token_over_valid_comparable_ar_rows(
    tmp_path: Path,
) -> None:
    fixture_path = tmp_path / "ar-latency-median.jsonl"
    fixture_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "kind": "cluster_benchmark_metric",
                        "cluster_path": "exo_api_bench_chat_completions",
                        "api_url": "http://127.0.0.1:52415",
                        "endpoint": "/bench/chat/completions",
                        "model_id": "kernelpool/MiMo-V2.5-Pro-6bit",
                        "prompt_hash": "same-prompt",
                        "temperature": 0.0,
                        "max_tokens": 16,
                        "mode": "ar",
                        "mtp_enabled": False,
                        "generation_tps": 20.0,
                        "generation_tokens": 16,
                        "repeat_index": 0,
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "kind": "cluster_benchmark_metric",
                        "cluster_path": "exo_api_bench_chat_completions",
                        "api_url": "http://127.0.0.1:52415",
                        "endpoint": "/bench/chat/completions",
                        "model_id": "kernelpool/MiMo-V2.5-Pro-6bit",
                        "prompt_hash": "same-prompt",
                        "temperature": 0.0,
                        "max_tokens": 16,
                        "mode": "ar",
                        "mtp_enabled": False,
                        "generation_tps": 40.0,
                        "generation_tokens": 16,
                        "repeat_index": 1,
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "kind": "cluster_benchmark_metric",
                        "cluster_path": "exo_api_bench_chat_completions",
                        "api_url": "http://127.0.0.1:52415",
                        "endpoint": "/bench/chat/completions",
                        "model_id": "kernelpool/MiMo-V2.5-Pro-6bit",
                        "prompt_hash": "same-prompt",
                        "temperature": 0.0,
                        "max_tokens": 16,
                        "mode": "ar",
                        "mtp_enabled": False,
                        "generation_tps": 0.0,
                        "generation_tokens": 16,
                        "repeat_index": 2,
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "kind": "cluster_benchmark_metric",
                        "cluster_path": "exo_api_bench_chat_completions",
                        "api_url": "http://127.0.0.1:52415",
                        "endpoint": "/bench/chat/completions",
                        "model_id": "kernelpool/MiMo-V2.5-Pro-6bit",
                        "prompt_hash": "same-prompt",
                        "temperature": 0.0,
                        "max_tokens": 16,
                        "mode": "mtp-d1",
                        "accepted_execution_path": "mimo_mtp_fastpath",
                        "generation_tps": 60.0,
                        "generation_tokens": 16,
                        "repeat_index": 0,
                    },
                    sort_keys=True,
                ),
            ]
        ),
        encoding="utf-8",
    )

    report = calculator.build_budget_report([fixture_path])

    budget = cast(dict[str, object], report["budget"])
    assert budget["ar_row_count"] == 2
    assert budget["ar_median_tok_s"] == 30.0
    assert budget["ar_ms_per_token"] == 37.5


def test_calculator_reports_ar_plus_mtp_budget_classifications_and_bottlenecks() -> (
    None
):
    fixture_path = FIXTURE_DIR / "mimo_mtp_budget_ar_plus_mtp.jsonl"

    report = calculator.build_budget_report([fixture_path])

    assert report["budget"] == {
        "kind": "mimo_mtp_speedup_budget",
        "ar_row_count": 2,
        "mtp_row_count": 2,
        "ar_median_tok_s": 21.0,
        "ar_ms_per_token": 47.727273,
        "mtp_median_tok_s": 35.0,
        "mtp_target_gap_to_30_tok_s": 0.0,
        "mtp_target_gap_to_40_tok_s": 5.0,
        "mtp_vs_ar_speedup_ratio": 1.666667,
        "status": "same_cluster_budget_ready",
        "next_step": "use this budget only with same-cluster AR-vs-MTP evidence rows; Slice 5 still requires a real MTP win without fallback concerns",
    }
    assert report["classifications"] == {
        "same_cluster_evidence": "same_cluster_budget_ready",
        "budget_classification": "pass",
        "mtp_throughput_threshold": "at_least_30_tok_s",
        "slice_5_gate": "eligible_for_guarded_review",
        "slice_5_review_allowed": True,
        "slice_5_minimum_speedup_ratio": 1.15,
        "speedup_claim_allowed": True,
        "target_30_tok_s_claim_allowed": True,
        "target_40_tok_s_claim_allowed": False,
    }
    assert report["bottlenecks"] == [
        "acceptance_rate_low",
        "mtp_beats_ar",
        "mtp_reaches_30",
    ]


def test_calculator_reports_all_budget_and_bottleneck_labels_when_telemetry_exists() -> (
    None
):
    fixture_path = FIXTURE_DIR / "mimo_mtp_budget_full_telemetry.jsonl"

    report = calculator.build_budget_report([fixture_path])

    assert report["classifications"] == {
        "same_cluster_evidence": "same_cluster_budget_ready",
        "budget_classification": "pass",
        "mtp_throughput_threshold": "at_least_40_tok_s",
        "slice_5_gate": "blocked_fallback_or_correctness_concern",
        "slice_5_review_allowed": False,
        "slice_5_minimum_speedup_ratio": 1.15,
        "speedup_claim_allowed": True,
        "target_30_tok_s_claim_allowed": True,
        "target_40_tok_s_claim_allowed": True,
    }
    assert report["bottlenecks"] == [
        "acceptance_rate_low",
        "proposal_too_slow",
        "verifier_too_slow",
        "fallback_too_high",
        "depth_too_aggressive",
        "mtp_beats_ar",
        "mtp_reaches_30",
        "mtp_reaches_40",
    ]


def test_calculator_marks_same_cluster_mtp_slower_than_ar_as_fail() -> None:
    report = calculator.build_budget_report(
        [FIXTURE_DIR / "mimo_mtp_budget_mtp_slower.jsonl"]
    )

    assert (
        cast(dict[str, object], report["classifications"])["budget_classification"]
        == "fail"
    )
    assert (
        cast(dict[str, object], report["classifications"])["speedup_claim_allowed"]
        is False
    )
    assert report["bottlenecks"] == ["mtp_not_faster_than_ar"]


def test_calculator_does_not_emit_target_labels_without_ar_baseline(
    tmp_path: Path,
) -> None:
    fixture_path = tmp_path / "mtp-only-high-throughput.jsonl"
    fixture_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "kind": "cluster_benchmark_metric",
                        "cluster_path": "exo_api_bench_chat_completions",
                        "live_execution_path": "exo_cluster_tensor_parallel",
                        "accepted_execution_path": "mimo_mtp_fastpath",
                        "mode": "mtp-d2",
                        "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                        "repeat_index": 0,
                        "http_status": 200,
                        "generation_tps": 45.0,
                        "generation_tokens": 16,
                    },
                    sort_keys=True,
                )
            ]
        ),
        encoding="utf-8",
    )

    report = calculator.build_budget_report([fixture_path])

    assert (
        cast(dict[str, object], report["budget"])["status"] == "blocked_missing_ar_rows"
    )
    assert report["bottlenecks"] == ["missing_ar_baseline_rows"]


def test_evidence_sufficiency_marks_malformed_benchmark_evidence_as_insufficient_data(
    tmp_path: Path,
) -> None:
    fixture_path = tmp_path / "malformed.jsonl"
    fixture_path.write_text("{not valid json}\n", encoding="utf-8")

    report = calculator.build_budget_report([fixture_path])

    assert report["evidence_sufficiency"] == {
        "status": "insufficient_data",
        "reason": "malformed_benchmark_evidence",
        "budget_status": "blocked_incomplete_telemetry",
        "valid_row_count": 0,
        "error_count": 1,
        "blocked_row_count": 0,
        "ar_row_count": 0,
        "mtp_row_count": 0,
        "live_mtp_row_count": 0,
        "performance_claim_allowed": False,
        "next_step": "fix malformed benchmark JSONL before evaluating AR-vs-MTP sufficiency",
    }


def test_evidence_sufficiency_marks_missing_required_fields_as_insufficient_data(
    tmp_path: Path,
) -> None:
    fixture_path = tmp_path / "missing-required-fields.jsonl"
    fixture_path.write_text(
        json.dumps(
            {
                "kind": "cluster_benchmark_metric",
                "cluster_path": "exo_api_bench_chat_completions",
                "mode": "ar",
                "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                "generation_tps": 20.0,
                "generation_tokens": 16,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = calculator.build_budget_report([fixture_path])

    assert report["evidence_sufficiency"] == {
        "status": "insufficient_data",
        "reason": "missing_required_benchmark_fields",
        "budget_status": "blocked_incomplete_telemetry",
        "valid_row_count": 0,
        "error_count": 1,
        "blocked_row_count": 0,
        "ar_row_count": 0,
        "mtp_row_count": 0,
        "live_mtp_row_count": 0,
        "performance_claim_allowed": False,
        "next_step": "repair benchmark rows to the canonical required field contract before evaluating sufficiency",
    }


def test_evidence_sufficiency_marks_ar_only_rows_as_insufficient_data() -> None:
    fixture_path = FIXTURE_DIR / "mimo_mtp_budget_ar_only.jsonl"

    report = calculator.build_budget_report([fixture_path])

    assert report["evidence_sufficiency"] == {
        "status": "insufficient_data",
        "reason": "absent_mtp_benchmark_evidence",
        "budget_status": "blocked_missing_mtp_rows",
        "valid_row_count": 2,
        "error_count": 0,
        "blocked_row_count": 0,
        "ar_row_count": 2,
        "mtp_row_count": 0,
        "live_mtp_row_count": 0,
        "performance_claim_allowed": False,
        "next_step": "collect same-cluster guarded MTP rows before making speedup or target claims",
    }


def test_evidence_sufficiency_marks_absent_mtp_with_blocked_record_as_blocked(
    tmp_path: Path,
) -> None:
    fixture_path = tmp_path / "blocked-mtp.jsonl"
    fixture_path.write_text(
        json.dumps(
            {
                "schema_version": "mimo_mtp_benchmark_evidence/v1",
                "evidence_kind": "blocked_benchmark_row",
                "row_status": "blocked_with_command",
                "benchmark_session_id": "sub-ac-3-blocked",
                "api_url": "http://127.0.0.1:52415",
                "endpoint": "/bench/chat/completions",
                "model_id": "kernelpool/MiMo-V2.5-Pro-6bit",
                "mode": "mtp-d1",
                "mtp_enabled": True,
                "mtp_depth": 1,
                "mtp_execution_state": "blocked_unavailable",
                "mtp_disable_reason": "cluster_api_unavailable",
                "commands": [
                    "python3 scripts/bench_mimo_mtp_cluster.py --mode-label mtp-d1 --payload-extra-json '{\"mimo_mtp_fastpath\":true}'"
                ],
                "environment_assumptions": ["exo cluster API is reachable"],
                "output_paths": [
                    ".goose-ultrawork/evidence/sub-ac-3-blocked-mtp.jsonl"
                ],
                "non_fabrication_statement": "No performance result is claimed; this row records a blocked rerun command only.",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = calculator.build_budget_report([fixture_path])

    assert report["evidence_sufficiency"] == {
        "status": "blocked",
        "reason": "blocked_benchmark_evidence",
        "budget_status": "blocked_incomplete_telemetry",
        "valid_row_count": 0,
        "error_count": 0,
        "blocked_row_count": 1,
        "ar_row_count": 0,
        "mtp_row_count": 0,
        "live_mtp_row_count": 0,
        "performance_claim_allowed": False,
        "next_step": "run the recorded blocked benchmark command against an available guarded cluster before evaluating sufficiency",
    }


def test_calculator_cli_prints_stable_json_report_for_fixture() -> None:
    fixture_path = FIXTURE_DIR / "mimo_mtp_budget_ar_plus_mtp.jsonl"
    entrypoint = REPO_ROOT / "scripts" / "mimo_mtp_budget_calculator.py"

    completed = subprocess.run(
        [sys.executable, str(entrypoint), str(fixture_path)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    report = cast(dict[str, object], json.loads(completed.stdout))
    assert report["kind"] == "mimo_mtp_budget_report"
    assert cast(dict[str, object], report["budget"])["mtp_median_tok_s"] == 35.0
    assert (
        cast(dict[str, object], report["classifications"])["slice_5_gate"]
        == "eligible_for_guarded_review"
    )


def test_calculator_cli_writes_report_when_output_path_is_provided(
    tmp_path: Path,
) -> None:
    fixture_path = FIXTURE_DIR / "mimo_mtp_budget_ar_only.jsonl"
    output_path = tmp_path / "budget-report.json"
    entrypoint = REPO_ROOT / "scripts" / "mimo_mtp_budget_calculator.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(entrypoint),
            str(fixture_path),
            "--output",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stdout == ""
    written_report = cast(dict[str, object], json.loads(output_path.read_text()))
    assert written_report["kind"] == "mimo_mtp_budget_report"
    assert (
        cast(dict[str, object], written_report["classifications"])[
            "speedup_claim_allowed"
        ]
        is False
    )


def test_calculator_cli_help_documents_user_facing_entrypoint() -> None:
    entrypoint = REPO_ROOT / "scripts" / "mimo_mtp_budget_calculator.py"

    completed = subprocess.run(
        [sys.executable, str(entrypoint), "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert (
        "Compute MiMo MTP budget metrics from AR/MTP JSONL benchmark rows"
        in completed.stdout
    )
    assert "jsonl_paths" in completed.stdout
    assert "--output" in completed.stdout


def test_optimization_loop_recommends_exactly_one_family_from_measured_bottleneck() -> (
    None
):
    report = calculator.build_budget_report(
        [FIXTURE_DIR / "mimo_mtp_budget_ar_plus_mtp.jsonl"]
    )

    optimization_loop = cast(dict[str, object], report["optimization_loop"])

    assert optimization_loop["next_optimization_family"] == "acceptance_semantics"
    assert optimization_loop["recommendation_status"] == "measured_bottleneck_ready"
    assert optimization_loop["blind_optimization_allowed"] is False
    assert optimization_loop["candidate_families"] == ["acceptance_semantics"]


def test_optimization_loop_marks_missing_mtp_bottleneck_telemetry_insufficient_data(
    tmp_path: Path,
) -> None:
    fixture_path = tmp_path / "same-cluster-win-without-bottleneck-telemetry.jsonl"
    common_metadata = {
        "kind": "cluster_benchmark_metric",
        "cluster_path": "exo_api_bench_chat_completions",
        "live_execution_path": "exo_cluster_tensor_parallel",
        "api_url": "http://cluster-a.example",
        "benchmark_session_id": "session-a",
        "endpoint": "/bench/chat/completions",
        "model_id": "kernelpool/MiMo-V2.5-Pro-6bit",
        "prompt_hash": "prompt-a",
        "temperature": 0.0,
        "max_tokens": 64,
        "http_status": 200,
        "generation_tokens": 64,
    }
    fixture_path.write_text(
        "\n".join(
            json.dumps(row, sort_keys=True)
            for row in [
                {
                    **common_metadata,
                    "accepted_execution_path": "ar",
                    "mode": "ar",
                    "repeat_index": 0,
                    "generation_tps": 20.0,
                },
                {
                    **common_metadata,
                    "accepted_execution_path": "mimo_mtp_fastpath",
                    "mode": "mtp-d2",
                    "repeat_index": 0,
                    "generation_tps": 35.0,
                },
            ]
        ),
        encoding="utf-8",
    )

    report = calculator.build_budget_report([fixture_path])
    optimization_loop = cast(dict[str, object], report["optimization_loop"])

    assert report["bottlenecks"] == [
        "mtp_beats_ar",
        "mtp_reaches_30",
        "insufficient_bottleneck_telemetry",
    ]
    assert optimization_loop["recommendation_status"] == "insufficient_data"
    assert optimization_loop["next_optimization_family"] is None
    assert optimization_loop["candidate_families"] == []
    assert optimization_loop["blocking_reason"] == (
        "collect benchmark-grade MTP bottleneck telemetry before choosing an "
        "optimization family or Slice 5 review"
    )


def test_optimization_loop_blocks_blind_optimization_without_measured_or_budgeted_bottleneck() -> (
    None
):
    report = calculator.build_budget_report(
        [FIXTURE_DIR / "mimo_mtp_budget_ar_only.jsonl"]
    )

    optimization_loop = cast(dict[str, object], report["optimization_loop"])

    assert optimization_loop["next_optimization_family"] is None
    assert (
        optimization_loop["recommendation_status"] == "blocked_no_measured_bottleneck"
    )
    assert optimization_loop["blind_optimization_allowed"] is False
    assert optimization_loop["candidate_families"] == []
    assert optimization_loop["blocking_reason"] == (
        "collect same-cluster AR and guarded live MTP rows before optimizing; "
        "missing_live_mtp_rows is an evidence collection blocker, not an optimization family"
    )


def test_optimization_loop_auto_depth_uses_measured_depth_rows_without_assuming_d3(
    tmp_path: Path,
) -> None:
    fixture_path = tmp_path / "auto-depth-measured.jsonl"
    fixture_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "kind": "cluster_benchmark_metric",
                        "cluster_path": "exo_api_bench_chat_completions",
                        "live_execution_path": "exo_cluster_tensor_parallel",
                        "accepted_execution_path": "ar",
                        "mode": "ar",
                        "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                        "repeat_index": 0,
                        "http_status": 200,
                        "generation_tps": 20.0,
                        "generation_tokens": 16,
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "kind": "cluster_benchmark_metric",
                        "cluster_path": "exo_api_bench_chat_completions",
                        "live_execution_path": "exo_cluster_tensor_parallel",
                        "accepted_execution_path": "mimo_mtp_fastpath",
                        "mode": "mtp-d1",
                        "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                        "repeat_index": 0,
                        "http_status": 200,
                        "generation_tps": 30.0,
                        "generation_tokens": 16,
                        "generation_stats": {
                            "mimo_mtp_depth": 1,
                            "mimo_mtp_attempted_tokens": 16,
                            "mimo_mtp_accepted_tokens": 12,
                            "mimo_mtp_fallback_count": 0,
                        },
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "kind": "cluster_benchmark_metric",
                        "cluster_path": "exo_api_bench_chat_completions",
                        "live_execution_path": "exo_cluster_tensor_parallel",
                        "accepted_execution_path": "mimo_mtp_fastpath",
                        "mode": "mtp-d2",
                        "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                        "repeat_index": 0,
                        "http_status": 200,
                        "generation_tps": 34.0,
                        "generation_tokens": 16,
                        "generation_stats": {
                            "mimo_mtp_depth": 2,
                            "mimo_mtp_attempted_tokens": 16,
                            "mimo_mtp_accepted_tokens": 12,
                            "mimo_mtp_fallback_count": 0,
                        },
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "kind": "cluster_benchmark_metric",
                        "cluster_path": "exo_api_bench_chat_completions",
                        "live_execution_path": "exo_cluster_tensor_parallel",
                        "accepted_execution_path": "mimo_mtp_fastpath",
                        "mode": "mtp-d3",
                        "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                        "repeat_index": 0,
                        "http_status": 200,
                        "generation_tps": 28.0,
                        "generation_tokens": 16,
                        "generation_stats": {
                            "mimo_mtp_depth": 3,
                            "mimo_mtp_attempted_tokens": 16,
                            "mimo_mtp_accepted_tokens": 7,
                            "mimo_mtp_fallback_count": 0,
                        },
                    },
                    sort_keys=True,
                ),
            ]
        ),
        encoding="utf-8",
    )

    report = calculator.build_budget_report([fixture_path])
    optimization_loop = cast(dict[str, object], report["optimization_loop"])
    depth_policy = cast(dict[str, object], optimization_loop["depth_policy"])

    assert optimization_loop["next_optimization_family"] == "slice_5_review"
    assert depth_policy["policy"] == "auto_depth_from_measured_rows"
    assert depth_policy["recommended_depth"] == 2
    assert depth_policy["d3_assumed_best"] is False
    assert depth_policy["measured_depths"] == [1, 2, 3]


def test_slice_5_gate_blocks_small_speedup_below_meaningful_margin(
    tmp_path: Path,
) -> None:
    fixture_path = tmp_path / "small-speedup.jsonl"
    fixture_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "kind": "cluster_benchmark_metric",
                        "cluster_path": "exo_api_bench_chat_completions",
                        "live_execution_path": "exo_cluster_tensor_parallel",
                        "accepted_execution_path": "ar",
                        "mode": "ar",
                        "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                        "repeat_index": 0,
                        "http_status": 200,
                        "generation_tps": 28.0,
                        "generation_tokens": 16,
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "kind": "cluster_benchmark_metric",
                        "cluster_path": "exo_api_bench_chat_completions",
                        "live_execution_path": "exo_cluster_tensor_parallel",
                        "accepted_execution_path": "mimo_mtp_fastpath",
                        "mode": "mtp-d1",
                        "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                        "repeat_index": 0,
                        "http_status": 200,
                        "generation_tps": 29.0,
                        "generation_tokens": 16,
                        "generation_stats": {
                            "mimo_mtp_depth": 1,
                            "mimo_mtp_attempted_tokens": 16,
                            "mimo_mtp_accepted_tokens": 14,
                            "mimo_mtp_fallback_count": 0,
                        },
                    },
                    sort_keys=True,
                ),
            ]
        ),
        encoding="utf-8",
    )

    report = calculator.build_budget_report([fixture_path])

    classifications = cast(dict[str, object], report["classifications"])
    assert classifications["speedup_claim_allowed"] is True
    assert classifications["target_30_tok_s_claim_allowed"] is False
    assert classifications["slice_5_gate"] == "blocked_margin_below_15_percent"
    assert classifications["slice_5_review_allowed"] is False
    assert classifications["slice_5_minimum_speedup_ratio"] == 1.15


def test_slice_5_gate_blocks_fallback_concern_despite_target_throughput(
    tmp_path: Path,
) -> None:
    fixture_path = tmp_path / "fallback-concern.jsonl"
    fixture_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "kind": "cluster_benchmark_metric",
                        "cluster_path": "exo_api_bench_chat_completions",
                        "live_execution_path": "exo_cluster_tensor_parallel",
                        "accepted_execution_path": "ar",
                        "mode": "ar",
                        "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                        "repeat_index": 0,
                        "http_status": 200,
                        "generation_tps": 20.0,
                        "generation_tokens": 16,
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "kind": "cluster_benchmark_metric",
                        "cluster_path": "exo_api_bench_chat_completions",
                        "live_execution_path": "exo_cluster_tensor_parallel",
                        "accepted_execution_path": "mimo_mtp_fastpath",
                        "mode": "mtp-d2",
                        "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                        "repeat_index": 0,
                        "http_status": 200,
                        "generation_tps": 35.0,
                        "generation_tokens": 16,
                        "generation_stats": {
                            "mimo_mtp_depth": 2,
                            "mimo_mtp_attempted_tokens": 16,
                            "mimo_mtp_accepted_tokens": 12,
                            "mimo_mtp_fallback_count": 5,
                        },
                    },
                    sort_keys=True,
                ),
            ]
        ),
        encoding="utf-8",
    )

    report = calculator.build_budget_report([fixture_path])

    classifications = cast(dict[str, object], report["classifications"])
    assert classifications["speedup_claim_allowed"] is True
    assert classifications["target_30_tok_s_claim_allowed"] is True
    assert classifications["slice_5_gate"] == "blocked_fallback_or_correctness_concern"
    assert classifications["slice_5_review_allowed"] is False


def test_slice_5_gate_allows_explicit_target_with_real_win_without_fallback(
    tmp_path: Path,
) -> None:
    fixture_path = tmp_path / "target-win-below-margin.jsonl"
    fixture_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "kind": "cluster_benchmark_metric",
                        "cluster_path": "exo_api_bench_chat_completions",
                        "live_execution_path": "exo_cluster_tensor_parallel",
                        "accepted_execution_path": "ar",
                        "mode": "ar",
                        "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                        "repeat_index": 0,
                        "http_status": 200,
                        "generation_tps": 29.0,
                        "generation_tokens": 16,
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "kind": "cluster_benchmark_metric",
                        "cluster_path": "exo_api_bench_chat_completions",
                        "live_execution_path": "exo_cluster_tensor_parallel",
                        "accepted_execution_path": "mimo_mtp_fastpath",
                        "mode": "mtp-d1",
                        "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                        "repeat_index": 0,
                        "http_status": 200,
                        "generation_tps": 31.0,
                        "generation_tokens": 16,
                        "generation_stats": {
                            "mimo_mtp_depth": 1,
                            "mimo_mtp_attempted_tokens": 16,
                            "mimo_mtp_accepted_tokens": 14,
                            "mimo_mtp_fallback_count": 0,
                        },
                    },
                    sort_keys=True,
                ),
            ]
        ),
        encoding="utf-8",
    )

    report = calculator.build_budget_report([fixture_path])

    classifications = cast(dict[str, object], report["classifications"])
    assert classifications["speedup_claim_allowed"] is True
    assert classifications["target_30_tok_s_claim_allowed"] is True
    assert classifications["slice_5_gate"] == "blocked_margin_below_15_percent"
    assert classifications["slice_5_review_allowed"] is False


def test_slice_5_refuses_high_mtp_throughput_without_meaningful_margin(
    tmp_path: Path,
) -> None:
    fixture_path = tmp_path / "ar-plus-mtp-high-throughput-low-margin.jsonl"
    fixture_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "kind": "cluster_benchmark_metric",
                        "cluster_path": "exo_api_bench_chat_completions",
                        "live_execution_path": "exo_cluster_tensor_parallel",
                        "accepted_execution_path": "ar",
                        "mode": "ar",
                        "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                        "repeat_index": 0,
                        "http_status": 200,
                        "generation_tps": 30.0,
                        "generation_tokens": 16,
                        "payload_extra_keys": [],
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "kind": "cluster_benchmark_metric",
                        "cluster_path": "exo_api_bench_chat_completions",
                        "live_execution_path": "exo_cluster_tensor_parallel",
                        "accepted_execution_path": "mimo_mtp_fastpath",
                        "mode": "mtp-d2",
                        "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                        "repeat_index": 0,
                        "http_status": 200,
                        "generation_tps": 32.0,
                        "generation_tokens": 16,
                        "payload_extra_keys": ["mimo_mtp_depth", "mimo_mtp_fastpath"],
                        "generation_stats": {
                            "mimo_mtp_depth": 2,
                            "mimo_mtp_attempted_tokens": 20,
                            "mimo_mtp_accepted_tokens": 12,
                            "mimo_mtp_fallback_count": 0,
                        },
                    },
                    sort_keys=True,
                ),
            ]
        ),
        encoding="utf-8",
    )

    report = calculator.build_budget_report([fixture_path])

    classifications = cast(dict[str, object], report["classifications"])
    assert classifications["slice_5_minimum_speedup_ratio"] == 1.15
    assert classifications["slice_5_gate"] == "blocked_margin_below_15_percent"
    assert classifications["slice_5_review_allowed"] is False
    assert classifications["speedup_claim_allowed"] is True
    assert classifications["target_30_tok_s_claim_allowed"] is True


def test_ar_only_rows_with_mtp_shaped_fields_emit_explicit_refusal_reason(
    tmp_path: Path,
) -> None:
    fixture_path = tmp_path / "ar-only-with-mtp-shaped-fields.jsonl"
    fixture_path.write_text(
        json.dumps(
            {
                "kind": "cluster_benchmark_metric",
                "cluster_path": "exo_api_bench_chat_completions",
                "live_execution_path": "exo_cluster_tensor_parallel",
                "accepted_execution_path": "ar",
                "mode": "ar",
                "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                "repeat_index": 0,
                "http_status": 200,
                "generation_tps": 50.0,
                "generation_tokens": 16,
                "payload_extra_keys": ["mimo_mtp_depth", "mimo_mtp_fastpath"],
                "mtp_enabled": False,
                "mtp_execution_state": "disabled_default",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = calculator.build_budget_report([fixture_path])

    classifications = cast(dict[str, object], report["classifications"])
    assert classifications["slice_5_gate"] == "blocked_missing_mtp_rows"
    assert classifications["slice_5_review_allowed"] is False
    assert classifications["speedup_claim_allowed"] is False
    assert classifications["target_30_tok_s_claim_allowed"] is False
    assert classifications["target_40_tok_s_claim_allowed"] is False
    assert classifications["refusal_reasons"] == [
        "ar_only_evidence_no_guarded_live_mtp_rows"
    ]


def _speedup_fixture_rows() -> list[dict[str, object]]:
    common_metadata: dict[str, object] = {
        "kind": "cluster_benchmark_metric",
        "cluster_path": "exo_api_bench_chat_completions",
        "live_execution_path": "exo_cluster_tensor_parallel",
        "api_url": "http://127.0.0.1:52415",
        "benchmark_session_id": "claim-suppression-session",
        "endpoint": "/bench/chat/completions",
        "model_id": "kernelpool/MiMo-V2.5-Pro-6bit",
        "model": "kernelpool/MiMo-V2.5-Pro-6bit",
        "prompt_hash": "claim-suppression-prompt",
        "temperature": 0.0,
        "max_tokens": 16,
        "http_status": 200,
        "generation_tokens": 16,
    }
    return [
        {
            **common_metadata,
            "accepted_execution_path": "ar",
            "mode": "ar",
            "repeat_index": 0,
            "generation_tps": 20.0,
            "mtp_enabled": False,
            "mtp_execution_state": "disabled_default",
        },
        {
            **common_metadata,
            "accepted_execution_path": "mimo_mtp_fastpath",
            "mode": "mtp-d2",
            "repeat_index": 0,
            "generation_tps": 35.0,
            "mtp_enabled": True,
            "mtp_depth": 2,
            "mtp_execution_state": "successful_mtp",
            "telemetry_completeness": "complete",
            "generation_stats": {
                "mimo_mtp_depth": 2,
                "mimo_mtp_attempted_tokens": 16,
                "mimo_mtp_accepted_tokens": 14,
                "mimo_mtp_fallback_count": 0,
                "acceptance_rate": 0.875,
                "fallback_count": 0,
            },
        },
    ]


def test_analyzer_suppresses_speedup_claims_when_malformed_evidence_is_present(
    tmp_path: Path,
) -> None:
    fixture_path = tmp_path / "valid-speedup-plus-malformed.jsonl"
    fixture_path.write_text(
        "\n".join(
            [
                *(json.dumps(row, sort_keys=True) for row in _speedup_fixture_rows()),
                "{not valid json}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = calculator.build_budget_report([fixture_path])

    classifications = cast(dict[str, object], report["classifications"])
    assert (
        cast(dict[str, object], report["evidence_sufficiency"])["status"]
        == "insufficient_data"
    )
    assert classifications["budget_classification"] == "ambiguous"
    assert classifications["slice_5_gate"] == "blocked_malformed_benchmark_evidence"
    assert classifications["slice_5_review_allowed"] is False
    assert classifications["speedup_claim_allowed"] is False
    assert classifications["target_30_tok_s_claim_allowed"] is False
    assert classifications["target_40_tok_s_claim_allowed"] is False
    assert classifications["refusal_reasons"] == ["malformed_benchmark_evidence"]


def test_analyzer_suppresses_speedup_claims_when_blocked_evidence_is_present(
    tmp_path: Path,
) -> None:
    fixture_path = tmp_path / "valid-speedup-plus-blocked.jsonl"
    blocked_row = {
        "schema_version": "mimo_mtp_benchmark_evidence/v1",
        "evidence_kind": "blocked_benchmark_row",
        "row_status": "blocked_with_command",
        "benchmark_session_id": "claim-suppression-session",
        "api_url": "http://127.0.0.1:52415",
        "endpoint": "/bench/chat/completions",
        "model_id": "kernelpool/MiMo-V2.5-Pro-6bit",
        "mode": "mtp-d3",
        "mtp_enabled": True,
        "mtp_depth": 3,
        "mtp_execution_state": "blocked_unavailable",
        "mtp_disable_reason": "cluster_api_unavailable",
        "commands": [
            "python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --mode-label mtp-d3"
        ],
        "environment_assumptions": ["exo cluster API is reachable"],
        "output_paths": [".goose-ultrawork/evidence/claim-suppression-mtp-d3.jsonl"],
        "non_fabrication_statement": "No performance result is claimed; this row records a blocked rerun command only.",
    }
    fixture_path.write_text(
        "\n".join(
            json.dumps(row, sort_keys=True)
            for row in [*_speedup_fixture_rows(), blocked_row]
        )
        + "\n",
        encoding="utf-8",
    )

    report = calculator.build_budget_report([fixture_path])

    classifications = cast(dict[str, object], report["classifications"])
    assert (
        cast(dict[str, object], report["evidence_sufficiency"])["status"] == "blocked"
    )
    assert classifications["budget_classification"] == "ambiguous"
    assert classifications["slice_5_gate"] == "blocked_benchmark_evidence"
    assert classifications["slice_5_review_allowed"] is False
    assert classifications["speedup_claim_allowed"] is False
    assert classifications["target_30_tok_s_claim_allowed"] is False
    assert classifications["target_40_tok_s_claim_allowed"] is False
    assert classifications["refusal_reasons"] == ["blocked_benchmark_evidence"]
