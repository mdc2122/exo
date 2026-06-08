from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "scripts" / "fixtures"
ENTRYPOINT = REPO_ROOT / "scripts" / "mimo_mtp_budget_calculator.py"


def _run_fixture(fixture_name: str) -> dict[str, Any]:
    fixture_path = FIXTURE_DIR / fixture_name
    completed = subprocess.run(
        [sys.executable, str(ENTRYPOINT), str(fixture_path)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src:."},
    )
    assert completed.returncode == 0, completed.stderr
    report = cast(dict[str, Any], json.loads(completed.stdout))
    assert report["kind"] == "mimo_mtp_budget_report"
    assert report["schema_version"] == 1
    assert report["source_files"] == [str(fixture_path)]
    return report


def _section(report: dict[str, Any], name: str) -> dict[str, Any]:
    return cast(dict[str, Any], report[name])


def test_sample_fixture_pass_contract_is_independently_runnable() -> None:
    report = _run_fixture("mimo_mtp_budget_sample_pass.jsonl")

    assert _section(report, "evidence_sufficiency")["status"] == "sufficient"
    assert _section(report, "budget")["status"] == "same_cluster_budget_ready"
    assert _section(report, "classifications") == {
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
    assert _section(report, "optimization_loop")["recommendation_status"] == "measured_bottleneck_ready"


def test_sample_fixture_fail_contract_is_independently_runnable() -> None:
    report = _run_fixture("mimo_mtp_budget_sample_fail.jsonl")

    assert _section(report, "evidence_sufficiency") == {
        "status": "insufficient_data",
        "reason": "same_cluster_mtp_not_faster_than_ar",
        "budget_status": "same_cluster_budget_ready",
        "valid_row_count": 2,
        "error_count": 0,
        "blocked_row_count": 0,
        "ar_row_count": 1,
        "mtp_row_count": 1,
        "live_mtp_row_count": 1,
        "performance_claim_allowed": False,
        "next_step": "use this budget only with same-cluster AR-vs-MTP evidence rows; Slice 5 still requires a real MTP win without fallback concerns",
    }
    classifications = _section(report, "classifications")
    assert classifications["budget_classification"] == "fail"
    assert classifications["slice_5_gate"] == "blocked_mtp_not_faster_than_ar"
    assert classifications["speedup_claim_allowed"] is False
    assert report["bottlenecks"] == ["mtp_not_faster_than_ar"]


def test_sample_fixture_ambiguous_absent_live_mtp_contract_is_independently_runnable() -> None:
    report = _run_fixture("mimo_mtp_budget_sample_ambiguous_absent_live_mtp.jsonl")

    assert _section(report, "evidence_sufficiency")["reason"] == "absent_live_mtp_benchmark_evidence"
    classifications = _section(report, "classifications")
    assert classifications["budget_classification"] == "ambiguous"
    assert classifications["slice_5_gate"] == "blocked_missing_live_mtp_rows"
    assert classifications["refusal_reasons"] == ["mtp_labeled_rows_without_guarded_live_fastpath"]


def test_sample_fixture_blocked_contract_is_independently_runnable() -> None:
    report = _run_fixture("mimo_mtp_budget_sample_blocked.jsonl")

    assert _section(report, "evidence_sufficiency")["status"] == "blocked"
    assert _section(report, "evidence_sufficiency")["reason"] == "blocked_benchmark_evidence"
    classifications = _section(report, "classifications")
    assert classifications["slice_5_gate"] == "blocked_benchmark_evidence"
    assert classifications["speedup_claim_allowed"] is False
    assert classifications["refusal_reasons"] == ["blocked_benchmark_evidence"]


def test_sample_fixture_insufficient_data_contract_is_independently_runnable() -> None:
    report = _run_fixture("mimo_mtp_budget_sample_insufficient_data_missing_required.jsonl")

    assert _section(report, "evidence_sufficiency")["status"] == "insufficient_data"
    assert _section(report, "evidence_sufficiency")["reason"] == "missing_required_benchmark_fields"
    classifications = _section(report, "classifications")
    assert classifications["budget_classification"] == "ambiguous"
    assert classifications["slice_5_gate"] == "blocked_missing_required_benchmark_fields"
    assert classifications["speedup_claim_allowed"] is False


def test_sample_fixture_ar_only_absent_mtp_contract_is_independently_runnable() -> None:
    report = _run_fixture("mimo_mtp_budget_sample_ar_only_absent_mtp.jsonl")

    assert _section(report, "evidence_sufficiency")["reason"] == "absent_mtp_benchmark_evidence"
    assert _section(report, "budget")["status"] == "blocked_missing_mtp_rows"
    classifications = _section(report, "classifications")
    assert classifications["slice_5_gate"] == "blocked_missing_mtp_rows"
    assert classifications["refusal_reasons"] == ["ar_only_evidence_no_guarded_live_mtp_rows"]


def test_sample_fixture_ar_only_mtp_shaped_fields_contract_is_independently_runnable() -> None:
    report = _run_fixture("mimo_mtp_budget_sample_ar_only_mtp_shaped_fields.jsonl")

    assert _section(report, "ingestion")["ar_row_count"] == 1
    assert _section(report, "ingestion")["mtp_row_count"] == 0
    classifications = _section(report, "classifications")
    assert classifications["slice_5_gate"] == "blocked_missing_mtp_rows"
    assert classifications["speedup_claim_allowed"] is False
    assert classifications["target_30_tok_s_claim_allowed"] is False
    assert classifications["refusal_reasons"] == ["ar_only_evidence_no_guarded_live_mtp_rows"]


def test_sample_fixture_partial_telemetry_contract_is_independently_runnable() -> None:
    report = _run_fixture("mimo_mtp_budget_sample_partial_telemetry.jsonl")

    assert _section(report, "evidence_sufficiency")["status"] == "sufficient"
    assert _section(report, "budget")["status"] == "same_cluster_budget_ready"
    classifications = _section(report, "classifications")
    assert classifications["budget_classification"] == "pass"
    assert classifications["speedup_claim_allowed"] is True
    assert classifications["slice_5_gate"] == "blocked_insufficient_bottleneck_telemetry"
    assert classifications["slice_5_review_allowed"] is False
    assert report["bottlenecks"] == [
        "mtp_beats_ar",
        "mtp_reaches_30",
        "insufficient_bottleneck_telemetry",
    ]
    assert _section(report, "optimization_loop")["recommendation_status"] == "insufficient_data"


def test_sample_fixture_non_same_cluster_contract_is_independently_runnable() -> None:
    report = _run_fixture("mimo_mtp_budget_sample_non_same_cluster.jsonl")

    assert _section(report, "evidence_sufficiency")["reason"] == "non_comparable_cluster_evidence"
    assert _section(report, "budget")["status"] == "ambiguous_non_comparable_cluster_rows"
    classifications = _section(report, "classifications")
    assert classifications["budget_classification"] == "ambiguous"
    assert classifications["slice_5_gate"] == "ambiguous_non_comparable_cluster_rows"
    assert classifications["refusal_reasons"] == ["non_comparable_cluster_evidence"]
    assert classifications["speedup_claim_allowed"] is False
