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

    assert report == {
        "kind": "mimo_mtp_budget_report",
        "schema_version": 1,
        "source_files": [str(fixture_path)],
        "ingestion": {
            "valid_row_count": 2,
            "error_count": 0,
            "ar_row_count": 2,
            "mtp_row_count": 0,
            "live_mtp_row_count": 0,
        },
        "budget": {
            "kind": "mimo_mtp_speedup_budget",
            "ar_row_count": 2,
            "mtp_row_count": 0,
            "ar_median_tok_s": 21.0,
            "ar_ms_per_token": 47.619048,
            "mtp_median_tok_s": None,
            "mtp_target_gap_to_30_tok_s": None,
            "mtp_target_gap_to_40_tok_s": None,
            "mtp_vs_ar_speedup_ratio": None,
            "status": "blocked_missing_mtp_rows",
            "next_step": "collect same-cluster guarded MTP rows before making speedup or target claims",
        },
        "classifications": {
            "same_cluster_evidence": "blocked_missing_mtp_rows",
            "mtp_throughput_threshold": None,
            "slice_5_gate": "blocked_missing_mtp_rows",
            "speedup_claim_allowed": False,
            "target_30_tok_s_claim_allowed": False,
            "target_40_tok_s_claim_allowed": False,
        },
        "bottlenecks": ["missing_live_mtp_rows"],
        "next_step": "collect same-cluster guarded MTP rows before making speedup or target claims",
    }


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
        "ar_ms_per_token": 47.619048,
        "mtp_median_tok_s": 35.0,
        "mtp_target_gap_to_30_tok_s": 0.0,
        "mtp_target_gap_to_40_tok_s": 5.0,
        "mtp_vs_ar_speedup_ratio": 1.666667,
        "status": "same_cluster_budget_ready",
        "next_step": "use this budget only with same-cluster AR-vs-MTP evidence rows; Slice 5 still requires a real MTP win without fallback concerns",
    }
    assert report["classifications"] == {
        "same_cluster_evidence": "same_cluster_budget_ready",
        "mtp_throughput_threshold": "at_least_30_tok_s",
        "slice_5_gate": "eligible_for_guarded_review",
        "speedup_claim_allowed": True,
        "target_30_tok_s_claim_allowed": True,
        "target_40_tok_s_claim_allowed": False,
    }
    assert report["bottlenecks"] == ["acceptance_rate_low"]


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
