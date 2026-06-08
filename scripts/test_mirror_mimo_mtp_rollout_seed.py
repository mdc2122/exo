from __future__ import annotations

from pathlib import Path

from scripts import mirror_mimo_mtp_rollout_seed as mirror_seed

SEED_TEXT = """goal: "Implement the optimized MiMo V2.5 Pro MTP rollout strategy."
acceptance_criteria:
  - |
    AC-P0 Optimized rollout seed and repo context are validated.
  - |
    AC-P1 Cluster AR baseline harness is canonical and budget-ready.
"""


def _write_seed(path: Path) -> Path:
    path.write_text(SEED_TEXT, encoding="utf-8")
    return path


def test_mirror_rollout_seed_creates_todo_and_runlog_when_missing(
    tmp_path: Path,
) -> None:
    seed_path = _write_seed(tmp_path / "seed.yaml")
    todo_path = tmp_path / "TODO.md"
    runlog_path = tmp_path / "docs" / "plans" / "MIMO_V25_PRO_MTP_FOUNDATION_RUNLOG.md"

    result = mirror_seed.mirror_rollout_seed(
        seed_path=seed_path,
        todo_path=todo_path,
        runlog_path=runlog_path,
    )

    assert result.todo_action == "created"
    assert result.runlog_action == "created"
    todo_text = todo_path.read_text(encoding="utf-8")
    runlog_text = runlog_path.read_text(encoding="utf-8")
    assert todo_text.count(mirror_seed.TODO_SECTION_HEADING) == 1
    assert runlog_text.count(mirror_seed.RUNLOG_SECTION_HEADING) == 1
    assert "guarded exo-cluster MTP vertical slice" in todo_text
    assert "same-cluster AR-vs-MTP rows" in runlog_text


def test_mirror_rollout_seed_updates_existing_sections_without_losing_other_content(
    tmp_path: Path,
) -> None:
    seed_path = _write_seed(tmp_path / "seed.yaml")
    todo_path = tmp_path / "TODO.md"
    runlog_path = tmp_path / "docs" / "plans" / "MIMO_V25_PRO_MTP_FOUNDATION_RUNLOG.md"
    todo_path.write_text(
        "# Existing TODO\n\n"
        f"{mirror_seed.TODO_SECTION_HEADING}\n\n"
        "- stale broad rollout wording\n\n"
        "## Later TODO\n\n"
        "- keep me\n",
        encoding="utf-8",
    )
    runlog_path.parent.mkdir(parents=True)
    runlog_path.write_text(
        "# Existing Runlog\n\n"
        f"{mirror_seed.RUNLOG_SECTION_HEADING}\n\n"
        "stale runlog body\n\n"
        "## Later Runlog\n\n"
        "keep me too\n",
        encoding="utf-8",
    )

    result = mirror_seed.mirror_rollout_seed(
        seed_path=seed_path,
        todo_path=todo_path,
        runlog_path=runlog_path,
    )

    assert result.todo_action == "updated"
    assert result.runlog_action == "updated"
    todo_text = todo_path.read_text(encoding="utf-8")
    runlog_text = runlog_path.read_text(encoding="utf-8")
    assert "stale broad rollout wording" not in todo_text
    assert "stale runlog body" not in runlog_text
    assert "## Later TODO\n\n- keep me" in todo_text
    assert "## Later Runlog\n\nkeep me too" in runlog_text
    assert todo_text.count(mirror_seed.TODO_SECTION_HEADING) == 1
    assert runlog_text.count(mirror_seed.RUNLOG_SECTION_HEADING) == 1


def test_mirror_rollout_seed_repeated_runs_do_not_duplicate_sections(
    tmp_path: Path,
) -> None:
    seed_path = _write_seed(tmp_path / "seed.yaml")
    todo_path = tmp_path / "TODO.md"
    runlog_path = tmp_path / "docs" / "plans" / "MIMO_V25_PRO_MTP_FOUNDATION_RUNLOG.md"

    first = mirror_seed.mirror_rollout_seed(
        seed_path=seed_path,
        todo_path=todo_path,
        runlog_path=runlog_path,
    )
    second = mirror_seed.mirror_rollout_seed(
        seed_path=seed_path,
        todo_path=todo_path,
        runlog_path=runlog_path,
    )
    third = mirror_seed.mirror_rollout_seed(
        seed_path=seed_path,
        todo_path=todo_path,
        runlog_path=runlog_path,
    )

    assert first.todo_action == "created"
    assert first.runlog_action == "created"
    assert second.todo_action == "unchanged"
    assert second.runlog_action == "unchanged"
    assert third.todo_action == "unchanged"
    assert third.runlog_action == "unchanged"
    assert (
        todo_path.read_text(encoding="utf-8").count(mirror_seed.TODO_SECTION_HEADING)
        == 1
    )
    assert (
        runlog_path.read_text(encoding="utf-8").count(
            mirror_seed.RUNLOG_SECTION_HEADING
        )
        == 1
    )


def test_append_runlog_bookkeeping_record_appends_without_rewriting_existing_runlog(
    tmp_path: Path,
) -> None:
    runlog_path = tmp_path / "docs" / "plans" / "MIMO_V25_PRO_MTP_FOUNDATION_RUNLOG.md"
    runlog_path.parent.mkdir(parents=True)
    runlog_path.write_text(
        "# Existing Runlog\n\nkeep existing evidence\n", encoding="utf-8"
    )

    result = mirror_seed.append_runlog_bookkeeping_record(
        runlog_path=runlog_path,
        acceptance_criterion="AC-P7",
        timestamp="2026-06-06T22:47:00",
        record_kind="benchmark_blocker",
        summary="Cluster API unavailable; live rows blocked rather than fabricated.",
        evidence=(
            "command: uv run python3 scripts/bench_mimo_mtp_cluster.py --mode-label ar",
            "benchmark rows: blocked_with_command",
        ),
        remaining_risk="Same-cluster AR-vs-MTP evidence is still missing; Slice 5 blocked.",
    )

    assert result == mirror_seed.RunlogBookkeepingAppendResult(
        runlog_action="updated",
        appended_heading="### 2026-06-06T22:47:00 AC-P7 benchmark_blocker",
    )
    runlog_text = runlog_path.read_text(encoding="utf-8")
    assert runlog_text.startswith("# Existing Runlog\n\nkeep existing evidence\n")
    assert "### 2026-06-06T22:47:00 AC-P7 benchmark_blocker" in runlog_text
    assert (
        "- Summary: Cluster API unavailable; live rows blocked rather than fabricated."
        in runlog_text
    )
    assert (
        "- command: uv run python3 scripts/bench_mimo_mtp_cluster.py --mode-label ar"
        in runlog_text
    )
    assert "- benchmark rows: blocked_with_command" in runlog_text
    assert (
        "- Remaining risk: Same-cluster AR-vs-MTP evidence is still missing; Slice 5 blocked."
        in runlog_text
    )


def test_append_runlog_bookkeeping_record_does_not_touch_todo_or_beads(
    tmp_path: Path,
) -> None:
    runlog_path = tmp_path / "runlog.md"
    todo_path = tmp_path / "TODO.md"
    todo_path.write_text("# TODO\n\n- keep unchanged\n", encoding="utf-8")

    result = mirror_seed.append_runlog_bookkeeping_record(
        runlog_path=runlog_path,
        acceptance_criterion="AC-P3",
        timestamp="2026-06-06T22:47:00",
        record_kind="rollout_bookkeeping",
        summary="Guarded MTP request contract tests passed.",
        evidence=("files changed: src/exo/api/adapters/chat_completions.py",),
        remaining_risk="Distributed MTP execution remains guarded/unwired.",
    )

    assert result.runlog_action == "created"
    assert todo_path.read_text(encoding="utf-8") == "# TODO\n\n- keep unchanged\n"
    assert not hasattr(result, "todo_action")
    assert not hasattr(result, "beads_result")
    assert "Guarded MTP request contract tests passed." in runlog_path.read_text(
        encoding="utf-8"
    )


def test_sync_runlog_tracking_entry_upserts_canonical_payload_with_gate_and_bottlenecks(
    tmp_path: Path,
) -> None:
    runlog_path = tmp_path / "docs" / "plans" / "MIMO_V25_PRO_MTP_FOUNDATION_RUNLOG.md"
    runlog_path.parent.mkdir(parents=True)
    runlog_path.write_text(
        "# Existing Runlog\n\n"
        "## Prior Wave\n\n"
        "keep prior benchmark history\n\n"
        "## Later Manual Note\n\n"
        "keep later operator note\n",
        encoding="utf-8",
    )
    payload = {
        "tracking_id": "ac-p7-same-cluster-matrix",
        "acceptance_criterion": "AC-P7",
        "timestamp": "2026-06-07T01:48:00Z",
        "summary": "Cluster API unavailable; live AR-vs-MTP rows are blocked.",
        "evidence": [
            ".goose-ultrawork/evidence/ac-p7-live-matrix-attempt.jsonl",
            "command: uv run python3 scripts/bench_mimo_mtp_cluster.py --mode-label ar",
        ],
        "benchmark_rows": {
            "same_cluster_ar_rows": 0,
            "same_cluster_mtp_rows": 0,
            "live_rows_status": "blocked_with_command",
        },
        "gate_status": {
            "default_generation": "ar",
            "slice_5_gate": "blocked_missing_live_mtp_rows",
            "speedup_claim_allowed": False,
        },
        "bottleneck_analysis": {
            "recommendation_status": "blocked_no_measured_bottleneck",
            "bottlenecks": ["missing_live_ar_rows", "missing_live_mtp_rows"],
            "next_step": "Start exo cluster API and rerun the matrix command.",
        },
        "remaining_risk": "Slice 5 remains blocked until same-cluster MTP beats AR.",
    }

    first_result = mirror_seed.sync_runlog_tracking_entry_from_payload(
        runlog_path=runlog_path,
        payload=payload,
    )
    updated_payload = {
        **payload,
        "timestamp": "2026-06-07T02:10:00Z",
        "summary": "Same tracking row refreshed after rerun; cluster still unavailable.",
        "benchmark_rows": {
            "same_cluster_ar_rows": 1,
            "same_cluster_mtp_rows": 0,
            "live_rows_status": "blocked_missing_live_mtp_rows",
        },
        "bottleneck_analysis": {
            "recommendation_status": "blocked_no_measured_bottleneck",
            "bottlenecks": ["missing_live_mtp_rows"],
            "next_step": "Rerun guarded MTP matrix after worker fastpath is available.",
        },
    }
    second_result = mirror_seed.sync_runlog_tracking_entry_from_payload(
        runlog_path=runlog_path,
        payload=updated_payload,
    )

    runlog_text = runlog_path.read_text(encoding="utf-8")
    assert first_result.runlog_action == "updated"
    assert second_result.runlog_action == "updated"
    assert first_result.tracking_id == "ac-p7-same-cluster-matrix"
    assert second_result.tracking_id == "ac-p7-same-cluster-matrix"
    assert (
        runlog_text.count("### MiMo MTP rollout tracking: ac-p7-same-cluster-matrix")
        == 1
    )
    assert "## Prior Wave\n\nkeep prior benchmark history" in runlog_text
    assert "## Later Manual Note\n\nkeep later operator note" in runlog_text
    assert "- Last updated: 2026-06-07T02:10:00Z" in runlog_text
    assert "- Acceptance criterion: AC-P7" in runlog_text
    assert (
        "- Summary: Same tracking row refreshed after rerun; cluster still unavailable."
        in runlog_text
    )
    assert "- same_cluster_ar_rows: 1" in runlog_text
    assert "- same_cluster_mtp_rows: 0" in runlog_text
    assert "- live_rows_status: blocked_missing_live_mtp_rows" in runlog_text
    assert "- default_generation: ar" in runlog_text
    assert "- slice_5_gate: blocked_missing_live_mtp_rows" in runlog_text
    assert "- speedup_claim_allowed: False" in runlog_text
    assert "- recommendation_status: blocked_no_measured_bottleneck" in runlog_text
    assert "- bottlenecks: missing_live_mtp_rows" in runlog_text
    assert (
        "- next_step: Rerun guarded MTP matrix after worker fastpath is available."
        in runlog_text
    )
    assert ".goose-ultrawork/evidence/ac-p7-live-matrix-attempt.jsonl" in runlog_text
    assert "Slice 5 remains blocked until same-cluster MTP beats AR." in runlog_text
    assert (
        "Cluster API unavailable; live AR-vs-MTP rows are blocked." not in runlog_text
    )
