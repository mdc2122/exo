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
