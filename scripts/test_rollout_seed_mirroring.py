"""Tests for pure rollout seed mirroring planning."""

from __future__ import annotations

from pathlib import Path

from scripts.rollout_seed_mirroring import (
    AcMirroringBoundaryDependencies,
    AcMirroringBoundaryResult,
    BeadsNoteOperation,
    FileAppendOperation,
    RolloutSeed,
    RolloutTodoChecklistEntry,
    RolloutTodoPayload,
    orchestrate_ac_mirroring_boundary,
    persist_todo_mirror,
    plan_rollout_seed_mirroring,
    render_rollout_todo_checklist,
    sync_rollout_todo_payload,
)


def _optimized_seed() -> RolloutSeed:
    return RolloutSeed(
        goal=(
            "Implement the optimized MiMo V2.5 Pro MTP rollout strategy: "
            "guarded exo-cluster MTP vertical slice."
        ),
        acceptance_criteria=(
            "AC-P0 Optimized rollout seed and repo context are validated:\n"
            "- mirror ACs into TODO/runlog/Beads where practical\n"
            "- identify focused verification commands for API/task/worker/benchmark changes",
            "AC-P1 Cluster AR baseline harness is canonical and budget-ready:\n"
            "- keep scripts/bench_mimo_mtp_cluster.py as the live benchmark path",
        ),
        ontology_name="MimoMtpOptimizedRollout",
        required_concepts=(
            "cluster_baseline",
            "mtp_vertical_slice",
            "benchmark_telemetry",
            "speedup_budget",
            "slice_5_gate",
        ),
        seed_id="mimo_mtp_optimized_rollout_20260606T203800",
    )


def test_plan_rollout_seed_mirroring_derives_todo_runlog_and_beads_operations() -> None:
    plan = plan_rollout_seed_mirroring(
        _optimized_seed(),
        todo_path="TODO.md",
        runlog_path="docs/plans/MIMO_V25_PRO_MTP_FOUNDATION_RUNLOG.md",
        beads_epic_id="mimo-v25-pro-mtp-d1-live-preflight-20260601-5ge",
        timestamp="2026-06-06T20:44:00",
    )

    assert (
        plan.summary
        == "Mirror 2 rollout ACs from mimo_mtp_optimized_rollout_20260606T203800"
    )
    assert plan.operations == (
        FileAppendOperation(
            path="TODO.md",
            heading="MiMo MTP optimized rollout mirror — 2026-06-06T20:44:00",
            content=(
                "- Seed `mimo_mtp_optimized_rollout_20260606T203800`: Implement the optimized MiMo V2.5 Pro MTP rollout strategy: guarded exo-cluster MTP vertical slice.\n"
                "- [ ] AC-P0 Optimized rollout seed and repo context are validated — cluster_baseline, mtp_vertical_slice, benchmark_telemetry, speedup_budget, slice_5_gate\n"
                "- [ ] AC-P1 Cluster AR baseline harness is canonical and budget-ready — cluster_baseline, benchmark_telemetry, speedup_budget, slice_5_gate\n"
                "- Slice 5 gate: default generation remains AR until same-cluster AR-vs-MTP rows prove a real MTP speed win."
            ),
        ),
        FileAppendOperation(
            path="docs/plans/MIMO_V25_PRO_MTP_FOUNDATION_RUNLOG.md",
            heading="2026-06-06T20:44:00 Optimized rollout seed mirror",
            content=(
                "Seed: `mimo_mtp_optimized_rollout_20260606T203800`\n\n"
                "Goal: Implement the optimized MiMo V2.5 Pro MTP rollout strategy: guarded exo-cluster MTP vertical slice.\n\n"
                "Mirrored acceptance criteria:\n"
                "1. AC-P0 Optimized rollout seed and repo context are validated\n"
                "2. AC-P1 Cluster AR baseline harness is canonical and budget-ready\n\n"
                "Required ontology concepts: cluster_baseline, mtp_vertical_slice, benchmark_telemetry, speedup_budget, slice_5_gate\n\n"
                "Evidence policy: benchmark rows are required before any >=30 tok/s, >=40 tok/s, or MTP speedup claim; absent live cluster rows must be recorded as blocked rather than fabricated."
            ),
        ),
        BeadsNoteOperation(
            issue_id="mimo-v25-pro-mtp-d1-live-preflight-20260601-5ge.1",
            note=(
                "[seed mirror 2026-06-06T20:44:00] AC-P0 Optimized rollout seed and repo context are validated; "
                "concepts=cluster_baseline, mtp_vertical_slice, benchmark_telemetry, speedup_budget, slice_5_gate; "
                "Slice 5 remains blocked without same-cluster AR-vs-MTP evidence."
            ),
        ),
        BeadsNoteOperation(
            issue_id="mimo-v25-pro-mtp-d1-live-preflight-20260601-5ge.2",
            note=(
                "[seed mirror 2026-06-06T20:44:00] AC-P1 Cluster AR baseline harness is canonical and budget-ready; "
                "concepts=cluster_baseline, benchmark_telemetry, speedup_budget, slice_5_gate; "
                "Slice 5 remains blocked without same-cluster AR-vs-MTP evidence."
            ),
        ),
    )


def test_plan_rollout_seed_mirroring_is_pure_and_returns_no_external_commands() -> None:
    plan = plan_rollout_seed_mirroring(
        _optimized_seed(),
        todo_path="TODO.md",
        runlog_path="docs/plans/MIMO_V25_PRO_MTP_FOUNDATION_RUNLOG.md",
        beads_epic_id=None,
        timestamp="2026-06-06T20:44:00",
    )

    assert (
        plan.operations
        == plan_rollout_seed_mirroring(
            _optimized_seed(),
            todo_path="TODO.md",
            runlog_path="docs/plans/MIMO_V25_PRO_MTP_FOUNDATION_RUNLOG.md",
            beads_epic_id=None,
            timestamp="2026-06-06T20:44:00",
        ).operations
    )
    assert all(not hasattr(operation, "command") for operation in plan.operations)
    assert not any(
        isinstance(operation, BeadsNoteOperation) for operation in plan.operations
    )


def test_plan_rollout_seed_mirroring_rejects_seed_without_acceptance_criteria() -> None:
    seed = RolloutSeed(
        goal="guarded rollout",
        acceptance_criteria=(),
        ontology_name="MimoMtpOptimizedRollout",
        required_concepts=("slice_5_gate",),
        seed_id="empty_seed",
    )

    try:
        plan_rollout_seed_mirroring(
            seed,
            todo_path="TODO.md",
            runlog_path="docs/plans/MIMO_V25_PRO_MTP_FOUNDATION_RUNLOG.md",
            beads_epic_id="mimo-v25-pro-mtp-d1-live-preflight-20260601-5ge",
            timestamp="2026-06-06T20:44:00",
        )
    except ValueError as exc:
        assert str(exc) == "rollout seed must contain at least one acceptance criterion"
    else:
        raise AssertionError("expected ValueError")


def _rollout_todo_payload() -> RolloutTodoPayload:
    return RolloutTodoPayload(
        seed_id="mimo_mtp_optimized_rollout_20260606T203800",
        goal="Guarded cluster MTP vertical slice with benchmark-grade telemetry.",
        entries=(
            RolloutTodoChecklistEntry(
                acceptance_id="AC-P0",
                title="Optimized rollout seed and repo context are validated",
                status="done",
                concepts=("cluster_baseline", "slice_5_gate"),
                evidence=(
                    "files changed: scripts/rollout_seed_mirroring.py",
                    "tests: uv run pytest scripts/test_rollout_seed_mirroring.py",
                ),
                remaining_risk="No live cluster rows for this bookkeeping-only Sub-AC.",
            ),
            RolloutTodoChecklistEntry(
                acceptance_id="AC-P1",
                title="Cluster AR baseline harness is canonical and budget-ready",
                status="blocked",
                concepts=(
                    "cluster_baseline",
                    "benchmark_telemetry",
                    "speedup_budget",
                ),
                evidence=(
                    "blocked command: uv run python3 scripts/bench_mimo_mtp_cluster.py --mode-label ar",
                ),
                remaining_risk="Cluster API unavailable; live AR row remains blocked.",
            ),
        ),
        slice_5_gate="blocked: default generation remains AR until same-cluster AR-vs-MTP rows prove a real win.",
    )


def test_render_rollout_todo_checklist_turns_payload_into_checklist_entries() -> None:
    checklist = render_rollout_todo_checklist(_rollout_todo_payload())

    assert checklist == (
        "- Seed `mimo_mtp_optimized_rollout_20260606T203800`: Guarded cluster MTP vertical slice with benchmark-grade telemetry.\n"
        "- Slice 5 gate: blocked: default generation remains AR until same-cluster AR-vs-MTP rows prove a real win.\n"
        "- [x] AC-P0 Optimized rollout seed and repo context are validated — concepts=cluster_baseline, slice_5_gate; evidence=files changed: scripts/rollout_seed_mirroring.py | tests: uv run pytest scripts/test_rollout_seed_mirroring.py; remaining_risk=No live cluster rows for this bookkeeping-only Sub-AC.\n"
        "- [!] AC-P1 Cluster AR baseline harness is canonical and budget-ready — concepts=cluster_baseline, benchmark_telemetry, speedup_budget; evidence=blocked command: uv run python3 scripts/bench_mimo_mtp_cluster.py --mode-label ar; remaining_risk=Cluster API unavailable; live AR row remains blocked."
    )


def test_sync_rollout_todo_payload_updates_entries_idempotently_without_losing_unrelated_content(
    tmp_path: Path,
) -> None:
    todo_path = tmp_path / "TODO.md"
    heading = "MiMo MTP optimized rollout checklist"
    todo_path.write_text(
        "# Existing TODO\n\n"
        f"## {heading}\n\n"
        "- Seed `mimo_mtp_optimized_rollout_20260606T203800`: stale goal.\n"
        "- [ ] AC-P0 stale status and stale title — old evidence\n"
        "- [ ] AC-P1 stale status and stale title — old evidence\n\n"
        "## Preserve unrelated TODOs\n\n"
        "- [ ] keep this manual item\n",
        encoding="utf-8",
    )

    first_action = sync_rollout_todo_payload(
        _rollout_todo_payload(),
        todo_path=todo_path,
        heading=heading,
    )
    second_action = sync_rollout_todo_payload(
        _rollout_todo_payload(),
        todo_path=todo_path,
        heading=heading,
    )

    todo_text = todo_path.read_text(encoding="utf-8")
    assert first_action == "updated"
    assert second_action == "unchanged"
    assert todo_text.count(f"## {heading}") == 1
    assert todo_text.count("AC-P0 Optimized rollout seed") == 1
    assert todo_text.count("AC-P1 Cluster AR baseline") == 1
    assert "stale status" not in todo_text
    assert "## Preserve unrelated TODOs\n\n- [ ] keep this manual item" in todo_text


def test_todo_mirror_sink_creates_todo_entries_when_file_is_missing(
    tmp_path: Path,
) -> None:
    todo_path = tmp_path / "TODO.md"
    operation = FileAppendOperation(
        path=str(todo_path),
        heading="MiMo MTP optimized rollout mirror — 2026-06-06T22:47:00",
        content=(
            "- Seed `mimo_mtp_optimized_rollout_20260606T203800`: guarded rollout.\n"
            "- [ ] AC-P0 Optimized rollout seed and repo context are validated — slice_5_gate\n"
            "- [ ] AC-P1 Cluster AR baseline harness is canonical and budget-ready — cluster_baseline\n"
            "- Slice 5 gate: default generation remains AR until same-cluster AR-vs-MTP rows prove a real MTP speed win."
        ),
    )

    action = persist_todo_mirror(operation)

    todo_text = todo_path.read_text(encoding="utf-8")
    assert action == "created"
    assert "## MiMo MTP optimized rollout mirror — 2026-06-06T22:47:00" in todo_text
    assert (
        "- [ ] AC-P0 Optimized rollout seed and repo context are validated" in todo_text
    )
    assert (
        "- [ ] AC-P1 Cluster AR baseline harness is canonical and budget-ready"
        in todo_text
    )
    assert "Slice 5 gate: default generation remains AR" in todo_text


def test_todo_mirror_sink_updates_existing_todo_entries_without_duplicate_sections(
    tmp_path: Path,
) -> None:
    todo_path = tmp_path / "TODO.md"
    heading = "MiMo MTP optimized rollout mirror — 2026-06-06T22:47:00"
    todo_path.write_text(
        "# Operator TODO\n\n"
        f"## {heading}\n\n"
        "- [ ] AC-P0 stale title — old_concept\n"
        "- [ ] AC-P1 stale title — old_concept\n\n"
        "## Preserve unrelated TODOs\n\n"
        "- [ ] keep this manual item\n",
        encoding="utf-8",
    )
    operation = FileAppendOperation(
        path=str(todo_path),
        heading=heading,
        content=(
            "- Seed `mimo_mtp_optimized_rollout_20260606T203800`: guarded rollout.\n"
            "- [x] AC-P0 Optimized rollout seed and repo context are validated — slice_5_gate\n"
            "- [ ] AC-P1 Cluster AR baseline harness is canonical and budget-ready — cluster_baseline\n"
            "- Slice 5 gate: default generation remains AR until same-cluster AR-vs-MTP rows prove a real MTP speed win."
        ),
    )

    action = persist_todo_mirror(operation)

    todo_text = todo_path.read_text(encoding="utf-8")
    assert action == "updated"
    assert todo_text.count(f"## {heading}") == 1
    assert "AC-P0 stale title" not in todo_text
    assert (
        "- [x] AC-P0 Optimized rollout seed and repo context are validated" in todo_text
    )
    assert (
        "- [ ] AC-P1 Cluster AR baseline harness is canonical and budget-ready"
        in todo_text
    )
    assert "## Preserve unrelated TODOs\n\n- [ ] keep this manual item" in todo_text


def test_todo_mirror_sink_does_not_invoke_runlog_or_beads_dependencies(
    tmp_path: Path,
) -> None:
    todo_path = tmp_path / "TODO.md"
    calls: list[str] = []
    operation = FileAppendOperation(
        path=str(todo_path),
        heading="MiMo MTP optimized rollout mirror — 2026-06-06T22:47:00",
        content="- [ ] AC-P0 Optimized rollout seed and repo context are validated — slice_5_gate",
    )

    def forbidden_runlog_write(_operation: FileAppendOperation) -> None:
        calls.append("runlog")
        raise AssertionError("TODO sink must not invoke runlog persistence")

    def forbidden_beads_write(_operation: BeadsNoteOperation) -> None:
        calls.append("beads")
        raise AssertionError("TODO sink must not invoke Beads persistence")

    action = persist_todo_mirror(
        operation,
        persist_runlog=forbidden_runlog_write,
        persist_beads=forbidden_beads_write,
    )

    assert action == "created"
    assert calls == []


def test_ac_mirroring_boundary_invokes_only_mirroring_dependencies() -> None:
    calls: list[str] = []

    def plan_mirror() -> str:
        calls.append("plan_mirror")
        return "planned"

    def persist_files(plan: str) -> str:
        calls.append(f"persist_files:{plan}")
        return "files persisted"

    def persist_beads(plan: str) -> str:
        calls.append(f"persist_beads:{plan}")
        return "beads persisted"

    def validate_repo_root() -> None:
        calls.append("validate_repo_root")
        raise AssertionError("repo root validation must not run in mirroring boundary")

    def discover_verification_commands() -> None:
        calls.append("discover_verification_commands")
        raise AssertionError(
            "verification command discovery must not run in mirroring boundary"
        )

    result = orchestrate_ac_mirroring_boundary(
        AcMirroringBoundaryDependencies(
            plan_mirror=plan_mirror,
            persist_files=persist_files,
            persist_beads=persist_beads,
            validate_repo_root=validate_repo_root,
            discover_verification_commands=discover_verification_commands,
        )
    )

    assert result == AcMirroringBoundaryResult(
        plan="planned",
        file_result="files persisted",
        beads_result="beads persisted",
    )
    assert calls == [
        "plan_mirror",
        "persist_files:planned",
        "persist_beads:planned",
    ]
