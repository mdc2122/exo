from __future__ import annotations

from pathlib import Path
from typing import NoReturn

from scripts.rollout_bookkeeping_sinks import (
    AcBookkeepingMetadata,
    BeadsBookkeepingOperation,
    BookkeepingDispatchFailure,
    BookkeepingDispatchResult,
    BookkeepingSinkConfig,
    BookkeepingSinkStatus,
    FileBookkeepingOperation,
    discover_bookkeeping_sinks,
    route_ac_bookkeeping,
    route_and_dispatch_ac_bookkeeping,
)


def _metadata() -> AcBookkeepingMetadata:
    return AcBookkeepingMetadata(
        seed_id="mimo_mtp_optimized_rollout_20260606T203800",
        acceptance_index=1,
        acceptance_id="AC-P1",
        title="Cluster AR baseline harness is canonical and budget-ready",
        summary="Collect same-cluster AR baseline rows before MTP claims.",
        evidence_path=".goose-ultrawork/evidence/ac-p1-summary.md",
    )


def _path_exists(path: Path, existing_paths: set[Path]) -> bool:
    return path in existing_paths


def test_discover_bookkeeping_sinks_marks_configured_available_enabled_targets(
    tmp_path: Path,
) -> None:
    todo_path = tmp_path / "TODO.md"
    runlog_path = tmp_path / "docs" / "plans" / "MIMO_V25_PRO_MTP_FOUNDATION_RUNLOG.md"
    existing_paths = {todo_path, runlog_path.parent}

    statuses = discover_bookkeeping_sinks(
        BookkeepingSinkConfig(
            enabled_sinks=frozenset({"todo", "runlog", "beads"}),
            todo_path=todo_path,
            runlog_path=runlog_path,
            beads_epic_id="mimo-v25-pro-mtp-d1-live-preflight-20260601-5ge",
            beads_command="bd",
        ),
        path_exists=lambda path: _path_exists(path, existing_paths),
        command_available=lambda command: command == "bd",
    )

    assert statuses == (
        BookkeepingSinkStatus(
            name="todo",
            enabled=True,
            configured=True,
            available=True,
            target=str(todo_path),
            reason="enabled configured target is available",
        ),
        BookkeepingSinkStatus(
            name="runlog",
            enabled=True,
            configured=True,
            available=True,
            target=str(runlog_path),
            reason="enabled configured target is available",
        ),
        BookkeepingSinkStatus(
            name="beads",
            enabled=True,
            configured=True,
            available=True,
            target="mimo-v25-pro-mtp-d1-live-preflight-20260601-5ge",
            reason="enabled configured target is available",
        ),
    )


def test_discover_bookkeeping_sinks_records_disabled_unconfigured_and_unavailable_targets(
    tmp_path: Path,
) -> None:
    runlog_path = tmp_path / "docs" / "plans" / "RUNLOG.md"

    statuses = discover_bookkeeping_sinks(
        BookkeepingSinkConfig(
            enabled_sinks=frozenset({"todo", "beads"}),
            todo_path=tmp_path / "missing" / "TODO.md",
            runlog_path=runlog_path,
            beads_epic_id=None,
            beads_command="bd",
        ),
        path_exists=lambda _path: False,
        command_available=lambda _command: False,
    )

    assert statuses == (
        BookkeepingSinkStatus(
            name="todo",
            enabled=True,
            configured=True,
            available=False,
            target=str(tmp_path / "missing" / "TODO.md"),
            reason="configured target is unavailable",
        ),
        BookkeepingSinkStatus(
            name="runlog",
            enabled=False,
            configured=True,
            available=False,
            target=str(runlog_path),
            reason="sink is disabled",
        ),
        BookkeepingSinkStatus(
            name="beads",
            enabled=True,
            configured=False,
            available=False,
            target=None,
            reason="sink is not configured",
        ),
    )


def test_route_ac_bookkeeping_emits_only_enabled_configured_available_sinks() -> None:
    operations = route_ac_bookkeeping(
        _metadata(),
        sink_statuses=(
            BookkeepingSinkStatus(
                name="todo",
                enabled=True,
                configured=True,
                available=True,
                target="TODO.md",
                reason="enabled configured target is available",
            ),
            BookkeepingSinkStatus(
                name="runlog",
                enabled=False,
                configured=True,
                available=True,
                target="docs/plans/RUNLOG.md",
                reason="sink is disabled",
            ),
            BookkeepingSinkStatus(
                name="beads",
                enabled=True,
                configured=True,
                available=False,
                target="mimo-v25-pro-mtp-d1-live-preflight-20260601-5ge",
                reason="configured target is unavailable",
            ),
        ),
        timestamp="2026-06-06T22:47:00",
    )

    assert operations == (
        FileBookkeepingOperation(
            sink="todo",
            path="TODO.md",
            heading="MiMo MTP rollout bookkeeping — 2026-06-06T22:47:00",
            content=(
                "- [ ] AC-P1 Cluster AR baseline harness is canonical and budget-ready "
                "— seed=mimo_mtp_optimized_rollout_20260606T203800; "
                "evidence=.goose-ultrawork/evidence/ac-p1-summary.md; "
                "summary=Collect same-cluster AR baseline rows before MTP claims."
            ),
        ),
    )


def test_route_ac_bookkeeping_uses_ac_metadata_for_todo_runlog_and_beads_notes() -> (
    None
):
    operations = route_ac_bookkeeping(
        _metadata(),
        sink_statuses=(
            BookkeepingSinkStatus(
                name="todo",
                enabled=True,
                configured=True,
                available=True,
                target="TODO.md",
                reason="enabled configured target is available",
            ),
            BookkeepingSinkStatus(
                name="runlog",
                enabled=True,
                configured=True,
                available=True,
                target="docs/plans/RUNLOG.md",
                reason="enabled configured target is available",
            ),
            BookkeepingSinkStatus(
                name="beads",
                enabled=True,
                configured=True,
                available=True,
                target="mimo-v25-pro-mtp-d1-live-preflight-20260601-5ge",
                reason="enabled configured target is available",
            ),
        ),
        timestamp="2026-06-06T22:47:00",
    )

    assert operations == (
        FileBookkeepingOperation(
            sink="todo",
            path="TODO.md",
            heading="MiMo MTP rollout bookkeeping — 2026-06-06T22:47:00",
            content=(
                "- [ ] AC-P1 Cluster AR baseline harness is canonical and budget-ready "
                "— seed=mimo_mtp_optimized_rollout_20260606T203800; "
                "evidence=.goose-ultrawork/evidence/ac-p1-summary.md; "
                "summary=Collect same-cluster AR baseline rows before MTP claims.; "
                "mirror_refs=todo:TODO.md;runlog:docs/plans/RUNLOG.md;"
                "beads:mimo-v25-pro-mtp-d1-live-preflight-20260601-5ge.1"
            ),
        ),
        FileBookkeepingOperation(
            sink="runlog",
            path="docs/plans/RUNLOG.md",
            heading="2026-06-06T22:47:00 AC-P1 Cluster AR baseline harness is canonical and budget-ready",
            content=(
                "Seed: `mimo_mtp_optimized_rollout_20260606T203800`\n\n"
                "Acceptance criterion: AC-P1 Cluster AR baseline harness is canonical and budget-ready\n\n"
                "Summary: Collect same-cluster AR baseline rows before MTP claims.\n\n"
                "Evidence: .goose-ultrawork/evidence/ac-p1-summary.md\n\n"
                "mirror_refs=todo:TODO.md;runlog:docs/plans/RUNLOG.md;"
                "beads:mimo-v25-pro-mtp-d1-live-preflight-20260601-5ge.1"
            ),
        ),
        BeadsBookkeepingOperation(
            issue_id="mimo-v25-pro-mtp-d1-live-preflight-20260601-5ge.1",
            note=(
                "[rollout bookkeeping 2026-06-06T22:47:00] AC-P1 Cluster AR baseline harness is canonical and budget-ready; "
                "seed=mimo_mtp_optimized_rollout_20260606T203800; "
                "evidence=.goose-ultrawork/evidence/ac-p1-summary.md; "
                "summary=Collect same-cluster AR baseline rows before MTP claims.; "
                "mirror_refs=todo:TODO.md;runlog:docs/plans/RUNLOG.md;"
                "beads:mimo-v25-pro-mtp-d1-live-preflight-20260601-5ge.1"
            ),
        ),
    )


def _raise_runtime_error(_operation: object) -> NoReturn:
    raise RuntimeError("todo sink failed")


def test_route_and_dispatch_ac_bookkeeping_skips_unavailable_sinks_and_isolates_failures() -> (
    None
):
    dispatched_sinks: list[str] = []

    def dispatch_todo(operation: FileBookkeepingOperation) -> None:
        dispatched_sinks.append(operation.sink)
        _raise_runtime_error(operation)

    def dispatch_runlog(operation: FileBookkeepingOperation) -> None:
        dispatched_sinks.append(operation.sink)

    def dispatch_beads(_operation: BeadsBookkeepingOperation) -> None:
        dispatched_sinks.append("beads")

    result = route_and_dispatch_ac_bookkeeping(
        _metadata(),
        sink_statuses=(
            BookkeepingSinkStatus(
                name="todo",
                enabled=True,
                configured=True,
                available=True,
                target="TODO.md",
                reason="enabled configured target is available",
            ),
            BookkeepingSinkStatus(
                name="runlog",
                enabled=True,
                configured=True,
                available=True,
                target="docs/plans/RUNLOG.md",
                reason="enabled configured target is available",
            ),
            BookkeepingSinkStatus(
                name="beads",
                enabled=True,
                configured=False,
                available=False,
                target=None,
                reason="sink is not configured",
            ),
        ),
        timestamp="2026-06-06T22:47:00",
        dispatch_todo=dispatch_todo,
        dispatch_runlog=dispatch_runlog,
        dispatch_beads=dispatch_beads,
    )

    assert dispatched_sinks == ["todo", "runlog"]
    assert result == BookkeepingDispatchResult(
        successful_operations=(
            FileBookkeepingOperation(
                sink="runlog",
                path="docs/plans/RUNLOG.md",
                heading="2026-06-06T22:47:00 AC-P1 Cluster AR baseline harness is canonical and budget-ready",
                content=(
                    "Seed: `mimo_mtp_optimized_rollout_20260606T203800`\n\n"
                    "Acceptance criterion: AC-P1 Cluster AR baseline harness is canonical and budget-ready\n\n"
                    "Summary: Collect same-cluster AR baseline rows before MTP claims.\n\n"
                    "Evidence: .goose-ultrawork/evidence/ac-p1-summary.md\n\n"
                    "mirror_refs=todo:TODO.md;runlog:docs/plans/RUNLOG.md"
                ),
            ),
        ),
        failures=(
            BookkeepingDispatchFailure(
                sink="todo",
                target="TODO.md",
                error="todo sink failed",
            ),
        ),
    )


def test_route_ac_bookkeeping_cross_references_todo_runlog_and_beads_mirrors() -> None:
    operations = route_ac_bookkeeping(
        _metadata(),
        sink_statuses=(
            BookkeepingSinkStatus(
                name="todo",
                enabled=True,
                configured=True,
                available=True,
                target="TODO.md",
                reason="enabled configured target is available",
            ),
            BookkeepingSinkStatus(
                name="runlog",
                enabled=True,
                configured=True,
                available=True,
                target="docs/plans/RUNLOG.md",
                reason="enabled configured target is available",
            ),
            BookkeepingSinkStatus(
                name="beads",
                enabled=True,
                configured=True,
                available=True,
                target="mimo-v25-pro-mtp-d1-live-preflight-20260601-5ge",
                reason="enabled configured target is available",
            ),
        ),
        timestamp="2026-06-07T03:40:00",
    )

    todo, runlog, beads = operations
    expected_refs = (
        "mirror_refs=todo:TODO.md;"
        "runlog:docs/plans/RUNLOG.md;"
        "beads:mimo-v25-pro-mtp-d1-live-preflight-20260601-5ge.1"
    )

    assert isinstance(todo, FileBookkeepingOperation)
    assert isinstance(runlog, FileBookkeepingOperation)
    assert isinstance(beads, BeadsBookkeepingOperation)
    assert expected_refs in todo.content
    assert expected_refs in runlog.content
    assert expected_refs in beads.note
