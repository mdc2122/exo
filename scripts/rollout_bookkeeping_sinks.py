"""Guarded rollout bookkeeping sink discovery and routing helpers.

The functions in this module are pure aside from injected availability probes.
They discover which operator bookkeeping sinks are explicitly enabled,
configured, and currently available, then route AC metadata only to those sinks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

BookkeepingSinkName: TypeAlias = Literal["todo", "runlog", "beads"]
BookkeepingPathProbe: TypeAlias = Callable[[Path], bool]
BookkeepingCommandProbe: TypeAlias = Callable[[str], bool]

_ALL_SINKS: tuple[BookkeepingSinkName, ...] = ("todo", "runlog", "beads")


@dataclass(frozen=True, slots=True)
class AcBookkeepingMetadata:
    """Acceptance-criterion metadata to mirror into operator bookkeeping."""

    seed_id: str
    acceptance_index: int
    acceptance_id: str
    title: str
    summary: str
    evidence_path: str


@dataclass(frozen=True, slots=True)
class BookkeepingSinkConfig:
    """Explicit sink configuration for guarded bookkeeping discovery."""

    enabled_sinks: frozenset[BookkeepingSinkName]
    todo_path: Path | None = None
    runlog_path: Path | None = None
    beads_epic_id: str | None = None
    beads_command: str = "bd"


@dataclass(frozen=True, slots=True)
class BookkeepingSinkStatus:
    """Discovered state for a single bookkeeping sink."""

    name: BookkeepingSinkName
    enabled: bool
    configured: bool
    available: bool
    target: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class FileBookkeepingOperation:
    """A planned append operation for a file-backed bookkeeping sink."""

    sink: Literal["todo", "runlog"]
    path: str
    heading: str
    content: str


@dataclass(frozen=True, slots=True)
class BeadsBookkeepingOperation:
    """A planned Beads note operation for AC bookkeeping."""

    issue_id: str
    note: str


BookkeepingOperation: TypeAlias = FileBookkeepingOperation | BeadsBookkeepingOperation
FileBookkeepingDispatcher: TypeAlias = Callable[[FileBookkeepingOperation], None]
BeadsBookkeepingDispatcher: TypeAlias = Callable[[BeadsBookkeepingOperation], None]


@dataclass(frozen=True, slots=True)
class BookkeepingDispatchFailure:
    """A sink dispatch failure captured without stopping other sinks."""

    sink: BookkeepingSinkName
    target: str
    error: str


@dataclass(frozen=True, slots=True)
class BookkeepingDispatchResult:
    """Outcome from guarded bookkeeping dispatch."""

    successful_operations: tuple[BookkeepingOperation, ...]
    failures: tuple[BookkeepingDispatchFailure, ...]


def _target_path_is_available(path: Path, path_exists: BookkeepingPathProbe) -> bool:
    return path_exists(path) or path_exists(path.parent)


def _file_sink_status(
    *,
    name: Literal["todo", "runlog"],
    enabled: bool,
    path: Path | None,
    path_exists: BookkeepingPathProbe,
) -> BookkeepingSinkStatus:
    configured = path is not None
    target = None if path is None else str(path)
    if not enabled:
        return BookkeepingSinkStatus(
            name=name,
            enabled=False,
            configured=configured,
            available=False,
            target=target,
            reason="sink is disabled",
        )
    if path is None:
        return BookkeepingSinkStatus(
            name=name,
            enabled=True,
            configured=False,
            available=False,
            target=None,
            reason="sink is not configured",
        )
    if not _target_path_is_available(path, path_exists):
        return BookkeepingSinkStatus(
            name=name,
            enabled=True,
            configured=True,
            available=False,
            target=target,
            reason="configured target is unavailable",
        )
    return BookkeepingSinkStatus(
        name=name,
        enabled=True,
        configured=True,
        available=True,
        target=target,
        reason="enabled configured target is available",
    )


def _beads_sink_status(
    *,
    enabled: bool,
    epic_id: str | None,
    command: str,
    command_available: BookkeepingCommandProbe,
) -> BookkeepingSinkStatus:
    configured = epic_id is not None and epic_id != "" and command != ""
    if not enabled:
        return BookkeepingSinkStatus(
            name="beads",
            enabled=False,
            configured=configured,
            available=False,
            target=epic_id,
            reason="sink is disabled",
        )
    if not configured:
        return BookkeepingSinkStatus(
            name="beads",
            enabled=True,
            configured=False,
            available=False,
            target=epic_id,
            reason="sink is not configured",
        )
    if not command_available(command):
        return BookkeepingSinkStatus(
            name="beads",
            enabled=True,
            configured=True,
            available=False,
            target=epic_id,
            reason="configured target is unavailable",
        )
    return BookkeepingSinkStatus(
        name="beads",
        enabled=True,
        configured=True,
        available=True,
        target=epic_id,
        reason="enabled configured target is available",
    )


def discover_bookkeeping_sinks(
    config: BookkeepingSinkConfig,
    *,
    path_exists: BookkeepingPathProbe,
    command_available: BookkeepingCommandProbe,
) -> tuple[BookkeepingSinkStatus, ...]:
    """Discover configured and available bookkeeping sinks.

    A sink is routable only when it is explicitly enabled, configured, and
    available. Disabled sinks are always reported as unavailable even when their
    targets exist, preserving fail-closed routing semantics.
    """

    enabled_sinks = config.enabled_sinks
    return (
        _file_sink_status(
            name="todo",
            enabled="todo" in enabled_sinks,
            path=config.todo_path,
            path_exists=path_exists,
        ),
        _file_sink_status(
            name="runlog",
            enabled="runlog" in enabled_sinks,
            path=config.runlog_path,
            path_exists=path_exists,
        ),
        _beads_sink_status(
            enabled="beads" in enabled_sinks,
            epic_id=config.beads_epic_id,
            command=config.beads_command,
            command_available=command_available,
        ),
    )


def _status_by_name(
    statuses: tuple[BookkeepingSinkStatus, ...],
) -> dict[BookkeepingSinkName, BookkeepingSinkStatus]:
    return {status.name: status for status in statuses if status.name in _ALL_SINKS}


def _is_routable(status: BookkeepingSinkStatus | None) -> bool:
    return (
        status is not None
        and status.enabled
        and status.configured
        and status.available
        and status.target is not None
    )


def _todo_operation(
    metadata: AcBookkeepingMetadata,
    *,
    target: str,
    timestamp: str,
    mirror_refs: str = "",
) -> FileBookkeepingOperation:
    return FileBookkeepingOperation(
        sink="todo",
        path=target,
        heading=f"MiMo MTP rollout bookkeeping — {timestamp}",
        content=(
            f"- [ ] {metadata.acceptance_id} {metadata.title} "
            f"— seed={metadata.seed_id}; "
            f"evidence={metadata.evidence_path}; "
            f"summary={metadata.summary}"
            f"{'; ' + mirror_refs if mirror_refs else ''}"
        ),
    )


def _runlog_operation(
    metadata: AcBookkeepingMetadata,
    *,
    target: str,
    timestamp: str,
    mirror_refs: str = "",
) -> FileBookkeepingOperation:
    return FileBookkeepingOperation(
        sink="runlog",
        path=target,
        heading=f"{timestamp} {metadata.acceptance_id} {metadata.title}",
        content=(
            f"Seed: `{metadata.seed_id}`\n\n"
            f"Acceptance criterion: {metadata.acceptance_id} {metadata.title}\n\n"
            f"Summary: {metadata.summary}\n\n"
            f"Evidence: {metadata.evidence_path}"
            f"{chr(10) + chr(10) + mirror_refs if mirror_refs else ''}"
        ),
    )


def _beads_operation(
    metadata: AcBookkeepingMetadata,
    *,
    target: str,
    timestamp: str,
    mirror_refs: str = "",
) -> BeadsBookkeepingOperation:
    return BeadsBookkeepingOperation(
        issue_id=f"{target}.{metadata.acceptance_index}",
        note=(
            f"[rollout bookkeeping {timestamp}] {metadata.acceptance_id} {metadata.title}; "
            f"seed={metadata.seed_id}; "
            f"evidence={metadata.evidence_path}; "
            f"summary={metadata.summary}"
            f"{'; ' + mirror_refs if mirror_refs else ''}"
        ),
    )


def _mirror_refs_for_routable_sinks(
    metadata: AcBookkeepingMetadata,
    statuses: dict[BookkeepingSinkName, BookkeepingSinkStatus],
) -> str:
    refs: list[str] = []

    todo_status = statuses.get("todo")
    if _is_routable(todo_status) and todo_status is not None and todo_status.target:
        refs.append(f"todo:{todo_status.target}")

    runlog_status = statuses.get("runlog")
    if (
        _is_routable(runlog_status)
        and runlog_status is not None
        and runlog_status.target
    ):
        refs.append(f"runlog:{runlog_status.target}")

    beads_status = statuses.get("beads")
    if _is_routable(beads_status) and beads_status is not None and beads_status.target:
        refs.append(f"beads:{beads_status.target}.{metadata.acceptance_index}")

    return "mirror_refs=" + ";".join(refs) if len(refs) > 1 else ""


def route_ac_bookkeeping(
    metadata: AcBookkeepingMetadata,
    *,
    sink_statuses: tuple[BookkeepingSinkStatus, ...],
    timestamp: str,
) -> tuple[BookkeepingOperation, ...]:
    """Route AC metadata only to enabled, configured, available sinks."""

    statuses = _status_by_name(sink_statuses)
    operations: list[BookkeepingOperation] = []
    mirror_refs = _mirror_refs_for_routable_sinks(metadata, statuses)

    todo_status = statuses.get("todo")
    if _is_routable(todo_status) and todo_status is not None and todo_status.target:
        operations.append(
            _todo_operation(
                metadata,
                target=todo_status.target,
                timestamp=timestamp,
                mirror_refs=mirror_refs,
            )
        )

    runlog_status = statuses.get("runlog")
    if (
        _is_routable(runlog_status)
        and runlog_status is not None
        and runlog_status.target
    ):
        operations.append(
            _runlog_operation(
                metadata,
                target=runlog_status.target,
                timestamp=timestamp,
                mirror_refs=mirror_refs,
            )
        )

    beads_status = statuses.get("beads")
    if _is_routable(beads_status) and beads_status is not None and beads_status.target:
        operations.append(
            _beads_operation(
                metadata,
                target=beads_status.target,
                timestamp=timestamp,
                mirror_refs=mirror_refs,
            )
        )

    return tuple(operations)


def _operation_sink(operation: BookkeepingOperation) -> BookkeepingSinkName:
    if isinstance(operation, FileBookkeepingOperation):
        return operation.sink
    return "beads"


def _operation_target(operation: BookkeepingOperation) -> str:
    if isinstance(operation, FileBookkeepingOperation):
        return operation.path
    return operation.issue_id


def route_and_dispatch_ac_bookkeeping(
    metadata: AcBookkeepingMetadata,
    *,
    sink_statuses: tuple[BookkeepingSinkStatus, ...],
    timestamp: str,
    dispatch_todo: FileBookkeepingDispatcher,
    dispatch_runlog: FileBookkeepingDispatcher,
    dispatch_beads: BeadsBookkeepingDispatcher,
) -> BookkeepingDispatchResult:
    """Route and dispatch AC metadata with per-sink error isolation.

    Disabled, unconfigured, or unavailable sinks are graceful no-ops because
    `route_ac_bookkeeping` emits operations only for routable sinks. Dispatch
    failures are captured per operation so one failing sink does not prevent
    later configured sinks from executing.
    """

    successful_operations: list[BookkeepingOperation] = []
    failures: list[BookkeepingDispatchFailure] = []

    for operation in route_ac_bookkeeping(
        metadata,
        sink_statuses=sink_statuses,
        timestamp=timestamp,
    ):
        try:
            if isinstance(operation, BeadsBookkeepingOperation):
                dispatch_beads(operation)
            elif operation.sink == "todo":
                dispatch_todo(operation)
            else:
                dispatch_runlog(operation)
        except Exception as exc:  # noqa: BLE001 - isolate bookkeeping sink failures
            failures.append(
                BookkeepingDispatchFailure(
                    sink=_operation_sink(operation),
                    target=_operation_target(operation),
                    error=str(exc),
                )
            )
            continue
        successful_operations.append(operation)

    return BookkeepingDispatchResult(
        successful_operations=tuple(successful_operations),
        failures=tuple(failures),
    )
