#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

from scripts.rollout_seed_mirroring import RolloutTodoChecklistEntry, RolloutTodoPayload

JsonObject = dict[str, object]
BeadsRunner = Callable[[list[str]], "CommandResult"]
BeadMirrorAction = Literal[
    "created", "updated", "skipped_unconfigured", "skipped_unavailable"
]


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class BeadMirrorEntry:
    seed_id: str
    acceptance_index: int
    title: str
    description: str
    acceptance_criteria: str
    labels: tuple[str, ...]
    priority: str
    issue_type: str
    parent_id: str | None


@dataclass(frozen=True, slots=True)
class BeadMirrorResult:
    seed_mirror_key: str
    action: BeadMirrorAction
    bead_id: str | None


@dataclass(frozen=True, slots=True)
class BeadsMirrorConfig:
    enabled: bool


class BeadMirrorError(RuntimeError):
    pass


def _run_command(command: list[str]) -> CommandResult:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _seed_mirror_key(entry: BeadMirrorEntry) -> str:
    return f"{entry.seed_id}:ac-{entry.acceptance_index}"


def _metadata_json(entry: BeadMirrorEntry, seed_mirror_key: str) -> str:
    metadata: JsonObject = {
        "acceptance_index": entry.acceptance_index,
        "seed_id": entry.seed_id,
        "seed_mirror_key": seed_mirror_key,
    }
    return json.dumps(metadata, separators=(",", ":"))


def _shared_write_flags(entry: BeadMirrorEntry, seed_mirror_key: str) -> list[str]:
    flags = [
        "--type",
        entry.issue_type,
        "--priority",
        entry.priority,
        "--description",
        entry.description,
        "--acceptance",
        entry.acceptance_criteria,
        "--metadata",
        _metadata_json(entry, seed_mirror_key),
    ]
    if entry.labels:
        flags[8:8] = ["--labels", ",".join(entry.labels)]
    if entry.parent_id is not None:
        flags.extend(["--parent", entry.parent_id])
    return flags


def _create_command(entry: BeadMirrorEntry, seed_mirror_key: str) -> list[str]:
    return ["bd", "create", entry.title, *_shared_write_flags(entry, seed_mirror_key)]


def _update_command(
    entry: BeadMirrorEntry, seed_mirror_key: str, bead_id: str
) -> list[str]:
    flags = _shared_write_flags(entry, seed_mirror_key)
    labels_index = flags.index("--labels") if "--labels" in flags else -1
    if labels_index >= 0:
        flags[labels_index] = "--set-labels"
    return ["bd", "update", bead_id, "--title", entry.title, *flags]


def _query_command(seed_mirror_key: str) -> list[str]:
    return [
        "bd",
        "list",
        "--json",
        "--metadata-field",
        f"seed_mirror_key={seed_mirror_key}",
        "--limit",
        "0",
    ]


def _run_checked(runner: BeadsRunner, command: list[str]) -> CommandResult:
    result = runner(command)
    if result.returncode != 0:
        raise BeadMirrorError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr}"
        )
    return result


def _is_beads_available(runner: BeadsRunner) -> bool:
    try:
        return runner(["bd", "--version"]).returncode == 0
    except FileNotFoundError:
        return False


def _skipped_results(
    entries: Iterable[BeadMirrorEntry], action: BeadMirrorAction
) -> list[BeadMirrorResult]:
    return [
        BeadMirrorResult(
            seed_mirror_key=_seed_mirror_key(entry),
            action=action,
            bead_id=None,
        )
        for entry in entries
    ]


def _write_result_artifact(
    result_path: Path | None, results: Iterable[BeadMirrorResult]
) -> None:
    if result_path is None:
        return
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        "".join(
            f"{json.dumps(asdict(result), sort_keys=True)}\n" for result in results
        ),
        encoding="utf-8",
    )


def _entry_concepts_text(entry: RolloutTodoChecklistEntry) -> str:
    return ", ".join(entry.concepts)


def _entry_evidence_text(entry: RolloutTodoChecklistEntry) -> str:
    return " | ".join(entry.evidence)


def entry_from_rollout_payload_entry(
    *,
    payload: RolloutTodoPayload,
    entry: RolloutTodoChecklistEntry,
    acceptance_index: int,
    labels: Sequence[str] = ("mimo", "mtp", "rollout", "payload-mirror"),
    priority: str = "P1",
    issue_type: str = "task",
    parent_id: str | None = None,
) -> BeadMirrorEntry:
    """Convert one canonical rollout status entry into a Beads item payload."""

    title = f"{entry.acceptance_id} {entry.title}"
    description = (
        f"status={entry.status}; "
        f"concepts={_entry_concepts_text(entry)}; "
        f"evidence={_entry_evidence_text(entry)}; "
        f"remaining_risk={entry.remaining_risk}; "
        f"slice_5_gate={payload.slice_5_gate}"
    )
    return BeadMirrorEntry(
        seed_id=payload.seed_id,
        acceptance_index=acceptance_index,
        title=title,
        description=description,
        acceptance_criteria=title,
        labels=tuple(labels),
        priority=priority,
        issue_type=issue_type,
        parent_id=parent_id,
    )


def entries_from_rollout_payload(
    payload: RolloutTodoPayload,
    *,
    labels: Sequence[str] = ("mimo", "mtp", "rollout", "payload-mirror"),
    priority: str = "P1",
    issue_type: str = "task",
    parent_id: str | None = None,
) -> list[BeadMirrorEntry]:
    """Convert canonical rollout payload entries into Beads mirror entries."""

    return [
        entry_from_rollout_payload_entry(
            payload=payload,
            entry=entry,
            acceptance_index=index,
            labels=labels,
            priority=priority,
            issue_type=issue_type,
            parent_id=parent_id,
        )
        for index, entry in enumerate(payload.entries)
    ]


def _decode_beads_list(raw_output: str) -> list[JsonObject]:
    decoded = cast(object, json.loads(raw_output))
    if not isinstance(decoded, list):
        raise BeadMirrorError("bd list --json returned a non-list JSON value")
    decoded_items = cast(list[object], decoded)
    beads: list[JsonObject] = []
    for item in decoded_items:
        if not isinstance(item, dict):
            raise BeadMirrorError("bd list --json returned a non-object bead row")
        beads.append(cast(JsonObject, item))
    return beads


def _require_bead_id(bead: JsonObject) -> str:
    bead_id = bead.get("id")
    if not isinstance(bead_id, str) or bead_id == "":
        raise BeadMirrorError("bd list --json returned a bead row without a string id")
    return bead_id


def _find_existing_beads(seed_mirror_key: str, runner: BeadsRunner) -> list[JsonObject]:
    result = _run_checked(runner, _query_command(seed_mirror_key))
    return _decode_beads_list(result.stdout)


def mirror_entries_to_configured_beads(
    entries: Iterable[BeadMirrorEntry],
    *,
    config: BeadsMirrorConfig,
    runner: BeadsRunner = _run_command,
) -> list[BeadMirrorResult]:
    materialized_entries = tuple(entries)
    if not config.enabled:
        return _skipped_results(materialized_entries, "skipped_unconfigured")
    if not _is_beads_available(runner):
        return _skipped_results(materialized_entries, "skipped_unavailable")
    return mirror_entries_to_beads(materialized_entries, runner=runner)


def sync_rollout_payload_to_configured_beads(
    payload: RolloutTodoPayload,
    *,
    config: BeadsMirrorConfig,
    runner: BeadsRunner = _run_command,
    result_path: Path | None = None,
    labels: Sequence[str] = ("mimo", "mtp", "rollout", "payload-mirror"),
    priority: str = "P1",
    issue_type: str = "task",
    parent_id: str | None = None,
) -> list[BeadMirrorResult]:
    """Synchronize canonical rollout payload entries into optional Beads items.

    Beads remains opt-in through ``config.enabled``. When disabled or unavailable,
    the function performs no Beads writes and returns/records deterministic skip
    results for every canonical payload entry.
    """

    entries = entries_from_rollout_payload(
        payload,
        labels=labels,
        priority=priority,
        issue_type=issue_type,
        parent_id=parent_id,
    )
    results = mirror_entries_to_configured_beads(
        entries,
        config=config,
        runner=runner,
    )
    _write_result_artifact(result_path, results)
    return results


def mirror_entries_to_beads(
    entries: Iterable[BeadMirrorEntry], runner: BeadsRunner = _run_command
) -> list[BeadMirrorResult]:
    results: list[BeadMirrorResult] = []
    for entry in entries:
        seed_mirror_key = _seed_mirror_key(entry)
        existing_beads = _find_existing_beads(seed_mirror_key, runner)
        if len(existing_beads) > 1:
            bead_ids = ", ".join(_require_bead_id(bead) for bead in existing_beads)
            raise BeadMirrorError(
                f"seed mirror key {seed_mirror_key} matched multiple Beads entries: {bead_ids}"
            )
        if not existing_beads:
            _run_checked(runner, _create_command(entry, seed_mirror_key))
            results.append(
                BeadMirrorResult(
                    seed_mirror_key=seed_mirror_key,
                    action="created",
                    bead_id=None,
                )
            )
            continue

        bead_id = _require_bead_id(existing_beads[0])
        _run_checked(runner, _update_command(entry, seed_mirror_key, bead_id))
        results.append(
            BeadMirrorResult(
                seed_mirror_key=seed_mirror_key,
                action="updated",
                bead_id=bead_id,
            )
        )
    return results


def _load_seed_json(seed_path: Path) -> JsonObject:
    try:
        import yaml
    except ImportError as exc:
        raise BeadMirrorError(
            "PyYAML is required to read rollout seed YAML files"
        ) from exc

    loaded = cast(object, yaml.safe_load(seed_path.read_text()))
    if not isinstance(loaded, dict):
        raise BeadMirrorError(f"seed file {seed_path} did not contain a YAML object")
    return cast(JsonObject, loaded)


def _short_title(raw_acceptance_criterion: str, fallback_index: int) -> str:
    first_line = next(
        (
            line.strip()
            for line in raw_acceptance_criterion.splitlines()
            if line.strip()
        ),
        f"AC-P{fallback_index} rollout task",
    )
    return first_line.rstrip(":")


def entries_from_seed(
    seed_path: Path,
    *,
    labels: Sequence[str] = ("mimo", "mtp", "rollout", "seed-mirror"),
    priority: str = "P1",
    issue_type: str = "task",
    parent_id: str | None = None,
) -> list[BeadMirrorEntry]:
    seed = _load_seed_json(seed_path)
    raw_metadata = seed.get("metadata")
    if not isinstance(raw_metadata, dict):
        raise BeadMirrorError("seed metadata is missing or not an object")
    metadata = cast(JsonObject, raw_metadata)
    seed_id = metadata.get("seed_id")
    if not isinstance(seed_id, str) or seed_id == "":
        raise BeadMirrorError("seed metadata.seed_id is missing or not a string")
    raw_acceptance_criteria = seed.get("acceptance_criteria")
    if not isinstance(raw_acceptance_criteria, list):
        raise BeadMirrorError("seed acceptance_criteria is missing or not a list")
    acceptance_criteria = cast(list[object], raw_acceptance_criteria)

    entries: list[BeadMirrorEntry] = []
    for index, raw_acceptance_criterion in enumerate(acceptance_criteria):
        if not isinstance(raw_acceptance_criterion, str):
            raise BeadMirrorError("seed acceptance_criteria entries must be strings")
        title = _short_title(raw_acceptance_criterion, index)
        entries.append(
            BeadMirrorEntry(
                seed_id=seed_id,
                acceptance_index=index,
                title=title,
                description=raw_acceptance_criterion.strip(),
                acceptance_criteria=raw_acceptance_criterion.strip(),
                labels=tuple(labels),
                priority=priority,
                issue_type=issue_type,
                parent_id=parent_id,
            )
        )
    return entries


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Idempotently mirror rollout seed acceptance criteria into Beads. "
            "Each mirrored Bead is keyed by metadata.seed_mirror_key so repeated "
            "runs update instead of creating duplicates."
        )
    )
    parser.add_argument("seed_path", type=Path)
    parser.add_argument("--parent-id", default=None)
    parser.add_argument("--priority", default="P1")
    parser.add_argument("--type", default="task", dest="issue_type")
    parser.add_argument(
        "--labels",
        default="mimo,mtp,rollout,seed-mirror",
        help="Comma-separated labels to set on mirrored Beads.",
    )
    parser.add_argument(
        "--enable-beads-mirror",
        action="store_true",
        help="Create or update Beads items when the bd CLI is available.",
    )
    return parser


def main(argv: list[str] | None = None, runner: BeadsRunner = _run_command) -> int:
    namespace = _parser().parse_args(argv)
    labels = tuple(
        label.strip()
        for label in cast(str, namespace.labels).split(",")
        if label.strip()
    )
    entries = entries_from_seed(
        cast(Path, namespace.seed_path),
        labels=labels,
        priority=cast(str, namespace.priority),
        issue_type=cast(str, namespace.issue_type),
        parent_id=cast(str | None, namespace.parent_id),
    )
    results = mirror_entries_to_configured_beads(
        entries,
        config=BeadsMirrorConfig(enabled=cast(bool, namespace.enable_beads_mirror)),
        runner=runner,
    )
    for result in results:
        print(json.dumps(asdict(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except BeadMirrorError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
