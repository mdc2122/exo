from __future__ import annotations

import json
from dataclasses import dataclass
from typing import cast

from scripts import mirror_rollout_seed_to_beads as mirror


@dataclass(slots=True)
class FakeBeadsCli:
    existing_by_key: dict[str, dict[str, object]]
    commands: list[list[str]]

    def __call__(self, command: list[str]) -> mirror.CommandResult:
        self.commands.append(command)
        if command[:3] == ["bd", "list", "--json"]:
            metadata_field = command[command.index("--metadata-field") + 1]
            key = metadata_field.removeprefix("seed_mirror_key=")
            existing = self.existing_by_key.get(key)
            rows = [] if existing is None else [existing]
            return mirror.CommandResult(
                returncode=0, stdout=json.dumps(rows), stderr=""
            )
        if command[:2] in (["bd", "create"], ["bd", "update"]):
            return mirror.CommandResult(returncode=0, stdout="ok\n", stderr="")
        raise AssertionError(f"unexpected command: {command}")


def _entry(index: int = 3, title: str = "Sub-AC 3") -> mirror.BeadMirrorEntry:
    return mirror.BeadMirrorEntry(
        seed_id="mimo_mtp_optimized_rollout_20260606T203800",
        acceptance_index=index,
        title=title,
        description="Implement idempotent Beads persistence.",
        acceptance_criteria="create-vs-update and no duplicates",
        labels=("mimo", "mtp", "rollout", "seed-mirror"),
        priority="P1",
        issue_type="task",
        parent_id=None,
    )


def test_mirror_creates_bead_when_seed_mirror_key_is_absent() -> None:
    fake = FakeBeadsCli(existing_by_key={}, commands=[])

    results = mirror.mirror_entries_to_beads([_entry()], runner=fake)

    assert results == [
        mirror.BeadMirrorResult(
            seed_mirror_key="mimo_mtp_optimized_rollout_20260606T203800:ac-3",
            action="created",
            bead_id=None,
        )
    ]
    assert fake.commands == [
        [
            "bd",
            "list",
            "--json",
            "--metadata-field",
            "seed_mirror_key=mimo_mtp_optimized_rollout_20260606T203800:ac-3",
            "--limit",
            "0",
        ],
        [
            "bd",
            "create",
            "Sub-AC 3",
            "--type",
            "task",
            "--priority",
            "P1",
            "--description",
            "Implement idempotent Beads persistence.",
            "--acceptance",
            "create-vs-update and no duplicates",
            "--labels",
            "mimo,mtp,rollout,seed-mirror",
            "--metadata",
            '{"acceptance_index":3,"seed_id":"mimo_mtp_optimized_rollout_20260606T203800","seed_mirror_key":"mimo_mtp_optimized_rollout_20260606T203800:ac-3"}',
        ],
    ]


def test_mirror_updates_existing_bead_instead_of_creating_duplicate() -> None:
    fake = FakeBeadsCli(
        existing_by_key={
            "mimo_mtp_optimized_rollout_20260606T203800:ac-3": {
                "id": "mimo-v25-pro-mtp-d1-live-preflight-20260601-5ge.3",
                "title": "Old title",
            }
        },
        commands=[],
    )

    results = mirror.mirror_entries_to_beads(
        [_entry(title="Updated Sub-AC 3")], runner=fake
    )

    assert results == [
        mirror.BeadMirrorResult(
            seed_mirror_key="mimo_mtp_optimized_rollout_20260606T203800:ac-3",
            action="updated",
            bead_id="mimo-v25-pro-mtp-d1-live-preflight-20260601-5ge.3",
        )
    ]
    create_commands = [
        command for command in fake.commands if command[:2] == ["bd", "create"]
    ]
    update_commands = [
        command for command in fake.commands if command[:2] == ["bd", "update"]
    ]
    assert create_commands == []
    assert update_commands == [
        [
            "bd",
            "update",
            "mimo-v25-pro-mtp-d1-live-preflight-20260601-5ge.3",
            "--title",
            "Updated Sub-AC 3",
            "--type",
            "task",
            "--priority",
            "P1",
            "--description",
            "Implement idempotent Beads persistence.",
            "--acceptance",
            "create-vs-update and no duplicates",
            "--set-labels",
            "mimo,mtp,rollout,seed-mirror",
            "--metadata",
            '{"acceptance_index":3,"seed_id":"mimo_mtp_optimized_rollout_20260606T203800","seed_mirror_key":"mimo_mtp_optimized_rollout_20260606T203800:ac-3"}',
        ]
    ]


def test_repeated_mirror_run_does_not_create_duplicate_beads() -> None:
    created_ids: dict[str, dict[str, object]] = {}

    def fake_runner(command: list[str]) -> mirror.CommandResult:
        if command[:3] == ["bd", "list", "--json"]:
            metadata_field = command[command.index("--metadata-field") + 1]
            key = metadata_field.removeprefix("seed_mirror_key=")
            existing = created_ids.get(key)
            return mirror.CommandResult(
                returncode=0,
                stdout=json.dumps([] if existing is None else [existing]),
                stderr="",
            )
        if command[:2] == ["bd", "create"]:
            metadata_json = command[command.index("--metadata") + 1]
            metadata = cast(dict[str, object], json.loads(metadata_json))
            key = metadata["seed_mirror_key"]
            assert isinstance(key, str)
            created_ids[key] = {"id": "bead-1", "metadata": metadata}
            return mirror.CommandResult(returncode=0, stdout="bead-1\n", stderr="")
        if command[:2] == ["bd", "update"]:
            return mirror.CommandResult(returncode=0, stdout="ok\n", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    first_results = mirror.mirror_entries_to_beads([_entry()], runner=fake_runner)
    second_results = mirror.mirror_entries_to_beads([_entry()], runner=fake_runner)

    assert first_results[0].action == "created"
    assert second_results == [
        mirror.BeadMirrorResult(
            seed_mirror_key="mimo_mtp_optimized_rollout_20260606T203800:ac-3",
            action="updated",
            bead_id="bead-1",
        )
    ]
    assert len(created_ids) == 1


def test_duplicate_existing_seed_mirror_key_fails_before_write() -> None:
    fake = FakeBeadsCli(
        existing_by_key={
            "mimo_mtp_optimized_rollout_20260606T203800:ac-3": {
                "id": "bead-1",
                "title": "First duplicate",
            }
        },
        commands=[],
    )

    def duplicate_list(command: list[str]) -> mirror.CommandResult:
        result = fake(command)
        if command[:3] != ["bd", "list", "--json"]:
            return result
        return mirror.CommandResult(
            returncode=0,
            stdout=json.dumps(
                [
                    {"id": "bead-1", "title": "First duplicate"},
                    {"id": "bead-2", "title": "Second duplicate"},
                ]
            ),
            stderr="",
        )

    try:
        mirror.mirror_entries_to_beads([_entry()], runner=duplicate_list)
    except mirror.BeadMirrorError as exc:
        assert str(exc) == (
            "seed mirror key mimo_mtp_optimized_rollout_20260606T203800:ac-3 "
            "matched multiple Beads entries: bead-1, bead-2"
        )
    else:
        raise AssertionError("expected duplicate Beads entries to fail closed")

    assert [
        command
        for command in fake.commands
        if command[:2] in (["bd", "create"], ["bd", "update"])
    ] == []
