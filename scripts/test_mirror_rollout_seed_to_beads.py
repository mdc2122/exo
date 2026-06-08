from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from scripts import mirror_rollout_seed_to_beads as mirror
from scripts.rollout_seed_mirroring import RolloutTodoChecklistEntry, RolloutTodoPayload


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


def _rollout_payload() -> RolloutTodoPayload:
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


def test_payload_sync_creates_beads_from_canonical_rollout_payload() -> None:
    fake = FakeBeadsCli(existing_by_key={}, commands=[])

    def available_runner(command: list[str]) -> mirror.CommandResult:
        if command == ["bd", "--version"]:
            fake.commands.append(command)
            return mirror.CommandResult(returncode=0, stdout="bd 1.0\n", stderr="")
        return fake(command)

    results = mirror.sync_rollout_payload_to_configured_beads(
        _rollout_payload(),
        config=mirror.BeadsMirrorConfig(enabled=True),
        runner=available_runner,
    )

    assert results == [
        mirror.BeadMirrorResult(
            seed_mirror_key="mimo_mtp_optimized_rollout_20260606T203800:ac-0",
            action="created",
            bead_id=None,
        ),
        mirror.BeadMirrorResult(
            seed_mirror_key="mimo_mtp_optimized_rollout_20260606T203800:ac-1",
            action="created",
            bead_id=None,
        ),
    ]
    create_commands = [
        command for command in fake.commands if command[:2] == ["bd", "create"]
    ]
    assert create_commands[0] == [
        "bd",
        "create",
        "AC-P0 Optimized rollout seed and repo context are validated",
        "--type",
        "task",
        "--priority",
        "P1",
        "--description",
        (
            "status=done; concepts=cluster_baseline, slice_5_gate; "
            "evidence=files changed: scripts/rollout_seed_mirroring.py | tests: uv run pytest scripts/test_rollout_seed_mirroring.py; "
            "remaining_risk=No live cluster rows for this bookkeeping-only Sub-AC.; "
            "slice_5_gate=blocked: default generation remains AR until same-cluster AR-vs-MTP rows prove a real win."
        ),
        "--acceptance",
        "AC-P0 Optimized rollout seed and repo context are validated",
        "--labels",
        "mimo,mtp,rollout,payload-mirror",
        "--metadata",
        '{"acceptance_index":0,"seed_id":"mimo_mtp_optimized_rollout_20260606T203800","seed_mirror_key":"mimo_mtp_optimized_rollout_20260606T203800:ac-0"}',
    ]


def test_payload_sync_records_unavailable_skip_result_without_beads_writes(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "beads-sync-results.jsonl"
    commands: list[list[str]] = []

    def unavailable_runner(command: list[str]) -> mirror.CommandResult:
        commands.append(command)
        if command == ["bd", "--version"]:
            return mirror.CommandResult(returncode=127, stdout="", stderr="bd missing")
        raise AssertionError(
            f"no Beads write commands expected when unavailable: {command}"
        )

    results = mirror.sync_rollout_payload_to_configured_beads(
        _rollout_payload(),
        config=mirror.BeadsMirrorConfig(enabled=True),
        runner=unavailable_runner,
        result_path=result_path,
    )

    assert results == [
        mirror.BeadMirrorResult(
            seed_mirror_key="mimo_mtp_optimized_rollout_20260606T203800:ac-0",
            action="skipped_unavailable",
            bead_id=None,
        ),
        mirror.BeadMirrorResult(
            seed_mirror_key="mimo_mtp_optimized_rollout_20260606T203800:ac-1",
            action="skipped_unavailable",
            bead_id=None,
        ),
    ]
    assert commands == [["bd", "--version"]]
    assert [json.loads(line) for line in result_path.read_text().splitlines()] == [
        {
            "action": "skipped_unavailable",
            "bead_id": None,
            "seed_mirror_key": "mimo_mtp_optimized_rollout_20260606T203800:ac-0",
        },
        {
            "action": "skipped_unavailable",
            "bead_id": None,
            "seed_mirror_key": "mimo_mtp_optimized_rollout_20260606T203800:ac-1",
        },
    ]


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


def _write_seed(path: Path) -> None:
    path.write_text(
        """metadata:\n  seed_id: mimo_mtp_optimized_rollout_20260606T203800\nacceptance_criteria:\n  - |\n    AC-P0 Optimized rollout seed and repo context are validated.\n""",
        encoding="utf-8",
    )


def test_cli_skips_beads_by_default_without_calling_bd(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.yaml"
    _write_seed(seed_path)

    def fail_if_called(command: list[str]) -> mirror.CommandResult:
        raise AssertionError(f"bd must not be called by default: {command}")

    assert mirror.main([str(seed_path)], runner=fail_if_called) == 0


def test_cli_creates_bead_when_explicitly_enabled_and_available(
    tmp_path: Path,
) -> None:
    seed_path = tmp_path / "seed.yaml"
    _write_seed(seed_path)
    fake = FakeBeadsCli(existing_by_key={}, commands=[])

    def available_runner(command: list[str]) -> mirror.CommandResult:
        if command == ["bd", "--version"]:
            fake.commands.append(command)
            return mirror.CommandResult(returncode=0, stdout="bd 1.0\n", stderr="")
        return fake(command)

    assert (
        mirror.main([str(seed_path), "--enable-beads-mirror"], runner=available_runner)
        == 0
    )

    assert fake.commands[0] == ["bd", "--version"]
    assert fake.commands[-1][:3] == [
        "bd",
        "create",
        "AC-P0 Optimized rollout seed and repo context are validated.",
    ]


def test_configured_sink_skips_without_calling_bd_when_not_configured() -> None:
    def fail_if_called(command: list[str]) -> mirror.CommandResult:
        raise AssertionError(f"bd must not be called when unconfigured: {command}")

    results = mirror.mirror_entries_to_configured_beads(
        [_entry()],
        config=mirror.BeadsMirrorConfig(enabled=False),
        runner=fail_if_called,
    )

    assert results == [
        mirror.BeadMirrorResult(
            seed_mirror_key="mimo_mtp_optimized_rollout_20260606T203800:ac-3",
            action="skipped_unconfigured",
            bead_id=None,
        )
    ]


def test_configured_sink_skips_writes_when_bd_binary_is_missing() -> None:
    def missing_binary_runner(command: list[str]) -> mirror.CommandResult:
        assert command == ["bd", "--version"]
        raise FileNotFoundError("bd")

    results = mirror.mirror_entries_to_configured_beads(
        [_entry()],
        config=mirror.BeadsMirrorConfig(enabled=True),
        runner=missing_binary_runner,
    )

    assert results == [
        mirror.BeadMirrorResult(
            seed_mirror_key="mimo_mtp_optimized_rollout_20260606T203800:ac-3",
            action="skipped_unavailable",
            bead_id=None,
        )
    ]


def test_configured_sink_skips_writes_when_bd_is_unavailable() -> None:
    commands: list[list[str]] = []

    def unavailable_runner(command: list[str]) -> mirror.CommandResult:
        commands.append(command)
        if command == ["bd", "--version"]:
            return mirror.CommandResult(returncode=127, stdout="", stderr="bd missing")
        raise AssertionError(
            f"no Beads write commands expected when unavailable: {command}"
        )

    results = mirror.mirror_entries_to_configured_beads(
        [_entry()],
        config=mirror.BeadsMirrorConfig(enabled=True),
        runner=unavailable_runner,
    )

    assert results == [
        mirror.BeadMirrorResult(
            seed_mirror_key="mimo_mtp_optimized_rollout_20260606T203800:ac-3",
            action="skipped_unavailable",
            bead_id=None,
        )
    ]
    assert commands == [["bd", "--version"]]


def test_configured_sink_creates_bead_when_configured_and_available() -> None:
    fake = FakeBeadsCli(existing_by_key={}, commands=[])

    def available_runner(command: list[str]) -> mirror.CommandResult:
        if command == ["bd", "--version"]:
            fake.commands.append(command)
            return mirror.CommandResult(returncode=0, stdout="bd 1.0\n", stderr="")
        return fake(command)

    results = mirror.mirror_entries_to_configured_beads(
        [_entry()],
        config=mirror.BeadsMirrorConfig(enabled=True),
        runner=available_runner,
    )

    assert results[0].action == "created"
    assert fake.commands[:2] == [
        ["bd", "--version"],
        [
            "bd",
            "list",
            "--json",
            "--metadata-field",
            "seed_mirror_key=mimo_mtp_optimized_rollout_20260606T203800:ac-3",
            "--limit",
            "0",
        ],
    ]
    assert fake.commands[-1][:3] == ["bd", "create", "Sub-AC 3"]


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
