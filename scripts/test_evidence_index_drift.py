"""Baseline drift test for the canonical evidence index.

This test serializes the rollout tracking evidence index for a fixed
canonical input and compares the output against a committed baseline
fixture. Any structural or content divergence — a new field, a changed
concept mapping, a reordered gate tag, an altered verification command,
or even a changed default — causes the test to fail with a diff, making
index evolution an intentional, reviewable act rather than a silent
regression.

To update the baseline after a deliberate index change, regenerate the
fixture using the canonical serializer and commit the result:

    uv run python3 -c "
    from scripts.test_evidence_index_drift import _canonical_payload_json, _FIXTURE_PATH
    _FIXTURE_PATH.write_text(_canonical_payload_json())
    "
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

from scripts.mimo_mtp_rollout_tracking import (
    normalize_rollout_acceptance_criteria,
    serialize_evidence_index_to_json,
)

_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "evidence_index_drift_baseline.json"

_DRIFT_TEST_ACCEPTANCE_CRITERIA: tuple[str, ...] = (
    """AC-P2 MTP speedup budget model exists before optimization claims:
- add a script or documented calculator that consumes AR/MTP JSONL rows
- compute AR ms/token, MTP tok/s, target gaps for 30 and 40 tok/s, speedup ratio, and pass/fail/ambiguous classification
- classify bottlenecks as acceptance_rate_low, proposal_too_slow, verifier_too_slow, fallback_too_high, depth_too_aggressive, mtp_beats_ar, mtp_reaches_30, and mtp_reaches_40 where telemetry exists
- do not require live MTP rows to pass tests; handle absent rows honestly""",
    """AC-P0 Optimized rollout seed and repo context are validated:
- confirm repo root, branch, git cleanliness, latest MiMo MTP commits, and existing fastpath modules
- record that exo cluster tensor parallelization is the live model path, not single-Studio full-model loading
- mirror ACs into TODO/runlog/Beads where practical
- identify focused verification commands for API/task/worker/benchmark changes""",
    """AC-P10 Verification and closeout:
- focused pytest, ruff, ruff format check, basedpyright, and git diff --check pass for touched surfaces, or failures are documented as pre-existing/blocked
- docs/runlog/TODO/Beads are updated with AC statuses, commands, evidence, and blockers
- final report identifies completed ACs, remaining ACs, worktrees reviewed/merged, tests run, and next best step""",
    """AC-P1 Cluster AR baseline harness is canonical and budget-ready:
- keep scripts/bench_mimo_mtp_cluster.py as the live benchmark path through /bench/chat/completions
- ensure rows include generation_tps, generation_tokens, prompt_tps, power_usage, model id, mode label, repeat index, and payload extra keys
- add or update tests for AR baseline row parsing and unavailable-cluster failure behavior
- provide exact commands for max_tokens=16 and max_tokens=64 AR baseline rows
- if a cluster is available, collect rows; otherwise record a clear blocked-with-command status""",
    """AC-P9 Slice 5 gate remains strict:
- default generation remains AR
- guarded production integration review is allowed only if same-cluster MTP median tok/s beats AR by a meaningful margin or reaches explicit target with no correctness/fallback concern
- recommended margin: >=10-15% median tok/s improvement and preferably >=30 tok/s; 40+ tok/s is preferred target
- docs/TODO/Beads explicitly state whether Slice 5 is blocked or eligible""",
)

_DRIFT_TEST_REQUIRED_CONCEPTS: tuple[str, ...] = (
    "cluster_baseline",
    "mtp_vertical_slice",
    "benchmark_telemetry",
    "speedup_budget",
    "slice_5_gate",
)

_DRIFT_TEST_SEED_ID = "mimo_mtp_optimized_rollout_20260606T203800"
_DRIFT_TEST_ONTOLOGY_NAME = "MimoMtpOptimizedRollout"
_DRIFT_TEST_STATUS_BY_ID: dict[str, str] = {
    "AC-P10": "done",
    "AC-P1": "blocked_with_command",
}
_DRIFT_TEST_EVIDENCE_BY_ID: dict[str, tuple[str, ...]] = {
    "AC-P1": (".goose-ultrawork/evidence/ac-p1-summary.md",),
    "AC-P9": (".goose-ultrawork/evidence/ac-p9-slice-5-strict-gate.md",),
}


def _canonical_payload_json() -> str:
    """Produce the canonical serialized index for the fixed drift-test input."""
    payload = normalize_rollout_acceptance_criteria(
        acceptance_criteria=_DRIFT_TEST_ACCEPTANCE_CRITERIA,
        seed_id=_DRIFT_TEST_SEED_ID,
        ontology_name=_DRIFT_TEST_ONTOLOGY_NAME,
        required_concepts=_DRIFT_TEST_REQUIRED_CONCEPTS,
        status_by_id=_DRIFT_TEST_STATUS_BY_ID,
        evidence_by_id=_DRIFT_TEST_EVIDENCE_BY_ID,
    )
    return serialize_evidence_index_to_json(payload)


def _reserialize_with_mutation(
    canonical_json: str, mutator: Any,  # noqa: ANN401 — intentional callback for test isolation
) -> str:
    """Deserialize, apply a mutating callback, and re-serialize with canonical settings.

    The mutator receives the parsed dict and may modify it in place.
    Returns the re-serialized JSON string with the same canonical
    settings as ``serialize_evidence_index_to_json``.
    """
    data: dict[str, Any] = json.loads(canonical_json)
    mutator(data)
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def test_evidence_index_matches_stored_baseline_without_drift() -> None:
    """The serialized evidence index for a fixed input must match the committed baseline fixture exactly.

    This test guards against silent drift in the evidence index: any
    change to concept mappings, gate defaults, verification commands,
    AC ordering, status normalization, or serialization format will
    produce a diff against the committed baseline and fail the suite.

    When the drift is intentional (e.g. adding a new concept alias or
    changing a gate default), regenerate the fixture and commit it.
    """
    runtime_json = _canonical_payload_json()
    baseline_json = _FIXTURE_PATH.read_text(encoding="utf-8")

    if runtime_json == baseline_json:
        return  # no drift detected

    diff_lines = list(
        difflib.unified_diff(
            baseline_json.splitlines(keepends=True),
            runtime_json.splitlines(keepends=True),
            fromfile="scripts/fixtures/evidence_index_drift_baseline.json (committed)",
            tofile="runtime serialize_evidence_index_to_json output",
        )
    )
    diff_text = "".join(diff_lines)
    raise AssertionError(
        "Evidence index drift detected — the serialized index for the fixed canonical "
        "input no longer matches the committed baseline fixture.\n\n"
        "If this drift is intentional, regenerate the baseline fixture:\n"
        "  uv run python3 -c \"from scripts.test_evidence_index_drift import "
        "_canonical_payload_json, _FIXTURE_PATH; _FIXTURE_PATH.write_text(_canonical_payload_json())\"\n\n"
        f"Diff:\n{diff_text}"
    )


def test_evidence_index_drift_detects_added_field() -> None:
    """Verify the drift test would catch a new field added to an index item."""

    def _add_field(data: dict[str, Any]) -> None:
        data["items"][0]["hypothetical_new_field"] = "should_cause_drift"

    baseline_json = _canonical_payload_json()
    mutated_json = _reserialize_with_mutation(baseline_json, _add_field)

    assert mutated_json != baseline_json, (
        "Adding a new field to an index item must produce a different serialization"
    )

    diff_lines = list(
        difflib.unified_diff(
            baseline_json.splitlines(keepends=True),
            mutated_json.splitlines(keepends=True),
        )
    )
    diff_text = "".join(diff_lines)
    assert "hypothetical_new_field" in diff_text, (
        "The diff must identify the new field causing the drift"
    )


def test_evidence_index_drift_detects_changed_concept() -> None:
    """Verify the drift test would catch a changed concept mapping."""

    def _change_concept(data: dict[str, Any]) -> None:
        for item in data["items"]:
            if item.get("acceptance_id") == "AC-P0":
                item["concepts"] = ["changed_concept", "different_concept"]
                break

    baseline_json = _canonical_payload_json()
    mutated_json = _reserialize_with_mutation(baseline_json, _change_concept)

    assert mutated_json != baseline_json, (
        "Changing concept mappings must produce a different serialization"
    )


def test_evidence_index_drift_detects_changed_gate_default() -> None:
    """Verify the drift test would catch a changed gate default value."""

    def _change_gate(data: dict[str, Any]) -> None:
        data["gate"]["target_tok_s"] = 50.0

    baseline_json = _canonical_payload_json()
    mutated_json = _reserialize_with_mutation(baseline_json, _change_gate)

    assert mutated_json != baseline_json, (
        "Changing a gate default value must produce a different serialization"
    )


def test_evidence_index_drift_detects_reordered_item() -> None:
    """Verify the drift test would catch a reordering of index items."""

    def _reorder_items(data: dict[str, Any]) -> None:
        items = data["items"]
        if len(items) >= 2:
            items[0], items[1] = items[1], items[0]

    baseline_json = _canonical_payload_json()
    mutated_json = _reserialize_with_mutation(baseline_json, _reorder_items)

    assert mutated_json != baseline_json, (
        "Reordering index items must produce a different serialization"
    )


if __name__ == "__main__":
    import sys

    if "--regenerate" in sys.argv:
        canonical = _canonical_payload_json()
        _FIXTURE_PATH.write_text(canonical, encoding="utf-8")
        print(f"Baseline fixture regenerated: {_FIXTURE_PATH}")
        sys.exit(0)

    # Run the main drift test when executed directly
    test_evidence_index_matches_stored_baseline_without_drift()
    print("No drift detected — evidence index matches committed baseline.")
