"""Canonical rollout-tracking payloads for the optimized MiMo MTP AC tree.

The normalizer in this module is intentionally pure: it converts raw acceptance
criterion text plus operator status/evidence overlays into a deterministic sync
payload that bookkeeping sinks, ledgers, or external issue trackers can consume
without re-parsing Seed prose.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Literal, TypeAlias

from scripts.rollout_bookkeeping_sinks import (
    AcBookkeepingMetadata,
    BeadsBookkeepingDispatcher,
    BeadsBookkeepingOperation,
    BookkeepingDispatchFailure,
    BookkeepingOperation,
    BookkeepingSinkName,
    BookkeepingSinkStatus,
    FileBookkeepingDispatcher,
    FileBookkeepingOperation,
    route_ac_bookkeeping,
)
from scripts.verification_command_registry import (
    ChangeCategory,
    get_verification_commands_for_change_category,
)

RolloutAcStatus: TypeAlias = Literal["pending", "in_progress", "complete", "blocked"]

_SCHEMA_VERSION = "mimo-mtp-rollout-sync/v1"
_AC_ID_PATTERN = re.compile(
    r"^\s*(AC-P(?P<order>\d+))\s+(?P<title>[^\n:]+):?", re.ASCII
)

_STATUS_ALIASES: dict[str, RolloutAcStatus] = {
    "blocked": "blocked",
    "blocked_with_command": "blocked",
    "blocked_with_evidence": "blocked",
    "blocked_with_commands": "blocked",
    "complete": "complete",
    "completed": "complete",
    "done": "complete",
    "pass": "complete",
    "passed": "complete",
    "pending": "pending",
    "todo": "pending",
    "not_started": "pending",
    "open": "pending",
    "in_progress": "in_progress",
    "running": "in_progress",
    "started": "in_progress",
}

_CONCEPT_ALIASES: dict[str, tuple[str, ...]] = {
    "cluster_baseline": (
        "ar baseline",
        "ar-vs-mtp",
        "baseline",
        "bench/chat/completions",
        "same-cluster",
    ),
    "mtp_vertical_slice": (
        "explicit guarded mtp",
        "fail-closed",
        "mimo_mtp",
        "request contract",
        "sidecar-backed mtp",
        "task params",
        "vertical slice",
        "worker/generator",
    ),
    "benchmark_telemetry": (
        "benchmark",
        "bottleneck",
        "generation_tps",
        "jsonl rows",
        "row parsing",
        "telemetry",
        "timing",
    ),
    "speedup_budget": (
        "budget",
        "gap",
        "mtp beats ar",
        "speedup",
        "target",
        "tok/s",
    ),
    "slice_5_gate": (
        "default generation remains ar",
        "default requests",
        "production",
        "slice 5",
    ),
}

_CONCEPTS_BY_AC_ID: dict[str, tuple[str, ...]] = {
    "AC-P0": (
        "cluster_baseline",
        "mtp_vertical_slice",
        "benchmark_telemetry",
        "speedup_budget",
        "slice_5_gate",
    ),
    "AC-P1": (
        "cluster_baseline",
        "benchmark_telemetry",
        "speedup_budget",
        "slice_5_gate",
    ),
    "AC-P2": (
        "cluster_baseline",
        "benchmark_telemetry",
        "speedup_budget",
        "slice_5_gate",
    ),
    "AC-P3": ("mtp_vertical_slice", "slice_5_gate"),
    "AC-P4": ("mtp_vertical_slice", "slice_5_gate"),
    "AC-P5": ("mtp_vertical_slice", "benchmark_telemetry", "slice_5_gate"),
    "AC-P6": ("benchmark_telemetry", "slice_5_gate"),
    "AC-P7": (
        "cluster_baseline",
        "benchmark_telemetry",
        "speedup_budget",
        "slice_5_gate",
    ),
    "AC-P8": ("benchmark_telemetry", "speedup_budget", "slice_5_gate"),
    "AC-P9": ("cluster_baseline", "speedup_budget", "slice_5_gate"),
    "AC-P10": ("slice_5_gate",),
}

_GATE_TAGS_BY_AC_ID: dict[str, tuple[str, ...]] = {
    "AC-P0": ("slice_5_blocked",),
    "AC-P1": ("cluster_baseline", "same_cluster_evidence", "slice_5_blocked"),
    "AC-P2": ("benchmark_telemetry", "speedup_budget", "slice_5_blocked"),
    "AC-P3": ("slice_5_blocked",),
    "AC-P4": ("slice_5_blocked",),
    "AC-P5": ("slice_5_blocked",),
    "AC-P6": ("benchmark_telemetry", "slice_5_blocked"),
    "AC-P7": (
        "benchmark_telemetry",
        "cluster_baseline",
        "same_cluster_evidence",
        "slice_5_blocked",
        "speedup_budget",
    ),
    "AC-P8": ("benchmark_telemetry", "speedup_budget", "slice_5_blocked"),
    "AC-P9": (
        "default_ar",
        "production_gate",
        "same_cluster_evidence",
        "slice_5_blocked",
        "speedup_budget",
    ),
    "AC-P10": ("slice_5_blocked",),
}

_FALLBACK_GATE_TAG_ORDER: tuple[str, ...] = (
    "benchmark_telemetry",
    "cluster_baseline",
    "default_ar",
    "production_gate",
    "same_cluster_evidence",
    "speedup_budget",
    "slice_5_blocked",
)


@dataclass(frozen=True, slots=True)
class AcceptanceCriterionGateMetadata:
    """Global rollout gate metadata shared by all canonical AC items."""

    default_generation_mode: Literal["ar"]
    slice_5_status: Literal["blocked", "eligible"]
    requires_same_cluster_rows: bool
    requires_real_mtp_speed_win: bool
    target_tok_s: float
    preferred_tok_s: float
    required_evidence: tuple[str, ...]
    guardrail: str


@dataclass(frozen=True, slots=True)
class VerificationCommandIndexEntry:
    """Deterministic evidence-contract index entry for one preserved command."""

    index: int
    category: ChangeCategory
    command: str
    purpose: str
    requires_live_cluster: bool


@dataclass(frozen=True, slots=True)
class AcceptanceCriterionSyncItem:
    """One canonical acceptance criterion row for rollout sync."""

    stable_id: str
    acceptance_id: str
    order: int
    title: str
    status: RolloutAcStatus
    concepts: tuple[str, ...]
    evidence_paths: tuple[str, ...]
    gate_tags: tuple[str, ...]
    evidence_contract_notes: tuple[str, ...] = ()
    verification_command_index: tuple[VerificationCommandIndexEntry, ...] = ()


@dataclass(frozen=True, slots=True)
class RolloutSyncPayload:
    """Deterministic canonical payload for external AC synchronization."""

    seed_id: str
    ontology_name: str
    schema_version: str
    gate: AcceptanceCriterionGateMetadata
    items: tuple[AcceptanceCriterionSyncItem, ...]


@dataclass(frozen=True, slots=True)
class RolloutArtifactSyncResult:
    """Per-artifact outcome from rollout tracking synchronization."""

    acceptance_id: str
    artifact: BookkeepingSinkName
    target: str
    succeeded: bool
    error: str | None


@dataclass(frozen=True, slots=True)
class RolloutTrackingSyncResult:
    """Aggregate outcome for a rollout-tracking synchronization pass."""

    payload: RolloutSyncPayload
    artifact_results: tuple[RolloutArtifactSyncResult, ...]
    failures: tuple[BookkeepingDispatchFailure, ...]


@dataclass(frozen=True, slots=True)
class _ParsedAcceptanceCriterion:
    acceptance_id: str
    order: int
    title: str
    text: str


def _default_gate() -> AcceptanceCriterionGateMetadata:
    return AcceptanceCriterionGateMetadata(
        default_generation_mode="ar",
        slice_5_status="blocked",
        requires_same_cluster_rows=True,
        requires_real_mtp_speed_win=True,
        target_tok_s=30.0,
        preferred_tok_s=40.0,
        required_evidence=(
            "same_cluster_ar_vs_mtp_rows",
            "no_correctness_or_fallback_concern",
        ),
        guardrail=(
            "Slice 5 remains blocked until same-cluster AR-vs-MTP rows prove "
            "a real MTP speed win; default exo generation remains AR."
        ),
    )


def _parse_criterion(criterion: str) -> _ParsedAcceptanceCriterion:
    match = _AC_ID_PATTERN.search(criterion)
    if match is None:
        raise ValueError("acceptance criterion is missing canonical AC-P id")

    acceptance_id = match.group(1)
    title = match.group("title").strip().removesuffix(":")
    return _ParsedAcceptanceCriterion(
        acceptance_id=acceptance_id,
        order=int(match.group("order")),
        title=title,
        text=criterion.strip(),
    )


def _normalize_status(*, acceptance_id: str, raw_status: str | None) -> RolloutAcStatus:
    if raw_status is None:
        return "pending"

    normalized_key = raw_status.strip().lower().replace("-", "_").replace(" ", "_")
    status = _STATUS_ALIASES.get(normalized_key)
    if status is None:
        raise ValueError(f"unknown rollout AC status for {acceptance_id}: {raw_status}")
    return status


def _matches_concept(concept: str, lowered_text: str) -> bool:
    aliases = _CONCEPT_ALIASES.get(concept, (concept,))
    return any(alias in lowered_text for alias in aliases)


def _concepts_for_criterion(
    criterion: _ParsedAcceptanceCriterion,
    required_concepts: tuple[str, ...],
) -> tuple[str, ...]:
    canonical_concepts = _CONCEPTS_BY_AC_ID.get(criterion.acceptance_id)
    if canonical_concepts is not None:
        return tuple(
            concept for concept in canonical_concepts if concept in required_concepts
        )

    lowered_text = criterion.text.lower()
    concepts = [
        concept
        for concept in required_concepts
        if _matches_concept(concept, lowered_text)
    ]
    if "slice_5_gate" in required_concepts and "slice_5_gate" not in concepts:
        concepts.append("slice_5_gate")
    return tuple(concepts)


def _evidence_contract_notes_for_criterion(
    criterion: _ParsedAcceptanceCriterion,
) -> tuple[str, ...]:
    lowered_text = criterion.text.lower()
    tensor_parallel_note = (
        "exo cluster tensor parallelization is the live model path, "
        "not single-Studio full-model loading"
    )
    if tensor_parallel_note.lower() in lowered_text:
        return (tensor_parallel_note,)
    return ()


def _verification_command_index_for_criterion(
    criterion: _ParsedAcceptanceCriterion,
) -> tuple[VerificationCommandIndexEntry, ...]:
    lowered_text = criterion.text.lower()
    if not any(
        marker in lowered_text
        for marker in ("verification commands", "focused pytest", "git diff --check")
    ):
        return ()

    indexed_entries: list[VerificationCommandIndexEntry] = []
    seen_commands: set[str] = set()
    for category in ("api", "task_type", "worker", "benchmark_telemetry"):
        descriptor = get_verification_commands_for_change_category(category)[0]
        if descriptor.command in seen_commands:
            continue
        seen_commands.add(descriptor.command)
        indexed_entries.append(
            VerificationCommandIndexEntry(
                index=len(indexed_entries),
                category=descriptor.category,
                command=descriptor.command,
                purpose=descriptor.purpose,
                requires_live_cluster=descriptor.requires_live_cluster,
            )
        )
    return tuple(indexed_entries)


def _gate_tags_for_criterion(criterion: _ParsedAcceptanceCriterion) -> tuple[str, ...]:
    canonical_tags = _GATE_TAGS_BY_AC_ID.get(criterion.acceptance_id)
    if canonical_tags is not None:
        return canonical_tags

    lowered_text = criterion.text.lower()
    tags: set[str] = {"slice_5_blocked"}

    if any(marker in lowered_text for marker in ("cluster", "baseline", "ar-vs-mtp")):
        tags.add("cluster_baseline")
    if any(
        marker in lowered_text for marker in ("same-cluster", "ar-vs-mtp", "cluster")
    ):
        tags.add("same_cluster_evidence")
    if any(marker in lowered_text for marker in ("benchmark", "telemetry", "timing")):
        tags.add("benchmark_telemetry")
    if any(
        marker in lowered_text for marker in ("budget", "speedup", "30", "40", "tok/s")
    ):
        tags.add("speedup_budget")
    if any(
        marker in lowered_text
        for marker in ("default generation remains ar", "default requests")
    ):
        tags.add("default_ar")
    if any(marker in lowered_text for marker in ("production", "slice 5 gate")):
        tags.add("production_gate")

    return tuple(tag for tag in _FALLBACK_GATE_TAG_ORDER if tag in tags)


def _canonical_items(
    *,
    acceptance_criteria: tuple[str, ...],
    seed_id: str,
    required_concepts: tuple[str, ...],
    status_by_id: dict[str, str],
    evidence_by_id: dict[str, tuple[str, ...]],
) -> tuple[AcceptanceCriterionSyncItem, ...]:
    seen_ids: set[str] = set()
    parsed_criteria: list[_ParsedAcceptanceCriterion] = []

    for criterion in acceptance_criteria:
        parsed = _parse_criterion(criterion)
        if parsed.acceptance_id in seen_ids:
            raise ValueError(
                f"duplicate acceptance criterion id: {parsed.acceptance_id}"
            )
        seen_ids.add(parsed.acceptance_id)
        parsed_criteria.append(parsed)

    return tuple(
        AcceptanceCriterionSyncItem(
            stable_id=f"{seed_id}:{criterion.acceptance_id}",
            acceptance_id=criterion.acceptance_id,
            order=criterion.order,
            title=criterion.title,
            status=_normalize_status(
                acceptance_id=criterion.acceptance_id,
                raw_status=status_by_id.get(criterion.acceptance_id),
            ),
            concepts=_concepts_for_criterion(criterion, required_concepts),
            evidence_paths=evidence_by_id.get(criterion.acceptance_id, ()),
            gate_tags=_gate_tags_for_criterion(criterion),
            evidence_contract_notes=_evidence_contract_notes_for_criterion(criterion),
            verification_command_index=_verification_command_index_for_criterion(
                criterion
            ),
        )
        for criterion in sorted(parsed_criteria, key=lambda item: item.order)
    )


def normalize_rollout_acceptance_criteria(
    *,
    acceptance_criteria: tuple[str, ...],
    seed_id: str,
    ontology_name: str,
    required_concepts: tuple[str, ...],
    status_by_id: dict[str, str] | None = None,
    evidence_by_id: dict[str, tuple[str, ...]] | None = None,
) -> RolloutSyncPayload:
    """Normalize optimized MiMo MTP AC prose into a canonical sync payload.

    Ordering is derived from the numeric suffix of canonical IDs such as
    ``AC-P10`` so payloads remain stable even when input criteria arrive in a
    different order. Slice 5 metadata is deliberately fail-closed by default.
    """

    return RolloutSyncPayload(
        seed_id=seed_id,
        ontology_name=ontology_name,
        schema_version=_SCHEMA_VERSION,
        gate=_default_gate(),
        items=_canonical_items(
            acceptance_criteria=acceptance_criteria,
            seed_id=seed_id,
            required_concepts=required_concepts,
            status_by_id={} if status_by_id is None else status_by_id,
            evidence_by_id={} if evidence_by_id is None else evidence_by_id,
        ),
    )


def _sync_metadata_for_item(item: AcceptanceCriterionSyncItem) -> AcBookkeepingMetadata:
    first_evidence_path = item.evidence_paths[0] if item.evidence_paths else ""
    return AcBookkeepingMetadata(
        seed_id=item.stable_id.split(":", maxsplit=1)[0],
        acceptance_index=item.order,
        acceptance_id=item.acceptance_id,
        title=item.title,
        summary=(
            f"status={item.status}; "
            f"concepts={','.join(item.concepts)}; "
            f"gate_tags={','.join(item.gate_tags)}"
        ),
        evidence_path=first_evidence_path,
    )


def _artifact_for_operation(operation: BookkeepingOperation) -> BookkeepingSinkName:
    if isinstance(operation, FileBookkeepingOperation):
        return operation.sink
    return "beads"


def _target_for_operation(operation: BookkeepingOperation) -> str:
    if isinstance(operation, FileBookkeepingOperation):
        return operation.path
    return operation.issue_id


def _dispatch_rollout_artifact(
    operation: BookkeepingOperation,
    *,
    dispatch_todo: FileBookkeepingDispatcher,
    dispatch_runlog: FileBookkeepingDispatcher,
    dispatch_beads: BeadsBookkeepingDispatcher,
) -> None:
    if isinstance(operation, BeadsBookkeepingOperation):
        dispatch_beads(operation)
    elif operation.sink == "todo":
        dispatch_todo(operation)
    else:
        dispatch_runlog(operation)


def orchestrate_rollout_tracking_sync(
    *,
    acceptance_criteria: tuple[str, ...],
    seed_id: str,
    ontology_name: str,
    required_concepts: tuple[str, ...],
    sink_statuses: tuple[BookkeepingSinkStatus, ...],
    timestamp: str,
    dispatch_todo: FileBookkeepingDispatcher,
    dispatch_runlog: FileBookkeepingDispatcher,
    dispatch_beads: BeadsBookkeepingDispatcher,
    status_by_id: dict[str, str] | None = None,
    evidence_by_id: dict[str, tuple[str, ...]] | None = None,
) -> RolloutTrackingSyncResult:
    """Synchronize normalized rollout tracking into configured artifacts.

    The normalized payload is the source of truth for artifact updates. TODO and
    runlog operations are always attempted when their sinks are routable; Beads
    is optional and is invoked only when its sink is explicitly enabled,
    configured, and available. Each artifact update is isolated so a failing
    sink records a partial failure without preventing later successful updates.
    """

    payload = normalize_rollout_acceptance_criteria(
        acceptance_criteria=acceptance_criteria,
        seed_id=seed_id,
        ontology_name=ontology_name,
        required_concepts=required_concepts,
        status_by_id=status_by_id,
        evidence_by_id=evidence_by_id,
    )
    artifact_results: list[RolloutArtifactSyncResult] = []
    failures: list[BookkeepingDispatchFailure] = []

    for item in payload.items:
        metadata = _sync_metadata_for_item(item)
        for operation in route_ac_bookkeeping(
            metadata,
            sink_statuses=sink_statuses,
            timestamp=timestamp,
        ):
            artifact = _artifact_for_operation(operation)
            target = _target_for_operation(operation)
            try:
                _dispatch_rollout_artifact(
                    operation,
                    dispatch_todo=dispatch_todo,
                    dispatch_runlog=dispatch_runlog,
                    dispatch_beads=dispatch_beads,
                )
            except Exception as exc:  # noqa: BLE001 - report partial sync failures
                error = str(exc)
                failures.append(
                    BookkeepingDispatchFailure(
                        sink=artifact,
                        target=target,
                        error=error,
                    )
                )
                artifact_results.append(
                    RolloutArtifactSyncResult(
                        acceptance_id=item.acceptance_id,
                        artifact=artifact,
                        target=target,
                        succeeded=False,
                        error=error,
                    )
                )
                continue

            artifact_results.append(
                RolloutArtifactSyncResult(
                    acceptance_id=item.acceptance_id,
                    artifact=artifact,
                    target=target,
                    succeeded=True,
                    error=None,
                )
            )

    return RolloutTrackingSyncResult(
        payload=payload,
        artifact_results=tuple(artifact_results),
        failures=tuple(failures),
    )


def serialize_evidence_index_to_json(
    payload: RolloutSyncPayload,
) -> str:
    """Serialize a rollout sync payload to a canonical, deterministic JSON string.

    This function provides a single canonical serialization path so that
    baseline drift tests and any future consumer produce byte-identical
    output from the same input.  Determinism is ensured by:
    - ``dataclasses.asdict`` for structural conversion (respects field order)
    - ``sort_keys=True`` for stable key ordering across Python versions
    - ``ensure_ascii=True`` to avoid locale-dependent encoding differences
    - ``indent=2`` for human-readable, diff-friendly output
    - a trailing newline so the fixture file ends conventionally
    """
    return json.dumps(asdict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
