"""Pure planning helpers for mirroring rollout seeds into operator ledgers.

This module deliberately returns planned operations only. Callers can inspect,
review, or execute those operations elsewhere; no files, Beads issues, shell
commands, or external services are touched here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeAlias, TypeVar


@dataclass(frozen=True, slots=True)
class RolloutSeed:
    """Minimal rollout seed data needed to plan mirroring operations."""

    goal: str
    acceptance_criteria: tuple[str, ...]
    ontology_name: str
    required_concepts: tuple[str, ...]
    seed_id: str


@dataclass(frozen=True, slots=True)
class FileAppendOperation:
    """A pure description of content to append under a markdown heading."""

    path: str
    heading: str
    content: str


@dataclass(frozen=True, slots=True)
class BeadsNoteOperation:
    """A pure description of a Beads issue note update."""

    issue_id: str
    note: str


MirrorOperation: TypeAlias = FileAppendOperation | BeadsNoteOperation

PlanT = TypeVar("PlanT")
FileResultT = TypeVar("FileResultT")
BeadsResultT = TypeVar("BeadsResultT")


@dataclass(frozen=True, slots=True)
class AcMirroringBoundaryDependencies(Generic[PlanT, FileResultT, BeadsResultT]):
    """Injected operations allowed at the AC mirroring boundary.

    Repo validation and verification discovery callables are present so tests can
    prove this orchestrator does not cross into unrelated validation logic.
    """

    plan_mirror: Callable[[], PlanT]
    persist_files: Callable[[PlanT], FileResultT]
    persist_beads: Callable[[PlanT], BeadsResultT]
    validate_repo_root: Callable[[], object]
    discover_verification_commands: Callable[[], object]


@dataclass(frozen=True, slots=True)
class AcMirroringBoundaryResult(Generic[PlanT, FileResultT, BeadsResultT]):
    """Result from executing the narrow AC mirroring boundary."""

    plan: PlanT
    file_result: FileResultT
    beads_result: BeadsResultT


@dataclass(frozen=True, slots=True)
class RolloutSeedMirrorPlan:
    """Pure rollout seed mirror plan."""

    summary: str
    operations: tuple[MirrorOperation, ...]


@dataclass(frozen=True, slots=True)
class _AcceptanceCriterionSummary:
    identifier: str
    title: str
    concepts: tuple[str, ...]


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0].strip()


def _criterion_summary(
    criterion: str, required_concepts: tuple[str, ...]
) -> _AcceptanceCriterionSummary:
    first_line = _first_line(criterion).rstrip(":")
    if " " in first_line:
        identifier, title = first_line.split(" ", 1)
    else:
        identifier = first_line
        title = ""
    concepts = _matching_concepts(
        identifier=identifier,
        criterion=criterion,
        required_concepts=required_concepts,
    )
    return _AcceptanceCriterionSummary(
        identifier=identifier,
        title=title,
        concepts=concepts,
    )


def _matching_concepts(
    *,
    identifier: str,
    criterion: str,
    required_concepts: tuple[str, ...],
) -> tuple[str, ...]:
    if identifier == "AC-P0":
        return required_concepts

    lowered_criterion = criterion.lower()
    aliases: dict[str, tuple[str, ...]] = {
        "cluster_baseline": ("baseline", "cluster", "ar row", "ar-vs-mtp"),
        "mtp_vertical_slice": (
            "vertical slice",
            "request contract",
            "worker/generator",
        ),
        "benchmark_telemetry": ("benchmark", "telemetry", "row", "timing"),
        "speedup_budget": ("budget", "speedup", "30", "40", "tok/s", "baseline"),
        "slice_5_gate": ("slice 5", "default", "production"),
    }
    return tuple(
        concept
        for concept in required_concepts
        if concept == "slice_5_gate"
        or any(alias in lowered_criterion for alias in aliases.get(concept, (concept,)))
    )


def _short_goal(goal: str) -> str:
    return goal.strip().removesuffix(".")


def _concepts_text(concepts: tuple[str, ...]) -> str:
    return ", ".join(concepts)


def _todo_content(
    seed: RolloutSeed, summaries: tuple[_AcceptanceCriterionSummary, ...]
) -> str:
    lines = [f"- Seed `{seed.seed_id}`: {_short_goal(seed.goal)}."]
    for summary in summaries:
        concepts = _concepts_text(summary.concepts)
        lines.append(f"- [ ] {summary.identifier} {summary.title} — {concepts}")
    lines.append(
        "- Slice 5 gate: default generation remains AR until same-cluster "
        "AR-vs-MTP rows prove a real MTP speed win."
    )
    return "\n".join(lines)


def _runlog_content(
    seed: RolloutSeed, summaries: tuple[_AcceptanceCriterionSummary, ...]
) -> str:
    mirrored_criteria = "\n".join(
        f"{index}. {summary.identifier} {summary.title}"
        for index, summary in enumerate(summaries, start=1)
    )
    return (
        f"Seed: `{seed.seed_id}`\n\n"
        f"Goal: {_short_goal(seed.goal)}.\n\n"
        "Mirrored acceptance criteria:\n"
        f"{mirrored_criteria}\n\n"
        f"Required ontology concepts: {_concepts_text(seed.required_concepts)}\n\n"
        "Evidence policy: benchmark rows are required before any >=30 tok/s, "
        ">=40 tok/s, or MTP speedup claim; absent live cluster rows must be "
        "recorded as blocked rather than fabricated."
    )


def _beads_operations(
    *,
    epic_id: str,
    timestamp: str,
    summaries: tuple[_AcceptanceCriterionSummary, ...],
) -> tuple[BeadsNoteOperation, ...]:
    return tuple(
        BeadsNoteOperation(
            issue_id=f"{epic_id}.{index}",
            note=(
                f"[seed mirror {timestamp}] {summary.identifier} {summary.title}; "
                f"concepts={_concepts_text(summary.concepts)}; "
                "Slice 5 remains blocked without same-cluster AR-vs-MTP evidence."
            ),
        )
        for index, summary in enumerate(summaries, start=1)
    )


def orchestrate_ac_mirroring_boundary(
    dependencies: AcMirroringBoundaryDependencies[PlanT, FileResultT, BeadsResultT],
) -> AcMirroringBoundaryResult[PlanT, FileResultT, BeadsResultT]:
    """Invoke only mirroring dependencies for the governed AC boundary."""

    plan = dependencies.plan_mirror()
    file_result = dependencies.persist_files(plan)
    beads_result = dependencies.persist_beads(plan)
    return AcMirroringBoundaryResult(
        plan=plan,
        file_result=file_result,
        beads_result=beads_result,
    )


def plan_rollout_seed_mirroring(
    seed: RolloutSeed,
    *,
    todo_path: str,
    runlog_path: str,
    beads_epic_id: str | None,
    timestamp: str,
) -> RolloutSeedMirrorPlan:
    """Derive TODO, runlog, and optional Beads mirror operations from a seed.

    The function is referentially transparent: the same arguments produce the
    same immutable plan, and no I/O or external update is performed.
    """

    if not seed.acceptance_criteria:
        raise ValueError("rollout seed must contain at least one acceptance criterion")

    summaries = tuple(
        _criterion_summary(criterion, seed.required_concepts)
        for criterion in seed.acceptance_criteria
    )
    operations: tuple[MirrorOperation, ...] = (
        FileAppendOperation(
            path=todo_path,
            heading=f"MiMo MTP optimized rollout mirror — {timestamp}",
            content=_todo_content(seed, summaries),
        ),
        FileAppendOperation(
            path=runlog_path,
            heading=f"{timestamp} Optimized rollout seed mirror",
            content=_runlog_content(seed, summaries),
        ),
    )
    if beads_epic_id is not None:
        operations = operations + _beads_operations(
            epic_id=beads_epic_id,
            timestamp=timestamp,
            summaries=summaries,
        )
    return RolloutSeedMirrorPlan(
        summary=f"Mirror {len(seed.acceptance_criteria)} rollout ACs from {seed.seed_id}",
        operations=operations,
    )
