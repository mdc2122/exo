#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

PersistenceAction = Literal["created", "updated", "unchanged"]

TODO_SECTION_HEADING = "## MiMo MTP optimized rollout ultrawork — 2026-06-06"
RUNLOG_SECTION_HEADING = "## 2026-06-06 Optimized Rollout Ultrawork Seed"
_DEFAULT_SEED_PATH = Path(".goose-ultrawork/seed.yaml")
_DEFAULT_TODO_PATH = Path("TODO.md")
_DEFAULT_RUNLOG_PATH = Path("docs/plans/MIMO_V25_PRO_MTP_FOUNDATION_RUNLOG.md")


@dataclass(frozen=True, slots=True)
class MirrorPersistenceResult:
    todo_action: PersistenceAction
    runlog_action: PersistenceAction


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Idempotently mirror the optimized MiMo MTP rollout seed into the "
            "operator TODO ledger and foundation runlog."
        )
    )
    parser.add_argument("--seed-path", default=str(_DEFAULT_SEED_PATH))
    parser.add_argument("--todo-path", default=str(_DEFAULT_TODO_PATH))
    parser.add_argument("--runlog-path", default=str(_DEFAULT_RUNLOG_PATH))
    return parser


def _extract_quoted_yaml_scalar(seed_text: str, key: str, fallback: str) -> str:
    match = re.search(
        rf'^\s*{re.escape(key)}:\s*"([^"]*)"\s*$', seed_text, re.MULTILINE
    )
    if match is None:
        return fallback
    return match.group(1)


def _extract_acceptance_criteria(seed_text: str) -> tuple[str, ...]:
    criteria: list[str] = []
    for match in re.finditer(r"^\s*(AC-P\d+[^\n]*)", seed_text, re.MULTILINE):
        criteria.append(match.group(1).strip())
    return tuple(criteria)


def _render_todo_section(seed_text: str) -> str:
    goal = _extract_quoted_yaml_scalar(
        seed_text,
        "goal",
        "Implement the optimized MiMo V2.5 Pro MTP rollout strategy.",
    )
    criteria = _extract_acceptance_criteria(seed_text)
    lines = [
        TODO_SECTION_HEADING,
        "",
        "- Seed mirror: `.goose-ultrawork/seed.yaml` defines the performance-first optimized rollout tree.",
        f"- Goal: {goal}",
        "- One-shot target: guarded exo-cluster MTP vertical slice with benchmark-grade telemetry, AR baseline support, bottleneck analysis, and strict Slice 5 gate.",
        "- Probability optimization: baseline first, full vertical slice over partial plumbing, fail-closed honesty, sidecar lifecycle guard, acceptance-rate telemetry, auto-depth policy, and no Slice 5 without same-cluster speedup evidence.",
        "- Slice 5 remains blocked until same-cluster AR-vs-MTP rows prove a real win; 30+ tok/s target, 40+ tok/s preferred.",
    ]
    if criteria:
        lines.extend(["- Mirrored acceptance criteria:"])
        lines.extend(f"  - {criterion}" for criterion in criteria)
    return "\n".join(lines).rstrip() + "\n"


def _render_runlog_section(seed_text: str) -> str:
    criteria = _extract_acceptance_criteria(seed_text)
    lines = [
        RUNLOG_SECTION_HEADING,
        "",
        "Created or refreshed `.goose-ultrawork/seed.yaml` as the source of truth for a performance-first MiMo MTP rollout AC tree.",
        "The tree optimizes for a one-shot guarded cluster vertical slice rather than a broad production rollout. The critical path is:",
        "",
        "1. collect or require same-cluster AR baseline rows through `/bench/chat/completions`,",
        "2. compute 30+/40+ tok/s budget and bottleneck classification,",
        "3. add explicit guarded MTP request/task contract,",
        "4. route compatible requests through production-shaped MTP scaffolding or fail/fallback with honest telemetry,",
        "5. preserve benchmark-grade MTP telemetry in cluster rows,",
        "6. run AR-vs-MTP matrix when cluster and guarded MTP path are available,",
        "7. keep Slice 5 blocked unless same-cluster AR-vs-MTP rows prove a real speed win.",
        "",
        "This seed explicitly corrects the local single-Studio load framing: live MiMo rows must use the exo tensor-parallel cluster path, while local scripts remain for sidecar, synthetic, and provider diagnostics.",
    ]
    if criteria:
        lines.extend(["", "Mirrored acceptance criteria:"])
        lines.extend(f"- {criterion}" for criterion in criteria)
    return "\n".join(lines).rstrip() + "\n"


def _section_pattern(heading: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?ms)^{re.escape(heading)}\n.*?(?=^##\s|\Z)",
    )


def _upsert_section(
    existing_text: str | None, section_text: str, heading: str
) -> tuple[str, PersistenceAction]:
    if existing_text is None:
        return section_text, "created"

    pattern = _section_pattern(heading)
    match = pattern.search(existing_text)
    if match is None:
        separator = (
            "" if existing_text.endswith("\n\n") or existing_text == "" else "\n"
        )
        new_text = (
            f"{existing_text}{separator}\n{section_text}"
            if existing_text
            else section_text
        )
        return new_text, "updated"

    if match.group(0).rstrip() == section_text.rstrip():
        return existing_text, "unchanged"

    new_text = (
        f"{existing_text[: match.start()]}{section_text}{existing_text[match.end() :]}"
    )
    return new_text, "updated"


def _read_optional(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def mirror_rollout_seed(
    *,
    seed_path: Path,
    todo_path: Path,
    runlog_path: Path,
) -> MirrorPersistenceResult:
    seed_text = seed_path.read_text(encoding="utf-8")
    todo_text, todo_action = _upsert_section(
        _read_optional(todo_path), _render_todo_section(seed_text), TODO_SECTION_HEADING
    )
    runlog_text, runlog_action = _upsert_section(
        _read_optional(runlog_path),
        _render_runlog_section(seed_text),
        RUNLOG_SECTION_HEADING,
    )

    if todo_action != "unchanged":
        _write_text(todo_path, todo_text)
    if runlog_action != "unchanged":
        _write_text(runlog_path, runlog_text)

    return MirrorPersistenceResult(todo_action=todo_action, runlog_action=runlog_action)


def main(argv: list[str] | None = None) -> int:
    namespace = _parser().parse_args(argv)
    result = mirror_rollout_seed(
        seed_path=Path(cast(str, namespace.seed_path)),
        todo_path=Path(cast(str, namespace.todo_path)),
        runlog_path=Path(cast(str, namespace.runlog_path)),
    )
    print(f"todo: {result.todo_action}")
    print(f"runlog: {result.runlog_action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
