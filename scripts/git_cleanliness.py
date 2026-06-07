#!/usr/bin/env python3
"""Git working-tree cleanliness validation helpers for governed dispatch checks."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GitCleanlinessVerdict:
    """Result of checking that git has no unexpected local changes."""

    clean: bool
    unexpected_changes: list[str]
    reasons: list[str]


def _run_git_status(repository_path: Path) -> str:
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repository_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout


def _status_path(status_line: str) -> str:
    path = status_line[3:]
    if " -> " in path:
        return path.rsplit(" -> ", maxsplit=1)[1]
    return path


def _status_reason(status_line: str) -> str:
    index_status = status_line[0]
    working_tree_status = status_line[1]
    path = _status_path(status_line)

    if status_line.startswith("?? "):
        return f"untracked: {path}"
    if index_status != " ":
        return f"index modified: {path}"
    if working_tree_status != " ":
        return f"working tree modified: {path}"
    return f"changed: {path}"


def validate_git_cleanliness(
    repository_path: str | Path,
    *,
    expected_changed_paths: Iterable[str] = (),
) -> GitCleanlinessVerdict:
    """Confirm the working tree and index contain no unexpected changes.

    Args:
        repository_path: Path inside the git repository to check.
        expected_changed_paths: Relative paths that are allowed to be changed.

    Returns:
        A verdict with every unexpected changed path and a human-readable reason.
    """

    repository = Path(repository_path)
    expected_paths = set(expected_changed_paths)
    unexpected_changes: list[str] = []
    reasons: list[str] = []

    for raw_line in _run_git_status(repository).splitlines():
        if raw_line == "":
            continue
        changed_path = _status_path(raw_line)
        if changed_path in expected_paths:
            continue
        unexpected_changes.append(changed_path)
        reasons.append(_status_reason(raw_line))

    return GitCleanlinessVerdict(
        clean=not unexpected_changes,
        unexpected_changes=unexpected_changes,
        reasons=reasons,
    )


__all__: Sequence[str] = ("GitCleanlinessVerdict", "validate_git_cleanliness")
