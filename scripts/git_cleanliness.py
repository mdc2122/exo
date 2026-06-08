#!/usr/bin/env python3
"""Git working-tree cleanliness validation helpers for governed dispatch checks."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

EXPECTED_MIMO_MTP_WORK_BRANCH: Final[str] = (
    "feature/mimo-v25-pro-mtp-fastpath-single-stream-20260601"
)


@dataclass(frozen=True)
class GitCleanlinessVerdict:
    """Result of checking that git has no unexpected local changes."""

    clean: bool
    unexpected_changes: list[str]
    reasons: list[str]


@dataclass(frozen=True)
class GitBranchValidationVerdict:
    """Result of checking that the active branch matches the governed work branch."""

    valid: bool
    branch: str
    expected_branch: str
    reasons: list[str]


@dataclass(frozen=True)
class GitRepositoryInspection:
    """Current git baseline context and cleanliness for dispatch checks."""

    repository_root: Path
    branch: str
    git_status_short: tuple[str, ...]
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


def _run_git_repository_root(repository_path: Path) -> Path:
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=repository_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return Path(completed.stdout.strip()).resolve()


def _run_git_branch(repository_path: Path) -> str:
    completed = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repository_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _status_path(status_line: str) -> str:
    path = status_line[3:]
    if " -> " in path:
        return path.rsplit(" -> ", maxsplit=1)[1]
    return path


def _status_reasons(status_line: str) -> list[str]:
    index_status = status_line[0]
    working_tree_status = status_line[1]
    path = _status_path(status_line)

    if status_line.startswith("?? "):
        return [f"untracked: {path}"]

    reasons: list[str] = []
    if index_status != " ":
        reasons.append(f"index modified: {path}")
    if working_tree_status != " ":
        reasons.append(f"working tree modified: {path}")
    if reasons:
        return reasons
    return [f"changed: {path}"]


def get_current_branch(repository_path: str | Path) -> str:
    """Return the current git branch name for the given repository path.

    Uses ``git branch --show-current`` which returns an empty string when
    the repository is in a detached HEAD state.
    """
    return _run_git_branch(Path(repository_path))


def validate_current_branch(
    repository_path: str | Path,
    *,
    expected_branch: str = EXPECTED_MIMO_MTP_WORK_BRANCH,
) -> GitBranchValidationVerdict:
    """Confirm the active git branch is the governed MiMo MTP work branch."""

    branch = _run_git_branch(Path(repository_path))
    if branch == expected_branch:
        return GitBranchValidationVerdict(
            valid=True,
            branch=branch,
            expected_branch=expected_branch,
            reasons=[],
        )
    if branch == "":
        reason = "active git branch is detached or unavailable"
    else:
        reason = (
            f"active git branch {branch!r} does not match expected "
            f"MiMo V2.5 Pro MTP work branch {expected_branch!r}"
        )
    return GitBranchValidationVerdict(
        valid=False,
        branch=branch,
        expected_branch=expected_branch,
        reasons=[reason],
    )


def inspect_git_repository(
    repository_path: str | Path,
    *,
    expected_changed_paths: Iterable[str] = (),
    allow_dirty: bool = False,
) -> GitRepositoryInspection:
    """Report canonical repo context, current branch, and cleanliness."""

    repository = Path(repository_path)
    git_status = _run_git_status(repository)
    cleanliness = _validate_git_cleanliness_from_status(
        git_status,
        expected_changed_paths=expected_changed_paths,
        allow_dirty=allow_dirty,
    )
    return GitRepositoryInspection(
        repository_root=_run_git_repository_root(repository),
        branch=_run_git_branch(repository),
        git_status_short=tuple(git_status.splitlines()),
        clean=cleanliness.clean,
        unexpected_changes=cleanliness.unexpected_changes,
        reasons=cleanliness.reasons,
    )


def validate_git_cleanliness(
    repository_path: str | Path,
    *,
    expected_changed_paths: Iterable[str] = (),
    allow_dirty: bool = False,
) -> GitCleanlinessVerdict:
    """Confirm the working tree and index contain no unexpected changes.

    Args:
        repository_path: Path inside the git repository to check.
        expected_changed_paths: Relative paths that are allowed to be changed.
        allow_dirty: When true, local changes are accepted but still recorded in
            reasons for governed-dispatch evidence.

    Returns:
        A verdict with every unexpected changed path and a human-readable reason.
    """

    return _validate_git_cleanliness_from_status(
        _run_git_status(Path(repository_path)),
        expected_changed_paths=expected_changed_paths,
        allow_dirty=allow_dirty,
    )


def _validate_git_cleanliness_from_status(
    git_status: str,
    *,
    expected_changed_paths: Iterable[str] = (),
    allow_dirty: bool = False,
) -> GitCleanlinessVerdict:
    expected_paths = set(expected_changed_paths)
    unexpected_changes: list[str] = []
    reasons: list[str] = []

    for raw_line in git_status.splitlines():
        if raw_line == "":
            continue
        changed_path = _status_path(raw_line)
        if changed_path in expected_paths:
            continue
        if changed_path not in unexpected_changes:
            unexpected_changes.append(changed_path)
        reasons.extend(_status_reasons(raw_line))

    if allow_dirty and unexpected_changes:
        return GitCleanlinessVerdict(
            clean=True,
            unexpected_changes=[],
            reasons=["explicitly allowed dirty tree", *reasons],
        )

    return GitCleanlinessVerdict(
        clean=not unexpected_changes,
        unexpected_changes=unexpected_changes,
        reasons=reasons,
    )


@dataclass(frozen=True)
class WorktreeStatus:
    """Git working-tree status: clean or dirty, with branch and change summary."""

    status: Literal["clean", "dirty"]
    branch: str
    repository_root: Path
    change_count: int
    reasons: tuple[str, ...]


def get_worktree_status(
    repository_path: str | Path = ".",
) -> WorktreeStatus:
    """Return whether the git working tree is clean or dirty.

    Uses ``git status --porcelain=v1`` to detect any changes (staged,
    unstaged, or untracked). Returns a :class:`WorktreeStatus` with the
    literal status ``"clean"`` or ``"dirty"``, the current branch, the
    repository root, and a summary of change reasons.

    A *clean* tree has zero porcelain lines — no staged, unstaged, or
    untracked changes. A *dirty* tree has one or more.
    """
    repository = Path(repository_path)
    git_status = _run_git_status(repository)
    lines = [line for line in git_status.splitlines() if line.strip()]
    branch = _run_git_branch(repository)
    repo_root = _run_git_repository_root(repository)
    reasons: list[str] = []
    for raw_line in lines:
        reasons.extend(_status_reasons(raw_line))
    is_clean = len(lines) == 0
    return WorktreeStatus(
        status="clean" if is_clean else "dirty",
        branch=branch,
        repository_root=repo_root,
        change_count=len(lines),
        reasons=tuple(reasons),
    )


__all__: Sequence[str] = (
    "EXPECTED_MIMO_MTP_WORK_BRANCH",
    "GitBranchValidationVerdict",
    "GitCleanlinessVerdict",
    "GitRepositoryInspection",
    "WorktreeStatus",
    "get_current_branch",
    "get_worktree_status",
    "inspect_git_repository",
    "validate_current_branch",
    "validate_git_cleanliness",
)
