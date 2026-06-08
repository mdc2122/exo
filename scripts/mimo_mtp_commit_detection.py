#!/usr/bin/env python3
"""Detect whether required latest MiMo MTP commits are present in git history."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final


@dataclass(frozen=True)
class ExpectedMimoMtpCommit:
    """A MiMo MTP commit expected to be reachable from HEAD."""

    sha: str
    description: str


@dataclass(frozen=True)
class MimoMtpCommitStatus:
    """Presence status for one expected MiMo MTP commit."""

    sha: str
    description: str
    present: bool


@dataclass(frozen=True)
class MimoMtpCommitReport:
    """Aggregated MiMo MTP commit detection report."""

    repository_available: bool
    complete: bool
    commit_statuses: list[MimoMtpCommitStatus]
    missing_shas: list[str]
    reasons: list[str]


ACCEPTED_MIMO_MTP_COMMIT_IDENTIFIER: Final[ExpectedMimoMtpCommit] = (
    ExpectedMimoMtpCommit(
        sha="6911a9f545f07276516e29b16872622e3357a68a",
        description="feat: scaffold guarded MiMo MTP cluster rollout",
    )
)

EXPECTED_LATEST_MIMO_MTP_COMMITS: Final[tuple[ExpectedMimoMtpCommit, ...]] = (
    ExpectedMimoMtpCommit(
        sha="bc93219273d18c79436dba6aaa0fa1b1744b8f29",
        description="feat: harden MiMo MTP fastpath readiness",
    ),
    ExpectedMimoMtpCommit(
        sha="ff9f28efb5b269b7d9590a2f05d68a1df296b449",
        description="feat: add MiMo MTP benchmark survival diagnostics",
    ),
    ExpectedMimoMtpCommit(
        sha="e88444fd575d02be3ef116d6d73d31c830d2d46d",
        description="feat: add MiMo MTP cluster benchmark harness",
    ),
    ACCEPTED_MIMO_MTP_COMMIT_IDENTIFIER,
)


def _git(
    repository_path: Path, args: Sequence[str], *, check: bool
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repository_path,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _repository_is_available(repository_path: Path) -> bool:
    if not repository_path.is_dir():
        return False
    completed = _git(
        repository_path,
        ["rev-parse", "--is-inside-work-tree"],
        check=False,
    )
    return completed.returncode == 0 and completed.stdout.strip() == "true"


def _commit_is_in_head_history(repository_path: Path, commit_sha: str) -> bool:
    completed = _git(
        repository_path,
        ["merge-base", "--is-ancestor", commit_sha, "HEAD"],
        check=False,
    )
    return completed.returncode == 0


def inspect_mimo_mtp_commits(
    repository_path: str | Path,
    *,
    expected_commits: Sequence[
        ExpectedMimoMtpCommit
    ] = EXPECTED_LATEST_MIMO_MTP_COMMITS,
) -> MimoMtpCommitReport:
    """Report whether each expected latest MiMo MTP commit is in HEAD history.

    The check is fail-closed: if git cannot confirm the repository, every expected
    commit is reported missing rather than treating unknown history as present.
    """

    repository = Path(repository_path)
    if not _repository_is_available(repository):
        missing_shas = [commit.sha for commit in expected_commits]
        commit_statuses = [
            MimoMtpCommitStatus(
                sha=commit.sha,
                description=commit.description,
                present=False,
            )
            for commit in expected_commits
        ]
        return MimoMtpCommitReport(
            repository_available=False,
            complete=False,
            commit_statuses=commit_statuses,
            missing_shas=missing_shas,
            reasons=["git repository unavailable"],
        )

    commit_statuses: list[MimoMtpCommitStatus] = []
    missing_shas: list[str] = []
    reasons: list[str] = []

    for commit in expected_commits:
        present = _commit_is_in_head_history(repository, commit.sha)
        commit_statuses.append(
            MimoMtpCommitStatus(
                sha=commit.sha,
                description=commit.description,
                present=present,
            )
        )
        if not present:
            missing_shas.append(commit.sha)
            reasons.append(
                f"missing commit in HEAD ancestry: {commit.sha} {commit.description}"
            )

    return MimoMtpCommitReport(
        repository_available=True,
        complete=not missing_shas,
        commit_statuses=commit_statuses,
        missing_shas=missing_shas,
        reasons=reasons,
    )


def _repository_argument(argv: Sequence[str] | None = None) -> str:
    arguments = sys.argv[1:] if argv is None else list(argv)
    if len(arguments) > 1:
        raise ValueError("usage: mimo_mtp_commit_detection.py [repository]")
    if len(arguments) == 0:
        return "."
    return arguments[0]


def main(argv: Sequence[str] | None = None) -> int:
    repository = _repository_argument(argv)
    report = inspect_mimo_mtp_commits(Path(repository))
    print(json.dumps(asdict(report), indent=2))
    return 0 if report.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__: Sequence[str] = (
    "ACCEPTED_MIMO_MTP_COMMIT_IDENTIFIER",
    "EXPECTED_LATEST_MIMO_MTP_COMMITS",
    "ExpectedMimoMtpCommit",
    "MimoMtpCommitReport",
    "MimoMtpCommitStatus",
    "inspect_mimo_mtp_commits",
)
