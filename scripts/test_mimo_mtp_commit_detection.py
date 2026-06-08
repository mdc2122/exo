import subprocess
from collections.abc import Sequence
from pathlib import Path

from scripts.mimo_mtp_commit_detection import (
    ACCEPTED_MIMO_MTP_COMMIT_IDENTIFIER,
    EXPECTED_LATEST_MIMO_MTP_COMMITS,
    ExpectedMimoMtpCommit,
    inspect_mimo_mtp_commits,
)


def _git(repo_path: Path, args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _write(repo_path: Path, relative_path: str, content: str) -> None:
    path = repo_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _commit(repo_path: Path, relative_path: str, message: str, content: str) -> str:
    _write(repo_path, relative_path, content)
    _git(repo_path, ["add", relative_path])
    _git(repo_path, ["commit", "-m", message])
    return _git(repo_path, ["rev-parse", "HEAD"])


def _init_repo(repo_path: Path) -> None:
    _git(repo_path, ["init"])
    _git(repo_path, ["config", "user.email", "tests@example.invalid"])
    _git(repo_path, ["config", "user.name", "Tests"])


def test_expected_latest_mimo_mtp_commit_set_matches_rollout_evidence() -> None:
    expected_shas = [commit.sha for commit in EXPECTED_LATEST_MIMO_MTP_COMMITS]

    assert expected_shas == [
        "bc93219273d18c79436dba6aaa0fa1b1744b8f29",
        "ff9f28efb5b269b7d9590a2f05d68a1df296b449",
        "e88444fd575d02be3ef116d6d73d31c830d2d46d",
        "6911a9f545f07276516e29b16872622e3357a68a",
    ]


def test_accepted_mimo_mtp_commit_identifier_is_discoverable_and_unchanged() -> None:
    assert ACCEPTED_MIMO_MTP_COMMIT_IDENTIFIER.sha == (
        "6911a9f545f07276516e29b16872622e3357a68a"
    )
    assert ACCEPTED_MIMO_MTP_COMMIT_IDENTIFIER.description == (
        "feat: scaffold guarded MiMo MTP cluster rollout"
    )
    assert ACCEPTED_MIMO_MTP_COMMIT_IDENTIFIER in EXPECTED_LATEST_MIMO_MTP_COMMITS


def test_inspect_mimo_mtp_commits_reports_all_present_when_in_head_history(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    first_sha = _commit(tmp_path, "tracked.txt", "first", "first\n")
    second_sha = _commit(tmp_path, "tracked.txt", "second", "second\n")

    report = inspect_mimo_mtp_commits(
        tmp_path,
        expected_commits=(
            ExpectedMimoMtpCommit(sha=first_sha, description="first"),
            ExpectedMimoMtpCommit(sha=second_sha, description="second"),
        ),
    )

    assert report.repository_available is True
    assert report.complete is True
    assert report.missing_shas == []
    assert [status.present for status in report.commit_statuses] == [True, True]
    assert report.reasons == []


def test_inspect_mimo_mtp_commits_reports_missing_commit_not_in_head_history(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    present_sha = _commit(tmp_path, "tracked.txt", "present", "present\n")
    missing_sha = "0" * 40

    report = inspect_mimo_mtp_commits(
        tmp_path,
        expected_commits=(
            ExpectedMimoMtpCommit(sha=present_sha, description="present"),
            ExpectedMimoMtpCommit(sha=missing_sha, description="missing"),
        ),
    )

    assert report.repository_available is True
    assert report.complete is False
    assert report.missing_shas == [missing_sha]
    assert [status.present for status in report.commit_statuses] == [True, False]
    assert report.reasons == [f"missing commit in HEAD ancestry: {missing_sha} missing"]


def test_inspect_mimo_mtp_commits_fails_closed_outside_git_repository(
    tmp_path: Path,
) -> None:
    report = inspect_mimo_mtp_commits(
        tmp_path,
        expected_commits=(ExpectedMimoMtpCommit(sha="1" * 40, description="required"),),
    )

    assert report.repository_available is False
    assert report.complete is False
    assert report.missing_shas == ["1" * 40]
    assert [status.present for status in report.commit_statuses] == [False]
    assert report.reasons == ["git repository unavailable"]


def test_inspect_mimo_mtp_commits_fails_closed_when_repository_path_is_missing(
    tmp_path: Path,
) -> None:
    missing_repository = tmp_path / "missing"

    report = inspect_mimo_mtp_commits(
        missing_repository,
        expected_commits=(ExpectedMimoMtpCommit(sha="2" * 40, description="required"),),
    )

    assert report.repository_available is False
    assert report.complete is False
    assert report.missing_shas == ["2" * 40]
    assert [status.present for status in report.commit_statuses] == [False]
    assert report.reasons == ["git repository unavailable"]
