from __future__ import annotations

import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path

import scripts.git_cleanliness as git_cleanliness
from scripts.git_cleanliness import get_current_branch, get_worktree_status, validate_git_cleanliness


def _git(repo_path: Path, args: Sequence[str]) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _write(repo_path: Path, relative_path: str, content: str) -> None:
    path = repo_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _init_repo(repo_path: Path) -> None:
    _git(repo_path, ["init"])
    _git(repo_path, ["config", "user.email", "tests@example.invalid"])
    _git(repo_path, ["config", "user.name", "Tests"])
    _write(repo_path, "tracked.txt", "initial\n")
    _git(repo_path, ["add", "tracked.txt"])
    _git(repo_path, ["commit", "-m", "initial"])


def _assert_contains_reason(reasons: Iterable[str], expected_fragment: str) -> None:
    assert any(expected_fragment in reason for reason in reasons)


def test_clean_repository_has_no_unexpected_changes(tmp_path: Path) -> None:
    _init_repo(tmp_path)

    verdict = validate_git_cleanliness(tmp_path)

    assert verdict.clean is True
    assert verdict.unexpected_changes == []
    assert verdict.reasons == []


def test_git_repository_inspection_reports_branch_and_cleanliness(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    _git(tmp_path, ["checkout", "-b", "feature/git-inspection"])
    _write(tmp_path, "tracked.txt", "modified\n")

    inspection = git_cleanliness.inspect_git_repository(tmp_path)

    assert inspection.branch == "feature/git-inspection"
    assert inspection.clean is False
    assert inspection.unexpected_changes == ["tracked.txt"]
    _assert_contains_reason(inspection.reasons, "working tree modified: tracked.txt")


def test_git_repository_inspection_captures_baseline_context_without_accepting_dirty_tree(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    nested_worker_dir = tmp_path / ".goose-ultrawork" / "waves" / "worker-1"
    nested_worker_dir.mkdir(parents=True)
    _git(tmp_path, ["checkout", "-b", "feature/baseline-context"])
    _write(tmp_path, "tracked.txt", "modified\n")
    _write(tmp_path, "notes/evidence.md", "draft\n")

    inspection = git_cleanliness.inspect_git_repository(nested_worker_dir)

    assert inspection.repository_root == tmp_path.resolve()
    assert inspection.branch == "feature/baseline-context"
    assert inspection.git_status_short == (
        " M tracked.txt",
        "?? notes/evidence.md",
    )
    assert inspection.clean is False
    assert inspection.unexpected_changes == ["tracked.txt", "notes/evidence.md"]


def test_current_branch_validation_accepts_expected_mimo_work_branch(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    _git(tmp_path, ["checkout", "-b", git_cleanliness.EXPECTED_MIMO_MTP_WORK_BRANCH])

    verdict = git_cleanliness.validate_current_branch(tmp_path)

    assert verdict.valid is True
    assert verdict.branch == git_cleanliness.EXPECTED_MIMO_MTP_WORK_BRANCH
    assert verdict.expected_branch == git_cleanliness.EXPECTED_MIMO_MTP_WORK_BRANCH
    assert verdict.reasons == []


def test_current_branch_validation_rejects_unexpected_branch(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    _git(tmp_path, ["checkout", "-b", "feature/unrelated"])

    verdict = git_cleanliness.validate_current_branch(tmp_path)

    assert verdict.valid is False
    assert verdict.branch == "feature/unrelated"
    assert verdict.expected_branch == git_cleanliness.EXPECTED_MIMO_MTP_WORK_BRANCH
    assert (
        "active git branch 'feature/unrelated' does not match expected "
        f"MiMo V2.5 Pro MTP work branch "
        f"'{git_cleanliness.EXPECTED_MIMO_MTP_WORK_BRANCH}'"
    ) in verdict.reasons


def test_current_branch_validation_rejects_detached_head(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        text=True,
    ).strip()
    _git(tmp_path, ["checkout", commit])

    verdict = git_cleanliness.validate_current_branch(tmp_path)

    assert verdict.valid is False
    assert verdict.branch == ""
    assert verdict.expected_branch == git_cleanliness.EXPECTED_MIMO_MTP_WORK_BRANCH
    assert "active git branch is detached or unavailable" in verdict.reasons


def test_dirty_working_tree_is_unexpected(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _write(tmp_path, "tracked.txt", "modified\n")

    verdict = validate_git_cleanliness(tmp_path)

    assert verdict.clean is False
    assert verdict.unexpected_changes == ["tracked.txt"]
    _assert_contains_reason(verdict.reasons, "working tree modified: tracked.txt")


def test_dirty_index_is_unexpected(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _write(tmp_path, "tracked.txt", "staged\n")
    _git(tmp_path, ["add", "tracked.txt"])

    verdict = validate_git_cleanliness(tmp_path)

    assert verdict.clean is False
    assert verdict.unexpected_changes == ["tracked.txt"]
    _assert_contains_reason(verdict.reasons, "index modified: tracked.txt")


def test_same_path_staged_and_unstaged_changes_reports_both_states(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    _write(tmp_path, "tracked.txt", "staged\n")
    _git(tmp_path, ["add", "tracked.txt"])
    _write(tmp_path, "tracked.txt", "unstaged\n")

    verdict = validate_git_cleanliness(tmp_path)

    assert verdict.clean is False
    assert verdict.unexpected_changes == ["tracked.txt"]
    _assert_contains_reason(verdict.reasons, "index modified: tracked.txt")
    _assert_contains_reason(verdict.reasons, "working tree modified: tracked.txt")


def test_untracked_file_is_unexpected(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _write(tmp_path, "new-file.txt", "new\n")

    verdict = validate_git_cleanliness(tmp_path)

    assert verdict.clean is False
    assert verdict.unexpected_changes == ["new-file.txt"]
    _assert_contains_reason(verdict.reasons, "untracked: new-file.txt")


def test_expected_paths_are_not_unexpected(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _write(tmp_path, "tracked.txt", "modified\n")
    _write(tmp_path, "scratch/notes.txt", "note\n")

    verdict = validate_git_cleanliness(
        tmp_path,
        expected_changed_paths=("tracked.txt", "scratch/notes.txt"),
    )

    assert verdict.clean is True
    assert verdict.unexpected_changes == []
    assert verdict.reasons == []


def test_explicit_allow_dirty_accepts_all_local_changes_but_records_reasons(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    _write(tmp_path, "tracked.txt", "modified\n")
    _write(tmp_path, "new-file.txt", "new\n")

    verdict = validate_git_cleanliness(tmp_path, allow_dirty=True)

    assert verdict.clean is True
    assert verdict.unexpected_changes == []
    _assert_contains_reason(verdict.reasons, "explicitly allowed dirty tree")
    _assert_contains_reason(verdict.reasons, "working tree modified: tracked.txt")
    _assert_contains_reason(verdict.reasons, "untracked: new-file.txt")


def test_get_current_branch_returns_expected_baseline(tmp_path: Path) -> None:
    """get_current_branch returns the branch name matching the expected baseline."""
    _init_repo(tmp_path)
    _git(tmp_path, ["checkout", "-b", git_cleanliness.EXPECTED_MIMO_MTP_WORK_BRANCH])
    branch = get_current_branch(tmp_path)
    assert branch == git_cleanliness.EXPECTED_MIMO_MTP_WORK_BRANCH


def test_get_current_branch_returns_actual_worktree_branch() -> None:
    """get_current_branch returns the real git branch of this worktree.

    This test runs against the actual repository worktree (not a temp repo)
    and asserts the returned branch matches the canonical expected baseline
    branch for the MiMo V2.5 Pro MTP rollout.
    """
    worktree_root = Path(__file__).resolve().parent.parent
    branch = get_current_branch(worktree_root)
    expected = git_cleanliness.EXPECTED_MIMO_MTP_WORK_BRANCH
    assert (
        branch == expected
    ), f"current branch {branch!r} does not match expected baseline {expected!r}"


# ---------------------------------------------------------------------------
# get_worktree_status tests
# ---------------------------------------------------------------------------


def test_get_worktree_status_returns_clean_for_fresh_repo(tmp_path: Path) -> None:
    """A freshly initialised repo with no changes must report status='clean'."""
    _init_repo(tmp_path)
    result = get_worktree_status(tmp_path)
    assert result.status == "clean"
    assert result.change_count == 0
    assert result.reasons == ()
    assert result.branch != ""
    assert result.repository_root == tmp_path.resolve()


def test_get_worktree_status_returns_dirty_when_working_tree_modified(
    tmp_path: Path,
) -> None:
    """Modifying a tracked file must flip the status to 'dirty'."""
    _init_repo(tmp_path)
    _write(tmp_path, "tracked.txt", "modified\n")
    result = get_worktree_status(tmp_path)
    assert result.status == "dirty"
    assert result.change_count >= 1
    _assert_contains_reason(result.reasons, "working tree modified: tracked.txt")


def test_get_worktree_status_returns_dirty_when_index_modified(tmp_path: Path) -> None:
    """Staging a change must flip the status to 'dirty'."""
    _init_repo(tmp_path)
    _write(tmp_path, "tracked.txt", "staged\n")
    _git(tmp_path, ["add", "tracked.txt"])
    result = get_worktree_status(tmp_path)
    assert result.status == "dirty"
    assert result.change_count >= 1
    _assert_contains_reason(result.reasons, "index modified: tracked.txt")


def test_get_worktree_status_returns_dirty_for_untracked(tmp_path: Path) -> None:
    """An untracked file must flip the status to 'dirty'."""
    _init_repo(tmp_path)
    _write(tmp_path, "new-untracked.txt", "new\n")
    result = get_worktree_status(tmp_path)
    assert result.status == "dirty"
    assert result.change_count >= 1
    _assert_contains_reason(result.reasons, "untracked: new-untracked.txt")


def test_get_worktree_status_reports_branch(tmp_path: Path) -> None:
    """The branch field must match the current git branch."""
    _init_repo(tmp_path)
    _git(tmp_path, ["checkout", "-b", "feature/worktree-branch-test"])
    result = get_worktree_status(tmp_path)
    assert result.branch == "feature/worktree-branch-test"


def test_get_worktree_status_resolves_repository_root_from_subdirectory(
    tmp_path: Path,
) -> None:
    """Calling get_worktree_status from a subdirectory must still resolve
    the repository root to the top-level directory."""
    _init_repo(tmp_path)
    sub = tmp_path / "deep" / "nested"
    sub.mkdir(parents=True)
    result = get_worktree_status(sub)
    assert result.repository_root == tmp_path.resolve()


def test_get_worktree_status_matches_baseline_for_actual_worktree() -> None:
    """get_worktree_status against this worktree must return 'dirty' because
    the governed MiMo MTP rollout worktree has in-progress changes, and the
    branch must match the canonical expected baseline branch.

    This is the key baseline-matching assertion: the function must correctly
    report the actual worktree state (dirty) and the expected branch.
    """
    worktree_root = Path(__file__).resolve().parent.parent
    result = get_worktree_status(worktree_root)
    # The worktree is actively being worked on, so it must be dirty
    assert result.status == "dirty", (
        f"expected status='dirty' for active worktree but got status={result.status!r}"
    )
    assert result.change_count > 0, "dirty worktree must have change_count > 0"
    # Branch must match the governed work branch
    expected_branch = git_cleanliness.EXPECTED_MIMO_MTP_WORK_BRANCH
    assert result.branch == expected_branch, (
        f"branch {result.branch!r} does not match expected baseline {expected_branch!r}"
    )
    # Repository root must be this worktree (not the main tree)
    assert result.repository_root == worktree_root.resolve(), (
        f"repository_root {result.repository_root!r} does not match "
        f"worktree root {worktree_root.resolve()!r}"
    )
