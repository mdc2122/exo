import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path

from scripts.git_cleanliness import validate_git_cleanliness


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
