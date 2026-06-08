from __future__ import annotations

from pathlib import Path

import pytest

from scripts import mimo_v25_pro_repo_root as repo_root


def test_validate_repo_root_accepts_expected_root(tmp_path: Path) -> None:
    expected_root = tmp_path / "exo"
    (expected_root / "src" / "exo").mkdir(parents=True)
    (expected_root / "pyproject.toml").write_text(
        '[project]\nname = "exo"\n', encoding="utf-8"
    )
    (expected_root / "Cargo.toml").write_text(
        "[workspace]\nmembers = []\n", encoding="utf-8"
    )

    validation = repo_root.validate_repo_root(
        start_path=expected_root,
        expected_root=expected_root,
    )

    assert validation.root == expected_root.resolve()
    assert validation.expected_root == expected_root.resolve()
    assert validation.start_path == expected_root.resolve()
    assert validation.markers == (
        "pyproject.toml",
        "Cargo.toml",
        "src/exo",
    )


def test_validate_repo_root_detects_expected_root_from_nested_path(
    tmp_path: Path,
) -> None:
    expected_root = tmp_path / "exo"
    nested_path = expected_root / ".goose-ultrawork" / "waves" / "worker-1"
    nested_path.mkdir(parents=True)
    (expected_root / "src" / "exo").mkdir(parents=True)
    (expected_root / "pyproject.toml").write_text(
        '[project]\nname = "exo"\n', encoding="utf-8"
    )
    (expected_root / "Cargo.toml").write_text(
        "[workspace]\nmembers = []\n", encoding="utf-8"
    )

    validation = repo_root.validate_repo_root(
        start_path=nested_path,
        expected_root=expected_root,
    )

    assert validation.root == expected_root.resolve()
    assert validation.start_path == nested_path.resolve()


def test_main_reports_repo_root_from_arbitrary_current_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected_root = tmp_path / "exo"
    arbitrary_working_directory = (
        expected_root / ".goose-ultrawork" / "waves" / "worker-1"
    )
    arbitrary_working_directory.mkdir(parents=True)
    (expected_root / "src" / "exo").mkdir(parents=True)
    (expected_root / "pyproject.toml").write_text(
        '[project]\nname = "exo"\n', encoding="utf-8"
    )
    (expected_root / "Cargo.toml").write_text(
        "[workspace]\nmembers = []\n", encoding="utf-8"
    )
    monkeypatch.chdir(arbitrary_working_directory)

    exit_code = repo_root.main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == f"{expected_root.resolve()}\n"
    assert captured.err == ""


def test_validate_repo_root_rejects_non_exo_directory(tmp_path: Path) -> None:
    wrong_root = tmp_path / "not-exo"
    wrong_root.mkdir()

    with pytest.raises(repo_root.RepoRootValidationError) as exc_info:
        repo_root.validate_repo_root(start_path=wrong_root)

    assert "could not find expected exo repository root" in str(exc_info.value)
    assert str(wrong_root.resolve()) in str(exc_info.value)


def test_main_reports_clear_diagnostic_when_outside_repo_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    wrong_root = tmp_path / "not-exo"
    wrong_root.mkdir()
    monkeypatch.chdir(wrong_root)

    exit_code = repo_root.main([])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "error: could not find expected exo repository root" in captured.err
    assert str(wrong_root.resolve()) in captured.err
    assert "required markers: pyproject.toml, Cargo.toml, src/exo" in captured.err
    assert "Traceback" not in captured.err


def test_validate_repo_root_rejects_mismatched_expected_root(tmp_path: Path) -> None:
    detected_root = tmp_path / "exo"
    mismatched_root = tmp_path / "other-exo"
    (detected_root / "src" / "exo").mkdir(parents=True)
    (detected_root / "pyproject.toml").write_text(
        '[project]\nname = "exo"\n', encoding="utf-8"
    )
    (detected_root / "Cargo.toml").write_text(
        "[workspace]\nmembers = []\n", encoding="utf-8"
    )
    mismatched_root.mkdir()

    with pytest.raises(repo_root.RepoRootValidationError) as exc_info:
        repo_root.validate_repo_root(
            start_path=detected_root,
            expected_root=mismatched_root,
        )

    message = str(exc_info.value)
    assert "detected repository root does not match expected root" in message
    assert str(detected_root.resolve()) in message
    assert str(mismatched_root.resolve()) in message
