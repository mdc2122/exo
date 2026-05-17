import shutil
from pathlib import Path

from pytest import MonkeyPatch

import exo.utils.info_gatherer.info_gatherer as info_gatherer


def _path_lookup(_: str) -> str:
    return "/path/bin/macmon"


def _missing_lookup(_: str) -> None:
    return None


def test_find_macmon_path_prefers_explicit_env(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("EXO_MACMON_PATH", "/custom/bin/macmon")
    monkeypatch.setattr(shutil, "which", _path_lookup)

    assert info_gatherer.find_macmon_path() == "/custom/bin/macmon"


def test_find_macmon_path_uses_path_lookup(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("EXO_MACMON_PATH", raising=False)
    monkeypatch.setattr(shutil, "which", _path_lookup)

    assert info_gatherer.find_macmon_path() == "/path/bin/macmon"


def test_find_macmon_path_checks_homebrew_paths(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    macmon = tmp_path / "macmon"
    macmon.write_text("#!/bin/sh\n")
    macmon.chmod(0o755)

    monkeypatch.delenv("EXO_MACMON_PATH", raising=False)
    monkeypatch.setattr(shutil, "which", _missing_lookup)
    monkeypatch.setattr(info_gatherer, "DEFAULT_MACMON_PATHS", (str(macmon),))

    assert info_gatherer.find_macmon_path() == str(macmon)


def test_find_macmon_path_returns_none_without_candidate(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-macmon"

    monkeypatch.delenv("EXO_MACMON_PATH", raising=False)
    monkeypatch.setattr(shutil, "which", _missing_lookup)
    monkeypatch.setattr(info_gatherer, "DEFAULT_MACMON_PATHS", (str(missing),))

    assert info_gatherer.find_macmon_path() is None
