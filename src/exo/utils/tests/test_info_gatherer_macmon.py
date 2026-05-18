import shutil
from pathlib import Path

from pytest import MonkeyPatch

import exo.utils.info_gatherer.info_gatherer as info_gatherer
from exo.utils.info_gatherer.macmon import MacmonMetrics, RawMacmonMetrics


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


def _raw_macmon_metrics(*, cpu_temp: float, gpu_temp: float) -> RawMacmonMetrics:
    return RawMacmonMetrics.model_validate(
        {
            "timestamp": "2026-05-18T05:12:05.940286+00:00",
            "temp": {"cpu_temp_avg": cpu_temp, "gpu_temp_avg": gpu_temp},
            "memory": {
                "ram_total": 549755813888,
                "ram_usage": 426740629504,
                "swap_total": 1073741824,
                "swap_usage": 54591488,
            },
            "ecpu_usage": (1118, 0.02628539502620697),
            "pcpu_usage": (1092, 0.003991651348769665),
            "gpu_usage": (0, 0.0),
            "all_power": 0.3829617202281952,
            "ane_power": 0.0,
            "cpu_power": 0.3829617202281952,
            "gpu_power": 0.0,
            "gpu_ram_power": 0.0,
            "ram_power": 0.6776460409164429,
            "sys_power": 16.085901260375977,
        }
    )


def test_macmon_metrics_uses_gpu_temperature_when_valid() -> None:
    metrics = MacmonMetrics.from_raw(_raw_macmon_metrics(cpu_temp=44.0, gpu_temp=41.5))

    assert metrics.system_profile.temp == 41.5


def test_macmon_metrics_falls_back_to_cpu_temp() -> None:
    metrics = MacmonMetrics.from_raw(
        _raw_macmon_metrics(cpu_temp=44.0, gpu_temp=-4.706249237060547)
    )

    assert metrics.system_profile.temp == 44.0
