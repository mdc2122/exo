from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from scripts import bench_mimo_mtp_fastpath as bench


def test_bench_dry_run_contract_only_accepts_missing_sidecar_without_model_load(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_sidecar = tmp_path / "model_mtp.safetensors"

    exit_code = bench.main(
        [
            "--sidecar-path",
            str(missing_sidecar),
            "--dry-run-contract-only",
        ]
    )

    assert exit_code == 0
    row = cast(dict[str, object], json.loads(capsys.readouterr().out))
    assert row["kind"] == "contract_probe"
    assert row["ready"] is False
    assert row["sidecar_status"] == "missing"
    assert row["sidecar_path"] == str(missing_sidecar)


def test_bench_non_dry_run_reports_clear_missing_model_path_without_loading_sidecar(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_model = tmp_path / "missing-model"
    missing_sidecar = tmp_path / "missing-sidecar.safetensors"

    exit_code = bench.main(
        [
            "--model-path",
            str(missing_model),
            "--sidecar-path",
            str(missing_sidecar),
        ]
    )

    assert exit_code == 2
    row = cast(dict[str, object], json.loads(capsys.readouterr().out))
    assert row == {
        "kind": "validation_error",
        "field": "model_path",
        "path": str(missing_model),
        "error": "MiMo MTP benchmark model path does not exist",
        "next_step": "provide an existing local MiMo model snapshot path or use --dry-run-contract-only",
    }


def test_bench_non_dry_run_reports_clear_invalid_sidecar_before_model_load(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_path = tmp_path / "model"
    missing_sidecar = tmp_path / "missing-sidecar.safetensors"
    model_path.mkdir()

    exit_code = bench.main(
        [
            "--model-path",
            str(model_path),
            "--sidecar-path",
            str(missing_sidecar),
        ]
    )

    assert exit_code == 2
    row = cast(dict[str, object], json.loads(capsys.readouterr().out))
    assert row["kind"] == "validation_error"
    assert row["field"] == "sidecar_path"
    assert row["path"] == str(missing_sidecar)
    assert row["error"] == "MiMo MTP sidecar is not ready"
    assert row["sidecar_status"] == "missing"
    assert (
        row["next_step"]
        == "run --dry-run-contract-only or provide a valid official-layout sidecar"
    )
