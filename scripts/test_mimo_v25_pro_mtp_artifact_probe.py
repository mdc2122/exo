import json
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import cast

import safetensors.torch as safetensors_torch
import torch

from scripts import mimo_v25_pro_mtp_artifact_probe as probe

SaveFile = Callable[
    [dict[str, torch.Tensor], str | PathLike[str], dict[str, str] | None],
    None,
]
SAVE_FILE = cast(SaveFile, safetensors_torch.save_file)


def _write_fixture(path: Path) -> None:
    SAVE_FILE(
        {
            "model.mtp.layers.0.self_attn.qkv_proj.weight": torch.zeros(
                (4, 4), dtype=torch.float16
            ),
            "model.mtp.layers.0.self_attn.qkv_proj.weight_scale_inv": torch.ones(
                (1, 1), dtype=torch.float32
            ),
            "model.mtp.layers.1.self_attn.qkv_proj.weight": torch.zeros(
                (4, 4), dtype=torch.float16
            ),
            "model.mtp.layers.2.mlp.down_proj.weight": torch.zeros(
                (4, 4), dtype=torch.float16
            ),
            "model.layers.0.mlp.down_proj.weight": torch.zeros(
                (4, 4), dtype=torch.float16
            ),
        },
        path,
        None,
    )


def test_probe_reports_only_mtp_tensors(tmp_path: Path) -> None:
    shard = tmp_path / "model_mtp.safetensors"
    _write_fixture(shard)

    report = probe.probe_mtp_shard(shard)

    assert report["source_file"] == str(shard)
    assert report["tensor_count"] == 4
    assert report["layers"] == [0, 1, 2]
    assert "model.layers.0.mlp.down_proj.weight" not in report["tensors"]
    assert report["tensors"]["model.mtp.layers.0.self_attn.qkv_proj.weight"][
        "shape"
    ] == [4, 4]


def test_probe_marks_missing_expected_layers(tmp_path: Path) -> None:
    shard = tmp_path / "model_mtp.safetensors"
    SAVE_FILE(
        {
            "model.mtp.layers.0.self_attn.qkv_proj.weight": torch.zeros(
                (4, 4), dtype=torch.float16
            )
        },
        shard,
        None,
    )

    report = probe.probe_mtp_shard(shard)

    assert report["layers"] == [0]
    assert report["missing_expected_layers"] == [1, 2]
    assert report["complete_expected_layers"] is False


def test_probe_cli_writes_json(tmp_path: Path) -> None:
    shard = tmp_path / "model_mtp.safetensors"
    output = tmp_path / "report.json"
    _write_fixture(shard)

    exit_code = probe.main([str(shard), "--json-out", str(output)])

    assert exit_code == 0
    data = cast(dict[str, object], json.loads(output.read_text()))
    assert data["complete_expected_layers"] is True
