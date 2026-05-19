import json
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import cast

import mlx.core as mx
import pytest
import safetensors.torch as safetensors_torch
import torch

from scripts import mimo_v25_pro_mtp_quantize as mtp_quantize

SaveFile = Callable[
    [dict[str, torch.Tensor], str | PathLike[str], dict[str, str] | None],
    None,
]
SAVE_FILE = cast(SaveFile, safetensors_torch.save_file)


def _write_mtp_fixture(path: Path) -> None:
    SAVE_FILE(
        {
            "model.mtp.layers.0.eh_proj.weight": torch.arange(
                128, dtype=torch.float16
            ).reshape(2, 64),
            "model.mtp.layers.0.enorm.weight": torch.ones(
                (2,), dtype=torch.float16
            ),
            "model.mtp.layers.0.self_attn.qkv_proj.weight": torch.arange(
                128, dtype=torch.float16
            ).reshape(2, 64),
            "model.mtp.layers.0.self_attn.attention_sink_bias": torch.zeros(
                (2,), dtype=torch.float16
            ),
            "model.layers.0.mlp.down_proj.weight": torch.ones(
                (2, 64), dtype=torch.float16
            ),
        },
        path,
        None,
    )


@pytest.fixture()
def mtp_fixture_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_shard = source_dir / "model_mtp.safetensors"
    output_dir = tmp_path / "kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental"
    _write_mtp_fixture(source_shard)
    monkeypatch.setattr(mtp_quantize, "MTP_SOURCE_SHARD", source_shard)
    monkeypatch.setattr(mtp_quantize, "MTP_OUTPUT_ROOT", output_dir)
    monkeypatch.setattr(mtp_quantize, "MTP_HEADROOM_BYTES", 0)
    return source_shard, output_dir


def test_mtp_output_guard_rejects_production_artifact_path() -> None:
    with pytest.raises(ValueError, match="experimental MTP output"):
        mtp_quantize.guard_mtp_output_path(
            mtp_quantize.PRODUCTION_6BIT_ARTIFACT_ROOT
        )


def test_mtp_source_guard_rejects_non_default_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="MTP source shard"):
        mtp_quantize.guard_mtp_source_shard(tmp_path / "model_mtp.safetensors")


def test_mtp_output_guard_accepts_experimental_root_and_child(
    mtp_fixture_paths: tuple[Path, Path],
) -> None:
    _source_shard, output_dir = mtp_fixture_paths

    mtp_quantize.guard_mtp_output_path(output_dir)
    mtp_quantize.guard_mtp_output_path(output_dir / "run-0001")


def test_build_mtp_manifest_filters_only_mtp_tensors(
    mtp_fixture_paths: tuple[Path, Path],
) -> None:
    source_shard, output_dir = mtp_fixture_paths

    manifest = mtp_quantize.build_mtp_dry_run_manifest(
        mtp_quantize.MtpQuantizationPlan(source_shard, output_dir)
    )

    shard = manifest["shards"][mtp_quantize.MTP_OUTPUT_SHARD_NAME]
    assert shard["status"] == "dry-run"
    assert "model.mtp.layers.0.eh_proj.weight" in shard["tensors"]
    assert "model.layers.0.mlp.down_proj.weight" not in shard["tensors"]
    assert shard["tensors"]["model.mtp.layers.0.eh_proj.weight"]["quantized"] is True
    assert (
        shard["tensors"]["model.mtp.layers.0.enorm.weight"]["quantized"] is False
    )


def test_mtp_dry_run_writes_index_manifest_config_and_model_card(
    mtp_fixture_paths: tuple[Path, Path],
) -> None:
    source_shard, output_dir = mtp_fixture_paths

    manifest = mtp_quantize.run_mtp_dry_run(
        mtp_quantize.MtpQuantizationPlan(source_shard, output_dir)
    )

    assert manifest["model_id"] == mtp_quantize.MTP_EXPERIMENTAL_MODEL_ID
    assert (output_dir / mtp_quantize.MTP_MANIFEST_NAME).is_file()
    assert (output_dir / "model.safetensors.index.json").is_file()
    assert (output_dir / "config.json").is_file()
    assert (output_dir / "README.md").is_file()

    index = json.loads((output_dir / "model.safetensors.index.json").read_text())
    assert index["metadata"]["artifact_kind"] == "mimo-v25-pro-mtp-only"
    assert (
        index["weight_map"]["model.mtp.layers.0.eh_proj.weight.scales"]
        == mtp_quantize.MTP_OUTPUT_SHARD_NAME
    )

    config = json.loads((output_dir / "config.json").read_text())
    assert config["artifact_kind"] == "mimo-v25-pro-mtp-only"
    assert config["base_model_id"] == mtp_quantize.PRODUCTION_6BIT_MODEL_ID
    assert config["mtp_config"] == {"enabled": True, "num_layers": 3}

    readme = (output_dir / "README.md").read_text()
    assert mtp_quantize.MTP_EXPERIMENTAL_MODEL_ID in readme
    assert "not a standalone runtime model" in readme


def test_mtp_conversion_writes_quantized_shard_and_no_tmp_files(
    mtp_fixture_paths: tuple[Path, Path],
) -> None:
    source_shard, output_dir = mtp_fixture_paths

    manifest = mtp_quantize.run_mtp_conversion(
        mtp_quantize.MtpQuantizationPlan(source_shard, output_dir)
    )

    shard = manifest["shards"][mtp_quantize.MTP_OUTPUT_SHARD_NAME]
    assert shard["status"] == "complete"
    assert shard["output_size"] > 0
    assert not list(output_dir.glob("*.tmp"))
    assert not list(output_dir.glob(".*.tmp"))

    loaded = mx.load(str(output_dir / mtp_quantize.MTP_OUTPUT_SHARD_NAME))
    assert loaded["model.mtp.layers.0.eh_proj.weight"].dtype == mx.uint32
    assert loaded["model.mtp.layers.0.eh_proj.weight.scales"].shape[0] == 2
    assert loaded["model.mtp.layers.0.enorm.weight"].shape == (2,)
    assert "model.layers.0.mlp.down_proj.weight" not in loaded
