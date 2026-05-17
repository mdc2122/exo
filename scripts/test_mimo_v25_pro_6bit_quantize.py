import json
from pathlib import Path

import mlx.core as mx
import pytest
import torch
from safetensors.torch import save_file

from scripts import mimo_v25_pro_6bit_quantize as quantize


def _write_minimal_source_fixture(source: Path) -> None:
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "_name_or_path": "XiaomiMiMo/MiMo-V2.5-Pro",
                "architectures": ["MiMoV2ForCausalLM"],
                "model_type": "mimo_v2",
                "quantization_config": {"quant_method": "fp8"},
            }
        )
    )
    (source / "tokenizer.json").write_text("{}")
    save_file(
        {
            "layer.weight": torch.arange(128, dtype=torch.float16).reshape(2, 64),
            "layer.bias": torch.ones(2, dtype=torch.float16),
        },
        source / "model-00001-of-00001.safetensors",
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 260, "save_format": "pt"},
                "weight_map": {
                    "layer.weight": "model-00001-of-00001.safetensors",
                    "layer.bias": "model-00001-of-00001.safetensors",
                },
            }
        )
    )


@pytest.fixture()
def approved_fixture_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    source = tmp_path / "official-source"
    output = tmp_path / "approved-output"
    _write_minimal_source_fixture(source)
    monkeypatch.setattr(quantize, "SOURCE_CHECKPOINT", source)
    monkeypatch.setattr(quantize, "OUTPUT_ROOT", output)
    monkeypatch.setattr(quantize, "HEADROOM_BYTES", 0)
    return source, output


def test_output_path_guard_rejects_source_overlap() -> None:
    with pytest.raises(ValueError, match="must be"):
        quantize.guard_output_path(Path("/tmp/not-the-approved-mimo-output"))

    with pytest.raises(ValueError, match="must be"):
        quantize.guard_output_path(quantize.SOURCE_CHECKPOINT)


def test_output_path_guard_accepts_approved_root_and_child() -> None:
    quantize.guard_output_path(quantize.OUTPUT_ROOT)
    quantize.guard_output_path(quantize.OUTPUT_ROOT / "20260505T000000Z")


def test_source_path_guard_rejects_non_official_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="official checkpoint"):
        quantize.guard_source_path(tmp_path)


def test_quantized_payload_nbytes_for_mlx_affine_6bit_group64() -> None:
    # MLX packs 6-bit values into uint32 columns: 64 source columns -> 12 uint32s.
    # Scales and biases are one fp32 each per group per row.
    assert quantize.quantized_payload_nbytes((4, 64), group_size=64, bits=6) == (
        4 * 12 * 4 + 4 * 1 * 4 + 4 * 1 * 4
    )


def test_tensor_plan_quantizes_fp8_weight_with_same_shard_scale() -> None:
    entry = quantize.estimate_tensor_entry(
        "model.layers.0.mlp.down_proj.weight",
        [6144, 16384],
        "F8_E4M3",
        100_663_296,
        {
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.mlp.down_proj.weight_scale_inv",
        },
        group_size=64,
        bits=6,
    )

    assert entry["quantized"] is True
    assert entry["output_keys"] == [
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.mlp.down_proj.weight.scales",
        "model.layers.0.mlp.down_proj.weight.biases",
    ]
    assert entry["estimated_output_nbytes"] < entry["source_nbytes"]


def test_tensor_plan_skips_unpaired_fp8_weight() -> None:
    entry = quantize.estimate_tensor_entry(
        "layer.weight",
        [128, 128],
        "F8_E4M3",
        16_384,
        {"layer.weight"},
        group_size=64,
        bits=6,
    )

    assert entry["quantized"] is False
    assert "no same-shard" in entry["reason"]


def test_broadcast_scale_inv_crops_extra_fp8_block_rows_for_mtp_qkv() -> None:
    scale_inv = torch.arange(216 * 48, dtype=torch.float32).reshape(216, 48)

    broadcast = quantize._broadcast_scale_inv(scale_inv, (27136, 6144))

    assert broadcast.shape == (27136, 6144)
    assert broadcast[0, 0] == scale_inv[0, 0]
    assert broadcast[127, 127] == scale_inv[0, 0]
    assert broadcast[128, 0] == scale_inv[1, 0]
    assert broadcast[27135, 6143] == scale_inv[211, 47]


def test_broadcast_scale_inv_preserves_normal_divisible_fp8_blocks() -> None:
    scale_inv = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

    broadcast = quantize._broadcast_scale_inv(scale_inv, (256, 256))

    assert broadcast.shape == (256, 256)
    assert torch.all(broadcast[:128, :128] == 1.0)
    assert torch.all(broadcast[:128, 128:] == 2.0)
    assert torch.all(broadcast[128:, :128] == 3.0)
    assert torch.all(broadcast[128:, 128:] == 4.0)


def test_write_config_replaces_official_fp8_quantization_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["MiMoV2ForCausalLM"],
                "model_type": "mimo_v2",
                "quantization_config": {"quant_method": "fp8"},
            }
        )
    )
    manifest: quantize.RunManifest = {
        "quantization": {"format": "mlx-affine-6bit-sharded"},
        "shards": {},
    }

    quantize.write_config(source, output, manifest)

    rewritten = json.loads((output / "config.json").read_text())
    assert rewritten["quantization"] == {"group_size": 64, "bits": 6, "mode": "affine"}
    assert rewritten["quantization_config"]["quant_method"] == "mlx-affine"
    assert (
        rewritten["quantization_config"]["custom_model_id"] == quantize.CUSTOM_MODEL_ID
    )


def test_manifest_round_trip_and_output_index_generation(tmp_path: Path) -> None:
    manifest_path = tmp_path / "quantization_manifest.json"
    manifest: quantize.RunManifest = {
        "model_id": quantize.CUSTOM_MODEL_ID,
        "shards": {
            "model-00001.safetensors": {
                "status": "complete",
                "tensors": {
                    "layer.weight": {
                        "estimated_output_nbytes": 224,
                        "quantized": True,
                        "output_keys": [
                            "layer.weight",
                            "layer.weight.scales",
                            "layer.weight.biases",
                        ],
                    },
                    "layer.weight_scale_inv": {
                        "estimated_output_nbytes": 16,
                        "quantized": False,
                    },
                },
            }
        },
    }

    quantize.write_manifest(manifest_path, manifest)
    loaded = quantize.load_manifest(manifest_path)
    assert loaded["model_id"] == quantize.CUSTOM_MODEL_ID

    quantize.write_output_index(tmp_path, loaded)
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert index["metadata"]["save_format"] == "mlx-affine-6bit"
    assert index["metadata"]["total_size"] == 224
    assert index["weight_map"]["layer.weight.scales"] == "model-00001.safetensors"
    assert "layer.weight_scale_inv" not in index["weight_map"]


def test_dry_run_writes_sizing_support_files_config_index_and_manifest(
    approved_fixture_paths: tuple[Path, Path],
) -> None:
    source, output = approved_fixture_paths

    manifest = quantize.run_dry_run(quantize.QuantizationPlan(source, output))

    manifest_path = output / "quantization_manifest.json"
    assert manifest_path.is_file()
    assert (output / "tokenizer.json").is_file()
    assert manifest["model_id"] == quantize.CUSTOM_MODEL_ID
    assert manifest["base_model_id"] == quantize.BASE_MODEL_ID
    assert manifest["expected_output_size_bytes"] == 116
    assert manifest["shards"]["model-00001-of-00001.safetensors"]["status"] == "dry-run"

    rewritten_config = json.loads((output / "config.json").read_text())
    assert "_name_or_path" not in rewritten_config
    assert (
        rewritten_config["quantization_config"]["base_model"] == quantize.BASE_MODEL_ID
    )
    assert (
        rewritten_config["quantization_config"]["custom_model_id"]
        == quantize.CUSTOM_MODEL_ID
    )

    output_index = json.loads((output / "model.safetensors.index.json").read_text())
    assert output_index["metadata"]["total_size"] == 116
    assert (
        output_index["weight_map"]["layer.weight.scales"]
        == "model-00001-of-00001.safetensors"
    )
    assert (
        output_index["weight_map"]["layer.bias"] == "model-00001-of-00001.safetensors"
    )


def test_conversion_pilot_writes_quantized_dtype_metadata_manifest_and_no_tmp_files(
    approved_fixture_paths: tuple[Path, Path],
) -> None:
    source, output = approved_fixture_paths

    manifest = quantize.run_conversion(quantize.QuantizationPlan(source, output))

    shard = manifest["shards"]["model-00001-of-00001.safetensors"]
    assert shard["status"] == "complete"
    assert shard["output_size"] > 0
    assert not list(output.glob("*.tmp"))
    assert not list(output.glob(".*.tmp"))

    loaded = mx.load(str(output / "model-00001-of-00001.safetensors"))
    assert loaded["layer.weight"].dtype == mx.uint32
    assert loaded["layer.weight.scales"].dtype in {mx.float16, mx.float32}
    assert loaded["layer.weight.biases"].dtype in {mx.float16, mx.float32}
    assert loaded["layer.bias"].dtype == mx.float16

    output_index = json.loads((output / "model.safetensors.index.json").read_text())
    assert output_index["metadata"]["save_format"] == "mlx-affine-6bit"
    assert (
        output_index["weight_map"]["layer.weight.biases"]
        == "model-00001-of-00001.safetensors"
    )


def test_run_dry_run_preserves_completed_shard_for_resumability(
    approved_fixture_paths: tuple[Path, Path],
) -> None:
    source, output = approved_fixture_paths
    output.mkdir()
    quantize.write_manifest(
        output / "quantization_manifest.json",
        {
            "model_id": quantize.CUSTOM_MODEL_ID,
            "shards": {
                "model-00001-of-00001.safetensors": {
                    "status": "complete",
                    "output_file": "model-00001-of-00001.safetensors",
                    "output_size": 123,
                    "tensors": {
                        "layer.weight": {
                            "estimated_output_nbytes": 112,
                            "quantized": True,
                            "output_keys": [
                                "layer.weight",
                                "layer.weight.scales",
                                "layer.weight.biases",
                            ],
                        }
                    },
                }
            },
        },
    )
    (output / "model-00001-of-00001.safetensors").write_bytes(b"already done")

    manifest = quantize.run_dry_run(quantize.QuantizationPlan(source, output))

    shard = manifest["shards"]["model-00001-of-00001.safetensors"]
    assert shard["status"] == "complete"
    assert shard["output_size"] == 123


def test_cleanup_incomplete_removes_hidden_tmp_files_and_marks_incomplete(
    approved_fixture_paths: tuple[Path, Path],
) -> None:
    _source, output = approved_fixture_paths
    output.mkdir()
    (output / ".model-00001-of-00001.safetensors.tmp").write_text("partial")
    (output / ".model-00002-of-00002.safetensors.tmp.safetensors").write_text("partial")
    quantize.write_manifest(
        output / "quantization_manifest.json",
        {
            "model_id": quantize.CUSTOM_MODEL_ID,
            "shards": {
                "model-00001-of-00001.safetensors": {
                    "status": "running",
                    "tensors": {},
                }
            },
        },
    )

    quantize.cleanup_incomplete(output)

    assert not (output / ".model-00001-of-00001.safetensors.tmp").exists()
    assert not (output / ".model-00002-of-00002.safetensors.tmp.safetensors").exists()
    manifest = quantize.load_manifest(output / "quantization_manifest.json")
    assert (
        manifest["shards"]["model-00001-of-00001.safetensors"]["status"]
        == "cleanup-required"
    )
