import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import mlx.core as mx
import pytest

from scripts import mimo_v25_pro_mtp_module_probe as probe

EXPECTED_MTP_OUTPUT_SHARD_NAME = "model_mtp-00001-of-00001.safetensors"
SaveSafetensors = Callable[
    [str | Path, dict[str, mx.array], dict[str, str] | None], object
]
SAVE_SAFETENSORS = cast(SaveSafetensors, mx.save_safetensors)


def _write_tiny_quantized_artifact(root: Path) -> None:
    root.mkdir()
    eh_weight, eh_scales, eh_biases = mx.quantize(
        mx.ones((2, 64)), group_size=64, bits=6, mode="affine"
    )
    qkv_weight, qkv_scales, qkv_biases = mx.quantize(
        mx.ones((4, 64)), group_size=64, bits=6, mode="affine"
    )
    SAVE_SAFETENSORS(
        root / EXPECTED_MTP_OUTPUT_SHARD_NAME,
        {
            "model.mtp.layers.0.eh_proj.weight": eh_weight,
            "model.mtp.layers.0.eh_proj.weight.scales": eh_scales,
            "model.mtp.layers.0.eh_proj.weight.biases": eh_biases,
            "model.mtp.layers.0.self_attn.qkv_proj.weight": qkv_weight,
            "model.mtp.layers.0.self_attn.qkv_proj.weight.scales": qkv_scales,
            "model.mtp.layers.0.self_attn.qkv_proj.weight.biases": qkv_biases,
            "model.mtp.layers.0.enorm.weight": mx.ones((2,)),
        },
        {"format": "mlx-affine-6bit-mtp-only"},
    )
    index = {
        "metadata": {
            "artifact_kind": "mimo-v25-pro-mtp-only",
            "mtp_layers": [0],
            "save_format": "mlx-affine-6bit",
        },
        "weight_map": {
            "model.mtp.layers.0.eh_proj.weight": EXPECTED_MTP_OUTPUT_SHARD_NAME,
            "model.mtp.layers.0.eh_proj.weight.scales": EXPECTED_MTP_OUTPUT_SHARD_NAME,
            "model.mtp.layers.0.eh_proj.weight.biases": EXPECTED_MTP_OUTPUT_SHARD_NAME,
            "model.mtp.layers.0.self_attn.qkv_proj.weight": EXPECTED_MTP_OUTPUT_SHARD_NAME,
            "model.mtp.layers.0.self_attn.qkv_proj.weight.scales": EXPECTED_MTP_OUTPUT_SHARD_NAME,
            "model.mtp.layers.0.self_attn.qkv_proj.weight.biases": EXPECTED_MTP_OUTPUT_SHARD_NAME,
            "model.mtp.layers.0.enorm.weight": EXPECTED_MTP_OUTPUT_SHARD_NAME,
        },
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )


def test_probe_uses_expected_output_shard_name() -> None:
    assert probe.MTP_OUTPUT_SHARD_NAME == EXPECTED_MTP_OUTPUT_SHARD_NAME


def test_probe_rejects_production_artifact_path() -> None:
    with pytest.raises(ValueError, match="experimental MTP artifact"):
        probe.guard_probe_artifact_path(probe.PRODUCTION_6BIT_ARTIFACT_ROOT)


def test_probe_validates_keys_and_runs_synthetic_matmuls(tmp_path: Path) -> None:
    artifact = tmp_path / "kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental"
    _write_tiny_quantized_artifact(artifact)

    report = probe.probe_mtp_artifact(
        artifact,
        layers=(0,),
        eh_input_size=64,
        qkv_input_size=64,
    )

    assert report["artifact_kind"] == "mimo-v25-pro-mtp-only"
    assert report["layers"] == [0]
    assert report["loaded_tensor_count"] == 7
    assert report["synthetic_forwards"]["layer_0_eh_proj"] == [1, 1, 2]
    assert report["synthetic_forwards"]["layer_0_qkv_proj"] == [1, 1, 4]


def test_probe_fails_when_required_quantized_key_is_missing(tmp_path: Path) -> None:
    artifact = tmp_path / "kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental"
    _write_tiny_quantized_artifact(artifact)
    index = cast(
        dict[str, object],
        json.loads((artifact / "model.safetensors.index.json").read_text()),
    )
    weight_map = cast(dict[str, str], index["weight_map"])
    weight_map.pop("model.mtp.layers.0.eh_proj.weight.biases")
    (artifact / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )

    with pytest.raises(ValueError, match="missing required MTP keys"):
        probe.probe_mtp_artifact(
            artifact,
            layers=(0,),
            eh_input_size=64,
            qkv_input_size=64,
        )
