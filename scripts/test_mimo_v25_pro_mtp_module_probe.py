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
    tensors: dict[str, mx.array] = {}
    weight_map: dict[str, str] = {}
    for layer in probe.DEFAULT_LAYERS:
        eh_weight, eh_scales, eh_biases = mx.quantize(
            mx.ones((2, 64)), group_size=64, bits=6, mode="affine"
        )
        qkv_weight, qkv_scales, qkv_biases = mx.quantize(
            mx.ones((4, 64)), group_size=64, bits=6, mode="affine"
        )
        layer_prefix = f"model.mtp.layers.{layer}"
        tensors[f"{layer_prefix}.eh_proj.weight"] = eh_weight
        tensors[f"{layer_prefix}.eh_proj.weight.scales"] = eh_scales
        tensors[f"{layer_prefix}.eh_proj.weight.biases"] = eh_biases
        tensors[f"{layer_prefix}.self_attn.qkv_proj.weight"] = qkv_weight
        tensors[f"{layer_prefix}.self_attn.qkv_proj.weight.scales"] = qkv_scales
        tensors[f"{layer_prefix}.self_attn.qkv_proj.weight.biases"] = qkv_biases
        tensors[f"{layer_prefix}.enorm.weight"] = mx.ones((2,))
        weight_map.update(
            {
                f"{layer_prefix}.eh_proj.weight": EXPECTED_MTP_OUTPUT_SHARD_NAME,
                f"{layer_prefix}.eh_proj.weight.scales": EXPECTED_MTP_OUTPUT_SHARD_NAME,
                f"{layer_prefix}.eh_proj.weight.biases": EXPECTED_MTP_OUTPUT_SHARD_NAME,
                f"{layer_prefix}.self_attn.qkv_proj.weight": EXPECTED_MTP_OUTPUT_SHARD_NAME,
                f"{layer_prefix}.self_attn.qkv_proj.weight.scales": EXPECTED_MTP_OUTPUT_SHARD_NAME,
                f"{layer_prefix}.self_attn.qkv_proj.weight.biases": EXPECTED_MTP_OUTPUT_SHARD_NAME,
                f"{layer_prefix}.enorm.weight": EXPECTED_MTP_OUTPUT_SHARD_NAME,
            }
        )
    SAVE_SAFETENSORS(
        root / EXPECTED_MTP_OUTPUT_SHARD_NAME,
        tensors,
        {"format": "mlx-affine-6bit-mtp-only"},
    )
    index = {
        "metadata": {
            "artifact_kind": "mimo-v25-pro-mtp-only",
            "mtp_layers": list(probe.DEFAULT_LAYERS),
            "save_format": "mlx-affine-6bit",
        },
        "weight_map": weight_map,
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )


def test_probe_uses_expected_output_shard_name() -> None:
    assert probe.MTP_OUTPUT_SHARD_NAME == EXPECTED_MTP_OUTPUT_SHARD_NAME


def test_probe_rejects_production_artifact_path() -> None:
    with pytest.raises(ValueError, match="experimental MTP artifact"):
        probe.guard_probe_artifact_path(probe.PRODUCTION_6BIT_ARTIFACT_ROOT)


def test_probe_rejects_relative_traversal_path() -> None:
    with pytest.raises(ValueError, match=r"\.\."):
        probe.guard_probe_artifact_path(
            Path("fixtures") / ".." / "kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental"
        )


def test_probe_validates_keys_and_runs_synthetic_matmuls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    artifact = Path("kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental")
    _write_tiny_quantized_artifact(artifact)

    report = probe.probe_mtp_artifact(
        artifact,
        layers=(0,),
        eh_input_size=64,
        qkv_input_size=64,
    )

    assert report["artifact_kind"] == "mimo-v25-pro-mtp-only"
    assert report["layers"] == [0]
    assert report["loaded_tensor_count"] == 21
    assert report["synthetic_forwards"]["layer_0_eh_proj"] == [1, 1, 2]
    assert report["synthetic_forwards"]["layer_0_qkv_proj"] == [1, 1, 4]


def test_probe_cli_main_writes_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    artifact = Path("kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental")
    json_out = Path("report.json")
    _write_tiny_quantized_artifact(artifact)
    original_probe = probe.probe_mtp_artifact

    def _tiny_probe(artifact_dir: Path) -> probe.MtpProbeReport:
        return original_probe(
            artifact_dir,
            eh_input_size=64,
            qkv_input_size=64,
        )

    monkeypatch.setattr(probe, "probe_mtp_artifact", _tiny_probe)

    exit_code = probe.main(["--artifact", str(artifact), "--json-out", str(json_out)])

    assert exit_code == 0
    data = cast(dict[str, object], json.loads(json_out.read_text()))
    assert data["artifact_kind"] == "mimo-v25-pro-mtp-only"
    assert data["layers"] == [0, 1, 2]
    assert data["loaded_tensor_count"] == 21


def test_probe_cli_requires_artifact_argument() -> None:
    with pytest.raises(SystemExit, match="2"):
        probe.main([])


def test_probe_fails_when_required_quantized_key_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    artifact = Path("kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental")
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
