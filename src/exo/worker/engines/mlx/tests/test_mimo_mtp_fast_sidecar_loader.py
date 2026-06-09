from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

import mlx.core as mx
import numpy as np
import pytest

from exo.worker.engines.mlx.mimo_mtp_fast.sidecar_contract import (
    MIMO_MTP_LAYER_COUNT,
    MIMO_MTP_REQUIRED_SUFFIXES,
    official_mimo_mtp_key,
)
from exo.worker.engines.mlx.mimo_mtp_fast.sidecar_loader import (
    MimoMtpSidecarLoadError,
    load_mimo_mtp_sidecar_tensors,
)


class _SaveFileFn(Protocol):
    def __call__(
        self,
        tensor_dict: dict[str, np.ndarray],
        filename: str,
        metadata: dict[str, str] | None = None,
    ) -> None: ...


_safetensors_numpy = import_module("safetensors.numpy")
_save_file = cast(_SaveFileFn, _safetensors_numpy.save_file)


def _tiny_tensor_for_suffix(layer_index: int, suffix: str) -> np.ndarray:
    base_value = float(layer_index + 1)
    if suffix.endswith((".weight.scales", ".weight.biases")):
        return np.full((1, 1), base_value, dtype=np.float32)
    if suffix.endswith(".weight") and suffix not in {
        "enorm.weight",
        "hnorm.weight",
        "final_layernorm.weight",
        "input_layernorm.weight",
        "pre_mlp_layernorm.weight",
    }:
        return np.full((2, 2), base_value, dtype=np.float32)
    return np.full((2,), base_value, dtype=np.float32)


def _write_synthetic_official_sidecar(
    path: Path, *, omit_key: str | None = None
) -> None:
    tensors: dict[str, np.ndarray] = {}
    for layer_index in range(MIMO_MTP_LAYER_COUNT):
        for suffix in MIMO_MTP_REQUIRED_SUFFIXES:
            key = official_mimo_mtp_key(layer_index, suffix)
            if key == omit_key:
                continue
            tensors[key] = _tiny_tensor_for_suffix(layer_index, suffix)
    _save_file(tensors, str(path))


def test_loads_three_official_layout_layer_records(tmp_path: Path) -> None:
    sidecar_path = tmp_path / "model_mtp.safetensors"
    _write_synthetic_official_sidecar(sidecar_path)

    sidecar = load_mimo_mtp_sidecar_tensors(sidecar_path)

    assert sidecar.path == sidecar_path
    assert len(sidecar.layers) == 3
    assert [layer.layer_index for layer in sidecar.layers] == [0, 1, 2]
    layer_one = sidecar.layers[1]
    assert layer_one.eh_proj_weight.shape == (2, 2)
    assert layer_one.qkv_proj_weight.shape == (2, 2)
    assert layer_one.qkv_proj_scale_inv.shape == (1, 1)
    assert layer_one.qkv_proj_biases.shape == (1, 1)
    assert layer_one.gate_proj_weight.shape == (2, 2)
    assert layer_one.up_proj_weight.shape == (2, 2)
    assert layer_one.down_proj_weight.shape == (2, 2)
    assert float(mx.array(layer_one.down_proj_weight)[0, 0]) == 2.0


def test_get_returns_tensor_by_official_suffix(tmp_path: Path) -> None:
    sidecar_path = tmp_path / "model_mtp.safetensors"
    _write_synthetic_official_sidecar(sidecar_path)

    sidecar = load_mimo_mtp_sidecar_tensors(sidecar_path)

    tensor = sidecar.layers[2].get("mlp.down_proj.weight.scales")
    assert tensor.shape == (1, 1)
    assert float(mx.array(tensor)[0, 0]) == 3.0


def test_loader_refuses_invalid_sidecar_before_materializing(tmp_path: Path) -> None:
    sidecar_path = tmp_path / "model_mtp.safetensors"
    omitted_key = official_mimo_mtp_key(0, "mlp.up_proj.weight")
    _write_synthetic_official_sidecar(sidecar_path, omit_key=omitted_key)

    with pytest.raises(MimoMtpSidecarLoadError) as exc_info:
        load_mimo_mtp_sidecar_tensors(sidecar_path)

    assert omitted_key in str(exc_info.value)
