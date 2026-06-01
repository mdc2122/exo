from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

import numpy as np

from exo.worker.engines.mlx.mimo_mtp_fast.sidecar_contract import (
    MIMO_MTP_LAYER_COUNT,
    MIMO_MTP_REQUIRED_SUFFIXES,
    official_mimo_mtp_key,
    probe_mimo_mtp_sidecar,
    required_mimo_mtp_keys,
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


def _tiny_tensor_for_suffix(suffix: str) -> np.ndarray:
    if suffix.endswith(".weight_scale_inv"):
        return np.ones((1, 1), dtype=np.float32)
    if suffix.endswith(".weight") and suffix not in {
        "enorm.weight",
        "hnorm.weight",
        "final_layernorm.weight",
        "input_layernorm.weight",
        "pre_mlp_layernorm.weight",
    }:
        return np.ones((2, 2), dtype=np.float32)
    return np.ones((2,), dtype=np.float32)


def _write_synthetic_official_sidecar(path: Path, *, omit_key: str | None = None) -> None:
    tensors: dict[str, np.ndarray] = {}
    for layer_index in range(MIMO_MTP_LAYER_COUNT):
        for suffix in MIMO_MTP_REQUIRED_SUFFIXES:
            key = official_mimo_mtp_key(layer_index, suffix)
            if key == omit_key:
                continue
            tensors[key] = _tiny_tensor_for_suffix(suffix)
    _save_file(tensors, str(path))


def test_missing_sidecar_fails_closed(tmp_path: Path) -> None:
    missing_path = tmp_path / "model_mtp.safetensors"

    probe = probe_mimo_mtp_sidecar(missing_path)

    assert not probe.ready
    assert probe.status == "missing"
    assert probe.path == missing_path
    assert probe.layer_count == 0
    assert probe.tensors == ()
    assert probe.missing_keys == required_mimo_mtp_keys()
    assert probe.error is not None
    assert str(missing_path) in probe.error


def test_synthetic_official_layout_sidecar_passes(tmp_path: Path) -> None:
    sidecar_path = tmp_path / "model_mtp.safetensors"
    _write_synthetic_official_sidecar(sidecar_path)

    probe = probe_mimo_mtp_sidecar(sidecar_path)

    assert probe.ready
    assert probe.status == "ready"
    assert probe.layer_count == 3
    assert probe.missing_keys == ()
    assert probe.error is None
    assert len(probe.tensors) == len(required_mimo_mtp_keys())
    tensor_by_key = {tensor.key: tensor for tensor in probe.tensors}
    sample_key = official_mimo_mtp_key(2, "self_attn.qkv_proj.weight")
    assert tensor_by_key[sample_key].shape == (2, 2)


def test_missing_required_tensor_reports_exact_key(tmp_path: Path) -> None:
    sidecar_path = tmp_path / "model_mtp.safetensors"
    omitted_key = official_mimo_mtp_key(1, "mlp.down_proj.weight_scale_inv")
    _write_synthetic_official_sidecar(sidecar_path, omit_key=omitted_key)

    probe = probe_mimo_mtp_sidecar(sidecar_path)

    assert not probe.ready
    assert probe.status == "invalid"
    assert probe.layer_count == 3
    assert probe.missing_keys == (omitted_key,)
    assert probe.error == "MiMo MTP sidecar is missing required official-layout tensors"
