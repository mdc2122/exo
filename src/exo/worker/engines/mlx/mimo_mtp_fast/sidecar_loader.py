from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import mlx.core as mx

from exo.worker.engines.mlx.mimo_mtp_fast.sidecar_contract import (
    MIMO_MTP_LAYER_COUNT,
    MIMO_MTP_REQUIRED_SUFFIXES,
    official_mimo_mtp_key,
    probe_mimo_mtp_sidecar,
)


class MimoMtpSidecarLoadError(RuntimeError):
    """Raised when a MiMo MTP sidecar cannot be loaded safely."""


@dataclass(frozen=True, slots=True)
class MimoMtpLayerTensors:
    layer_index: int
    tensors: Mapping[str, mx.array]

    def get(self, suffix: str) -> mx.array:
        try:
            return self.tensors[suffix]
        except KeyError as exc:
            raise KeyError(
                f"MiMo MTP layer {self.layer_index} missing tensor suffix {suffix!r}"
            ) from exc

    @property
    def eh_proj_weight(self) -> mx.array:
        return self.get("eh_proj.weight")

    @property
    def qkv_proj_weight(self) -> mx.array:
        return self.get("self_attn.qkv_proj.weight")

    @property
    def qkv_proj_scale_inv(self) -> mx.array:
        return self.get("self_attn.qkv_proj.weight.scales")

    @property
    def qkv_proj_biases(self) -> mx.array:
        return self.get("self_attn.qkv_proj.weight.biases")

    @property
    def gate_proj_weight(self) -> mx.array:
        return self.get("mlp.gate_proj.weight")

    @property
    def up_proj_weight(self) -> mx.array:
        return self.get("mlp.up_proj.weight")

    @property
    def down_proj_weight(self) -> mx.array:
        return self.get("mlp.down_proj.weight")

    def quantized_linear_tensors(self, prefix: str) -> tuple[mx.array, mx.array, mx.array]:
        return (
            self.get(f"{prefix}.weight"),
            self.get(f"{prefix}.weight.scales"),
            self.get(f"{prefix}.weight.biases"),
        )


@dataclass(frozen=True, slots=True)
class MimoMtpSidecarTensors:
    path: Path
    layers: tuple[MimoMtpLayerTensors, ...]


def _local_suffix_for_key(layer_index: int, key: str) -> str:
    prefix = f"model.mtp.layers.{layer_index}."
    if not key.startswith(prefix):
        raise ValueError(f"Expected key for layer {layer_index}, got {key!r}")
    return key.removeprefix(prefix)


def _format_missing_keys(missing_keys: tuple[str, ...]) -> str:
    if not missing_keys:
        return "[]"
    visible_keys = ", ".join(missing_keys[:5])
    suffix = "" if len(missing_keys) <= 5 else f", ... ({len(missing_keys)} total)"
    return f"[{visible_keys}{suffix}]"


def load_mimo_mtp_sidecar_tensors(path: str | Path) -> MimoMtpSidecarTensors:
    probe = probe_mimo_mtp_sidecar(path)
    if not probe.ready:
        raise MimoMtpSidecarLoadError(
            "MiMo MTP sidecar is not ready: "
            f"status={probe.status} missing={_format_missing_keys(probe.missing_keys)} "
            f"error={probe.error}"
        )

    sidecar_path = Path(path).expanduser()
    loaded = cast(Mapping[str, mx.array], mx.load(str(sidecar_path)))
    layers: list[MimoMtpLayerTensors] = []
    for layer_index in range(MIMO_MTP_LAYER_COUNT):
        layer_tensors: dict[str, mx.array] = {}
        for suffix in MIMO_MTP_REQUIRED_SUFFIXES:
            official_key = official_mimo_mtp_key(layer_index, suffix)
            tensor = loaded[official_key]
            layer_tensors[_local_suffix_for_key(layer_index, official_key)] = tensor
        layers.append(
            MimoMtpLayerTensors(layer_index=layer_index, tensors=layer_tensors)
        )
    return MimoMtpSidecarTensors(path=sidecar_path, layers=tuple(layers))
