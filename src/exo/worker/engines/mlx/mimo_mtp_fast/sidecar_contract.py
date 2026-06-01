from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol, cast

MIMO_MTP_LAYER_COUNT: Final[int] = 3
MIMO_MTP_REQUIRED_SUFFIXES: Final[tuple[str, ...]] = (
    "eh_proj.weight",
    "enorm.weight",
    "hnorm.weight",
    "final_layernorm.weight",
    "input_layernorm.weight",
    "pre_mlp_layernorm.weight",
    "self_attn.attention_sink_bias",
    "self_attn.o_proj.weight",
    "self_attn.qkv_proj.weight",
    "self_attn.qkv_proj.weight_scale_inv",
    "mlp.down_proj.weight",
    "mlp.down_proj.weight_scale_inv",
    "mlp.gate_proj.weight",
    "mlp.gate_proj.weight_scale_inv",
    "mlp.up_proj.weight",
    "mlp.up_proj.weight_scale_inv",
)

MimoMtpSidecarStatus = Literal["ready", "missing", "invalid"]


class _SafeTensorSlice(Protocol):
    def get_dtype(self) -> str: ...

    def get_shape(self) -> Sequence[int]: ...


class _SafeTensorReader(Protocol):
    def keys(self) -> list[str]: ...

    def get_slice(self, name: str) -> _SafeTensorSlice: ...


@dataclass(frozen=True, slots=True)
class MimoMtpTensorSpec:
    key: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MimoMtpSidecarProbe:
    path: Path
    status: MimoMtpSidecarStatus
    layer_count: int
    tensors: tuple[MimoMtpTensorSpec, ...]
    missing_keys: tuple[str, ...]
    error: str | None

    @property
    def ready(self) -> bool:
        return self.status == "ready"


def official_mimo_mtp_key(layer_index: int, suffix: str) -> str:
    return f"model.mtp.layers.{layer_index}.{suffix}"


def required_mimo_mtp_keys() -> tuple[str, ...]:
    return tuple(
        official_mimo_mtp_key(layer_index, suffix)
        for layer_index in range(MIMO_MTP_LAYER_COUNT)
        for suffix in MIMO_MTP_REQUIRED_SUFFIXES
    )


def probe_mimo_mtp_sidecar(path: str | Path) -> MimoMtpSidecarProbe:
    sidecar_path = Path(path).expanduser()
    required_keys = required_mimo_mtp_keys()
    if not sidecar_path.exists():
        return MimoMtpSidecarProbe(
            path=sidecar_path,
            status="missing",
            layer_count=0,
            tensors=(),
            missing_keys=required_keys,
            error=f"MiMo MTP sidecar does not exist: {sidecar_path}",
        )

    try:
        from safetensors import safe_open

        tensor_specs: list[MimoMtpTensorSpec] = []
        with safe_open(str(sidecar_path), framework="np") as raw_handle:
            handle = cast(_SafeTensorReader, cast(object, raw_handle))
            keys = tuple(handle.keys())
            key_set = set(keys)
            for key in keys:
                tensor_slice = handle.get_slice(key)
                tensor_specs.append(
                    MimoMtpTensorSpec(
                        key=key,
                        dtype=tensor_slice.get_dtype(),
                        shape=tuple(int(dim) for dim in tensor_slice.get_shape()),
                    )
                )
    except Exception as exc:
        return MimoMtpSidecarProbe(
            path=sidecar_path,
            status="invalid",
            layer_count=0,
            tensors=(),
            missing_keys=required_keys,
            error=f"Failed to inspect MiMo MTP sidecar {sidecar_path}: {exc}",
        )

    missing_keys = tuple(key for key in required_keys if key not in key_set)
    layer_count = sum(
        1
        for layer_index in range(MIMO_MTP_LAYER_COUNT)
        if any(key.startswith(f"model.mtp.layers.{layer_index}.") for key in key_set)
    )
    status: MimoMtpSidecarStatus = (
        "ready"
        if not missing_keys and layer_count == MIMO_MTP_LAYER_COUNT
        else "invalid"
    )
    return MimoMtpSidecarProbe(
        path=sidecar_path,
        status=status,
        layer_count=layer_count,
        tensors=tuple(sorted(tensor_specs, key=lambda spec: spec.key)),
        missing_keys=missing_keys,
        error=None
        if status == "ready"
        else "MiMo MTP sidecar is missing required official-layout tensors",
    )
