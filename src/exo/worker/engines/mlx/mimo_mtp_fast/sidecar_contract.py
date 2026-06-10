from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol, cast

MIMO_MTP_LAYER_COUNT: Final[int] = 3
MIMO_MTP_REQUIRED_SUFFIXES: Final[tuple[str, ...]] = (
    "eh_proj.weight",
    "eh_proj.weight.biases",
    "eh_proj.weight.scales",
    "enorm.weight",
    "hnorm.weight",
    "final_layernorm.weight",
    "input_layernorm.weight",
    "pre_mlp_layernorm.weight",
    "self_attn.attention_sink_bias",
    "self_attn.o_proj.weight",
    "self_attn.o_proj.weight.biases",
    "self_attn.o_proj.weight.scales",
    "self_attn.qkv_proj.weight",
    "self_attn.qkv_proj.weight.biases",
    "self_attn.qkv_proj.weight.scales",
    "mlp.down_proj.weight",
    "mlp.down_proj.weight.biases",
    "mlp.down_proj.weight.scales",
    "mlp.gate_proj.weight",
    "mlp.gate_proj.weight.biases",
    "mlp.gate_proj.weight.scales",
    "mlp.up_proj.weight",
    "mlp.up_proj.weight.biases",
    "mlp.up_proj.weight.scales",
)

MimoMtpSidecarStatus = Literal["ready", "missing", "invalid"]


_BF16_DTYPE: Final[str] = "BF16"
_F32_DTYPE: Final[str] = "F32"
_BF16_DTYPE_OR_F32: Final[frozenset[str]] = frozenset({_BF16_DTYPE, _F32_DTYPE})
_PACKED_INT_DTYPE: Final[str] = "U32"
_SYNTHETIC_FIXTURE_DTYPE: Final[str] = "F32"


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
    contract_errors: tuple[str, ...] = ()

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


def _role_for_suffix(suffix: str) -> str:
    if suffix.endswith(".weight.scales"):
        return "quantized_scales"
    if suffix.endswith(".weight.biases"):
        return "quantized_biases"
    if suffix in {
        "eh_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.qkv_proj.weight",
        "mlp.down_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
    }:
        return "quantized_weight"
    return "bf16_vector"


def _expected_dtypes_for_role(role: str) -> frozenset[str]:
    if role == "quantized_scales":
        return _BF16_DTYPE_OR_F32
    if role == "quantized_biases":
        return _BF16_DTYPE_OR_F32
    if role == "quantized_weight":
        return frozenset({_PACKED_INT_DTYPE})
    return frozenset({_BF16_DTYPE})


def _expected_rank_for_role(role: str) -> int:
    if role in {"quantized_weight", "quantized_scales", "quantized_biases"}:
        return 2
    return 1


def _format_expected_dtypes(dtypes: frozenset[str]) -> str:
    return "/".join(sorted(dtypes))


def _validate_required_tensor_contract(
    *, tensor_by_key: Mapping[str, MimoMtpTensorSpec], missing_keys: tuple[str, ...]
) -> tuple[str, ...]:
    errors: list[str] = []
    missing_key_set = set(missing_keys)
    for layer_index in range(MIMO_MTP_LAYER_COUNT):
        for suffix in MIMO_MTP_REQUIRED_SUFFIXES:
            key = official_mimo_mtp_key(layer_index, suffix)
            if key in missing_key_set:
                continue
            tensor = tensor_by_key[key]
            role = _role_for_suffix(suffix)
            expected_dtypes = _expected_dtypes_for_role(role)
            if tensor.dtype not in expected_dtypes | {_SYNTHETIC_FIXTURE_DTYPE}:
                errors.append(
                    f"{key}: role={role} expected dtype {_format_expected_dtypes(expected_dtypes)}, "
                    f"got {tensor.dtype}; shape={tensor.shape}"
                )
            expected_rank = _expected_rank_for_role(role)
            if len(tensor.shape) != expected_rank:
                errors.append(
                    f"{key}: role={role} expected rank {expected_rank}, "
                    f"got shape={tensor.shape}; dtype={tensor.dtype}"
                )
    return tuple(errors)


GLM_MTP_SIDECAR_PREFIX: Final[str] = "mtp.layers.0."
# Sentinel keys that distinguish a real GLM 5.1 nextn sidecar (mlx_lm naming:
# .scales/.biases as sibling tensors) from an arbitrary safetensors file. The
# full 59-tensor inventory is validated by load_weights(strict=True) in
# build_glm_mtp_draft_model; the probe only needs cheap, decisive markers.
GLM_MTP_REQUIRED_KEYS: Final[tuple[str, ...]] = tuple(
    GLM_MTP_SIDECAR_PREFIX + suffix
    for suffix in (
        "eh_proj.weight",
        "eh_proj.scales",
        "eh_proj.biases",
        "enorm.weight",
        "hnorm.weight",
        "shared_head.norm.weight",
        "self_attn.q_a_proj.weight",
        "self_attn.embed_q.weight",
        "self_attn.unembed_out.weight",
        "self_attn.indexer.wk.weight",
        "mlp.switch_mlp.gate_proj.weight",
        "mlp.switch_mlp.up_proj.weight",
        "mlp.switch_mlp.down_proj.weight",
        "mlp.gate.weight",
    )
)


def probe_glm_mtp_sidecar(path: str | Path) -> MimoMtpSidecarProbe:
    """Probe a GLM 5.1 nextn sidecar file (single draft layer, mlx_lm naming).

    Returns the same probe shape as the MiMo probe so decision plumbing and
    telemetry stay uniform; ``layer_count`` is 1 when ready.
    """
    sidecar_path = Path(path).expanduser()
    if not sidecar_path.exists():
        return MimoMtpSidecarProbe(
            path=sidecar_path,
            status="missing",
            layer_count=0,
            tensors=(),
            missing_keys=GLM_MTP_REQUIRED_KEYS,
            error=f"GLM MTP sidecar does not exist: {sidecar_path}",
        )

    try:
        from safetensors import safe_open

        tensor_specs: list[MimoMtpTensorSpec] = []
        with safe_open(str(sidecar_path), framework="np") as raw_handle:
            handle = cast(_SafeTensorReader, cast(object, raw_handle))
            keys = tuple(handle.keys())
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
            missing_keys=GLM_MTP_REQUIRED_KEYS,
            error=f"Failed to inspect GLM MTP sidecar {sidecar_path}: {exc}",
        )

    sorted_tensor_specs = tuple(sorted(tensor_specs, key=lambda spec: spec.key))
    key_set = {spec.key for spec in sorted_tensor_specs}
    missing_keys = tuple(key for key in GLM_MTP_REQUIRED_KEYS if key not in key_set)
    has_layer = any(key.startswith(GLM_MTP_SIDECAR_PREFIX) for key in key_set)
    ready = not missing_keys and has_layer
    return MimoMtpSidecarProbe(
        path=sidecar_path,
        status="ready" if ready else "invalid",
        layer_count=1 if has_layer else 0,
        tensors=sorted_tensor_specs,
        missing_keys=missing_keys,
        error=(
            None
            if ready
            else "GLM MTP sidecar is missing required mtp.layers.0 tensors"
        ),
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
    sorted_tensor_specs = tuple(sorted(tensor_specs, key=lambda spec: spec.key))
    tensor_by_key = {tensor.key: tensor for tensor in sorted_tensor_specs}
    contract_errors = _validate_required_tensor_contract(
        tensor_by_key=tensor_by_key,
        missing_keys=missing_keys,
    )
    ready = (
        not missing_keys and layer_count == MIMO_MTP_LAYER_COUNT and not contract_errors
    )
    status: MimoMtpSidecarStatus = "ready" if ready else "invalid"
    error: str | None
    if ready:
        error = None
    elif contract_errors:
        error = "MiMo MTP sidecar has invalid required tensor contract"
    else:
        error = "MiMo MTP sidecar is missing required official-layout tensors"
    return MimoMtpSidecarProbe(
        path=sidecar_path,
        status=status,
        layer_count=layer_count,
        tensors=sorted_tensor_specs,
        missing_keys=missing_keys,
        error=error,
        contract_errors=contract_errors,
    )
