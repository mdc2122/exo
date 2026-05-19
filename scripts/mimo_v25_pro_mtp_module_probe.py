#!/usr/bin/env python3
"""Probe the standalone MiMo V2.5 Pro experimental MTP MLX artifact."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, TypedDict, cast

import mlx.core as mx

PRODUCTION_6BIT_ARTIFACT_ROOT: Final[Path] = Path(
    "/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit"
)
MTP_ARTIFACT_ROOT: Final[Path] = Path(
    "/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/"
    "kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental"
)
MTP_OUTPUT_SHARD_NAME: Final[str] = "model_mtp-00001-of-00001.safetensors"
DEFAULT_LAYERS: Final[tuple[int, ...]] = (0, 1, 2)
DEFAULT_HIDDEN_SIZE: Final[int] = 6144
DEFAULT_EH_INPUT_SIZE: Final[int] = 12288
DEFAULT_QKV_OUTPUT_SIZE: Final[int] = 27136
ARTIFACT_KIND: Final[str] = "mimo-v25-pro-mtp-only"
INDEX_NAME: Final[str] = "model.safetensors.index.json"


class MtpProbeReport(TypedDict):
    artifact_dir: str
    artifact_kind: str
    layers: list[int]
    loaded_tensor_count: int
    required_key_count: int
    synthetic_forwards: dict[str, list[int]]


@dataclass(frozen=True)
class CliArgs:
    artifact: Path
    json_out: Path | None


class _MxLoadFn(Protocol):
    def __call__(self, file: str | Path) -> dict[str, mx.array]: ...


class _MxEvalFn(Protocol):
    def __call__(self, *arrays: object) -> object: ...


class _MxQuantizedMatmulFn(Protocol):
    def __call__(
        self,
        input_tensor: mx.array,
        weight: mx.array,
        scales: mx.array,
        biases: mx.array,
        *,
        transpose: bool,
        group_size: int,
        bits: int,
        mode: str,
    ) -> mx.array: ...


MX_LOAD = cast(_MxLoadFn, mx.load)
MX_EVAL = cast(_MxEvalFn, mx.eval)
MX_QUANTIZED_MATMUL = cast(_MxQuantizedMatmulFn, mx.quantized_matmul)


def _is_within(resolved_path: Path, resolved_root: Path) -> bool:
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def guard_probe_artifact_path(artifact_dir: Path) -> None:
    resolved = artifact_dir.resolve()
    production_root = PRODUCTION_6BIT_ARTIFACT_ROOT.resolve()
    experimental_root = MTP_ARTIFACT_ROOT.resolve()

    if ".." in artifact_dir.parts:
        raise ValueError(
            f"Refusing artifact path {artifact_dir}; relative traversal with '..' is "
            "not allowed"
        )
    if _is_within(resolved, production_root):
        raise ValueError(
            f"Refusing artifact path {artifact_dir}; use the experimental MTP artifact "
            f"root {MTP_ARTIFACT_ROOT}, not the production 6-bit artifact tree"
        )
    if artifact_dir.is_absolute() and not _is_within(resolved, experimental_root):
        raise ValueError(
            f"Refusing artifact path {artifact_dir}; absolute paths must stay under "
            f"the experimental MTP artifact root {MTP_ARTIFACT_ROOT}"
        )


def load_index(artifact_dir: Path) -> dict[str, object]:
    index_path = artifact_dir / INDEX_NAME
    raw_obj = cast(object, json.loads(index_path.read_text()))
    if not isinstance(raw_obj, dict):
        raise ValueError(f"Invalid safetensors index: {index_path}")
    raw = cast(dict[object, object], raw_obj)
    return {str(key): value for key, value in raw.items()}


def required_mtp_keys(layers: Sequence[int]) -> list[str]:
    required: list[str] = []
    for layer in layers:
        prefix = f"model.mtp.layers.{layer}"
        required.extend(
            [
                f"{prefix}.eh_proj.weight",
                f"{prefix}.eh_proj.weight.scales",
                f"{prefix}.eh_proj.weight.biases",
                f"{prefix}.self_attn.qkv_proj.weight",
                f"{prefix}.self_attn.qkv_proj.weight.scales",
                f"{prefix}.self_attn.qkv_proj.weight.biases",
                f"{prefix}.enorm.weight",
            ]
        )
    return required


def _weight_map(index: dict[str, object], artifact_dir: Path) -> dict[str, str]:
    weight_map_obj = index.get("weight_map")
    if not isinstance(weight_map_obj, dict):
        raise ValueError(f"Invalid safetensors index: {artifact_dir / INDEX_NAME}")
    weight_map = cast(dict[object, object], weight_map_obj)
    return {str(key): str(value) for key, value in weight_map.items()}


def _artifact_kind(index: dict[str, object], artifact_dir: Path) -> str:
    metadata_obj = index.get("metadata")
    if not isinstance(metadata_obj, dict):
        raise ValueError(f"Invalid safetensors index: {artifact_dir / INDEX_NAME}")
    metadata = cast(dict[str, object], metadata_obj)
    artifact_kind_obj = metadata.get("artifact_kind")
    if not isinstance(artifact_kind_obj, str):
        raise ValueError(f"Missing artifact_kind in {artifact_dir / INDEX_NAME}")
    return artifact_kind_obj


def _validate_required_keys(weight_map: dict[str, str], layers: Sequence[int]) -> None:
    missing = sorted(
        key for key in required_mtp_keys(layers) if key not in weight_map
    )
    if missing:
        raise ValueError(
            "missing required MTP keys: " + ", ".join(missing)
        )


def _quantized_matmul(
    input_tensor: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
) -> mx.array:
    return MX_QUANTIZED_MATMUL(
        input_tensor,
        weight,
        scales,
        biases,
        transpose=True,
        group_size=64,
        bits=6,
        mode="affine",
    )


def _shape_list(array: mx.array) -> list[int]:
    return [int(dimension) for dimension in array.shape]


def probe_mtp_artifact(
    artifact_dir: Path,
    *,
    layers: Sequence[int] = DEFAULT_LAYERS,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    eh_input_size: int = DEFAULT_EH_INPUT_SIZE,
    qkv_input_size: int | None = None,
) -> MtpProbeReport:
    guard_probe_artifact_path(artifact_dir)
    index = load_index(artifact_dir)
    artifact_kind = _artifact_kind(index, artifact_dir)
    if artifact_kind != ARTIFACT_KIND:
        raise ValueError(
            f"Expected artifact_kind {ARTIFACT_KIND}, got {artifact_kind}"
        )

    layer_list = [int(layer) for layer in layers]
    weight_map = _weight_map(index, artifact_dir)
    _validate_required_keys(weight_map, layer_list)

    shard_path = artifact_dir / MTP_OUTPUT_SHARD_NAME
    loaded_tensors = MX_LOAD(str(shard_path))
    probe_layer = layer_list[0]
    qkv_input_width = hidden_size if qkv_input_size is None else qkv_input_size

    eh_output = _quantized_matmul(
        mx.ones((1, 1, eh_input_size)),
        loaded_tensors[f"model.mtp.layers.{probe_layer}.eh_proj.weight"],
        loaded_tensors[f"model.mtp.layers.{probe_layer}.eh_proj.weight.scales"],
        loaded_tensors[f"model.mtp.layers.{probe_layer}.eh_proj.weight.biases"],
    )
    qkv_output = _quantized_matmul(
        mx.ones((1, 1, qkv_input_width)),
        loaded_tensors[f"model.mtp.layers.{probe_layer}.self_attn.qkv_proj.weight"],
        loaded_tensors[
            f"model.mtp.layers.{probe_layer}.self_attn.qkv_proj.weight.scales"
        ],
        loaded_tensors[
            f"model.mtp.layers.{probe_layer}.self_attn.qkv_proj.weight.biases"
        ],
    )
    MX_EVAL(eh_output, qkv_output)

    synthetic_forwards = {
        f"layer_{probe_layer}_eh_proj": _shape_list(eh_output),
        f"layer_{probe_layer}_qkv_proj": _shape_list(qkv_output),
    }
    return {
        "artifact_dir": str(artifact_dir),
        "artifact_kind": artifact_kind,
        "layers": layer_list,
        "loaded_tensor_count": len(loaded_tensors),
        "required_key_count": len(required_mtp_keys(layer_list)),
        "synthetic_forwards": synthetic_forwards,
    }


def _parse_args(argv: Sequence[str]) -> CliArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, default=None)
    namespace = parser.parse_args(argv)
    artifact = cast(object, namespace.artifact)
    json_out = cast(object, namespace.json_out)

    if not isinstance(artifact, Path):
        raise TypeError("--artifact must be a pathlib.Path")
    if json_out is not None and not isinstance(json_out, Path):
        raise TypeError("--json-out must be a pathlib.Path when provided")

    return CliArgs(artifact=artifact, json_out=json_out)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    report = probe_mtp_artifact(args.artifact)
    text = json.dumps(cast(object, report), indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
