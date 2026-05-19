#!/usr/bin/env python3
"""Build an experimental MLX affine 6-bit artifact for MiMo MTP weights only."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, TypedDict, cast

import mlx.core as mx
import torch
from safetensors import safe_open

from scripts import mimo_v25_pro_6bit_quantize as base_quantize

SOURCE_CHECKPOINT: Final[Path] = base_quantize.SOURCE_CHECKPOINT
MTP_SOURCE_SHARD: Final[Path] = SOURCE_CHECKPOINT / "model_mtp.safetensors"
PRODUCTION_6BIT_MODEL_ID: Final[str] = "kernelpool/MiMo-V2.5-Pro-6bit"
PRODUCTION_6BIT_ARTIFACT_ROOT: Final[Path] = Path(
    "/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit"
)
MTP_EXPERIMENTAL_MODEL_ID: Final[str] = (
    "kernelpool/MiMo-V2.5-Pro-6bit-mtp-experimental"
)
MTP_OUTPUT_ROOT: Final[Path] = Path(
    "/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/"
    "kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental"
)
MTP_OUTPUT_SHARD_NAME: Final[str] = "model_mtp-00001-of-00001.safetensors"
MTP_MANIFEST_NAME: Final[str] = "mtp_quantization_manifest.json"
DEFAULT_GROUP_SIZE: Final[int] = base_quantize.DEFAULT_GROUP_SIZE
DEFAULT_BITS: Final[int] = base_quantize.DEFAULT_BITS
EXPECTED_MTP_LAYERS: Final[tuple[int, ...]] = (0, 1, 2)
MTP_HEADROOM_BYTES: Final[int] = 8 * 1024**3
ARTIFACT_KIND: Final[str] = "mimo-v25-pro-mtp-only"
SOURCE_MODEL_ID: Final[str] = base_quantize.BASE_MODEL_ID

SourceNbytesFn = Callable[[Sequence[int], str], int]
QuantizeTensorFn = Callable[
    [str, torch.Tensor, torch.Tensor | None, int, int], dict[str, mx.array]
]
TorchToMxFn = Callable[[torch.Tensor], mx.array]
SaveSafetensorsFn = Callable[[str | Path, dict[str, mx.array], dict[str, str] | None], object]

SOURCE_NBYTES = cast(
    SourceNbytesFn,
    base_quantize._source_nbytes,  # pyright: ignore[reportPrivateUsage]
)
QUANTIZE_TENSOR = cast(
    QuantizeTensorFn,
    base_quantize._quantize_tensor,  # pyright: ignore[reportPrivateUsage]
)
TORCH_TO_MX = cast(
    TorchToMxFn,
    base_quantize._torch_to_mx,  # pyright: ignore[reportPrivateUsage]
)
SAVE_SAFETENSORS = cast(SaveSafetensorsFn, mx.save_safetensors)


class TensorManifestEntry(TypedDict):
    source_dtype: str
    output_dtype: str
    shape: list[int]
    layer: int
    source_nbytes: int
    estimated_output_nbytes: int
    quantized: bool
    source_scale_inv: str
    output_keys: list[str]
    reason: str


class ShardManifestEntry(TypedDict):
    source_file: str
    output_file: str
    status: str
    source_size: int
    output_size: int
    started_at: float
    completed_at: float
    tensors: dict[str, TensorManifestEntry]
    error: str


class MtpManifest(TypedDict):
    model_id: str
    artifact_kind: str
    base_model_id: str
    source_model_id: str
    source_shard: str
    output_dir: str
    output_shard: str
    mtp_layers: list[int]
    expected_mtp_layers: list[int]
    missing_expected_layers: list[int]
    complete_expected_layers: bool
    quantization: dict[str, object]
    expected_output_size_bytes: int
    created_at: float
    updated_at: float
    complete: bool
    shards: dict[str, ShardManifestEntry]


class OutputIndexMetadata(TypedDict):
    artifact_kind: str
    total_size: int
    save_format: str
    quantization_format: str
    base_model: str
    mtp_layers: list[int]


class OutputIndex(TypedDict):
    metadata: OutputIndexMetadata
    weight_map: dict[str, str]


@dataclass(frozen=True)
class MtpQuantizationPlan:
    source_shard: Path
    output_dir: Path
    group_size: int = DEFAULT_GROUP_SIZE
    bits: int = DEFAULT_BITS


@dataclass(frozen=True)
class CliArgs:
    source_shard: Path
    output_dir: Path
    dry_run: bool
    convert: bool
    cleanup_incomplete: bool
    summary_json: Path | None


class _SafeSlice(Protocol):
    def get_dtype(self) -> str: ...

    def get_shape(self) -> Sequence[int]: ...


class _SafeOpenHandle(Protocol):
    def keys(self) -> Sequence[str]: ...

    def get_slice(self, name: str) -> _SafeSlice: ...

    def get_tensor(self, name: str) -> torch.Tensor: ...


def _is_mtp_tensor(name: str) -> bool:
    return name.startswith("model.mtp.layers.")


def _mtp_layer(name: str) -> int | None:
    prefix = "model.mtp.layers."
    if not name.startswith(prefix):
        return None
    remainder = name[len(prefix) :]
    layer_text, _, _suffix = remainder.partition(".")
    return int(layer_text) if layer_text.isdigit() else None


def _manifest_path(output_dir: Path) -> Path:
    return output_dir / MTP_MANIFEST_NAME


def _index_path(output_dir: Path) -> Path:
    return output_dir / "model.safetensors.index.json"


def _tmp_output_path(output_dir: Path) -> Path:
    return output_dir / f".{MTP_OUTPUT_SHARD_NAME}.tmp.safetensors"


def _empty_manifest() -> MtpManifest:
    return {
        "model_id": MTP_EXPERIMENTAL_MODEL_ID,
        "artifact_kind": ARTIFACT_KIND,
        "base_model_id": PRODUCTION_6BIT_MODEL_ID,
        "source_model_id": SOURCE_MODEL_ID,
        "source_shard": str(MTP_SOURCE_SHARD),
        "output_dir": str(MTP_OUTPUT_ROOT),
        "output_shard": MTP_OUTPUT_SHARD_NAME,
        "mtp_layers": [],
        "expected_mtp_layers": list(EXPECTED_MTP_LAYERS),
        "missing_expected_layers": list(EXPECTED_MTP_LAYERS),
        "complete_expected_layers": False,
        "quantization": {
            "format": "mlx-affine-6bit-mtp-only",
            "quant_method": "mlx-affine",
            "bits": DEFAULT_BITS,
            "group_size": DEFAULT_GROUP_SIZE,
            "mode": "affine",
        },
        "expected_output_size_bytes": 0,
        "created_at": 0.0,
        "updated_at": 0.0,
        "complete": False,
        "shards": {},
    }


def _json_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _json_str(value: object, default: str) -> str:
    return value if isinstance(value, str) else default


def _json_int_list(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    output: list[int] = []
    for item in cast(list[object], value):
        if isinstance(item, bool):
            output.append(int(item))
        elif isinstance(item, int):
            output.append(item)
        elif isinstance(item, float):
            output.append(int(item))
        elif isinstance(item, str):
            try:
                output.append(int(item))
            except ValueError:
                continue
    return output


def _load_output_index(path: Path) -> OutputIndex:
    if not path.exists():
        return {
            "metadata": {
                "artifact_kind": ARTIFACT_KIND,
                "total_size": 0,
                "save_format": "mlx-affine-6bit",
                "quantization_format": "mlx-affine-6bit-mtp-only",
                "base_model": PRODUCTION_6BIT_MODEL_ID,
                "mtp_layers": [],
            },
            "weight_map": {},
        }

    raw_obj = cast(object, json.loads(path.read_text()))
    if not isinstance(raw_obj, dict):
        raise ValueError(f"Invalid output index {path}")
    raw = cast(dict[str, object], raw_obj)
    metadata_obj = raw.get("metadata", {})
    weight_map_obj = raw.get("weight_map", {})
    if not isinstance(metadata_obj, dict) or not isinstance(weight_map_obj, dict):
        raise ValueError(f"Invalid output index {path}")
    metadata = cast(dict[str, object], metadata_obj)
    weight_map = cast(dict[object, object], weight_map_obj)

    mtp_layers = _json_int_list(metadata.get("mtp_layers", []))
    return {
        "metadata": {
            "artifact_kind": _json_str(
                metadata.get("artifact_kind", ARTIFACT_KIND), ARTIFACT_KIND
            ),
            "total_size": _json_int(metadata.get("total_size", 0), 0),
            "save_format": _json_str(
                metadata.get("save_format", "mlx-affine-6bit"), "mlx-affine-6bit"
            ),
            "quantization_format": _json_str(
                metadata.get("quantization_format", "mlx-affine-6bit-mtp-only"),
                "mlx-affine-6bit-mtp-only",
            ),
            "base_model": _json_str(
                metadata.get("base_model", PRODUCTION_6BIT_MODEL_ID),
                PRODUCTION_6BIT_MODEL_ID,
            ),
            "mtp_layers": mtp_layers,
        },
        "weight_map": {
            str(name): str(shard_name) for name, shard_name in weight_map.items()
        },
    }


def guard_mtp_source_shard(source_shard: Path) -> None:
    resolved = source_shard.resolve()
    expected = MTP_SOURCE_SHARD.resolve()
    if resolved != expected:
        raise ValueError(
            f"Refusing MTP source shard {source_shard}; expected exact MTP source shard "
            f"{MTP_SOURCE_SHARD}"
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"MTP source shard not found: {resolved}")


def guard_mtp_output_path(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    approved_root = MTP_OUTPUT_ROOT.resolve()
    production_root = PRODUCTION_6BIT_ARTIFACT_ROOT.resolve()
    source_root = SOURCE_CHECKPOINT.resolve()

    if resolved != approved_root and approved_root not in resolved.parents:
        raise ValueError(
            f"Refusing output path {output_dir}; must be the experimental MTP output "
            f"root {MTP_OUTPUT_ROOT} or a child"
        )
    if resolved == production_root or production_root in resolved.parents:
        raise ValueError(
            f"Refusing output path {output_dir}; production artifact root "
            f"{PRODUCTION_6BIT_ARTIFACT_ROOT} must remain unchanged"
        )
    if resolved == source_root or source_root in resolved.parents:
        raise ValueError(
            f"Refusing output path {output_dir}; output path must not overlap "
            f"source checkpoint {SOURCE_CHECKPOINT}"
        )


def _mtp_tensor_names(source_shard: Path) -> list[str]:
    with safe_open(str(source_shard), framework="pt") as raw_handle:
        handle = cast(_SafeOpenHandle, cast(object, raw_handle))
        names = list(handle.keys())
        return sorted(name for name in names if _is_mtp_tensor(name))


def load_mtp_manifest(path: Path) -> MtpManifest:
    if not path.exists():
        return _empty_manifest()
    raw_obj = cast(object, json.loads(path.read_text()))
    if not isinstance(raw_obj, dict):
        raise ValueError(f"Invalid MTP manifest {path}")
    manifest = cast(MtpManifest, cast(object, raw_obj))
    return manifest


def _refresh_complete_flag(manifest: MtpManifest) -> None:
    shards = manifest.get("shards", {})
    manifest["complete"] = bool(shards) and all(
        shard.get("status") == "complete" for shard in shards.values()
    )


def write_mtp_manifest(path: Path, manifest: MtpManifest) -> None:
    _refresh_complete_flag(manifest)
    manifest["updated_at"] = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def build_mtp_dry_run_manifest(plan: MtpQuantizationPlan) -> MtpManifest:
    guard_mtp_source_shard(plan.source_shard)
    guard_mtp_output_path(plan.output_dir)

    previous_manifest = load_mtp_manifest(_manifest_path(plan.output_dir))
    tensor_names = _mtp_tensor_names(plan.source_shard)
    tensor_name_set = set(tensor_names)
    tensors: dict[str, TensorManifestEntry] = {}
    layers: set[int] = set()
    total_estimate = 0

    with safe_open(str(plan.source_shard), framework="pt") as raw_handle:
        handle = cast(_SafeOpenHandle, cast(object, raw_handle))
        for tensor_name in tensor_names:
            tensor_slice = handle.get_slice(tensor_name)
            shape = tensor_slice.get_shape()
            dtype = tensor_slice.get_dtype()
            source_nbytes = SOURCE_NBYTES(shape, dtype)
            base_entry = base_quantize.estimate_tensor_entry(
                tensor_name,
                shape,
                dtype,
                source_nbytes,
                tensor_name_set,
                plan.group_size,
                plan.bits,
            )
            layer = _mtp_layer(tensor_name)
            if layer is None:
                raise ValueError(f"Unexpected non-layer MTP tensor name: {tensor_name}")
            layers.add(layer)
            entry: TensorManifestEntry = {
                "source_dtype": str(base_entry.get("source_dtype", dtype)),
                "output_dtype": "",
                "shape": [
                    int(dim)
                    for dim in cast(Sequence[int], base_entry.get("shape", shape))
                ],
                "layer": layer,
                "source_nbytes": int(base_entry.get("source_nbytes", source_nbytes)),
                "estimated_output_nbytes": int(
                    base_entry.get("estimated_output_nbytes", source_nbytes)
                ),
                "quantized": bool(base_entry.get("quantized", False)),
                "source_scale_inv": "",
                "output_keys": [
                    str(key)
                    for key in cast(
                        Sequence[str], base_entry.get("output_keys", [tensor_name])
                    )
                ],
                "reason": str(base_entry.get("reason", "")),
            }
            tensors[tensor_name] = entry
            if not tensor_name.endswith(".weight_scale_inv"):
                total_estimate += entry["estimated_output_nbytes"]

    missing_layers = [layer for layer in EXPECTED_MTP_LAYERS if layer not in layers]
    manifest: MtpManifest = {
        "model_id": MTP_EXPERIMENTAL_MODEL_ID,
        "artifact_kind": ARTIFACT_KIND,
        "base_model_id": PRODUCTION_6BIT_MODEL_ID,
        "source_model_id": SOURCE_MODEL_ID,
        "source_shard": str(plan.source_shard),
        "output_dir": str(plan.output_dir),
        "output_shard": MTP_OUTPUT_SHARD_NAME,
        "mtp_layers": sorted(layers),
        "expected_mtp_layers": list(EXPECTED_MTP_LAYERS),
        "missing_expected_layers": missing_layers,
        "complete_expected_layers": not missing_layers,
        "quantization": {
            "format": "mlx-affine-6bit-mtp-only",
            "quant_method": "mlx-affine",
            "bits": plan.bits,
            "group_size": plan.group_size,
            "mode": "affine",
        },
        "expected_output_size_bytes": total_estimate,
        "created_at": previous_manifest.get("created_at", time.time()),
        "complete": False,
        "shards": {
            MTP_OUTPUT_SHARD_NAME: {
                "source_file": str(plan.source_shard),
                "output_file": MTP_OUTPUT_SHARD_NAME,
                "status": "dry-run",
                "source_size": plan.source_shard.stat().st_size,
                "output_size": 0,
                "started_at": 0.0,
                "completed_at": 0.0,
                "tensors": tensors,
                "error": "",
            }
        },
        "updated_at": previous_manifest["updated_at"],
    }
    return manifest


def check_mtp_free_space(output_dir: Path, expected_output_size: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output_dir).free
    required = expected_output_size + MTP_HEADROOM_BYTES
    if free < required:
        raise RuntimeError(
            f"Insufficient free space at {output_dir}: free={free}, required={required} "
            f"(expected_output={expected_output_size}, headroom={MTP_HEADROOM_BYTES})"
        )


def write_mtp_config(output_dir: Path, manifest: MtpManifest) -> None:
    quantization = manifest["quantization"]
    config = {
        "artifact_kind": ARTIFACT_KIND,
        "model_id": manifest["model_id"],
        "base_model_id": PRODUCTION_6BIT_MODEL_ID,
        "source_model_id": SOURCE_MODEL_ID,
        "source_mtp_shard": manifest["source_shard"],
        "architecture": "MiMoV2ForCausalLM",
        "hidden_size": 6144,
        "num_mtp_layers": 3,
        "mtp_config": {"enabled": True, "num_layers": 3},
        "quantization": quantization,
        "quantization_config": {
            "quant_method": "mlx-affine",
            "group_size": DEFAULT_GROUP_SIZE,
            "bits": DEFAULT_BITS,
            "mode": "affine",
            "base_model": PRODUCTION_6BIT_MODEL_ID,
            "model_id": manifest["model_id"],
        },
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )


def write_mtp_readme(output_dir: Path, manifest: MtpManifest) -> None:
    text = (
        f"# {manifest['model_id']}\n\n"
        "This artifact contains only quantized MTP module weights extracted from "
        "`model_mtp.safetensors`.\n\n"
        "It is not a standalone runtime model and must not be registered as a "
        "normal exo runtime model card yet.\n\n"
        f"Base model artifact: `{PRODUCTION_6BIT_MODEL_ID}`\n"
        f"Source model: `{SOURCE_MODEL_ID}`\n"
        f"Source shard: `{manifest['source_shard']}`\n"
    )
    (output_dir / "README.md").write_text(text)


def write_mtp_output_index(output_dir: Path, manifest: MtpManifest) -> None:
    weight_map: dict[str, str] = {}
    total_size = 0
    shard_name = manifest["output_shard"]
    for tensor_name, tensor_entry in manifest["shards"][shard_name]["tensors"].items():
        if tensor_name.endswith(".weight_scale_inv") and not tensor_entry.get(
            "quantized", False
        ):
            continue
        for output_key in tensor_entry.get("output_keys", [tensor_name]):
            weight_map[output_key] = shard_name
        total_size += int(tensor_entry.get("estimated_output_nbytes", 0))

    index = {
        "metadata": {
            "artifact_kind": ARTIFACT_KIND,
            "total_size": total_size,
            "save_format": "mlx-affine-6bit",
            "quantization_format": "mlx-affine-6bit-mtp-only",
            "base_model": PRODUCTION_6BIT_MODEL_ID,
            "mtp_layers": manifest["mtp_layers"],
        },
        "weight_map": weight_map,
    }
    _index_path(output_dir).write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )


def run_mtp_dry_run(plan: MtpQuantizationPlan) -> MtpManifest:
    if plan.bits != DEFAULT_BITS or plan.group_size != DEFAULT_GROUP_SIZE:
        raise ValueError("This MiMo MTP path is fixed to MLX affine 6-bit group_size=64")
    manifest = build_mtp_dry_run_manifest(plan)
    check_mtp_free_space(plan.output_dir, int(manifest["expected_output_size_bytes"]))
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    write_mtp_manifest(_manifest_path(plan.output_dir), manifest)
    write_mtp_config(plan.output_dir, manifest)
    write_mtp_readme(plan.output_dir, manifest)
    write_mtp_output_index(plan.output_dir, manifest)
    return manifest


def convert_mtp_shard(plan: MtpQuantizationPlan, manifest: MtpManifest) -> None:
    shard_manifest = manifest["shards"][MTP_OUTPUT_SHARD_NAME]
    output_file = plan.output_dir / MTP_OUTPUT_SHARD_NAME
    tmp_file = _tmp_output_path(plan.output_dir)

    if output_file.exists() and shard_manifest.get("status") == "complete":
        return

    shard_manifest["status"] = "running"
    shard_manifest["started_at"] = time.time()
    shard_manifest.pop("error", None)
    write_mtp_manifest(_manifest_path(plan.output_dir), manifest)

    tensor_names = sorted(shard_manifest.get("tensors", {}))
    tensor_name_set = set(tensor_names)
    output_tensors: dict[str, mx.array] = {}

    try:
        with safe_open(str(plan.source_shard), framework="pt") as raw_handle:
            handle = cast(_SafeOpenHandle, cast(object, raw_handle))
            for tensor_name in tensor_names:
                entry = shard_manifest["tensors"][tensor_name]
                if tensor_name.endswith(".weight_scale_inv"):
                    continue
                tensor = handle.get_tensor(tensor_name)
                if entry["quantized"]:
                    scale_name = f"{tensor_name}_scale_inv"
                    scale_tensor: torch.Tensor | None = None
                    if scale_name in tensor_name_set:
                        scale_tensor = handle.get_tensor(scale_name)
                        entry["source_scale_inv"] = scale_name
                    output_tensors.update(
                        QUANTIZE_TENSOR(
                            tensor_name,
                            tensor,
                            scale_tensor,
                            plan.group_size,
                            plan.bits,
                        )
                    )
                    entry["output_dtype"] = "uint32+scales+biases"
                else:
                    output_tensor = TORCH_TO_MX(tensor)
                    output_tensors[tensor_name] = output_tensor
                    entry["output_dtype"] = str(output_tensor.dtype)

        SAVE_SAFETENSORS(str(tmp_file), output_tensors, {"format": "mlx-affine-6bit"})
        tmp_file.replace(output_file)
        shard_manifest["output_size"] = output_file.stat().st_size
        shard_manifest["status"] = "complete"
        shard_manifest["completed_at"] = time.time()
    except Exception as exc:
        shard_manifest["status"] = "failed"
        shard_manifest["error"] = repr(exc)
        if tmp_file.exists():
            tmp_file.unlink()
        raise
    finally:
        write_mtp_manifest(_manifest_path(plan.output_dir), manifest)


def run_mtp_conversion(plan: MtpQuantizationPlan) -> MtpManifest:
    manifest = run_mtp_dry_run(plan)
    convert_mtp_shard(plan, manifest)
    write_mtp_output_index(plan.output_dir, manifest)
    write_mtp_manifest(_manifest_path(plan.output_dir), manifest)
    return manifest


def cleanup_incomplete(output_dir: Path) -> None:
    guard_mtp_output_path(output_dir)
    for pattern in ("*.tmp", "*.tmp.safetensors", ".*.tmp", ".*.tmp.safetensors"):
        for tmp_path in output_dir.glob(pattern):
            tmp_path.unlink()

    manifest_path = _manifest_path(output_dir)
    if manifest_path.exists():
        manifest = load_mtp_manifest(manifest_path)
        for shard in manifest.get("shards", {}).values():
            if shard.get("status") in {"running", "failed"}:
                shard["status"] = "cleanup-required"
        write_mtp_manifest(manifest_path, manifest)


def sha256_file(path: Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_summary(output_dir: Path, summary_json: Path | None = None) -> dict[str, object]:
    manifest = load_mtp_manifest(_manifest_path(output_dir))
    index = _load_output_index(_index_path(output_dir))
    output_shard = manifest["output_shard"]
    output_path = output_dir / output_shard
    summary: dict[str, object] = {
        "model_id": manifest["model_id"],
        "artifact_kind": manifest["artifact_kind"],
        "output_dir": str(output_dir),
        "output_shard": output_shard,
        "output_shard_size": output_path.stat().st_size if output_path.exists() else 0,
        "output_shard_sha256": sha256_file(output_path) if output_path.exists() else None,
        "tensor_count": len(index["weight_map"]),
        "mtp_layers": manifest["mtp_layers"],
        "complete": manifest["complete"],
    }
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _parse_args(argv: Sequence[str]) -> CliArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-shard", type=Path, default=MTP_SOURCE_SHARD)
    parser.add_argument("--output", type=Path, default=MTP_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--convert", action="store_true")
    parser.add_argument("--cleanup-incomplete", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=None)
    namespace = parser.parse_args(argv)

    source_shard = cast(object, namespace.source_shard)
    output_dir = cast(object, namespace.output)
    summary_json = cast(object, namespace.summary_json)
    dry_run = cast(bool, namespace.dry_run)
    convert = cast(bool, namespace.convert)
    cleanup = cast(bool, namespace.cleanup_incomplete)

    if not isinstance(source_shard, Path):
        raise TypeError("--source-shard must be a pathlib.Path")
    if not isinstance(output_dir, Path):
        raise TypeError("--output must be a pathlib.Path")
    if summary_json is not None and not isinstance(summary_json, Path):
        raise TypeError("--summary-json must be a pathlib.Path when provided")

    return CliArgs(
        source_shard=source_shard,
        output_dir=output_dir,
        dry_run=dry_run,
        convert=convert,
        cleanup_incomplete=cleanup,
        summary_json=summary_json,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    plan = MtpQuantizationPlan(
        source_shard=args.source_shard,
        output_dir=args.output_dir,
    )

    if args.cleanup_incomplete:
        cleanup_incomplete(plan.output_dir)
    elif args.convert == args.dry_run:
        raise ValueError("Choose exactly one of --dry-run or --convert")
    elif args.convert:
        run_mtp_conversion(plan)
    else:
        run_mtp_dry_run(plan)

    summary = write_summary(plan.output_dir, args.summary_json)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
