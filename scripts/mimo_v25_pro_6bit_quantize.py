#!/usr/bin/env python3
"""Create a custom MLX affine 6-bit MiMo V2.5-Pro artifact shard by shard.

This script is intentionally narrow: it only accepts the official local
XiaomiMiMo/MiMo-V2.5-Pro source checkpoint and only writes below the approved
MiMo 6-bit quantized artifact root. It never rewrites source files.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence, TypedDict, cast

import mlx.core as mx
import torch
from safetensors import safe_open

SOURCE_CHECKPOINT: Final[Path] = Path(
    "/Volumes/GLM5-NVMe/exo/mimo-v25-pro/hf/XiaomiMiMo--MiMo-V2.5-Pro"
)
OUTPUT_ROOT: Final[Path] = Path(
    "/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/MiMo-V2.5-Pro-6bit"
)
CUSTOM_MODEL_ID: Final[str] = "XiaomiMiMo/MiMo-V2.5-Pro-6bit-MLX"
BASE_MODEL_ID: Final[str] = "XiaomiMiMo/MiMo-V2.5-Pro"
DEFAULT_GROUP_SIZE: Final[int] = 64
DEFAULT_BITS: Final[int] = 6
FP8_BLOCK_SIZE: Final[int] = 128
HEADROOM_BYTES: Final[int] = 250 * 1024**3
COPIED_FILE_NAMES: Final[frozenset[str]] = frozenset(
    {
        ".gitattributes",
        "README.md",
        "added_tokens.json",
        "config.json",
        "configuration_mimo_v2.py",
        "merges.txt",
        "modeling_mimo_v2.py",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
)


class IndexMetadata(TypedDict, total=False):
    total_size: int
    save_format: str
    tp_size: int
    quantization_format: str
    base_model: str


class SafetensorsIndex(TypedDict):
    metadata: IndexMetadata
    weight_map: dict[str, str]


class TensorManifestEntry(TypedDict, total=False):
    source_dtype: str
    output_dtype: str
    shape: list[int]
    source_nbytes: int
    estimated_output_nbytes: int
    quantized: bool
    source_scale_inv: str
    output_keys: list[str]
    reason: str


class ShardManifestEntry(TypedDict, total=False):
    source_file: str
    output_file: str
    status: str
    source_size: int
    output_size: int
    started_at: float
    completed_at: float
    tensors: dict[str, TensorManifestEntry]
    error: str


class RunManifest(TypedDict, total=False):
    model_id: str
    base_model_id: str
    source_checkpoint: str
    output_dir: str
    quantization: dict[str, object]
    source_index_metadata: IndexMetadata
    expected_output_size_bytes: int
    created_at: float
    updated_at: float
    shards: dict[str, ShardManifestEntry]


@dataclass(frozen=True)
class QuantizationPlan:
    source_dir: Path
    output_dir: Path
    group_size: int = DEFAULT_GROUP_SIZE
    bits: int = DEFAULT_BITS
    include_shards: tuple[str, ...] = ()
    include_tensors: tuple[str, ...] = ()


def guard_source_path(source_dir: Path) -> None:
    resolved = source_dir.resolve()
    if resolved != SOURCE_CHECKPOINT.resolve():
        raise ValueError(
            f"Refusing source path {source_dir}; expected exact official checkpoint "
            f"{SOURCE_CHECKPOINT}"
        )
    if not (resolved / "config.json").is_file():
        raise FileNotFoundError(f"Missing source config.json under {resolved}")
    if not (resolved / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"Missing source safetensors index under {resolved}")


def guard_output_path(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    root = OUTPUT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(
            f"Refusing output path {output_dir}; must be {OUTPUT_ROOT} or child"
        )
    if (
        resolved == SOURCE_CHECKPOINT.resolve()
        or SOURCE_CHECKPOINT.resolve() in resolved.parents
    ):
        raise ValueError("Output path must not overlap the official source checkpoint")


def load_index(source_dir: Path) -> SafetensorsIndex:
    index_path = source_dir / "model.safetensors.index.json"
    raw = json.loads(index_path.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("weight_map"), dict):
        raise ValueError(f"Invalid safetensors index: {index_path}")
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "metadata": cast(IndexMetadata, metadata),
        "weight_map": {str(k): str(v) for k, v in raw["weight_map"].items()},
    }


def load_manifest(path: Path) -> RunManifest:
    if not path.exists():
        return {"shards": {}}
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid manifest {path}")
    raw.setdefault("shards", {})
    return cast(RunManifest, raw)


def write_manifest(path: Path, manifest: RunManifest) -> None:
    manifest["updated_at"] = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def source_shards(index: SafetensorsIndex) -> list[str]:
    return sorted(set(index["weight_map"].values()))


def shard_weight_names(index: SafetensorsIndex, shard_name: str) -> list[str]:
    return sorted(k for k, v in index["weight_map"].items() if v == shard_name)


def _matches_any(value: str, patterns: Sequence[str]) -> bool:
    return not patterns or any(fnmatch.fnmatch(value, pattern) for pattern in patterns)


def selected_shards(
    index: SafetensorsIndex, include_shards: Sequence[str]
) -> list[str]:
    return [name for name in source_shards(index) if _matches_any(name, include_shards)]


def sha256_file(path: Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def quantized_payload_nbytes(shape: Sequence[int], group_size: int, bits: int) -> int:
    if len(shape) < 2 or shape[-1] % group_size != 0:
        return 0
    rows = 1
    for dim in shape[:-1]:
        rows *= dim
    packed_cols = shape[-1] * bits // 32
    groups = shape[-1] // group_size
    weight_bytes = rows * packed_cols * 4
    scale_bytes = rows * groups * 4
    bias_bytes = rows * groups * 4
    return weight_bytes + scale_bytes + bias_bytes


def should_quantize_tensor(
    tensor_name: str,
    shape: Sequence[int],
    dtype: str,
    names_in_shard: set[str],
    group_size: int,
) -> tuple[bool, str]:
    if tensor_name.endswith(".weight_scale_inv"):
        return (
            False,
            "source fp8 scale tensor is consumed when quantizing its paired weight",
        )
    if len(shape) < 2:
        return False, "rank is below 2"
    if shape[-1] % group_size != 0:
        return False, f"last dimension is not divisible by group_size={group_size}"
    if dtype == "F8_E4M3":
        scale_name = f"{tensor_name}_scale_inv"
        if scale_name not in names_in_shard:
            return False, "fp8 tensor has no same-shard weight_scale_inv"
        return True, "fp8 e4m3 tensor with paired scale_inv"
    if dtype in {"BF16", "F16", "F32"} and tensor_name.endswith(".weight"):
        return True, "floating point matrix weight"
    return False, f"dtype {dtype} is copied without weight quantization"


def estimate_tensor_entry(
    tensor_name: str,
    shape: Sequence[int],
    dtype: str,
    source_nbytes: int,
    names_in_shard: set[str],
    group_size: int,
    bits: int,
) -> TensorManifestEntry:
    quantized, reason = should_quantize_tensor(
        tensor_name, shape, dtype, names_in_shard, group_size
    )
    output_keys = [tensor_name]
    estimated = source_nbytes
    if quantized:
        output_keys = [tensor_name, f"{tensor_name}.scales", f"{tensor_name}.biases"]
        estimated = quantized_payload_nbytes(shape, group_size, bits)
    return {
        "source_dtype": dtype,
        "shape": list(shape),
        "source_nbytes": source_nbytes,
        "estimated_output_nbytes": estimated,
        "quantized": quantized,
        "output_keys": output_keys,
        "reason": reason,
    }


def build_dry_run_manifest(plan: QuantizationPlan) -> RunManifest:
    guard_source_path(plan.source_dir)
    guard_output_path(plan.output_dir)
    index = load_index(plan.source_dir)
    previous_manifest = load_manifest(plan.output_dir / "quantization_manifest.json")
    manifest: RunManifest = {
        "model_id": CUSTOM_MODEL_ID,
        "base_model_id": BASE_MODEL_ID,
        "source_checkpoint": str(plan.source_dir),
        "output_dir": str(plan.output_dir),
        "quantization": {
            "format": "mlx-affine-6bit-sharded",
            "bits": plan.bits,
            "group_size": plan.group_size,
            "mode": "affine",
            "source_fp8_dequantization": "torch float8_e4m3fn -> fp32, multiplied by paired *_scale_inv over 128x128 source blocks",
        },
        "source_index_metadata": index["metadata"],
        "created_at": previous_manifest.get("created_at", time.time()),
        "shards": {},
    }
    total_estimate = 0
    for shard_name in selected_shards(index, plan.include_shards):
        previous_shard = previous_manifest.get("shards", {}).get(shard_name)
        if (
            previous_shard is not None
            and previous_shard.get("status") == "complete"
            and (plan.output_dir / shard_name).is_file()
        ):
            manifest["shards"][shard_name] = previous_shard
            for tensor_entry in previous_shard.get("tensors", {}).values():
                total_estimate += int(tensor_entry.get("estimated_output_nbytes", 0))
            continue

        names = shard_weight_names(index, shard_name)
        names_set = set(names)
        tensors: dict[str, TensorManifestEntry] = {}
        source_file = plan.source_dir / shard_name
        with safe_open(source_file, framework="pt", device="cpu") as handle:
            for tensor_name in names:
                if not _matches_any(tensor_name, plan.include_tensors):
                    continue
                tensor_slice = handle.get_slice(tensor_name)
                shape = tensor_slice.get_shape()
                dtype = tensor_slice.get_dtype()
                source_nbytes = _source_nbytes(shape, dtype)
                entry = estimate_tensor_entry(
                    tensor_name,
                    shape,
                    dtype,
                    source_nbytes,
                    names_set,
                    plan.group_size,
                    plan.bits,
                )
                tensors[tensor_name] = entry
                if not tensor_name.endswith(".weight_scale_inv"):
                    total_estimate += entry["estimated_output_nbytes"]
        manifest["shards"][shard_name] = {
            "source_file": str(source_file),
            "output_file": shard_name,
            "status": "dry-run",
            "source_size": source_file.stat().st_size,
            "tensors": tensors,
        }
    manifest["expected_output_size_bytes"] = total_estimate
    return manifest


def _source_nbytes(shape: Sequence[int], dtype: str) -> int:
    element_count = 1
    for dim in shape:
        element_count *= dim
    bytes_per_element = {
        "F8_E4M3": 1,
        "BF16": 2,
        "F16": 2,
        "F32": 4,
        "I64": 8,
        "I32": 4,
        "U8": 1,
    }.get(dtype, 0)
    return element_count * bytes_per_element


def check_free_space(
    output_dir: Path, expected_output_size: int, *, require_headroom: bool
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output_dir).free
    required = expected_output_size + (HEADROOM_BYTES if require_headroom else 0)
    if free < required:
        raise RuntimeError(
            f"Insufficient free space at {output_dir}: free={free}, "
            f"required={required} (expected_output={expected_output_size}, "
            f"headroom={HEADROOM_BYTES if require_headroom else 0})"
        )


def estimate_remaining_output_bytes(manifest: RunManifest) -> int:
    """Return estimated bytes still to be written for incomplete shards.

    The total model estimate includes already completed/resumable shards. A
    resumed conversion should require free space for the remaining shards plus
    headroom, not the full model plus headroom again.
    """

    remaining = 0
    for shard in manifest.get("shards", {}).values():
        if shard.get("status") == "complete":
            continue
        for tensor_entry in shard.get("tensors", {}).values():
            remaining += int(tensor_entry.get("estimated_output_nbytes", 0))
    return remaining


def write_config(source_dir: Path, output_dir: Path, manifest: RunManifest) -> None:
    config = json.loads((source_dir / "config.json").read_text())
    config.pop("_name_or_path", None)
    config["quantization"] = {
        "group_size": DEFAULT_GROUP_SIZE,
        "bits": DEFAULT_BITS,
        "mode": "affine",
    }
    config["quantization_config"] = {
        "quant_method": "mlx-affine",
        "group_size": DEFAULT_GROUP_SIZE,
        "bits": DEFAULT_BITS,
        "mode": "affine",
        "base_model": BASE_MODEL_ID,
        "custom_model_id": CUSTOM_MODEL_ID,
    }
    config["_custom_exo_quantization"] = manifest["quantization"]
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )


def copy_support_files(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.iterdir():
        if (
            path.is_file()
            and path.name in COPIED_FILE_NAMES
            and path.name != "config.json"
        ):
            shutil.copy2(path, output_dir / path.name)
        elif path.is_dir() and path.name == "assets":
            target = output_dir / path.name
            if not target.exists():
                shutil.copytree(path, target)


def write_model_card_readme(output_dir: Path, manifest: RunManifest) -> None:
    readme = output_dir / "README.md"
    existing = readme.read_text() if readme.exists() else ""
    header = (
        "\n\n---\n"
        "# exo custom MLX 6-bit artifact\n\n"
        f"Model id: `{CUSTOM_MODEL_ID}`\n\n"
        f"Base model: `{BASE_MODEL_ID}`\n\n"
        "Quantization format: MLX affine packed 6-bit weights with per-group "
        f"scales/biases (`bits={DEFAULT_BITS}`, `group_size={DEFAULT_GROUP_SIZE}`, `mode=affine`).\n\n"
        "Generated by `scripts/mimo_v25_pro_6bit_quantize.py`. See "
        "`quantization_manifest.json` for source/output shard details.\n"
    )
    if "# exo custom MLX 6-bit artifact" not in existing:
        readme.write_text(existing + header)


def write_output_index(output_dir: Path, manifest: RunManifest) -> None:
    weight_map: dict[str, str] = {}
    total_size = 0
    for shard_name, shard in manifest["shards"].items():
        if shard.get("status") not in {"complete", "dry-run"}:
            continue
        for tensor_name, tensor_entry in shard.get("tensors", {}).items():
            if (
                tensor_name.endswith(".weight_scale_inv")
                and tensor_entry.get("quantized") is False
            ):
                continue
            for output_key in tensor_entry.get("output_keys", [tensor_name]):
                weight_map[output_key] = shard_name
            total_size += int(tensor_entry.get("estimated_output_nbytes", 0))
    index = {
        "metadata": {
            "total_size": total_size,
            "save_format": "mlx-affine-6bit",
            "quantization_format": "mlx-affine-6bit-sharded",
            "base_model": BASE_MODEL_ID,
        },
        "weight_map": weight_map,
    }
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError(f"denominator must be positive, got {denominator}")
    return -(-numerator // denominator)


def _broadcast_scale_inv(scale_inv: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    if len(shape) != 2:
        raise ValueError(
            f"FP8 dequantization currently expects rank-2 weights, got {shape}"
        )
    if scale_inv.ndim != 2:
        raise ValueError(f"FP8 scale_inv must be rank-2, got {tuple(scale_inv.shape)}")
    rows, cols = int(shape[0]), int(shape[1])
    scale_rows, scale_cols = int(scale_inv.shape[0]), int(scale_inv.shape[1])
    if rows <= 0 or cols <= 0:
        raise ValueError(f"FP8 target shape must be positive, got {shape}")
    if scale_rows <= 0 or scale_cols <= 0:
        raise ValueError(
            f"FP8 scale_inv shape must be positive, got {tuple(scale_inv.shape)}"
        )

    required_scale_rows = _ceil_div(rows, FP8_BLOCK_SIZE)
    required_scale_cols = _ceil_div(cols, FP8_BLOCK_SIZE)
    if scale_rows < required_scale_rows or scale_cols < required_scale_cols:
        raise ValueError(
            "FP8 scale_inv does not cover target shape with 128x128 source blocks: "
            f"scale_inv={tuple(scale_inv.shape)} target={tuple(shape)} "
            f"required_scale_shape={(required_scale_rows, required_scale_cols)}"
        )

    if scale_rows > required_scale_rows:
        # Scales laid out per row-group (e.g. the MiMo fused qkv: one scale
        # grid per KV-head group, whose 3392-row stride is not a multiple of
        # 128). A top-aligned uniform expansion mis-scales every row past the
        # first group boundary, corrupting all K/V projections.
        n_groups = next(
            (
                g
                for g in range(2, scale_rows + 1)
                if rows % g == 0
                and g * _ceil_div(rows // g, FP8_BLOCK_SIZE) == scale_rows
            ),
            None,
        )
        if n_groups is None:
            raise ValueError(
                "FP8 scale_inv rows do not match a uniform or grouped "
                f"128-block layout: scale_inv={tuple(scale_inv.shape)} "
                f"target={tuple(shape)}"
            )
        rows_per_group = rows // n_groups
        scale_rows_per_group = scale_rows // n_groups
        expanded_rows = torch.cat(
            [
                scale_inv[
                    g * scale_rows_per_group : (g + 1) * scale_rows_per_group
                ].repeat_interleave(FP8_BLOCK_SIZE, dim=0)[:rows_per_group]
                for g in range(n_groups)
            ],
            dim=0,
        )
    else:
        expanded_rows = scale_inv.repeat_interleave(FP8_BLOCK_SIZE, dim=0)[:rows]
    expanded = expanded_rows.repeat_interleave(FP8_BLOCK_SIZE, dim=1)
    return expanded[:rows, :cols]


def _torch_to_mx(tensor: torch.Tensor) -> mx.array:
    if tensor.dtype is torch.bfloat16:
        return mx.array(tensor.float().numpy(), dtype=mx.bfloat16)
    return mx.array(tensor.cpu().numpy())


def _quantize_tensor(
    tensor_name: str,
    tensor: torch.Tensor,
    scale_inv: torch.Tensor | None,
    group_size: int,
    bits: int,
) -> dict[str, mx.array]:
    if tensor.dtype is torch.float8_e4m3fn:
        if scale_inv is None:
            raise ValueError(f"Missing scale_inv for fp8 tensor {tensor_name}")
        dequantized = tensor.float() * _broadcast_scale_inv(
            scale_inv.float(), tensor.shape
        )
        source = mx.array(dequantized.numpy(), dtype=mx.float32)
    else:
        source = _torch_to_mx(tensor)
    q_weight, q_scales, q_biases = mx.quantize(
        source, group_size=group_size, bits=bits, mode="affine"
    )
    mx.eval(q_weight, q_scales, q_biases)
    return {
        tensor_name: q_weight,
        f"{tensor_name}.scales": q_scales,
        f"{tensor_name}.biases": q_biases,
    }


def convert_shard(
    plan: QuantizationPlan, shard_name: str, manifest: RunManifest
) -> None:
    source_file = plan.source_dir / shard_name
    output_file = plan.output_dir / shard_name
    tmp_file = plan.output_dir / f".{shard_name}.tmp.safetensors"
    shard_manifest = manifest["shards"][shard_name]
    if output_file.exists() and shard_manifest.get("status") == "complete":
        return
    shard_manifest["status"] = "running"
    shard_manifest["started_at"] = time.time()
    write_manifest(plan.output_dir / "quantization_manifest.json", manifest)

    names_to_convert = [
        name
        for name in shard_manifest.get("tensors", {})
        if _matches_any(name, plan.include_tensors)
    ]
    names_set = set(names_to_convert)
    output_tensors: dict[str, mx.array] = {}
    try:
        with safe_open(source_file, framework="pt", device="cpu") as handle:
            for tensor_name in names_to_convert:
                entry = shard_manifest["tensors"][tensor_name]
                if tensor_name.endswith(".weight_scale_inv"):
                    continue
                tensor = handle.get_tensor(tensor_name)
                if entry["quantized"]:
                    scale = None
                    scale_name = f"{tensor_name}_scale_inv"
                    if scale_name in names_set:
                        scale = handle.get_tensor(scale_name)
                        entry["source_scale_inv"] = scale_name
                    output_tensors.update(
                        _quantize_tensor(
                            tensor_name, tensor, scale, plan.group_size, plan.bits
                        )
                    )
                    entry["output_dtype"] = "uint32+scales+biases"
                else:
                    output_tensors[tensor_name] = _torch_to_mx(tensor)
                    entry["output_dtype"] = str(output_tensors[tensor_name].dtype)
        mx.save_safetensors(
            str(tmp_file), output_tensors, metadata={"format": "mlx-affine-6bit"}
        )
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
        write_manifest(plan.output_dir / "quantization_manifest.json", manifest)


def run_dry_run(plan: QuantizationPlan) -> RunManifest:
    manifest = build_dry_run_manifest(plan)
    check_free_space(
        plan.output_dir,
        estimate_remaining_output_bytes(manifest),
        require_headroom=True,
    )
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(plan.output_dir / "quantization_manifest.json", manifest)
    copy_support_files(plan.source_dir, plan.output_dir)
    write_config(plan.source_dir, plan.output_dir, manifest)
    write_model_card_readme(plan.output_dir, manifest)
    write_output_index(plan.output_dir, manifest)
    return manifest


def run_conversion(plan: QuantizationPlan) -> RunManifest:
    manifest = run_dry_run(plan)
    for shard_name in selected_shards(load_index(plan.source_dir), plan.include_shards):
        convert_shard(plan, shard_name, manifest)
        write_output_index(plan.output_dir, manifest)
    return manifest


def cleanup_incomplete(output_dir: Path) -> None:
    guard_output_path(output_dir)
    for tmp_path in output_dir.glob("*.tmp"):
        tmp_path.unlink()
    for tmp_path in output_dir.glob("*.tmp.safetensors"):
        tmp_path.unlink()
    for tmp_path in output_dir.glob(".*.tmp"):
        tmp_path.unlink()
    for tmp_path in output_dir.glob(".*.tmp.safetensors"):
        tmp_path.unlink()
    manifest_path = output_dir / "quantization_manifest.json"
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        for shard in manifest.get("shards", {}).values():
            if shard.get("status") in {"running", "failed"}:
                shard["status"] = "cleanup-required"
        write_manifest(manifest_path, manifest)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--bits", type=int, default=DEFAULT_BITS)
    parser.add_argument("--include-shard", action="append", default=[])
    parser.add_argument("--include-tensor", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--convert", action="store_true")
    parser.add_argument("--cleanup-incomplete", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    plan = QuantizationPlan(
        source_dir=args.source,
        output_dir=args.output,
        group_size=args.group_size,
        bits=args.bits,
        include_shards=tuple(args.include_shard),
        include_tensors=tuple(args.include_tensor),
    )
    if plan.bits != DEFAULT_BITS or plan.group_size != DEFAULT_GROUP_SIZE:
        raise ValueError("This MiMo path is fixed to MLX affine 6-bit group_size=64")
    if args.cleanup_incomplete:
        cleanup_incomplete(plan.output_dir)
        return 0
    if args.convert == args.dry_run:
        raise ValueError("Choose exactly one of --dry-run or --convert")
    manifest = run_conversion(plan) if args.convert else run_dry_run(plan)
    print(
        json.dumps(
            {
                "manifest": str(plan.output_dir / "quantization_manifest.json"),
                "expected_output_size_bytes": manifest.get(
                    "expected_output_size_bytes", 0
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
