# MiMo V2.5 Pro MTP Artifact Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an experimental MLX affine 6-bit MTP-only artifact and prove it can be loaded independently of the full MiMo V2.5 Pro model.

**Architecture:** This plan adds a dedicated MTP artifact lane beside the existing 6-bit MiMo artifact. The new quantizer reuses the existing MLX affine 6-bit conversion helpers, writes only under an experimental output directory, generates artifact-local identity files, and stops after an MTP-only module load probe. It does not modify the main 778 GiB artifact, register a default runtime model, implement MTP decode, start exo, or load a second full MiMo instance.

**Tech Stack:** Python 3.13, pytest, basedpyright, ruff, safetensors, torch, MLX, existing `scripts/mimo_v25_pro_6bit_quantize.py` helpers, git.

---

## Scope Boundary

This plan implements Stage 2 and Stage 3 from the approved MTP design:

- Experimental MTP artifact identity and artifact-local model card.
- `model_mtp.safetensors` conversion into MLX affine 6-bit output.
- MTP-only quantization manifest and safetensors index.
- Standalone MTP-only module load probe using synthetic inputs.

It intentionally does not:

- Stop, restart, or start exo.
- Load the full 778 GiB MiMo V2.5 Pro 6-bit artifact.
- Modify the production artifact at `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit`.
- Register a runtime model card under `resources/inference_model_cards`.
- Change `mimo_v2_flash` sanitize behavior.
- Implement proposal, verification, acceptance, or distributed decode.
- Make MTP the default runtime path.

The experimental artifact path is:

```text
/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental
```

The experimental artifact identity is:

```text
kernelpool/MiMo-V2.5-Pro-6bit-mtp-experimental
```

## Current Evidence To Preserve

- Foundation plan: `docs/superpowers/plans/2026-05-19-mimo-v25-pro-mtp-foundation.md`
- Foundation runlog: `docs/plans/MIMO_V25_PRO_MTP_FOUNDATION_RUNLOG.md`
- Source MTP shape report: `docs/plans/artifacts/mimo-v25-pro-mtp-shape-report-20260519.json`
- Source MTP shard: `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/hf/XiaomiMiMo--MiMo-V2.5-Pro/model_mtp.safetensors`
- Source MTP tensor count: `48`
- Source MTP layers: `[0, 1, 2]`
- Current production 6-bit artifact: `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit`

## File Structure

- Create `scripts/mimo_v25_pro_mtp_quantize.py`
  - Dedicated MTP-only quantizer. It reads exactly `model_mtp.safetensors`, filters `model.mtp.layers.*`, reuses the existing MLX affine 6-bit quantization helpers, and writes one experimental MTP safetensors shard plus manifest/index/card files.
- Create `scripts/test_mimo_v25_pro_mtp_quantize.py`
  - Unit tests for path guards, MTP-only filtering, manifest/index generation, artifact-local model card generation, dry run, tiny conversion, and production-artifact protection.
- Create `scripts/mimo_v25_pro_mtp_module_probe.py`
  - Standalone MTP-only loader/probe. It loads only the experimental MTP artifact shard, validates required layer keys, and runs bounded synthetic MLX quantized matmuls for `eh_proj` and `qkv_proj`.
- Create `scripts/test_mimo_v25_pro_mtp_module_probe.py`
  - Unit tests for the probe with a tiny quantized fixture, missing-key failures, and production-artifact path rejection.
- Create `docs/plans/MIMO_V25_PRO_MTP_ARTIFACT_RUNLOG.md`
  - Operator evidence for artifact generation, real conversion output, real module probe output, verification commands, and safety notes.
- Generate outside git: `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental/model_mtp-00001-of-00001.safetensors`
  - Real MTP-only quantized shard. This is not committed.
- Generate outside git: `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental/mtp_quantization_manifest.json`
  - Real conversion manifest. This is not committed because it records local absolute paths and generated sizes.
- Generate outside git: `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental/model.safetensors.index.json`
  - Experimental MTP-only safetensors index. This is not committed.
- Generate outside git: `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental/README.md`
  - Artifact-local model card. This is not an exo registry card.
- Generate and commit summary only: `docs/plans/artifacts/mimo-v25-pro-mtp-artifact-summary-20260519.json`
  - Small JSON summary with tensor count, layers, output path, output size, and checksums.
- Generate and commit probe evidence: `docs/plans/artifacts/mimo-v25-pro-mtp-module-probe-20260519.json`
  - Small JSON report from the standalone MTP-only probe.

---

### Task 1: Add MTP Quantizer Tests

**Files:**
- Create: `scripts/test_mimo_v25_pro_mtp_quantize.py`
- Verify existing helper context: `scripts/mimo_v25_pro_6bit_quantize.py`

- [ ] **Step 1: Write failing tests for MTP-only artifact conversion**

Create `scripts/test_mimo_v25_pro_mtp_quantize.py`:

```python
import json
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import cast

import mlx.core as mx
import pytest
import safetensors.torch as safetensors_torch
import torch

from scripts import mimo_v25_pro_mtp_quantize as mtp_quantize

SaveFile = Callable[
    [dict[str, torch.Tensor], str | PathLike[str], dict[str, str] | None],
    None,
]
SAVE_FILE = cast(SaveFile, safetensors_torch.save_file)


def _write_mtp_fixture(path: Path) -> None:
    SAVE_FILE(
        {
            "model.mtp.layers.0.eh_proj.weight": torch.arange(
                128, dtype=torch.float16
            ).reshape(2, 64),
            "model.mtp.layers.0.enorm.weight": torch.ones(
                (2,), dtype=torch.float16
            ),
            "model.mtp.layers.0.self_attn.qkv_proj.weight": torch.arange(
                128, dtype=torch.float16
            ).reshape(2, 64),
            "model.mtp.layers.0.self_attn.attention_sink_bias": torch.zeros(
                (2,), dtype=torch.float16
            ),
            "model.layers.0.mlp.down_proj.weight": torch.ones(
                (2, 64), dtype=torch.float16
            ),
        },
        path,
        None,
    )


@pytest.fixture()
def mtp_fixture_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_shard = source_dir / "model_mtp.safetensors"
    output_dir = tmp_path / "kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental"
    _write_mtp_fixture(source_shard)
    monkeypatch.setattr(mtp_quantize, "MTP_SOURCE_SHARD", source_shard)
    monkeypatch.setattr(mtp_quantize, "MTP_OUTPUT_ROOT", output_dir)
    monkeypatch.setattr(mtp_quantize, "MTP_HEADROOM_BYTES", 0)
    return source_shard, output_dir


def test_mtp_output_guard_rejects_production_artifact_path() -> None:
    with pytest.raises(ValueError, match="experimental MTP output"):
        mtp_quantize.guard_mtp_output_path(
            mtp_quantize.PRODUCTION_6BIT_ARTIFACT_ROOT
        )


def test_mtp_source_guard_rejects_non_default_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="MTP source shard"):
        mtp_quantize.guard_mtp_source_shard(tmp_path / "model_mtp.safetensors")


def test_mtp_output_guard_accepts_experimental_root_and_child(
    mtp_fixture_paths: tuple[Path, Path],
) -> None:
    _source_shard, output_dir = mtp_fixture_paths

    mtp_quantize.guard_mtp_output_path(output_dir)
    mtp_quantize.guard_mtp_output_path(output_dir / "run-0001")


def test_build_mtp_manifest_filters_only_mtp_tensors(
    mtp_fixture_paths: tuple[Path, Path],
) -> None:
    source_shard, output_dir = mtp_fixture_paths

    manifest = mtp_quantize.build_mtp_dry_run_manifest(
        mtp_quantize.MtpQuantizationPlan(source_shard, output_dir)
    )

    shard = manifest["shards"][mtp_quantize.MTP_OUTPUT_SHARD_NAME]
    assert shard["status"] == "dry-run"
    assert "model.mtp.layers.0.eh_proj.weight" in shard["tensors"]
    assert "model.layers.0.mlp.down_proj.weight" not in shard["tensors"]
    assert shard["tensors"]["model.mtp.layers.0.eh_proj.weight"]["quantized"] is True
    assert (
        shard["tensors"]["model.mtp.layers.0.enorm.weight"]["quantized"] is False
    )


def test_mtp_dry_run_writes_index_manifest_config_and_model_card(
    mtp_fixture_paths: tuple[Path, Path],
) -> None:
    source_shard, output_dir = mtp_fixture_paths

    manifest = mtp_quantize.run_mtp_dry_run(
        mtp_quantize.MtpQuantizationPlan(source_shard, output_dir)
    )

    assert manifest["model_id"] == mtp_quantize.MTP_EXPERIMENTAL_MODEL_ID
    assert (output_dir / mtp_quantize.MTP_MANIFEST_NAME).is_file()
    assert (output_dir / "model.safetensors.index.json").is_file()
    assert (output_dir / "config.json").is_file()
    assert (output_dir / "README.md").is_file()

    index = json.loads((output_dir / "model.safetensors.index.json").read_text())
    assert index["metadata"]["artifact_kind"] == "mimo-v25-pro-mtp-only"
    assert (
        index["weight_map"]["model.mtp.layers.0.eh_proj.weight.scales"]
        == mtp_quantize.MTP_OUTPUT_SHARD_NAME
    )

    config = json.loads((output_dir / "config.json").read_text())
    assert config["artifact_kind"] == "mimo-v25-pro-mtp-only"
    assert config["base_model_id"] == mtp_quantize.PRODUCTION_6BIT_MODEL_ID
    assert config["mtp_config"] == {"enabled": True, "num_layers": 3}

    readme = (output_dir / "README.md").read_text()
    assert mtp_quantize.MTP_EXPERIMENTAL_MODEL_ID in readme
    assert "not a standalone runtime model" in readme


def test_mtp_conversion_writes_quantized_shard_and_no_tmp_files(
    mtp_fixture_paths: tuple[Path, Path],
) -> None:
    source_shard, output_dir = mtp_fixture_paths

    manifest = mtp_quantize.run_mtp_conversion(
        mtp_quantize.MtpQuantizationPlan(source_shard, output_dir)
    )

    shard = manifest["shards"][mtp_quantize.MTP_OUTPUT_SHARD_NAME]
    assert shard["status"] == "complete"
    assert shard["output_size"] > 0
    assert not list(output_dir.glob("*.tmp"))
    assert not list(output_dir.glob(".*.tmp"))

    loaded = mx.load(str(output_dir / mtp_quantize.MTP_OUTPUT_SHARD_NAME))
    assert loaded["model.mtp.layers.0.eh_proj.weight"].dtype == mx.uint32
    assert loaded["model.mtp.layers.0.eh_proj.weight.scales"].shape[0] == 2
    assert loaded["model.mtp.layers.0.enorm.weight"].shape == (2,)
    assert "model.layers.0.mlp.down_proj.weight" not in loaded
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest scripts/test_mimo_v25_pro_mtp_quantize.py -q
```

Expected:

```text
ImportError: cannot import name 'mimo_v25_pro_mtp_quantize' from 'scripts'
```

The exact pytest header may vary. The important failure is that the new MTP quantizer module does not exist yet.

- [ ] **Step 3: Commit the failing tests**

Run:

```bash
git add scripts/test_mimo_v25_pro_mtp_quantize.py
git commit -m "Specify MiMo MTP quantized artifact behavior" \
  -m "Add failing tests for a dedicated MTP-only MLX affine 6-bit artifact converter before implementing the converter." \
  -m "Constraint: The production MiMo 6-bit artifact must not be mutated" \
  -m "Confidence: high" \
  -m "Scope-risk: narrow" \
  -m "Tested: uv run pytest scripts/test_mimo_v25_pro_mtp_quantize.py -q fails because the module is missing" \
  -m "Not-tested: Real MTP conversion"
```

---

### Task 2: Implement MTP-Only Quantizer

**Files:**
- Create: `scripts/mimo_v25_pro_mtp_quantize.py`
- Test: `scripts/test_mimo_v25_pro_mtp_quantize.py`

- [ ] **Step 1: Create the dedicated MTP quantizer module**

Create `scripts/mimo_v25_pro_mtp_quantize.py` with these definitions and behavior:

```python
#!/usr/bin/env python3
"""Create an experimental MLX affine 6-bit MiMo V2.5 Pro MTP-only artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, TypedDict, cast

import mlx.core as mx
import torch
from safetensors import safe_open

from scripts import mimo_v25_pro_6bit_quantize as base_quantize

SOURCE_CHECKPOINT: Final[Path] = base_quantize.SOURCE_CHECKPOINT
MTP_SOURCE_SHARD: Path = SOURCE_CHECKPOINT / "model_mtp.safetensors"
PRODUCTION_6BIT_MODEL_ID: Final[str] = "kernelpool/MiMo-V2.5-Pro-6bit"
PRODUCTION_6BIT_ARTIFACT_ROOT: Final[Path] = Path(
    "/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit"
)
MTP_EXPERIMENTAL_MODEL_ID: Final[str] = (
    "kernelpool/MiMo-V2.5-Pro-6bit-mtp-experimental"
)
MTP_OUTPUT_ROOT: Path = Path(
    "/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/"
    "kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental"
)
MTP_OUTPUT_SHARD_NAME: Final[str] = "model_mtp-00001-of-00001.safetensors"
MTP_MANIFEST_NAME: Final[str] = "mtp_quantization_manifest.json"
DEFAULT_GROUP_SIZE: Final[int] = base_quantize.DEFAULT_GROUP_SIZE
DEFAULT_BITS: Final[int] = base_quantize.DEFAULT_BITS
EXPECTED_MTP_LAYERS: Final[tuple[int, ...]] = (0, 1, 2)
MTP_HEADROOM_BYTES: int = 8 * 1024**3


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


class MtpManifest(TypedDict, total=False):
    model_id: str
    artifact_kind: str
    base_model_id: str
    source_model_id: str
    source_mtp_shard: str
    output_dir: str
    output_shard: str
    mtp_layers: list[int]
    quantization: dict[str, object]
    expected_output_size_bytes: int
    created_at: float
    updated_at: float
    shards: dict[str, ShardManifestEntry]


@dataclass(frozen=True)
class MtpQuantizationPlan:
    source_shard: Path
    output_dir: Path
    group_size: int = DEFAULT_GROUP_SIZE
    bits: int = DEFAULT_BITS


class _SafeSlice(Protocol):
    def get_dtype(self) -> str:
        raise NotImplementedError

    def get_shape(self) -> Sequence[int]:
        raise NotImplementedError


class _SafeOpenHandle(Protocol):
    def keys(self) -> Sequence[str]:
        raise NotImplementedError

    def get_slice(self, name: str) -> _SafeSlice:
        raise NotImplementedError

    def get_tensor(self, name: str) -> torch.Tensor:
        raise NotImplementedError


def _is_mtp_tensor(name: str) -> bool:
    return name.startswith("model.mtp.layers.")


def guard_mtp_source_shard(source_shard: Path) -> None:
    resolved = source_shard.resolve()
    expected = MTP_SOURCE_SHARD.resolve()
    if resolved != expected:
        raise ValueError(
            f"Refusing MTP source shard {source_shard}; expected {MTP_SOURCE_SHARD}"
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"MTP source shard does not exist: {resolved}")


def guard_mtp_output_path(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    root = MTP_OUTPUT_ROOT.resolve()
    production = PRODUCTION_6BIT_ARTIFACT_ROOT.resolve()
    source = SOURCE_CHECKPOINT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(
            f"Refusing output path {output_dir}; must be under experimental MTP output {MTP_OUTPUT_ROOT}"
        )
    if resolved == production or production in resolved.parents:
        raise ValueError("Refusing to write MTP output inside the production 6-bit artifact")
    if resolved == source or source in resolved.parents:
        raise ValueError("Refusing to write MTP output inside the source checkpoint")


def load_mtp_manifest(path: Path) -> MtpManifest:
    if not path.exists():
        return {"shards": {}}
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid MTP manifest: {path}")
    raw.setdefault("shards", {})
    return cast(MtpManifest, raw)


def write_mtp_manifest(path: Path, manifest: MtpManifest) -> None:
    manifest["updated_at"] = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def _mtp_tensor_names(source_shard: Path) -> list[str]:
    with safe_open(str(source_shard), framework="pt") as raw_handle:
        handle = cast(_SafeOpenHandle, cast(object, raw_handle))
        return sorted(name for name in handle.keys() if _is_mtp_tensor(name))


def build_mtp_dry_run_manifest(plan: MtpQuantizationPlan) -> MtpManifest:
    guard_mtp_source_shard(plan.source_shard)
    guard_mtp_output_path(plan.output_dir)
    previous_manifest = load_mtp_manifest(plan.output_dir / MTP_MANIFEST_NAME)
    names = _mtp_tensor_names(plan.source_shard)
    names_set = set(names)
    tensors: dict[str, TensorManifestEntry] = {}
    expected_output_size = 0

    with safe_open(str(plan.source_shard), framework="pt") as raw_handle:
        handle = cast(_SafeOpenHandle, cast(object, raw_handle))
        for tensor_name in names:
            tensor_slice = handle.get_slice(tensor_name)
            shape = [int(dim) for dim in tensor_slice.get_shape()]
            dtype = tensor_slice.get_dtype()
            source_nbytes = base_quantize._source_nbytes(shape, dtype)
            entry = base_quantize.estimate_tensor_entry(
                tensor_name,
                shape,
                dtype,
                source_nbytes,
                names_set,
                plan.group_size,
                plan.bits,
            )
            tensors[tensor_name] = cast(TensorManifestEntry, entry)
            if not tensor_name.endswith(".weight_scale_inv"):
                expected_output_size += int(entry["estimated_output_nbytes"])

    return {
        "model_id": MTP_EXPERIMENTAL_MODEL_ID,
        "artifact_kind": "mimo-v25-pro-mtp-only",
        "base_model_id": PRODUCTION_6BIT_MODEL_ID,
        "source_model_id": base_quantize.BASE_MODEL_ID,
        "source_mtp_shard": str(plan.source_shard),
        "output_dir": str(plan.output_dir),
        "output_shard": MTP_OUTPUT_SHARD_NAME,
        "mtp_layers": list(EXPECTED_MTP_LAYERS),
        "quantization": {
            "format": "mlx-affine-6bit-mtp-only",
            "bits": plan.bits,
            "group_size": plan.group_size,
            "mode": "affine",
            "source_fp8_dequantization": "torch float8_e4m3fn -> fp32 with paired *_scale_inv blocks",
        },
        "expected_output_size_bytes": expected_output_size,
        "created_at": previous_manifest.get("created_at", time.time()),
        "shards": {
            MTP_OUTPUT_SHARD_NAME: {
                "source_file": str(plan.source_shard),
                "output_file": MTP_OUTPUT_SHARD_NAME,
                "status": "dry-run",
                "source_size": plan.source_shard.stat().st_size,
                "tensors": tensors,
            }
        },
    }


def check_mtp_free_space(output_dir: Path, expected_output_size: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output_dir).free
    required = expected_output_size + MTP_HEADROOM_BYTES
    if free < required:
        raise RuntimeError(
            f"Insufficient free space at {output_dir}: free={free}, required={required}"
        )


def write_mtp_config(output_dir: Path, manifest: MtpManifest) -> None:
    config = {
        "artifact_kind": "mimo-v25-pro-mtp-only",
        "model_id": MTP_EXPERIMENTAL_MODEL_ID,
        "base_model_id": PRODUCTION_6BIT_MODEL_ID,
        "source_model_id": base_quantize.BASE_MODEL_ID,
        "source_mtp_shard": manifest["source_mtp_shard"],
        "architecture": "MiMoV2ForCausalLM",
        "hidden_size": 6144,
        "num_mtp_layers": len(EXPECTED_MTP_LAYERS),
        "mtp_config": {"enabled": True, "num_layers": len(EXPECTED_MTP_LAYERS)},
        "quantization": {"group_size": DEFAULT_GROUP_SIZE, "bits": DEFAULT_BITS, "mode": "affine"},
        "quantization_config": {
            "quant_method": "mlx-affine",
            "group_size": DEFAULT_GROUP_SIZE,
            "bits": DEFAULT_BITS,
            "mode": "affine",
            "artifact_kind": "mimo-v25-pro-mtp-only",
        },
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )


def write_mtp_readme(output_dir: Path, manifest: MtpManifest) -> None:
    text = f"""# MiMo V2.5 Pro 6-bit MTP Experimental Artifact

Model id: `{MTP_EXPERIMENTAL_MODEL_ID}`

Base runtime artifact: `{PRODUCTION_6BIT_MODEL_ID}`

Artifact kind: `mimo-v25-pro-mtp-only`

This is not a standalone runtime model. It contains only quantized MTP module weights
derived from `model_mtp.safetensors` and must be paired with the base MiMo V2.5 Pro
6-bit artifact in later opt-in runtime work.

Quantization: MLX affine packed 6-bit, `group_size=64`, `bits=6`, `mode=affine`.

Generated manifest: `{MTP_MANIFEST_NAME}`
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text(text)


def write_mtp_output_index(output_dir: Path, manifest: MtpManifest) -> None:
    shard = manifest["shards"][MTP_OUTPUT_SHARD_NAME]
    weight_map: dict[str, str] = {}
    total_size = 0
    for tensor_name, entry in shard.get("tensors", {}).items():
        if tensor_name.endswith(".weight_scale_inv") and entry.get("quantized") is False:
            continue
        for output_key in entry.get("output_keys", [tensor_name]):
            weight_map[output_key] = MTP_OUTPUT_SHARD_NAME
        total_size += int(entry.get("estimated_output_nbytes", 0))
    index = {
        "metadata": {
            "artifact_kind": "mimo-v25-pro-mtp-only",
            "total_size": total_size,
            "save_format": "mlx-affine-6bit",
            "quantization_format": "mlx-affine-6bit-mtp-only",
            "base_model": PRODUCTION_6BIT_MODEL_ID,
            "mtp_layers": list(EXPECTED_MTP_LAYERS),
        },
        "weight_map": weight_map,
    }
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )


def run_mtp_dry_run(plan: MtpQuantizationPlan) -> MtpManifest:
    if plan.bits != DEFAULT_BITS or plan.group_size != DEFAULT_GROUP_SIZE:
        raise ValueError("MTP artifact is fixed to MLX affine 6-bit group_size=64")
    manifest = build_mtp_dry_run_manifest(plan)
    check_mtp_free_space(plan.output_dir, manifest["expected_output_size_bytes"])
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    write_mtp_manifest(plan.output_dir / MTP_MANIFEST_NAME, manifest)
    write_mtp_config(plan.output_dir, manifest)
    write_mtp_readme(plan.output_dir, manifest)
    write_mtp_output_index(plan.output_dir, manifest)
    return manifest


def convert_mtp_shard(plan: MtpQuantizationPlan, manifest: MtpManifest) -> None:
    shard_manifest = manifest["shards"][MTP_OUTPUT_SHARD_NAME]
    output_file = plan.output_dir / MTP_OUTPUT_SHARD_NAME
    tmp_file = plan.output_dir / f".{MTP_OUTPUT_SHARD_NAME}.tmp.safetensors"
    shard_manifest["status"] = "running"
    shard_manifest["started_at"] = time.time()
    write_mtp_manifest(plan.output_dir / MTP_MANIFEST_NAME, manifest)
    output_tensors: dict[str, mx.array] = {}
    names_set = set(shard_manifest["tensors"])
    try:
        with safe_open(str(plan.source_shard), framework="pt") as raw_handle:
            handle = cast(_SafeOpenHandle, cast(object, raw_handle))
            for tensor_name, entry in shard_manifest["tensors"].items():
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
                        base_quantize._quantize_tensor(
                            tensor_name,
                            tensor,
                            scale,
                            plan.group_size,
                            plan.bits,
                        )
                    )
                    entry["output_dtype"] = "uint32+scales+biases"
                else:
                    output_tensors[tensor_name] = base_quantize._torch_to_mx(tensor)
                    entry["output_dtype"] = str(output_tensors[tensor_name].dtype)
        mx.save_safetensors(
            str(tmp_file),
            output_tensors,
            metadata={"format": "mlx-affine-6bit-mtp-only"},
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
        write_mtp_manifest(plan.output_dir / MTP_MANIFEST_NAME, manifest)


def run_mtp_conversion(plan: MtpQuantizationPlan) -> MtpManifest:
    manifest = run_mtp_dry_run(plan)
    convert_mtp_shard(plan, manifest)
    write_mtp_output_index(plan.output_dir, manifest)
    return manifest


def cleanup_incomplete(output_dir: Path) -> None:
    guard_mtp_output_path(output_dir)
    for pattern in ("*.tmp", "*.tmp.safetensors", ".*.tmp", ".*.tmp.safetensors"):
        for tmp_path in output_dir.glob(pattern):
            tmp_path.unlink()
    manifest_path = output_dir / MTP_MANIFEST_NAME
    if manifest_path.exists():
        manifest = load_mtp_manifest(manifest_path)
        for shard in manifest.get("shards", {}).values():
            if shard.get("status") in {"running", "failed"}:
                shard["status"] = "cleanup-required"
        write_mtp_manifest(manifest_path, manifest)


def write_summary(output_dir: Path, summary_path: Path) -> dict[str, object]:
    manifest = load_mtp_manifest(output_dir / MTP_MANIFEST_NAME)
    shard_path = output_dir / MTP_OUTPUT_SHARD_NAME
    index = json.loads((output_dir / "model.safetensors.index.json").read_text())
    summary = {
        "model_id": MTP_EXPERIMENTAL_MODEL_ID,
        "artifact_kind": "mimo-v25-pro-mtp-only",
        "output_dir": str(output_dir),
        "output_shard": str(shard_path),
        "output_shard_size": shard_path.stat().st_size if shard_path.exists() else 0,
        "output_shard_sha256": sha256_file(shard_path) if shard_path.exists() else "",
        "tensor_count": len(index["weight_map"]),
        "mtp_layers": list(EXPECTED_MTP_LAYERS),
        "complete": manifest["shards"][MTP_OUTPUT_SHARD_NAME].get("status") == "complete",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-shard", type=Path, default=MTP_SOURCE_SHARD)
    parser.add_argument("--output", type=Path, default=MTP_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--convert", action="store_true")
    parser.add_argument("--cleanup-incomplete", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    plan = MtpQuantizationPlan(args.source_shard, args.output)
    if args.cleanup_incomplete:
        cleanup_incomplete(plan.output_dir)
        return 0
    if args.convert == args.dry_run:
        raise ValueError("Choose exactly one of --dry-run or --convert")
    manifest = run_mtp_conversion(plan) if args.convert else run_mtp_dry_run(plan)
    summary: dict[str, object] = {
        "manifest": str(plan.output_dir / MTP_MANIFEST_NAME),
        "expected_output_size_bytes": manifest["expected_output_size_bytes"],
    }
    if args.summary_json is not None:
        summary = write_summary(plan.output_dir, args.summary_json)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run MTP quantizer tests**

Run:

```bash
uv run pytest scripts/test_mimo_v25_pro_mtp_quantize.py -q
```

Expected:

```text
6 passed
```

- [ ] **Step 3: Run lint and typecheck for the quantizer**

Run:

```bash
uv run ruff check scripts/mimo_v25_pro_mtp_quantize.py scripts/test_mimo_v25_pro_mtp_quantize.py
uv run basedpyright scripts/mimo_v25_pro_mtp_quantize.py scripts/test_mimo_v25_pro_mtp_quantize.py
```

Expected:

```text
All checks passed!
0 errors, 0 warnings, 0 notes
```

- [ ] **Step 4: Commit the MTP quantizer**

Run:

```bash
git add scripts/mimo_v25_pro_mtp_quantize.py scripts/test_mimo_v25_pro_mtp_quantize.py
git commit -m "Add MiMo MTP-only quantized artifact builder" \
  -m "Create a dedicated converter for the MiMo model_mtp.safetensors shard that writes only to the experimental MTP artifact path and produces MTP-only manifest, index, config, and model-card README files." \
  -m "Constraint: The production 6-bit artifact must remain unchanged" \
  -m "Rejected: Register an exo runtime model card in this phase | module loading and decode are not proven yet" \
  -m "Confidence: high" \
  -m "Scope-risk: moderate" \
  -m "Tested: uv run pytest scripts/test_mimo_v25_pro_mtp_quantize.py -q; uv run ruff check scripts/mimo_v25_pro_mtp_quantize.py scripts/test_mimo_v25_pro_mtp_quantize.py; uv run basedpyright scripts/mimo_v25_pro_mtp_quantize.py scripts/test_mimo_v25_pro_mtp_quantize.py" \
  -m "Not-tested: Real model_mtp.safetensors conversion"
```

---

### Task 3: Generate The Real Experimental MTP Artifact

**Files:**
- Generate outside git: `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental/model_mtp-00001-of-00001.safetensors`
- Generate outside git: `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental/mtp_quantization_manifest.json`
- Generate outside git: `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental/model.safetensors.index.json`
- Generate outside git: `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental/config.json`
- Generate outside git: `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental/README.md`
- Create: `docs/plans/artifacts/mimo-v25-pro-mtp-artifact-summary-20260519.json`
- Create or modify: `docs/plans/MIMO_V25_PRO_MTP_ARTIFACT_RUNLOG.md`

- [ ] **Step 1: Run a dry run against the real MTP source shard**

Run:

```bash
uv run python scripts/mimo_v25_pro_mtp_quantize.py --dry-run
python3 - <<'PY'
import json
from pathlib import Path
root = Path("/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental")
manifest = json.loads((root / "mtp_quantization_manifest.json").read_text())
index = json.loads((root / "model.safetensors.index.json").read_text())
print(manifest["model_id"])
print(manifest["shards"]["model_mtp-00001-of-00001.safetensors"]["status"])
print(len(manifest["shards"]["model_mtp-00001-of-00001.safetensors"]["tensors"]))
print(index["metadata"]["artifact_kind"])
print(index["metadata"]["mtp_layers"])
PY
```

Expected:

```text
kernelpool/MiMo-V2.5-Pro-6bit-mtp-experimental
dry-run
48
mimo-v25-pro-mtp-only
[0, 1, 2]
```

- [ ] **Step 2: Convert the real MTP shard**

Run:

```bash
uv run python scripts/mimo_v25_pro_mtp_quantize.py \
  --convert \
  --summary-json docs/plans/artifacts/mimo-v25-pro-mtp-artifact-summary-20260519.json
```

Expected:

- Command exits `0`.
- Output JSON contains `"complete": true`.
- Output shard exists at:

```text
/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental/model_mtp-00001-of-00001.safetensors
```

- [ ] **Step 3: Verify the production artifact was not modified**

Run:

```bash
test ! -e /Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit/model_mtp-00001-of-00001.safetensors
git diff --cached --name-only
git status --short
```

Expected:

- `test ! -e /Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit/model_mtp-00001-of-00001.safetensors` exits `0`.
- No staged files before staging the runlog and summary.
- Existing unrelated dirty KV/cache runtime files remain visible and unstaged.

- [ ] **Step 4: Validate the generated artifact summary**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path
summary = json.loads(Path("docs/plans/artifacts/mimo-v25-pro-mtp-artifact-summary-20260519.json").read_text())
assert summary["model_id"] == "kernelpool/MiMo-V2.5-Pro-6bit-mtp-experimental"
assert summary["artifact_kind"] == "mimo-v25-pro-mtp-only"
assert summary["complete"] is True
assert summary["output_shard_size"] > 0
assert len(summary["output_shard_sha256"]) == 64
assert summary["mtp_layers"] == [0, 1, 2]
print(summary["output_shard_size"])
print(summary["tensor_count"])
print(summary["complete"])
PY
```

Expected:

- First printed value is greater than `0`.
- Second printed value is greater than `48` because quantized weights add `.scales` and `.biases` keys.
- Third printed value is `True`.

- [ ] **Step 5: Write the artifact runlog**

Create `docs/plans/MIMO_V25_PRO_MTP_ARTIFACT_RUNLOG.md` with this content, updating command outputs with the observed values:

```markdown
# MiMo V2.5 Pro MTP Artifact Runlog

Date: 2026-05-19

## Scope

This run generated the experimental MTP-only artifact for MiMo V2.5 Pro. It did
not start exo, load the full main model, alter distributed decode, or modify the
production 6-bit artifact.

## Artifact Identity

- Experimental model id: `kernelpool/MiMo-V2.5-Pro-6bit-mtp-experimental`
- Artifact kind: `mimo-v25-pro-mtp-only`
- Base runtime artifact: `kernelpool/MiMo-V2.5-Pro-6bit`
- Source MTP shard: `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/hf/XiaomiMiMo--MiMo-V2.5-Pro/model_mtp.safetensors`
- Output directory: `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental`
- Output shard: `model_mtp-00001-of-00001.safetensors`

## Conversion Evidence

- Dry run: `uv run python scripts/mimo_v25_pro_mtp_quantize.py --dry-run`
  - Result: `model_id=kernelpool/MiMo-V2.5-Pro-6bit-mtp-experimental`, `status=dry-run`, `source_tensor_count=48`, `artifact_kind=mimo-v25-pro-mtp-only`, `mtp_layers=[0, 1, 2]`
- Conversion: `uv run python scripts/mimo_v25_pro_mtp_quantize.py --convert --summary-json docs/plans/artifacts/mimo-v25-pro-mtp-artifact-summary-20260519.json`
  - Result: complete; summary JSON written
- Summary: `docs/plans/artifacts/mimo-v25-pro-mtp-artifact-summary-20260519.json`
  - Result: `output_shard_size > 0`, `tensor_count > 48`, `complete=true`

## Safety Evidence

- Production artifact check: `test ! -e /Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit/model_mtp-00001-of-00001.safetensors`
  - Result: passed
- Staged state before commit: `git diff --cached --name-only`
  - Result: empty
- Working tree note: only pre-existing dirty KV/cache runtime files remained unstaged.

## Next Gate

Run the standalone MTP-only module probe before any runtime decode plan.
```

- [ ] **Step 6: Commit artifact generation evidence**

Run:

```bash
git add docs/plans/MIMO_V25_PRO_MTP_ARTIFACT_RUNLOG.md \
  docs/plans/artifacts/mimo-v25-pro-mtp-artifact-summary-20260519.json
git commit -m "Record MiMo MTP artifact generation" \
  -m "Record the experimental MTP-only artifact identity, conversion evidence, and safety checks after converting model_mtp.safetensors to MLX affine 6-bit output." \
  -m "Constraint: Generated MTP shard lives outside git under the experimental artifact directory" \
  -m "Constraint: Production MiMo 6-bit artifact remains untouched" \
  -m "Confidence: high" \
  -m "Scope-risk: narrow" \
  -m "Tested: dry-run conversion; real MTP conversion; summary validation; production artifact absence check" \
  -m "Not-tested: MTP module load or decode"
```

---

### Task 4: Add Standalone MTP-Only Module Probe Tests

**Files:**
- Create: `scripts/test_mimo_v25_pro_mtp_module_probe.py`

- [ ] **Step 1: Write failing tests for the standalone MTP-only probe**

Create `scripts/test_mimo_v25_pro_mtp_module_probe.py`:

```python
import json
from pathlib import Path
from typing import cast

import mlx.core as mx
import pytest

from scripts import mimo_v25_pro_mtp_module_probe as probe


def _write_tiny_quantized_artifact(root: Path) -> None:
    root.mkdir()
    eh_weight, eh_scales, eh_biases = mx.quantize(
        mx.ones((2, 64)), group_size=64, bits=6, mode="affine"
    )
    qkv_weight, qkv_scales, qkv_biases = mx.quantize(
        mx.ones((4, 64)), group_size=64, bits=6, mode="affine"
    )
    mx.save_safetensors(
        str(root / probe.MTP_OUTPUT_SHARD_NAME),
        {
            "model.mtp.layers.0.eh_proj.weight": eh_weight,
            "model.mtp.layers.0.eh_proj.weight.scales": eh_scales,
            "model.mtp.layers.0.eh_proj.weight.biases": eh_biases,
            "model.mtp.layers.0.self_attn.qkv_proj.weight": qkv_weight,
            "model.mtp.layers.0.self_attn.qkv_proj.weight.scales": qkv_scales,
            "model.mtp.layers.0.self_attn.qkv_proj.weight.biases": qkv_biases,
            "model.mtp.layers.0.enorm.weight": mx.ones((2,)),
        },
        metadata={"format": "mlx-affine-6bit-mtp-only"},
    )
    index = {
        "metadata": {
            "artifact_kind": "mimo-v25-pro-mtp-only",
            "mtp_layers": [0],
            "save_format": "mlx-affine-6bit",
        },
        "weight_map": {
            "model.mtp.layers.0.eh_proj.weight": probe.MTP_OUTPUT_SHARD_NAME,
            "model.mtp.layers.0.eh_proj.weight.scales": probe.MTP_OUTPUT_SHARD_NAME,
            "model.mtp.layers.0.eh_proj.weight.biases": probe.MTP_OUTPUT_SHARD_NAME,
            "model.mtp.layers.0.self_attn.qkv_proj.weight": probe.MTP_OUTPUT_SHARD_NAME,
            "model.mtp.layers.0.self_attn.qkv_proj.weight.scales": probe.MTP_OUTPUT_SHARD_NAME,
            "model.mtp.layers.0.self_attn.qkv_proj.weight.biases": probe.MTP_OUTPUT_SHARD_NAME,
            "model.mtp.layers.0.enorm.weight": probe.MTP_OUTPUT_SHARD_NAME,
        },
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest scripts/test_mimo_v25_pro_mtp_module_probe.py -q
```

Expected:

```text
ImportError: cannot import name 'mimo_v25_pro_mtp_module_probe' from 'scripts'
```

- [ ] **Step 3: Commit the failing probe tests**

Run:

```bash
git add scripts/test_mimo_v25_pro_mtp_module_probe.py
git commit -m "Specify MiMo MTP module probe behavior" \
  -m "Add failing tests for the standalone MTP-only module probe before implementing the probe." \
  -m "Constraint: Probe must load only the experimental MTP artifact, not the full MiMo model" \
  -m "Confidence: high" \
  -m "Scope-risk: narrow" \
  -m "Tested: uv run pytest scripts/test_mimo_v25_pro_mtp_module_probe.py -q fails because the module is missing" \
  -m "Not-tested: Real MTP artifact probe"
```

---

### Task 5: Implement Standalone MTP-Only Module Probe

**Files:**
- Create: `scripts/mimo_v25_pro_mtp_module_probe.py`
- Test: `scripts/test_mimo_v25_pro_mtp_module_probe.py`

- [ ] **Step 1: Create the standalone module probe**

Create `scripts/mimo_v25_pro_mtp_module_probe.py`:

```python
#!/usr/bin/env python3
"""Load and probe only the experimental MiMo V2.5 Pro MTP artifact."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final, TypedDict, cast

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


class MtpProbeReport(TypedDict):
    artifact_dir: str
    artifact_kind: str
    layers: list[int]
    loaded_tensor_count: int
    required_key_count: int
    synthetic_forwards: dict[str, list[int]]


def guard_probe_artifact_path(artifact_dir: Path) -> None:
    resolved = artifact_dir.resolve()
    root = MTP_ARTIFACT_ROOT.resolve()
    production = PRODUCTION_6BIT_ARTIFACT_ROOT.resolve()
    if resolved == production or production in resolved.parents:
        raise ValueError("Refusing to probe production artifact as an experimental MTP artifact")
    if resolved != root and root not in resolved.parents and artifact_dir.is_absolute():
        raise ValueError(f"Refusing artifact path outside experimental MTP artifact root: {artifact_dir}")


def load_index(artifact_dir: Path) -> dict[str, object]:
    index_path = artifact_dir / "model.safetensors.index.json"
    raw = json.loads(index_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid safetensors index: {index_path}")
    return cast(dict[str, object], raw)


def _required_quantized_projection_keys(layer: int, projection: str) -> list[str]:
    base = f"model.mtp.layers.{layer}.{projection}.weight"
    return [base, f"{base}.scales", f"{base}.biases"]


def required_mtp_keys(layers: Sequence[int]) -> list[str]:
    keys: list[str] = []
    for layer in layers:
        keys.extend(_required_quantized_projection_keys(layer, "eh_proj"))
        keys.extend(_required_quantized_projection_keys(layer, "self_attn.qkv_proj"))
        keys.append(f"model.mtp.layers.{layer}.enorm.weight")
    return keys


def _validate_required_keys(weight_map: dict[str, str], layers: Sequence[int]) -> None:
    missing = [key for key in required_mtp_keys(layers) if key not in weight_map]
    if missing:
        raise ValueError(f"missing required MTP keys: {missing[:8]}")


def _quantized_matmul(
    x: mx.array,
    tensors: dict[str, mx.array],
    base_key: str,
) -> mx.array:
    return mx.quantized_matmul(
        x,
        tensors[base_key],
        tensors[f"{base_key}.scales"],
        tensors[f"{base_key}.biases"],
        transpose=True,
        group_size=64,
        bits=6,
        mode="affine",
    )


def probe_mtp_artifact(
    artifact_dir: Path,
    *,
    layers: Sequence[int] = DEFAULT_LAYERS,
    eh_input_size: int = DEFAULT_EH_INPUT_SIZE,
    qkv_input_size: int = DEFAULT_HIDDEN_SIZE,
) -> MtpProbeReport:
    guard_probe_artifact_path(artifact_dir)
    index = load_index(artifact_dir)
    metadata = cast(dict[str, object], index.get("metadata", {}))
    weight_map = cast(dict[str, str], index.get("weight_map", {}))
    if metadata.get("artifact_kind") != "mimo-v25-pro-mtp-only":
        raise ValueError("MTP artifact index must declare artifact_kind=mimo-v25-pro-mtp-only")
    _validate_required_keys(weight_map, layers)

    shard_path = artifact_dir / MTP_OUTPUT_SHARD_NAME
    tensors = cast(dict[str, mx.array], mx.load(str(shard_path)))
    synthetic_forwards: dict[str, list[int]] = {}
    first_layer = int(layers[0])

    eh_input = mx.ones((1, 1, eh_input_size), dtype=mx.float16)
    eh_base = f"model.mtp.layers.{first_layer}.eh_proj.weight"
    eh_output = _quantized_matmul(eh_input, tensors, eh_base)
    mx.eval(eh_output)
    synthetic_forwards[f"layer_{first_layer}_eh_proj"] = [
        int(dim) for dim in eh_output.shape
    ]

    qkv_input = mx.ones((1, 1, qkv_input_size), dtype=mx.float16)
    qkv_base = f"model.mtp.layers.{first_layer}.self_attn.qkv_proj.weight"
    qkv_output = _quantized_matmul(qkv_input, tensors, qkv_base)
    mx.eval(qkv_output)
    synthetic_forwards[f"layer_{first_layer}_qkv_proj"] = [
        int(dim) for dim in qkv_output.shape
    ]

    return {
        "artifact_dir": str(artifact_dir),
        "artifact_kind": "mimo-v25-pro-mtp-only",
        "layers": [int(layer) for layer in layers],
        "loaded_tensor_count": len(tensors),
        "required_key_count": len(required_mtp_keys(layers)),
        "synthetic_forwards": synthetic_forwards,
    }


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=MTP_ARTIFACT_ROOT)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args(argv)


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
```

- [ ] **Step 2: Run probe tests**

Run:

```bash
uv run pytest scripts/test_mimo_v25_pro_mtp_module_probe.py -q
```

Expected:

```text
3 passed
```

- [ ] **Step 3: Run lint and typecheck for the probe**

Run:

```bash
uv run ruff check scripts/mimo_v25_pro_mtp_module_probe.py scripts/test_mimo_v25_pro_mtp_module_probe.py
uv run basedpyright scripts/mimo_v25_pro_mtp_module_probe.py scripts/test_mimo_v25_pro_mtp_module_probe.py
```

Expected:

```text
All checks passed!
0 errors, 0 warnings, 0 notes
```

- [ ] **Step 4: Commit the standalone module probe**

Run:

```bash
git add scripts/mimo_v25_pro_mtp_module_probe.py scripts/test_mimo_v25_pro_mtp_module_probe.py
git commit -m "Probe MiMo MTP artifact loading standalone" \
  -m "Add a standalone probe that loads only the experimental MTP artifact, validates required MTP keys, and runs bounded synthetic MLX quantized matmuls without loading the full MiMo model." \
  -m "Constraint: Full MiMo runtime loading and distributed decode remain out of scope" \
  -m "Confidence: high" \
  -m "Scope-risk: moderate" \
  -m "Tested: uv run pytest scripts/test_mimo_v25_pro_mtp_module_probe.py -q; uv run ruff check scripts/mimo_v25_pro_mtp_module_probe.py scripts/test_mimo_v25_pro_mtp_module_probe.py; uv run basedpyright scripts/mimo_v25_pro_mtp_module_probe.py scripts/test_mimo_v25_pro_mtp_module_probe.py" \
  -m "Not-tested: Real experimental MTP artifact probe"
```

---

### Task 6: Run Real MTP-Only Module Probe

**Files:**
- Create: `docs/plans/artifacts/mimo-v25-pro-mtp-module-probe-20260519.json`
- Modify: `docs/plans/MIMO_V25_PRO_MTP_ARTIFACT_RUNLOG.md`

- [ ] **Step 1: Run the standalone probe against the real experimental artifact**

Run:

```bash
uv run python scripts/mimo_v25_pro_mtp_module_probe.py \
  --artifact /Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental \
  --json-out docs/plans/artifacts/mimo-v25-pro-mtp-module-probe-20260519.json
python3 - <<'PY'
import json
from pathlib import Path
report = json.loads(Path("docs/plans/artifacts/mimo-v25-pro-mtp-module-probe-20260519.json").read_text())
assert report["artifact_kind"] == "mimo-v25-pro-mtp-only"
assert report["layers"] == [0, 1, 2]
assert report["loaded_tensor_count"] > 48
assert report["synthetic_forwards"]["layer_0_eh_proj"] == [1, 1, 6144]
assert report["synthetic_forwards"]["layer_0_qkv_proj"] == [1, 1, 27136]
print(report["loaded_tensor_count"])
print(report["synthetic_forwards"]["layer_0_eh_proj"])
print(report["synthetic_forwards"]["layer_0_qkv_proj"])
PY
```

Expected:

- Probe exits `0`.
- Printed tensor count is greater than `48`.
- Printed `eh_proj` output shape is `[1, 1, 6144]`.
- Printed `qkv_proj` output shape is `[1, 1, 27136]`.

- [ ] **Step 2: Append real probe evidence to the runlog**

Modify `docs/plans/MIMO_V25_PRO_MTP_ARTIFACT_RUNLOG.md` by adding:

```markdown
## Standalone MTP-Only Module Probe

- Probe: `uv run python scripts/mimo_v25_pro_mtp_module_probe.py --artifact /Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental --json-out docs/plans/artifacts/mimo-v25-pro-mtp-module-probe-20260519.json`
  - Result: probe completed; JSON report written
- Report: `docs/plans/artifacts/mimo-v25-pro-mtp-module-probe-20260519.json`
  - Result: `loaded_tensor_count > 48`, `layer_0_eh_proj=[1, 1, 6144]`, `layer_0_qkv_proj=[1, 1, 27136]`

## Decode Boundary

The MTP-only artifact and standalone module probe passed. Full distributed MTP
decode remains blocked until a separate decode-controller plan covers
propose/verify/accept semantics and fallback behavior.
```

- [ ] **Step 3: Commit real probe evidence**

Run:

```bash
git add docs/plans/MIMO_V25_PRO_MTP_ARTIFACT_RUNLOG.md \
  docs/plans/artifacts/mimo-v25-pro-mtp-module-probe-20260519.json
git commit -m "Record MiMo MTP module probe evidence" \
  -m "Record standalone MTP-only module load evidence after validating the experimental artifact with synthetic MLX quantized matmuls." \
  -m "Constraint: Probe loaded only the MTP artifact, not the full MiMo model" \
  -m "Confidence: high" \
  -m "Scope-risk: narrow" \
  -m "Tested: real MTP module probe; probe report assertions" \
  -m "Not-tested: Distributed MTP decode"
```

---

### Task 7: Final Verification And Handoff

**Files:**
- Verify: `scripts/mimo_v25_pro_mtp_quantize.py`
- Verify: `scripts/mimo_v25_pro_mtp_module_probe.py`
- Verify: `docs/plans/MIMO_V25_PRO_MTP_ARTIFACT_RUNLOG.md`

- [ ] **Step 1: Run focused tests**

Run:

```bash
uv run pytest \
  scripts/test_mimo_v25_pro_mtp_quantize.py \
  scripts/test_mimo_v25_pro_mtp_module_probe.py \
  scripts/test_mimo_v25_pro_mtp_artifact_probe.py \
  -q
```

Expected:

```text
12 passed
```

The count is `6 + 3 + 3` if the tests in this plan are implemented exactly.

- [ ] **Step 2: Run lint**

Run:

```bash
uv run ruff check \
  scripts/mimo_v25_pro_mtp_quantize.py \
  scripts/test_mimo_v25_pro_mtp_quantize.py \
  scripts/mimo_v25_pro_mtp_module_probe.py \
  scripts/test_mimo_v25_pro_mtp_module_probe.py \
  scripts/mimo_v25_pro_mtp_artifact_probe.py \
  scripts/test_mimo_v25_pro_mtp_artifact_probe.py
```

Expected:

```text
All checks passed!
```

- [ ] **Step 3: Run typecheck**

Run:

```bash
uv run basedpyright \
  scripts/mimo_v25_pro_mtp_quantize.py \
  scripts/test_mimo_v25_pro_mtp_quantize.py \
  scripts/mimo_v25_pro_mtp_module_probe.py \
  scripts/test_mimo_v25_pro_mtp_module_probe.py \
  scripts/mimo_v25_pro_mtp_artifact_probe.py \
  scripts/test_mimo_v25_pro_mtp_artifact_probe.py
```

Expected:

```text
0 errors, 0 warnings, 0 notes
```

- [ ] **Step 4: Verify artifact summaries**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path
artifact = json.loads(Path("docs/plans/artifacts/mimo-v25-pro-mtp-artifact-summary-20260519.json").read_text())
probe = json.loads(Path("docs/plans/artifacts/mimo-v25-pro-mtp-module-probe-20260519.json").read_text())
assert artifact["complete"] is True
assert artifact["mtp_layers"] == [0, 1, 2]
assert artifact["output_shard_size"] > 0
assert probe["artifact_kind"] == "mimo-v25-pro-mtp-only"
assert probe["layers"] == [0, 1, 2]
assert probe["synthetic_forwards"]["layer_0_eh_proj"] == [1, 1, 6144]
assert probe["synthetic_forwards"]["layer_0_qkv_proj"] == [1, 1, 27136]
print("artifact-ok")
print("probe-ok")
PY
```

Expected:

```text
artifact-ok
probe-ok
```

- [ ] **Step 5: Confirm runtime and production artifact boundaries**

Run:

```bash
git diff --cached --name-only
git diff --name-only -- src/exo/api src/exo/master src/exo/shared/models resources/inference_model_cards
git diff --name-only -- \
  src/exo/worker/engines/mlx/cache.py \
  src/exo/worker/engines/mlx/generator/batch_generate.py \
  src/exo/worker/engines/mlx/generator/generate.py \
  src/exo/worker/tests/unittests/test_mlx/test_kv_prefix_cache.py
test ! -e /Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit/model_mtp-00001-of-00001.safetensors
git status --short --branch
```

Expected:

- No staged files before final runlog commit.
- No diffs in `src/exo/api`, `src/exo/master`, `src/exo/shared/models`, or `resources/inference_model_cards`.
- The worker diff command prints exactly the four pre-existing KV/cache dirty files listed in the command, and no additional worker files.
- Production artifact does not contain `model_mtp-00001-of-00001.safetensors`.
- Existing unrelated dirty KV/cache files remain visible and unstaged.

- [ ] **Step 6: Append final verification to the runlog**

Modify `docs/plans/MIMO_V25_PRO_MTP_ARTIFACT_RUNLOG.md` by adding:

```markdown
## Final Verification

- Focused tests: `uv run pytest scripts/test_mimo_v25_pro_mtp_quantize.py scripts/test_mimo_v25_pro_mtp_module_probe.py scripts/test_mimo_v25_pro_mtp_artifact_probe.py -q`
  - Result: `12 passed`
- Lint: `uv run ruff check scripts/mimo_v25_pro_mtp_quantize.py scripts/test_mimo_v25_pro_mtp_quantize.py scripts/mimo_v25_pro_mtp_module_probe.py scripts/test_mimo_v25_pro_mtp_module_probe.py scripts/mimo_v25_pro_mtp_artifact_probe.py scripts/test_mimo_v25_pro_mtp_artifact_probe.py`
  - Result: `All checks passed!`
- Typecheck: `uv run basedpyright scripts/mimo_v25_pro_mtp_quantize.py scripts/test_mimo_v25_pro_mtp_quantize.py scripts/mimo_v25_pro_mtp_module_probe.py scripts/test_mimo_v25_pro_mtp_module_probe.py scripts/mimo_v25_pro_mtp_artifact_probe.py scripts/test_mimo_v25_pro_mtp_artifact_probe.py`
  - Result: `0 errors, 0 warnings, 0 notes`
- Boundary check: production artifact unchanged; runtime source files not modified by this plan.
- Handoff: write the next plan for tiny fixture MTP decode-controller semantics before any distributed full-model run.
```

- [ ] **Step 7: Commit final verification**

Run:

```bash
git add docs/plans/MIMO_V25_PRO_MTP_ARTIFACT_RUNLOG.md
git commit -m "Record MiMo MTP artifact final verification" \
  -m "Record focused tests, lint, typecheck, artifact summary assertions, module probe assertions, and runtime boundary evidence for the MTP artifact phase." \
  -m "Confidence: high" \
  -m "Scope-risk: narrow" \
  -m "Tested: focused pytest, ruff, basedpyright, artifact summary assertions, module probe assertions, boundary checks" \
  -m "Not-tested: Full-model MTP decode"
```

---

## Handoff To Next Plan

After this plan passes, write a third plan for a tiny fixture MTP decode controller. That next plan should cover:

- A tiny synthetic main-model/MTP-module fixture.
- Proposal, verification, and longest-prefix acceptance semantics.
- Rejection fallback to one-token normal decode.
- Telemetry for proposed, accepted, rejected, emitted tokens, and fallback reason.

Do not plan distributed full-model MTP execution until the tiny fixture decode controller passes.
