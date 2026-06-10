"""Export GLM 5.1's multi-token-prediction (nextn) layer as a quantized sidecar.

mlx_lm's deepseek_v32 sanitize() drops layers >= num_hidden_layers, so the
mlx-community 6-bit GLM 5.1 export contains no MTP weights despite
num_nextn_predict_layers=1. This script rebuilds them from the upstream bf16
checkpoint (zai-org/GLM-5.1, model.layers.78.*, 4 of 282 shards) applying the
same transforms sanitize() applies to full layers:

  1. stack the 256 routed experts into switch_mlp tensors
  2. split kv_b_proj into embed_q / unembed_out (MLA absorption)

then quantizes to match the served base model's per-tensor rules (8-bit g64
for attention/indexer/gate_proj/shared_experts, 6-bit g64 for switch_mlp
up/down and eh_proj; router gate and norms stay bf16).

Output: one safetensors file with names `mtp.layers.0.<mlx_lm-relative-path>`
(.scales/.biases as sibling tensors, mlx_lm convention — note this differs
from the MiMo sidecar's `<name>.weight.scales` nesting) plus an export
manifest JSON.

Usage:
  uv run python scripts/glm51_mtp_sidecar_export.py \
      [--source-dir DIR] [--base-config PATH] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Final

import mlx.core as mx

DEFAULT_SOURCE_DIR: Final[Path] = Path("/Volumes/GLM5-NVMe/exo/glm51-mtp/upstream")
DEFAULT_BASE_CONFIG: Final[Path] = Path(
    "~/.exo/models/mlx-community--GLM-5.1-8b-crit-6b-exp/config.json"
).expanduser()
DEFAULT_OUTPUT_DIR: Final[Path] = Path(
    "/Volumes/GLM5-NVMe/exo/glm51-mtp/sidecar-6bit-20260610"
)

NEXTN_LAYER_INDEX: Final[int] = 78
NEXTN_PREFIX: Final[str] = f"model.layers.{NEXTN_LAYER_INDEX}."
SIDECAR_PREFIX: Final[str] = "mtp.layers.0."
GROUP_SIZE: Final[int] = 64

# Per-suffix quantization bits, mirroring the base model's rules for its last
# MoE layer (layer 77). Suffixes not listed and not matching _DEFAULT_6BIT
# stay unquantized (router gate, e_score_correction_bias, all norms).
_8BIT_SUFFIXES: Final[tuple[str, ...]] = (
    "self_attn.q_a_proj",
    "self_attn.q_b_proj",
    "self_attn.kv_a_proj_with_mqa",
    "self_attn.o_proj",
    "self_attn.embed_q",
    "self_attn.unembed_out",
    "self_attn.indexer.wq_b",
    "self_attn.indexer.wk",
    "self_attn.indexer.weights_proj",
    "mlp.switch_mlp.gate_proj",
    "mlp.shared_experts.gate_proj",
    "mlp.shared_experts.up_proj",
    "mlp.shared_experts.down_proj",
)
_6BIT_SUFFIXES: Final[tuple[str, ...]] = (
    "mlp.switch_mlp.up_proj",
    "mlp.switch_mlp.down_proj",
    "eh_proj",
)


def _quantization_bits(relative_name: str) -> int | None:
    if not relative_name.endswith(".weight"):
        return None
    module_path = relative_name.removesuffix(".weight")
    if module_path in _8BIT_SUFFIXES:
        return 8
    if module_path in _6BIT_SUFFIXES:
        return 6
    return None


def _load_nextn_tensors(source_dir: Path) -> dict[str, mx.array]:
    index = json.loads((source_dir / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    shard_to_names: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if name.startswith(NEXTN_PREFIX):
            shard_to_names.setdefault(shard, []).append(name)
    if not shard_to_names:
        raise RuntimeError(f"no {NEXTN_PREFIX}* tensors in index")
    tensors: dict[str, mx.array] = {}
    for shard, names in sorted(shard_to_names.items()):
        shard_path = source_dir / shard
        if not shard_path.exists():
            raise FileNotFoundError(f"missing shard {shard_path}")
        loaded = mx.load(str(shard_path))
        for name in names:
            tensors[name.removeprefix(NEXTN_PREFIX)] = loaded[name]
        del loaded
        print(f"  {shard}: {len(names)} tensors")
    return tensors


def _stack_experts(tensors: dict[str, mx.array], num_experts: int) -> None:
    for projection in ("gate_proj", "up_proj", "down_proj"):
        first = f"mlp.experts.0.{projection}.weight"
        if first not in tensors:
            raise RuntimeError(f"missing {first}")
        stacked = mx.stack(
            [
                tensors.pop(f"mlp.experts.{e}.{projection}.weight")
                for e in range(num_experts)
            ]
        )
        tensors[f"mlp.switch_mlp.{projection}.weight"] = stacked
        mx.eval(stacked)


def _split_kv_b_proj(
    tensors: dict[str, mx.array],
    *,
    num_heads: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
) -> None:
    kv_b = tensors.pop("self_attn.kv_b_proj.weight")
    head_dim = qk_nope_head_dim + v_head_dim
    reshaped = kv_b.reshape(num_heads, head_dim, -1)
    embed_q = mx.contiguous(reshaped[:, :qk_nope_head_dim, :].swapaxes(-1, -2))
    unembed_out = mx.contiguous(reshaped[:, qk_nope_head_dim:, :])
    tensors["self_attn.embed_q.weight"] = embed_q
    tensors["self_attn.unembed_out.weight"] = unembed_out
    mx.eval(embed_q, unembed_out)


def export(source_dir: Path, base_config_path: Path, output_dir: Path) -> Path:
    base_config = json.loads(base_config_path.read_text())
    if int(base_config.get("num_nextn_predict_layers", 0)) != 1:
        raise RuntimeError("base config does not declare exactly one nextn layer")
    num_experts = int(base_config["n_routed_experts"])
    num_heads = int(base_config["num_attention_heads"])
    qk_nope_head_dim = int(base_config["qk_nope_head_dim"])
    v_head_dim = int(base_config["v_head_dim"])

    print(f"loading nextn tensors from {source_dir} ...")
    tensors = _load_nextn_tensors(source_dir)
    print(f"{len(tensors)} tensors loaded; stacking {num_experts} experts ...")
    _stack_experts(tensors, num_experts)
    print("splitting kv_b_proj into embed_q / unembed_out ...")
    _split_kv_b_proj(
        tensors,
        num_heads=num_heads,
        qk_nope_head_dim=qk_nope_head_dim,
        v_head_dim=v_head_dim,
    )

    output: dict[str, mx.array] = {}
    quantization_report: dict[str, str] = {}
    for relative_name in sorted(tensors):
        tensor = tensors[relative_name]
        bits = _quantization_bits(relative_name)
        sidecar_name = SIDECAR_PREFIX + relative_name
        if bits is None:
            output[sidecar_name] = tensor
            quantization_report[relative_name] = f"raw {tensor.dtype}"
            continue
        weight, scales, biases = mx.quantize(
            tensor, bits=bits, group_size=GROUP_SIZE
        )
        base = sidecar_name.removesuffix(".weight")
        output[base + ".weight"] = weight
        output[base + ".scales"] = scales
        output[base + ".biases"] = biases
        quantization_report[relative_name] = f"{bits}-bit g{GROUP_SIZE} affine"
    mx.eval(*output.values())

    output_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = output_dir / "model_mtp-00001-of-00001.safetensors"
    mx.save_safetensors(str(sidecar_path), output)
    total_bytes = sidecar_path.stat().st_size
    manifest = {
        "source_repo": "zai-org/GLM-5.1",
        "source_layer": NEXTN_LAYER_INDEX,
        "base_model": "mlx-community/GLM-5.1-8b-crit-6b-exp",
        "sidecar_prefix": SIDECAR_PREFIX,
        "naming_convention": "mlx_lm (.scales/.biases siblings)",
        "group_size": GROUP_SIZE,
        "tensor_count": len(output),
        "total_bytes": total_bytes,
        "quantization": quantization_report,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {sidecar_path} ({total_bytes / 1e9:.2f} GB, {len(output)} tensors)")
    return sidecar_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args()
    export(arguments.source_dir, arguments.base_config, arguments.output_dir)


if __name__ == "__main__":
    main()
