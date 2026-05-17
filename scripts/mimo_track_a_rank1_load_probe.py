#!/usr/bin/env python3
"""Offline MiMo Track A rank-1 shard load probe.

This script is intentionally narrow: it loads/evaluates only the requested local
MiMo V2.5 Pro custom 6-bit MLX layer interval and records diagnostics. It does
not initialize mx.distributed, JACCL, libp2p, exo services, placement, API calls,
barriers, or generation.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

MODEL_ID: Final[str] = "XiaomiMiMo/MiMo-V2.5-Pro-6bit-MLX"
BASE_MODEL_ID: Final[str] = "XiaomiMiMo/MiMo-V2.5-Pro"
DEFAULT_MODEL_PATH: Final[Path] = Path(
    "/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/XiaomiMiMo--MiMo-V2.5-Pro-6bit-MLX"
)
DEFAULT_INVESTIGATION_DIR: Final[Path] = Path(
    "/Volumes/GLM5-NVMe/exo/mimo-v25-pro/investigations/rank1-load-probe"
)
DEFAULT_START_LAYER: Final[int] = 36
DEFAULT_END_LAYER: Final[int] = 70
DEFAULT_DEVICE_RANK: Final[int] = 1
DEFAULT_WORLD_SIZE: Final[int] = 2
DEFAULT_N_LAYERS: Final[int] = 70
EXPECTED_ARCHITECTURE: Final[str] = "MiMoV2ForCausalLM"
EXPECTED_MODEL_TYPE: Final[str] = "mimo_v2"


@dataclass(frozen=True)
class ProbeConfig:
    model_path: Path
    start_layer: int
    end_layer: int
    device_rank: int
    world_size: int
    output_path: Path | None
    dry_run: bool
    no_eval: bool


@dataclass(frozen=True)
class ModelMetadata:
    model_id: str
    base_model_id: str
    architecture: str | None
    model_type: str | None
    n_layers: int
    config_path: Path


def parse_args(argv: Sequence[str] | None = None) -> ProbeConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Offline MiMo Track A rank-1 shard load probe. Defaults target "
            "layers 36..69 of XiaomiMiMo/MiMo-V2.5-Pro-6bit-MLX."
        )
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Local MLX model directory (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--start-layer",
        type=int,
        default=DEFAULT_START_LAYER,
        help="Inclusive global start layer for the local pipeline-style shard.",
    )
    parser.add_argument(
        "--end-layer",
        type=int,
        default=DEFAULT_END_LAYER,
        help="Exclusive global end layer for the local pipeline-style shard.",
    )
    parser.add_argument(
        "--device-rank",
        type=int,
        default=DEFAULT_DEVICE_RANK,
        help="Pipeline device rank to record in metadata only; no distributed init is performed.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=DEFAULT_WORLD_SIZE,
        help="Pipeline world size to record in metadata only; no distributed init is performed.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "JSONL diagnostics path. Defaults to a timestamped file under "
            f"{DEFAULT_INVESTIGATION_DIR}."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate model path/config/shard metadata and print the planned layer range without loading weights.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Load and slice the model but skip per-layer mx.eval calls. Implied by --dry-run.",
    )
    args = parser.parse_args(argv)
    dry_run = bool(args.dry_run)
    return ProbeConfig(
        model_path=args.model_path,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        device_rank=args.device_rank,
        world_size=args.world_size,
        output_path=args.output_path,
        dry_run=dry_run,
        no_eval=bool(args.no_eval) or dry_run,
    )


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def iso_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def default_output_path() -> Path:
    return (
        DEFAULT_INVESTIGATION_DIR
        / f"mimo-track-a-rank1-load-probe-{utc_timestamp()}.jsonl"
    )


def read_config_json(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing model config: {config_path}")
    with config_path.open("r", encoding="utf-8") as config_file:
        parsed = json.load(config_file)
    if not isinstance(parsed, dict):
        raise ValueError(f"config must be a JSON object: {config_path}")
    return parsed


def _extract_n_layers(config: dict[str, Any]) -> int:
    for key in (
        "num_hidden_layers",
        "num_layers",
        "n_layer",
        "n_layers",
        "num_decoder_layers",
        "decoder_layers",
    ):
        value = config.get(key)
        if isinstance(value, int):
            return value
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        return _extract_n_layers(text_config)
    return DEFAULT_N_LAYERS


def _extract_first_string(config: dict[str, Any], key: str) -> str | None:
    value = config.get(key)
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and isinstance(value[0], str):
        return value[0]
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        return _extract_first_string(text_config, key)
    return None


def validate_model_metadata(model_path: Path) -> ModelMetadata:
    if not model_path.exists():
        raise FileNotFoundError(f"model path does not exist: {model_path}")
    if not model_path.is_dir():
        raise NotADirectoryError(f"model path is not a directory: {model_path}")

    config = read_config_json(model_path)
    architecture = _extract_first_string(config, "architectures")
    model_type = _extract_first_string(config, "model_type")
    n_layers = _extract_n_layers(config)

    if architecture not in {EXPECTED_ARCHITECTURE, None}:
        raise ValueError(
            f"unexpected architecture {architecture!r}; expected {EXPECTED_ARCHITECTURE!r}"
        )
    if model_type not in {EXPECTED_MODEL_TYPE, None}:
        raise ValueError(
            f"unexpected model_type {model_type!r}; expected {EXPECTED_MODEL_TYPE!r}"
        )
    if n_layers <= 0:
        raise ValueError(f"invalid layer count: {n_layers}")

    return ModelMetadata(
        model_id=MODEL_ID,
        base_model_id=BASE_MODEL_ID,
        architecture=architecture,
        model_type=model_type,
        n_layers=n_layers,
        config_path=model_path / "config.json",
    )


def validate_shard_range(config: ProbeConfig, metadata: ModelMetadata) -> None:
    if config.start_layer < 0:
        raise ValueError("start-layer must be >= 0")
    if config.end_layer <= config.start_layer:
        raise ValueError("end-layer must be greater than start-layer")
    if config.end_layer > metadata.n_layers:
        raise ValueError(
            f"end-layer {config.end_layer} exceeds model layer count {metadata.n_layers}"
        )
    if config.device_rank < 0:
        raise ValueError("device-rank must be >= 0")
    if config.world_size <= 0:
        raise ValueError("world-size must be > 0")
    if config.device_rank >= config.world_size:
        raise ValueError("device-rank must be less than world-size")


def planned_layers(config: ProbeConfig) -> list[int]:
    return list(range(config.start_layer, config.end_layer))


def _collect_process_memory() -> dict[str, Any]:
    memory: dict[str, Any] = {"pid": os.getpid()}
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        memory["ru_maxrss"] = usage.ru_maxrss
    except Exception as exc:  # pragma: no cover - platform-specific defensive path
        memory["ru_maxrss_error"] = repr(exc)

    try:
        ps_output = subprocess.check_output(
            ["ps", "-o", "rss=,vsz=", "-p", str(os.getpid())],
            text=True,
            timeout=1,
        ).strip()
        if ps_output:
            rss_kb, vsz_kb = ps_output.split()[:2]
            memory["rss_kb"] = int(rss_kb)
            memory["vsz_kb"] = int(vsz_kb)
    except Exception as exc:  # pragma: no cover - platform-specific defensive path
        memory["ps_error"] = repr(exc)
    return memory


def _collect_mlx_memory(mx_module: Any) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for name in ("active", "cache", "peak"):
        try:
            stats[f"{name}_bytes"] = getattr(mx_module, f"get_{name}_memory")()
        except Exception as exc:  # pragma: no cover - MLX-version defensive path
            stats[f"{name}_error"] = repr(exc)
    return stats


def _collect_device_memory(mx_module: Any) -> dict[str, Any]:
    try:
        device_info = dict(mx_module.device_info())
    except Exception as exc:  # pragma: no cover - MLX-version defensive path
        return {"error": repr(exc)}

    fields: dict[str, Any] = {}
    for key in (
        "resource_limit",
        "memory_size",
        "max_recommended_working_set_size",
        "architecture",
    ):
        if key in device_info:
            fields[key] = device_info[key]
    return fields


def make_diagnostic_record(
    *,
    config: ProbeConfig,
    metadata: ModelMetadata,
    output_path: Path,
    stage: str,
    global_layer_index: int | None,
    local_layer_ordinal: int | None,
    mx_module: Any | None,
    error: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "timestamp": iso_timestamp(),
        "stage": stage,
        "model_id": metadata.model_id,
        "base_model_id": metadata.base_model_id,
        "model_path": str(config.model_path.resolve()),
        "output_path": str(output_path),
        "start_layer": config.start_layer,
        "end_layer": config.end_layer,
        "device_rank": config.device_rank,
        "world_size": config.world_size,
        "shard_metadata_type": "PipelineShardMetadata",
        "global_layer_index": global_layer_index,
        "local_layer_ordinal": local_layer_ordinal,
        "process_memory": _collect_process_memory(),
    }
    if mx_module is not None:
        record["mlx_memory"] = _collect_mlx_memory(mx_module)
        record["device_memory"] = _collect_device_memory(mx_module)
    if error is not None:
        record["error"] = error
    return record


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, sort_keys=True) + "\n")


def print_plan(config: ProbeConfig, metadata: ModelMetadata, output_path: Path) -> None:
    layers = planned_layers(config)
    print("MiMo Track A rank-1 load probe plan")
    print(f"  model_id: {metadata.model_id}")
    print(f"  base_model_id: {metadata.base_model_id}")
    print(f"  model_path: {config.model_path.resolve()}")
    print(f"  config_path: {metadata.config_path}")
    print(f"  architecture: {metadata.architecture}")
    print(f"  model_type: {metadata.model_type}")
    print(f"  n_layers: {metadata.n_layers}")
    print(
        "  pipeline-style shard: "
        f"rank {config.device_rank}/{config.world_size}, "
        f"global layers [{config.start_layer}, {config.end_layer})"
    )
    print(f"  planned_layers: {layers[0]}..{layers[-1]} ({len(layers)} layers)")
    print(f"  output_path: {output_path}")
    print("  distributed/JACCL/libp2p/API/generation: disabled")


def run_probe(config: ProbeConfig, metadata: ModelMetadata, output_path: Path) -> None:
    # Importing utils_mlx is deliberate: its module import registers the MiMo
    # mimo_v2 -> mimo_v2_flash MLX-LM alias and patches the MiMo sanitizer so
    # custom 6-bit qkv/quant key normalization is active before load_model().
    import mlx.core as mx
    import mlx.nn as nn

    from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
    from exo.shared.types.memory import Memory
    from exo.shared.types.worker.shards import PipelineShardMetadata
    from exo.worker.engines.mlx import utils_mlx as exo_mlx_utils
    from exo.worker.engines.mlx.auto_parallel import get_inner_model, get_layers

    local_shard_metadata = PipelineShardMetadata(
        model_card=ModelCard(
            model_id=ModelId(metadata.model_id),
            storage_size=Memory.from_bytes(0),
            n_layers=metadata.n_layers,
            hidden_size=1,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
            architecture=metadata.architecture or EXPECTED_ARCHITECTURE,
            family="mimo",
            quantization="6bit-mlx-affine",
            base_model=metadata.base_model_id,
        ),
        device_rank=config.device_rank,
        world_size=config.world_size,
        start_layer=config.start_layer,
        end_layer=config.end_layer,
        n_layers=metadata.n_layers,
    )

    append_jsonl(
        output_path,
        make_diagnostic_record(
            config=config,
            metadata=metadata,
            output_path=output_path,
            stage=f"before-load-model:{type(local_shard_metadata).__name__}",
            global_layer_index=None,
            local_layer_ordinal=None,
            mx_module=mx,
        ),
    )

    model, _ = exo_mlx_utils.load_model(config.model_path, lazy=True, strict=False)
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"expected mlx.nn.Module from load_model, got {type(model).__name__}"
        )

    inner_model = get_inner_model(model)
    layers = get_layers(inner_model)
    if len(layers) < metadata.n_layers:
        raise ValueError(
            f"loaded model exposes only {len(layers)} layers; metadata expected {metadata.n_layers}"
        )
    shard_layers = layers[config.start_layer : config.end_layer]

    append_jsonl(
        output_path,
        make_diagnostic_record(
            config=config,
            metadata=metadata,
            output_path=output_path,
            stage="after-load-model-before-layer-eval",
            global_layer_index=None,
            local_layer_ordinal=None,
            mx_module=mx,
        ),
    )

    for local_ordinal, layer in enumerate(shard_layers):
        global_index = config.start_layer + local_ordinal
        stage = "layer-sliced-no-eval" if config.no_eval else "after-layer-eval"
        try:
            if not config.no_eval:
                mx.eval(layer)
            append_jsonl(
                output_path,
                make_diagnostic_record(
                    config=config,
                    metadata=metadata,
                    output_path=output_path,
                    stage=stage,
                    global_layer_index=global_index,
                    local_layer_ordinal=local_ordinal,
                    mx_module=mx,
                ),
            )
            print(
                f"recorded {stage}: global_layer={global_index} local={local_ordinal}"
            )
        except BaseException as exc:
            append_jsonl(
                output_path,
                make_diagnostic_record(
                    config=config,
                    metadata=metadata,
                    output_path=output_path,
                    stage="layer-error",
                    global_layer_index=global_index,
                    local_layer_ordinal=local_ordinal,
                    mx_module=mx,
                    error=repr(exc),
                ),
            )
            raise

    append_jsonl(
        output_path,
        make_diagnostic_record(
            config=config,
            metadata=metadata,
            output_path=output_path,
            stage="complete",
            global_layer_index=None,
            local_layer_ordinal=None,
            mx_module=mx,
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(argv)
    output_path = config.output_path or default_output_path()
    try:
        metadata = validate_model_metadata(config.model_path)
        validate_shard_range(config, metadata)
        print_plan(config, metadata, output_path)
        if config.dry_run:
            print("dry-run: model weights were not loaded or evaluated")
            return 0
        run_probe(config, metadata, output_path)
        print(f"diagnostics written to {output_path}")
        return 0
    except Exception as exc:
        print(f"mimo rank-1 load probe failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
