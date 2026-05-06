#!/usr/bin/env python3
"""MiMo Track A rank-1 layer-load bisect wrapper.

This sidecar is intentionally narrow and offline. It never imports MLX, never
initializes mx.distributed/JACCL/libp2p/API/generation, and never calls exo
placement or generation endpoints. Real probing is performed only by shelling
out to scripts/mimo_track_a_rank1_load_probe.py, and only when --execute is
provided. By default it writes a dry-run manifest for safe tests/review.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

MODEL_ID: Final[str] = "XiaomiMiMo/MiMo-V2.5-Pro-6bit-MLX"
BASE_MODEL_ID: Final[str] = "XiaomiMiMo/MiMo-V2.5-Pro"
DEFAULT_MODEL_PATH: Final[Path] = Path(
    "/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/"
    "XiaomiMiMo--MiMo-V2.5-Pro-6bit-MLX"
)
DEFAULT_INVESTIGATION_DIR: Final[Path] = Path(
    "/Volumes/GLM5-NVMe/exo/mimo-v25-pro/investigations/rank1-load-probe"
)
DEFAULT_START_LAYER: Final[int] = 36
DEFAULT_MIN_END_LAYER: Final[int] = 60
DEFAULT_MAX_END_LAYER: Final[int] = 70
DEFAULT_DEVICE_RANK: Final[int] = 1
DEFAULT_WORLD_SIZE: Final[int] = 2
DEFAULT_PLAN_STYLE: Final[str] = "binary"
PROBE_SCRIPT: Final[Path] = Path("scripts/mimo_track_a_rank1_load_probe.py")


@dataclass(frozen=True)
class BisectConfig:
    model_path: Path
    start_layer: int
    min_end_layer: int
    max_end_layer: int
    device_rank: int
    world_size: int
    plan_style: str
    output_dir: Path
    manifest_path: Path | None
    execute: bool
    probe_script: Path
    python_executable: str
    no_eval: bool


@dataclass(frozen=True)
class PlannedAttempt:
    start_layer: int
    end_layer: int
    jsonl_path: Path
    stdout_path: Path
    stderr_path: Path


@dataclass(frozen=True)
class CompletedProcessLike:
    returncode: int
    stdout: str
    stderr: str


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def default_manifest_path(output_dir: Path) -> Path:
    return output_dir / f"mimo-track-a-rank1-bisect-manifest-{utc_timestamp()}.jsonl"


def parse_args(argv: Sequence[str] | None = None) -> BisectConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Offline/dry-run-safe MiMo Track A wrapper that bisects the "
            "rank-1 layer-load threshold by invoking the standalone load probe."
        )
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--start-layer", type=int, default=DEFAULT_START_LAYER)
    parser.add_argument("--min-end-layer", type=int, default=DEFAULT_MIN_END_LAYER)
    parser.add_argument("--max-end-layer", type=int, default=DEFAULT_MAX_END_LAYER)
    parser.add_argument("--device-rank", type=int, default=DEFAULT_DEVICE_RANK)
    parser.add_argument("--world-size", type=int, default=DEFAULT_WORLD_SIZE)
    parser.add_argument(
        "--plan-style",
        choices=("binary", "linear"),
        default=DEFAULT_PLAN_STYLE,
        help="Range plan. binary tests midpoint(s) then bounds; linear tests every end layer.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_INVESTIGATION_DIR)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--probe-script", type=Path, default=PROBE_SCRIPT)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run the standalone load probe. Without this flag only a dry-run manifest is written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicit alias for the default safe mode; heavy probing still requires --execute.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Forward --no-eval to the standalone probe during real execution.",
    )
    args = parser.parse_args(argv)
    config = BisectConfig(
        model_path=args.model_path,
        start_layer=args.start_layer,
        min_end_layer=args.min_end_layer,
        max_end_layer=args.max_end_layer,
        device_rank=args.device_rank,
        world_size=args.world_size,
        plan_style=args.plan_style,
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        execute=bool(args.execute),
        probe_script=args.probe_script,
        python_executable=str(args.python_executable),
        no_eval=bool(args.no_eval),
    )
    validate_config(config)
    return config


def validate_config(config: BisectConfig) -> None:
    if config.start_layer < 0:
        raise ValueError("start-layer must be >= 0")
    if config.min_end_layer <= config.start_layer:
        raise ValueError("min-end-layer must be greater than start-layer")
    if config.max_end_layer < config.min_end_layer:
        raise ValueError("max-end-layer must be >= min-end-layer")
    if config.device_rank < 0:
        raise ValueError("device-rank must be >= 0")
    if config.world_size <= 0:
        raise ValueError("world-size must be > 0")
    if config.device_rank >= config.world_size:
        raise ValueError("device-rank must be less than world-size")


def planned_end_layers(config: BisectConfig) -> list[int]:
    if config.plan_style == "linear":
        return list(range(config.min_end_layer, config.max_end_layer + 1))

    midpoint = (config.min_end_layer + config.max_end_layer) // 2
    candidates = [midpoint, config.min_end_layer, config.max_end_layer]
    # Add adjacent midpoint probes to make the threshold plan useful while still bounded.
    for delta in (-(config.max_end_layer - config.min_end_layer) // 4, (config.max_end_layer - config.min_end_layer) // 4):
        if delta:
            candidates.append(midpoint + delta)
    return sorted({end for end in candidates if config.min_end_layer <= end <= config.max_end_layer})


def make_attempts(config: BisectConfig, manifest_path: Path) -> list[PlannedAttempt]:
    stem = manifest_path.stem
    attempts: list[PlannedAttempt] = []
    for end_layer in planned_end_layers(config):
        prefix = f"{stem}-layers-{config.start_layer}-{end_layer}"
        attempts.append(
            PlannedAttempt(
                start_layer=config.start_layer,
                end_layer=end_layer,
                jsonl_path=config.output_dir / f"{prefix}.jsonl",
                stdout_path=config.output_dir / f"{prefix}.stdout.log",
                stderr_path=config.output_dir / f"{prefix}.stderr.log",
            )
        )
    return attempts


def command_for_attempt(config: BisectConfig, attempt: PlannedAttempt) -> list[str]:
    command = [
        config.python_executable,
        str(config.probe_script),
        "--model-path",
        str(config.model_path),
        "--start-layer",
        str(attempt.start_layer),
        "--end-layer",
        str(attempt.end_layer),
        "--device-rank",
        str(config.device_rank),
        "--world-size",
        str(config.world_size),
        "--output-path",
        str(attempt.jsonl_path),
    ]
    if config.no_eval:
        command.append("--no-eval")
    return command


def infer_signal(returncode: int) -> str | None:
    if returncode < 0:
        signal_number = -returncode
    elif returncode > 128:
        # Shells often encode signal death as 128 + signal number (e.g. 137 = SIGKILL).
        signal_number = returncode - 128
    else:
        return None
    try:
        return signal.Signals(signal_number).name
    except ValueError:
        return f"SIG{signal_number}"


def parse_last_jsonl_record(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    last_record: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                last_record = parsed
    return last_record


def write_text_log(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def append_manifest_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as manifest_file:
        manifest_file.write(json.dumps(record, sort_keys=True) + "\n")


def run_attempt(
    config: BisectConfig,
    attempt: PlannedAttempt,
    runner: Callable[..., CompletedProcessLike] = subprocess.run,
) -> dict[str, Any]:
    command = command_for_attempt(config, attempt)
    start = time.monotonic()
    completed = runner(command, text=True, capture_output=True, check=False)
    elapsed_seconds = time.monotonic() - start
    write_text_log(attempt.stdout_path, completed.stdout)
    write_text_log(attempt.stderr_path, completed.stderr)
    return {
        "type": "attempt",
        "mode": "execute",
        "model_id": MODEL_ID,
        "base_model_id": BASE_MODEL_ID,
        "start_layer": attempt.start_layer,
        "end_layer": attempt.end_layer,
        "command": command,
        "exit_code": completed.returncode,
        "signal": infer_signal(completed.returncode),
        "elapsed_seconds": elapsed_seconds,
        "stdout_log_path": str(attempt.stdout_path),
        "stderr_log_path": str(attempt.stderr_path),
        "jsonl_path": str(attempt.jsonl_path),
        "last_parsed_diagnostic_record": parse_last_jsonl_record(attempt.jsonl_path),
    }


def dry_run_record(config: BisectConfig, attempt: PlannedAttempt) -> dict[str, Any]:
    return {
        "type": "attempt",
        "mode": "dry-run",
        "model_id": MODEL_ID,
        "base_model_id": BASE_MODEL_ID,
        "start_layer": attempt.start_layer,
        "end_layer": attempt.end_layer,
        "command": command_for_attempt(config, attempt),
        "exit_code": None,
        "signal": None,
        "elapsed_seconds": 0.0,
        "stdout_log_path": str(attempt.stdout_path),
        "stderr_log_path": str(attempt.stderr_path),
        "jsonl_path": str(attempt.jsonl_path),
        "last_parsed_diagnostic_record": None,
    }


def header_record(config: BisectConfig, manifest_path: Path, attempts: Sequence[PlannedAttempt]) -> dict[str, Any]:
    return {
        "type": "manifest-header",
        "created_at": datetime.now(UTC).isoformat(),
        "model_id": MODEL_ID,
        "base_model_id": BASE_MODEL_ID,
        "observed_failure": "Studio1 rank-1 standalone layer-load probe signal-9 threshold near end_layer<=70 from start_layer=36",
        "safety_boundary": (
            "No exo restart, /place_instance, placement endpoint, generation endpoint, "
            "mx.distributed/JACCL/libp2p/API/generation initialization, or model download."
        ),
        "execute": config.execute,
        "plan_style": config.plan_style,
        "start_layer": config.start_layer,
        "min_end_layer": config.min_end_layer,
        "max_end_layer": config.max_end_layer,
        "device_rank": config.device_rank,
        "world_size": config.world_size,
        "manifest_path": str(manifest_path),
        "attempt_count": len(attempts),
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = parse_args(argv)
        manifest_path = config.manifest_path or default_manifest_path(config.output_dir)
        attempts = make_attempts(config, manifest_path)
        append_manifest_record(manifest_path, header_record(config, manifest_path, attempts))
        for attempt in attempts:
            record = run_attempt(config, attempt) if config.execute else dry_run_record(config, attempt)
            append_manifest_record(manifest_path, record)
        mode = "EXECUTE" if config.execute else "DRY-RUN"
        print(f"MiMo Track A rank-1 bisect probe {mode} manifest: {manifest_path}")
        if not config.execute:
            print("dry-run: standalone load probe was not launched; pass --execute for real Studio1 probing")
        return 0
    except Exception as exc:
        print(f"mimo rank-1 bisect probe failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
