#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from exo.worker.engines.mlx.mimo_mtp_fast.benchmark import (
    MimoMtpBenchmarkMode,
    build_contract_probe_row,
    build_metric_row,
    parse_benchmark_modes,
    render_json_line,
)


@dataclass(frozen=True, slots=True)
class _Args:
    model_path: Path | None
    sidecar_path: Path
    prompt: str
    max_tokens: int
    modes: str
    dry_run_contract_only: bool


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark or dry-run-probe the MiMo V2.5 Pro MTP fastpath."
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--sidecar-path", type=Path, required=True)
    parser.add_argument("--prompt", default="Write a Python function that parses JSON lines.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--modes", default="ar,d1,d2,d3,auto")
    parser.add_argument("--dry-run-contract-only", action="store_true")
    return parser


def _parse_args() -> _Args:
    namespace = _parser().parse_args()
    model_path = cast(Path | None, namespace.model_path)
    sidecar_path = cast(Path, namespace.sidecar_path)
    prompt = cast(str, namespace.prompt)
    max_tokens = cast(int, namespace.max_tokens)
    modes = cast(str, namespace.modes)
    dry_run_contract_only = cast(bool, namespace.dry_run_contract_only)
    return _Args(
        model_path=model_path,
        sidecar_path=sidecar_path,
        prompt=prompt,
        max_tokens=max_tokens,
        modes=modes,
        dry_run_contract_only=dry_run_contract_only,
    )


def main() -> int:
    args = _parse_args()
    if args.dry_run_contract_only:
        print(
            render_json_line(
                build_contract_probe_row(
                    sidecar_path=args.sidecar_path,
                    model_path=args.model_path,
                )
            )
        )
        return 0

    modes = parse_benchmark_modes(args.modes)
    ar_baseline_tok_s: float | None = None
    for mode in modes:
        if mode != MimoMtpBenchmarkMode.AR:
            row = build_metric_row(
                mode=mode,
                generated_tokens=0,
                decode_seconds=0.0,
                attempted_depth_counts={},
                accepted_depth_counts={},
                ar_baseline_tok_s=ar_baseline_tok_s,
            )
            row["next_step"] = (
                "full-model benchmark execution is not wired yet; implement MLX-backed AR/MTP runners next"
            )
            print(render_json_line(row))
            continue
        row = build_metric_row(
            mode=mode,
            generated_tokens=0,
            decode_seconds=0.0,
            attempted_depth_counts={},
            accepted_depth_counts={},
            ar_baseline_tok_s=None,
        )
        row["next_step"] = (
            "full-model AR baseline runner is not wired yet; implement MLX-backed AR/MTP runners next"
        )
        print(render_json_line(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
