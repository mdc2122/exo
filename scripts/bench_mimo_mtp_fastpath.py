#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import mlx.core as mx

from exo.download.download_utils import build_model_path
from exo.shared.types.common import ModelId
from exo.shared.types.mlx import Model
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.worker.engines.mlx.cache import encode_prompt, make_kv_cache
from exo.worker.engines.mlx.generator.generate import mlx_generate
from exo.worker.engines.mlx.mimo_mtp_fast.benchmark import (
    ArBenchmarkRequest,
    MimoMtpBenchmarkMode,
    build_ar_task_params,
    build_benchmark_runner,
    build_benchmark_stage_row,
    build_contract_probe_row,
    parse_benchmark_modes,
    render_json_line,
    run_ar_benchmark,
    run_benchmark_modes,
    run_mtp_benchmark,
)
from exo.worker.engines.mlx.mimo_mtp_fast.providers import (
    TargetVerifierModel,
    make_replay_mtp_one_cycle_runner,
)
from exo.worker.engines.mlx.mimo_mtp_fast.sidecar_contract import probe_mimo_mtp_sidecar
from exo.worker.engines.mlx.mimo_mtp_fast.sidecar_loader import (
    load_mimo_mtp_sidecar_tensors,
)
from exo.worker.engines.mlx.mimo_mtp_fast.sidecar_module import build_mimo_mtp_stack
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    load_model,
    load_tokenizer_for_model_id,
)

PromptBuilder = Callable[[object, TextGenerationTaskParams], str]
StageStatus = Literal["started", "completed", "failed"]


@dataclass(frozen=True, slots=True)
class _Args:
    model_path: Path | None
    model_id: str
    sidecar_path: Path
    prompt: str
    max_tokens: int
    modes: str
    dry_run_contract_only: bool
    preflight_only: bool
    load_only: bool


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark or dry-run-probe the MiMo V2.5 Pro MTP fastpath."
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--model-id", default="kernelpool/MiMo-V2.5-Pro-6bit")
    parser.add_argument("--sidecar-path", type=Path, required=True)
    parser.add_argument(
        "--prompt", default="Write a Python function that parses JSON lines."
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--modes", default="ar,d1,d2,d3,auto")
    parser.add_argument("--dry-run-contract-only", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate paths, modes, and sidecar contract without loading MiMo tensors.",
    )
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="Load the base model, tokenizer, sidecar, stack, and prompt tokens, then exit before generation.",
    )
    return parser


def _parse_args(argv: list[str] | None = None) -> _Args:
    namespace = _parser().parse_args(argv)
    model_path = cast(Path | None, namespace.model_path)
    model_id = cast(str, namespace.model_id)
    sidecar_path = cast(Path, namespace.sidecar_path)
    prompt = cast(str, namespace.prompt)
    max_tokens = cast(int, namespace.max_tokens)
    modes = cast(str, namespace.modes)
    dry_run_contract_only = cast(bool, namespace.dry_run_contract_only)
    preflight_only = cast(bool, namespace.preflight_only)
    load_only = cast(bool, namespace.load_only)
    return _Args(
        model_path=model_path,
        model_id=model_id,
        sidecar_path=sidecar_path,
        prompt=prompt,
        max_tokens=max_tokens,
        modes=modes,
        dry_run_contract_only=dry_run_contract_only,
        preflight_only=preflight_only,
        load_only=load_only,
    )


def _validation_error_row(
    *,
    field: str,
    path: Path,
    error: str,
    next_step: str,
    sidecar_status: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "kind": "validation_error",
        "field": field,
        "path": str(path),
        "error": error,
        "next_step": next_step,
    }
    if sidecar_status is not None:
        row["sidecar_status"] = sidecar_status
    return row


def _print_row(row: dict[str, object]) -> None:
    print(render_json_line(row), flush=True)


def _print_stage(
    *,
    stage: str,
    status: StageStatus,
    started_at: float | None = None,
    details: dict[str, object] | None = None,
) -> None:
    elapsed_seconds = (
        None if started_at is None else max(0.0, time.perf_counter() - started_at)
    )
    _print_row(
        build_benchmark_stage_row(
            stage=stage,
            status=status,
            elapsed_seconds=elapsed_seconds,
            details=details,
        )
    )


def _validate_model_path(model_path: Path) -> int | None:
    started_at = time.perf_counter()
    if model_path.exists():
        _print_stage(
            stage="model_path_validated",
            status="completed",
            started_at=started_at,
            details={"model_path": str(model_path)},
        )
        return None
    _print_stage(
        stage="model_path_validated",
        status="failed",
        started_at=started_at,
        details={"model_path": str(model_path)},
    )
    _print_row(
        _validation_error_row(
            field="model_path",
            path=model_path,
            error="MiMo MTP benchmark model path does not exist",
            next_step="provide an existing local MiMo model snapshot path or use --dry-run-contract-only",
        )
    )
    return 2


def _validate_sidecar_path(sidecar_path: Path) -> int | None:
    started_at = time.perf_counter()
    sidecar_probe = probe_mimo_mtp_sidecar(sidecar_path)
    if sidecar_probe.ready:
        _print_stage(
            stage="sidecar_contract_validated",
            status="completed",
            started_at=started_at,
            details={
                "sidecar_path": str(sidecar_probe.path),
                "sidecar_status": sidecar_probe.status,
                "sidecar_layer_count": sidecar_probe.layer_count,
            },
        )
        return None
    _print_stage(
        stage="sidecar_contract_validated",
        status="failed",
        started_at=started_at,
        details={
            "sidecar_path": str(sidecar_probe.path),
            "sidecar_status": sidecar_probe.status,
        },
    )
    _print_row(
        _validation_error_row(
            field="sidecar_path",
            path=sidecar_probe.path,
            error="MiMo MTP sidecar is not ready",
            next_step="run --dry-run-contract-only or provide a valid official-layout sidecar",
            sidecar_status=sidecar_probe.status,
        )
    )
    return 2


def _parse_modes_or_print_error(
    raw_modes: str,
) -> tuple[MimoMtpBenchmarkMode, ...] | int:
    started_at = time.perf_counter()
    try:
        modes = parse_benchmark_modes(raw_modes)
    except ValueError as exc:
        _print_stage(
            stage="modes_validated",
            status="failed",
            started_at=started_at,
            details={"modes": raw_modes, "error": str(exc)},
        )
        _print_row(
            {
                "kind": "validation_error",
                "field": "modes",
                "path": "",
                "error": str(exc),
                "next_step": "use a comma-separated subset of ar,d1,d2,d3,auto",
            }
        )
        return 2
    _print_stage(
        stage="modes_validated",
        status="completed",
        started_at=started_at,
        details={"modes": [mode.value for mode in modes]},
    )
    return modes


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
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

    parsed_modes = _parse_modes_or_print_error(args.modes)
    if isinstance(parsed_modes, int):
        return parsed_modes
    modes = parsed_modes
    model_id = ModelId(args.model_id)
    model_path = args.model_path or build_model_path(model_id)
    model_validation_exit_code = _validate_model_path(model_path)
    if model_validation_exit_code is not None:
        return model_validation_exit_code
    sidecar_validation_exit_code = _validate_sidecar_path(args.sidecar_path)
    if sidecar_validation_exit_code is not None:
        return sidecar_validation_exit_code
    if args.preflight_only:
        _print_row(
            {
                "kind": "benchmark_preflight",
                "ready": True,
                "model_path": str(model_path),
                "sidecar_path": str(args.sidecar_path),
                "modes": [mode.value for mode in modes],
                "next_step": "run --load-only to test model materialization before generation",
            }
        )
        return 0

    started_at = time.perf_counter()
    _print_stage(
        stage="base_model_materialization",
        status="started",
        details={"model_path": str(model_path), "lazy": True, "strict": False},
    )
    raw_model, _config = load_model(model_path, lazy=True, strict=False)
    model = cast(Model, raw_model)
    target_model = cast(TargetVerifierModel, cast(object, raw_model))
    mx.eval(raw_model)
    _print_stage(
        stage="base_model_materialization",
        status="completed",
        started_at=started_at,
        details={"model_path": str(model_path)},
    )

    started_at = time.perf_counter()
    _print_stage(stage="tokenizer_load", status="started")
    tokenizer = load_tokenizer_for_model_id(model_id, model_path)
    _print_stage(stage="tokenizer_load", status="completed", started_at=started_at)

    started_at = time.perf_counter()
    _print_stage(
        stage="sidecar_tensor_load",
        status="started",
        details={"sidecar_path": str(args.sidecar_path)},
    )
    sidecar = load_mimo_mtp_sidecar_tensors(args.sidecar_path)
    _print_stage(
        stage="sidecar_tensor_load",
        status="completed",
        started_at=started_at,
        details={"sidecar_path": str(args.sidecar_path)},
    )

    started_at = time.perf_counter()
    _print_stage(stage="sidecar_stack_build", status="started")
    stack = build_mimo_mtp_stack(sidecar, raw_model)
    _print_stage(stage="sidecar_stack_build", status="completed", started_at=started_at)

    prompt_builder = cast(PromptBuilder, apply_chat_template)
    benchmark_request = ArBenchmarkRequest(
        model=model,
        tokenizer=tokenizer,
        model_id=str(model_id),
        prompt=args.prompt,
        max_tokens=args.max_tokens,
    )
    started_at = time.perf_counter()
    _print_stage(stage="prompt_tokenization", status="started")
    prompt_for_tokens = apply_chat_template(
        tokenizer,
        build_ar_task_params(benchmark_request),
    )
    flattened_prompt_tokens = encode_prompt(tokenizer, prompt_for_tokens).reshape(-1)
    prompt_token_history = tuple(
        int(flattened_prompt_tokens[index].item())
        for index in range(int(flattened_prompt_tokens.size))
    )
    _print_stage(
        stage="prompt_tokenization",
        status="completed",
        started_at=started_at,
        details={"prompt_token_count": len(prompt_token_history)},
    )
    if args.load_only:
        _print_row(
            {
                "kind": "benchmark_load",
                "ready": True,
                "model_path": str(model_path),
                "sidecar_path": str(args.sidecar_path),
                "prompt_token_count": len(prompt_token_history),
                "next_step": "run a minimal live AR row with --modes ar --max-tokens 1",
            }
        )
        return 0

    runner = build_benchmark_runner(
        model=model,
        tokenizer=tokenizer,
        model_id=str(model_id),
        prompt=args.prompt,
        prompt_token_history=prompt_token_history,
        max_tokens=args.max_tokens,
        one_cycle=make_replay_mtp_one_cycle_runner(
            stack=stack,
            target_model=target_model,
            target_cache_factory=lambda: make_kv_cache(model=model),
            token_dtype=mx.int32,
        ),
        ar_benchmark=lambda request: run_ar_benchmark(
            request,
            prompt_builder=prompt_builder,
            generate=mlx_generate,
        ),
        mtp_benchmark=run_mtp_benchmark,
    )

    started_at = time.perf_counter()
    _print_stage(
        stage="benchmark_generation",
        status="started",
        details={
            "modes": [mode.value for mode in modes],
            "max_tokens": args.max_tokens,
        },
    )
    for row in run_benchmark_modes(modes=modes, runner=runner):
        print(render_json_line(row), flush=True)
    _print_stage(
        stage="benchmark_generation", status="completed", started_at=started_at
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
