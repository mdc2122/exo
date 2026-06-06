#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from exo.worker.engines.mlx.mimo_mtp_fast.benchmark import (
    MimoMtpBenchmarkMode,
    MtpBenchmarkRequest,
    acceptance_rate,
    run_mtp_benchmark,
)
from exo.worker.engines.mlx.mimo_mtp_fast.one_cycle import (
    MimoMtpCycleTiming,
    MimoMtpOneCycleResult,
)

JsonReport = dict[str, Any]


def _stringify_counts(counts: dict[int, int]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(counts.items())}


def _timing_report(timing: MimoMtpCycleTiming) -> dict[str, float]:
    return {
        "proposal": round(timing.proposal_seconds, 6),
        "verification": round(timing.verification_seconds, 6),
        "acceptance": round(timing.acceptance_seconds, 6),
        "fallback": round(timing.fallback_seconds, 6),
    }


def _make_synthetic_one_cycle() -> Callable[
    [tuple[int, ...], int], MimoMtpOneCycleResult
]:
    cycle_index = 0

    def one_cycle(
        _history: tuple[int, ...], requested_depth: int
    ) -> MimoMtpOneCycleResult:
        nonlocal cycle_index
        cycle_index += 1
        if cycle_index >= 4:
            proposed = (40,)
            return MimoMtpOneCycleResult(
                proposed_token_ids=proposed,
                accepted_token_ids=(),
                fallback_token_id=77,
                attempted_depth=1,
                accepted_depth=0,
                elapsed_seconds=0.001,
                timing=MimoMtpCycleTiming(
                    proposal_seconds=0.001,
                    verification_seconds=0.002,
                    acceptance_seconds=0.00025,
                    fallback_seconds=0.002,
                ),
            )
        proposed = tuple(range(10 * cycle_index, 10 * cycle_index + requested_depth))
        accepted = proposed[:1]
        return MimoMtpOneCycleResult(
            proposed_token_ids=proposed,
            accepted_token_ids=accepted,
            fallback_token_id=99,
            attempted_depth=len(proposed),
            accepted_depth=len(accepted),
            elapsed_seconds=0.001,
            timing=MimoMtpCycleTiming(
                proposal_seconds=0.001,
                verification_seconds=0.002,
                acceptance_seconds=0.00025,
            ),
        )

    return one_cycle


def run_synthetic_probe(*, max_tokens: int = 4, requested_depth: int = 3) -> JsonReport:
    result = run_mtp_benchmark(
        MtpBenchmarkRequest(
            mode=MimoMtpBenchmarkMode.D3,
            token_history=(1,),
            max_tokens=max_tokens,
            requested_depth=requested_depth,
        ),
        one_cycle=_make_synthetic_one_cycle(),
        elapsed_seconds_override=0.004,
    )
    return {
        "kind": "synthetic_mtp_fastpath_probe",
        "full_model_loaded": False,
        "generated_tokens": result.generated_tokens,
        "requested_depth": requested_depth,
        "attempted_depth_counts": _stringify_counts(result.attempted_depth_counts),
        "accepted_depth_counts": _stringify_counts(result.accepted_depth_counts),
        "acceptance_rate": acceptance_rate(
            attempted_depth_counts=result.attempted_depth_counts,
            accepted_depth_counts=result.accepted_depth_counts,
        ),
        "timing_breakdown_seconds": _timing_report(result.timing_totals),
        "decode_seconds": result.decode_seconds,
        "notes": [
            "synthetic/tiny providers only; no MiMo model materialized",
            "diagnostics are for code-path readiness, not speedup claims",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a synthetic MiMo MTP fastpath timing/acceptance probe without loading MiMo."
    )
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--requested-depth", type=int, default=3)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    namespace = _parser().parse_args(argv)
    max_tokens = cast(int, namespace.max_tokens)
    requested_depth = cast(int, namespace.requested_depth)
    json_out = cast(Path | None, namespace.json_out)
    report = run_synthetic_probe(
        max_tokens=max_tokens,
        requested_depth=requested_depth,
    )
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if json_out is None:
        print(serialized, end="")
    else:
        json_out.write_text(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
