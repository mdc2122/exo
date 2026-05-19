#!/usr/bin/env python3
"""Fail-closed safety guard before full MiMo V2.5 Pro startup."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any, Final, Sequence
from urllib import request

MIMO_MODEL_ID: Final[str] = "kernelpool/MiMo-V2.5-Pro-6bit"
MIMO_MODEL_MARKERS: Final[tuple[str, ...]] = (
    "MiMo-V2.5-Pro",
    "mimo-v25-pro",
    "kernelpool/MiMo-V2.5-Pro-6bit",
    "XiaomiMiMo/MiMo-V2.5-Pro",
)
DEFAULT_EXO_URL: Final[str] = "http://127.0.0.1:52415"
DEFAULT_MIN_AVAILABLE_GIB: Final[int] = 200


@dataclass(frozen=True)
class GuardVerdict:
    safe: bool
    reasons: list[str]


def _print_verdict(verdict: GuardVerdict) -> None:
    print(json.dumps(asdict(verdict), indent=2))


def _unwrap_instance(instance: dict[str, Any]) -> dict[str, Any]:
    if len(instance) != 1:
        return {}
    value = next(iter(instance.values()))
    return value if isinstance(value, dict) else {}


def _model_id_from_instance(instance: dict[str, Any]) -> str | None:
    inner = _unwrap_instance(instance)
    shard_assignments = inner.get("shardAssignments")
    if isinstance(shard_assignments, dict):
        model_id = shard_assignments.get("modelId")
        return str(model_id) if model_id is not None else None
    return None


def _is_mimo_text(value: str) -> bool:
    return any(marker in value for marker in MIMO_MODEL_MARKERS)


def evaluate_state(
    state: dict[str, Any],
    *,
    processes: Sequence[str],
    min_available_bytes: int,
) -> GuardVerdict:
    reasons: list[str] = []

    instances = state.get("instances", {})
    if isinstance(instances, dict):
        for instance_id, instance in instances.items():
            if isinstance(instance, dict):
                model_id = _model_id_from_instance(instance)
                if model_id is not None and _is_mimo_text(model_id):
                    reasons.append(f"active MiMo instance {instance_id}")

    runners = state.get("runners", {})
    if isinstance(runners, dict):
        for runner_id, runner in runners.items():
            if isinstance(runner, dict):
                if "RunnerLoading" in runner:
                    reasons.append(f"runner {runner_id} is loading")
                if "RunnerWarmingUp" in runner:
                    reasons.append(f"runner {runner_id} is warming")

    for proc in processes:
        if _is_mimo_text(proc) and "rg " not in proc:
            reasons.append(f"resident MiMo process: {proc[:180]}")

    node_memory = state.get("nodeMemory", {})
    if isinstance(node_memory, dict):
        for node_id, memory in node_memory.items():
            if not isinstance(memory, dict):
                continue
            available = (
                memory.get("ramAvailable", {}).get("inBytes", 0)
                if isinstance(memory.get("ramAvailable"), dict)
                else 0
            )
            if int(available) < min_available_bytes:
                reasons.append(f"node {node_id} below memory floor")

    return GuardVerdict(safe=not reasons, reasons=reasons)


def fetch_exo_state(base_url: str) -> dict[str, Any]:
    with request.urlopen(f"{base_url.rstrip('/')}/state", timeout=10) as response:
        data = json.loads(response.read().decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("exo /state did not return a JSON object")
    return data


def collect_processes() -> list[str]:
    output = subprocess.check_output(["ps", "axo", "command="], text=True)
    return [line.strip() for line in output.splitlines() if line.strip()]


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exo-url", default=DEFAULT_EXO_URL)
    parser.add_argument("--state-json", type=str, default=None)
    parser.add_argument(
        "--min-available-gib",
        type=int,
        default=DEFAULT_MIN_AVAILABLE_GIB,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.state_json:
        try:
            state = json.loads(args.state_json)
            if not isinstance(state, dict):
                raise ValueError("--state-json did not decode to a JSON object")
        except (ValueError, json.JSONDecodeError) as exc:
            _print_verdict(
                GuardVerdict(
                    safe=False, reasons=[f"failed to parse --state-json: {exc}"]
                )
            )
            return 2
    else:
        try:
            state = fetch_exo_state(args.exo_url)
        except Exception as exc:
            _print_verdict(
                GuardVerdict(
                    safe=False, reasons=[f"failed to fetch exo state: {exc}"]
                )
            )
            return 2

    try:
        processes = collect_processes()
    except Exception as exc:
        _print_verdict(
            GuardVerdict(safe=False, reasons=[f"failed to collect processes: {exc}"])
        )
        return 2

    verdict = evaluate_state(
        state,
        processes=processes,
        min_available_bytes=args.min_available_gib * 1024**3,
    )
    _print_verdict(verdict)
    return 0 if verdict.safe else 2


if __name__ == "__main__":
    raise SystemExit(main())
