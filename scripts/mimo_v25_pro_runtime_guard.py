#!/usr/bin/env python3
"""Fail-closed safety guard before full MiMo V2.5 Pro startup."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from http.client import HTTPResponse
from typing import Final, cast
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
type JsonObject = Mapping[str, object]


@dataclass(frozen=True)
class GuardVerdict:
    safe: bool
    reasons: list[str]


@dataclass(frozen=True)
class CliArgs:
    exo_url: str
    state_json: str | None
    min_available_gib: int


def _print_verdict(verdict: GuardVerdict) -> None:
    print(json.dumps(asdict(verdict), indent=2))


def _as_json_object(value: object) -> JsonObject | None:
    if not isinstance(value, dict):
        return None
    candidate = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in candidate):
        return None
    return cast(JsonObject, candidate)


def _load_json_object(raw_json: str, *, context: str) -> JsonObject:
    decoded = cast(object, json.loads(raw_json))
    json_object = _as_json_object(decoded)
    if json_object is None:
        raise ValueError(f"{context} did not decode to a JSON object")
    return json_object


def _coerce_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"invalid memory value {value!r}") from exc
    raise ValueError(f"invalid memory value {value!r}")


def _unwrap_instance(instance: JsonObject) -> JsonObject:
    if len(instance) != 1:
        return {}
    value = next(iter(instance.values()))
    inner = _as_json_object(value)
    return inner if inner is not None else {}


def _model_id_from_instance(instance: JsonObject) -> str | None:
    inner = _unwrap_instance(instance)
    shard_assignments = _as_json_object(inner.get("shardAssignments"))
    if shard_assignments is not None:
        model_id = shard_assignments.get("modelId")
        return str(model_id) if model_id is not None else None
    return None


def _is_mimo_text(value: str) -> bool:
    return any(marker in value for marker in MIMO_MODEL_MARKERS)


def evaluate_state(
    state: JsonObject,
    *,
    processes: Sequence[str],
    min_available_bytes: int,
) -> GuardVerdict:
    reasons: list[str] = []

    instances = _as_json_object(state.get("instances"))
    if instances is not None:
        for instance_id, instance in instances.items():
            instance_data = _as_json_object(instance)
            if instance_data is None:
                continue
            model_id = _model_id_from_instance(instance_data)
            if model_id is not None and _is_mimo_text(model_id):
                reasons.append(f"active MiMo instance {instance_id}")

    runners = _as_json_object(state.get("runners"))
    if runners is not None:
        for runner_id, runner in runners.items():
            runner_data = _as_json_object(runner)
            if runner_data is None:
                continue
            if "RunnerLoading" in runner_data:
                reasons.append(f"runner {runner_id} is loading")
            if "RunnerWarmingUp" in runner_data:
                reasons.append(f"runner {runner_id} is warming")

    for proc in processes:
        if _is_mimo_text(proc) and "rg " not in proc:
            reasons.append(f"resident MiMo process: {proc[:180]}")

    node_memory = _as_json_object(state.get("nodeMemory"))
    if node_memory is not None:
        for node_id, memory in node_memory.items():
            memory_data = _as_json_object(memory)
            if memory_data is None:
                continue
            ram_available = _as_json_object(memory_data.get("ramAvailable"))
            available = _coerce_int(
                ram_available.get("inBytes") if ram_available is not None else 0
            )
            if available < min_available_bytes:
                reasons.append(f"node {node_id} below memory floor")

    return GuardVerdict(safe=not reasons, reasons=reasons)


def fetch_exo_state(base_url: str) -> JsonObject:
    response_handle = cast(
        HTTPResponse, request.urlopen(f"{base_url.rstrip('/')}/state", timeout=10)
    )
    with response_handle as response:
        payload = response.read().decode("utf-8")
    return _load_json_object(payload, context="exo /state")


def collect_processes() -> list[str]:
    output = subprocess.check_output(["ps", "axo", "command="], text=True)
    return [line.strip() for line in output.splitlines() if line.strip()]


def _parse_args(argv: Sequence[str]) -> CliArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exo-url", default=DEFAULT_EXO_URL)
    parser.add_argument("--state-json", type=str, default=None)
    parser.add_argument(
        "--min-available-gib",
        type=int,
        default=DEFAULT_MIN_AVAILABLE_GIB,
    )
    namespace = parser.parse_args(argv)

    exo_url = cast(object, namespace.exo_url)
    state_json = cast(object, namespace.state_json)
    min_available_gib = cast(object, namespace.min_available_gib)

    if not isinstance(exo_url, str):
        raise TypeError("--exo-url must be a string")
    if state_json is not None and not isinstance(state_json, str):
        raise TypeError("--state-json must be a string when provided")
    if not isinstance(min_available_gib, int):
        raise TypeError("--min-available-gib must be an integer")

    return CliArgs(
        exo_url=exo_url,
        state_json=state_json,
        min_available_gib=min_available_gib,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.state_json:
        try:
            state = _load_json_object(args.state_json, context="--state-json")
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

    try:
        verdict = evaluate_state(
            state,
            processes=processes,
            min_available_bytes=args.min_available_gib * 1024**3,
        )
    except (TypeError, ValueError) as exc:
        _print_verdict(
            GuardVerdict(safe=False, reasons=[f"failed to evaluate state: {exc}"])
        )
        return 2

    _print_verdict(verdict)
    return 0 if verdict.safe else 2


if __name__ == "__main__":
    raise SystemExit(main())
