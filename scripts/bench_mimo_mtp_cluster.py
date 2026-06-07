#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol, cast

from exo.worker.engines.mlx.mimo_mtp_fast.benchmark import render_json_line

JsonObject = dict[str, object]
HttpJsonGet = Callable[[str, float], tuple[int, JsonObject]]
HttpJsonPost = Callable[[str, JsonObject, float], tuple[int, JsonObject]]


class _HttpResponse(Protocol):
    status: int

    def read(self) -> bytes: ...

    def __enter__(self) -> _HttpResponse: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class ClusterBenchmarkArgs:
    api_base: str
    model_id: str
    prompt: str
    max_tokens: int
    repeats: int
    mode_label: str
    timeout_seconds: float
    payload_extra_json: str | None
    list_models: bool


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark MiMo MTP work through an already-running exo cluster API "
            "instead of materializing the full model in this process."
        )
    )
    parser.add_argument("--api-base", default="http://127.0.0.1:52415")
    parser.add_argument("--model-id", default="kernelpool/MiMo-V2.5-Pro-6bit")
    parser.add_argument(
        "--prompt", default="Write a Python function that parses JSON lines."
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--mode-label",
        default="ar",
        help=(
            "Label for the row being collected, for example ar or mtp-d1. This "
            "script does not itself enable MTP; use --payload-extra-json only "
            "after a guarded cluster MTP request flag exists."
        ),
    )
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--payload-extra-json",
        default=None,
        help=(
            "Optional JSON object merged into the /bench/chat/completions payload "
            "for guarded experimental cluster flags."
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Emit a /v1/models probe row before benchmark requests.",
    )
    return parser


def _parse_args(argv: list[str] | None = None) -> ClusterBenchmarkArgs:
    namespace = _parser().parse_args(argv)
    return ClusterBenchmarkArgs(
        api_base=cast(str, namespace.api_base).rstrip("/"),
        model_id=cast(str, namespace.model_id),
        prompt=cast(str, namespace.prompt),
        max_tokens=cast(int, namespace.max_tokens),
        repeats=cast(int, namespace.repeats),
        mode_label=cast(str, namespace.mode_label),
        timeout_seconds=cast(float, namespace.timeout_seconds),
        payload_extra_json=cast(str | None, namespace.payload_extra_json),
        list_models=cast(bool, namespace.list_models),
    )


def _decode_json_object(raw_body: str) -> JsonObject:
    decoded = cast(object, json.loads(raw_body))
    if not isinstance(decoded, dict):
        raise ValueError("expected JSON object response")
    return cast(JsonObject, decoded)


def _http_json_get(url: str, timeout_seconds: float) -> tuple[int, JsonObject]:
    request = urllib.request.Request(url, method="GET")
    try:
        raw_response = cast(
            _HttpResponse, urllib.request.urlopen(request, timeout=timeout_seconds)
        )
        with raw_response as response:
            body = response.read().decode("utf-8", errors="replace")
            return int(response.status), _decode_json_object(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return int(exc.code), _decode_json_object(body)


def _http_json_post(
    url: str, payload: JsonObject, timeout_seconds: float
) -> tuple[int, JsonObject]:
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        raw_response = cast(
            _HttpResponse, urllib.request.urlopen(request, timeout=timeout_seconds)
        )
        with raw_response as response:
            body = response.read().decode("utf-8", errors="replace")
            return int(response.status), _decode_json_object(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return int(exc.code), _decode_json_object(body)


def _payload_extra(raw_payload_extra_json: str | None) -> JsonObject:
    if raw_payload_extra_json is None:
        return {}
    decoded = cast(object, json.loads(raw_payload_extra_json))
    if not isinstance(decoded, dict):
        raise ValueError("--payload-extra-json must decode to a JSON object")
    return cast(JsonObject, decoded)


def build_cluster_chat_payload(
    *,
    model_id: str,
    prompt: str,
    max_tokens: int,
    payload_extra: Mapping[str, object],
) -> JsonObject:
    payload: JsonObject = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0.0,
    }
    payload.update(payload_extra)
    return payload


def _generation_stats(response: JsonObject) -> JsonObject | None:
    raw_stats = response.get("generation_stats")
    if isinstance(raw_stats, dict):
        return cast(JsonObject, raw_stats)
    return None


def _cluster_metric_row(
    *,
    api_base: str,
    model_id: str,
    mode_label: str,
    repeat_index: int,
    elapsed_seconds: float,
    status_code: int,
    response: JsonObject,
    payload_extra_keys: list[str],
) -> JsonObject:
    stats = _generation_stats(response)
    generation_tps = None if stats is None else stats.get("generation_tps")
    generation_tokens = None if stats is None else stats.get("generation_tokens")
    return {
        "kind": "cluster_benchmark_metric",
        "cluster_path": "exo_api_bench_chat_completions",
        "api_base": api_base,
        "mode": mode_label,
        "model": model_id,
        "repeat_index": repeat_index,
        "http_status": status_code,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "generation_tps": generation_tps,
        "generation_tokens": generation_tokens,
        "generation_stats": stats,
        "power_usage": response.get("power_usage"),
        "payload_extra_keys": payload_extra_keys,
        "next_step": (
            "use this as the distributed AR baseline"
            if mode_label == "ar"
            else "compare only against same-cluster AR rows and confirm the response used the guarded MTP path"
        ),
    }


def _cluster_error_row(
    *,
    api_base: str,
    mode_label: str,
    repeat_index: int | None,
    error: str,
    stage: str,
) -> JsonObject:
    return {
        "kind": "cluster_benchmark_error",
        "cluster_path": "exo_api_bench_chat_completions",
        "api_base": api_base,
        "mode": mode_label,
        "repeat_index": repeat_index,
        "stage": stage,
        "error": error,
        "next_step": "start or point to an exo cluster API that can serve /bench/chat/completions",
    }


def _models_probe_row(
    *,
    api_base: str,
    status_code: int,
    response: JsonObject,
) -> JsonObject:
    data: object = response.get("data")
    model_ids: list[str] = []
    if isinstance(data, list):
        for raw_item in cast(list[object], data):
            item = raw_item
            if isinstance(item, dict):
                mapped_item = cast(Mapping[str, object], item)
                model_id = mapped_item.get("id")
                if isinstance(model_id, str):
                    model_ids.append(model_id)
    return {
        "kind": "cluster_models_probe",
        "api_base": api_base,
        "http_status": status_code,
        "model_ids": model_ids,
        "model_count": len(model_ids),
    }


def run_cluster_benchmark(
    *,
    args: ClusterBenchmarkArgs,
    http_post: HttpJsonPost,
    http_get: HttpJsonGet,
    timer: Callable[[], float] = time.perf_counter,
) -> list[JsonObject]:
    rows: list[JsonObject] = []
    try:
        payload_extra = _payload_extra(args.payload_extra_json)
    except (json.JSONDecodeError, ValueError) as exc:
        return [
            _cluster_error_row(
                api_base=args.api_base,
                mode_label=args.mode_label,
                repeat_index=None,
                stage="payload_validation",
                error=str(exc),
            )
        ]

    if args.list_models:
        try:
            status_code, response = http_get(
                f"{args.api_base}/v1/models", args.timeout_seconds
            )
            rows.append(
                _models_probe_row(
                    api_base=args.api_base,
                    status_code=status_code,
                    response=response,
                )
            )
        except (OSError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            rows.append(
                _cluster_error_row(
                    api_base=args.api_base,
                    mode_label=args.mode_label,
                    repeat_index=None,
                    stage="models_probe",
                    error=str(exc),
                )
            )
            return rows

    payload = build_cluster_chat_payload(
        model_id=args.model_id,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        payload_extra=payload_extra,
    )
    payload_extra_keys = sorted(payload_extra.keys())
    for repeat_index in range(args.repeats):
        start = timer()
        try:
            status_code, response = http_post(
                f"{args.api_base}/bench/chat/completions",
                payload,
                args.timeout_seconds,
            )
        except (OSError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            rows.append(
                _cluster_error_row(
                    api_base=args.api_base,
                    mode_label=args.mode_label,
                    repeat_index=repeat_index,
                    stage="bench_chat_completions",
                    error=str(exc),
                )
            )
            continue
        rows.append(
            _cluster_metric_row(
                api_base=args.api_base,
                model_id=args.model_id,
                mode_label=args.mode_label,
                repeat_index=repeat_index,
                elapsed_seconds=max(0.0, timer() - start),
                status_code=status_code,
                response=response,
                payload_extra_keys=payload_extra_keys,
            )
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = run_cluster_benchmark(
        args=args, http_post=_http_json_post, http_get=_http_json_get
    )
    for row in rows:
        print(render_json_line(row), flush=True)
    return 1 if any(row["kind"] == "cluster_benchmark_error" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
