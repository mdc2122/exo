from __future__ import annotations

import json
from collections.abc import Iterator
from typing import cast

import pytest

from scripts import bench_mimo_mtp_cluster as cluster_bench


def _timer(values: list[float]) -> Iterator[float]:
    yield from values


def test_build_cluster_chat_payload_merges_guarded_extra_fields() -> None:
    payload = cluster_bench.build_cluster_chat_payload(
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        prompt="hello",
        max_tokens=4,
        payload_extra={"mimo_mtp_fastpath": True, "mimo_mtp_depth": 1},
    )

    assert payload == {
        "model": "kernelpool/MiMo-V2.5-Pro-6bit",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 4,
        "stream": False,
        "temperature": 0.0,
        "mimo_mtp_fastpath": True,
        "mimo_mtp_depth": 1,
    }


def test_run_cluster_benchmark_emits_models_probe_and_metric_row() -> None:
    posts: list[dict[str, object]] = []

    def fake_get(url: str, timeout_seconds: float) -> tuple[int, dict[str, object]]:
        assert url == "http://cluster.example/v1/models"
        assert timeout_seconds == 12.0
        return 200, {"data": [{"id": "kernelpool/MiMo-V2.5-Pro-6bit"}]}

    def fake_post(
        url: str, payload: dict[str, object], timeout_seconds: float
    ) -> tuple[int, dict[str, object]]:
        posts.append({"url": url, "payload": payload, "timeout": timeout_seconds})
        return 200, {
            "generation_stats": {
                "prompt_tps": 100.0,
                "generation_tps": 31.5,
                "prompt_tokens": 8,
                "generation_tokens": 4,
                "peak_memory_usage": {"bytes": 123},
            },
            "power_usage": {"elapsed_seconds": 1.0, "nodes": []},
        }

    timer = _timer([10.0, 11.0])
    rows = cluster_bench.run_cluster_benchmark(
        args=cluster_bench.ClusterBenchmarkArgs(
            api_base="http://cluster.example",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            prompt="hello",
            max_tokens=4,
            repeats=1,
            mode_label="ar",
            timeout_seconds=12.0,
            payload_extra_json=None,
            list_models=True,
        ),
        http_get=fake_get,
        http_post=fake_post,
        timer=lambda: next(timer),
    )

    assert rows[0] == {
        "kind": "cluster_models_probe",
        "api_base": "http://cluster.example",
        "http_status": 200,
        "model_ids": ["kernelpool/MiMo-V2.5-Pro-6bit"],
        "model_count": 1,
    }
    assert rows[1]["kind"] == "cluster_benchmark_metric"
    assert rows[1]["cluster_path"] == "exo_api_bench_chat_completions"
    assert rows[1]["mode"] == "ar"
    assert rows[1]["generation_tps"] == 31.5
    assert rows[1]["generation_tokens"] == 4
    assert rows[1]["elapsed_seconds"] == 1.0
    assert posts == [
        {
            "url": "http://cluster.example/bench/chat/completions",
            "payload": {
                "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 4,
                "stream": False,
                "temperature": 0.0,
            },
            "timeout": 12.0,
        }
    ]


def test_run_cluster_benchmark_reports_connection_error() -> None:
    def fake_get(_url: str, _timeout_seconds: float) -> tuple[int, dict[str, object]]:
        raise OSError("connection refused")

    def fake_post(
        _url: str, _payload: dict[str, object], _timeout_seconds: float
    ) -> tuple[int, dict[str, object]]:
        raise AssertionError("models probe failure should stop before benchmark")

    rows = cluster_bench.run_cluster_benchmark(
        args=cluster_bench.ClusterBenchmarkArgs(
            api_base="http://127.0.0.1:52415",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            prompt="hello",
            max_tokens=1,
            repeats=1,
            mode_label="ar",
            timeout_seconds=1.0,
            payload_extra_json=None,
            list_models=True,
        ),
        http_get=fake_get,
        http_post=fake_post,
    )

    assert rows == [
        {
            "kind": "cluster_benchmark_error",
            "cluster_path": "exo_api_bench_chat_completions",
            "api_base": "http://127.0.0.1:52415",
            "mode": "ar",
            "repeat_index": None,
            "stage": "models_probe",
            "error": "connection refused",
            "next_step": "start or point to an exo cluster API that can serve /bench/chat/completions",
        }
    ]


def test_main_prints_ndjson_and_returns_error_for_unavailable_cluster(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(_url: str, _timeout_seconds: float) -> tuple[int, dict[str, object]]:
        raise OSError("connection refused")

    monkeypatch.setattr(cluster_bench, "_http_json_get", fake_get)

    exit_code = cluster_bench.main(["--list-models", "--timeout-seconds", "1"])

    assert exit_code == 1
    decoded = cast(object, json.loads(capsys.readouterr().out))
    assert isinstance(decoded, dict)
    row = cast(dict[str, object], decoded)
    assert row["kind"] == "cluster_benchmark_error"
    assert row["stage"] == "models_probe"
