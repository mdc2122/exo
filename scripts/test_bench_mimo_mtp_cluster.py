from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from scripts import bench_mimo_mtp_cluster as cluster_bench


def _timer(values: list[float]) -> Iterator[float]:
    yield from values


def test_bench_mimo_mtp_cluster_is_canonical_runnable_entrypoint() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    entrypoint = repo_root / "scripts" / "bench_mimo_mtp_cluster.py"

    assert entrypoint.is_file()

    completed = subprocess.run(
        [sys.executable, str(entrypoint), "--help"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert (
        "Benchmark MiMo MTP work through an already-running exo cluster API"
        in completed.stdout
    )
    assert "--mode-label" in completed.stdout


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


def test_run_cluster_benchmark_labels_ar_requests_and_results() -> None:
    posted_payloads: list[dict[str, object]] = []

    def fake_get(_url: str, _timeout_seconds: float) -> tuple[int, dict[str, object]]:
        raise AssertionError("models probe should not run when list_models is false")

    def fake_post(
        _url: str, payload: dict[str, object], _timeout_seconds: float
    ) -> tuple[int, dict[str, object]]:
        posted_payloads.append(dict(payload))
        return 200, {
            "generation_stats": {
                "prompt_tps": 100.0,
                "generation_tps": 31.5,
                "generation_tokens": 4,
            },
        }

    timer = _timer([0.0, 1.0])
    rows = cluster_bench.run_cluster_benchmark(
        args=cluster_bench.ClusterBenchmarkArgs(
            api_base="http://cluster.example",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            prompt="hello",
            max_tokens=4,
            matrix_max_tokens=(4,),
            repeats=1,
            mode_label="ar",
            timeout_seconds=12.0,
            payload_extra_json=None,
            list_models=False,
        ),
        http_get=fake_get,
        http_post=fake_post,
        timer=lambda: next(timer),
    )

    assert posted_payloads[0]["benchmark_mode_label"] == "ar"
    assert rows[0]["mode"] == "ar"
    assert rows[0]["accepted_execution_path"] == "ar"


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
            matrix_max_tokens=(4,),
            repeats=1,
            mode_label="ar",
            timeout_seconds=12.0,
            payload_extra_json=None,
            list_models=True,
            matrix_commands=False,
            mimo_mtp_sidecar_path=None,
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
    assert rows[1]["prompt_tps"] == 100.0
    assert rows[1]["power_usage"] == {"elapsed_seconds": 1.0, "nodes": []}
    assert rows[1]["payload_extra_keys"] == []
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
                "benchmark_mode_label": "ar",
                "benchmark_repeat_index": 0,
            },
            "timeout": 12.0,
        }
    ]


def test_run_cluster_benchmark_sends_repeat_index_for_each_request() -> None:
    posted_payloads: list[dict[str, object]] = []

    def fake_get(_url: str, _timeout_seconds: float) -> tuple[int, dict[str, object]]:
        raise AssertionError("models probe should not run when list_models is false")

    def fake_post(
        _url: str, payload: dict[str, object], _timeout_seconds: float
    ) -> tuple[int, dict[str, object]]:
        posted_payloads.append(dict(payload))
        return 200, {
            "generation_stats": {
                "generation_tps": 30.0 + len(posted_payloads),
                "generation_tokens": 4,
            },
        }

    timer = _timer([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    rows = cluster_bench.run_cluster_benchmark(
        args=cluster_bench.ClusterBenchmarkArgs(
            api_base="http://cluster.example",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            prompt="hello",
            max_tokens=4,
            repeats=3,
            mode_label="mtp-d1",
            timeout_seconds=12.0,
            payload_extra_json=json.dumps(
                {"mimo_mtp_fastpath": True, "benchmark_repeat_index": 99}
            ),
            list_models=False,
        ),
        http_get=fake_get,
        http_post=fake_post,
        timer=lambda: next(timer),
    )

    assert [row["repeat_index"] for row in rows] == [0, 1, 2]
    assert [payload["benchmark_repeat_index"] for payload in posted_payloads] == [
        0,
        1,
        2,
    ]
    assert all(payload["mimo_mtp_fastpath"] is True for payload in posted_payloads)
    assert all(
        row["payload_extra_keys"] == ["benchmark_repeat_index", "mimo_mtp_fastpath"]
        for row in rows
    )


def test_validate_rollout_baseline_rejects_single_studio_full_model_metric() -> None:
    validation = cluster_bench.validate_rollout_baseline_row(
        {
            "kind": "benchmark_metric",
            "mode": "ar",
            "generated_tokens": 16,
            "decode_tok_s": 18.0,
            "model": "kernelpool/MiMo-V2.5-Pro-6bit",
        }
    )

    assert validation == {
        "kind": "rollout_baseline_validation",
        "allowed": False,
        "baseline_source": "single_studio_full_model_load",
        "reason": "single-Studio full-model benchmark rows are diagnostic only and cannot be used as MiMo rollout baselines",
        "required_source": "exo cluster /bench/chat/completions AR row",
        "next_step": "collect an AR row with scripts/bench_mimo_mtp_cluster.py against a running exo cluster API",
    }


def test_validate_rollout_baseline_allows_cluster_ar_metric() -> None:
    validation = cluster_bench.validate_rollout_baseline_row(
        {
            "kind": "cluster_benchmark_metric",
            "cluster_path": "exo_api_bench_chat_completions",
            "mode": "ar",
            "model": "kernelpool/MiMo-V2.5-Pro-6bit",
            "generation_tps": 22.0,
            "generation_tokens": 16,
        }
    )

    assert validation == {
        "kind": "rollout_baseline_validation",
        "allowed": True,
        "baseline_source": "exo_cluster_bench_chat_completions",
        "reason": "cluster AR row is eligible as a MiMo rollout baseline",
        "required_source": "exo cluster /bench/chat/completions AR row",
        "next_step": "compare only with same-cluster guarded MTP rows",
    }


def test_validate_rollout_baseline_rejects_cluster_row_classified_single_studio_full_model() -> (
    None
):
    validation = cluster_bench.validate_rollout_baseline_row(
        {
            "kind": "cluster_benchmark_metric",
            "cluster_path": "exo_api_bench_chat_completions",
            "mode": "ar",
            "model_path_classification": {
                "path": "single_studio_full_model_load",
                "is_exo_cluster_tensor_parallel": False,
                "disable_reason": "response execution path is not tensor-parallel across multiple cluster nodes",
            },
        }
    )

    assert validation == {
        "kind": "rollout_baseline_validation",
        "allowed": False,
        "baseline_source": "single_studio_full_model_load",
        "reason": "single-Studio full-model loading is not an allowed MiMo rollout baseline",
        "required_source": "exo cluster /bench/chat/completions AR row",
        "next_step": "collect an AR row with scripts/bench_mimo_mtp_cluster.py against a running tensor-parallel exo cluster API",
    }


def test_cluster_metric_row_classifies_tensor_parallel_live_execution_path() -> None:
    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="ar",
        repeat_index=0,
        elapsed_seconds=1.25,
        status_code=200,
        response={
            "generation_stats": {
                "prompt_tps": 100.0,
                "generation_tps": 31.5,
                "prompt_tokens": 8,
                "generation_tokens": 4,
                "peak_memory_usage": {"bytes": 123},
            },
            "power_usage": {"elapsed_seconds": 1.0, "nodes": []},
            "execution_path": {
                "sharding": "Tensor",
                "world_size": 2,
                "engine": "mlx",
                "model_path": "/Users/studio2/.cache/exo/downloads/kernelpool--MiMo-V2.5-Pro-6bit",
            },
        },
        payload_extra_keys=[],
    )

    assert row["live_execution_path"] == "exo_cluster_tensor_parallel"
    assert row["model_path_classification"] == {
        "path": "exo_cluster_tensor_parallel",
        "is_exo_cluster_tensor_parallel": True,
        "source": "response.execution_path",
        "sharding": "Tensor",
        "world_size": 2,
        "model_path": "/Users/studio2/.cache/exo/downloads/kernelpool--MiMo-V2.5-Pro-6bit",
        "disable_reason": None,
    }


def test_cluster_metric_row_records_accepted_live_execution_path_from_response() -> (
    None
):
    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="mtp-d2",
        repeat_index=0,
        elapsed_seconds=1.25,
        status_code=200,
        response={
            "generation_stats": {
                "prompt_tps": 100.0,
                "generation_tps": 35.5,
                "prompt_tokens": 8,
                "generation_tokens": 4,
                "accepted_execution_path": "mimo_mtp_fastpath",
            },
            "power_usage": {"elapsed_seconds": 1.0, "nodes": []},
        },
        payload_extra_keys=["mimo_mtp_depth", "mimo_mtp_fastpath"],
    )

    assert row["accepted_execution_path"] == "mimo_mtp_fastpath"


def test_mtp_metric_row_classifies_generation_tps_below_30_tok_s() -> None:
    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="mtp-d1",
        repeat_index=0,
        elapsed_seconds=1.25,
        status_code=200,
        response={
            "generation_stats": {
                "generation_tps": 29.99,
                "generation_tokens": 4,
                "accepted_execution_path": "mimo_mtp_fastpath",
            },
        },
        payload_extra_keys=["mimo_mtp_fastpath"],
    )

    assert row["mtp_throughput_threshold"] == "below_30_tok_s"
    assert row["mtp_speedup_budget"] == {
        "generation_tps": 29.99,
        "threshold_30_tok_s_met": False,
        "threshold_40_tok_s_met": False,
        "tok_s_gap_to_30": 0.01,
        "tok_s_gap_to_40": 10.01,
    }


def test_mtp_metric_row_classifies_generation_tps_at_least_30_tok_s() -> None:
    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="mtp-d1",
        repeat_index=0,
        elapsed_seconds=1.25,
        status_code=200,
        response={
            "generation_stats": {
                "generation_tps": 30.0,
                "generation_tokens": 4,
                "accepted_execution_path": "mimo_mtp_fastpath",
            },
        },
        payload_extra_keys=["mimo_mtp_fastpath"],
    )

    assert row["mtp_throughput_threshold"] == "at_least_30_tok_s"
    assert row["mtp_speedup_budget"] == {
        "generation_tps": 30.0,
        "threshold_30_tok_s_met": True,
        "threshold_40_tok_s_met": False,
        "tok_s_gap_to_30": 0.0,
        "tok_s_gap_to_40": 10.0,
    }


def test_mtp_metric_row_classifies_generation_tps_at_least_40_tok_s() -> None:
    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="mtp-d1",
        repeat_index=0,
        elapsed_seconds=1.25,
        status_code=200,
        response={
            "generation_stats": {
                "generation_tps": 40.0,
                "generation_tokens": 4,
                "accepted_execution_path": "mimo_mtp_fastpath",
            },
        },
        payload_extra_keys=["mimo_mtp_fastpath"],
    )

    assert row["mtp_throughput_threshold"] == "at_least_40_tok_s"
    assert row["mtp_speedup_budget"] == {
        "generation_tps": 40.0,
        "threshold_30_tok_s_met": True,
        "threshold_40_tok_s_met": True,
        "tok_s_gap_to_30": 0.0,
        "tok_s_gap_to_40": 0.0,
    }


def test_ar_metric_row_does_not_emit_mtp_throughput_classification() -> None:
    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="ar",
        repeat_index=0,
        elapsed_seconds=1.25,
        status_code=200,
        response={
            "generation_stats": {
                "generation_tps": 40.0,
                "generation_tokens": 4,
            },
        },
        payload_extra_keys=[],
    )

    assert "mtp_throughput_threshold" not in row
    assert "mtp_speedup_budget" not in row


def test_cluster_metric_row_marks_non_ar_live_path_unknown_without_response_classification() -> (
    None
):
    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="mtp-d2",
        repeat_index=0,
        elapsed_seconds=1.25,
        status_code=200,
        response={
            "generation_stats": {
                "prompt_tps": 100.0,
                "generation_tps": 35.5,
                "prompt_tokens": 8,
                "generation_tokens": 4,
            },
            "power_usage": {"elapsed_seconds": 1.0, "nodes": []},
        },
        payload_extra_keys=["mimo_mtp_depth", "mimo_mtp_fastpath"],
    )

    assert row["accepted_execution_path"] == "unknown"


def test_classify_mtp_vs_ar_baseline_marks_mtp_speed_win() -> None:
    classification = cluster_bench.classify_mtp_vs_ar_baseline(
        ar_row={
            "kind": "cluster_benchmark_metric",
            "cluster_path": "exo_api_bench_chat_completions",
            "mode": "ar",
            "model": "kernelpool/MiMo-V2.5-Pro-6bit",
            "generation_tps": 22.0,
            "generation_tokens": 64,
        },
        mtp_row={
            "kind": "cluster_benchmark_metric",
            "cluster_path": "exo_api_bench_chat_completions",
            "mode": "mtp-d2",
            "model": "kernelpool/MiMo-V2.5-Pro-6bit",
            "generation_tps": 33.0,
            "generation_tokens": 64,
            "generation_stats": {
                "acceptance_rate": 0.82,
                "attempted_depth_counts": {"2": 32},
                "accepted_depth_counts": {"2": 28, "0": 4},
            },
        },
    )

    assert classification == {
        "kind": "mtp_vs_ar_baseline_classification",
        "classification": "pass",
        "bottlenecks": ["mtp_beats_ar", "mtp_reaches_30"],
        "ar_generation_tps": 22.0,
        "ar_ms_per_token": 45.4545,
        "mtp_generation_tps": 33.0,
        "mtp_ms_per_token": 30.303,
        "speedup_ratio": 1.5,
        "target_gap_tps_30": 0.0,
        "target_gap_tps_40": 7.0,
        "same_cluster_comparison": True,
        "reason": "MTP row beats same-cluster AR baseline",
        "next_step": "eligible for measured bottleneck analysis and guarded Slice 5 review only if median rows repeat this win",
    }


def test_classify_mtp_vs_ar_baseline_marks_mtp_non_win() -> None:
    classification = cluster_bench.classify_mtp_vs_ar_baseline(
        ar_row={
            "kind": "cluster_benchmark_metric",
            "cluster_path": "exo_api_bench_chat_completions",
            "mode": "ar",
            "model": "kernelpool/MiMo-V2.5-Pro-6bit",
            "generation_tps": 28.0,
            "generation_tokens": 64,
        },
        mtp_row={
            "kind": "cluster_benchmark_metric",
            "cluster_path": "exo_api_bench_chat_completions",
            "mode": "mtp-d1",
            "model": "kernelpool/MiMo-V2.5-Pro-6bit",
            "generation_tps": 24.0,
            "generation_tokens": 64,
            "generation_stats": {"acceptance_rate": 0.35},
        },
    )

    assert classification == {
        "kind": "mtp_vs_ar_baseline_classification",
        "classification": "fail",
        "bottlenecks": ["acceptance_rate_low"],
        "ar_generation_tps": 28.0,
        "ar_ms_per_token": 35.7143,
        "mtp_generation_tps": 24.0,
        "mtp_ms_per_token": 41.6667,
        "speedup_ratio": 0.8571,
        "target_gap_tps_30": 6.0,
        "target_gap_tps_40": 16.0,
        "same_cluster_comparison": True,
        "reason": "MTP row does not beat same-cluster AR baseline",
        "next_step": "do not claim MTP speedup; inspect acceptance/proposal/verification/fallback telemetry before further optimization",
    }


def test_classify_mtp_vs_ar_baseline_marks_absent_mtp_telemetry_ambiguous() -> None:
    classification = cluster_bench.classify_mtp_vs_ar_baseline(
        ar_row={
            "kind": "cluster_benchmark_metric",
            "cluster_path": "exo_api_bench_chat_completions",
            "mode": "ar",
            "model": "kernelpool/MiMo-V2.5-Pro-6bit",
            "generation_tps": 28.0,
            "generation_tokens": 64,
        },
        mtp_row={
            "kind": "cluster_benchmark_metric",
            "cluster_path": "exo_api_bench_chat_completions",
            "mode": "mtp-d1",
            "model": "kernelpool/MiMo-V2.5-Pro-6bit",
            "generation_tokens": 64,
            "accepted_execution_path": "unknown",
        },
    )

    assert classification == {
        "kind": "mtp_vs_ar_baseline_classification",
        "classification": "ambiguous",
        "bottlenecks": ["absent_mtp_telemetry"],
        "ar_generation_tps": 28.0,
        "ar_ms_per_token": 35.7143,
        "mtp_generation_tps": None,
        "mtp_ms_per_token": None,
        "speedup_ratio": None,
        "target_gap_tps_30": None,
        "target_gap_tps_40": None,
        "same_cluster_comparison": True,
        "reason": "MTP row lacks live fastpath generation_tps telemetry; cannot classify pass or fail",
        "next_step": "collect same-cluster guarded MTP rows with accepted_execution_path=mimo_mtp_fastpath and generation_tps before making speedup or target claims",
    }


def test_render_ar_baseline_commands_includes_exact_16_and_64_token_rows() -> None:
    commands = cluster_bench.render_ar_baseline_commands(
        api_base="http://127.0.0.1:52415",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
    )

    assert commands == [
        "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 16 --repeats 1 --mode-label ar --list-models",
        "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 64 --repeats 1 --mode-label ar --list-models",
    ]


def test_run_cluster_benchmark_records_blocked_status_without_success_rows_when_cluster_is_unavailable() -> (
    None
):
    def fake_get(_url: str, _timeout_seconds: float) -> tuple[int, dict[str, object]]:
        raise AssertionError("models probe should not run without --list-models")

    def fake_post(
        _url: str, _payload: dict[str, object], _timeout_seconds: float
    ) -> tuple[int, dict[str, object]]:
        raise OSError("connection refused")

    rows = cluster_bench.run_cluster_benchmark(
        args=cluster_bench.ClusterBenchmarkArgs(
            api_base="http://127.0.0.1:52415",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            prompt="hello",
            max_tokens=1,
            matrix_max_tokens=(1,),
            repeats=1,
            mode_label="ar",
            timeout_seconds=1.0,
            payload_extra_json=None,
            list_models=False,
        ),
        http_get=fake_get,
        http_post=fake_post,
    )

    success_rows = [row for row in rows if row["kind"] == "cluster_benchmark_metric"]
    assert success_rows == []
    assert rows == [
        {
            "kind": "cluster_benchmark_error",
            "cluster_path": "exo_api_bench_chat_completions",
            "api_base": "http://127.0.0.1:52415",
            "mode": "ar",
            "repeat_index": 0,
            "stage": "bench_chat_completions",
            "error": "connection refused",
            "status": "blocked_with_command",
            "success_row_count": 0,
            "rerun_command": "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 1 --repeats 1 --mode-label ar",
            "next_step": "start or point to an exo cluster API that can serve /bench/chat/completions, then rerun the command in rerun_command",
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
            matrix_max_tokens=(1,),
            repeats=1,
            mode_label="ar",
            timeout_seconds=1.0,
            payload_extra_json=None,
            list_models=True,
            matrix_commands=False,
            mimo_mtp_sidecar_path=None,
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
            "status": "blocked_with_command",
            "success_row_count": 0,
            "rerun_command": "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 1 --repeats 1 --mode-label ar --list-models",
            "next_step": "start or point to an exo cluster API that can serve /bench/chat/completions, then rerun the command in rerun_command",
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


def test_calculate_mtp_speedup_budget_for_ar_only_rows() -> None:
    budget = cluster_bench.calculate_mtp_speedup_budget(
        [
            {
                "kind": "cluster_benchmark_metric",
                "cluster_path": "exo_api_bench_chat_completions",
                "mode": "ar",
                "generation_tps": 20.0,
                "generation_tokens": 16,
            },
            {
                "kind": "cluster_benchmark_metric",
                "cluster_path": "exo_api_bench_chat_completions",
                "mode": "ar",
                "generation_tps": 25.0,
                "generation_tokens": 64,
            },
        ]
    )

    assert budget == {
        "kind": "mimo_mtp_speedup_budget",
        "ar_row_count": 2,
        "mtp_row_count": 0,
        "ar_median_tok_s": 22.5,
        "ar_ms_per_token": 44.444444,
        "mtp_median_tok_s": None,
        "mtp_target_gap_to_30_tok_s": None,
        "mtp_target_gap_to_40_tok_s": None,
        "mtp_vs_ar_speedup_ratio": None,
        "status": "blocked_missing_mtp_rows",
        "next_step": "collect same-cluster guarded MTP rows before making speedup or target claims",
    }


def test_calculate_mtp_speedup_budget_for_ar_and_mtp_rows() -> None:
    budget = cluster_bench.calculate_mtp_speedup_budget(
        [
            {
                "kind": "cluster_benchmark_metric",
                "cluster_path": "exo_api_bench_chat_completions",
                "mode": "ar",
                "generation_tps": 20.0,
                "generation_tokens": 16,
            },
            {
                "kind": "cluster_benchmark_metric",
                "cluster_path": "exo_api_bench_chat_completions",
                "mode": "ar",
                "generation_tps": 22.0,
                "generation_tokens": 64,
            },
            {
                "kind": "cluster_benchmark_metric",
                "cluster_path": "exo_api_bench_chat_completions",
                "mode": "mtp-d1",
                "generation_tps": 32.0,
                "generation_tokens": 16,
                "accepted_execution_path": "mimo_mtp_fastpath",
            },
            {
                "kind": "cluster_benchmark_metric",
                "cluster_path": "exo_api_bench_chat_completions",
                "mode": "mtp-d1",
                "generation_tps": 34.0,
                "generation_tokens": 64,
                "accepted_execution_path": "mimo_mtp_fastpath",
            },
        ]
    )

    assert budget == {
        "kind": "mimo_mtp_speedup_budget",
        "ar_row_count": 2,
        "mtp_row_count": 2,
        "ar_median_tok_s": 21.0,
        "ar_ms_per_token": 47.619048,
        "mtp_median_tok_s": 33.0,
        "mtp_target_gap_to_30_tok_s": 0.0,
        "mtp_target_gap_to_40_tok_s": 7.0,
        "mtp_vs_ar_speedup_ratio": 1.571429,
        "status": "same_cluster_budget_ready",
        "next_step": "use this budget only with same-cluster AR-vs-MTP evidence rows; Slice 5 still requires a real MTP win without fallback concerns",
    }


def test_calculate_mtp_speedup_budget_marks_incomplete_telemetry_without_fabricating_metrics() -> (
    None
):
    budget = cluster_bench.calculate_mtp_speedup_budget(
        [
            {
                "kind": "cluster_benchmark_metric",
                "cluster_path": "exo_api_bench_chat_completions",
                "mode": "ar",
                "generation_tps": None,
                "generation_tokens": 16,
            },
            {
                "kind": "cluster_benchmark_metric",
                "cluster_path": "exo_api_bench_chat_completions",
                "mode": "mtp-d1",
                "generation_tokens": 16,
                "accepted_execution_path": "unknown",
            },
            {
                "kind": "cluster_benchmark_error",
                "status": "blocked_with_command",
                "rerun_command": "uv run python3 scripts/bench_mimo_mtp_cluster.py --mode-label ar",
            },
        ]
    )

    assert budget == {
        "kind": "mimo_mtp_speedup_budget",
        "ar_row_count": 0,
        "mtp_row_count": 0,
        "ar_median_tok_s": None,
        "ar_ms_per_token": None,
        "mtp_median_tok_s": None,
        "mtp_target_gap_to_30_tok_s": None,
        "mtp_target_gap_to_40_tok_s": None,
        "mtp_vs_ar_speedup_ratio": None,
        "status": "blocked_incomplete_telemetry",
        "next_step": "rerun same-cluster AR and guarded MTP benchmarks until generation_tps is present on both row sets",
    }


def test_render_benchmark_matrix_command_rows_includes_ar_and_guarded_mtp() -> None:
    rows = cluster_bench.render_benchmark_matrix_command_rows(
        api_base="http://127.0.0.1:52415",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        max_tokens_values=(16, 64),
        repeats=3,
        mimo_mtp_sidecar_path="/tmp/model_mtp.safetensors",
    )

    assert len(rows) == 8
    assert rows[0]["mode"] == "ar"
    assert rows[0]["max_tokens"] == 16
    assert rows[0]["payload_extra_json"] is None
    assert "--list-models" in str(rows[0]["command"])
    mtp_row = rows[1]
    assert mtp_row["mode"] == "mtp-d1"
    assert mtp_row["max_tokens"] == 16
    assert mtp_row["payload_extra_json"] == (
        '{"mimo_mtp_depth":1,"mimo_mtp_fail_closed":true,'
        '"mimo_mtp_fastpath":true,'
        '"mimo_mtp_sidecar_path":"/tmp/model_mtp.safetensors"}'
    )
    assert "--payload-extra-json" in str(mtp_row["command"])
    assert rows[-1]["mode"] == "mtp-d3"
    assert rows[-1]["max_tokens"] == 64


def test_main_matrix_commands_prints_commands_without_contacting_cluster(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_get(_url: str, _timeout_seconds: float) -> tuple[int, dict[str, object]]:
        raise AssertionError("matrix command rendering should not contact the cluster")

    monkeypatch.setattr(cluster_bench, "_http_json_get", fail_get)

    exit_code = cluster_bench.main(
        [
            "--matrix-commands",
            "--max-tokens",
            "16",
            "--max-tokens",
            "64",
            "--repeats",
            "3",
            "--mimo-mtp-sidecar-path",
            "/tmp/model_mtp.safetensors",
        ]
    )

    assert exit_code == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(rows) == 8
    assert rows[0]["kind"] == "cluster_benchmark_command"
    assert rows[0]["mode"] == "ar"
    assert rows[1]["mode"] == "mtp-d1"
    assert rows[-1]["mode"] == "mtp-d3"
