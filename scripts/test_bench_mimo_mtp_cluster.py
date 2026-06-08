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


def _decode_json_object(raw_json: str) -> dict[str, object]:
    decoded_json = cast(object, json.loads(raw_json))
    assert isinstance(decoded_json, dict)
    return cast(dict[str, object], decoded_json)


def _expected_ar_baseline_blocker(rerun_command: str) -> dict[str, object]:
    return {
        "kind": "ar_baseline_blocker",
        "status": "blocked_with_command",
        "live_metrics_status": "unavailable",
        "non_fabrication_statement": (
            "No live benchmark metrics are claimed in this blocked row; run "
            "remediation_command before using performance, speedup, or Slice 5 evidence."
        ),
        "remediation_command": rerun_command,
        "remediation_message": (
            "AR baseline collection is blocked because the exo cluster API is unavailable; "
            "start exo or set --api-base to a running cluster, then rerun: "
            + rerun_command
        ),
    }


def _assert_blocked_error_row_has_non_fabricated_evidence(
    row: dict[str, object],
    *,
    stage: str,
    error: str,
    repeat_index: int | None,
    rerun_command: str,
) -> None:
    assert row["kind"] == "cluster_benchmark_error"
    assert row["cluster_path"] == "exo_api_bench_chat_completions"
    assert row["api_base"] == "http://127.0.0.1:52415"
    assert row["mode"] == "ar"
    assert row["repeat_index"] == repeat_index
    assert row["stage"] == stage
    assert row["error"] == error
    assert row["status"] == "blocked_with_command"
    assert row["success_row_count"] == 0
    assert row["rerun_command"] == rerun_command
    assert row["next_step"] == (
        "start or point to an exo cluster API that can serve /bench/chat/completions, "
        "then rerun the command in rerun_command"
    )
    assert row["ar_baseline_blocker"] == _expected_ar_baseline_blocker(rerun_command)
    live_metric_fields = {
        "generation_tps",
        "generation_tokens",
        "prompt_tps",
        "power_usage",
        "generation_stats",
        "mtp_speedup_budget",
        "mtp_throughput_threshold",
    }
    assert live_metric_fields.isdisjoint(row)
    assert row["live_metrics_status"] == "unavailable"
    assert row["non_fabrication_statement"] == (
        "No live benchmark metrics are claimed in this blocked row; run rerun_command before using performance, speedup, or Slice 5 evidence."
    )
    blocked_evidence = row["blocked_evidence"]
    assert isinstance(blocked_evidence, dict)
    assert blocked_evidence["evidence_kind"] == "blocked_evidence"
    assert blocked_evidence["endpoint"] == "/bench/chat/completions"
    assert blocked_evidence["runnable_command"] == rerun_command
    assert blocked_evidence["blocker_reason"] == error
    assert blocked_evidence["blocked_stage"] == stage
    assert blocked_evidence["generation_tps"] is None
    assert blocked_evidence["generation_tokens"] is None
    assert blocked_evidence["prompt_tps"] is None
    assert blocked_evidence["power_usage"] is None
    assert blocked_evidence["non_fabrication_statement"] == (
        "No performance result is claimed because the target was unavailable before benchmark execution."
    )


def _tensor_parallel_execution_path(
    *, world_size: int = 2, instance_id: str = "instance-tensor-2"
) -> dict[str, object]:
    model_id = "kernelpool/MiMo-V2.5-Pro-6bit"
    return {
        "sharding": "Tensor",
        "world_size": world_size,
        "engine": "mlx",
        "model_path": "/Users/studio2/.cache/exo/downloads/kernelpool--MiMo-V2.5-Pro-6bit",
        "instance_id": instance_id,
        "cluster_nodes": [
            {
                "node_id": f"node-{rank}",
                "runner_id": f"runner-{rank}",
                "runner_status": "RunnerReady",
            }
            for rank in range(world_size)
        ],
        "tensor_parallel_shards": [
            {
                "node_id": f"node-{rank}",
                "runner_id": f"runner-{rank}",
                "model_id": model_id,
                "device_rank": rank,
                "world_size": world_size,
                "start_layer": 0,
                "end_layer": 70,
                "n_layers": 70,
            }
            for rank in range(world_size)
        ],
    }


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


def test_bench_mimo_mtp_cluster_builds_importable_cli_parser() -> None:
    parser = cluster_bench.build_cli_parser()

    namespace = parser.parse_args(["--matrix-commands"])

    api_base = cast(str, namespace.api_base)
    mode_label = cast(str, namespace.mode_label)
    matrix_commands = cast(bool, namespace.matrix_commands)

    assert api_base == "http://127.0.0.1:52415"
    assert mode_label == "ar"
    assert matrix_commands is True
    assert (
        "Benchmark MiMo MTP work through an already-running exo cluster API"
        in parser.format_help()
    )


def test_bench_mimo_mtp_cluster_help_identifies_supported_ar_baseline_harness() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    entrypoint = repo_root / "scripts" / "bench_mimo_mtp_cluster.py"

    completed = subprocess.run(
        [sys.executable, str(entrypoint), "--help"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    normalized_stdout = " ".join(completed.stdout.split())
    assert "Supported same-cluster AR baseline harness" in normalized_stdout
    assert "default mode-label is ar" in normalized_stdout


def test_cli_ar_baseline_request_sets_max_tokens_to_16(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    posted_payloads: list[dict[str, object]] = []

    def fake_get(url: str, timeout_seconds: float) -> tuple[int, dict[str, object]]:
        assert url == "http://cluster.example/v1/models"
        assert timeout_seconds == 12.0
        return 200, {"data": [{"id": "kernelpool/MiMo-V2.5-Pro-6bit"}]}

    def fake_post(
        url: str, payload: dict[str, object], timeout_seconds: float
    ) -> tuple[int, dict[str, object]]:
        posted_payloads.append(
            {"url": url, "payload": dict(payload), "timeout": timeout_seconds}
        )
        return 200, {
            "generation_stats": {
                "generation_tps": 31.5,
                "generation_tokens": 16,
            }
        }

    monkeypatch.setattr(cluster_bench, "_http_json_get", fake_get)
    monkeypatch.setattr(cluster_bench, "_http_json_post", fake_post)

    exit_code = cluster_bench.main(
        [
            "--api-base",
            "http://cluster.example",
            "--model-id",
            "kernelpool/MiMo-V2.5-Pro-6bit",
            "--prompt",
            "hello",
            "--max-tokens",
            "16",
            "--repeats",
            "1",
            "--mode-label",
            "ar",
            "--timeout-seconds",
            "12",
        ]
    )

    assert exit_code == 0
    assert posted_payloads == [
        {
            "url": "http://cluster.example/bench/chat/completions",
            "payload": {
                "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 16,
                "stream": False,
                "temperature": 0.0,
                "benchmark_mode_label": "ar",
                "benchmark_repeat_index": 0,
            },
            "timeout": 12.0,
        }
    ]
    row = _decode_json_object(capsys.readouterr().out)
    assert row["kind"] == "cluster_benchmark_metric"
    assert row["mode"] == "ar"
    assert row["generation_tokens"] == 16


def test_cli_ar_baseline_request_sets_max_tokens_to_64(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    posted_payloads: list[dict[str, object]] = []

    def fake_get(url: str, timeout_seconds: float) -> tuple[int, dict[str, object]]:
        assert url == "http://cluster.example/v1/models"
        assert timeout_seconds == 12.0
        return 200, {"data": [{"id": "kernelpool/MiMo-V2.5-Pro-6bit"}]}

    def fake_post(
        url: str, payload: dict[str, object], timeout_seconds: float
    ) -> tuple[int, dict[str, object]]:
        posted_payloads.append(
            {"url": url, "payload": dict(payload), "timeout": timeout_seconds}
        )
        return 200, {
            "generation_stats": {
                "generation_tps": 31.5,
                "generation_tokens": 64,
            }
        }

    monkeypatch.setattr(cluster_bench, "_http_json_get", fake_get)
    monkeypatch.setattr(cluster_bench, "_http_json_post", fake_post)

    exit_code = cluster_bench.main(
        [
            "--api-base",
            "http://cluster.example",
            "--model-id",
            "kernelpool/MiMo-V2.5-Pro-6bit",
            "--prompt",
            "hello",
            "--max-tokens",
            "64",
            "--repeats",
            "1",
            "--mode-label",
            "ar",
            "--timeout-seconds",
            "12",
        ]
    )

    assert exit_code == 0
    assert posted_payloads == [
        {
            "url": "http://cluster.example/bench/chat/completions",
            "payload": {
                "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 64,
                "stream": False,
                "temperature": 0.0,
                "benchmark_mode_label": "ar",
                "benchmark_repeat_index": 0,
            },
            "timeout": 12.0,
        }
    ]
    row = _decode_json_object(capsys.readouterr().out)
    assert row["kind"] == "cluster_benchmark_metric"
    assert row["mode"] == "ar"
    assert row["generation_tokens"] == 64


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


def test_ar_baseline_metric_row_preserves_non_canonical_payload_extra_values() -> None:
    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="ar",
        repeat_index=0,
        elapsed_seconds=1.0,
        status_code=200,
        response={
            "generation_stats": {
                "generation_tps": 24.5,
                "generation_tokens": 64,
            }
        },
        payload_extra={"diagnostic_flag": True, "run_tag": "same-cluster-ar"},
        requested_max_tokens=64,
    )

    assert row["mode"] == "ar"
    assert row["generation_tps"] == 24.5
    assert row["payload_extra_keys"] == ["diagnostic_flag", "run_tag"]
    assert row["payload_extra"] == {
        "diagnostic_flag": True,
        "run_tag": "same-cluster-ar",
    }
    assert row["payload_extra_canonical_collision_keys"] == []


def test_ar_baseline_metric_row_keeps_canonical_fields_when_payload_extra_collides() -> (
    None
):
    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="ar",
        repeat_index=0,
        elapsed_seconds=1.0,
        status_code=200,
        response={
            "generation_stats": {
                "generation_tps": 24.5,
                "generation_tokens": 64,
            }
        },
        payload_extra={
            "mode": "mtp-d999",
            "generation_tps": 999.0,
            "diagnostic_flag": True,
        },
        requested_max_tokens=64,
    )

    assert row["mode"] == "ar"
    assert row["generation_tps"] == 24.5
    assert row["payload_extra"] == {
        "mode": "mtp-d999",
        "generation_tps": 999.0,
        "diagnostic_flag": True,
    }
    assert row["payload_extra_canonical_collision_keys"] == [
        "generation_tps",
        "mode",
    ]


def test_extract_response_payload_extra_preserves_unrecognized_keys() -> None:
    extra = cluster_bench.extract_response_payload_extra(
        {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "kernelpool/MiMo-V2.5-Pro-6bit",
            "choices": [],
            "usage": {"prompt_tokens": 8, "completion_tokens": 16, "total_tokens": 24},
            "generation_stats": {"generation_tps": 24.5, "generation_tokens": 64},
            "power_usage": {"elapsed_seconds": 1.0, "nodes": []},
            "custom_metadata": {"run_group": "nightly"},
            "model_version": "6bit-v3",
        }
    )
    assert extra == {
        "custom_metadata": {"run_group": "nightly"},
        "model_version": "6bit-v3",
    }


def test_extract_response_payload_extra_excludes_all_known_keys() -> None:
    response_with_only_known_keys: dict[str, object] = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "kernelpool/MiMo-V2.5-Pro-6bit",
        "choices": [],
        "usage": {"prompt_tokens": 8, "completion_tokens": 16, "total_tokens": 24},
        "service_tier": None,
        "generation_stats": {
            "generation_tps": 24.5,
            "generation_tokens": 64,
            "prompt_tps": 91.0,
            "prompt_tokens": 8,
            "peak_memory_usage": {},
        },
        "power_usage": {"elapsed_seconds": 1.0, "nodes": []},
        "accepted_execution_path": "ar",
        "mtp_enabled": False,
        "mtp_depth": None,
        "requested_mtp_depth": None,
        "mtp_sidecar_status": None,
        "mtp_disable_reason": None,
        "mtp_fallback_reason": None,
        "attempted_depth_counts": None,
        "accepted_depth_counts": None,
        "fallback_count": None,
        "timing_breakdown_seconds": None,
        "acceptance_rate": None,
        "execution_path": {"path": "exo_cluster_tensor_parallel"},
        "mtp_execution_state": "disabled_default",
    }
    extra = cluster_bench.extract_response_payload_extra(response_with_only_known_keys)
    assert extra == {}


def test_extract_response_payload_extra_returns_empty_for_empty_response() -> None:
    extra = cluster_bench.extract_response_payload_extra({})
    assert extra == {}


def test_build_cluster_metric_row_preserves_response_extra_keys_in_payload_extra() -> (
    None
):
    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="ar",
        repeat_index=0,
        elapsed_seconds=1.0,
        status_code=200,
        response={
            "generation_stats": {
                "generation_tps": 24.5,
                "generation_tokens": 64,
            },
            "custom_metadata": {"run_group": "nightly"},
            "model_version": "6bit-v3",
        },
        payload_extra={"diagnostic_flag": True},
        requested_max_tokens=64,
    )
    assert row["mode"] == "ar"
    assert row["generation_tps"] == 24.5
    assert row["payload_extra"]["diagnostic_flag"] is True
    assert row["payload_extra"]["custom_metadata"] == {"run_group": "nightly"}
    assert row["payload_extra"]["model_version"] == "6bit-v3"
    assert "custom_metadata" in row["payload_extra_keys"]
    assert "model_version" in row["payload_extra_keys"]
    assert "diagnostic_flag" in row["payload_extra_keys"]


def test_build_cluster_metric_row_response_extra_does_not_override_request_payload_extra() -> (
    None
):
    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="ar",
        repeat_index=0,
        elapsed_seconds=1.0,
        status_code=200,
        response={
            "generation_stats": {
                "generation_tps": 24.5,
                "generation_tokens": 64,
            },
            "model_version": "response-version",
        },
        payload_extra={"model_version": "request-version", "run_tag": "baseline"},
        requested_max_tokens=64,
    )
    assert row["payload_extra"]["model_version"] == "request-version"
    assert row["payload_extra"]["run_tag"] == "baseline"


def test_canonical_benchmark_row_contract_accepts_valid_ar_row() -> None:
    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="ar",
        repeat_index=0,
        elapsed_seconds=1.0,
        status_code=200,
        response={
            "generation_stats": {
                "prompt_tps": 100.0,
                "generation_tps": 24.5,
                "generation_tokens": 64,
            }
        },
        payload_extra={},
        requested_max_tokens=64,
        benchmark_session_id="session-ar-contract",
        prompt_hash="sha256:test-prompt",
    )

    assert row["schema_version"] == cluster_bench.CANONICAL_BENCHMARK_ROW_SCHEMA_VERSION
    assert row["evidence_kind"] == "benchmark_row"
    assert row["row_status"] == "live"
    assert row["api_url"] == "http://cluster.example"
    assert row["endpoint"] == "/bench/chat/completions"
    assert row["temperature"] == 0.0
    assert row["mtp_enabled"] is False
    assert row["mtp_execution_state"] == "disabled_default"
    assert row["telemetry_completeness"] == "complete"

    validation = cluster_bench.validate_canonical_benchmark_row(row)

    assert validation == {
        "kind": "canonical_benchmark_row_validation",
        "valid": True,
        "schema_version": cluster_bench.CANONICAL_BENCHMARK_ROW_SCHEMA_VERSION,
        "row_status": "live",
        "missing_required_fields": [],
        "invalid_fields": [],
        "required_one_of_missing": [],
    }


def test_canonical_benchmark_row_contract_rejects_missing_required_field() -> None:
    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="ar",
        repeat_index=0,
        elapsed_seconds=1.0,
        status_code=200,
        response={
            "generation_stats": {
                "prompt_tps": 100.0,
                "generation_tps": 24.5,
                "generation_tokens": 64,
            }
        },
        payload_extra={},
        requested_max_tokens=64,
        benchmark_session_id="session-ar-contract",
        prompt_hash="sha256:test-prompt",
    )
    del row["model_id"]

    validation = cluster_bench.validate_canonical_benchmark_row(row)

    assert validation["valid"] is False
    assert validation["missing_required_fields"] == ["model_id"]
    assert validation["invalid_fields"] == []


def test_canonical_benchmark_row_contract_rejects_missing_each_required_field() -> (
    None,
):
    """Omitting any single required field must make validation fail and list that field as missing."""
    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="ar",
        repeat_index=0,
        elapsed_seconds=1.0,
        status_code=200,
        response={
            "generation_stats": {
                "prompt_tps": 100.0,
                "generation_tps": 24.5,
                "generation_tokens": 64,
            }
        },
        payload_extra={},
        requested_max_tokens=64,
        benchmark_session_id="session-ar-contract",
        prompt_hash="sha256:test-prompt",
    )
    validation = cluster_bench.validate_canonical_benchmark_row(row)
    assert validation["valid"] is True, (
        "baseline row must be valid before field-removal checks"
    )
    for required_field in cluster_bench.CANONICAL_BENCHMARK_ROW_REQUIRED_FIELDS:
        incomplete_row = dict(row)
        del incomplete_row[required_field]
        incomplete_validation = cluster_bench.validate_canonical_benchmark_row(
            incomplete_row,
        )
        assert incomplete_validation["valid"] is False, (
            f"row must be invalid when required field {required_field!r} is omitted"
        )
        assert required_field in incomplete_validation["missing_required_fields"], (
            f"required field {required_field!r} must appear in missing_required_fields"
        )


def test_canonical_benchmark_row_contract_rejects_missing_each_required_one_of_group() -> (
    None,
):
    """Removing all fields in a required-one-of group must make validation fail."""
    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="ar",
        repeat_index=0,
        elapsed_seconds=1.0,
        status_code=200,
        response={
            "generation_stats": {
                "prompt_tps": 100.0,
                "generation_tps": 24.5,
                "generation_tokens": 64,
            }
        },
        payload_extra={},
        requested_max_tokens=64,
        benchmark_session_id="session-ar-contract",
        prompt_hash="sha256:test-prompt",
    )
    for group in cluster_bench._CANONICAL_BENCHMARK_ROW_REQUIRED_ONE_OF:
        incomplete_row = dict(row)
        for field in group:
            incomplete_row.pop(field, None)
        incomplete_validation = cluster_bench.validate_canonical_benchmark_row(
            incomplete_row,
        )
        assert incomplete_validation["valid"] is False, (
            f"row must be invalid when all fields in required-one-of group "
            f"{group!r} are omitted"
        )
        assert list(group) in incomplete_validation["required_one_of_missing"], (
            f"required-one-of group {group!r} must appear in required_one_of_missing"
        )


def test_canonical_benchmark_row_contract_rejects_unknown_row_status() -> None:
    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="ar",
        repeat_index=0,
        elapsed_seconds=1.0,
        status_code=200,
        response={
            "generation_stats": {
                "prompt_tps": 100.0,
                "generation_tps": 24.5,
                "generation_tokens": 64,
            }
        },
        payload_extra={},
        requested_max_tokens=64,
        benchmark_session_id="session-ar-contract",
        prompt_hash="sha256:test-prompt",
    )
    row["row_status"] = "successful_mtp"

    validation = cluster_bench.validate_canonical_benchmark_row(row)

    assert validation["valid"] is False
    assert validation["invalid_fields"] == [
        {
            "field": "row_status",
            "reason": "must be one of: blocked, fail_closed, fail_open, invalid, live",
            "value": "successful_mtp",
        }
    ]


def test_canonical_benchmark_row_classification_recognizes_valid_ar_row() -> None:
    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="ar",
        repeat_index=0,
        elapsed_seconds=1.0,
        status_code=200,
        response={
            "generation_stats": {
                "prompt_tps": 101.0,
                "generation_tps": 25.0,
                "generation_tokens": 32,
            }
        },
        payload_extra={},
        requested_max_tokens=32,
        benchmark_session_id="session-ar-classification",
        prompt_hash="sha256:test-prompt",
    )

    classification = cluster_bench.classify_canonical_benchmark_row(row)

    assert classification == {
        "kind": "canonical_benchmark_row_classification",
        "valid": True,
        "row_classification": "ar_live",
        "mode": "ar",
        "mtp_execution_state": "disabled_default",
        "telemetry_required_fields_missing": [],
        "cluster_identity_fields_present": ["api_url", "prompt_hash"],
    }


def test_canonical_benchmark_row_validation_requires_ar_identity_and_throughput_fields() -> (
    None
):
    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="ar",
        repeat_index=0,
        elapsed_seconds=1.0,
        status_code=200,
        response={
            "generation_stats": {
                "prompt_tps": 101.0,
                "generation_tps": 25.0,
                "generation_tokens": 32,
            }
        },
        payload_extra={},
        requested_max_tokens=32,
        benchmark_session_id="session-ar-classification",
        prompt_hash="sha256:test-prompt",
    )
    del row["api_url"]
    del row["prompt_hash"]
    row["generation_tps"] = None

    validation = cluster_bench.validate_canonical_benchmark_row(row)
    classification = cluster_bench.classify_canonical_benchmark_row(row)

    assert validation["valid"] is False
    assert validation["required_one_of_missing"] == [
        ["api_url", "cluster_id"],
        ["prompt_id", "prompt_hash"],
    ]
    assert {
        "field": "generation_tps",
        "reason": "live rows require numeric timing/throughput telemetry",
        "value": None,
    } in validation["invalid_fields"]
    assert classification["row_classification"] == "invalid"


def test_canonical_benchmark_row_classification_recognizes_successful_mtp_row() -> None:
    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="mtp-d2",
        repeat_index=0,
        elapsed_seconds=1.0,
        status_code=200,
        response={
            "accepted_execution_path": "mimo_mtp_fastpath",
            "generation_stats": {
                "prompt_tps": 99.0,
                "generation_tps": 31.0,
                "generation_tokens": 64,
                "mtp_enabled": True,
                "mtp_depth": 2,
                "mtp_sidecar_status": "loaded",
                "attempted_depth_counts": {"2": 20},
                "accepted_depth_counts": {"2": 12},
                "acceptance_rate": 0.6,
                "fallback_count": 0,
                "timing_breakdown_seconds": {"proposal": 0.2, "verifier": 0.3},
            },
        },
        payload_extra={"mimo_mtp_fastpath": True, "mimo_mtp_depth": 2},
        requested_max_tokens=64,
        benchmark_session_id="session-mtp-classification",
        prompt_hash="sha256:test-prompt",
    )

    classification = cluster_bench.classify_canonical_benchmark_row(row)

    assert classification["valid"] is True
    assert classification["row_classification"] == "mtp_live_successful"
    assert classification["telemetry_required_fields_missing"] == []


def test_canonical_benchmark_row_validation_rejects_successful_mtp_without_fastpath_telemetry() -> (
    None
):
    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="mtp-d2",
        repeat_index=0,
        elapsed_seconds=1.0,
        status_code=200,
        response={
            "generation_stats": {
                "prompt_tps": 99.0,
                "generation_tps": 31.0,
                "generation_tokens": 64,
            },
        },
        payload_extra={"mimo_mtp_fastpath": True, "mimo_mtp_depth": 2},
        requested_max_tokens=64,
        benchmark_session_id="session-mtp-invalid",
        prompt_hash="sha256:test-prompt",
    )
    row["mtp_enabled"] = True
    row["mtp_depth"] = 2
    row["mtp_execution_state"] = "successful_mtp"

    validation = cluster_bench.validate_canonical_benchmark_row(row)
    classification = cluster_bench.classify_canonical_benchmark_row(row)

    assert validation["valid"] is False
    assert {
        "field": "mtp_execution_state",
        "reason": "successful_mtp rows require fastpath attempt telemetry and benchmark-grade MTP telemetry fields",
        "value": "successful_mtp",
    } in validation["invalid_fields"]
    assert classification["row_classification"] == "invalid"
    assert classification["telemetry_required_fields_missing"] == [
        "accepted_execution_path=mimo_mtp_fastpath",
        "accepted_depth_counts",
        "acceptance_rate",
        "attempted_depth_counts",
        "fallback_count",
        "mtp_sidecar_status",
        "timing_breakdown_seconds",
    ]


def test_canonical_benchmark_row_classification_recognizes_mtp_fail_closed_row() -> (
    None
):
    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="mtp-d2",
        repeat_index=0,
        elapsed_seconds=1.0,
        status_code=200,
        response={
            "generation_stats": {
                "prompt_tps": 99.0,
                "generation_tps": 28.0,
                "generation_tokens": 64,
                "mtp_enabled": False,
                "mtp_depth": 2,
                "mtp_disable_reason": "missing_sidecar",
                "mtp_sidecar_status": "missing",
            },
        },
        payload_extra={"mimo_mtp_fastpath": True, "mimo_mtp_depth": 2},
        requested_max_tokens=64,
        benchmark_session_id="session-mtp-fail-closed",
        prompt_hash="sha256:test-prompt",
    )
    row["row_status"] = "fail_closed"

    classification = cluster_bench.classify_canonical_benchmark_row(row)

    assert classification["valid"] is True
    assert classification["row_classification"] == "mtp_fail_closed"
    assert classification["telemetry_required_fields_missing"] == []


def test_run_cluster_benchmark_labels_ar_requests_and_results() -> None:
    posted_payloads: list[dict[str, object]] = []

    def fake_get(url: str, timeout_seconds: float) -> tuple[int, dict[str, object]]:
        assert url == "http://cluster.example/v1/models"
        assert timeout_seconds == 12.0
        return 200, {"data": [{"id": "kernelpool/MiMo-V2.5-Pro-6bit"}]}

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


def test_run_cluster_benchmark_posts_to_exact_bench_chat_completions_endpoint() -> None:
    posted_urls: list[str] = []

    def fake_get(url: str, timeout_seconds: float) -> tuple[int, dict[str, object]]:
        assert url == "http://cluster.example/v1/models"
        assert timeout_seconds == 12.0
        return 200, {"data": [{"id": "kernelpool/MiMo-V2.5-Pro-6bit"}]}

    def fake_post(
        url: str, _payload: dict[str, object], _timeout_seconds: float
    ) -> tuple[int, dict[str, object]]:
        posted_urls.append(url)
        return 200, {
            "generation_stats": {"generation_tps": 1.0, "generation_tokens": 1}
        }

    timer = _timer([0.0, 1.0])
    cluster_bench.run_cluster_benchmark(
        args=cluster_bench.ClusterBenchmarkArgs(
            api_base="http://cluster.example/",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            prompt="hello",
            max_tokens=1,
            matrix_max_tokens=(1,),
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

    assert posted_urls == ["http://cluster.example/bench/chat/completions"]


def test_ar_baseline_metric_row_parses_top_level_metric_telemetry() -> None:
    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="ar",
        repeat_index=0,
        elapsed_seconds=1.0,
        status_code=200,
        response={
            "generation_tps": 27.25,
            "generation_tokens": 16,
            "prompt_tps": 91.5,
            "power_usage": {
                "elapsed_seconds": 1.0,
                "nodes": [],
                "total_avg_sys_power_watts": 42.0,
                "total_energy_joules": 42.0,
            },
        },
        payload_extra_keys=[],
    )

    assert row["mode"] == "ar"
    assert row["generation_tps"] == 27.25
    assert row["generation_tokens"] == 16
    assert row["prompt_tps"] == 91.5
    assert row["power_usage"] == {
        "elapsed_seconds": 1.0,
        "nodes": [],
        "total_avg_sys_power_watts": 42.0,
        "total_energy_joules": 42.0,
    }


def test_ar_baseline_metric_row_exposes_budget_ready_schema_aliases() -> None:
    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="ar",
        repeat_index=2,
        elapsed_seconds=1.25,
        status_code=200,
        response={
            "generation_stats": {
                "prompt_tps": 100.0,
                "generation_tps": 31.5,
                "prompt_tokens": 8,
                "generation_tokens": 16,
            },
            "power_usage": {"elapsed_seconds": 1.0, "nodes": []},
        },
        payload_extra_keys=["diagnostic_flag"],
    )

    assert row["model"] == "kernelpool/MiMo-V2.5-Pro-6bit"
    assert row["model_id"] == "kernelpool/MiMo-V2.5-Pro-6bit"
    assert row["mode"] == "ar"
    assert row["mode_label"] == "ar"
    assert row["repeat_index"] == 2
    assert row["generation_tps"] == 31.5
    assert row["generation_tokens"] == 16
    assert row["prompt_tps"] == 100.0
    assert row["power_usage"] == {"elapsed_seconds": 1.0, "nodes": []}
    assert row["payload_extra_keys"] == ["diagnostic_flag"]


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
        return 200, {"data": [{"id": "kernelpool/MiMo-V2.5-Pro-6bit"}]}

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
            matrix_max_tokens=(4,),
            repeats=3,
            mode_label="mtp-d1",
            timeout_seconds=12.0,
            payload_extra_json=json.dumps(
                {
                    "mimo_mtp_fastpath": True,
                    "mimo_mtp_depth": 1,
                    "mimo_mtp_fail_closed": True,
                    "benchmark_repeat_index": 99,
                }
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
        row["payload_extra_keys"]
        == [
            "benchmark_repeat_index",
            "mimo_mtp_depth",
            "mimo_mtp_fail_closed",
            "mimo_mtp_fastpath",
        ]
        for row in rows
    )


def test_ar_baseline_available_cluster_plan_runs_every_repeat_for_each_token_budget() -> (
    None
):
    posted_payloads: list[dict[str, object]] = []

    def fake_get(url: str, timeout_seconds: float) -> tuple[int, dict[str, object]]:
        assert url == "http://cluster.example/v1/models"
        assert timeout_seconds == 12.0
        return 200, {"data": [{"id": "kernelpool/MiMo-V2.5-Pro-6bit"}]}

    def fake_post(
        url: str, payload: dict[str, object], timeout_seconds: float
    ) -> tuple[int, dict[str, object]]:
        assert url == "http://cluster.example/bench/chat/completions"
        assert timeout_seconds == 12.0
        posted_payloads.append(dict(payload))
        return 200, {
            "generation_stats": {
                "generation_tps": 20.0 + len(posted_payloads),
                "generation_tokens": payload["max_tokens"],
            },
            "execution_path": {"sharding": "Tensor", "world_size": 2},
        }

    timer = _timer([float(index) for index in range(8)])
    rows = cluster_bench.run_cluster_benchmark(
        args=cluster_bench.ClusterBenchmarkArgs(
            api_base="http://cluster.example",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            prompt="hello",
            max_tokens=64,
            matrix_max_tokens=(16, 64),
            repeats=2,
            mode_label="ar",
            timeout_seconds=12.0,
            payload_extra_json=None,
            list_models=False,
        ),
        http_get=fake_get,
        http_post=fake_post,
        timer=lambda: next(timer),
    )

    metric_rows = [row for row in rows if row["kind"] == "cluster_benchmark_metric"]
    assert [payload["max_tokens"] for payload in posted_payloads] == [16, 16, 64, 64]
    assert [payload["benchmark_repeat_index"] for payload in posted_payloads] == [
        0,
        1,
        0,
        1,
    ]
    assert all(payload["benchmark_mode_label"] == "ar" for payload in posted_payloads)
    assert all("placement" not in payload for payload in posted_payloads)
    assert [row["requested_max_tokens"] for row in metric_rows] == [16, 16, 64, 64]
    assert [row["repeat_index"] for row in metric_rows] == [0, 1, 0, 1]


def test_multi_budget_persistence_blocker_preserves_full_rerun_command(
    tmp_path: Path,
) -> None:
    def fake_get(_url: str, _timeout_seconds: float) -> tuple[int, dict[str, object]]:
        raise OSError("connection refused")

    def fake_post(
        _url: str, _payload: dict[str, object], _timeout_seconds: float
    ) -> tuple[int, dict[str, object]]:
        raise AssertionError("benchmark must not run after unavailable cluster probe")

    rows = cluster_bench.run_cluster_benchmark(
        args=cluster_bench.ClusterBenchmarkArgs(
            api_base="http://127.0.0.1:52415",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            prompt="hello",
            max_tokens=64,
            matrix_max_tokens=(16, 64),
            repeats=2,
            mode_label="ar",
            timeout_seconds=1.0,
            payload_extra_json=None,
            list_models=True,
            output_dir=tmp_path,
        ),
        http_get=fake_get,
        http_post=fake_post,
    )

    expected_command = (
        "uv run python3 scripts/bench_mimo_mtp_cluster.py "
        "--api-base http://127.0.0.1:52415 "
        "--model-id kernelpool/MiMo-V2.5-Pro-6bit "
        "--prompt hello "
        "--max-tokens 16 --max-tokens 64 "
        "--repeats 2 --mode-label ar --list-models "
        f"--output-dir {tmp_path}"
    )
    assert rows[0]["rerun_command"] == expected_command
    assert rows[0]["ar_baseline_blocker"] == _expected_ar_baseline_blocker(
        expected_command
    )


def test_persist_ar_baseline_rows_saves_each_repeat_for_each_requested_token_budget(
    tmp_path: Path,
) -> None:
    posted_max_tokens: list[int] = []

    def fake_get(_url: str, _timeout_seconds: float) -> tuple[int, dict[str, object]]:
        return 200, {"data": [{"id": "kernelpool/MiMo-V2.5-Pro-6bit"}]}

    def fake_post(
        _url: str, payload: dict[str, object], _timeout_seconds: float
    ) -> tuple[int, dict[str, object]]:
        max_tokens = payload["max_tokens"]
        assert isinstance(max_tokens, int)
        posted_max_tokens.append(max_tokens)
        return 200, {
            "generation_stats": {
                "prompt_tps": 100.0,
                "generation_tps": float(max_tokens),
                "generation_tokens": max_tokens,
            },
            "execution_path": {"sharding": "Tensor", "world_size": 2},
        }

    timer = _timer([float(value) for value in range(8)])
    rows = cluster_bench.run_cluster_benchmark(
        args=cluster_bench.ClusterBenchmarkArgs(
            api_base="http://cluster.example",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            prompt="hello",
            max_tokens=64,
            matrix_max_tokens=(16, 64),
            repeats=2,
            mode_label="ar",
            timeout_seconds=12.0,
            payload_extra_json=None,
            list_models=False,
            output_dir=tmp_path,
        ),
        http_get=fake_get,
        http_post=fake_post,
        timer=lambda: next(timer),
    )

    metric_rows = [row for row in rows if row["kind"] == "cluster_benchmark_metric"]
    assert posted_max_tokens == [16, 16, 64, 64]
    assert [(row["max_tokens"], row["repeat_index"]) for row in metric_rows] == [
        (16, 0),
        (16, 1),
        (64, 0),
        (64, 1),
    ]

    for max_tokens in (16, 64):
        output_file = tmp_path / f"ar-baseline-max-tokens-{max_tokens}.jsonl"
        saved_rows = [
            _decode_json_object(line) for line in output_file.read_text().splitlines()
        ]
        assert [row["repeat_index"] for row in saved_rows] == [0, 1]
        assert all(row["mode"] == "ar" for row in saved_rows)
        assert all(row["max_tokens"] == max_tokens for row in saved_rows)
        assert all(row["generation_tokens"] == max_tokens for row in saved_rows)


def test_collect_ar_baseline_rows_defaults_to_required_token_budgets_and_persists(
    tmp_path: Path,
) -> None:
    posted_payloads: list[dict[str, object]] = []

    def fake_get(_url: str, _timeout_seconds: float) -> tuple[int, dict[str, object]]:
        return 200, {"data": [{"id": "kernelpool/MiMo-V2.5-Pro-6bit"}]}

    def fake_post(
        _url: str, payload: dict[str, object], _timeout_seconds: float
    ) -> tuple[int, dict[str, object]]:
        posted_payloads.append(dict(payload))
        max_tokens = payload["max_tokens"]
        assert isinstance(max_tokens, int)
        return 200, {
            "generation_stats": {
                "prompt_tps": 100.0,
                "generation_tps": float(max_tokens),
                "generation_tokens": max_tokens,
            },
            "execution_path": _tensor_parallel_execution_path(),
        }

    timer = _timer([float(value) for value in range(4)])
    rows = cluster_bench.collect_ar_baseline_rows(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        prompt="hello",
        repeats=1,
        timeout_seconds=12.0,
        output_dir=tmp_path,
        http_get=fake_get,
        http_post=fake_post,
        timer=lambda: next(timer),
    )

    metric_rows = [row for row in rows if row["kind"] == "cluster_benchmark_metric"]
    assert [payload["max_tokens"] for payload in posted_payloads] == [16, 64]
    assert all(payload["benchmark_mode_label"] == "ar" for payload in posted_payloads)
    assert [row["max_tokens"] for row in metric_rows] == [16, 64]
    assert all(row["mode"] == "ar" for row in metric_rows)
    assert all(row["accepted_execution_path"] == "ar" for row in metric_rows)

    for max_tokens in (16, 64):
        output_file = tmp_path / f"ar-baseline-max-tokens-{max_tokens}.jsonl"
        saved_rows = [
            _decode_json_object(line) for line in output_file.read_text().splitlines()
        ]
        assert len(saved_rows) == 1
        assert saved_rows[0]["kind"] == "cluster_benchmark_metric"
        assert saved_rows[0]["mode"] == "ar"
        assert saved_rows[0]["max_tokens"] == max_tokens
        assert saved_rows[0]["generation_tokens"] == max_tokens


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


def test_validate_rollout_baseline_flags_cluster_ar_metric_without_tensor_parallel_classification() -> (
    None
):
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
        "allowed": False,
        "baseline_source": "unknown_model_path",
        "reason": "cluster AR row is missing tensor-parallel execution-path evidence",
        "required_source": "exo cluster /bench/chat/completions AR row",
        "next_step": "collect an AR row whose response.execution_path reports Tensor sharding with world_size > 1",
    }


def test_validate_rollout_baseline_allows_cluster_ar_metric_with_tensor_parallel_classification() -> (
    None
):
    validation = cluster_bench.validate_rollout_baseline_row(
        {
            "kind": "cluster_benchmark_metric",
            "cluster_path": "exo_api_bench_chat_completions",
            "mode": "ar",
            "model": "kernelpool/MiMo-V2.5-Pro-6bit",
            "generation_tps": 22.0,
            "generation_tokens": 16,
            "model_path_classification": {
                "path": "exo_cluster_tensor_parallel",
                "is_exo_cluster_tensor_parallel": True,
                "source": "response.execution_path",
                "sharding": "Tensor",
                "world_size": 2,
                "model_path": "/Users/studio2/.cache/exo/downloads/kernelpool--MiMo-V2.5-Pro-6bit",
                "disable_reason": None,
            },
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


def test_validate_live_model_path_identifies_exo_cluster_tensor_parallelization() -> (
    None
):
    validation = cluster_bench.validate_live_model_path(
        {
            "execution_path": _tensor_parallel_execution_path(
                world_size=4, instance_id="instance-tensor-4"
            )
        }
    )

    assert validation == {
        "kind": "live_model_path_validation",
        "allowed": True,
        "live_model_path": "exo_cluster_tensor_parallel",
        "is_exo_cluster_tensor_parallel": True,
        "source": "response.execution_path",
        "sharding": "Tensor",
        "world_size": 4,
        "model_path": "/Users/studio2/.cache/exo/downloads/kernelpool--MiMo-V2.5-Pro-6bit",
        "instance_id": "instance-tensor-4",
        "reason": "exo cluster tensor parallelization is the live model path",
        "disable_reason": None,
        "next_step": "use this row as eligible same-cluster rollout evidence only with matching guarded MTP rows",
    }


def test_validate_live_model_path_rejects_tensor_world_size_without_participant_evidence() -> (
    None
):
    validation = cluster_bench.validate_live_model_path(
        {
            "execution_path": {
                "sharding": "Tensor",
                "world_size": 4,
                "engine": "mlx",
                "model_path": "/Users/studio2/.cache/exo/downloads/kernelpool--MiMo-V2.5-Pro-6bit",
                "instance_id": "legacy-tensor-world-size-only",
            }
        }
    )

    assert validation == {
        "kind": "live_model_path_validation",
        "allowed": False,
        "live_model_path": "single_studio_full_model_load",
        "is_exo_cluster_tensor_parallel": False,
        "source": "response.execution_path",
        "sharding": "Tensor",
        "world_size": 4,
        "model_path": "/Users/studio2/.cache/exo/downloads/kernelpool--MiMo-V2.5-Pro-6bit",
        "instance_id": "legacy-tensor-world-size-only",
        "reason": "tensor-parallel execution path lacks multi-participant tensor shard evidence",
        "disable_reason": "response execution path does not prove multiple cluster participants host tensor shards for the requested model",
        "next_step": "collect an AR row with scripts/bench_mimo_mtp_cluster.py against a running tensor-parallel exo cluster API",
    }


def test_validate_live_model_path_rejects_self_reported_single_studio_full_model_even_with_tensor_world_size() -> (
    None
):
    validation = cluster_bench.validate_live_model_path(
        {
            "execution_path": {
                "path": "single_studio_full_model_load",
                "sharding": "Tensor",
                "world_size": 2,
                "engine": "mlx",
                "model_path": "/Users/studio2/.cache/exo/downloads/kernelpool--MiMo-V2.5-Pro-6bit",
                "instance_id": "single-studio-diagnostic",
            }
        }
    )

    assert validation == {
        "kind": "live_model_path_validation",
        "allowed": False,
        "live_model_path": "single_studio_full_model_load",
        "is_exo_cluster_tensor_parallel": False,
        "source": "response.execution_path",
        "sharding": "Tensor",
        "world_size": 2,
        "model_path": "/Users/studio2/.cache/exo/downloads/kernelpool--MiMo-V2.5-Pro-6bit",
        "instance_id": "single-studio-diagnostic",
        "reason": "single-Studio full-model loading is not valid tensor-parallel evidence",
        "disable_reason": "response execution path self-reported single-Studio full-model loading",
        "next_step": "collect an AR row with scripts/bench_mimo_mtp_cluster.py against a running tensor-parallel exo cluster API",
    }


def test_validate_live_model_path_rejects_non_cluster_full_model_path_even_with_tensor_world_size() -> (
    None
):
    validation = cluster_bench.validate_live_model_path(
        {
            "execution_path": {
                "path": "local_full_model_load",
                "sharding": "Tensor",
                "world_size": 8,
                "engine": "mlx",
                "model_path": "/Users/studio2/.cache/exo/downloads/kernelpool--MiMo-V2.5-Pro-6bit",
                "instance_id": "local-full-model-diagnostic",
            }
        }
    )

    assert validation == {
        "kind": "live_model_path_validation",
        "allowed": False,
        "live_model_path": "local_full_model_load",
        "is_exo_cluster_tensor_parallel": False,
        "source": "response.execution_path",
        "sharding": "Tensor",
        "world_size": 8,
        "model_path": "/Users/studio2/.cache/exo/downloads/kernelpool--MiMo-V2.5-Pro-6bit",
        "instance_id": "local-full-model-diagnostic",
        "reason": "non-cluster full-model loading is not valid tensor-parallel evidence",
        "disable_reason": "response execution path is a non-cluster full-model path",
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
            "execution_path": _tensor_parallel_execution_path(),
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


def test_cluster_metric_row_emits_live_model_path_validation_result() -> None:
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
            },
            "execution_path": _tensor_parallel_execution_path(),
        },
        payload_extra_keys=[],
    )

    assert row["live_model_path_validation"] == {
        "kind": "live_model_path_validation",
        "allowed": True,
        "live_model_path": "exo_cluster_tensor_parallel",
        "is_exo_cluster_tensor_parallel": True,
        "source": "response.execution_path",
        "sharding": "Tensor",
        "world_size": 2,
        "model_path": "/Users/studio2/.cache/exo/downloads/kernelpool--MiMo-V2.5-Pro-6bit",
        "instance_id": "instance-tensor-2",
        "reason": "exo cluster tensor parallelization is the live model path",
        "disable_reason": None,
        "next_step": "use this row as eligible same-cluster rollout evidence only with matching guarded MTP rows",
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


def test_classify_same_cluster_comparability_requires_matching_contract_fields() -> (
    None
):
    base_ar_row = {
        "kind": "cluster_benchmark_metric",
        "cluster_path": "exo_api_bench_chat_completions",
        "api_base": "http://127.0.0.1:52415",
        "endpoint": "/bench/chat/completions",
        "benchmark_session_id": "session-a",
        "run_window_id": "window-a",
        "repeat_policy": "repeats=3",
        "mode": "ar",
        "model_id": "kernelpool/MiMo-V2.5-Pro-6bit",
        "prompt_hash": "sha256:prompt-a",
        "temperature": 0.0,
        "max_tokens": 64,
        "request_schema": "chat_completions_v1",
        "node_topology": {"nodes": ["studio1", "studio2"]},
        "generation_tps": 28.0,
        "generation_tokens": 64,
    }
    base_mtp_row = {
        **base_ar_row,
        "mode": "mtp-d1",
        "accepted_execution_path": "mimo_mtp_fastpath",
        "generation_tps": 34.0,
    }

    assert cluster_bench.classify_same_cluster_comparability(
        ar_row=base_ar_row, mtp_row=base_mtp_row
    ) == {
        "kind": "same_cluster_comparability_classification",
        "classification": "comparable",
        "same_cluster_comparison": True,
        "blocked_reasons": [],
        "ambiguous_reasons": [],
        "matched_fields": [
            "cluster_identity",
            "endpoint",
            "model_id",
            "prompt",
            "temperature",
            "max_tokens_bucket",
            "request_schema",
            "benchmark_session_or_run_window",
            "repeat_policy",
            "node_topology",
        ],
        "next_step": "rows are comparable for AR-vs-MTP budget analysis",
    }

    different_cluster_row = {**base_mtp_row, "api_base": "http://other.cluster:52415"}
    assert (
        cluster_bench.classify_same_cluster_comparability(
            ar_row=base_ar_row, mtp_row=different_cluster_row
        )["classification"]
        == "ambiguous"
    )
    assert (
        "cluster_identity_mismatch"
        in cluster_bench.classify_same_cluster_comparability(
            ar_row=base_ar_row, mtp_row=different_cluster_row
        )["ambiguous_reasons"]
    )

    missing_cluster_row = {**base_ar_row}
    del missing_cluster_row["api_base"]
    assert cluster_bench.classify_same_cluster_comparability(
        ar_row=missing_cluster_row, mtp_row=base_mtp_row
    ) == {
        "kind": "same_cluster_comparability_classification",
        "classification": "blocked",
        "same_cluster_comparison": False,
        "blocked_reasons": ["missing_cluster_identity"],
        "ambiguous_reasons": [],
        "matched_fields": [
            "endpoint",
            "model_id",
            "prompt",
            "temperature",
            "max_tokens_bucket",
            "request_schema",
            "benchmark_session_or_run_window",
            "repeat_policy",
            "node_topology",
        ],
        "next_step": "rerun AR and guarded MTP rows with api_base or cluster_id before making speedup claims",
    }


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
            "accepted_execution_path": "mimo_mtp_fastpath",
            "telemetry_completeness": "complete",
            "mtp_enabled": True,
            "mtp_depth": 2,
            "mtp_execution_state": "successful_mtp",
            "mtp_sidecar_status": "ready",
            "fallback_count": 0,
            "timing_breakdown_seconds": {
                "proposal": 0.10,
                "verification": 0.20,
                "acceptance": 0.02,
                "fallback": 0.0,
            },
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


def test_classify_mtp_vs_ar_baseline_refuses_pass_when_mtp_telemetry_is_partial() -> (
    None
):
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
            "generation_tps": 44.0,
            "generation_tokens": 64,
            "accepted_execution_path": "mimo_mtp_fastpath",
            "telemetry_completeness": "partial",
            "generation_stats": {
                "acceptance_rate": 0.82,
            },
        },
    )

    assert classification == {
        "kind": "mtp_vs_ar_baseline_classification",
        "classification": "ambiguous",
        "bottlenecks": ["incomplete_mtp_telemetry"],
        "ar_generation_tps": 22.0,
        "ar_ms_per_token": 45.4545,
        "mtp_generation_tps": 44.0,
        "mtp_ms_per_token": 22.7273,
        "speedup_ratio": None,
        "target_gap_tps_30": 0.0,
        "target_gap_tps_40": 0.0,
        "same_cluster_comparison": True,
        "reason": "MTP row lacks benchmark-grade fastpath telemetry; cannot classify pass",
        "next_step": "rerun guarded MTP benchmark until telemetry_completeness=complete and depth, acceptance, fallback, sidecar, and timing telemetry are present",
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


def test_classify_mtp_vs_ar_baseline_emits_proposal_too_slow_from_stage_timing() -> (
    None
):
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
            "mode": "mtp-d2",
            "model": "kernelpool/MiMo-V2.5-Pro-6bit",
            "generation_tps": 24.0,
            "generation_tokens": 64,
            "generation_stats": {
                "acceptance_rate": 0.80,
                "mimo_mtp_proposed_tokens": 20,
                "timing_breakdown_seconds": {
                    "proposal": 2.0,
                    "verification": 0.1,
                    "acceptance": 0.02,
                    "fallback": 0.0,
                },
            },
        },
    )

    assert classification["classification"] == "fail"
    assert classification["bottlenecks"] == ["proposal_too_slow"]


def test_classify_mtp_vs_ar_baseline_emits_verifier_too_slow_from_stage_timing() -> (
    None
):
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
            "mode": "mtp-d2",
            "model": "kernelpool/MiMo-V2.5-Pro-6bit",
            "generation_tps": 24.0,
            "generation_tokens": 64,
            "generation_stats": {
                "acceptance_rate": 0.80,
                "timing_breakdown_seconds": {
                    "proposal": 0.1,
                    "verification": 0.7,
                    "acceptance": 0.02,
                    "fallback": 0.0,
                },
            },
        },
    )

    assert classification["classification"] == "fail"
    assert classification["bottlenecks"] == ["verifier_too_slow"]


def test_classify_mtp_vs_ar_baseline_does_not_emit_stage_speed_labels_without_stage_timing() -> (
    None
):
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
            "mode": "mtp-d2",
            "model": "kernelpool/MiMo-V2.5-Pro-6bit",
            "generation_tps": 24.0,
            "generation_tokens": 64,
        },
    )

    assert classification["classification"] == "fail"
    assert classification["bottlenecks"] == []


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


def test_classify_mtp_vs_ar_baseline_marks_non_live_mtp_row_ambiguous_even_with_tps() -> (
    None
):
    classification = cluster_bench.classify_mtp_vs_ar_baseline(
        ar_row={
            "kind": "cluster_benchmark_metric",
            "cluster_path": "exo_api_bench_chat_completions",
            "mode": "ar",
            "model": "kernelpool/MiMo-V2.5-Pro-6bit",
            "generation_tps": 28.0,
            "generation_tokens": 64,
            "accepted_execution_path": "ar",
        },
        mtp_row={
            "kind": "cluster_benchmark_metric",
            "cluster_path": "exo_api_bench_chat_completions",
            "mode": "mtp-d1",
            "model": "kernelpool/MiMo-V2.5-Pro-6bit",
            "generation_tps": 100.0,
            "generation_tokens": 64,
            "accepted_execution_path": "ar_fallback",
        },
    )

    assert classification == {
        "kind": "mtp_vs_ar_baseline_classification",
        "classification": "ambiguous",
        "bottlenecks": ["absent_live_mtp_fastpath"],
        "ar_generation_tps": 28.0,
        "ar_ms_per_token": 35.7143,
        "mtp_generation_tps": 100.0,
        "mtp_ms_per_token": 10.0,
        "speedup_ratio": None,
        "target_gap_tps_30": 0.0,
        "target_gap_tps_40": 0.0,
        "same_cluster_comparison": True,
        "reason": "MTP-labeled row did not execute accepted_execution_path=mimo_mtp_fastpath; cannot classify pass or fail",
        "next_step": "collect same-cluster guarded MTP rows with accepted_execution_path=mimo_mtp_fastpath and generation_tps before making speedup or target claims",
    }


def test_collect_ar_baseline_rows_records_blocked_status_with_required_commands_when_cluster_unavailable() -> (
    None
):
    def fake_get(_url: str, _timeout_seconds: float) -> tuple[int, dict[str, object]]:
        raise OSError("connection refused")

    def fake_post(
        _url: str, _payload: dict[str, object], _timeout_seconds: float
    ) -> tuple[int, dict[str, object]]:
        raise AssertionError(
            "unavailable cluster preflight must not post benchmark rows"
        )

    rows = cluster_bench.collect_ar_baseline_rows(
        api_base="http://127.0.0.1:52415",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        prompt="hello",
        repeats=1,
        timeout_seconds=1.0,
        output_dir=None,
        http_get=fake_get,
        http_post=fake_post,
    )

    assert rows == [
        cluster_bench.build_ar_baseline_blocked_status(
            api_base="http://127.0.0.1:52415",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            reason="cluster availability preflight failed: connection refused",
        )
    ]


def test_blocked_ar_baseline_status_emits_non_fabrication_safeguards() -> None:
    status = cluster_bench.build_ar_baseline_blocked_status(
        api_base="http://127.0.0.1:52415",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        reason="cluster availability preflight failed: connection refused",
    )

    live_metric_fields = {
        "generation_tps",
        "generation_tokens",
        "prompt_tps",
        "power_usage",
        "generation_stats",
        "mtp_speedup_budget",
        "mtp_throughput_threshold",
    }
    assert live_metric_fields.isdisjoint(status)
    assert status["live_metrics_status"] == "unavailable"
    assert status["non_fabrication_statement"] == (
        "No live benchmark metrics are claimed in this blocked row; run the exact commands before using performance, speedup, or Slice 5 evidence."
    )


def test_blocked_cluster_benchmark_error_emits_non_fabrication_safeguards() -> None:
    def fake_get(_url: str, _timeout_seconds: float) -> tuple[int, dict[str, object]]:
        raise OSError("connection refused")

    def fake_post(
        _url: str, _payload: dict[str, object], _timeout_seconds: float
    ) -> tuple[int, dict[str, object]]:
        raise AssertionError("benchmark execution must not continue")

    rows = cluster_bench.run_cluster_benchmark(
        args=cluster_bench.ClusterBenchmarkArgs(
            api_base="http://127.0.0.1:52415",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            prompt="hello",
            max_tokens=16,
            matrix_max_tokens=(16,),
            repeats=1,
            mode_label="ar",
            timeout_seconds=1.0,
            payload_extra_json=None,
            list_models=False,
        ),
        http_get=fake_get,
        http_post=fake_post,
    )

    row = rows[0]
    live_metric_fields = {
        "generation_tps",
        "generation_tokens",
        "prompt_tps",
        "power_usage",
        "generation_stats",
        "mtp_speedup_budget",
        "mtp_throughput_threshold",
    }
    assert live_metric_fields.isdisjoint(row)
    assert row["live_metrics_status"] == "unavailable"
    assert row["non_fabrication_statement"] == (
        "No live benchmark metrics are claimed in this blocked row; run rerun_command before using performance, speedup, or Slice 5 evidence."
    )


def test_render_ar_baseline_commands_includes_exact_16_and_64_token_rows() -> None:
    commands = cluster_bench.render_ar_baseline_commands(
        api_base="http://127.0.0.1:52415",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
    )

    assert commands == [
        "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 16 --repeats 1 --mode-label ar --list-models",
        "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 64 --repeats 1 --mode-label ar --list-models",
    ]


def test_build_exact_ar_baseline_command_row_for_max_tokens_64_targets_bench_chat_completions() -> (
    None
):
    row = cluster_bench.build_exact_ar_baseline_command_row(
        api_base="http://127.0.0.1:52415/",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        max_tokens=64,
    )

    assert row == {
        "kind": "cluster_benchmark_command",
        "cluster_path": "exo_api_bench_chat_completions",
        "benchmark_endpoint": "/bench/chat/completions",
        "api_base": "http://127.0.0.1:52415",
        "mode": "ar",
        "model": "kernelpool/MiMo-V2.5-Pro-6bit",
        "model_id": "kernelpool/MiMo-V2.5-Pro-6bit",
        "temperature": 0.0,
        "max_tokens": 64,
        "repeats": 1,
        "mtp_enabled": False,
        "payload_extra_json": None,
        "command": "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 64 --repeats 1 --mode-label ar --list-models",
        "next_step": "run this same-cluster AR command before making MTP speedup or Slice 5 claims",
    }


def test_build_ar_baseline_blocked_status_records_required_16_and_64_commands() -> None:
    status = cluster_bench.build_ar_baseline_blocked_status(
        api_base="http://127.0.0.1:52415",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        reason="cluster availability preflight failed: connection refused",
    )

    assert status == {
        "kind": "ar_baseline_collection_status",
        "status": "blocked_with_command",
        "cluster_path": "exo_api_bench_chat_completions",
        "benchmark_endpoint": "/bench/chat/completions",
        "api_base": "http://127.0.0.1:52415",
        "model": "kernelpool/MiMo-V2.5-Pro-6bit",
        "model_id": "kernelpool/MiMo-V2.5-Pro-6bit",
        "mode": "ar",
        "reason": "cluster availability preflight failed: connection refused",
        "required_max_tokens": [16, 64],
        "commands": [
            {
                "max_tokens": 16,
                "command": "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 16 --repeats 1 --mode-label ar --list-models",
            },
            {
                "max_tokens": 64,
                "command": "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 64 --repeats 1 --mode-label ar --list-models",
            },
        ],
        "live_metrics_status": "unavailable",
        "non_fabrication_statement": (
            "No live benchmark metrics are claimed in this blocked row; run the exact "
            "commands before using performance, speedup, or Slice 5 evidence."
        ),
        "next_step": "start or point to an exo cluster API, then run each command in commands to collect same-cluster AR baseline rows before MTP speedup or Slice 5 claims",
    }


def test_render_ar_baseline_command_documentation_lists_required_commands() -> None:
    documentation = cluster_bench.render_ar_baseline_command_documentation(
        api_base="http://127.0.0.1:52415",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
    )

    documented_commands = cluster_bench.render_ar_baseline_commands(
        api_base="http://127.0.0.1:52415",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
    )
    for command in documented_commands:
        assert command in documentation

    assert "## Exact AR baseline commands" in documentation
    assert "--api-base http://127.0.0.1:52415" in documentation
    assert "--model-id kernelpool/MiMo-V2.5-Pro-6bit" in documentation
    assert documentation.count("--mode-label ar") == 2
    assert documentation.count("--list-models") == 2
    assert documentation.count("--repeats 1") == 2
    assert "--max-tokens 16" in documentation
    assert "--max-tokens 64" in documentation
    assert "--payload-extra-json" not in documentation
    assert "--mode-label mtp" not in documentation


def test_preflight_cluster_for_ar_baseline_marks_available_cluster() -> None:
    def fake_get(url: str, timeout_seconds: float) -> tuple[int, dict[str, object]]:
        assert url == "http://127.0.0.1:52415/v1/models"
        assert timeout_seconds == 1.0
        return 200, {"data": [{"id": "kernelpool/MiMo-V2.5-Pro-6bit"}]}

    result = cluster_bench.preflight_cluster_for_ar_baseline(
        api_base="http://127.0.0.1:52415/",
        timeout_seconds=1.0,
        http_get=fake_get,
        list_models=False,
    )

    assert result == cluster_bench.ClusterAvailabilityPreflightResult(
        available=True,
        api_base="http://127.0.0.1:52415",
        probe_url="http://127.0.0.1:52415/v1/models",
        stage="cluster_availability_probe",
        status_code=200,
        response={"data": [{"id": "kernelpool/MiMo-V2.5-Pro-6bit"}]},
        error=None,
    )


def test_preflight_cluster_for_ar_baseline_marks_unavailable_cluster() -> None:
    def fake_get(url: str, timeout_seconds: float) -> tuple[int, dict[str, object]]:
        assert url == "http://127.0.0.1:52415/v1/models"
        assert timeout_seconds == 1.0
        return 503, {
            "status": "blocked",
            "detail": "cluster API is unavailable for model listing",
        }

    result = cluster_bench.preflight_cluster_for_ar_baseline(
        api_base="http://127.0.0.1:52415",
        timeout_seconds=1.0,
        http_get=fake_get,
        list_models=True,
    )

    assert result == cluster_bench.ClusterAvailabilityPreflightResult(
        available=False,
        api_base="http://127.0.0.1:52415",
        probe_url="http://127.0.0.1:52415/v1/models",
        stage="models_probe",
        status_code=503,
        response={
            "status": "blocked",
            "detail": "cluster API is unavailable for model listing",
        },
        error="HTTP 503 blocked: cluster API is unavailable for model listing",
    )


def test_run_cluster_benchmark_does_not_emit_metric_row_for_blocked_http_response() -> (
    None
):
    def fake_get(url: str, timeout_seconds: float) -> tuple[int, dict[str, object]]:
        assert url == "http://127.0.0.1:52415/v1/models"
        assert timeout_seconds == 1.0
        return 200, {"data": [{"id": "kernelpool/MiMo-V2.5-Pro-6bit"}]}

    def fake_post(
        _url: str, _payload: dict[str, object], _timeout_seconds: float
    ) -> tuple[int, dict[str, object]]:
        return 503, {
            "status": "blocked",
            "detail": "cluster API is unavailable for benchmark generation",
        }

    rows = cluster_bench.run_cluster_benchmark(
        args=cluster_bench.ClusterBenchmarkArgs(
            api_base="http://127.0.0.1:52415",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            prompt="hello",
            max_tokens=16,
            matrix_max_tokens=(16,),
            repeats=1,
            mode_label="ar",
            timeout_seconds=1.0,
            payload_extra_json=None,
            list_models=False,
        ),
        http_get=fake_get,
        http_post=fake_post,
    )

    assert [row for row in rows if row["kind"] == "cluster_benchmark_metric"] == []
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "cluster_benchmark_error"
    assert row["cluster_path"] == "exo_api_bench_chat_completions"
    assert row["api_base"] == "http://127.0.0.1:52415"
    assert row["mode"] == "ar"
    assert row["repeat_index"] == 0
    assert row["stage"] == "bench_chat_completions"
    assert (
        row["error"]
        == "HTTP 503 blocked: cluster API is unavailable for benchmark generation"
    )
    assert row["status"] == "blocked_with_command"
    assert row["success_row_count"] == 0
    assert row["live_metrics_status"] == "unavailable"
    assert row["blocked_evidence"]["evidence_kind"] == "blocked_evidence"
    assert row["blocked_evidence"]["row_status"] == "blocked"
    assert row["blocked_evidence"]["generation_tps"] is None
    assert row["blocked_evidence"]["generation_tokens"] is None
    assert row["blocked_evidence"]["prompt_tps"] is None
    assert row["blocked_evidence"]["blocker_reason"] == (
        "HTTP 503 blocked: cluster API is unavailable for benchmark generation"
    )
    assert row["blocked_evidence"]["blocked_stage"] == "bench_chat_completions"
    assert row["rerun_command"] == (
        "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 "
        "--model-id kernelpool/MiMo-V2.5-Pro-6bit --prompt hello --max-tokens 16 "
        "--repeats 1 --mode-label ar"
    )


def test_unavailable_cluster_error_row_includes_canonical_blocked_evidence_contract() -> (
    None
):
    def fake_get(url: str, timeout_seconds: float) -> tuple[int, dict[str, object]]:
        assert url == "http://127.0.0.1:52415/v1/models"
        assert timeout_seconds == 1.0
        raise OSError("cluster probe refused")

    def fake_post(
        _url: str, _payload: dict[str, object], _timeout_seconds: float
    ) -> tuple[int, dict[str, object]]:
        raise AssertionError(
            "benchmark execution must not continue after probe failure"
        )

    rows = cluster_bench.run_cluster_benchmark(
        args=cluster_bench.ClusterBenchmarkArgs(
            api_base="http://127.0.0.1:52415",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            prompt="hello",
            max_tokens=16,
            matrix_max_tokens=(16,),
            repeats=1,
            mode_label="ar",
            timeout_seconds=1.0,
            payload_extra_json=None,
            list_models=False,
        ),
        http_get=fake_get,
        http_post=fake_post,
    )

    row = rows[0]
    assert row["kind"] == "cluster_benchmark_error"
    blocked_evidence = row["blocked_evidence"]
    assert blocked_evidence == {
        "schema_version": "mimo_mtp_benchmark_evidence.v1",
        "evidence_kind": "blocked_evidence",
        "row_status": "blocked",
        "benchmark_session_id": "blocked:http://127.0.0.1:52415:/bench/chat/completions:kernelpool/MiMo-V2.5-Pro-6bit:ar:16",
        "api_url": "http://127.0.0.1:52415",
        "cluster_id": None,
        "endpoint": "/bench/chat/completions",
        "model_id": "kernelpool/MiMo-V2.5-Pro-6bit",
        "prompt_id": None,
        "prompt_hash": "2cf24dba5fb0a30e",
        "temperature": 0.0,
        "max_tokens": 16,
        "mode": "ar",
        "repeat_index": None,
        "generation_tps": None,
        "generation_tokens": None,
        "prompt_tps": None,
        "power_usage": None,
        "payload_extra": {},
        "mtp_enabled": False,
        "mtp_depth": None,
        "mtp_execution_state": "blocked_unavailable",
        "mtp_disable_reason": "cluster_unavailable",
        "telemetry_completeness": "blocked_no_live_telemetry",
        "runnable_command": "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --prompt hello --max-tokens 16 --repeats 1 --mode-label ar",
        "expected_output_path": None,
        "blocker_reason": "cluster probe refused",
        "blocked_stage": "cluster_availability_probe",
        "non_fabrication_statement": "No performance result is claimed because the target was unavailable before benchmark execution.",
    }


def test_run_cluster_benchmark_blocks_when_availability_probe_fails_without_posting() -> (
    None
):
    def fake_get(url: str, timeout_seconds: float) -> tuple[int, dict[str, object]]:
        assert url == "http://127.0.0.1:52415/v1/models"
        assert timeout_seconds == 1.0
        raise OSError("cluster probe refused")

    def fake_post(
        _url: str, _payload: dict[str, object], _timeout_seconds: float
    ) -> tuple[int, dict[str, object]]:
        raise AssertionError(
            "benchmark execution must not continue after probe failure"
        )

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

    assert [row for row in rows if row["kind"] == "cluster_benchmark_metric"] == []
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "cluster_benchmark_error"
    assert row["cluster_path"] == "exo_api_bench_chat_completions"
    assert row["api_base"] == "http://127.0.0.1:52415"
    assert row["mode"] == "ar"
    assert row["repeat_index"] is None
    assert row["stage"] == "cluster_availability_probe"
    assert row["error"] == "cluster probe refused"
    assert row["status"] == "blocked_with_command"
    assert row["success_row_count"] == 0
    assert row["blocked_reason_kind"] == "unavailable_cluster_api_target"
    assert row["benchmark_execution_attempted"] is False
    assert row["benchmark_endpoint"] == "/bench/chat/completions"
    assert row["live_metrics_status"] == "unavailable"
    assert row["blocked_evidence"]["evidence_kind"] == "blocked_evidence"
    assert row["blocked_evidence"]["row_status"] == "blocked"
    assert row["blocked_evidence"]["generation_tps"] is None
    assert row["blocked_evidence"]["blocker_reason"] == "cluster probe refused"
    assert row["blocked_evidence"]["blocked_stage"] == "cluster_availability_probe"
    assert row["blocked_evidence"]["non_fabrication_statement"] == (
        "No performance result is claimed because the target was unavailable before benchmark execution."
    )
    assert row["rerun_command"] == (
        "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 "
        "--model-id kernelpool/MiMo-V2.5-Pro-6bit --prompt hello --max-tokens 1 "
        "--repeats 1 --mode-label ar"
    )


def test_run_cluster_benchmark_blocks_when_models_probe_returns_unavailable_status_without_posting() -> (
    None
):
    def fake_get(url: str, timeout_seconds: float) -> tuple[int, dict[str, object]]:
        assert url == "http://127.0.0.1:52415/v1/models"
        assert timeout_seconds == 1.0
        return 503, {
            "status": "blocked",
            "detail": "cluster API is unavailable for model listing",
        }

    def fake_post(
        _url: str, _payload: dict[str, object], _timeout_seconds: float
    ) -> tuple[int, dict[str, object]]:
        raise AssertionError(
            "benchmark execution must not continue after unavailable models probe"
        )

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
        ),
        http_get=fake_get,
        http_post=fake_post,
    )

    assert [row for row in rows if row["kind"] == "cluster_benchmark_metric"] == []
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "cluster_benchmark_error"
    assert row["cluster_path"] == "exo_api_bench_chat_completions"
    assert row["api_base"] == "http://127.0.0.1:52415"
    assert row["mode"] == "ar"
    assert row["repeat_index"] is None
    assert row["stage"] == "models_probe"
    assert (
        row["error"] == "HTTP 503 blocked: cluster API is unavailable for model listing"
    )
    assert row["status"] == "blocked_with_command"
    assert row["success_row_count"] == 0
    assert row["blocked_reason_kind"] == "unavailable_cluster_api_target"
    assert row["benchmark_execution_attempted"] is False
    assert row["benchmark_endpoint"] == "/bench/chat/completions"
    assert row["live_metrics_status"] == "unavailable"
    assert row["blocked_evidence"]["evidence_kind"] == "blocked_evidence"
    assert row["blocked_evidence"]["row_status"] == "blocked"
    assert row["blocked_evidence"]["generation_tps"] is None
    assert row["blocked_evidence"]["blocker_reason"] == (
        "HTTP 503 blocked: cluster API is unavailable for model listing"
    )
    assert row["blocked_evidence"]["blocked_stage"] == "models_probe"
    assert row["blocked_evidence"]["runnable_command"] == (
        "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 "
        "--model-id kernelpool/MiMo-V2.5-Pro-6bit --prompt hello --max-tokens 1 "
        "--repeats 1 --mode-label ar --list-models"
    )


def test_run_cluster_benchmark_formats_ar_baseline_blocker_with_exact_remediation() -> (
    None
):
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
            max_tokens=16,
            matrix_max_tokens=(16,),
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

    rerun_command = "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --prompt hello --max-tokens 16 --repeats 1 --mode-label ar --list-models"
    assert rows[0]["ar_baseline_blocker"] == _expected_ar_baseline_blocker(
        rerun_command
    )


def test_run_cluster_benchmark_records_blocked_status_without_success_rows_when_cluster_is_unavailable() -> (
    None
):
    def fake_get(_url: str, _timeout_seconds: float) -> tuple[int, dict[str, object]]:
        return 200, {"data": [{"id": "kernelpool/MiMo-V2.5-Pro-6bit"}]}

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
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "cluster_benchmark_error"
    assert row["cluster_path"] == "exo_api_bench_chat_completions"
    assert row["api_base"] == "http://127.0.0.1:52415"
    assert row["mode"] == "ar"
    assert row["repeat_index"] == 0
    assert row["stage"] == "bench_chat_completions"
    assert row["error"] == "connection refused"
    assert row["status"] == "blocked_with_command"
    assert row["success_row_count"] == 0
    assert row["live_metrics_status"] == "unavailable"
    assert row["blocked_evidence"]["evidence_kind"] == "blocked_evidence"
    assert row["blocked_evidence"]["row_status"] == "blocked"
    assert row["blocked_evidence"]["generation_tps"] is None
    assert row["blocked_evidence"]["generation_tokens"] is None
    assert row["blocked_evidence"]["prompt_tps"] is None
    assert row["blocked_evidence"]["blocker_reason"] == "connection refused"
    assert row["blocked_evidence"]["blocked_stage"] == "bench_chat_completions"
    assert row["rerun_command"] == (
        "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 "
        "--model-id kernelpool/MiMo-V2.5-Pro-6bit --prompt hello --max-tokens 1 "
        "--repeats 1 --mode-label ar"
    )


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

    assert [row for row in rows if row["kind"] == "cluster_benchmark_metric"] == []
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "cluster_benchmark_error"
    assert row["cluster_path"] == "exo_api_bench_chat_completions"
    assert row["api_base"] == "http://127.0.0.1:52415"
    assert row["mode"] == "ar"
    assert row["repeat_index"] is None
    assert row["stage"] == "models_probe"
    assert row["error"] == "connection refused"
    assert row["status"] == "blocked_with_command"
    assert row["success_row_count"] == 0
    assert row["blocked_reason_kind"] == "unavailable_cluster_api_target"
    assert row["benchmark_execution_attempted"] is False
    assert row["benchmark_endpoint"] == "/bench/chat/completions"
    assert row["live_metrics_status"] == "unavailable"
    assert row["blocked_evidence"]["evidence_kind"] == "blocked_evidence"
    assert row["blocked_evidence"]["row_status"] == "blocked"
    assert row["blocked_evidence"]["generation_tps"] is None
    assert row["blocked_evidence"]["generation_tokens"] is None
    assert row["blocked_evidence"]["prompt_tps"] is None
    assert row["blocked_evidence"]["blocker_reason"] == "connection refused"
    assert row["blocked_evidence"]["blocked_stage"] == "models_probe"
    assert row["blocked_evidence"]["runnable_command"] == (
        "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 "
        "--model-id kernelpool/MiMo-V2.5-Pro-6bit --prompt hello --max-tokens 1 "
        "--repeats 1 --mode-label ar --list-models"
    )


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


def test_run_cluster_benchmark_unavailable_preflight_row_marks_no_benchmark_execution_attempted() -> (
    None
):
    post_attempted = False

    def fake_get(_url: str, _timeout_seconds: float) -> tuple[int, dict[str, object]]:
        raise OSError("connection refused")

    def fake_post(
        _url: str, _payload: dict[str, object], _timeout_seconds: float
    ) -> tuple[int, dict[str, object]]:
        nonlocal post_attempted
        post_attempted = True
        raise AssertionError("unavailable target must not execute benchmark request")

    rows = cluster_bench.run_cluster_benchmark(
        args=cluster_bench.ClusterBenchmarkArgs(
            api_base="http://127.0.0.1:52415",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            prompt="hello",
            max_tokens=16,
            matrix_max_tokens=(16,),
            repeats=1,
            mode_label="ar",
            timeout_seconds=1.0,
            payload_extra_json=None,
            list_models=False,
        ),
        http_get=fake_get,
        http_post=fake_post,
    )

    assert post_attempted is False
    assert rows[0]["kind"] == "cluster_benchmark_error"
    assert rows[0]["status"] == "blocked_with_command"
    assert rows[0]["stage"] == "cluster_availability_probe"
    assert rows[0]["blocked_reason_kind"] == "unavailable_cluster_api_target"
    assert rows[0]["benchmark_execution_attempted"] is False
    assert rows[0]["benchmark_endpoint"] == "/bench/chat/completions"


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
        "ar_ms_per_token": 45.0,
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
        "ar_ms_per_token": 47.727273,
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
        prompt="Write a Python function that parses JSON lines.",
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


def test_ar_matrix_command_row_for_16_tokens_declares_exact_endpoint_and_disabled_mtp() -> (
    None
):
    rows = cluster_bench.render_benchmark_matrix_command_rows(
        api_base="http://127.0.0.1:52415/",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        prompt="Write a Python function that parses JSON lines.",
        max_tokens_values=(16,),
        repeats=1,
        mimo_mtp_sidecar_path=None,
    )

    ar_row = rows[0]

    assert ar_row["kind"] == "cluster_benchmark_command"
    assert ar_row["cluster_path"] == "exo_api_bench_chat_completions"
    assert ar_row["benchmark_endpoint"] == "/bench/chat/completions"
    assert ar_row["api_base"] == "http://127.0.0.1:52415"
    assert ar_row["mode"] == "ar"
    assert ar_row["max_tokens"] == 16
    assert ar_row["temperature"] == 0.0
    assert ar_row["payload_extra_json"] is None
    assert ar_row["mtp_enabled"] is False
    assert ar_row["mtp_depth"] is None
    assert ar_row["mtp_execution_state"] == "disabled_default"
    assert ar_row["mtp_disable_reason"] == "default_ar_no_mtp_payload"
    assert ar_row["telemetry_completeness"] == "command_only_no_live_execution"
    assert str(ar_row["command"]) == (
        "uv run python3 scripts/bench_mimo_mtp_cluster.py "
        "--api-base http://127.0.0.1:52415 "
        "--model-id kernelpool/MiMo-V2.5-Pro-6bit "
        "--prompt 'Write a Python function that parses JSON lines.' "
        "--max-tokens 16 --repeats 1 --mode-label ar --list-models"
    )
    assert "--payload-extra-json" not in str(ar_row["command"])
    assert "--mode-label mtp" not in str(ar_row["command"])


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


def test_blocked_rerun_command_preserves_prompt_for_same_prompt_matrix_invariant() -> (
    None
):
    def fake_get(_url: str, _timeout_seconds: float) -> tuple[int, dict[str, object]]:
        raise OSError("connection refused")

    def fake_post(
        _url: str, _payload: dict[str, object], _timeout_seconds: float
    ) -> tuple[int, dict[str, object]]:
        raise AssertionError("availability failure must stop before benchmark")

    rows = cluster_bench.run_cluster_benchmark(
        args=cluster_bench.ClusterBenchmarkArgs(
            api_base="http://127.0.0.1:52415",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            prompt="same prompt for every row",
            max_tokens=16,
            matrix_max_tokens=(16,),
            repeats=3,
            mode_label="mtp-d1",
            timeout_seconds=1.0,
            payload_extra_json=json.dumps(
                {
                    "mimo_mtp_fastpath": True,
                    "mimo_mtp_depth": 1,
                    "mimo_mtp_fail_closed": True,
                },
                sort_keys=True,
            ),
            list_models=False,
        ),
        http_get=fake_get,
        http_post=fake_post,
    )

    rerun_command = rows[0]["rerun_command"]
    assert isinstance(rerun_command, str)
    assert "--prompt 'same prompt for every row'" in rerun_command
    assert "--repeats 3" in rerun_command
    assert "--mode-label mtp-d1" in rerun_command


def test_benchmark_matrix_command_rows_pin_same_cluster_invariants() -> None:
    rows = cluster_bench.render_benchmark_matrix_command_rows(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        prompt="same prompt for every row",
        max_tokens_values=(16, 64),
        repeats=3,
        mimo_mtp_sidecar_path="/tmp/model_mtp.safetensors",
    )

    assert len(rows) == 8
    assert {row["api_base"] for row in rows} == {"http://cluster.example"}
    assert {row["benchmark_endpoint"] for row in rows} == {"/bench/chat/completions"}
    assert {row["cluster_path"] for row in rows} == {"exo_api_bench_chat_completions"}
    assert {row["model"] for row in rows} == {"kernelpool/MiMo-V2.5-Pro-6bit"}
    assert {row["prompt"] for row in rows} == {"same prompt for every row"}
    assert {row["temperature"] for row in rows} == {0.0}
    assert {row["repeats"] for row in rows} == {3}
    assert {row["max_tokens"] for row in rows} == {16, 64}
    assert [row["mode"] for row in rows] == [
        "ar",
        "mtp-d1",
        "mtp-d2",
        "mtp-d3",
        "ar",
        "mtp-d1",
        "mtp-d2",
        "mtp-d3",
    ]
    assert all(
        "--prompt 'same prompt for every row'" in str(row["command"]) for row in rows
    )
    assert all("--repeats 3" in str(row["command"]) for row in rows)
    first_command = rows[0]["command"]
    assert isinstance(first_command, str)
    assert first_command.count("--repeats") == 1


def test_summarize_benchmark_matrix_rows_groups_by_mode_depth_and_max_tokens() -> None:
    rows = [
        {
            "kind": "cluster_benchmark_metric",
            "cluster_path": "exo_api_bench_chat_completions",
            "mode": "ar",
            "max_tokens": 16,
            "generation_tps": 20.0,
            "accepted_execution_path": "ar",
            "mtp_enabled": False,
        },
        {
            "kind": "cluster_benchmark_metric",
            "cluster_path": "exo_api_bench_chat_completions",
            "mode": "ar",
            "max_tokens": 16,
            "generation_tps": 24.0,
            "accepted_execution_path": "ar",
            "mtp_enabled": False,
        },
        {
            "kind": "cluster_benchmark_metric",
            "cluster_path": "exo_api_bench_chat_completions",
            "mode": "mtp-d2",
            "max_tokens": 16,
            "generation_tps": 30.0,
            "accepted_execution_path": "mimo_mtp_fastpath",
            "mtp_enabled": True,
            "mtp_depth": 2,
            "acceptance_rate": 0.75,
        },
        {
            "kind": "cluster_benchmark_metric",
            "cluster_path": "exo_api_bench_chat_completions",
            "mode": "mtp-d2",
            "max_tokens": 16,
            "generation_tps": 36.0,
            "accepted_execution_path": "mimo_mtp_fastpath",
            "mtp_enabled": True,
            "mtp_depth": 2,
            "acceptance_rate": 0.85,
        },
        {
            "kind": "cluster_benchmark_error",
            "cluster_path": "exo_api_bench_chat_completions",
            "mode": "mtp-d3",
            "status": "blocked_with_command",
            "rerun_command": "uv run python3 scripts/bench_mimo_mtp_cluster.py --mode-label mtp-d3",
        },
    ]

    summary = cluster_bench.summarize_benchmark_matrix_rows(rows)

    assert summary == {
        "kind": "cluster_benchmark_matrix_summary",
        "cluster_path": "exo_api_bench_chat_completions",
        "status": "blocked_partial_matrix",
        "metric_row_count": 4,
        "blocked_row_count": 1,
        "groups": [
            {
                "mode": "ar",
                "mtp_depth": None,
                "max_tokens": 16,
                "row_count": 2,
                "median_tok_s": 22.0,
                "best_tok_s": 24.0,
                "accepted_mtp_row_count": 0,
                "acceptance_rate_median": None,
                "acceptance_rate_best": None,
            },
            {
                "mode": "mtp-d2",
                "mtp_depth": 2,
                "max_tokens": 16,
                "row_count": 2,
                "median_tok_s": 33.0,
                "best_tok_s": 36.0,
                "accepted_mtp_row_count": 2,
                "acceptance_rate_median": 0.8,
                "acceptance_rate_best": 0.85,
            },
        ],
        "blocked_commands": [
            "uv run python3 scripts/bench_mimo_mtp_cluster.py --mode-label mtp-d3"
        ],
        "next_step": "rerun blocked commands before making complete AR-vs-MTP performance claims",
    }


def test_cluster_metric_row_marks_ar_as_mtp_disabled() -> None:
    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="ar",
        repeat_index=0,
        elapsed_seconds=1.0,
        status_code=200,
        response={
            "generation_stats": {
                "prompt_tps": 100.0,
                "generation_tps": 20.0,
                "generation_tokens": 4,
            }
        },
        payload_extra_keys=[],
    )

    assert row["mode"] == "ar"
    assert row["accepted_execution_path"] == "ar"
    assert row["mtp_enabled"] is False
    assert row["mtp_depth"] is None


def test_cluster_metric_row_preserves_top_level_mtp_telemetry() -> None:
    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="mtp-d3",
        repeat_index=1,
        elapsed_seconds=2.0,
        status_code=200,
        response={
            "accepted_execution_path": "mimo_mtp_fastpath",
            "mtp_enabled": True,
            "requested_mtp_depth": 3,
            "mtp_depth": 3,
            "mtp_sidecar_status": "ready",
            "attempted_depth_counts": {"3": 10},
            "accepted_depth_counts": {"0": 1, "3": 9},
            "acceptance_rate": 0.9,
            "fallback_count": 1,
            "timing_breakdown_seconds": {
                "proposal": 0.12,
                "verification": 0.34,
                "acceptance": 0.05,
                "fallback": 0.01,
            },
            "generation_stats": {
                "prompt_tps": 100.0,
                "generation_tps": 35.0,
                "generation_tokens": 30,
            },
        },
        payload_extra_keys=["mimo_mtp_fastpath", "mimo_mtp_depth"],
    )

    assert row["accepted_execution_path"] == "mimo_mtp_fastpath"
    assert row["mtp_enabled"] is True
    assert row["requested_mtp_depth"] == 3
    assert row["mtp_depth"] == 3
    assert row["mtp_sidecar_status"] == "ready"
    assert row["attempted_depth_counts"] == {"3": 10}
    assert row["accepted_depth_counts"] == {"0": 1, "3": 9}
    assert row["acceptance_rate"] == 0.9
    assert row["fallback_count"] == 1
    assert row["timing_breakdown_seconds"] == {
        "proposal": 0.12,
        "verification": 0.34,
        "acceptance": 0.05,
        "fallback": 0.01,
    }


def test_cluster_metric_row_derives_mtp_telemetry_from_generation_stats_when_needed() -> (
    None
):
    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="mtp-d2",
        repeat_index=0,
        elapsed_seconds=2.0,
        status_code=200,
        response={
            "generation_stats": {
                "prompt_tps": 100.0,
                "generation_tps": 28.0,
                "generation_tokens": 12,
                "accepted_execution_path": "mimo_mtp_fastpath",
                "mtp_enabled": True,
                "requested_mtp_depth": 2,
                "mtp_depth": 2,
                "mimo_mtp_attempted_depth_counts": {"2": 6},
                "mimo_mtp_accepted_depth_counts": {"1": 2, "2": 4},
                "mimo_mtp_attempted_tokens": 12,
                "mimo_mtp_accepted_tokens": 10,
                "mimo_mtp_fallback_count": 2,
                "timing_breakdown_seconds": {
                    "proposal": 0.2,
                    "verification": 0.6,
                    "acceptance": 0.04,
                    "fallback": 0.03,
                },
            }
        },
        payload_extra_keys=["mimo_mtp_fastpath"],
    )

    assert row["accepted_execution_path"] == "mimo_mtp_fastpath"
    assert row["mtp_enabled"] is True
    assert row["requested_mtp_depth"] == 2
    assert row["mtp_depth"] == 2
    assert row["attempted_depth_counts"] == {"2": 6}
    assert row["accepted_depth_counts"] == {"1": 2, "2": 4}
    assert row["acceptance_rate"] == 10 / 12
    assert row["fallback_count"] == 2
    assert row["timing_breakdown_seconds"] == {
        "proposal": 0.2,
        "verification": 0.6,
        "acceptance": 0.04,
        "fallback": 0.03,
    }


def test_cluster_metric_row_preserves_fail_open_missing_sidecar_fallback_reason() -> (
    None
):
    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="mtp-d1",
        repeat_index=0,
        elapsed_seconds=1.0,
        status_code=200,
        response={
            "accepted_execution_path": "ar",
            "mtp_enabled": False,
            "requested_mtp_depth": 1,
            "mtp_depth": None,
            "mtp_sidecar_status": "missing",
            "mtp_disable_reason": "missing_sidecar",
            "mtp_fallback_reason": "fail_open_missing_sidecar",
            "fallback_count": 1,
            "generation_stats": {
                "prompt_tps": 100.0,
                "generation_tps": 20.0,
                "generation_tokens": 4,
            },
        },
        payload_extra_keys=["mimo_mtp_fastpath"],
    )

    assert row["accepted_execution_path"] == "ar"
    assert row["mtp_enabled"] is False
    assert row["mtp_disable_reason"] == "missing_sidecar"
    assert row["mtp_fallback_reason"] == "fail_open_missing_sidecar"
    assert row["mtp_fallback_kind"] == "missing_sidecar"
    assert row["fallback_count"] == 1


@pytest.mark.parametrize(
    ("disable_reason", "fallback_reason", "expected_kind"),
    [
        ("unsupported_model", "fail_open_unsupported_model", "unsupported"),
        ("missing_sidecar", "fail_open_missing_sidecar", "missing_sidecar"),
        ("low_acceptance", "fail_open_low_acceptance", "low_acceptance"),
        ("runtime_error", "fail_open_runtime_error", "runtime_error"),
    ],
)
def test_cluster_metric_row_identifies_fail_open_fallback_taxonomy(
    disable_reason: str, fallback_reason: str, expected_kind: str
) -> None:
    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="mtp-d1",
        repeat_index=0,
        elapsed_seconds=1.0,
        status_code=200,
        response={
            "accepted_execution_path": "ar",
            "mtp_enabled": False,
            "requested_mtp_depth": 1,
            "mtp_depth": None,
            "mtp_disable_reason": disable_reason,
            "mtp_fallback_reason": fallback_reason,
            "generation_stats": {
                "prompt_tps": 100.0,
                "generation_tps": 20.0,
                "generation_tokens": 4,
            },
        },
        payload_extra_keys=["mimo_mtp_fastpath"],
    )

    assert row["mtp_fallback_reason"] == fallback_reason
    assert row["mtp_fallback_mode"] == "fail_open"
    assert row["mtp_fallback_kind"] == expected_kind


def test_mtp_metric_row_does_not_claim_success_when_response_reports_disabled_mtp() -> (
    None
):
    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="mtp-d2",
        repeat_index=0,
        elapsed_seconds=1.0,
        status_code=200,
        response={
            "accepted_execution_path": "mimo_mtp_fastpath",
            "mtp_enabled": False,
            "requested_mtp_depth": 2,
            "mtp_depth": None,
            "mtp_disable_reason": "missing_sidecar",
            "generation_stats": {
                "prompt_tps": 100.0,
                "generation_tps": 20.0,
                "generation_tokens": 4,
            },
        },
        payload_extra={
            "mimo_mtp_fastpath": True,
            "mimo_mtp_depth": 2,
            "mimo_mtp_fail_closed": True,
        },
    )

    assert row["mtp_execution_state"] == "missing_sidecar"
    assert row["accepted_execution_path"] != "mimo_mtp_fastpath"
    assert row["accepted_execution_path"] == "rejected"
    assert row["mtp_enabled"] is False
    assert row["mtp_depth"] is None
    assert row["mtp_disable_reason"] == "missing_sidecar"
    assert "mtp_speedup_budget" not in row
    assert "mtp_throughput_threshold" not in row


def test_mtp_metric_row_does_not_honor_explicit_success_state_for_fail_open_fallback() -> (
    None
):
    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="mtp-d1",
        repeat_index=0,
        elapsed_seconds=1.0,
        status_code=200,
        response={
            "accepted_execution_path": "ar",
            "mtp_enabled": False,
            "requested_mtp_depth": 1,
            "mtp_depth": None,
            "mtp_execution_state": "successful_mtp",
            "mtp_disable_reason": "mimo_mtp_distributed_generator_unwired",
            "mtp_fallback_reason": "fail_open_distributed_generator_unwired",
            "generation_stats": {
                "prompt_tps": 100.0,
                "generation_tps": 20.0,
                "generation_tokens": 4,
            },
        },
        payload_extra={
            "mimo_mtp_fastpath": True,
            "mimo_mtp_depth": 1,
            "mimo_mtp_fail_closed": False,
        },
    )

    assert row["mtp_execution_state"] == "fail_open_fallback"
    assert row["accepted_execution_path"] == "ar"
    assert row["mtp_enabled"] is False
    assert row["mtp_fallback_mode"] == "fail_open"
    assert "mtp_speedup_budget" not in row
    assert "mtp_throughput_threshold" not in row


def test_blocked_benchmark_row_construction_preserves_request_evidence_without_live_execution(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def forbidden_live_runner() -> None:
        calls.append("live benchmark was invoked")
        raise AssertionError(
            "blocked row construction must not invoke live benchmark execution"
        )

    output_path = tmp_path / "blocked" / "mimo-mtp-d2.jsonl"
    row = cluster_bench.build_blocked_benchmark_row_from_context(
        api_base="http://cluster.example/",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        prompt="hello",
        max_tokens=64,
        repeats=2,
        mode_label="mtp-d2",
        payload_extra={
            "mimo_mtp_fastpath": True,
            "mimo_mtp_depth": 2,
            "mimo_mtp_fail_closed": True,
        },
        output_path=output_path,
        blocker_reason="cluster API unavailable during evaluator run",
        benchmark_session_id="session-20260607T033700Z",
        environment_assumptions={
            "exo_cluster_api": "operator must start exo before rerun",
            "sidecar_path": "operator-provided model_mtp.safetensors path required",
        },
        live_runner=forbidden_live_runner,
    )

    assert calls == []
    assert row["kind"] == "cluster_benchmark_blocked_row"
    assert row["evidence_kind"] == "blocked_evidence"
    assert row["schema_version"] == "mimo_mtp_benchmark_row.v1"
    assert row["row_status"] == "blocked_unavailable"
    assert row["cluster_path"] == "exo_api_bench_chat_completions"
    assert row["api_base"] == "http://cluster.example"
    assert row["endpoint"] == "/bench/chat/completions"
    assert row["model"] == "kernelpool/MiMo-V2.5-Pro-6bit"
    assert row["model_id"] == "kernelpool/MiMo-V2.5-Pro-6bit"
    assert row["mode"] == "mtp-d2"
    assert row["benchmark_session_id"] == "session-20260607T033700Z"
    assert row["temperature"] == 0.0
    assert row["max_tokens"] == 64
    assert row["repeat_index"] is None
    assert row["generation_tps"] is None
    assert row["generation_tokens"] is None
    assert row["prompt_tps"] is None
    assert row["power_usage"] is None
    assert row["mtp_enabled"] is True
    assert row["mtp_depth"] == 2
    assert row["mtp_execution_state"] == "blocked_unavailable"
    assert row["mtp_disable_reason"] == "cluster API unavailable during evaluator run"
    assert row["telemetry_completeness"] == "blocked_no_live_execution"
    assert row["no_performance_result_claimed"] is True
    assert row["blocker_reason"] == "cluster API unavailable during evaluator run"
    assert row["expected_output_path"] == str(output_path)
    assert row["environment_assumptions"] == {
        "exo_cluster_api": "operator must start exo before rerun",
        "sidecar_path": "operator-provided model_mtp.safetensors path required",
    }
    assert row["payload_extra"] == {
        "mimo_mtp_fastpath": True,
        "mimo_mtp_depth": 2,
        "mimo_mtp_fail_closed": True,
    }
    assert row["payload_extra_keys"] == [
        "mimo_mtp_depth",
        "mimo_mtp_fail_closed",
        "mimo_mtp_fastpath",
    ]
    assert row["commands"] == [
        {
            "command_kind": "rerun_benchmark",
            "command": (
                "uv run python3 scripts/bench_mimo_mtp_cluster.py "
                "--api-base http://cluster.example "
                "--model-id kernelpool/MiMo-V2.5-Pro-6bit "
                "--prompt hello --max-tokens 64 --repeats 2 --mode-label mtp-d2 "
                '--payload-extra-json \'{"mimo_mtp_depth":2,"mimo_mtp_fail_closed":true,"mimo_mtp_fastpath":true}\' '
                f"--output-dir {output_path.parent}"
            ),
        }
    ]


def test_fail_closed_harness_default_request_omits_mtp_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posted_payloads: list[dict[str, object]] = []

    def fake_get(_url: str, _timeout_seconds: float) -> tuple[int, dict[str, object]]:
        return 200, {"data": [{"id": "kernelpool/MiMo-V2.5-Pro-6bit"}]}

    def fake_post(
        _url: str, payload: dict[str, object], _timeout_seconds: float
    ) -> tuple[int, dict[str, object]]:
        posted_payloads.append(dict(payload))
        return 200, {
            "generation_stats": {
                "prompt_tps": 100.0,
                "generation_tps": 22.0,
                "generation_tokens": payload["max_tokens"],
            },
            "execution_path": _tensor_parallel_execution_path(),
        }

    monkeypatch.setattr(cluster_bench, "_http_json_get", fake_get)
    monkeypatch.setattr(cluster_bench, "_http_json_post", fake_post)

    exit_code = cluster_bench.main(
        [
            "--api-base",
            "http://cluster.example",
            "--max-tokens",
            "16",
            "--repeats",
            "1",
        ]
    )

    assert exit_code == 0
    assert len(posted_payloads) == 1
    payload = posted_payloads[0]
    assert payload["benchmark_mode_label"] == "ar"
    assert "mimo_mtp_fastpath" not in payload
    assert "mimo_mtp_depth" not in payload
    assert "mimo_mtp_fail_closed" not in payload


def test_harness_live_ar_row_emitted_by_run_cluster_benchmark_passes_canonical_contract() -> (
    None,
):
    """End-to-end: run_cluster_benchmark with a fake available cluster produces a live AR
    metric row that passes validate_canonical_benchmark_row()."""

    def fake_get(url: str, timeout_seconds: float) -> tuple[int, dict[str, object]]:
        assert url == "http://cluster.example/v1/models"
        return 200, {"data": [{"id": "kernelpool/MiMo-V2.5-Pro-6bit"}]}

    def fake_post(
        url: str, payload: dict[str, object], timeout_seconds: float
    ) -> tuple[int, dict[str, object]]:
        return 200, {
            "generation_stats": {
                "prompt_tps": 100.0,
                "generation_tps": 24.5,
                "generation_tokens": 64,
            },
            "execution_path": _tensor_parallel_execution_path(),
        }

    timer = _timer([0.0, 1.0])
    rows = cluster_bench.run_cluster_benchmark(
        args=cluster_bench.ClusterBenchmarkArgs(
            api_base="http://cluster.example",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            prompt="Write a Python function that parses JSON lines.",
            max_tokens=64,
            matrix_max_tokens=(64,),
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
    metric_rows = [
        row
        for row in rows
        if row.get("kind") == "cluster_benchmark_metric"
        and row.get("cluster_path") == "exo_api_bench_chat_completions"
    ]
    assert len(metric_rows) == 1
    row = metric_rows[0]
    assert row["mode"] == "ar"
    assert row["row_status"] == "live"
    assert row["mtp_enabled"] is False
    assert row["mtp_execution_state"] == "disabled_default"
    validation = cluster_bench.validate_canonical_benchmark_row(row)
    assert validation["valid"] is True, (
        f"Live AR metric row must pass canonical contract validation; "
        f"missing={validation['missing_required_fields']} "
        f"invalid={validation['invalid_fields']} "
        f"one_of_missing={validation['required_one_of_missing']}"
    )


def test_harness_live_successful_mtp_row_emitted_by_run_cluster_benchmark_passes_canonical_contract() -> (
    None,
):
    """End-to-end: run_cluster_benchmark with MTP response carrying successful_mtp
    fastpath telemetry produces a live metric row that passes the canonical contract."""

    def fake_get(url: str, timeout_seconds: float) -> tuple[int, dict[str, object]]:
        return 200, {"data": [{"id": "kernelpool/MiMo-V2.5-Pro-6bit"}]}

    def fake_post(
        url: str, payload: dict[str, object], timeout_seconds: float
    ) -> tuple[int, dict[str, object]]:
        return 200, {
            "accepted_execution_path": "mimo_mtp_fastpath",
            "generation_stats": {
                "prompt_tps": 99.0,
                "generation_tps": 31.0,
                "generation_tokens": 64,
                "mtp_enabled": True,
                "mtp_depth": 2,
                "mtp_sidecar_status": "loaded",
                "attempted_depth_counts": {"2": 20},
                "accepted_depth_counts": {"2": 12},
                "acceptance_rate": 0.6,
                "fallback_count": 0,
                "timing_breakdown_seconds": {"proposal": 0.2, "verification": 0.3},
            },
        }

    timer = _timer([0.0, 1.0])
    rows = cluster_bench.run_cluster_benchmark(
        args=cluster_bench.ClusterBenchmarkArgs(
            api_base="http://cluster.example",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            prompt="Write a Python function that parses JSON lines.",
            max_tokens=64,
            matrix_max_tokens=(64,),
            repeats=1,
            mode_label="mtp-d2",
            timeout_seconds=12.0,
            payload_extra_json=json.dumps(
                {
                    "mimo_mtp_fastpath": True,
                    "mimo_mtp_depth": 2,
                    "mimo_mtp_fail_closed": True,
                },
                sort_keys=True,
            ),
            list_models=False,
        ),
        http_get=fake_get,
        http_post=fake_post,
        timer=lambda: next(timer),
    )
    metric_rows = [
        row
        for row in rows
        if row.get("kind") == "cluster_benchmark_metric"
        and row.get("cluster_path") == "exo_api_bench_chat_completions"
    ]
    assert len(metric_rows) == 1
    row = metric_rows[0]
    assert row["mode"] == "mtp-d2"
    assert row["row_status"] == "live"
    assert row["accepted_execution_path"] == "mimo_mtp_fastpath"
    validation = cluster_bench.validate_canonical_benchmark_row(row)
    assert validation["valid"] is True, (
        f"Live successful_mtp metric row must pass canonical contract validation; "
        f"missing={validation['missing_required_fields']} "
        f"invalid={validation['invalid_fields']} "
        f"one_of_missing={validation['required_one_of_missing']}"
    )


def test_harness_live_fail_closed_mtp_row_passes_canonical_contract() -> None:
    """End-to-end: build_cluster_metric_row with a fail_closed MTP response
    produces a live metric row that passes the canonical contract."""

    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="mtp-d2",
        repeat_index=0,
        elapsed_seconds=1.0,
        status_code=200,
        response={
            "generation_stats": {
                "prompt_tps": 99.0,
                "generation_tps": 28.0,
                "generation_tokens": 64,
                "mtp_enabled": False,
                "mtp_depth": 2,
                "mtp_disable_reason": "missing_sidecar",
                "mtp_sidecar_status": "missing",
            },
        },
        payload_extra={"mimo_mtp_fastpath": True, "mimo_mtp_depth": 2},
        requested_max_tokens=64,
        benchmark_session_id="session-mtp-fail-closed-e2e",
        prompt_hash="sha256:test-prompt",
    )
    assert row["mtp_execution_state"] == "missing_sidecar"
    assert row["mtp_enabled"] is False
    validation = cluster_bench.validate_canonical_benchmark_row(row)
    assert validation["valid"] is True, (
        f"Live fail_closed MTP metric row must pass canonical contract validation; "
        f"missing={validation['missing_required_fields']} "
        f"invalid={validation['invalid_fields']} "
        f"one_of_missing={validation['required_one_of_missing']}"
    )


def test_harness_live_fail_open_mtp_row_passes_canonical_contract() -> None:
    """End-to-end: build_cluster_metric_row with a fail_open MTP response
    produces a live metric row that passes the canonical contract."""

    row = cluster_bench.build_cluster_metric_row(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        mode_label="mtp-d1",
        repeat_index=0,
        elapsed_seconds=1.0,
        status_code=200,
        response={
            "accepted_execution_path": "ar",
            "mtp_enabled": False,
            "requested_mtp_depth": 1,
            "mtp_depth": None,
            "mtp_sidecar_status": "missing",
            "mtp_disable_reason": "missing_sidecar",
            "mtp_fallback_reason": "fail_open_missing_sidecar",
            "fallback_count": 1,
            "generation_stats": {
                "prompt_tps": 100.0,
                "generation_tps": 20.0,
                "generation_tokens": 64,
            },
        },
        payload_extra={"mimo_mtp_fastpath": True, "mimo_mtp_depth": 1},
        requested_max_tokens=64,
        benchmark_session_id="session-mtp-fail-open-e2e",
        prompt_hash="sha256:test-prompt",
    )
    assert row["mtp_execution_state"] == "fail_open_fallback"
    assert row["mtp_enabled"] is False
    validation = cluster_bench.validate_canonical_benchmark_row(row)
    assert validation["valid"] is True, (
        f"Live fail_open MTP metric row must pass canonical contract validation; "
        f"missing={validation['missing_required_fields']} "
        f"invalid={validation['invalid_fields']} "
        f"one_of_missing={validation['required_one_of_missing']}"
    )


def test_harness_blocked_evidence_row_contains_all_canonical_required_fields() -> None:
    """Blocked evidence rows from _blocked_evidence_contract carry every required
    canonical benchmark row field so downstream tooling can validate them uniformly."""

    row = cluster_bench._blocked_evidence_contract(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        prompt="hello",
        max_tokens=64,
        mode_label="ar",
        repeat_index=0,
        error="cluster probe refused",
        stage="cluster_availability_probe",
        rerun_command="uv run python3 scripts/bench_mimo_mtp_cluster.py --mode-label ar",
    )
    missing_required = [
        field
        for field in cluster_bench.CANONICAL_BENCHMARK_ROW_REQUIRED_FIELDS
        if field not in row
    ]
    assert missing_required == [], (
        f"Blocked evidence row must contain all canonical required fields; "
        f"missing: {missing_required}"
    )
    # Blocked rows must satisfy the required-one-of groups as well
    for group in cluster_bench._CANONICAL_BENCHMARK_ROW_REQUIRED_ONE_OF:
        assert any(field in row for field in group), (
            f"Blocked evidence row must satisfy required-one-of group {group}"
        )


def test_harness_blocked_evidence_row_for_mtp_mode_contains_all_canonical_required_fields() -> (
    None,
):
    """Blocked MTP evidence rows carry every required canonical benchmark row field."""

    row = cluster_bench._blocked_evidence_contract(
        api_base="http://cluster.example",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        prompt="hello",
        max_tokens=64,
        mode_label="mtp-d2",
        repeat_index=0,
        error="cluster probe refused",
        stage="cluster_availability_probe",
        rerun_command="uv run python3 scripts/bench_mimo_mtp_cluster.py --mode-label mtp-d2",
        payload_extra={
            "mimo_mtp_fastpath": True,
            "mimo_mtp_depth": 2,
            "mimo_mtp_fail_closed": True,
        },
    )
    missing_required = [
        field
        for field in cluster_bench.CANONICAL_BENCHMARK_ROW_REQUIRED_FIELDS
        if field not in row
    ]
    assert missing_required == [], (
        f"Blocked MTP evidence row must contain all canonical required fields; "
        f"missing: {missing_required}"
    )
    assert row["mtp_enabled"] is True
    assert row["mtp_depth"] == 2
    assert row["mtp_execution_state"] == "blocked_unavailable"
    for group in cluster_bench._CANONICAL_BENCHMARK_ROW_REQUIRED_ONE_OF:
        assert any(field in row for field in group), (
            f"Blocked MTP evidence row must satisfy required-one-of group {group}"
        )


def test_harness_matrix_command_rows_carry_canonical_mtp_execution_state_fields() -> (
    None,
):
    """Matrix command rows emit mode, mtp_enabled, mtp_depth, mtp_execution_state,
    mtp_disable_reason, and telemetry_completeness matching the canonical contract."""

    rows = cluster_bench.render_benchmark_matrix_command_rows(
        api_base="http://127.0.0.1:52415",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        prompt="Write a Python function that parses JSON lines.",
        max_tokens_values=(16, 64),
        repeats=1,
        mimo_mtp_sidecar_path=None,
    )
    ar_rows = [row for row in rows if row["mode"] == "ar"]
    mtp_rows = [row for row in rows if row["mode"] != "ar"]
    assert len(ar_rows) == 2
    assert len(mtp_rows) == 6
    for row in ar_rows:
        assert row["mtp_enabled"] is False
        assert row["mtp_depth"] is None
        assert row["mtp_execution_state"] == "disabled_default"
        assert row["mtp_disable_reason"] == "default_ar_no_mtp_payload"
        assert row["telemetry_completeness"] == "command_only_no_live_execution"
    for row in mtp_rows:
        assert row["mtp_enabled"] is True
        assert isinstance(row["mtp_depth"], int)
        assert row["mtp_depth"] >= 1
        assert row["mtp_execution_state"] == "enabled_intent"
        assert row["mtp_disable_reason"] is None
        assert row["telemetry_completeness"] == "command_only_no_live_execution"


def test_mtp_mode_without_explicit_fail_closed_guard_is_rejected_before_post() -> None:
    post_calls: list[dict[str, object]] = []

    def fake_get(_url: str, _timeout_seconds: float) -> tuple[int, dict[str, object]]:
        raise AssertionError(
            "MTP guard validation should fail before cluster preflight"
        )

    def fake_post(
        _url: str, payload: dict[str, object], _timeout_seconds: float
    ) -> tuple[int, dict[str, object]]:
        post_calls.append(payload)
        raise AssertionError("unguarded MTP benchmark must not post")

    rows = cluster_bench.run_cluster_benchmark(
        args=cluster_bench.ClusterBenchmarkArgs(
            api_base="http://cluster.example",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            prompt="hello",
            max_tokens=16,
            matrix_max_tokens=(16,),
            repeats=1,
            mode_label="mtp-d1",
            timeout_seconds=12.0,
            payload_extra_json=json.dumps({"mimo_mtp_fastpath": True}),
            list_models=False,
        ),
        http_get=fake_get,
        http_post=fake_post,
    )

    assert post_calls == []
    assert rows == [
        {
            "kind": "cluster_benchmark_error",
            "cluster_path": "exo_api_bench_chat_completions",
            "api_base": "http://cluster.example",
            "mode": "mtp-d1",
            "repeat_index": None,
            "stage": "mtp_guard_validation",
            "error": "MTP benchmark mode requires guarded payload fields: mimo_mtp_fastpath=true, mimo_mtp_fail_closed=true, and mimo_mtp_depth matching --mode-label depth",
            "status": "fail_closed",
            "success_row_count": 0,
            "live_metrics_status": "unavailable",
            "non_fabrication_statement": "No live benchmark metrics are claimed in this blocked row; run rerun_command before using performance, speedup, or Slice 5 evidence.",
            "blocked_evidence": {
                "schema_version": "mimo_mtp_benchmark_evidence.v1",
                "evidence_kind": "blocked_evidence",
                "row_status": "fail_closed",
                "benchmark_session_id": "blocked:http://cluster.example:/bench/chat/completions:kernelpool/MiMo-V2.5-Pro-6bit:mtp-d1:16",
                "api_url": "http://cluster.example",
                "cluster_id": None,
                "endpoint": "/bench/chat/completions",
                "model_id": "kernelpool/MiMo-V2.5-Pro-6bit",
                "prompt_id": None,
                "prompt_hash": "2cf24dba5fb0a30e",
                "temperature": 0.0,
                "max_tokens": 16,
                "mode": "mtp-d1",
                "repeat_index": None,
                "generation_tps": None,
                "generation_tokens": None,
                "prompt_tps": None,
                "power_usage": None,
                "payload_extra": {"mimo_mtp_fastpath": True},
                "mtp_enabled": True,
                "mtp_depth": None,
                "mtp_execution_state": "fail_closed_error",
                "mtp_disable_reason": "MTP benchmark mode requires guarded payload fields: mimo_mtp_fastpath=true, mimo_mtp_fail_closed=true, and mimo_mtp_depth matching --mode-label depth",
                "telemetry_completeness": "fail_closed_no_live_execution",
                "runnable_command": "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://cluster.example --model-id kernelpool/MiMo-V2.5-Pro-6bit --prompt hello --max-tokens 16 --repeats 1 --mode-label mtp-d1 --payload-extra-json '{\"mimo_mtp_fastpath\": true}'",
                "expected_output_path": None,
                "blocker_reason": "MTP benchmark mode requires guarded payload fields: mimo_mtp_fastpath=true, mimo_mtp_fail_closed=true, and mimo_mtp_depth matching --mode-label depth",
                "blocked_stage": "mtp_guard_validation",
                "non_fabrication_statement": "No performance result is claimed because guarded MTP request validation failed before live benchmark execution.",
            },
            "rerun_command": "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://cluster.example --model-id kernelpool/MiMo-V2.5-Pro-6bit --prompt hello --max-tokens 16 --repeats 1 --mode-label mtp-d1 --payload-extra-json '{\"mimo_mtp_fastpath\": true}'",
            "next_step": "rerun with guarded MTP payload fields or use --mode-label ar for the default AR benchmark path",
            "blocked_reason_kind": "unguarded_mtp_benchmark_request",
            "benchmark_execution_attempted": False,
            "benchmark_endpoint": "/bench/chat/completions",
        }
    ]
