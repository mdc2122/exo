from __future__ import annotations

import dataclasses

import pytest

from scripts.verification_command_discovery import (
    API_VERIFICATION_COMMANDS,
    BENCHMARK_VERIFICATION_COMMANDS,
)
from scripts.verification_command_registry import (
    VerificationCommandDescriptor,
    get_verification_commands_for_change_category,
)

ROLLOUT_CLOSEOUT_FOCUSED_VERIFICATION_COMMANDS = (
    "uv run pytest -q scripts/test_bench_mimo_mtp_cluster.py scripts/test_git_cleanliness.py scripts/test_mimo_mtp_benchmark_ingest.py scripts/test_mimo_mtp_bottleneck_classifier.py scripts/test_mimo_mtp_budget_calculator.py scripts/test_mimo_v25_pro_repo_root.py scripts/test_mimo_v25_pro_runtime_guard.py scripts/test_mirror_mimo_mtp_rollout_seed.py scripts/test_mirror_rollout_seed_to_beads.py scripts/test_rollout_seed_mirroring.py scripts/test_validate_mimo_mtp_request_docs.py scripts/test_verification_command_discovery.py scripts/test_mimo_mtp_commit_detection.py scripts/test_mimo_mtp_model_path_evidence.py scripts/test_mimo_mtp_rollout_tracking.py scripts/test_rollout_bookkeeping_sinks.py scripts/test_verification_command_registry.py src/exo/api/tests/test_mimo_mtp_normal_execution_path.py src/exo/api/tests/test_mimo_mtp_request_schema.py src/exo/api/tests/test_mimo_v25_pro_chat_adapter.py src/exo/api/tests/test_mimo_v25_pro_preview_path.py src/exo/shared/tests/test_mimo_mtp_task_params.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_module_validation.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_worker_fastpath.py src/exo/worker/tests/unittests/test_runner/test_mimo_mtp_worker_generator_routing.py",
    "uv run ruff check scripts/bench_mimo_mtp_cluster.py scripts/git_cleanliness.py scripts/mimo_mtp_benchmark_ingest.py scripts/mimo_mtp_bottleneck_classifier.py scripts/mimo_mtp_budget_calculator.py scripts/mimo_v25_pro_repo_root.py scripts/mimo_v25_pro_runtime_guard.py scripts/mirror_mimo_mtp_rollout_seed.py scripts/mirror_rollout_seed_to_beads.py scripts/rollout_seed_mirroring.py scripts/validate_mimo_mtp_request_docs.py scripts/verification_command_discovery.py scripts/mimo_mtp_commit_detection.py scripts/mimo_mtp_model_path_evidence.py scripts/mimo_mtp_rollout_tracking.py scripts/rollout_bookkeeping_sinks.py scripts/verification_command_registry.py scripts/test_bench_mimo_mtp_cluster.py scripts/test_git_cleanliness.py scripts/test_mimo_mtp_benchmark_ingest.py scripts/test_mimo_mtp_bottleneck_classifier.py scripts/test_mimo_mtp_budget_calculator.py scripts/test_mimo_v25_pro_repo_root.py scripts/test_mimo_v25_pro_runtime_guard.py scripts/test_mirror_mimo_mtp_rollout_seed.py scripts/test_mirror_rollout_seed_to_beads.py scripts/test_rollout_seed_mirroring.py scripts/test_validate_mimo_mtp_request_docs.py scripts/test_verification_command_discovery.py scripts/test_mimo_mtp_commit_detection.py scripts/test_mimo_mtp_model_path_evidence.py scripts/test_mimo_mtp_rollout_tracking.py scripts/test_rollout_bookkeeping_sinks.py scripts/test_verification_command_registry.py src/exo/api/main.py src/exo/api/types/api.py src/exo/api/adapters/chat_completions.py src/exo/api/tests/test_mimo_mtp_normal_execution_path.py src/exo/api/tests/test_mimo_mtp_request_schema.py src/exo/api/tests/test_mimo_v25_pro_chat_adapter.py src/exo/api/tests/test_mimo_v25_pro_preview_path.py src/exo/shared/types/text_generation.py src/exo/shared/tests/test_mimo_mtp_task_params.py src/exo/worker/engines/mlx/mimo_mtp_fast/module_validation.py src/exo/worker/engines/mlx/mimo_mtp_fast/worker_fastpath.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_module_validation.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_worker_fastpath.py src/exo/worker/runner/llm_inference/batch_generator.py src/exo/worker/tests/unittests/test_runner/test_mimo_mtp_worker_generator_routing.py",
    "uv run ruff format --check scripts/bench_mimo_mtp_cluster.py scripts/git_cleanliness.py scripts/mimo_mtp_benchmark_ingest.py scripts/mimo_mtp_bottleneck_classifier.py scripts/mimo_mtp_budget_calculator.py scripts/mimo_v25_pro_repo_root.py scripts/mimo_v25_pro_runtime_guard.py scripts/mirror_mimo_mtp_rollout_seed.py scripts/mirror_rollout_seed_to_beads.py scripts/rollout_seed_mirroring.py scripts/validate_mimo_mtp_request_docs.py scripts/verification_command_discovery.py scripts/mimo_mtp_commit_detection.py scripts/mimo_mtp_model_path_evidence.py scripts/mimo_mtp_rollout_tracking.py scripts/rollout_bookkeeping_sinks.py scripts/verification_command_registry.py scripts/test_bench_mimo_mtp_cluster.py scripts/test_git_cleanliness.py scripts/test_mimo_mtp_benchmark_ingest.py scripts/test_mimo_mtp_bottleneck_classifier.py scripts/test_mimo_mtp_budget_calculator.py scripts/test_mimo_v25_pro_repo_root.py scripts/test_mimo_v25_pro_runtime_guard.py scripts/test_mirror_mimo_mtp_rollout_seed.py scripts/test_mirror_rollout_seed_to_beads.py scripts/test_rollout_seed_mirroring.py scripts/test_validate_mimo_mtp_request_docs.py scripts/test_verification_command_discovery.py scripts/test_mimo_mtp_commit_detection.py scripts/test_mimo_mtp_model_path_evidence.py scripts/test_mimo_mtp_rollout_tracking.py scripts/test_rollout_bookkeeping_sinks.py scripts/test_verification_command_registry.py src/exo/api/main.py src/exo/api/types/api.py src/exo/api/adapters/chat_completions.py src/exo/api/tests/test_mimo_mtp_normal_execution_path.py src/exo/api/tests/test_mimo_mtp_request_schema.py src/exo/api/tests/test_mimo_v25_pro_chat_adapter.py src/exo/api/tests/test_mimo_v25_pro_preview_path.py src/exo/shared/types/text_generation.py src/exo/shared/tests/test_mimo_mtp_task_params.py src/exo/worker/engines/mlx/mimo_mtp_fast/module_validation.py src/exo/worker/engines/mlx/mimo_mtp_fast/worker_fastpath.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_module_validation.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_worker_fastpath.py src/exo/worker/runner/llm_inference/batch_generator.py src/exo/worker/tests/unittests/test_runner/test_mimo_mtp_worker_generator_routing.py",
    "uv run basedpyright scripts/bench_mimo_mtp_cluster.py scripts/git_cleanliness.py scripts/mimo_mtp_benchmark_ingest.py scripts/mimo_mtp_bottleneck_classifier.py scripts/mimo_mtp_budget_calculator.py scripts/mimo_v25_pro_repo_root.py scripts/mimo_v25_pro_runtime_guard.py scripts/mirror_mimo_mtp_rollout_seed.py scripts/mirror_rollout_seed_to_beads.py scripts/rollout_seed_mirroring.py scripts/validate_mimo_mtp_request_docs.py scripts/verification_command_discovery.py scripts/mimo_mtp_commit_detection.py scripts/mimo_mtp_model_path_evidence.py scripts/mimo_mtp_rollout_tracking.py scripts/rollout_bookkeeping_sinks.py scripts/verification_command_registry.py scripts/test_bench_mimo_mtp_cluster.py scripts/test_git_cleanliness.py scripts/test_mimo_mtp_benchmark_ingest.py scripts/test_mimo_mtp_bottleneck_classifier.py scripts/test_mimo_mtp_budget_calculator.py scripts/test_mimo_v25_pro_repo_root.py scripts/test_mimo_v25_pro_runtime_guard.py scripts/test_mirror_mimo_mtp_rollout_seed.py scripts/test_mirror_rollout_seed_to_beads.py scripts/test_rollout_seed_mirroring.py scripts/test_validate_mimo_mtp_request_docs.py scripts/test_verification_command_discovery.py scripts/test_mimo_mtp_commit_detection.py scripts/test_mimo_mtp_model_path_evidence.py scripts/test_mimo_mtp_rollout_tracking.py scripts/test_rollout_bookkeeping_sinks.py scripts/test_verification_command_registry.py src/exo/api/main.py src/exo/api/types/api.py src/exo/api/adapters/chat_completions.py src/exo/api/tests/test_mimo_mtp_normal_execution_path.py src/exo/api/tests/test_mimo_mtp_request_schema.py src/exo/api/tests/test_mimo_v25_pro_chat_adapter.py src/exo/api/tests/test_mimo_v25_pro_preview_path.py src/exo/shared/types/text_generation.py src/exo/shared/tests/test_mimo_mtp_task_params.py src/exo/worker/engines/mlx/mimo_mtp_fast/module_validation.py src/exo/worker/engines/mlx/mimo_mtp_fast/worker_fastpath.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_module_validation.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_worker_fastpath.py src/exo/worker/runner/llm_inference/batch_generator.py src/exo/worker/tests/unittests/test_runner/test_mimo_mtp_worker_generator_routing.py",
    "git diff --check",
    "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --repeats 1 --matrix-commands",
    "uv run python3 scripts/mimo_mtp_budget_calculator.py scripts/fixtures/mimo_mtp_budget_ar_plus_mtp.jsonl",
    "uv run python3 scripts/mimo_mtp_bottleneck_classifier.py scripts/fixtures/mimo_mtp_budget_full_telemetry.jsonl",
    "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 16 --repeats 1 --mode-label ar --list-models",
)


def test_api_change_category_returns_typed_command_descriptors() -> None:
    descriptors = get_verification_commands_for_change_category("api")

    assert tuple(descriptor.command for descriptor in descriptors) == tuple(
        API_VERIFICATION_COMMANDS
    )
    assert tuple(descriptor.category for descriptor in descriptors) == (
        "api",
        "api",
        "api",
        "api",
    )
    assert tuple(descriptor.requires_live_cluster for descriptor in descriptors) == (
        False,
        False,
        False,
        False,
    )
    assert descriptors[0].purpose == (
        "Run focused API tests for request/response contract changes."
    )
    assert all(
        isinstance(descriptor, VerificationCommandDescriptor)
        for descriptor in descriptors
    )


def test_rollout_closeout_category_preserves_exact_focused_verification_commands() -> (
    None
):
    descriptors = get_verification_commands_for_change_category("rollout_closeout")

    assert tuple(descriptor.command for descriptor in descriptors) == (
        ROLLOUT_CLOSEOUT_FOCUSED_VERIFICATION_COMMANDS
    )
    assert len(descriptors) == len(ROLLOUT_CLOSEOUT_FOCUSED_VERIFICATION_COMMANDS)
    assert tuple(descriptor.category for descriptor in descriptors) == (
        "rollout_closeout",
    ) * len(ROLLOUT_CLOSEOUT_FOCUSED_VERIFICATION_COMMANDS)
    assert [
        descriptor.command
        for descriptor in descriptors
        if descriptor.requires_live_cluster
    ] == [
        ROLLOUT_CLOSEOUT_FOCUSED_VERIFICATION_COMMANDS[-1],
    ]


def test_registry_descriptors_are_immutable() -> None:
    descriptor = get_verification_commands_for_change_category("worker")[0]

    command_field_name = "command"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(descriptor, command_field_name, "uv run pytest")


def test_benchmark_telemetry_category_uses_specific_telemetry_purpose() -> None:
    descriptors = get_verification_commands_for_change_category("benchmark_telemetry")

    assert descriptors[0].command == BENCHMARK_VERIFICATION_COMMANDS[0]
    assert (
        descriptors[0].purpose
        == "Run benchmark telemetry, budget, and bottleneck tests."
    )


def test_benchmark_telemetry_category_includes_runnable_matrix_and_gate_entries() -> (
    None
):
    descriptors = get_verification_commands_for_change_category("benchmark_telemetry")
    commands_by_purpose = {
        descriptor.purpose: descriptor.command for descriptor in descriptors
    }

    assert commands_by_purpose[
        "Emit same-cluster AR-vs-guarded-MTP benchmark matrix commands without contacting the cluster."
    ] == (
        "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 "
        "--model-id kernelpool/MiMo-V2.5-Pro-6bit --repeats 1 --matrix-commands"
    )
    assert commands_by_purpose[
        "Compute 30/40 tok/s speedup budget and strict Slice 5 gate from benchmark JSONL rows."
    ] == (
        "uv run python3 scripts/mimo_mtp_budget_calculator.py "
        "scripts/fixtures/mimo_mtp_budget_ar_plus_mtp.jsonl"
    )
    assert any(
        command
        == (
            "uv run pytest scripts/test_bench_mimo_mtp_cluster.py "
            "scripts/test_bench_mimo_mtp_fastpath.py "
            "scripts/test_mimo_mtp_benchmark_ingest.py "
            "scripts/test_mimo_mtp_budget_calculator.py "
            "scripts/test_mimo_mtp_bottleneck_classifier.py "
            "src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py -q"
        )
        for command in commands_by_purpose.values()
    )
    assert all(
        descriptor.category == "benchmark_telemetry" for descriptor in descriptors
    )
    assert [
        descriptor.command
        for descriptor in descriptors
        if descriptor.requires_live_cluster
    ] == [
        "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 "
        "--model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 16 --repeats 1 --mode-label ar --list-models"
    ]


def test_cluster_benchmark_category_includes_live_cluster_smoke_descriptor() -> None:
    descriptors = get_verification_commands_for_change_category("cluster_benchmark")

    live_cluster_descriptors = [
        descriptor for descriptor in descriptors if descriptor.requires_live_cluster
    ]
    assert len(live_cluster_descriptors) == 1
    assert live_cluster_descriptors[0].command == (
        "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 "
        "--model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 16 --repeats 1 --mode-label ar --list-models"
    )
    assert "cluster API" in live_cluster_descriptors[0].purpose


def test_lookup_rejects_unknown_change_category() -> None:
    with pytest.raises(
        ValueError, match="unknown verification command change category"
    ):
        get_verification_commands_for_change_category("dashboard")


def test_verification_command_evidence_contract_does_not_expand_ac_p0_implementation_scope() -> (
    None
):
    descriptors = get_verification_commands_for_change_category(
        "verification_command_evidence"
    )

    commands = tuple(descriptor.command for descriptor in descriptors)

    assert commands == (
        "uv run pytest scripts/test_verification_command_registry.py scripts/test_verification_command_discovery.py -q",
        "uv run ruff check scripts/verification_command_registry.py scripts/verification_command_discovery.py scripts/test_verification_command_registry.py scripts/test_verification_command_discovery.py",
        "uv run basedpyright scripts/verification_command_registry.py scripts/verification_command_discovery.py scripts/test_verification_command_registry.py scripts/test_verification_command_discovery.py",
    )
    assert all(
        descriptor.category == "verification_command_evidence"
        for descriptor in descriptors
    )
    assert all(not descriptor.requires_live_cluster for descriptor in descriptors)
    assert all("src/exo/" not in command for command in commands)
    assert all(
        "scripts/bench_mimo_mtp_cluster.py" not in command for command in commands
    )
    assert all("--list-models" not in command for command in commands)


def test_worker_category_returns_focused_mimo_worker_verification_entries() -> None:
    descriptors = get_verification_commands_for_change_category("worker")

    commands = tuple(descriptor.command for descriptor in descriptors)

    assert commands == (
        "uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_module_validation.py src/exo/worker/tests/unittests/test_runner/test_batch_selection.py src/exo/worker/tests/unittests/test_mlx/test_kv_prefix_cache.py src/exo/worker/tests/unittests/test_mlx/test_warmup_inference.py -q",
        "uv run ruff check src/exo/worker/engines/mlx/mimo_mtp_fast/module_validation.py src/exo/worker/runner/llm_inference/batch_generator.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_module_validation.py src/exo/worker/tests/unittests/test_runner/test_batch_selection.py src/exo/worker/tests/unittests/test_mlx/test_kv_prefix_cache.py src/exo/worker/tests/unittests/test_mlx/test_warmup_inference.py",
        "uv run basedpyright src/exo/worker/engines/mlx/mimo_mtp_fast/module_validation.py src/exo/worker/runner/llm_inference/batch_generator.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_module_validation.py src/exo/worker/tests/unittests/test_runner/test_batch_selection.py src/exo/worker/tests/unittests/test_mlx/test_kv_prefix_cache.py src/exo/worker/tests/unittests/test_mlx/test_warmup_inference.py",
    )
    assert all(descriptor.category == "worker" for descriptor in descriptors)
    assert all(not descriptor.requires_live_cluster for descriptor in descriptors)
    assert all("src/exo/worker" in command for command in commands)
    assert all(
        "scripts/bench_mimo_mtp_cluster.py" not in command for command in commands
    )
    assert all(
        command
        != "uv run pytest src/exo/worker/tests src/exo/worker/engines/mlx/tests -q"
        for command in commands
    )
