from __future__ import annotations

from collections.abc import Sequence

CLUSTER_BENCHMARK_SMOKE_COMMAND = (
    "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 "
    "--model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 16 --repeats 1 --mode-label ar --list-models"
)
BENCHMARK_VERIFICATION_COMMANDS = [
    "uv run pytest scripts/test_bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_fastpath.py scripts/test_mimo_mtp_benchmark_ingest.py scripts/test_mimo_mtp_budget_calculator.py scripts/test_mimo_mtp_bottleneck_classifier.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py -q",
    "uv run ruff check scripts/bench_mimo_mtp_cluster.py scripts/bench_mimo_mtp_fastpath.py scripts/mimo_mtp_benchmark_ingest.py scripts/mimo_mtp_budget_calculator.py scripts/mimo_mtp_bottleneck_classifier.py scripts/test_bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_fastpath.py scripts/test_mimo_mtp_benchmark_ingest.py scripts/test_mimo_mtp_budget_calculator.py scripts/test_mimo_mtp_bottleneck_classifier.py src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py",
    "uv run basedpyright scripts/bench_mimo_mtp_cluster.py scripts/bench_mimo_mtp_fastpath.py scripts/mimo_mtp_benchmark_ingest.py scripts/mimo_mtp_budget_calculator.py scripts/mimo_mtp_bottleneck_classifier.py src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py",
    CLUSTER_BENCHMARK_SMOKE_COMMAND,
]
API_VERIFICATION_COMMANDS = [
    "uv run pytest src/exo/api/tests/test_mimo_mtp_request_schema.py src/exo/api/tests/test_mimo_v25_pro_chat_adapter.py src/exo/api/tests/test_mimo_mtp_normal_execution_path.py src/exo/api/tests/test_mimo_v25_pro_preview_path.py scripts/test_validate_mimo_mtp_request_docs.py -q",
    "uv run ruff check src/exo/api scripts/validate_mimo_mtp_request_docs.py scripts/test_validate_mimo_mtp_request_docs.py",
    "uv run basedpyright src/exo/api scripts/validate_mimo_mtp_request_docs.py scripts/test_validate_mimo_mtp_request_docs.py",
    "uv run python3 scripts/validate_mimo_mtp_request_docs.py",
]
WORKER_VERIFICATION_COMMANDS = [
    "uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_module_validation.py src/exo/worker/tests/unittests/test_runner/test_batch_selection.py src/exo/worker/tests/unittests/test_mlx/test_kv_prefix_cache.py src/exo/worker/tests/unittests/test_mlx/test_warmup_inference.py -q",
    "uv run ruff check src/exo/worker/engines/mlx/mimo_mtp_fast/module_validation.py src/exo/worker/runner/llm_inference/batch_generator.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_module_validation.py src/exo/worker/tests/unittests/test_runner/test_batch_selection.py src/exo/worker/tests/unittests/test_mlx/test_kv_prefix_cache.py src/exo/worker/tests/unittests/test_mlx/test_warmup_inference.py",
    "uv run basedpyright src/exo/worker/engines/mlx/mimo_mtp_fast/module_validation.py src/exo/worker/runner/llm_inference/batch_generator.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_module_validation.py src/exo/worker/tests/unittests/test_runner/test_batch_selection.py src/exo/worker/tests/unittests/test_mlx/test_kv_prefix_cache.py src/exo/worker/tests/unittests/test_mlx/test_warmup_inference.py",
]
WORKER_EXECUTION_CONTRACT_TEST_COMMAND = (
    "uv run pytest src/exo/worker/tests src/exo/worker/engines/mlx/tests -q"
)
TASK_TYPE_VERIFICATION_COMMANDS = [
    "uv run pytest src/exo/shared/tests/test_mimo_mtp_task_params.py src/exo/shared/tests/test_state_serialization.py -q",
    "uv run ruff check src/exo/shared/types/tasks.py src/exo/shared/types/text_generation.py src/exo/shared/models/model_cards.py src/exo/shared/tests/test_mimo_mtp_task_params.py src/exo/shared/tests/test_state_serialization.py",
    "uv run basedpyright src/exo/shared/types/tasks.py src/exo/shared/types/text_generation.py src/exo/shared/models/model_cards.py src/exo/shared/tests/test_mimo_mtp_task_params.py src/exo/shared/tests/test_state_serialization.py",
]


def _normalize_changed_path(path: str) -> str:
    normalized_path = path.strip()
    while normalized_path.startswith("./"):
        normalized_path = normalized_path.removeprefix("./")
    while normalized_path.startswith("../"):
        normalized_path = normalized_path.removeprefix("../")
    return normalized_path


def _is_cluster_benchmark_path(path: str) -> bool:
    return path in {
        "scripts/bench_mimo_mtp_cluster.py",
        "scripts/test_bench_mimo_mtp_cluster.py",
    }


def _is_benchmark_telemetry_path(path: str) -> bool:
    return path in {
        "scripts/bench_mimo_mtp_fastpath.py",
        "scripts/mimo_mtp_benchmark_ingest.py",
        "scripts/mimo_mtp_bottleneck_classifier.py",
        "scripts/mimo_mtp_budget_calculator.py",
        "scripts/test_bench_mimo_mtp_fastpath.py",
        "scripts/test_mimo_mtp_benchmark_ingest.py",
        "scripts/test_mimo_mtp_bottleneck_classifier.py",
        "scripts/test_mimo_mtp_budget_calculator.py",
        "src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py",
        "src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py",
    } or path.startswith("scripts/fixtures/mimo_mtp_budget_")


def _is_api_related_path(path: str) -> bool:
    return path == "docs/api.md" or path.startswith("src/exo/api/")


def _is_worker_related_path(path: str) -> bool:
    return path.startswith("src/exo/worker/")


def _is_worker_execution_contract_path(path: str) -> bool:
    return path in {
        "src/exo/shared/types/tasks.py",
        "src/exo/shared/types/text_generation.py",
    }


def _is_task_type_model_path(path: str) -> bool:
    return path in {
        "src/exo/shared/types/tasks.py",
        "src/exo/shared/types/text_generation.py",
        "src/exo/shared/models/model_cards.py",
        "src/exo/shared/tests/test_mimo_mtp_task_params.py",
        "src/exo/shared/tests/test_state_serialization.py",
    }


def _is_task_verification_test_path(path: str) -> bool:
    return path in {
        "src/exo/shared/tests/test_mimo_mtp_task_params.py",
        "src/exo/shared/tests/test_state_serialization.py",
    }


def _is_cluster_task_path(path: str) -> bool:
    return path.startswith(("src/exo/api/", "src/exo/shared/types/", "src/exo/worker/"))


def _worker_execution_contract_commands(contract_paths: Sequence[str]) -> list[str]:
    checked_contract_paths = " ".join(contract_paths)
    return [
        WORKER_EXECUTION_CONTRACT_TEST_COMMAND,
        f"uv run ruff check src/exo/worker {checked_contract_paths}",
        f"uv run basedpyright src/exo/worker {checked_contract_paths}",
    ]


def discover_verification_commands(changed_files: Sequence[str]) -> list[str]:
    """Return focused runnable checks for changed rollout-related files.

    Non-task changes intentionally return no commands. API-only changes return API
    pytest/static checks only. Benchmark, worker, and cross-layer cluster seam
    changes return the smallest relevant runnable checks rather than repository-wide
    commands.
    """
    normalized_paths = [_normalize_changed_path(path) for path in changed_files]
    normalized_paths = [path for path in dict.fromkeys(normalized_paths) if path]
    has_cluster_benchmark_change = any(
        _is_cluster_benchmark_path(path) for path in normalized_paths
    )
    has_benchmark_telemetry_change = any(
        _is_benchmark_telemetry_path(path) for path in normalized_paths
    )
    has_api_change = any(_is_api_related_path(path) for path in normalized_paths)
    has_worker_change = any(_is_worker_related_path(path) for path in normalized_paths)
    has_worker_execution_contract_change = any(
        _is_worker_execution_contract_path(path) for path in normalized_paths
    )
    has_task_type_model_change = any(
        _is_task_type_model_path(path) for path in normalized_paths
    )
    has_task_verification_test_change = any(
        _is_task_verification_test_path(path) for path in normalized_paths
    )
    has_cluster_task_change = any(
        _is_cluster_task_path(path) for path in normalized_paths
    )
    has_shared_task_change = any(
        path.startswith("src/exo/shared/types/") for path in normalized_paths
    )

    if has_cluster_benchmark_change:
        return [
            "uv run pytest scripts/test_bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_fastpath.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py -q",
            "uv run ruff check scripts/bench_mimo_mtp_cluster.py scripts/bench_mimo_mtp_fastpath.py scripts/test_bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_fastpath.py src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py",
            "uv run basedpyright scripts/bench_mimo_mtp_cluster.py scripts/bench_mimo_mtp_fastpath.py src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py",
            CLUSTER_BENCHMARK_SMOKE_COMMAND,
        ]

    if has_benchmark_telemetry_change:
        return list(BENCHMARK_VERIFICATION_COMMANDS)

    if has_api_change and not has_worker_change and not has_shared_task_change:
        return list(API_VERIFICATION_COMMANDS)

    if has_worker_change and not has_api_change and not has_shared_task_change:
        return list(WORKER_VERIFICATION_COMMANDS)

    if has_task_type_model_change and (
        has_task_verification_test_change
        or any(
            path == "src/exo/shared/models/model_cards.py" for path in normalized_paths
        )
    ):
        return list(TASK_TYPE_VERIFICATION_COMMANDS)

    if (
        has_worker_execution_contract_change
        and not has_api_change
        and not has_worker_change
        and not has_benchmark_telemetry_change
    ):
        contract_paths = [
            path
            for path in normalized_paths
            if _is_worker_execution_contract_path(path)
        ]
        return _worker_execution_contract_commands(contract_paths)

    if has_cluster_task_change:
        task_paths = [path for path in normalized_paths if _is_cluster_task_path(path)]
        checked_task_paths = " ".join(task_paths)
        return [
            "uv run pytest scripts/test_bench_mimo_mtp_cluster.py -q",
            f"uv run ruff check scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py {checked_task_paths}",
            f"uv run basedpyright scripts/bench_mimo_mtp_cluster.py {checked_task_paths}",
            CLUSTER_BENCHMARK_SMOKE_COMMAND,
        ]

    if has_worker_change:
        return WORKER_VERIFICATION_COMMANDS

    if has_api_change:
        return API_VERIFICATION_COMMANDS

    return []
