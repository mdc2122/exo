from __future__ import annotations

from collections.abc import Sequence

CLUSTER_BENCHMARK_SMOKE_COMMAND = (
    "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 "
    "--model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 16 --repeats 1 --mode-label ar --list-models"
)
API_VERIFICATION_COMMANDS = [
    "uv run pytest src/exo/api/tests -q",
    "uv run ruff check src/exo/api src/exo/api/tests",
    "uv run basedpyright src/exo/api",
]
WORKER_VERIFICATION_COMMANDS = [
    "uv run pytest src/exo/worker/tests src/exo/worker/engines/mlx/tests -q",
    "uv run ruff check src/exo/worker",
    "uv run basedpyright src/exo/worker",
]


def _is_cluster_benchmark_path(path: str) -> bool:
    return path in {
        "scripts/bench_mimo_mtp_cluster.py",
        "scripts/test_bench_mimo_mtp_cluster.py",
    }


def _is_api_related_path(path: str) -> bool:
    return path == "docs/api.md" or path.startswith("src/exo/api/")


def _is_worker_related_path(path: str) -> bool:
    return path.startswith("src/exo/worker/")


def _is_cluster_task_path(path: str) -> bool:
    return path.startswith(("src/exo/api/", "src/exo/shared/types/", "src/exo/worker/"))


def discover_verification_commands(changed_files: Sequence[str]) -> list[str]:
    """Return focused runnable checks for changed rollout-related files.

    Non-task changes intentionally return no commands. API-only changes return API
    pytest/static checks only. Benchmark, worker, and cross-layer cluster seam
    changes return the smallest relevant runnable checks rather than repository-wide
    commands.
    """
    normalized_paths = [path.strip().removeprefix("./") for path in changed_files]
    normalized_paths = [path for path in dict.fromkeys(normalized_paths) if path]
    has_benchmark_change = any(
        _is_cluster_benchmark_path(path) for path in normalized_paths
    )
    has_api_change = any(_is_api_related_path(path) for path in normalized_paths)
    has_worker_change = any(_is_worker_related_path(path) for path in normalized_paths)
    has_cluster_task_change = any(
        _is_cluster_task_path(path) for path in normalized_paths
    )
    has_shared_task_change = any(
        path.startswith("src/exo/shared/types/") for path in normalized_paths
    )

    if has_benchmark_change:
        return [
            "uv run pytest scripts/test_bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_fastpath.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py -q",
            "uv run ruff check scripts/bench_mimo_mtp_cluster.py scripts/bench_mimo_mtp_fastpath.py scripts/test_bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_fastpath.py src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py",
            "uv run basedpyright scripts/bench_mimo_mtp_cluster.py scripts/bench_mimo_mtp_fastpath.py src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py",
            CLUSTER_BENCHMARK_SMOKE_COMMAND,
        ]

    if has_api_change and not has_worker_change and not has_shared_task_change:
        return list(API_VERIFICATION_COMMANDS)

    if has_worker_change and not has_api_change and not has_shared_task_change:
        return list(WORKER_VERIFICATION_COMMANDS)

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
