from __future__ import annotations

from scripts.verification_command_discovery import discover_verification_commands


def test_benchmark_related_changes_return_focused_benchmark_checks_only() -> None:
    commands = discover_verification_commands(
        [
            "scripts/bench_mimo_mtp_cluster.py",
            "scripts/test_bench_mimo_mtp_cluster.py",
            "src/exo/api/main.py",
            "README.md",
        ]
    )

    assert commands == [
        "uv run pytest scripts/test_bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_fastpath.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py -q",
        "uv run ruff check scripts/bench_mimo_mtp_cluster.py scripts/bench_mimo_mtp_fastpath.py scripts/test_bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_fastpath.py src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py",
        "uv run basedpyright scripts/bench_mimo_mtp_cluster.py scripts/bench_mimo_mtp_fastpath.py src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py",
        "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 16 --repeats 1 --mode-label ar --list-models",
    ]


def test_cluster_api_or_task_changes_include_cluster_benchmark_checks() -> None:
    commands = discover_verification_commands(
        [
            "src/exo/api/types/api.py",
            "src/exo/shared/types/tasks.py",
            "src/exo/worker/main.py",
        ]
    )

    assert commands == [
        "uv run pytest scripts/test_bench_mimo_mtp_cluster.py -q",
        "uv run ruff check scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py src/exo/api/types/api.py src/exo/shared/types/tasks.py src/exo/worker/main.py",
        "uv run basedpyright scripts/bench_mimo_mtp_cluster.py src/exo/api/types/api.py src/exo/shared/types/tasks.py src/exo/worker/main.py",
        "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 16 --repeats 1 --mode-label ar --list-models",
    ]


def test_worker_related_changes_return_focused_runnable_worker_checks_only() -> None:
    commands = discover_verification_commands(
        [
            "src/exo/worker/main.py",
            "src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py",
            "docs/worker.md",
        ]
    )

    assert commands == [
        "uv run pytest src/exo/worker/tests src/exo/worker/engines/mlx/tests -q",
        "uv run ruff check src/exo/worker",
        "uv run basedpyright src/exo/worker",
    ]
    assert all("src/exo/worker" in command for command in commands)
    assert all("src/exo/api" not in command for command in commands)
    assert all(
        "scripts/bench_mimo_mtp_cluster.py" not in command for command in commands
    )


def test_unrelated_changes_do_not_return_benchmark_checks() -> None:
    assert discover_verification_commands(["dashboard/src/App.svelte", "LICENSE"]) == []


def test_api_related_changes_return_focused_runnable_api_checks_only() -> None:
    commands = discover_verification_commands(
        [
            "src/exo/api/main.py",
            "src/exo/api/adapters/chat_completions.py",
            "docs/api.md",
            "README.md",
        ]
    )

    assert commands == [
        "uv run pytest src/exo/api/tests -q",
        "uv run ruff check src/exo/api src/exo/api/tests",
        "uv run basedpyright src/exo/api",
    ]
    assert all("src/exo/worker" not in command for command in commands)
    assert all(
        "scripts/bench_mimo_mtp_cluster.py" not in command for command in commands
    )
