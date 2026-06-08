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
            "src/exo/worker/engines/mlx/generator/batch_generate.py",
            "docs/worker.md",
        ]
    )

    assert commands == [
        "uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_module_validation.py src/exo/worker/tests/unittests/test_runner/test_batch_selection.py src/exo/worker/tests/unittests/test_mlx/test_kv_prefix_cache.py src/exo/worker/tests/unittests/test_mlx/test_warmup_inference.py -q",
        "uv run ruff check src/exo/worker/engines/mlx/mimo_mtp_fast/module_validation.py src/exo/worker/runner/llm_inference/batch_generator.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_module_validation.py src/exo/worker/tests/unittests/test_runner/test_batch_selection.py src/exo/worker/tests/unittests/test_mlx/test_kv_prefix_cache.py src/exo/worker/tests/unittests/test_mlx/test_warmup_inference.py",
        "uv run basedpyright src/exo/worker/engines/mlx/mimo_mtp_fast/module_validation.py src/exo/worker/runner/llm_inference/batch_generator.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_module_validation.py src/exo/worker/tests/unittests/test_runner/test_batch_selection.py src/exo/worker/tests/unittests/test_mlx/test_kv_prefix_cache.py src/exo/worker/tests/unittests/test_mlx/test_warmup_inference.py",
    ]
    assert all(
        command
        != "uv run pytest src/exo/worker/tests src/exo/worker/engines/mlx/tests -q"
        for command in commands
    )
    assert all("src/exo/worker" in command for command in commands)
    assert all("src/exo/api" not in command for command in commands)
    assert all(
        "scripts/bench_mimo_mtp_cluster.py" not in command for command in commands
    )


def test_worker_execution_contract_changes_return_focused_worker_checks() -> None:
    commands = discover_verification_commands(
        [
            "./src/exo/shared/types/tasks.py",
            "src/exo/shared/types/text_generation.py",
            "README.md",
        ]
    )

    assert commands == [
        "uv run pytest src/exo/worker/tests src/exo/worker/engines/mlx/tests -q",
        "uv run ruff check src/exo/worker src/exo/shared/types/tasks.py src/exo/shared/types/text_generation.py",
        "uv run basedpyright src/exo/worker src/exo/shared/types/tasks.py src/exo/shared/types/text_generation.py",
    ]
    assert all("src/exo/api" not in command for command in commands)
    assert all(
        "scripts/bench_mimo_mtp_cluster.py" not in command for command in commands
    )


def test_task_type_model_changes_return_focused_runnable_task_checks() -> None:
    commands = discover_verification_commands(
        [
            "./src/exo/shared/types/tasks.py",
            "src/exo/shared/types/text_generation.py",
            "src/exo/shared/models/model_cards.py",
            "README.md",
        ]
    )

    assert commands == [
        "uv run pytest src/exo/shared/tests/test_mimo_mtp_task_params.py src/exo/shared/tests/test_state_serialization.py -q",
        "uv run ruff check src/exo/shared/types/tasks.py src/exo/shared/types/text_generation.py src/exo/shared/models/model_cards.py src/exo/shared/tests/test_mimo_mtp_task_params.py src/exo/shared/tests/test_state_serialization.py",
        "uv run basedpyright src/exo/shared/types/tasks.py src/exo/shared/types/text_generation.py src/exo/shared/models/model_cards.py src/exo/shared/tests/test_mimo_mtp_task_params.py src/exo/shared/tests/test_state_serialization.py",
    ]
    assert all(
        "scripts/bench_mimo_mtp_cluster.py" not in command for command in commands
    )
    assert all("--list-models" not in command for command in commands)


def test_task_param_test_changes_return_focused_runnable_task_checks() -> None:
    commands = discover_verification_commands(
        [
            "./src/exo/shared/types/tasks.py",
            "src/exo/shared/tests/test_mimo_mtp_task_params.py",
            "README.md",
        ]
    )

    assert commands == [
        "uv run pytest src/exo/shared/tests/test_mimo_mtp_task_params.py src/exo/shared/tests/test_state_serialization.py -q",
        "uv run ruff check src/exo/shared/types/tasks.py src/exo/shared/types/text_generation.py src/exo/shared/models/model_cards.py src/exo/shared/tests/test_mimo_mtp_task_params.py src/exo/shared/tests/test_state_serialization.py",
        "uv run basedpyright src/exo/shared/types/tasks.py src/exo/shared/types/text_generation.py src/exo/shared/models/model_cards.py src/exo/shared/tests/test_mimo_mtp_task_params.py src/exo/shared/tests/test_state_serialization.py",
    ]
    assert all("src/exo/worker" not in command for command in commands)
    assert all(
        "scripts/bench_mimo_mtp_cluster.py" not in command for command in commands
    )
    assert all("--list-models" not in command for command in commands)


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
        "uv run pytest src/exo/api/tests/test_mimo_mtp_request_schema.py src/exo/api/tests/test_mimo_v25_pro_chat_adapter.py src/exo/api/tests/test_mimo_mtp_normal_execution_path.py src/exo/api/tests/test_mimo_v25_pro_preview_path.py scripts/test_validate_mimo_mtp_request_docs.py -q",
        "uv run ruff check src/exo/api scripts/validate_mimo_mtp_request_docs.py scripts/test_validate_mimo_mtp_request_docs.py",
        "uv run basedpyright src/exo/api scripts/validate_mimo_mtp_request_docs.py scripts/test_validate_mimo_mtp_request_docs.py",
        "uv run python3 scripts/validate_mimo_mtp_request_docs.py",
    ]
    assert all("src/exo/worker" not in command for command in commands)
    assert all(
        "scripts/bench_mimo_mtp_cluster.py" not in command for command in commands
    )


def test_ultrawork_relative_api_changes_return_focused_runnable_api_checks() -> None:
    commands = discover_verification_commands(
        [
            "../src/exo/api/main.py",
            "../src/exo/api/adapters/chat_completions.py",
            "../docs/api.md",
            "../README.md",
        ]
    )

    assert commands == [
        "uv run pytest src/exo/api/tests/test_mimo_mtp_request_schema.py src/exo/api/tests/test_mimo_v25_pro_chat_adapter.py src/exo/api/tests/test_mimo_mtp_normal_execution_path.py src/exo/api/tests/test_mimo_v25_pro_preview_path.py scripts/test_validate_mimo_mtp_request_docs.py -q",
        "uv run ruff check src/exo/api scripts/validate_mimo_mtp_request_docs.py scripts/test_validate_mimo_mtp_request_docs.py",
        "uv run basedpyright src/exo/api scripts/validate_mimo_mtp_request_docs.py scripts/test_validate_mimo_mtp_request_docs.py",
        "uv run python3 scripts/validate_mimo_mtp_request_docs.py",
    ]
    assert all("src/exo/worker" not in command for command in commands)
    assert all(
        "scripts/bench_mimo_mtp_cluster.py" not in command for command in commands
    )


def test_benchmark_telemetry_changes_return_focused_runnable_benchmark_validation_commands() -> (
    None
):
    commands = discover_verification_commands(
        [
            "scripts/mimo_mtp_budget_calculator.py",
            "scripts/test_mimo_mtp_budget_calculator.py",
            "scripts/mimo_mtp_benchmark_ingest.py",
            "scripts/mimo_mtp_bottleneck_classifier.py",
            "scripts/fixtures/mimo_mtp_budget_full_telemetry.jsonl",
            "src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py",
            "README.md",
        ]
    )

    assert commands == [
        "uv run pytest scripts/test_bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_fastpath.py scripts/test_mimo_mtp_benchmark_ingest.py scripts/test_mimo_mtp_budget_calculator.py scripts/test_mimo_mtp_bottleneck_classifier.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py -q",
        "uv run ruff check scripts/bench_mimo_mtp_cluster.py scripts/bench_mimo_mtp_fastpath.py scripts/mimo_mtp_benchmark_ingest.py scripts/mimo_mtp_budget_calculator.py scripts/mimo_mtp_bottleneck_classifier.py scripts/test_bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_fastpath.py scripts/test_mimo_mtp_benchmark_ingest.py scripts/test_mimo_mtp_budget_calculator.py scripts/test_mimo_mtp_bottleneck_classifier.py src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py",
        "uv run basedpyright scripts/bench_mimo_mtp_cluster.py scripts/bench_mimo_mtp_fastpath.py scripts/mimo_mtp_benchmark_ingest.py scripts/mimo_mtp_budget_calculator.py scripts/mimo_mtp_bottleneck_classifier.py src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py",
        "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 16 --repeats 1 --mode-label ar --list-models",
    ]
    assert all("src/exo/api" not in command for command in commands)
    assert any("test_mimo_mtp_budget_calculator.py" in command for command in commands)
    assert any("test_mimo_mtp_benchmark_ingest.py" in command for command in commands)
    assert any(
        "test_mimo_mtp_bottleneck_classifier.py" in command for command in commands
    )


def test_fastpath_benchmark_implementation_change_returns_benchmark_validation_commands() -> (
    None
):
    commands = discover_verification_commands(
        [
            "src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py",
            "README.md",
        ]
    )

    assert commands == [
        "uv run pytest scripts/test_bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_fastpath.py scripts/test_mimo_mtp_benchmark_ingest.py scripts/test_mimo_mtp_budget_calculator.py scripts/test_mimo_mtp_bottleneck_classifier.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py -q",
        "uv run ruff check scripts/bench_mimo_mtp_cluster.py scripts/bench_mimo_mtp_fastpath.py scripts/mimo_mtp_benchmark_ingest.py scripts/mimo_mtp_budget_calculator.py scripts/mimo_mtp_bottleneck_classifier.py scripts/test_bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_fastpath.py scripts/test_mimo_mtp_benchmark_ingest.py scripts/test_mimo_mtp_budget_calculator.py scripts/test_mimo_mtp_bottleneck_classifier.py src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py",
        "uv run basedpyright scripts/bench_mimo_mtp_cluster.py scripts/bench_mimo_mtp_fastpath.py scripts/mimo_mtp_benchmark_ingest.py scripts/mimo_mtp_budget_calculator.py scripts/mimo_mtp_bottleneck_classifier.py src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py",
        "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 16 --repeats 1 --mode-label ar --list-models",
    ]
    assert all("src/exo/api" not in command for command in commands)
