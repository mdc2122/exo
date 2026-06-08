from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from scripts.verification_command_discovery import (
    API_VERIFICATION_COMMANDS,
    BENCHMARK_VERIFICATION_COMMANDS,
    CLUSTER_BENCHMARK_SMOKE_COMMAND,
    TASK_TYPE_VERIFICATION_COMMANDS,
    WORKER_EXECUTION_CONTRACT_TEST_COMMAND,
    WORKER_VERIFICATION_COMMANDS,
)

ChangeCategory = Literal[
    "api",
    "benchmark_telemetry",
    "cluster_benchmark",
    "cluster_task",
    "rollout_closeout",
    "task_type",
    "verification_command_evidence",
    "worker",
    "worker_execution_contract",
]


@dataclass(frozen=True, slots=True)
class VerificationCommandDescriptor:
    category: ChangeCategory
    command: str
    purpose: str
    requires_live_cluster: bool = False


CLUSTER_BENCHMARK_VALIDATION_COMMANDS: Final[tuple[str, ...]] = (
    "uv run pytest scripts/test_bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_fastpath.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py -q",
    "uv run ruff check scripts/bench_mimo_mtp_cluster.py scripts/bench_mimo_mtp_fastpath.py scripts/test_bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_fastpath.py src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py",
    "uv run basedpyright scripts/bench_mimo_mtp_cluster.py scripts/bench_mimo_mtp_fastpath.py src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py",
    CLUSTER_BENCHMARK_SMOKE_COMMAND,
)
CLUSTER_TASK_VALIDATION_COMMANDS: Final[tuple[str, ...]] = (
    "uv run pytest scripts/test_bench_mimo_mtp_cluster.py -q",
    "uv run ruff check scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py {changed_paths}",
    "uv run basedpyright scripts/bench_mimo_mtp_cluster.py {changed_paths}",
    CLUSTER_BENCHMARK_SMOKE_COMMAND,
)
WORKER_EXECUTION_CONTRACT_VALIDATION_COMMANDS: Final[tuple[str, ...]] = (
    WORKER_EXECUTION_CONTRACT_TEST_COMMAND,
    "uv run ruff check src/exo/worker {changed_paths}",
    "uv run basedpyright src/exo/worker {changed_paths}",
)
BENCHMARK_MATRIX_COMMAND: Final[str] = (
    "uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 "
    "--model-id kernelpool/MiMo-V2.5-Pro-6bit --repeats 1 --matrix-commands"
)
BENCHMARK_BUDGET_GATE_COMMAND: Final[str] = (
    "uv run python3 scripts/mimo_mtp_budget_calculator.py "
    "scripts/fixtures/mimo_mtp_budget_ar_plus_mtp.jsonl"
)
BENCHMARK_RUNTIME_EVIDENCE_COMMANDS: Final[tuple[str, ...]] = (
    BENCHMARK_MATRIX_COMMAND,
    BENCHMARK_BUDGET_GATE_COMMAND,
)
VERIFICATION_COMMAND_EVIDENCE_COMMANDS: Final[tuple[str, ...]] = (
    "uv run pytest scripts/test_verification_command_registry.py scripts/test_verification_command_discovery.py -q",
    "uv run ruff check scripts/verification_command_registry.py scripts/verification_command_discovery.py scripts/test_verification_command_registry.py scripts/test_verification_command_discovery.py",
    "uv run basedpyright scripts/verification_command_registry.py scripts/verification_command_discovery.py scripts/test_verification_command_registry.py scripts/test_verification_command_discovery.py",
)
ROLLOUT_CLOSEOUT_FOCUSED_VERIFICATION_COMMANDS: Final[tuple[str, ...]] = (
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
_COMMANDS_BY_CHANGE_CATEGORY: Final[dict[ChangeCategory, tuple[str, ...]]] = {
    "api": tuple(API_VERIFICATION_COMMANDS),
    "benchmark_telemetry": tuple(BENCHMARK_VERIFICATION_COMMANDS)
    + BENCHMARK_RUNTIME_EVIDENCE_COMMANDS,
    "cluster_benchmark": CLUSTER_BENCHMARK_VALIDATION_COMMANDS,
    "cluster_task": CLUSTER_TASK_VALIDATION_COMMANDS,
    "rollout_closeout": ROLLOUT_CLOSEOUT_FOCUSED_VERIFICATION_COMMANDS,
    "task_type": tuple(TASK_TYPE_VERIFICATION_COMMANDS),
    "verification_command_evidence": VERIFICATION_COMMAND_EVIDENCE_COMMANDS,
    "worker": tuple(WORKER_VERIFICATION_COMMANDS),
    "worker_execution_contract": WORKER_EXECUTION_CONTRACT_VALIDATION_COMMANDS,
}

_PURPOSES_BY_COMMAND_PREFIX: Final[tuple[tuple[str, str], ...]] = (
    (
        "uv run pytest src/exo/api/tests",
        "Run focused API tests for request/response contract changes.",
    ),
    ("uv run ruff check src/exo/api", "Lint focused API sources and tests."),
    ("uv run basedpyright src/exo/api", "Type-check focused API sources."),
    (
        "uv run pytest src/exo/worker",
        "Run focused worker tests for execution-path changes.",
    ),
    ("uv run ruff check src/exo/worker", "Lint focused worker sources and tests."),
    ("uv run basedpyright src/exo/worker", "Type-check focused worker sources."),
    (
        "uv run pytest scripts/test_bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_fastpath.py scripts/test_mimo_mtp_benchmark_ingest.py",
        "Run benchmark telemetry, budget, and bottleneck tests.",
    ),
    (
        "uv run pytest scripts/test_bench_mimo_mtp_cluster.py",
        "Run focused MiMo MTP benchmark tests.",
    ),
    (
        "uv run ruff check scripts/bench_mimo_mtp_cluster.py",
        "Lint focused MiMo MTP benchmark sources and tests.",
    ),
    (
        "uv run basedpyright scripts/bench_mimo_mtp_cluster.py",
        "Type-check focused MiMo MTP benchmark sources.",
    ),
    (
        BENCHMARK_MATRIX_COMMAND,
        "Emit same-cluster AR-vs-guarded-MTP benchmark matrix commands without contacting the cluster.",
    ),
    (
        BENCHMARK_BUDGET_GATE_COMMAND,
        "Compute 30/40 tok/s speedup budget and strict Slice 5 gate from benchmark JSONL rows.",
    ),
    (
        "uv run pytest scripts/test_verification_command_registry.py scripts/test_verification_command_discovery.py",
        "Run verification-command evidence contract guard tests without product-code scope.",
    ),
    (
        "uv run ruff check scripts/verification_command_registry.py scripts/verification_command_discovery.py",
        "Lint verification-command evidence contract surfaces only.",
    ),
    (
        "uv run basedpyright scripts/verification_command_registry.py scripts/verification_command_discovery.py",
        "Type-check verification-command evidence contract surfaces only.",
    ),
    (
        "uv run pytest -q scripts/test_bench_mimo_mtp_cluster.py",
        "Run focused rollout closeout verification commands preserved by the evidence contract.",
    ),
    ("uv run ruff format --check", "Check formatting for rollout closeout surfaces."),
    ("git diff --check", "Check the working tree diff for whitespace errors."),
    (
        "uv run python3 scripts/mimo_mtp_bottleneck_classifier.py",
        "Classify bottlenecks from benchmark-grade MTP telemetry rows.",
    ),
    (
        "uv run pytest src/exo/shared/tests/test_mimo_mtp_task_params.py",
        "Run focused shared task parameter serialization tests.",
    ),
    (
        "uv run ruff check src/exo/shared/types/tasks.py",
        "Lint focused shared task/model sources and tests.",
    ),
    (
        "uv run basedpyright src/exo/shared/types/tasks.py",
        "Type-check focused shared task/model sources and tests.",
    ),
)


def _coerce_change_category(category: str) -> ChangeCategory:
    match category:
        case (
            "api"
            | "benchmark_telemetry"
            | "cluster_benchmark"
            | "cluster_task"
            | "rollout_closeout"
            | "task_type"
            | "verification_command_evidence"
            | "worker"
            | "worker_execution_contract"
        ):
            return category
        case _:
            known_categories = ", ".join(sorted(_COMMANDS_BY_CHANGE_CATEGORY))
            raise ValueError(
                f"unknown verification command change category: {category!r}; known categories: {known_categories}"
            )


def _requires_live_cluster(command: str) -> bool:
    return command in {
        CLUSTER_BENCHMARK_SMOKE_COMMAND,
        ROLLOUT_CLOSEOUT_FOCUSED_VERIFICATION_COMMANDS[-1],
    }


def _purpose_for_command(command: str) -> str:
    if _requires_live_cluster(command):
        return "Probe the live cluster API for a same-cluster AR benchmark smoke row."
    for prefix, purpose in _PURPOSES_BY_COMMAND_PREFIX:
        if command.startswith(prefix):
            return purpose
    return "Run focused verification for this change category."


def _descriptor_for_command(
    *, category: ChangeCategory, command: str
) -> VerificationCommandDescriptor:
    return VerificationCommandDescriptor(
        category=category,
        command=command,
        purpose=_purpose_for_command(command),
        requires_live_cluster=_requires_live_cluster(command),
    )


def get_verification_commands_for_change_category(
    category: str,
) -> tuple[VerificationCommandDescriptor, ...]:
    """Return typed verification command descriptors for a change category."""
    change_category = _coerce_change_category(category)
    return tuple(
        _descriptor_for_command(category=change_category, command=command)
        for command in _COMMANDS_BY_CHANGE_CATEGORY[change_category]
    )
