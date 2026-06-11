from __future__ import annotations

from scripts.mimo_mtp_rollout_tracking import (
    AcceptanceCriterionGateMetadata,
    AcceptanceCriterionSyncItem,
    RolloutArtifactSyncResult,
    RolloutSyncPayload,
    RolloutTrackingSyncResult,
    VerificationCommandIndexEntry,
    normalize_rollout_acceptance_criteria,
    orchestrate_rollout_tracking_sync,
)
from scripts.rollout_bookkeeping_sinks import (
    BeadsBookkeepingOperation,
    BookkeepingDispatchFailure,
    BookkeepingSinkStatus,
    FileBookkeepingOperation,
)

_OPTIMIZED_MIMO_ACCEPTANCE_CRITERIA: tuple[str, ...] = (
    """AC-P2 MTP speedup budget model exists before optimization claims:
    - add a script or documented calculator that consumes AR/MTP JSONL rows
    - compute AR ms/token, MTP tok/s, target gaps for 30 and 40 tok/s, speedup ratio, and pass/fail/ambiguous classification
    - classify bottlenecks as acceptance_rate_low, proposal_too_slow, verifier_too_slow, fallback_too_high, depth_too_aggressive, mtp_beats_ar, mtp_reaches_30, and mtp_reaches_40 where telemetry exists
    - do not require live MTP rows to pass tests; handle absent rows honestly""",
    """AC-P0 Optimized rollout seed and repo context are validated:
    - confirm repo root, branch, git cleanliness, latest MiMo MTP commits, and existing fastpath modules
    - record that exo cluster tensor parallelization is the live model path, not single-Studio full-model loading
    - mirror ACs into TODO/runlog/Beads where practical
    - identify focused verification commands for API/task/worker/benchmark changes""",
    """AC-P10 Verification and closeout:
    - focused pytest, ruff, ruff format check, basedpyright, and git diff --check pass for touched surfaces, or failures are documented as pre-existing/blocked
    - docs/runlog/TODO/Beads are updated with AC statuses, commands, evidence, and blockers
    - final report identifies completed ACs, remaining ACs, worktrees reviewed/merged, tests run, and next best step""",
    """AC-P1 Cluster AR baseline harness is canonical and budget-ready:
    - keep scripts/bench_mimo_mtp_cluster.py as the live benchmark path through /bench/chat/completions
    - ensure rows include generation_tps, generation_tokens, prompt_tps, power_usage, model id, mode label, repeat index, and payload extra keys
    - add or update tests for AR baseline row parsing and unavailable-cluster failure behavior
    - provide exact commands for max_tokens=16 and max_tokens=64 AR baseline rows
    - if a cluster is available, collect rows; otherwise record a clear blocked-with-command status""",
    """AC-P9 Slice 5 gate remains strict:
    - default generation remains AR
    - guarded production integration review is allowed only if same-cluster MTP median tok/s beats AR by a meaningful margin or reaches explicit target with no correctness/fallback concern
    - recommended margin: >=10-15% median tok/s improvement and preferably >=30 tok/s; 40+ tok/s is preferred target
    - docs/TODO/Beads explicitly state whether Slice 5 is blocked or eligible""",
)


_REQUIRED_CONCEPTS: tuple[str, ...] = (
    "cluster_baseline",
    "mtp_vertical_slice",
    "benchmark_telemetry",
    "speedup_budget",
    "slice_5_gate",
)


_EXPECTED_VERIFICATION_COMMAND_INDEX: tuple[VerificationCommandIndexEntry, ...] = (
    VerificationCommandIndexEntry(
        index=0,
        category="api",
        command=(
            "uv run pytest src/exo/api/tests/test_mimo_mtp_request_schema.py "
            "src/exo/api/tests/test_mimo_v25_pro_chat_adapter.py "
            "src/exo/api/tests/test_mimo_mtp_normal_execution_path.py "
            "src/exo/api/tests/test_mimo_v25_pro_preview_path.py "
            "scripts/test_validate_mimo_mtp_request_docs.py -q"
        ),
        purpose="Run focused API tests for request/response contract changes.",
        requires_live_cluster=False,
    ),
    VerificationCommandIndexEntry(
        index=1,
        category="task_type",
        command=(
            "uv run pytest src/exo/shared/tests/test_mimo_mtp_task_params.py "
            "src/exo/shared/tests/test_state_serialization.py -q"
        ),
        purpose="Run focused shared task parameter serialization tests.",
        requires_live_cluster=False,
    ),
    VerificationCommandIndexEntry(
        index=2,
        category="worker",
        command=(
            "uv run pytest src/exo/worker/engines/mlx/tests/"
            "test_mimo_mtp_fast_module_validation.py "
            "src/exo/worker/tests/unittests/test_runner/test_batch_selection.py "
            "src/exo/worker/tests/unittests/test_mlx/test_kv_prefix_cache.py "
            "src/exo/worker/tests/unittests/test_mlx/test_warmup_inference.py -q"
        ),
        purpose="Run focused worker tests for execution-path changes.",
        requires_live_cluster=False,
    ),
    VerificationCommandIndexEntry(
        index=3,
        category="benchmark_telemetry",
        command=(
            "uv run pytest scripts/test_bench_mimo_mtp_cluster.py "
            "scripts/test_bench_mimo_mtp_fastpath.py "
            "scripts/test_mimo_mtp_benchmark_ingest.py "
            "scripts/test_mimo_mtp_budget_calculator.py "
            "scripts/test_mimo_mtp_bottleneck_classifier.py "
            "src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py -q"
        ),
        purpose="Run benchmark telemetry, budget, and bottleneck tests.",
        requires_live_cluster=False,
    ),
)


def test_normalize_rollout_acceptance_criteria_emits_deterministic_sync_payload() -> (
    None
):
    payload = normalize_rollout_acceptance_criteria(
        acceptance_criteria=_OPTIMIZED_MIMO_ACCEPTANCE_CRITERIA,
        seed_id="mimo_mtp_optimized_rollout_20260606T203800",
        ontology_name="MimoMtpOptimizedRollout",
        required_concepts=_REQUIRED_CONCEPTS,
        status_by_id={"AC-P10": "done", "AC-P1": "blocked_with_command"},
        evidence_by_id={
            "AC-P1": (".goose-ultrawork/evidence/ac-p1-summary.md",),
            "AC-P9": (".goose-ultrawork/evidence/ac-p9-slice-5-strict-gate.md",),
        },
    )

    assert payload == RolloutSyncPayload(
        seed_id="mimo_mtp_optimized_rollout_20260606T203800",
        ontology_name="MimoMtpOptimizedRollout",
        schema_version="mimo-mtp-rollout-sync/v1",
        gate=AcceptanceCriterionGateMetadata(
            default_generation_mode="ar",
            slice_5_status="blocked",
            requires_same_cluster_rows=True,
            requires_real_mtp_speed_win=True,
            target_tok_s=30.0,
            preferred_tok_s=40.0,
            required_evidence=(
                "same_cluster_ar_vs_mtp_rows",
                "no_correctness_or_fallback_concern",
            ),
            guardrail=(
                "Slice 5 remains blocked until same-cluster AR-vs-MTP rows prove "
                "a real MTP speed win; default exo generation remains AR."
            ),
        ),
        items=(
            AcceptanceCriterionSyncItem(
                stable_id="mimo_mtp_optimized_rollout_20260606T203800:AC-P0",
                acceptance_id="AC-P0",
                order=0,
                title="Optimized rollout seed and repo context are validated",
                status="pending",
                concepts=(
                    "cluster_baseline",
                    "mtp_vertical_slice",
                    "benchmark_telemetry",
                    "speedup_budget",
                    "slice_5_gate",
                ),
                evidence_paths=(),
                gate_tags=("slice_5_blocked",),
                evidence_contract_notes=(
                    "exo cluster tensor parallelization is the live model path, not single-Studio full-model loading",
                ),
                verification_command_index=_EXPECTED_VERIFICATION_COMMAND_INDEX,
            ),
            AcceptanceCriterionSyncItem(
                stable_id="mimo_mtp_optimized_rollout_20260606T203800:AC-P1",
                acceptance_id="AC-P1",
                order=1,
                title="Cluster AR baseline harness is canonical and budget-ready",
                status="blocked",
                concepts=(
                    "cluster_baseline",
                    "benchmark_telemetry",
                    "speedup_budget",
                    "slice_5_gate",
                ),
                evidence_paths=(".goose-ultrawork/evidence/ac-p1-summary.md",),
                gate_tags=(
                    "cluster_baseline",
                    "same_cluster_evidence",
                    "slice_5_blocked",
                ),
            ),
            AcceptanceCriterionSyncItem(
                stable_id="mimo_mtp_optimized_rollout_20260606T203800:AC-P2",
                acceptance_id="AC-P2",
                order=2,
                title="MTP speedup budget model exists before optimization claims",
                status="pending",
                concepts=(
                    "cluster_baseline",
                    "benchmark_telemetry",
                    "speedup_budget",
                    "slice_5_gate",
                ),
                evidence_paths=(),
                gate_tags=("benchmark_telemetry", "speedup_budget", "slice_5_blocked"),
            ),
            AcceptanceCriterionSyncItem(
                stable_id="mimo_mtp_optimized_rollout_20260606T203800:AC-P9",
                acceptance_id="AC-P9",
                order=9,
                title="Slice 5 gate remains strict",
                status="pending",
                concepts=("cluster_baseline", "speedup_budget", "slice_5_gate"),
                evidence_paths=(
                    ".goose-ultrawork/evidence/ac-p9-slice-5-strict-gate.md",
                ),
                gate_tags=(
                    "default_ar",
                    "production_gate",
                    "same_cluster_evidence",
                    "slice_5_blocked",
                    "speedup_budget",
                ),
            ),
            AcceptanceCriterionSyncItem(
                stable_id="mimo_mtp_optimized_rollout_20260606T203800:AC-P10",
                acceptance_id="AC-P10",
                order=10,
                title="Verification and closeout",
                status="complete",
                concepts=("slice_5_gate",),
                evidence_paths=(),
                gate_tags=("slice_5_blocked",),
                verification_command_index=_EXPECTED_VERIFICATION_COMMAND_INDEX,
            ),
        ),
    )


def test_canonical_evidence_index_preserves_tensor_parallel_live_model_path_note() -> (
    None
):
    payload = normalize_rollout_acceptance_criteria(
        acceptance_criteria=(
            """AC-P0 Optimized rollout seed and repo context are validated:
            - record that exo cluster tensor parallelization is the live model path, not single-Studio full-model loading""",
        ),
        seed_id="mimo_mtp_optimized_rollout_20260606T203800",
        ontology_name="MimoMtpOptimizedRollout",
        required_concepts=_REQUIRED_CONCEPTS,
    )

    ac_p0 = payload.items[0]
    assert ac_p0.evidence_contract_notes == (
        "exo cluster tensor parallelization is the live model path, not single-Studio full-model loading",
    )


def test_canonical_evidence_index_preserves_verification_commands_deterministically() -> (
    None
):
    payload = normalize_rollout_acceptance_criteria(
        acceptance_criteria=(
            """AC-P0 Optimized rollout seed and repo context are validated:
            - identify focused verification commands for API/task/worker/benchmark changes""",
            """AC-P10 Verification and closeout:
            - focused pytest, ruff, ruff format check, basedpyright, and git diff --check pass for touched surfaces""",
        ),
        seed_id="mimo_mtp_optimized_rollout_20260606T203800",
        ontology_name="MimoMtpOptimizedRollout",
        required_concepts=_REQUIRED_CONCEPTS,
    )

    ac_p0 = payload.items[0]
    ac_p10 = payload.items[1]

    assert ac_p0.verification_command_index == _EXPECTED_VERIFICATION_COMMAND_INDEX
    assert ac_p10.verification_command_index == ac_p0.verification_command_index

    indexed_commands = [entry.command for entry in ac_p0.verification_command_index]
    assert indexed_commands == sorted(indexed_commands, key=indexed_commands.index)
    assert len(indexed_commands) == len(set(indexed_commands))
    assert tuple(entry.index for entry in ac_p0.verification_command_index) == tuple(
        range(len(ac_p0.verification_command_index))
    )


def test_normalize_rollout_acceptance_criteria_rejects_missing_or_duplicate_ids() -> (
    None
):
    duplicate_criteria = (
        "AC-P1 Cluster AR baseline harness is canonical and budget-ready:",
        "AC-P1 Duplicate title:",
    )

    try:
        normalize_rollout_acceptance_criteria(
            acceptance_criteria=duplicate_criteria,
            seed_id="mimo_mtp_optimized_rollout_20260606T203800",
            ontology_name="MimoMtpOptimizedRollout",
            required_concepts=_REQUIRED_CONCEPTS,
        )
    except ValueError as exc:
        assert str(exc) == "duplicate acceptance criterion id: AC-P1"
    else:
        raise AssertionError("expected duplicate AC id rejection")

    try:
        normalize_rollout_acceptance_criteria(
            acceptance_criteria=("Cluster AR baseline harness is canonical",),
            seed_id="mimo_mtp_optimized_rollout_20260606T203800",
            ontology_name="MimoMtpOptimizedRollout",
            required_concepts=_REQUIRED_CONCEPTS,
        )
    except ValueError as exc:
        assert str(exc) == "acceptance criterion is missing canonical AC-P id"
    else:
        raise AssertionError("expected missing AC id rejection")


def test_normalize_rollout_acceptance_criteria_rejects_unknown_status() -> None:
    try:
        normalize_rollout_acceptance_criteria(
            acceptance_criteria=(
                "AC-P7 Same-cluster AR-vs-MTP benchmark matrix is runnable and evidence-gated:",
            ),
            seed_id="mimo_mtp_optimized_rollout_20260606T203800",
            ontology_name="MimoMtpOptimizedRollout",
            required_concepts=_REQUIRED_CONCEPTS,
            status_by_id={"AC-P7": "fast_enough"},
        )
    except ValueError as exc:
        assert str(exc) == "unknown rollout AC status for AC-P7: fast_enough"
    else:
        raise AssertionError("expected unknown status rejection")


def test_evidence_index_contains_no_entries_absent_from_input_sources() -> None:
    """AC Sub-AC 4b: The evidence index produced by the sync/indexing function
    must contain no entries absent from the input sources (no phantom/inflated
    entries). We inject a known superset of all verification command descriptors
    from the four hardcoded categories and assert every index entry's command
    string is found in that superset, proving strict subset containment.
    """
    from scripts.verification_command_registry import (
        get_verification_commands_for_change_category,
    )

    # Build the known superset: ALL descriptors from the four categories that
    # _verification_command_index_for_criterion iterates over.  The indexing
    # function only takes [0] from each category, so the full descriptor set
    # is guaranteed to be a superset of anything the index can produce.
    _INDEXED_CATEGORIES: tuple[str, ...] = (  # noqa: N806
        "api",
        "task_type",
        "worker",
        "benchmark_telemetry",
    )
    superset_commands: set[str] = set()
    superset_by_category: dict[str, set[str]] = {}
    for category in _INDEXED_CATEGORIES:
        category_commands: set[str] = set()
        for descriptor in get_verification_commands_for_change_category(category):
            superset_commands.add(descriptor.command)
            category_commands.add(descriptor.command)
        superset_by_category[category] = category_commands

    # Produce the evidence index via the normal sync function.
    payload = normalize_rollout_acceptance_criteria(
        acceptance_criteria=(
            """AC-P0 Optimized rollout seed and repo context are validated:
- identify focused verification commands for API/task/worker/benchmark changes""",
            """AC-P10 Verification and closeout:
- focused pytest, ruff, ruff format check, basedpyright, and git diff --check pass for touched surfaces""",
        ),
        seed_id="mimo_mtp_optimized_rollout_20260606T203800",
        ontology_name="MimoMtpOptimizedRollout",
        required_concepts=_REQUIRED_CONCEPTS,
    )

    # Collect ALL index entries across all items.
    all_index_entries: list[VerificationCommandIndexEntry] = []
    for item in payload.items:
        all_index_entries.extend(item.verification_command_index)

    assert len(all_index_entries) > 0, (
        "evidence index must contain entries for verification-command-bearing ACs"
    )

    # Core assertion: every index entry's command must exist in the superset.
    phantom_entries: list[VerificationCommandIndexEntry] = []
    for entry in all_index_entries:
        if entry.command not in superset_commands:
            phantom_entries.append(entry)

    assert phantom_entries == [], (
        f"evidence index contains {len(phantom_entries)} phantom entries "
        f"absent from all input source descriptors:\n"
        + "\n".join(
            f"  index={e.index} category={e.category} command={e.command!r}"
            for e in phantom_entries
        )
    )

    # Additionally verify the index is a strict subset (not just subset-of-self):
    # at least one superset command must NOT appear in the index, proving the
    # superset is genuinely larger and the index is a *strict* subset.
    indexed_commands = {entry.command for entry in all_index_entries}
    unindexed_superset_commands = superset_commands - indexed_commands
    assert len(unindexed_superset_commands) > 0, (
        "superset must be strictly larger than the index to prove strict-subset "
        "property; the indexing function takes only [0] per category so the "
        "remaining descriptors must produce superset surplus"
    )

    # Verify per-category subset containment: each index entry's category must
    # have its command present in that category's superset slice.
    for entry in all_index_entries:
        category_superset = superset_by_category.get(entry.category, set())
        assert entry.command in category_superset, (
            f"index entry category={entry.category} command={entry.command!r} "
            f"not found in that category's input source descriptors"
        )


def _available_statuses(
    *, beads_available: bool = True
) -> tuple[BookkeepingSinkStatus, ...]:
    return (
        BookkeepingSinkStatus(
            name="todo",
            enabled=True,
            configured=True,
            available=True,
            target="TODO.md",
            reason="enabled configured target is available",
        ),
        BookkeepingSinkStatus(
            name="runlog",
            enabled=True,
            configured=True,
            available=True,
            target="docs/plans/MIMO_V25_PRO_MTP_FOUNDATION_RUNLOG.md",
            reason="enabled configured target is available",
        ),
        BookkeepingSinkStatus(
            name="beads",
            enabled=True,
            configured=beads_available,
            available=beads_available,
            target=(
                "mimo-v25-pro-mtp-d1-live-preflight-20260601-5ge"
                if beads_available
                else None
            ),
            reason=(
                "enabled configured target is available"
                if beads_available
                else "sink is not configured"
            ),
        ),
    )


def test_orchestrate_rollout_tracking_sync_invokes_artifact_paths_and_reports_partial_failures() -> (
    None
):
    dispatched: list[str] = []

    def dispatch_todo(operation: FileBookkeepingOperation) -> None:
        dispatched.append(f"todo:{operation.path}")

    def dispatch_runlog(operation: FileBookkeepingOperation) -> None:
        dispatched.append(f"runlog:{operation.path}")
        raise RuntimeError("runlog write denied")

    def dispatch_beads(operation: BeadsBookkeepingOperation) -> None:
        dispatched.append(f"beads:{operation.issue_id}")

    result = orchestrate_rollout_tracking_sync(
        acceptance_criteria=(
            "AC-P0 Optimized rollout seed and repo context are validated:",
        ),
        seed_id="mimo_mtp_optimized_rollout_20260606T203800",
        ontology_name="MimoMtpOptimizedRollout",
        required_concepts=_REQUIRED_CONCEPTS,
        status_by_id={"AC-P0": "complete"},
        evidence_by_id={"AC-P0": (".goose-ultrawork/evidence/ac-p0.md",)},
        sink_statuses=_available_statuses(),
        timestamp="2026-06-07T01:55:00",
        dispatch_todo=dispatch_todo,
        dispatch_runlog=dispatch_runlog,
        dispatch_beads=dispatch_beads,
    )

    assert dispatched == [
        "todo:TODO.md",
        "runlog:docs/plans/MIMO_V25_PRO_MTP_FOUNDATION_RUNLOG.md",
        "beads:mimo-v25-pro-mtp-d1-live-preflight-20260601-5ge.0",
    ]
    assert result == RolloutTrackingSyncResult(
        payload=RolloutSyncPayload(
            seed_id="mimo_mtp_optimized_rollout_20260606T203800",
            ontology_name="MimoMtpOptimizedRollout",
            schema_version="mimo-mtp-rollout-sync/v1",
            gate=AcceptanceCriterionGateMetadata(
                default_generation_mode="ar",
                slice_5_status="blocked",
                requires_same_cluster_rows=True,
                requires_real_mtp_speed_win=True,
                target_tok_s=30.0,
                preferred_tok_s=40.0,
                required_evidence=(
                    "same_cluster_ar_vs_mtp_rows",
                    "no_correctness_or_fallback_concern",
                ),
                guardrail=(
                    "Slice 5 remains blocked until same-cluster AR-vs-MTP rows prove "
                    "a real MTP speed win; default exo generation remains AR."
                ),
            ),
            items=(
                AcceptanceCriterionSyncItem(
                    stable_id="mimo_mtp_optimized_rollout_20260606T203800:AC-P0",
                    acceptance_id="AC-P0",
                    order=0,
                    title="Optimized rollout seed and repo context are validated",
                    status="complete",
                    concepts=(
                        "cluster_baseline",
                        "mtp_vertical_slice",
                        "benchmark_telemetry",
                        "speedup_budget",
                        "slice_5_gate",
                    ),
                    evidence_paths=(".goose-ultrawork/evidence/ac-p0.md",),
                    gate_tags=("slice_5_blocked",),
                ),
            ),
        ),
        artifact_results=(
            RolloutArtifactSyncResult(
                acceptance_id="AC-P0",
                artifact="todo",
                target="TODO.md",
                succeeded=True,
                error=None,
            ),
            RolloutArtifactSyncResult(
                acceptance_id="AC-P0",
                artifact="runlog",
                target="docs/plans/MIMO_V25_PRO_MTP_FOUNDATION_RUNLOG.md",
                succeeded=False,
                error="runlog write denied",
            ),
            RolloutArtifactSyncResult(
                acceptance_id="AC-P0",
                artifact="beads",
                target="mimo-v25-pro-mtp-d1-live-preflight-20260601-5ge.0",
                succeeded=True,
                error=None,
            ),
        ),
        failures=(
            BookkeepingDispatchFailure(
                sink="runlog",
                target="docs/plans/MIMO_V25_PRO_MTP_FOUNDATION_RUNLOG.md",
                error="runlog write denied",
            ),
        ),
    )


def test_orchestrate_rollout_tracking_sync_skips_unavailable_optional_beads_path() -> (
    None
):
    dispatched: list[str] = []

    result = orchestrate_rollout_tracking_sync(
        acceptance_criteria=(
            "AC-P1 Cluster AR baseline harness is canonical and budget-ready:",
        ),
        seed_id="mimo_mtp_optimized_rollout_20260606T203800",
        ontology_name="MimoMtpOptimizedRollout",
        required_concepts=_REQUIRED_CONCEPTS,
        sink_statuses=_available_statuses(beads_available=False),
        timestamp="2026-06-07T01:55:00",
        dispatch_todo=lambda operation: dispatched.append(operation.sink),
        dispatch_runlog=lambda operation: dispatched.append(operation.sink),
        dispatch_beads=lambda _operation: dispatched.append("beads"),
    )

    assert dispatched == ["todo", "runlog"]
    assert result.artifact_results == (
        RolloutArtifactSyncResult(
            acceptance_id="AC-P1",
            artifact="todo",
            target="TODO.md",
            succeeded=True,
            error=None,
        ),
        RolloutArtifactSyncResult(
            acceptance_id="AC-P1",
            artifact="runlog",
            target="docs/plans/MIMO_V25_PRO_MTP_FOUNDATION_RUNLOG.md",
            succeeded=True,
            error=None,
        ),
    )
    assert result.failures == ()
