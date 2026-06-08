"""Tests for the canonical benchmark telemetry row model.

Verifies:
- Default instantiation produces a valid AR row.
- Frozen/strict mode rejects mutation and type coercion.
- Full MTP row instantiation and serialization round-trip.
- Blocked-evidence row instantiation.
- JSON round-trip preserves types and values.
- Extra fields are rejected (strict mode).
"""

# pyright: reportAny=false
import pytest
from pydantic import ValidationError

from exo.shared.types.benchmark_telemetry import (
    BENCHMARK_ROW_SCHEMA_VERSION,
    BenchmarkRow,
)

# ---------------------------------------------------------------------------
# Default AR row
# ---------------------------------------------------------------------------


class TestDefaultArRow:
    def test_default_instantiation_is_ar(self) -> None:
        row = BenchmarkRow()
        assert row.schema_version == BENCHMARK_ROW_SCHEMA_VERSION
        assert row.evidence_kind == "canonical_benchmark_row"
        assert row.row_status == "valid"
        assert row.mode == "ar"
        assert row.mtp_enabled is False
        assert row.mtp_execution_state == "disabled_default"
        assert row.mtp_depth is None
        assert row.mtp_disable_reason is None
        assert row.telemetry_completeness == "full"

    def test_default_ar_row_serializes_without_mtp_noise(self) -> None:
        row = BenchmarkRow()
        dumped = row.model_dump()
        assert dumped["mode"] == "ar"
        assert dumped["mtp_enabled"] is False
        assert dumped["mtp_depth"] is None
        assert dumped["mtp_execution_state"] == "disabled_default"

    def test_default_ar_row_json_round_trips(self) -> None:
        row = BenchmarkRow()
        raw = row.model_dump_json()
        restored = BenchmarkRow.model_validate_json(raw)
        assert restored == row


# ---------------------------------------------------------------------------
# Frozen model — mutation is rejected
# ---------------------------------------------------------------------------


class TestFrozenEnforcement:
    def test_frozen_rejects_field_mutation(self) -> None:
        row = BenchmarkRow()
        with pytest.raises(ValidationError):
            row.mode = "mtp"

    def test_frozen_rejects_generation_tps_mutation(self) -> None:
        row = BenchmarkRow()
        with pytest.raises(ValidationError):
            row.generation_tps = 99.0


# ---------------------------------------------------------------------------
# Strict mode — type coercion is rejected
# ---------------------------------------------------------------------------


class TestStrictEnforcement:
    def test_strict_rejects_coerced_generation_tps(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkRow(generation_tps="42.0")  # type: ignore[arg-type]

    def test_strict_rejects_coerced_mtp_enabled_string(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkRow(mtp_enabled="true")  # type: ignore[arg-type]

    def test_strict_rejects_coerced_repeat_index_float(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkRow(repeat_index=1.5)  # type: ignore[arg-type]

    def test_strict_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkRow(unexpected_controller_flag=True)  # type: ignore[call-arg]

    def test_strict_rejects_invalid_mode(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkRow(mode="speculative")  # type: ignore[arg-type]

    def test_strict_rejects_invalid_mtp_execution_state(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkRow(mtp_execution_state="maybe_mtp")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Full MTP row
# ---------------------------------------------------------------------------


class TestMtpRow:
    def test_full_mtp_row_instantiation(self) -> None:
        row = BenchmarkRow(
            benchmark_session_id="sess-001",
            api_url="http://localhost:52415",
            endpoint="/bench/chat/completions",
            model_id="mlx-community/Qwen3-30B-A3B-4bit",
            prompt_id="prompt-42",
            temperature=0.0,
            max_tokens=256,
            mode="mtp",
            repeat_index=0,
            generation_tps=35.0,
            generation_tokens=256,
            prompt_tps=800.0,
            mtp_enabled=True,
            mtp_depth=3,
            mtp_execution_state="successful_mtp",
            requested_mtp_depth=3,
            attempted_depth_counts={"1": 10, "2": 20, "3": 226},
            accepted_depth_counts={"1": 8, "2": 15, "3": 200},
            acceptance_rate=0.88,
            fallback_count=5,
            timing_breakdown_seconds={"draft": 0.3, "verify": 0.7},
            telemetry_completeness="full",
        )
        assert row.mode == "mtp"
        assert row.mtp_enabled is True
        assert row.mtp_depth == 3
        assert row.mtp_execution_state == "successful_mtp"
        assert row.acceptance_rate == 0.88
        assert row.fallback_count == 5

    def test_mtp_row_json_round_trips(self) -> None:
        row = BenchmarkRow(
            api_url="http://localhost:52415",
            endpoint="/bench/chat/completions",
            model_id="test-model",
            mode="mtp",
            mtp_enabled=True,
            mtp_depth=2,
            mtp_execution_state="fail_open_fallback",
            mtp_fallback_reason="sidecar_not_found",
            generation_tps=28.0,
            generation_tokens=128,
            telemetry_completeness="partial",
        )
        raw = row.model_dump_json()
        restored_from_json = BenchmarkRow.model_validate_json(raw)
        assert restored_from_json.mode == "mtp"
        assert restored_from_json.mtp_execution_state == "fail_open_fallback"
        assert restored_from_json.mtp_fallback_reason == "sidecar_not_found"
        restored = BenchmarkRow.model_validate_json(raw)
        assert restored == row


# ---------------------------------------------------------------------------
# Fail-closed / fail-open states
# ---------------------------------------------------------------------------


class TestMtpExecutionStates:
    @pytest.mark.parametrize(
        "state",
        [
            "disabled_default",
            "enabled_intent",
            "compatible_attempted",
            "successful_mtp",
            "fail_closed_error",
            "fail_open_fallback",
            "unsupported_model",
            "unsupported_depth",
            "missing_sidecar",
            "unwired_execution",
            "runtime_error",
            "blocked_unavailable",
        ],
    )
    def test_all_mtp_execution_states_accepted(self, state: str) -> None:
        row = BenchmarkRow(mtp_execution_state=state)  # pyright: ignore[reportArgumentType]
        assert row.mtp_execution_state == state

    def test_fail_closed_error_row(self) -> None:
        row = BenchmarkRow(
            mode="mtp",
            mtp_enabled=True,
            mtp_execution_state="fail_closed_error",
            mtp_disable_reason="unsupported_depth",
            generation_tps=0.0,
            telemetry_completeness="minimal",
        )
        assert row.mode == "mtp"
        assert row.mtp_execution_state == "fail_closed_error"
        assert row.generation_tps == 0.0

    def test_fail_open_fallback_row_labeled_not_successful_mtp(self) -> None:
        row = BenchmarkRow(
            mode="ar",
            mtp_enabled=True,
            mtp_execution_state="fail_open_fallback",
            mtp_fallback_reason="sidecar_cache_miss",
            generation_tps=22.0,
            generation_tokens=128,
            telemetry_completeness="partial",
        )
        # Fail-open fallback is AR behavior; mode must reflect actual generation
        assert row.mode == "ar"
        assert row.mtp_execution_state == "fail_open_fallback"
        assert row.mtp_enabled is True  # intent was MTP


# ---------------------------------------------------------------------------
# Blocked evidence row
# ---------------------------------------------------------------------------


class TestBlockedEvidenceRow:
    def test_blocked_evidence_row(self) -> None:
        row = BenchmarkRow(
            evidence_kind="blocked_evidence",
            row_status="blocked",
            api_url="http://localhost:52415",
            model_id="test-model",
            endpoint="/bench/chat/completions",
            mtp_execution_state="blocked_unavailable",
            telemetry_completeness="unavailable",
        )
        assert row.evidence_kind == "blocked_evidence"
        assert row.row_status == "blocked"
        assert row.mtp_execution_state == "blocked_unavailable"
        assert row.generation_tps == 0.0  # no performance claim

    def test_blocked_row_serializes_with_zero_tps(self) -> None:
        row = BenchmarkRow(
            evidence_kind="blocked_evidence",
            row_status="blocked",
        )
        dumped = row.model_dump()
        assert dumped["generation_tps"] == 0.0
        assert dumped["generation_tokens"] == 0


# ---------------------------------------------------------------------------
# Constraint validation
# ---------------------------------------------------------------------------


class TestFieldConstraints:
    def test_negative_generation_tps_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkRow(generation_tps=-1.0)

    def test_negative_repeat_index_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkRow(repeat_index=-1)

    def test_mtp_depth_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkRow(mtp_depth=0)

    def test_mtp_depth_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkRow(mtp_depth=-1)

    def test_acceptance_rate_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkRow(acceptance_rate=1.5)

    def test_acceptance_rate_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkRow(acceptance_rate=-0.1)

    def test_fallback_count_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkRow(fallback_count=-1)


# ---------------------------------------------------------------------------
# Blocked row where MTP was requested but could not proceed
# (AC Sub-AC 1.2: mtp_enabled=True + mtp_disable_reason is set)
# ---------------------------------------------------------------------------


class TestMtpRequestedButBlockedRow:
    """Verify that a benchmark telemetry row where mtp_enabled=True but
    mtp_disable_reason is also set — indicating MTP was *requested* but
    *blocked* — is accepted by the model and faithfully reflects the blocked
    condition.

    This is the "no silent AR-as-MTP" pattern: intent was MTP, outcome was
    not MTP, and both are recorded.
    """

    def test_mtp_enabled_with_disable_reason_is_accepted(self) -> None:
        """Core AC test: mtp_enabled=True + mtp_disable_reason set constructs
        a valid row that faithfully records the blocked condition."""
        row = BenchmarkRow(
            api_url="http://localhost:52415",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            endpoint="/bench/chat/completions",
            mode="ar",
            mtp_enabled=True,
            mtp_execution_state="fail_closed_error",
            mtp_disable_reason="missing_sidecar",
            generation_tps=0.0,
            telemetry_completeness="minimal",
        )
        assert row.mtp_enabled is True
        assert row.mtp_disable_reason == "missing_sidecar"
        assert row.mtp_execution_state == "fail_closed_error"
        assert row.mode == "ar"  # actual execution was AR

    def test_blocked_row_with_fail_open_fallback_records_both(self) -> None:
        """A fail-open fallback row with mtp_enabled=True records both MTP
        intent and the fallback reason."""
        row = BenchmarkRow(
            mode="ar",
            mtp_enabled=True,
            mtp_execution_state="fail_open_fallback",
            mtp_disable_reason="native_runtime_disabled",
            mtp_fallback_reason="fail_open_native_runtime_disabled",
            generation_tps=20.0,
            generation_tokens=128,
            telemetry_completeness="partial",
        )
        assert row.mtp_enabled is True
        assert row.mtp_disable_reason == "native_runtime_disabled"
        assert row.mtp_fallback_reason == "fail_open_native_runtime_disabled"
        assert row.mode == "ar"  # ran AR, not MTP

    @pytest.mark.parametrize(
        "execution_state, disable_reason",
        [
            ("fail_closed_error", "unsupported_model"),
            ("fail_closed_error", "missing_sidecar"),
            ("fail_open_fallback", "sidecar_load_failed"),
            ("unwired_execution", "mimo_mtp_distributed_generator_unwired"),
            ("blocked_unavailable", "cluster_unavailable"),
        ],
    )
    def test_parametrized_blocking_states(
        self, execution_state: str, disable_reason: str
    ) -> None:
        """Multiple blocking states with mtp_enabled=True + disable_reason
        are accepted and never labeled successful_mtp."""
        row = BenchmarkRow(
            mode="ar",
            mtp_enabled=True,
            mtp_execution_state=execution_state,  # type: ignore[arg-type]
            mtp_disable_reason=disable_reason,
            telemetry_completeness="minimal",
        )
        assert row.mtp_enabled is True
        assert row.mtp_disable_reason == disable_reason
        assert row.mtp_execution_state != "successful_mtp"

    def test_blocked_row_json_round_trip(self) -> None:
        """Blocked row survives JSON round-trip with mtp_enabled=True and
        mtp_disable_reason intact."""
        row = BenchmarkRow(
            api_url="http://10.0.0.1:52415",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            mode="ar",
            mtp_enabled=True,
            mtp_execution_state="fail_closed_error",
            mtp_disable_reason="unsupported_depth",
            generation_tps=0.0,
            telemetry_completeness="minimal",
        )
        raw = row.model_dump_json()
        restored = BenchmarkRow.model_validate_json(raw)
        assert restored == row
        assert restored.mtp_enabled is True
        assert restored.mtp_disable_reason == "unsupported_depth"

    def test_blocked_row_as_blocked_evidence(self) -> None:
        """A blocked row with evidence_kind='blocked_evidence' records
        mtp_enabled=True to show that MTP was requested but unavailable."""
        row = BenchmarkRow(
            evidence_kind="blocked_evidence",
            row_status="blocked",
            mtp_enabled=True,
            mtp_execution_state="blocked_unavailable",
            mtp_disable_reason="cluster_unavailable",
            telemetry_completeness="unavailable",
        )
        assert row.evidence_kind == "blocked_evidence"
        assert row.row_status == "blocked"
        assert row.mtp_enabled is True
        assert row.mtp_disable_reason == "cluster_unavailable"
        assert row.generation_tps == 0.0  # no performance claim


# ---------------------------------------------------------------------------
# Comparability helpers (not on the model itself, but verify fields exist)
# ---------------------------------------------------------------------------


class TestComparabilityFields:
    def test_comparability_fields_present_on_ar_row(self) -> None:
        row = BenchmarkRow(
            benchmark_session_id="sess-compare",
            api_url="http://10.0.0.1:52415",
            model_id="test-model",
            endpoint="/bench/chat/completions",
            prompt_id="prompt-abc",
            temperature=0.0,
            max_tokens=256,
            mode="ar",
        )
        assert row.benchmark_session_id == "sess-compare"
        assert row.api_url == "http://10.0.0.1:52415"
        assert row.model_id == "test-model"
        assert row.endpoint == "/bench/chat/completions"
        assert row.prompt_id == "prompt-abc"
        assert row.temperature == 0.0
        assert row.max_tokens == 256
        assert row.mode == "ar"

    def test_node_topology_field_records_cluster_shape(self) -> None:
        row = BenchmarkRow(
            node_topology={"nodes": 2, "chips": ["m3ultra", "m3ultra"]},
        )
        assert row.node_topology == {"nodes": 2, "chips": ["m3ultra", "m3ultra"]}

    def test_payload_extra_captures_bench_context(self) -> None:
        row = BenchmarkRow(
            payload_extra={"concurrency": 4, "warmup_runs": 2},
        )
        assert row.payload_extra["concurrency"] == 4
