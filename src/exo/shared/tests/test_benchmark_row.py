"""Unit tests for the canonical BenchmarkRow model (AC Sub-AC 1.1).

Verifies:
- Construction of a valid AR-mode row with mtp_enabled=False,
  mtp_depth=0, and a populated mtp_disable_reason.
- Round-trip serialization/deserialization preserves all field values and
  types exactly (no silent coercion).
- Strict mode rejects coerced types (e.g. passing a string where int is
  expected).
- Extra fields are forbidden.
- Frozen mode rejects mutation.
- MTP-mode row construction and round-trip.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from pydantic import ValidationError

from exo.shared.types.benchmark import BenchmarkRow, MtpExecutionState

# ---------------------------------------------------------------------------
# AR-mode row factory (typed kwargs to satisfy reportAny)
# ---------------------------------------------------------------------------


def _make_ar_row() -> BenchmarkRow:
    return BenchmarkRow(
        schema_version="1.0.0",
        evidence_kind="cluster_live",
        row_status="valid",
        benchmark_session_id="session-20260607-ar",
        api_url="http://localhost:52415",
        cluster_id=None,
        endpoint="/bench/chat/completions",
        model_id="mlx-community/Qwen3-30B-A3B-4bit",
        prompt_id="prompt-001",
        prompt_hash=None,
        temperature=0.0,
        max_tokens=128,
        mode="ar",
        repeat_index=0,
        generation_tps=18.5,
        generation_tokens=100,
        prompt_tps=450.0,
        power_usage=None,
        payload_extra={},
        mtp_enabled=False,
        mtp_depth=0,
        mtp_execution_state="disabled_default",
        mtp_disable_reason="mtp not requested",
        telemetry_completeness=1.0,
    )


def _make_ar_row_from_base(
    *,
    mtp_execution_state: MtpExecutionState | None = "disabled_default",
    mtp_disable_reason: str | None = "mtp not requested",
    mode: str = "ar",
) -> BenchmarkRow:
    return BenchmarkRow(
        schema_version="1.0.0",
        evidence_kind="cluster_live",
        row_status="valid",
        benchmark_session_id="session-20260607-ar",
        api_url="http://localhost:52415",
        endpoint="/bench/chat/completions",
        model_id="mlx-community/Qwen3-30B-A3B-4bit",
        prompt_id="prompt-001",
        temperature=0.0,
        max_tokens=128,
        mode=mode,  # type: ignore[arg-type]  # Literal["ar","mtp"] from str
        repeat_index=0,
        generation_tps=18.5,
        generation_tokens=100,
        prompt_tps=450.0,
        payload_extra={},
        mtp_enabled=False,
        mtp_depth=0,
        mtp_execution_state=mtp_execution_state,
        mtp_disable_reason=mtp_disable_reason,
        telemetry_completeness=1.0,
    )


# ---------------------------------------------------------------------------
# Tests: AR-mode row construction
# ---------------------------------------------------------------------------


class TestArModeRowConstruction:
    def test_constructs_with_all_fields(self) -> None:
        row = _make_ar_row()
        assert row.schema_version == "1.0.0"
        assert row.evidence_kind == "cluster_live"
        assert row.row_status == "valid"
        assert row.benchmark_session_id == "session-20260607-ar"
        assert row.api_url == "http://localhost:52415"
        assert row.cluster_id is None
        assert row.endpoint == "/bench/chat/completions"
        assert row.model_id == "mlx-community/Qwen3-30B-A3B-4bit"
        assert row.prompt_id == "prompt-001"
        assert row.prompt_hash is None
        assert row.temperature == 0.0
        assert row.max_tokens == 128
        assert row.mode == "ar"
        assert row.repeat_index == 0
        assert row.generation_tps == 18.5
        assert row.generation_tokens == 100
        assert row.prompt_tps == 450.0
        assert row.power_usage is None
        assert row.payload_extra == {}
        assert row.mtp_enabled is False
        assert row.mtp_depth == 0
        assert row.mtp_execution_state == "disabled_default"
        assert row.mtp_disable_reason == "mtp not requested"
        assert row.telemetry_completeness == 1.0

    def test_mtp_fields_on_ar_row(self) -> None:
        """AR-mode rows must carry mtp_enabled=False, mtp_depth=0, and a
        populated mtp_disable_reason so consumers can distinguish 'no MTP
        attempted' from 'MTP field absent'."""
        row = _make_ar_row()
        assert row.mtp_enabled is False
        assert row.mtp_depth == 0
        assert row.mtp_execution_state == "disabled_default"
        assert isinstance(row.mtp_disable_reason, str)
        assert row.mtp_disable_reason == "mtp not requested"

    def test_mtp_execution_state_literal_allows_all_seed_states(self) -> None:
        """Verify that every MtpExecutionState value from the Seed ontology
        is accepted by the model."""
        all_states: list[MtpExecutionState] = [
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
        ]
        for state in all_states:
            row = _make_ar_row_from_base(mtp_execution_state=state)
            assert row.mtp_execution_state == state


# ---------------------------------------------------------------------------
# Tests: serialization / deserialization round-trip
# ---------------------------------------------------------------------------


class TestArModeRowRoundTrip:
    def test_model_dump_round_trip(self) -> None:
        """model_dump() → model_validate() preserves all values."""
        row = _make_ar_row()
        dumped = row.model_dump()
        restored = BenchmarkRow.model_validate(dumped)
        assert restored == row

    def test_model_dump_json_round_trip(self) -> None:
        """model_dump_json() → model_validate_json() preserves all values."""
        row = _make_ar_row()
        json_bytes = row.model_dump_json()
        restored = BenchmarkRow.model_validate_json(json_bytes)
        assert restored == row

    def test_json_module_round_trip(self) -> None:
        """json.loads(model_dump_json()) → model_validate() also works."""
        row = _make_ar_row()
        json_str = row.model_dump_json()
        parsed = cast(dict[str, Any], json.loads(json_str))
        restored = BenchmarkRow.model_validate(parsed)
        assert restored == row

    def test_serialized_types_are_exact(self) -> None:
        """Ensure no silent type coercion in serialization."""
        row = _make_ar_row()
        dumped = row.model_dump()
        # bool fields stay bool
        assert isinstance(dumped["mtp_enabled"], bool)
        # int fields stay int
        assert isinstance(dumped["max_tokens"], int)
        assert isinstance(dumped["mtp_depth"], int)
        assert isinstance(dumped["repeat_index"], int)
        assert isinstance(dumped["generation_tokens"], int)
        # float fields stay float
        assert isinstance(dumped["temperature"], float)
        assert isinstance(dumped["generation_tps"], float)
        assert isinstance(dumped["prompt_tps"], float)
        assert isinstance(dumped["telemetry_completeness"], float)
        # string fields stay string
        assert isinstance(dumped["schema_version"], str)
        assert isinstance(dumped["model_id"], str)
        assert isinstance(dumped["mode"], str)
        assert isinstance(dumped["mtp_disable_reason"], str)

    def test_json_round_trip_preserves_nulls(self) -> None:
        """Nullable fields survive JSON round-trip as None."""
        row = _make_ar_row()
        assert row.cluster_id is None
        assert row.prompt_hash is None
        assert row.power_usage is None
        json_str = row.model_dump_json()
        restored = BenchmarkRow.model_validate_json(json_str)
        assert restored.cluster_id is None
        assert restored.prompt_hash is None
        assert restored.power_usage is None


# ---------------------------------------------------------------------------
# Tests: strict mode rejection of coerced types
# ---------------------------------------------------------------------------


class TestStrictMode:
    def test_rejects_string_max_tokens(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkRow(
                benchmark_session_id="s",
                model_id="m",
                max_tokens="128",  # type: ignore[arg-type]
                mode="ar",
                mtp_enabled=False,
            )

    def test_rejects_string_mtp_enabled(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkRow(
                benchmark_session_id="s",
                model_id="m",
                max_tokens=1,
                mode="ar",
                mtp_enabled="false",  # type: ignore[arg-type]
            )

    def test_rejects_string_mtp_depth(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkRow(
                benchmark_session_id="s",
                model_id="m",
                max_tokens=1,
                mode="ar",
                mtp_enabled=False,
                mtp_depth="0",  # type: ignore[arg-type]
            )

    def test_rejects_float_mtp_depth(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkRow(
                benchmark_session_id="s",
                model_id="m",
                max_tokens=1,
                mode="ar",
                mtp_enabled=False,
                mtp_depth=0.0,  # type: ignore[arg-type]
            )

    def test_accepts_int_for_float_temperature(self) -> None:
        """Pydantic strict=True allows int values for float fields (JSON
        semantics).  Verify that int 0 round-trips as float 0.0."""
        row = BenchmarkRow(
            benchmark_session_id="s",
            model_id="m",
            max_tokens=1,
            mode="ar",
            mtp_enabled=False,
            temperature=0,
        )
        assert row.temperature == 0.0

    def test_rejects_bool_for_float_temperature(self) -> None:
        """Bool should not be accepted for float fields."""
        with pytest.raises(ValidationError):
            BenchmarkRow(
                benchmark_session_id="s",
                model_id="m",
                max_tokens=1,
                mode="ar",
                mtp_enabled=False,
                temperature=True,
            )

    def test_rejects_invalid_mode_literal(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkRow(
                benchmark_session_id="s",
                model_id="m",
                max_tokens=1,
                mode="speculative",  # type: ignore[arg-type]
                mtp_enabled=False,
            )

    def test_rejects_invalid_execution_state_literal(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkRow(
                benchmark_session_id="s",
                model_id="m",
                max_tokens=1,
                mode="ar",
                mtp_enabled=False,
                mtp_execution_state="unknown_state",  # type: ignore[arg-type]
            )

    def test_rejects_invalid_evidence_kind_literal(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkRow(
                benchmark_session_id="s",
                model_id="m",
                max_tokens=1,
                mode="ar",
                mtp_enabled=False,
                evidence_kind="synthetic",  # type: ignore[arg-type]
            )

    def test_rejects_invalid_row_status_literal(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkRow(
                benchmark_session_id="s",
                model_id="m",
                max_tokens=1,
                mode="ar",
                mtp_enabled=False,
                row_status="pending",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# Tests: extra fields forbidden
# ---------------------------------------------------------------------------


class TestExtraForbidden:
    def test_rejects_unknown_field_via_model_validate(self) -> None:
        """Extra fields are rejected via model_validate (dict input)."""
        row = _make_ar_row()
        dumped = row.model_dump()
        dumped["unexpected_field"] = True
        with pytest.raises(ValidationError):
            BenchmarkRow.model_validate(dumped)


# ---------------------------------------------------------------------------
# Tests: frozen mode
# ---------------------------------------------------------------------------


class TestFrozenMode:
    def test_rejects_field_mutation(self) -> None:
        row = _make_ar_row()
        with pytest.raises(ValidationError):
            row.generation_tps = 99.0

    def test_rejects_mtp_depth_mutation(self) -> None:
        row = _make_ar_row()
        with pytest.raises(ValidationError):
            row.mtp_depth = 3


# ---------------------------------------------------------------------------
# Tests: MTP-mode row
# ---------------------------------------------------------------------------


class TestMtpModeRowConstruction:
    def _make_mtp_row(
        self,
        *,
        mtp_execution_state: MtpExecutionState = "successful_mtp",
        mtp_disable_reason: str | None = None,
        mode: str = "mtp",
    ) -> BenchmarkRow:
        return BenchmarkRow(
            benchmark_session_id="session-20260607-mtp",
            api_url="http://localhost:52415",
            model_id="mlx-community/Qwen3-30B-A3B-4bit",
            prompt_id="prompt-001",
            temperature=0.0,
            max_tokens=128,
            mode=mode,  # type: ignore[arg-type]
            repeat_index=0,
            generation_tps=22.0,
            generation_tokens=100,
            prompt_tps=450.0,
            mtp_enabled=True,
            mtp_depth=3,
            mtp_execution_state=mtp_execution_state,
            mtp_disable_reason=mtp_disable_reason,
            telemetry_completeness=1.0,
        )

    def test_constructs_mtp_row(self) -> None:
        row = self._make_mtp_row()
        assert row.mode == "mtp"
        assert row.mtp_enabled is True
        assert row.mtp_depth == 3
        assert row.mtp_execution_state == "successful_mtp"
        assert row.mtp_disable_reason is None

    def test_mtp_row_round_trip(self) -> None:
        row = self._make_mtp_row()
        json_str = row.model_dump_json()
        restored = BenchmarkRow.model_validate_json(json_str)
        assert restored == row

    def test_fail_closed_mtp_row(self) -> None:
        row = self._make_mtp_row(
            mtp_execution_state="fail_closed_error",
            mtp_disable_reason="sidecar not found",
        )
        assert row.mtp_execution_state == "fail_closed_error"
        assert row.mtp_disable_reason == "sidecar not found"
        restored = BenchmarkRow.model_validate_json(row.model_dump_json())
        assert restored == row

    def test_fail_open_mtp_row(self) -> None:
        """Fail-open fallback actually ran AR, so mode must be 'ar', not
        'mtp' — this prevents silent AR-as-MTP behavior."""
        row = self._make_mtp_row(
            mtp_execution_state="fail_open_fallback",
            mtp_disable_reason="sidecar load failed; fell back to AR",
            mode="ar",
        )
        assert row.mtp_execution_state == "fail_open_fallback"
        assert row.mode == "ar"  # ran AR, not successful MTP
        restored = BenchmarkRow.model_validate_json(row.model_dump_json())
        assert restored == row


# ---------------------------------------------------------------------------
# Tests: blocked evidence row
# ---------------------------------------------------------------------------


class TestBlockedEvidenceRow:
    def test_blocked_row_with_minimal_fields(self) -> None:
        row = BenchmarkRow(
            benchmark_session_id="session-blocked",
            model_id="mlx-community/Qwen3-30B-A3B-4bit",
            max_tokens=128,
            mode="ar",
            mtp_enabled=False,
            mtp_execution_state="blocked_unavailable",
            mtp_disable_reason="cluster unavailable",
            evidence_kind="blocked_evidence",
            row_status="blocked",
            telemetry_completeness=0.0,
        )
        assert row.evidence_kind == "blocked_evidence"
        assert row.row_status == "blocked"
        assert row.mtp_execution_state == "blocked_unavailable"
        assert row.generation_tps is None
        assert row.telemetry_completeness == 0.0
        # Round-trip
        restored = BenchmarkRow.model_validate_json(row.model_dump_json())
        assert restored == row


# ---------------------------------------------------------------------------
# Tests: validation constraints
# ---------------------------------------------------------------------------


class TestValidationConstraints:
    def test_max_tokens_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkRow(
                benchmark_session_id="s",
                model_id="m",
                max_tokens=0,
                mode="ar",
                mtp_enabled=False,
            )

    def test_repeat_index_must_be_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkRow(
                benchmark_session_id="s",
                model_id="m",
                max_tokens=1,
                mode="ar",
                mtp_enabled=False,
                repeat_index=-1,
            )

    def test_mtp_depth_must_be_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkRow(
                benchmark_session_id="s",
                model_id="m",
                max_tokens=1,
                mode="ar",
                mtp_enabled=False,
                mtp_depth=-1,
            )

    def test_telemetry_completeness_must_be_between_0_and_1(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkRow(
                benchmark_session_id="s",
                model_id="m",
                max_tokens=1,
                mode="ar",
                mtp_enabled=False,
                telemetry_completeness=1.5,
            )

    def test_telemetry_completeness_zero_is_valid(self) -> None:
        row = BenchmarkRow(
            benchmark_session_id="s",
            model_id="m",
            max_tokens=1,
            mode="ar",
            mtp_enabled=False,
            telemetry_completeness=0.0,
        )
        assert row.telemetry_completeness == 0.0

    def test_telemetry_completeness_one_is_valid(self) -> None:
        row = BenchmarkRow(
            benchmark_session_id="s",
            model_id="m",
            max_tokens=1,
            mode="ar",
            mtp_enabled=False,
            telemetry_completeness=1.0,
        )
        assert row.telemetry_completeness == 1.0


# ---------------------------------------------------------------------------
# Tests: default values
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_defaults_match_seed_schema(self) -> None:
        row = BenchmarkRow(
            benchmark_session_id="session-defaults",
            model_id="test-model",
            max_tokens=64,
            mode="ar",
            mtp_enabled=False,
        )
        assert row.schema_version == "1.0.0"
        assert row.evidence_kind == "cluster_live"
        assert row.row_status == "valid"
        assert row.endpoint == "/bench/chat/completions"
        assert row.repeat_index == 0
        assert row.mtp_depth == 0
        assert row.mtp_execution_state is None
        assert row.mtp_disable_reason is None
        assert row.telemetry_completeness == 1.0
        assert row.api_url is None
        assert row.cluster_id is None
        assert row.prompt_id is None
        assert row.prompt_hash is None
        assert row.generation_tps is None
        assert row.generation_tokens is None
        assert row.prompt_tps is None
        assert row.power_usage is None
        assert row.payload_extra == {}


# ---------------------------------------------------------------------------
# Tests: blocked row where MTP was requested but could not proceed
# (AC Sub-AC 1.2: mtp_enabled=True + mtp_disable_reason is set)
# ---------------------------------------------------------------------------


class TestMtpRequestedButBlockedRow:
    """Verify that a row where mtp_enabled=True but mtp_disable_reason is also
    set — indicating MTP was *requested* but *blocked* — is accepted by the
    model and that the fields faithfully reflect the blocked condition.

    This is the critical "no silent AR-as-MTP" evidence pattern: the caller
    intended MTP, MTP could not proceed, and the row must record both the
    intent (mtp_enabled=True) and the outcome (mtp_disable_reason populated,
    mtp_execution_state reflecting the blocking reason).
    """

    def test_blocked_row_with_mtp_enabled_and_disable_reason(self) -> None:
        """Core AC test: mtp_enabled=True + mtp_disable_reason set is accepted."""
        row = BenchmarkRow(
            benchmark_session_id="session-blocked-mtp-001",
            api_url="http://localhost:52415",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            endpoint="/bench/chat/completions",
            prompt_id="prompt-42",
            temperature=0.0,
            max_tokens=128,
            mode="ar",
            repeat_index=0,
            mtp_enabled=True,
            mtp_depth=0,
            mtp_execution_state="fail_closed_error",
            mtp_disable_reason="missing_sidecar",
            generation_tps=None,
            generation_tokens=None,
            evidence_kind="cluster_live",
            row_status="valid",
            telemetry_completeness=0.3,
        )
        assert row.mtp_enabled is True
        assert row.mtp_disable_reason == "missing_sidecar"
        assert row.mtp_execution_state == "fail_closed_error"
        assert row.mode == "ar"  # actual execution was AR, not MTP

    @pytest.mark.parametrize(
        "execution_state, disable_reason",
        [
            ("fail_closed_error", "unsupported_model"),
            ("fail_closed_error", "missing_sidecar"),
            ("fail_closed_error", "unsupported_depth"),
            ("fail_open_fallback", "sidecar_load_failed"),
            ("fail_open_fallback", "native_runtime_disabled"),
            ("unwired_execution", "mimo_mtp_distributed_generator_unwired"),
            ("runtime_error", "speculative_loop_exception"),
            ("blocked_unavailable", "cluster_unavailable"),
        ],
    )
    def test_blocked_row_accepts_all_mtp_blocking_states(
        self,
        execution_state: str,
        disable_reason: str,
    ) -> None:
        """Every MTP blocking state with mtp_enabled=True and a disable_reason
        is accepted and faithfully records both the intent and the block."""
        row = BenchmarkRow(
            benchmark_session_id="session-blocked-param",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            max_tokens=128,
            mode="ar",
            mtp_enabled=True,
            mtp_depth=0,
            mtp_execution_state=execution_state,  # type: ignore[arg-type]
            mtp_disable_reason=disable_reason,
            telemetry_completeness=0.2,
        )
        assert row.mtp_enabled is True
        assert row.mtp_disable_reason == disable_reason
        assert row.mtp_execution_state == execution_state
        # A blocked row must not claim successful MTP
        assert row.mtp_execution_state != "successful_mtp"

    def test_blocked_row_round_trips_through_json(self) -> None:
        """Blocked row survives JSON serialization/deserialization intact,
        preserving both mtp_enabled=True and the disable_reason."""
        row = BenchmarkRow(
            benchmark_session_id="session-blocked-rt",
            api_url="http://10.0.0.1:52415",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            endpoint="/bench/chat/completions",
            prompt_id="prompt-abc",
            temperature=0.0,
            max_tokens=256,
            mode="ar",
            mtp_enabled=True,
            mtp_depth=0,
            mtp_execution_state="fail_open_fallback",
            mtp_disable_reason="sidecar_cache_expired",
            generation_tps=18.0,
            generation_tokens=200,
            telemetry_completeness=0.5,
        )
        json_str = row.model_dump_json()
        restored = BenchmarkRow.model_validate_json(json_str)
        assert restored == row
        assert restored.mtp_enabled is True
        assert restored.mtp_disable_reason == "sidecar_cache_expired"
        assert restored.mtp_execution_state == "fail_open_fallback"
        assert restored.generation_tps == 18.0

    def test_blocked_row_as_blocked_evidence_kind(self) -> None:
        """A blocked row where MTP was requested but the cluster was
        unavailable uses evidence_kind='blocked_evidence' and
        row_status='blocked' with mtp_enabled=True."""
        row = BenchmarkRow(
            benchmark_session_id="session-blocked-evidence",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            max_tokens=128,
            mode="ar",
            mtp_enabled=True,
            mtp_depth=0,
            mtp_execution_state="blocked_unavailable",
            mtp_disable_reason="cluster_unavailable",
            evidence_kind="blocked_evidence",
            row_status="blocked",
            telemetry_completeness=0.0,
        )
        assert row.evidence_kind == "blocked_evidence"
        assert row.row_status == "blocked"
        assert row.mtp_enabled is True
        assert row.mtp_disable_reason == "cluster_unavailable"
        assert row.generation_tps is None  # no performance claim

    def test_blocked_row_preserves_intent_not_achievement(self) -> None:
        """The blocked row pattern must never allow mtp_enabled=True to be
        confused with successful MTP: mode='ar' and a non-success
        execution_state must be present alongside mtp_enabled=True."""
        row = BenchmarkRow(
            benchmark_session_id="session-intent-check",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            max_tokens=128,
            mode="ar",
            mtp_enabled=True,
            mtp_depth=0,
            mtp_execution_state="unwired_execution",
            mtp_disable_reason="mimo_mtp_distributed_generator_unwired",
            telemetry_completeness=0.1,
        )
        # Intent was MTP
        assert row.mtp_enabled is True
        # But execution was not MTP
        assert row.mode == "ar"
        assert row.mtp_execution_state != "successful_mtp"
        assert row.mtp_disable_reason is not None
        # generation_tps is absent: no MTP throughput claim
        assert row.generation_tps is None
