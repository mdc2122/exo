"""Canonical benchmark telemetry row model for AR/MTP evidence collection.

Every benchmark row — whether from live cluster execution or recorded as
blocked evidence — must conform to this schema.  The model is frozen and
strict so that once a row is created it cannot be silently mutated, and
coercion-based bugs are caught at validation time.

Default values represent a standard autoregressive (AR) row so that
omitting all MTP fields produces a valid AR evidence record.
"""

from typing import Literal

from pydantic import Field

from exo.utils.pydantic_ext import FrozenModel

# ---------------------------------------------------------------------------
# MTP execution state enum
# ---------------------------------------------------------------------------

MtpExecutionState = Literal[
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

# ---------------------------------------------------------------------------
# Row status enum
# ---------------------------------------------------------------------------

BenchmarkRowStatus = Literal[
    "valid",
    "blocked",
    "error",
]

# ---------------------------------------------------------------------------
# Evidence kind enum
# ---------------------------------------------------------------------------

EvidenceKind = Literal[
    "canonical_benchmark_row",
    "blocked_evidence",
]

# ---------------------------------------------------------------------------
# Telemetry completeness enum
# ---------------------------------------------------------------------------

TelemetryCompleteness = Literal[
    "full",
    "partial",
    "minimal",
    "unavailable",
]

# ---------------------------------------------------------------------------
# Canonical benchmark telemetry row
# ---------------------------------------------------------------------------

BENCHMARK_ROW_SCHEMA_VERSION: int = 1


class BenchmarkRow(FrozenModel):
    """A single canonical benchmark evidence row.

    Field semantics follow the Seed's ``canonical_benchmark_row`` schema.
    All MTP-related fields default to AR-equivalent values so that
    ``BenchmarkRow()`` with only the required identifying fields populated
    represents a valid autoregressive row (``mode='ar'``,
    ``mtp_enabled=False``, ``mtp_execution_state='disabled_default'``).
    """

    # -- Schema identity -----------------------------------------------------
    schema_version: int = BENCHMARK_ROW_SCHEMA_VERSION
    evidence_kind: EvidenceKind = "canonical_benchmark_row"
    row_status: BenchmarkRowStatus = "valid"

    # -- Session / comparability ---------------------------------------------
    benchmark_session_id: str = ""
    api_url: str = ""
    cluster_id: str = ""

    # -- Request identity ----------------------------------------------------
    endpoint: str = ""
    model_id: str = ""
    prompt_id: str = ""
    prompt_hash: str = ""
    temperature: float | None = None
    max_tokens: int | None = None

    # -- Mode ----------------------------------------------------------------
    mode: Literal["ar", "mtp"] = "ar"
    repeat_index: int = Field(default=0, ge=0)

    # -- Throughput / performance --------------------------------------------
    generation_tps: float = Field(default=0.0, ge=0.0)
    generation_tokens: int = Field(default=0, ge=0)
    prompt_tps: float = Field(default=0.0, ge=0.0)

    # -- Power ---------------------------------------------------------------
    power_usage: float | None = Field(default=None, ge=0.0)

    # -- Extra payload context -----------------------------------------------
    payload_extra: dict[str, object] = Field(default_factory=dict)

    # -- MTP telemetry -------------------------------------------------------
    mtp_enabled: bool = False
    mtp_depth: int | None = Field(default=None, ge=1)
    mtp_execution_state: MtpExecutionState = "disabled_default"
    mtp_disable_reason: str | None = None

    # -- Extended MTP telemetry (optional, populated when available) ---------
    mtp_sidecar_status: str | None = None
    mtp_fallback_reason: str | None = None
    requested_mtp_depth: int | None = Field(default=None, ge=1)
    attempted_depth_counts: dict[str, int] | None = None
    accepted_depth_counts: dict[str, int] | None = None
    acceptance_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    fallback_count: int | None = Field(default=None, ge=0)
    timing_breakdown_seconds: dict[str, float] | None = None

    # -- Telemetry quality ---------------------------------------------------
    telemetry_completeness: TelemetryCompleteness = "full"

    # -- Node topology (recorded when available) -----------------------------
    node_topology: dict[str, object] | None = None
