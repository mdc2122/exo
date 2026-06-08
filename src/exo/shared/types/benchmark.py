"""Canonical benchmark row model for AR/MTP evidence contracts.

Every benchmark evidence row produced by exo bench or telemetry must conform
to this schema so that downstream speedup/gate analyzers can compare rows
without ambiguity.  The model is frozen and strict per project conventions;
AR-mode rows carry mtp_enabled=False and a populated mtp_disable_reason, while
MTP-mode rows carry the MTP execution state and depth.
"""

from typing import Literal, final

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# MTP execution state – allowed labels per the Seed ontology
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
# Row-level status
# ---------------------------------------------------------------------------

BenchmarkRowStatus = Literal[
    "valid",
    "blocked",
    "ambiguous",
    "insufficient_data",
]

# ---------------------------------------------------------------------------
# Evidence kind discriminator
# ---------------------------------------------------------------------------

BenchmarkEvidenceKind = Literal[
    "cluster_live",
    "blocked_evidence",
]

# ---------------------------------------------------------------------------
# Canonical benchmark row
# ---------------------------------------------------------------------------


@final
class BenchmarkRow(BaseModel):
    """Canonical JSONL-serializable benchmark evidence row.

    Fields follow the Seed's ``canonical_benchmark_row`` schema exactly.
    ``frozen=True`` and ``strict=True`` enforce immutability and reject
    coerced types, matching project-wide Pydantic conventions.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    # -- identification -----------------------------------------------------
    schema_version: str = "1.0.0"
    evidence_kind: BenchmarkEvidenceKind = "cluster_live"
    row_status: BenchmarkRowStatus = "valid"
    benchmark_session_id: str

    # -- cluster / endpoint -------------------------------------------------
    api_url: str | None = None
    cluster_id: str | None = None
    endpoint: str = "/bench/chat/completions"
    model_id: str

    # -- prompt identity ----------------------------------------------------
    prompt_id: str | None = None
    prompt_hash: str | None = None

    # -- generation parameters ----------------------------------------------
    temperature: float = 0.0
    max_tokens: int = Field(ge=1)

    # -- mode ---------------------------------------------------------------
    mode: Literal["ar", "mtp"]

    # -- repeat identity ----------------------------------------------------
    repeat_index: int = Field(ge=0, default=0)

    # -- throughput / tokens ------------------------------------------------
    generation_tps: float | None = None
    generation_tokens: int | None = None
    prompt_tps: float | None = None

    # -- power (nullable – not always available) ----------------------------
    power_usage: float | None = None

    # -- opaque extras ------------------------------------------------------
    payload_extra: dict[str, object] = Field(default_factory=dict)

    # -- MTP telemetry (required on every row; AR rows populate the
    #    disable-reason so a consumer can distinguish "no MTP attempted"
    #    from "MTP field absent") -------------------------------------------
    mtp_enabled: bool
    mtp_depth: int = Field(ge=0, default=0)
    mtp_execution_state: MtpExecutionState | None = None
    mtp_disable_reason: str | None = None

    # -- telemetry quality --------------------------------------------------
    telemetry_completeness: float = Field(ge=0.0, le=1.0, default=1.0)
