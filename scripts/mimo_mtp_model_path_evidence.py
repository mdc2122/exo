from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

JsonObject = dict[str, object]
ModelPathEvidenceVerdict = Literal["accepted", "rejected", "blocked"]
SCHEMA_VERSION = "mimo-mtp-model-path-evidence/v1"
ARTIFACT_KIND = "mimo_mtp_model_path_evidence"
_VALID_VERDICTS: tuple[str, ...] = ("accepted", "rejected", "blocked")


@dataclass(frozen=True, slots=True)
class RecordedModelPathEvidence:
    """Persisted benchmark-grade model-path evidence artifact."""

    path: Path
    payload: JsonObject


def record_model_path_evidence(
    *,
    output_path: Path,
    verdict: str,
    model_id: str,
    captured_signals: Mapping[str, object],
    rejection_reasons: Sequence[str],
    cluster_topology: Mapping[str, object],
    timestamp_utc: str | None = None,
) -> RecordedModelPathEvidence:
    """Persist a benchmark-grade model-path evidence artifact.

    This helper is intentionally side-effect-light and opt-in. It records the
    evidence needed to decide whether a benchmark row came from the expected exo
    cluster tensor-parallel path without changing generation defaults or enabling
    MTP execution.
    """

    normalized_verdict = _normalize_verdict(verdict)
    payload = build_model_path_evidence_payload(
        verdict=normalized_verdict,
        model_id=model_id,
        captured_signals=captured_signals,
        rejection_reasons=rejection_reasons,
        cluster_topology=cluster_topology,
        timestamp_utc=_timestamp_utc() if timestamp_utc is None else timestamp_utc,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return RecordedModelPathEvidence(path=output_path, payload=payload)


def build_model_path_evidence_payload(
    *,
    verdict: ModelPathEvidenceVerdict,
    model_id: str,
    captured_signals: Mapping[str, object],
    rejection_reasons: Sequence[str],
    cluster_topology: Mapping[str, object],
    timestamp_utc: str,
) -> JsonObject:
    """Build the deterministic JSON payload for model-path evidence."""

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "timestamp_utc": timestamp_utc,
        "verdict": verdict,
        "model_id": model_id,
        "captured_signals": dict(captured_signals),
        "rejection_reasons": list(rejection_reasons),
        "cluster_topology_summary": summarize_cluster_topology(cluster_topology),
    }


def summarize_cluster_topology(cluster_topology: Mapping[str, object]) -> JsonObject:
    """Summarize cluster topology enough for benchmark evidence review."""

    nodes = _extract_nodes(cluster_topology)
    edge_count = _extract_int(cluster_topology.get("edge_count"))
    if edge_count is None:
        topology = _optional_mapping(cluster_topology.get("topology"))
        if topology is not None:
            edge_count = _extract_int(topology.get("edge_count"))
    tensor_parallel_world_size = _extract_int(
        cluster_topology.get("tensor_parallel_world_size")
    )
    if tensor_parallel_world_size is None:
        tensor_parallel_world_size = _extract_int(cluster_topology.get("world_size"))

    return {
        "node_count": len(nodes),
        "nodes": nodes,
        "edge_count": edge_count,
        "tensor_parallel_world_size": tensor_parallel_world_size,
    }


def _normalize_verdict(verdict: str) -> ModelPathEvidenceVerdict:
    normalized = verdict.strip().lower()
    if normalized not in _VALID_VERDICTS:
        raise ValueError(
            "model-path evidence verdict must be accepted, rejected, or blocked"
        )
    return cast(ModelPathEvidenceVerdict, normalized)


def _timestamp_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _extract_nodes(cluster_topology: Mapping[str, object]) -> list[str]:
    direct_nodes = _nodes_from_value(cluster_topology.get("nodes"))
    if direct_nodes is not None:
        return direct_nodes

    topology = _optional_mapping(cluster_topology.get("topology"))
    if topology is not None:
        topology_nodes = _nodes_from_value(topology.get("nodes"))
        if topology_nodes is not None:
            return topology_nodes

    return []


def _nodes_from_value(value: object) -> list[str] | None:
    if isinstance(value, dict):
        mapping = cast(Mapping[object, object], value)
        return sorted(str(key) for key in mapping)
    if isinstance(value, list | tuple):
        sequence = cast(Sequence[object], value)
        return [str(item) for item in sequence]
    return None


def _extract_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _optional_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, dict):
        return cast(Mapping[str, object], value)
    return None
