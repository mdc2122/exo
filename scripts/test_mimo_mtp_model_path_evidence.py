from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from scripts import mimo_mtp_model_path_evidence as evidence


def _read_json_object(path: Path) -> dict[str, object]:
    decoded = cast(object, json.loads(path.read_text(encoding="utf-8")))
    assert isinstance(decoded, dict)
    return cast(dict[str, object], decoded)


def test_record_model_path_evidence_persists_benchmark_grade_artifact(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "model-path-evidence.json"

    recorded = evidence.record_model_path_evidence(
        output_path=artifact_path,
        verdict="accepted",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        captured_signals={
            "cluster_path": "exo_api_bench_chat_completions",
            "mode": "ar",
            "generation_tps": 24.25,
            "live_model_path_validation": {
                "allowed": True,
                "live_model_path": "exo_cluster_tensor_parallel",
            },
        },
        rejection_reasons=(),
        cluster_topology={
            "nodes": ["studio-1", "studio-2"],
            "edge_count": 1,
            "tensor_parallel_world_size": 2,
        },
        timestamp_utc="2026-06-07T01:53:00Z",
    )

    assert recorded.path == artifact_path
    assert recorded.payload == {
        "schema_version": "mimo-mtp-model-path-evidence/v1",
        "artifact_kind": "mimo_mtp_model_path_evidence",
        "timestamp_utc": "2026-06-07T01:53:00Z",
        "verdict": "accepted",
        "model_id": "kernelpool/MiMo-V2.5-Pro-6bit",
        "captured_signals": {
            "cluster_path": "exo_api_bench_chat_completions",
            "mode": "ar",
            "generation_tps": 24.25,
            "live_model_path_validation": {
                "allowed": True,
                "live_model_path": "exo_cluster_tensor_parallel",
            },
        },
        "rejection_reasons": [],
        "cluster_topology_summary": {
            "node_count": 2,
            "nodes": ["studio-1", "studio-2"],
            "edge_count": 1,
            "tensor_parallel_world_size": 2,
        },
    }
    assert _read_json_object(artifact_path) == recorded.payload
    assert artifact_path.read_text(encoding="utf-8").endswith("\n")


def test_record_model_path_evidence_rejects_invalid_verdict(tmp_path: Path) -> None:
    try:
        evidence.record_model_path_evidence(
            output_path=tmp_path / "bad.json",
            verdict="maybe",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            captured_signals={},
            rejection_reasons=("response did not include execution_path telemetry",),
            cluster_topology={},
            timestamp_utc="2026-06-07T01:53:00Z",
        )
    except ValueError as exc:
        assert (
            str(exc)
            == "model-path evidence verdict must be accepted, rejected, or blocked"
        )
    else:
        raise AssertionError("expected invalid verdict rejection")


def test_record_model_path_evidence_derives_blocked_topology_summary(
    tmp_path: Path,
) -> None:
    recorded = evidence.record_model_path_evidence(
        output_path=tmp_path / "blocked.json",
        verdict="blocked",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        captured_signals={"cluster_api": "unavailable"},
        rejection_reasons=("cluster API unavailable",),
        cluster_topology={"topology": {"nodes": {"studio-1": {}, "studio-2": {}}}},
        timestamp_utc="2026-06-07T01:53:00Z",
    )

    assert recorded.payload["cluster_topology_summary"] == {
        "node_count": 2,
        "nodes": ["studio-1", "studio-2"],
        "edge_count": None,
        "tensor_parallel_world_size": None,
    }
    assert recorded.payload["rejection_reasons"] == ["cluster API unavailable"]
