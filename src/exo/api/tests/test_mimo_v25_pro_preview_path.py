from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, cast

import anyio
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from exo.api.main import API
from exo.api.types.api import CreateInstanceParams, PlacementPreviewResponse
from exo.shared.models.model_cards import (
    MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
    ModelId,
)
from exo.shared.types.profiling import MemoryUsage
from exo.shared.types.state import State
from exo.shared.types.worker.shards import PipelineShardMetadata, TensorShardMetadata


def _api_with_empty_topology() -> API:
    api = object.__new__(API)
    api.state = State()
    return api


def _preview_ranges(preview_response: PlacementPreviewResponse) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for preview in preview_response.previews:
        if preview.instance is None:
            continue
        for shard in preview.instance.shard_assignments.runner_to_shard.values():
            if isinstance(shard, PipelineShardMetadata):
                ranges.append((shard.start_layer, shard.end_layer))
    return sorted(set(ranges))


def test_mimo_v25_pro_empty_live_topology_preview_exposes_two_node_candidate() -> None:
    api = _api_with_empty_topology()

    response = anyio.run(
        api.get_placement_previews, MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID
    )

    assert _preview_ranges(response) == [(0, 35), (35, 70)]
    assert any(
        preview.instance is not None
        and preview.sharding.value == "Tensor"
        for preview in response.previews
    )


def test_non_mimo_empty_live_topology_preview_stays_empty() -> None:
    api = _api_with_empty_topology()

    response = anyio.run(
        api.get_placement_previews,
        ModelId("mlx-community/Llama-3.3-70B-Instruct-4bit"),
    )

    assert response.previews == []


def test_tensor_create_memory_evidence_uses_per_rank_storage() -> None:
    api = _api_with_empty_topology()
    response = anyio.run(
        api.get_placement_previews, MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID
    )
    preview = next(
        preview
        for preview in response.previews
        if preview.instance is not None
        and preview.sharding.value == "Tensor"
    )
    assert preview.instance is not None
    selected_nodes = list(preview.instance.shard_assignments.node_to_runner.keys())
    tensor_shards = list(preview.instance.shard_assignments.runner_to_shard.values())
    assert all(isinstance(shard, TensorShardMetadata) for shard in tensor_shards)
    assert {shard.start_layer for shard in tensor_shards} == {0}
    assert {shard.end_layer for shard in tensor_shards} == {70}

    now = datetime.now(timezone.utc)
    api.state = api.state.model_copy(
        update={
            "last_seen": {node_id: now for node_id in selected_nodes},
            "node_memory": {
                node_id: MemoryUsage.from_bytes(
                    ram_total=512 * 1024**3,
                    ram_available=455 * 1024**3,
                    swap_total=0,
                    swap_available=0,
                )
                for node_id in selected_nodes
            },
        }
    )
    sent_commands: list[object] = []

    async def fake_send(command: object) -> None:
        sent_commands.append(command)

    cast(Any, api)._send = fake_send

    create_response = anyio.run(
        api.create_instance, CreateInstanceParams(instance=preview.instance)
    )

    assert create_response.message == "Command received."
    assert create_response.model_card.model_id == MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID
    assert len(sent_commands) == 1


def test_create_instance_failure_reports_selected_worker_memory_evidence() -> None:
    api = _api_with_empty_topology()
    response = anyio.run(
        api.get_placement_previews, MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID
    )
    preview = next(
        preview
        for preview in response.previews
        if preview.instance is not None
        and preview.sharding.value == "Pipeline"
        and preview.instance_meta.value == "MlxRing"
    )
    assert preview.instance is not None
    selected_nodes = list(preview.instance.shard_assignments.node_to_runner.keys())
    now = datetime.now(timezone.utc)
    api.state = api.state.model_copy(
        update={
            "last_seen": {selected_nodes[0]: now},
            "node_memory": {
                selected_nodes[0]: MemoryUsage.from_bytes(
                    ram_total=512 * 1024**3,
                    ram_available=0,
                    swap_total=0,
                    swap_available=0,
                ),
            },
        }
    )

    async def fail_send(_: object) -> None:
        raise AssertionError("unsafe create must not be sent")

    cast(Any, api)._send = fail_send

    with pytest.raises(HTTPException) as exc_info:
        anyio.run(api.create_instance, CreateInstanceParams(instance=preview.instance))

    assert exc_info.value.status_code == 400
    detail = cast(dict[str, Any], cast(object, exc_info.value.detail))
    assert detail["available_bytes"] == 0
    assert detail["selected_node_ids"] == [str(node_id) for node_id in selected_nodes]
    selected_worker_memory = cast(
        list[dict[str, Any]], detail["selected_worker_memory"]
    )
    statuses = {
        worker["node_id"]: worker["status"]
        for worker in selected_worker_memory
    }
    assert statuses[str(selected_nodes[0])] == "non_positive"
    assert statuses[str(selected_nodes[1])] == "missing"


def test_create_instance_failure_reports_stale_selected_worker_memory() -> None:
    api = _api_with_empty_topology()
    response = anyio.run(
        api.get_placement_previews, MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID
    )
    preview = next(
        preview
        for preview in response.previews
        if preview.instance is not None
        and preview.sharding.value == "Pipeline"
        and preview.instance_meta.value == "MlxRing"
    )
    assert preview.instance is not None
    selected_nodes = list(preview.instance.shard_assignments.node_to_runner.keys())
    stale_time = datetime.now(timezone.utc) - timedelta(seconds=300)
    api.state = api.state.model_copy(
        update={
            "last_seen": {node_id: stale_time for node_id in selected_nodes},
            "node_memory": {
                node_id: MemoryUsage.from_bytes(
                    ram_total=512 * 1024**3,
                    ram_available=512 * 1024**3,
                    swap_total=0,
                    swap_available=0,
                )
                for node_id in selected_nodes
            },
        }
    )

    async def fail_send(_: object) -> None:
        raise AssertionError("stale create must not be sent")

    cast(Any, api)._send = fail_send

    with pytest.raises(HTTPException) as exc_info:
        anyio.run(api.create_instance, CreateInstanceParams(instance=preview.instance))

    detail = cast(dict[str, Any], cast(object, exc_info.value.detail))
    selected_worker_memory = cast(
        list[dict[str, Any]], detail["selected_worker_memory"]
    )
    statuses = {
        worker["node_id"]: worker["status"]
        for worker in selected_worker_memory
    }
    assert statuses == {str(node_id): "stale" for node_id in selected_nodes}


def test_http_exception_handler_preserves_structured_memory_detail() -> None:
    api = _api_with_empty_topology()
    detail = {
        "error": "Insufficient live selected-worker memory to create instance",
        "selected_node_ids": ["node-a"],
        "selected_worker_memory": [
            {
                "node_id": "node-a",
                "required_bytes": 1,
                "required_gb": 1 / 1024**3,
                "ram_available_bytes": 0,
                "ram_available_gb": 0,
                "ram_total_bytes": None,
                "last_seen": None,
                "status": "missing",
                "reason": "selected worker is absent from state.node_memory",
            }
        ],
    }

    response = anyio.run(
        api.http_exception_handler,
        Request({"type": "http", "method": "POST", "path": "/instance"}),
        HTTPException(status_code=400, detail=detail),
    )

    assert response.status_code == 400
    assert json.loads(bytes(response.body)) == {"error": detail}
