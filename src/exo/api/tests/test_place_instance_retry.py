from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import anyio
import pytest
from fastapi import HTTPException

from exo.api.main import API
from exo.api.types.api import PlaceInstanceParams
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.memory import Memory
from exo.shared.types.worker.instances import InstanceMeta
from exo.shared.types.worker.shards import Sharding


class _FakeTopology:
    def __init__(self, node_ids: list[str]) -> None:
        self._node_ids = node_ids

    def list_nodes(self) -> list[str]:
        return self._node_ids


def _model_card() -> ModelCard:
    return ModelCard(
        model_id=ModelId("mlx-community/GLM-5.1-8b-crit-6b-exp"),
        storage_size=Memory.from_bytes(1024),
        n_layers=78,
        hidden_size=6144,
        supports_tensor=True,
        num_key_value_heads=64,
        tasks=[ModelTask.TextGeneration],
        trust_remote_code=False,
        is_custom=True,
    )


def _api() -> API:
    api = object.__new__(API)
    cast(Any, api).state = SimpleNamespace(
        topology=_FakeTopology(["node-a", "node-b"]),
        instances={},
        node_memory={"node-a": object(), "node-b": object()},
        node_network={"node-a": object(), "node-b": object()},
        downloads={},
    )
    return api


def test_place_instance_retries_until_placement_becomes_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    sent_commands: list[Any] = []
    sleep_calls: list[float] = []
    placement_calls = {"count": 0}

    async def fake_send(command: object) -> None:
        sent_commands.append(command)

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    async def fake_load(_: ModelId) -> ModelCard:
        return _model_card()

    def fake_get_instance_placements(*_: object, **__: object) -> dict[str, object]:
        placement_calls["count"] += 1
        if placement_calls["count"] < 3:
            raise ValueError("No cycles found with sufficient memory")
        return {}

    monkeypatch.setattr(api, "_send", fake_send)
    monkeypatch.setattr("exo.api.main.anyio.sleep", fake_sleep)
    monkeypatch.setattr("exo.api.main.get_instance_placements", fake_get_instance_placements)
    monkeypatch.setattr("exo.api.main.ModelCard.load", fake_load)

    payload = PlaceInstanceParams(
        model_id=ModelId("mlx-community/GLM-5.1-8b-crit-6b-exp"),
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        min_nodes=2,
    )

    response = anyio.run(api.place_instance, payload)

    assert response.message == "Command received."
    assert len(sent_commands) == 1
    assert placement_calls["count"] == 3
    assert sleep_calls == [1.0, 1.0]


def test_place_instance_raises_on_non_retryable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()

    async def fake_send(_: Any) -> None:
        raise AssertionError("send should not be reached on non-retryable placement error")

    async def fake_load(_: ModelId) -> ModelCard:
        return _model_card()

    def fake_get_instance_placements(*_: object, **__: object) -> dict[str, object]:
        raise ValueError("Requested Tensor sharding but this model does not support tensor parallelism")

    monkeypatch.setattr(api, "_send", fake_send)
    monkeypatch.setattr("exo.api.main.get_instance_placements", fake_get_instance_placements)
    monkeypatch.setattr("exo.api.main.ModelCard.load", fake_load)

    payload = PlaceInstanceParams(
        model_id=ModelId("mlx-community/GLM-5.1-8b-crit-6b-exp"),
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        min_nodes=2,
    )

    with pytest.raises(HTTPException) as exc_info:
        anyio.run(api.place_instance, payload)

    assert exc_info.value.status_code == 409
    assert "does not support tensor parallelism" in str(exc_info.value.detail)
