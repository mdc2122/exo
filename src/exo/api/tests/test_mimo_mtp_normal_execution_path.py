from __future__ import annotations

import json
import sys
import types
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, cast

from fastapi.responses import JSONResponse
from loguru import logger

from exo.api.types import ChatCompletionMessage, ChatCompletionRequest
from exo.shared.models.model_cards import MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.commands import TextGeneration
from exo.shared.types.common import CommandId
from exo.shared.types.state import State

if TYPE_CHECKING:
    from exo.api.main import API


def _install_bootstrap_logger_stub() -> None:
    """Avoid importing MLX patches while this API-boundary test imports API."""
    bootstrap_stub = types.ModuleType("exo.worker.runner.bootstrap")
    cast(Any, bootstrap_stub).logger = logger
    sys.modules.setdefault("exo.worker.runner.bootstrap", bootstrap_stub)


def _api_with_empty_topology() -> "API":
    _install_bootstrap_logger_stub()
    from exo.api.main import API

    api = object.__new__(API)
    api.state = State()
    return api


async def test_normal_chat_completion_without_enabled_mtp_uses_non_mtp_text_generation_path() -> (
    None
):
    api = _api_with_empty_topology()
    preview_response = await api.get_placement_previews(
        MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID
    )
    preview = next(
        preview
        for preview in preview_response.previews
        if preview.instance is not None and preview.sharding.value == "Tensor"
    )
    assert preview.instance is not None
    api.state = api.state.model_copy(
        update={"instances": {preview.instance.instance_id: preview.instance}}
    )

    async def stream(_command_id: CommandId) -> AsyncGenerator[TokenChunk, None]:
        yield TokenChunk(
            model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
            text="ok",
            token_id=1,
            usage=None,
            finish_reason="stop",
            stats=None,
        )

    sent_commands: list[object] = []

    async def fake_send(command: object) -> None:
        sent_commands.append(command)

    object.__setattr__(api, "_token_chunk_stream", stream)
    object.__setattr__(api, "_send", fake_send)

    response = await api.chat_completions(
        ChatCompletionRequest(
            model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
            messages=[
                ChatCompletionMessage(role="user", content="Summarize MiMo Pro.")
            ],
            max_tokens=1,
        )
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 200
    response_body = (
        response.body.tobytes()
        if isinstance(response.body, memoryview)
        else response.body
    )
    assert json.loads(response_body)["choices"][0]["message"]["content"] == "ok"
    assert len(sent_commands) == 1
    command = sent_commands[0]
    assert isinstance(command, TextGeneration)
    assert command.task_params.bench is False
    assert command.task_params.stream is False
    assert command.task_params.input[0].content == "Summarize MiMo Pro."
    assert command.task_params.mimo_mtp_fastpath is None
    task_param_payload = command.task_params.model_dump()
    assert task_param_payload.get("mimo_mtp_fastpath") is None
    assert "mimo_mtp_depth" not in task_param_payload
    assert "mimo_mtp_sidecar_path" not in task_param_payload
    assert "mimo_mtp_fail_closed" not in task_param_payload
