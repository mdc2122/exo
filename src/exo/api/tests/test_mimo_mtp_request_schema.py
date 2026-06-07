from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
from fastapi import HTTPException

from exo.api.adapters.chat_completions import chat_request_to_text_generation
from exo.api.types import ChatCompletionMessage, ChatCompletionRequest
from exo.shared.models.model_cards import MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID
from exo.shared.types.common import ModelId

MIMO_MTP_REQUEST_FIELD_DEFAULTS: Mapping[str, object] = {
    "mimo_mtp_fastpath": False,
    "mimo_mtp_depth": None,
    "mimo_mtp_sidecar_path": None,
    "mimo_mtp_fail_closed": True,
}


def _json_object(value: object) -> Mapping[str, object]:
    assert isinstance(value, dict)
    return cast(Mapping[str, object], value)


def test_mimo_mtp_extension_schema_defaults_match_disabled_request_defaults() -> None:
    request = ChatCompletionRequest(
        model=ModelId("kernelpool/MiMo-V2.5-Pro-6bit"),
        messages=[ChatCompletionMessage(role="user", content="Hello")],
    )
    schema = _json_object(ChatCompletionRequest.model_json_schema())
    properties = _json_object(schema["properties"])

    for field_name, expected_default in MIMO_MTP_REQUEST_FIELD_DEFAULTS.items():
        property_schema = _json_object(properties[field_name])

        assert getattr(request, field_name) == expected_default
        assert property_schema["default"] == expected_default

    assert request.mimo_mtp_fastpath is False


def test_mimo_mtp_extension_fields_are_disabled_when_omitted() -> None:
    request = ChatCompletionRequest(
        model=ModelId("kernelpool/MiMo-V2.5-Pro-6bit"),
        messages=[ChatCompletionMessage(role="user", content="Hello")],
    )

    assert request.mimo_mtp_fastpath is False
    assert request.mimo_mtp_depth is None
    assert request.mimo_mtp_sidecar_path is None
    assert request.mimo_mtp_fail_closed is True


def test_mimo_mtp_extension_fields_parse_explicit_request_values() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "model": "kernelpool/MiMo-V2.5-Pro-6bit",
            "messages": [{"role": "user", "content": "Hello"}],
            "mimo_mtp_fastpath": True,
            "mimo_mtp_depth": 2,
            "mimo_mtp_sidecar_path": "/models/mimo-v25-pro-mtp-sidecar",
            "mimo_mtp_fail_closed": False,
        }
    )

    assert request.mimo_mtp_fastpath is True
    assert request.mimo_mtp_depth == 2
    assert request.mimo_mtp_sidecar_path == "/models/mimo-v25-pro-mtp-sidecar"
    assert request.mimo_mtp_fail_closed is False


def test_mimo_mtp_extension_fields_are_marked_experimental_in_schema() -> None:
    schema = _json_object(ChatCompletionRequest.model_json_schema())
    properties = _json_object(schema["properties"])

    for field_name in MIMO_MTP_REQUEST_FIELD_DEFAULTS:
        property_schema = _json_object(properties[field_name])
        description = property_schema["description"]

        assert property_schema["x-exo-extension"] is True
        assert isinstance(description, str)
        assert "experimental" in description.lower()


@pytest.mark.asyncio
async def test_mimo_mtp_fastpath_fail_closed_rejects_missing_sidecar(
    tmp_path: Path,
) -> None:
    missing_sidecar_path = tmp_path / "missing-model_mtp.safetensors"
    request = ChatCompletionRequest(
        model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
        messages=[ChatCompletionMessage(role="user", content="Hello")],
        mimo_mtp_fastpath=True,
        mimo_mtp_depth=2,
        mimo_mtp_sidecar_path=str(missing_sidecar_path),
        mimo_mtp_fail_closed=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await chat_request_to_text_generation(request)

    assert exc_info.value.status_code == 400
    assert (
        exc_info.value.detail
        == "MiMo MTP fastpath is disabled: required sidecar is missing at "
        f"{missing_sidecar_path}. disable_reason=missing_sidecar. "
        "Provide a valid model_mtp.safetensors sidecar or disable mimo_mtp_fastpath."
    )
