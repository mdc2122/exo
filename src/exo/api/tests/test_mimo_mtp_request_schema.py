from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from exo.api.adapters import chat_completions
from exo.api.adapters.chat_completions import chat_request_to_text_generation
from exo.api.types import (
    BenchChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    GenerationStats,
)
from exo.shared.models.model_cards import MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory

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
    request = ChatCompletionRequest.model_validate(
        {
            "model": "kernelpool/MiMo-V2.5-Pro-6bit",
            "messages": [{"role": "user", "content": "Hello"}],
        }
    )

    for field_name, expected_default in MIMO_MTP_REQUEST_FIELD_DEFAULTS.items():
        assert getattr(request, field_name) == expected_default

    assert request.mimo_mtp_fastpath is False
    assert request.mimo_mtp_depth is None
    assert request.mimo_mtp_sidecar_path is None
    assert request.mimo_mtp_fail_closed is True


def test_serializing_legacy_ar_request_does_not_add_omitted_mtp_fields() -> None:
    legacy_ar_payload = {
        "model": "kernelpool/MiMo-V2.5-Pro-6bit",
        "messages": [{"role": "user", "content": "Summarize MiMo Pro."}],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": False,
    }

    request = ChatCompletionRequest.model_validate(legacy_ar_payload)

    assert request.model_dump(exclude_none=True) == legacy_ar_payload
    assert request.model_dump_json(exclude_none=True) == (
        '{"model":"kernelpool/MiMo-V2.5-Pro-6bit",'
        '"messages":[{"role":"user","content":"Summarize MiMo Pro."}],'
        '"max_tokens":16,"stream":false,"temperature":0.0}'
    )


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


def test_mimo_mtp_extension_fields_reject_coerced_request_values() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(
            {
                "model": "kernelpool/MiMo-V2.5-Pro-6bit",
                "messages": [{"role": "user", "content": "Hello"}],
                "mimo_mtp_fastpath": "true",
                "mimo_mtp_depth": "2",
                "mimo_mtp_sidecar_path": 123,
                "mimo_mtp_fail_closed": "false",
            }
        )


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
async def test_mimo_mtp_fail_closed_rejects_non_mimo_model_with_typed_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_sidecar_probe(path: str) -> object:
        raise AssertionError(
            f"non-MiMo fail-closed guard must not probe sidecar: {path}"
        )

    monkeypatch.setattr(chat_completions, "probe_mimo_mtp_sidecar", fail_sidecar_probe)
    request = ChatCompletionRequest(
        model=ModelId("llama-3.2-1b"),
        messages=[ChatCompletionMessage(role="user", content="Hello")],
        mimo_mtp_fastpath=True,
        mimo_mtp_depth=1,
        mimo_mtp_sidecar_path="/tmp/must-not-be-probed-model_mtp.safetensors",
        mimo_mtp_fail_closed=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await chat_request_to_text_generation(request)

    assert exc_info.value.status_code == 400
    raw_detail = cast(object, exc_info.value.detail)
    assert isinstance(raw_detail, dict)
    detail = cast(Mapping[str, object], raw_detail)
    assert detail["error"] == "mimo_mtp_model_ineligible"
    assert detail["mtp_enabled"] is False
    assert detail["accepted_execution_path"] == "rejected"
    assert detail["requested_mtp_depth"] == 1
    assert detail["mtp_depth"] is None
    assert detail["mtp_sidecar_status"] == "not_loaded"
    assert detail["mtp_disable_reason"] == "unsupported_model"
    assert detail["mtp_execution_state"] == "unsupported_model"
    assert detail["mimo_mtp_fail_closed"] is True
    assert detail["accepted_execution_path"] != "mimo_mtp_fastpath"
    assert detail["mtp_execution_state"] != "successful_mtp"


@pytest.mark.asyncio
async def test_mimo_mtp_fastpath_rejects_non_mimo_model_with_disable_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_sidecar_probe(path: str) -> object:
        raise AssertionError(f"non-MiMo guard must not probe sidecar: {path}")

    monkeypatch.setattr(chat_completions, "probe_mimo_mtp_sidecar", fail_sidecar_probe)
    request = ChatCompletionRequest(
        model=ModelId("llama-3.2-1b"),
        messages=[ChatCompletionMessage(role="user", content="Hello")],
        mimo_mtp_fastpath=True,
        mimo_mtp_depth=1,
        mimo_mtp_sidecar_path="/tmp/must-not-be-probed-model_mtp.safetensors",
        mimo_mtp_fail_closed=False,
    )

    with pytest.raises(HTTPException) as exc_info:
        await chat_request_to_text_generation(request)

    assert exc_info.value.status_code == 400
    raw_detail = cast(object, exc_info.value.detail)
    assert isinstance(raw_detail, dict)
    detail = cast(Mapping[str, object], raw_detail)
    assert detail == {
        "error": "mimo_mtp_model_ineligible",
        "message": (
            "MiMo MTP fastpath is only supported for MiMo V2.5 Pro models; "
            "model llama-3.2-1b is not eligible. Disable mimo_mtp_fastpath "
            "or use a MiMo V2.5 Pro model."
        ),
        "mtp_enabled": False,
        "accepted_execution_path": "rejected",
        "requested_mtp_depth": 1,
        "mtp_depth": None,
        "mtp_sidecar_status": "not_loaded",
        "mtp_disable_reason": "non_mimo_model",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("supported_depth", [1, 2, 3])
async def test_mimo_mtp_fastpath_accepts_supported_depth_values(
    monkeypatch: pytest.MonkeyPatch,
    supported_depth: int,
) -> None:
    sidecar_path = f"/tmp/supported-depth-{supported_depth}.safetensors"

    class ReadySidecarProbe:
        status = "ready"
        path = Path(sidecar_path)

    monkeypatch.setattr(
        chat_completions,
        "probe_mimo_mtp_sidecar",
        lambda path: ReadySidecarProbe(),
    )
    request = ChatCompletionRequest(
        model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
        messages=[ChatCompletionMessage(role="user", content="Hello")],
        mimo_mtp_fastpath=True,
        mimo_mtp_depth=supported_depth,
        mimo_mtp_sidecar_path=sidecar_path,
        mimo_mtp_fail_closed=False,
    )

    params = await chat_request_to_text_generation(request)

    mtp_params = params.mimo_mtp_fastpath
    assert mtp_params is not None
    assert mtp_params.depth == supported_depth
    assert mtp_params.enabled is True
    assert mtp_params.fail_closed is False


@pytest.mark.asyncio
async def test_mimo_mtp_fastpath_rejects_invalid_sidecar_before_dispatch(
    tmp_path: Path,
) -> None:
    invalid_sidecar_path = tmp_path / "invalid-model_mtp.safetensors"
    invalid_sidecar_path.write_text("not a safetensors file")
    request = ChatCompletionRequest(
        model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
        messages=[ChatCompletionMessage(role="user", content="Hello")],
        mimo_mtp_fastpath=True,
        mimo_mtp_depth=1,
        mimo_mtp_sidecar_path=str(invalid_sidecar_path),
        mimo_mtp_fail_closed=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await chat_request_to_text_generation(request)

    assert exc_info.value.status_code == 400
    raw_detail = cast(object, exc_info.value.detail)
    assert isinstance(raw_detail, dict)
    detail = cast(Mapping[str, object], raw_detail)
    assert detail["error"] == "mimo_mtp_sidecar_invalid"
    assert detail["mtp_enabled"] is False
    assert detail["accepted_execution_path"] == "rejected"
    assert detail["requested_mtp_depth"] == 1
    assert detail["mtp_depth"] is None
    assert detail["mtp_sidecar_status"] == "invalid"
    assert detail["mtp_disable_reason"] == "invalid_sidecar"
    assert "disable_reason=invalid_sidecar" in str(detail["message"])


@pytest.mark.asyncio
async def test_mimo_mtp_fastpath_rejects_unsupported_depth_before_sidecar_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_sidecar_probe(path: str) -> object:
        raise AssertionError(f"unsupported-depth guard must not probe sidecar: {path}")

    monkeypatch.setattr(chat_completions, "probe_mimo_mtp_sidecar", fail_sidecar_probe)
    request = ChatCompletionRequest(
        model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
        messages=[ChatCompletionMessage(role="user", content="Hello")],
        mimo_mtp_fastpath=True,
        mimo_mtp_depth=4,
        mimo_mtp_sidecar_path="/tmp/not-probed-for-unsupported-depth.safetensors",
        mimo_mtp_fail_closed=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await chat_request_to_text_generation(request)

    assert exc_info.value.status_code == 400
    raw_detail = cast(object, exc_info.value.detail)
    assert isinstance(raw_detail, dict)
    detail = cast(Mapping[str, object], raw_detail)
    assert detail == {
        "error": "mimo_mtp_depth_unsupported",
        "message": (
            "MiMo MTP fastpath is disabled: unsupported MTP depth 4. "
            "supported_depths=1,2,3. disable_reason=unsupported_depth. "
            "Use a supported mimo_mtp_depth or disable mimo_mtp_fastpath."
        ),
        "mtp_enabled": False,
        "accepted_execution_path": "rejected",
        "requested_mtp_depth": 4,
        "mtp_depth": None,
        "supported_mtp_depths": [1, 2, 3],
        "mtp_sidecar_status": "not_loaded",
        "mtp_disable_reason": "unsupported_depth",
        "mtp_execution_state": "unsupported_depth",
    }
    assert detail.get("mode") != "mtp"
    assert detail.get("mtp_execution_state") != "successful_mtp"


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
    raw_detail = cast(object, exc_info.value.detail)
    assert isinstance(raw_detail, dict)
    detail = cast(Mapping[str, object], raw_detail)
    assert detail == {
        "error": "mimo_mtp_sidecar_missing",
        "message": (
            "MiMo MTP fastpath is disabled: required sidecar is missing at "
            f"{missing_sidecar_path}. disable_reason=missing_sidecar. "
            "Provide a valid model_mtp.safetensors sidecar or disable "
            "mimo_mtp_fastpath."
        ),
        "mtp_enabled": False,
        "accepted_execution_path": "rejected",
        "requested_mtp_depth": 2,
        "mtp_depth": None,
        "mtp_sidecar_status": "missing",
        "mtp_disable_reason": "missing_sidecar",
        "mtp_execution_state": "missing_sidecar",
    }


@pytest.mark.asyncio
async def test_mimo_mtp_fastpath_fail_closed_missing_sidecar_reports_typed_disabled_state_and_never_claims_mtp(
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
    raw_detail = cast(object, exc_info.value.detail)
    assert isinstance(raw_detail, dict)
    detail = cast(Mapping[str, object], raw_detail)
    assert detail["error"] == "mimo_mtp_sidecar_missing"
    assert detail["mtp_enabled"] is False
    assert detail["accepted_execution_path"] == "rejected"
    assert detail["mtp_execution_state"] == "missing_sidecar"
    assert detail["mtp_disable_reason"] == "missing_sidecar"
    assert detail["mtp_sidecar_status"] == "missing"
    assert detail["accepted_execution_path"] != "mimo_mtp_fastpath"
    assert detail["mtp_execution_state"] != "successful_mtp"
    assert "successful" not in str(detail["message"]).lower()


def test_bench_chat_completion_response_schema_carries_mtp_telemetry() -> None:
    response = BenchChatCompletionResponse(
        id="bench-1",
        created=1,
        model="kernelpool/MiMo-V2.5-Pro-6bit",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content="ok"),
                finish_reason="stop",
            )
        ],
        generation_stats=GenerationStats(
            prompt_tps=100.0,
            generation_tps=35.0,
            prompt_tokens=8,
            generation_tokens=16,
            peak_memory_usage=Memory.from_bytes(123),
            attempted_depth_counts={"3": 6},
            accepted_depth_counts={"0": 1, "3": 5},
            acceptance_rate=5 / 6,
            fallback_count=1,
            timing_breakdown_seconds={
                "proposal": 0.1,
                "verification": 0.2,
                "acceptance": 0.03,
                "fallback": 0.01,
            },
        ),
        accepted_execution_path="mimo_mtp_fastpath",
        mtp_enabled=True,
        requested_mtp_depth=3,
        mtp_depth=3,
        mtp_sidecar_status="ready",
        attempted_depth_counts={"3": 6},
        accepted_depth_counts={"0": 1, "3": 5},
        acceptance_rate=5 / 6,
        fallback_count=1,
        timing_breakdown_seconds={
            "proposal": 0.1,
            "verification": 0.2,
            "acceptance": 0.03,
            "fallback": 0.01,
        },
    )

    serialized = response.model_dump()
    assert serialized["mtp_enabled"] is True
    assert serialized["requested_mtp_depth"] == 3
    assert serialized["attempted_depth_counts"] == {"3": 6}
    assert serialized["accepted_depth_counts"] == {"0": 1, "3": 5}
    assert serialized["acceptance_rate"] == 5 / 6
    assert serialized["fallback_count"] == 1
    assert serialized["timing_breakdown_seconds"] == {
        "proposal": 0.1,
        "verification": 0.2,
        "acceptance": 0.03,
        "fallback": 0.01,
    }
    generation_stats = cast(Mapping[str, object], serialized["generation_stats"])
    assert generation_stats["attempted_depth_counts"] == {"3": 6}
    assert generation_stats["accepted_depth_counts"] == {"0": 1, "3": 5}
    assert generation_stats["acceptance_rate"] == 5 / 6
    assert generation_stats["fallback_count"] == 1
