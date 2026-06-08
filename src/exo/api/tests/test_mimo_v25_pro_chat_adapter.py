# pyright: reportAny=false

import pytest

from exo.api.adapters import chat_completions
from exo.api.adapters.chat_completions import (
    VideoValidationError,
    chat_request_to_text_generation,
)
from exo.api.types import (
    ChatCompletionMessage,
    ChatCompletionMessageImageUrl,
    ChatCompletionMessageText,
    ChatCompletionMessageVideoUrl,
    ChatCompletionRequest,
)
from exo.shared.models.model_cards import (
    MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
    MIMO_V25_PRO_MODEL_IDS,
)
from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import (
    InputMessage,
    MimoMtpFastpathParams,
    TextGenerationTaskParams,
)
from exo.shared.types.video_errors import (
    VIDEO_ERROR_CODE_STREAMING_UNSUPPORTED,
    VIDEO_ERROR_CODE_UNSUPPORTED_FORMAT,
)


def _normal_request_golden_task_params() -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
        input=[InputMessage(role="user", content="Summarize MiMo Pro.")],
        max_output_tokens=16,
        temperature=0.0,
        top_p=0.95,
        top_k=40,
        seed=1234,
        stream=False,
        stop=["</s>"],
        chat_template_messages=[{"role": "user", "content": "Summarize MiMo Pro."}],
        logprobs=False,
        min_p=0.05,
        repetition_penalty=1.05,
        repetition_context_size=1024,
        images=[],
        videos=[],
        video_sources=[],
        video_urls=[],
    )


def _assert_no_mtp_task_params_contract(task_params: TextGenerationTaskParams) -> None:
    assert task_params.mimo_mtp_fastpath is None
    assert "mimo_mtp_fastpath_params" not in task_params.model_fields_set
    dumped_params = task_params.model_dump()
    assert "mimo_mtp_fastpath" not in dumped_params
    assert "mimo_mtp_fastpath_params" not in dumped_params
    assert "mimo_mtp_depth" not in dumped_params
    assert "mimo_mtp_sidecar_path" not in dumped_params
    assert "mimo_mtp_fail_closed" not in dumped_params


@pytest.mark.asyncio
async def test_normal_request_without_enabled_mtp_matches_pre_mtp_task_params_golden() -> (
    None
):
    implicit_disabled_request = ChatCompletionRequest(
        model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
        messages=[ChatCompletionMessage(role="user", content="Summarize MiMo Pro.")],
        max_tokens=16,
        temperature=0.0,
        top_p=0.95,
        top_k=40,
        seed=1234,
        stop=["</s>"],
        min_p=0.05,
        repetition_penalty=1.05,
        repetition_context_size=1024,
    )
    explicit_disabled_request = ChatCompletionRequest(
        model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
        messages=[ChatCompletionMessage(role="user", content="Summarize MiMo Pro.")],
        max_tokens=16,
        temperature=0.0,
        top_p=0.95,
        top_k=40,
        seed=1234,
        stop=["</s>"],
        min_p=0.05,
        repetition_penalty=1.05,
        repetition_context_size=1024,
        mimo_mtp_fastpath=False,
        mimo_mtp_depth=None,
        mimo_mtp_sidecar_path=None,
        mimo_mtp_fail_closed=True,
    )

    golden_params = _normal_request_golden_task_params()

    implicit_params = await chat_request_to_text_generation(implicit_disabled_request)
    explicit_disabled_params = await chat_request_to_text_generation(
        explicit_disabled_request
    )

    assert implicit_params == golden_params
    assert explicit_disabled_params == golden_params
    assert implicit_params.model_dump() == explicit_disabled_params.model_dump()
    _assert_no_mtp_task_params_contract(implicit_params)
    _assert_no_mtp_task_params_contract(explicit_disabled_params)


@pytest.mark.asyncio
async def test_absent_mtp_request_fields_leave_internal_fastpath_params_absent() -> (
    None
):
    request = ChatCompletionRequest(
        model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
        messages=[ChatCompletionMessage(role="user", content="Summarize MiMo Pro.")],
        max_tokens=16,
    )

    params = await chat_request_to_text_generation(request)

    assert params.mimo_mtp_fastpath_params is None
    assert "mimo_mtp_fastpath" not in params.model_dump()


@pytest.mark.asyncio
async def test_mtp_shaped_fields_without_explicit_fastpath_intent_remain_ar_params() -> (
    None
):
    request = ChatCompletionRequest.model_validate(
        {
            "model": MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
            "messages": [
                {"role": "user", "content": "Summarize MiMo Pro."},
            ],
            "max_tokens": 16,
            "mimo_mtp_depth": 2,
            "mimo_mtp_sidecar_path": "/models/model_mtp.safetensors",
            "mimo_mtp_fail_closed": False,
        }
    )

    assert "mimo_mtp_fastpath" not in request.model_fields_set
    assert request.mimo_mtp_fastpath is False

    params = await chat_request_to_text_generation(request)

    _assert_no_mtp_task_params_contract(params)
    assert params.model == MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID
    assert params.max_output_tokens == 16
    assert params.input == [InputMessage(role="user", content="Summarize MiMo Pro.")]


@pytest.mark.asyncio
async def test_disabled_mtp_request_fields_leave_internal_fastpath_params_absent() -> (
    None
):
    request = ChatCompletionRequest(
        model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
        messages=[ChatCompletionMessage(role="user", content="Summarize MiMo Pro.")],
        max_tokens=16,
        mimo_mtp_fastpath=False,
        mimo_mtp_depth=2,
        mimo_mtp_sidecar_path="/models/model_mtp.safetensors",
        mimo_mtp_fail_closed=False,
    )

    params = await chat_request_to_text_generation(request)

    assert params.mimo_mtp_fastpath_params is None
    assert "mimo_mtp_fastpath" not in params.model_dump()


@pytest.mark.asyncio
async def test_enabled_mtp_request_maps_to_internal_fastpath_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReadySidecarProbe:
        status = "ready"

    monkeypatch.setattr(
        chat_completions,
        "probe_mimo_mtp_sidecar",
        lambda path: ReadySidecarProbe(),
    )
    request = ChatCompletionRequest(
        model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
        messages=[ChatCompletionMessage(role="user", content="Summarize MiMo Pro.")],
        max_tokens=16,
        mimo_mtp_fastpath=True,
        mimo_mtp_depth=2,
        mimo_mtp_sidecar_path="/models/model_mtp.safetensors",
        mimo_mtp_fail_closed=False,
    )

    params = await chat_request_to_text_generation(request)

    assert params.mimo_mtp_fastpath_params == MimoMtpFastpathParams(
        enabled=True,
        depth=2,
        sidecar_path="/models/model_mtp.safetensors",
        fail_closed=False,
    )
    assert params.mimo_mtp_fastpath == params.mimo_mtp_fastpath_params
    dumped = params.model_dump()
    assert dumped["mimo_mtp_fastpath"] == {
        "enabled": True,
        "depth": 2,
        "sidecar_path": "/models/model_mtp.safetensors",
        "fail_closed": False,
    }


@pytest.mark.asyncio
async def test_raw_guarded_mtp_request_shape_maps_to_typed_task_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReadySidecarProbe:
        status = "ready"

    probed_sidecar_paths: list[str] = []

    def ready_sidecar_probe(path: str) -> ReadySidecarProbe:
        probed_sidecar_paths.append(path)
        return ReadySidecarProbe()

    monkeypatch.setattr(
        chat_completions, "probe_mimo_mtp_sidecar", ready_sidecar_probe
    )
    request = ChatCompletionRequest.model_validate(
        {
            "model": str(MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID),
            "messages": [
                {"role": "user", "content": "Summarize MiMo Pro with MTP."}
            ],
            "max_tokens": 32,
            "temperature": 0.0,
            "mimo_mtp_fastpath": True,
            "mimo_mtp_depth": 3,
            "mimo_mtp_sidecar_path": "/models/mimo-v25-pro/model_mtp.safetensors",
            "mimo_mtp_fail_closed": True,
        }
    )

    params = await chat_request_to_text_generation(request)

    assert probed_sidecar_paths == ["/models/mimo-v25-pro/model_mtp.safetensors"]
    assert params.mimo_mtp_fastpath_params == MimoMtpFastpathParams(
        enabled=True,
        depth=3,
        sidecar_path="/models/mimo-v25-pro/model_mtp.safetensors",
        fail_closed=True,
    )
    assert params.mimo_mtp_fastpath is params.mimo_mtp_fastpath_params
    assert params.model_dump()["mimo_mtp_fastpath"] == {
        "enabled": True,
        "depth": 3,
        "sidecar_path": "/models/mimo-v25-pro/model_mtp.safetensors",
        "fail_closed": True,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("model_id", MIMO_V25_PRO_MODEL_IDS)
async def test_mimo_v25_pro_accepts_text_only_chat_request(model_id: ModelId) -> None:
    request = ChatCompletionRequest(
        model=model_id,
        messages=[ChatCompletionMessage(role="user", content="Summarize MiMo Pro.")],
    )

    params = await chat_request_to_text_generation(request)

    assert params.input[0].role == "user"
    assert params.input[0].content == "Summarize MiMo Pro."
    assert params.images == []
    assert params.videos == []
    assert params.video_sources == []
    assert params.video_urls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("model_id", MIMO_V25_PRO_MODEL_IDS)
async def test_mimo_v25_pro_rejects_image_before_fetch(
    monkeypatch: pytest.MonkeyPatch, model_id: ModelId
) -> None:
    async def _fail_fetch_image_url(url: str) -> str:
        raise AssertionError(f"MiMo Pro guard must run before image fetch: {url}")

    monkeypatch.setattr(chat_completions, "fetch_image_url", _fail_fetch_image_url)
    request = ChatCompletionRequest(
        model=model_id,
        messages=[
            ChatCompletionMessage(
                role="user",
                content=[
                    ChatCompletionMessageText(text="What is in this image?"),
                    ChatCompletionMessageImageUrl(
                        image_url={"url": "https://example.test/image.png"}
                    ),
                ],
            )
        ],
    )

    with pytest.raises(VideoValidationError) as exc_info:
        await chat_request_to_text_generation(request)

    assert exc_info.value.status_code == 400
    assert exc_info.value.error_code == VIDEO_ERROR_CODE_UNSUPPORTED_FORMAT
    assert "text-only" in exc_info.value.error_message
    assert "image" in exc_info.value.error_message


@pytest.mark.asyncio
@pytest.mark.parametrize("model_id", MIMO_V25_PRO_MODEL_IDS)
async def test_mimo_v25_pro_rejects_video_before_url_validation(
    model_id: ModelId,
) -> None:
    request = ChatCompletionRequest(
        model=model_id,
        messages=[
            ChatCompletionMessage(
                role="user",
                content=[
                    ChatCompletionMessageText(text="Describe this clip."),
                    ChatCompletionMessageVideoUrl(
                        video_url={"url": "file:///tmp/clip.mp4"}
                    ),
                ],
            )
        ],
    )

    with pytest.raises(VideoValidationError) as exc_info:
        await chat_request_to_text_generation(request)

    assert exc_info.value.error_code == VIDEO_ERROR_CODE_UNSUPPORTED_FORMAT
    assert "video" in exc_info.value.error_message
    assert "Local file paths" not in exc_info.value.error_message


@pytest.mark.asyncio
@pytest.mark.parametrize("model_id", MIMO_V25_PRO_MODEL_IDS)
async def test_mimo_v25_pro_rejects_dict_audio_media_part(
    model_id: ModelId,
) -> None:
    request = ChatCompletionRequest(
        model=model_id,
        messages=[
            ChatCompletionMessage(
                role="user",
                content=[
                    ChatCompletionMessageText(text="Transcribe this."),
                    {"type": "input_audio", "input_audio": {"data": "abc"}},
                ],
            )
        ],
    )

    with pytest.raises(VideoValidationError) as exc_info:
        await chat_request_to_text_generation(request)

    assert exc_info.value.error_code == VIDEO_ERROR_CODE_UNSUPPORTED_FORMAT
    assert "input_audio" in exc_info.value.error_message


@pytest.mark.asyncio
async def test_non_mimo_kimi_video_validation_path_is_preserved() -> None:
    request = ChatCompletionRequest(
        model=ModelId("mlx-community/Kimi-K2.5"),
        stream=True,
        messages=[
            ChatCompletionMessage(
                role="user",
                content=[
                    ChatCompletionMessageText(text="Describe this clip."),
                    ChatCompletionMessageVideoUrl(
                        video_url={"url": "https://example.test/clip.mp4"}
                    ),
                ],
            )
        ],
    )

    with pytest.raises(VideoValidationError) as exc_info:
        await chat_request_to_text_generation(request)

    assert exc_info.value.error_code == VIDEO_ERROR_CODE_STREAMING_UNSUPPORTED
    assert "Streaming video chat completions" in exc_info.value.error_message
