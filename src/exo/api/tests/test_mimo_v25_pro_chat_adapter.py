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
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
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
    for mtp_field_name in (
        "mimo_mtp_fastpath",
        "mimo_mtp_depth",
        "mimo_mtp_sidecar_path",
        "mimo_mtp_fail_closed",
    ):
        assert mtp_field_name not in implicit_params.model_dump()
        assert mtp_field_name not in explicit_disabled_params.model_dump()


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
