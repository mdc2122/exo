"""Canonical internal type for text generation task parameters.

All external API formats (Chat Completions, Claude Messages, OpenAI Responses)
are converted to TextGenerationTaskParams at the API boundary via adapters.
"""

from collections.abc import Mapping
from typing import Any, Final, Literal, Self, cast

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
)
from pydantic_core.core_schema import SerializerFunctionWrapHandler

from exo.shared.types.common import ModelId

MessageRole = Literal["user", "assistant", "system", "developer"]
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
SUPPORTED_MIMO_MTP_FASTPATH_DEPTHS: Final[tuple[int, ...]] = (1, 2, 3)


def _supported_mimo_mtp_fastpath_depths_text() -> str:
    return ",".join(str(depth) for depth in SUPPORTED_MIMO_MTP_FASTPATH_DEPTHS)


def resolve_reasoning_params(
    reasoning_effort: ReasoningEffort | None,
    enable_thinking: bool | None,
) -> tuple[ReasoningEffort | None, bool | None]:
    """
    enable_thinking=True  -> reasoning_effort="medium"
    enable_thinking=False -> reasoning_effort="none"
    reasoning_effort="none" -> enable_thinking=False
    reasoning_effort=<anything else> -> enable_thinking=True
    """
    resolved_effort: ReasoningEffort | None = reasoning_effort
    resolved_thinking: bool | None = enable_thinking

    if reasoning_effort is None and enable_thinking is not None:
        resolved_effort = "medium" if enable_thinking else "none"

    if enable_thinking is None and reasoning_effort is not None:
        resolved_thinking = reasoning_effort != "none"

    return resolved_effort, resolved_thinking


class InputMessage(BaseModel, frozen=True):
    """Internal message for text generation pipelines."""

    role: MessageRole
    content: str


class VideoSource(BaseModel, frozen=True):
    """Normalized internal representation for one validated video payload.

    V1 only supports bounded base64-encoded MP4 payloads from validated
    data:video/mp4;base64 URLs. The worker-side Kimi video pipeline still
    consumes raw base64 strings, so this type keeps source metadata at the API
    boundary and exposes the exact payload via as_pipeline_payload().
    """

    type: Literal["base64"] = "base64"
    media_type: Literal["video/mp4"] = "video/mp4"
    data: str
    byte_count: int = Field(ge=0)

    def as_pipeline_payload(self) -> str:
        return self.data


class MimoMtpFastpathParams(BaseModel, frozen=True):
    """Guarded experimental MiMo V2.5 Pro MTP request intent.

    This is absent for default autoregressive requests so normal generation
    stays unchanged unless callers explicitly opt in at the API boundary.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool = True
    depth: int | None = None
    sidecar_path: str | None = None
    fail_closed: bool = True

    @field_validator("depth")
    @classmethod
    def _validate_supported_depth(cls, depth: int | None) -> int | None:
        if depth is None or depth in SUPPORTED_MIMO_MTP_FASTPATH_DEPTHS:
            return depth
        supported_depths = _supported_mimo_mtp_fastpath_depths_text()
        raise ValueError(
            f"unsupported MTP depth {depth}; supported depths: {supported_depths}; "
            "disable_reason=unsupported_depth"
        )


class MimoMtpRequestFields(BaseModel, frozen=True):
    """Guarded MTP request fields extracted from a ChatCompletionRequest.

    These are the API boundary fields that gate MTP intent. They map
    1:1 to MimoMtpFastpathParams attributes when the enabled flag is true,
    or produce None (disabled_default) when the enabled flag is false.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    mimo_mtp_fastpath: bool
    mimo_mtp_depth: int | None
    mimo_mtp_sidecar_path: str | None
    mimo_mtp_fail_closed: bool


def map_request_mtp_fields_to_fastpath_params(
    fields: MimoMtpRequestFields,
) -> MimoMtpFastpathParams | None:
    """Map guarded API request fields to MimoMtpFastpathParams.

    This is a pure function that converts the API-boundary MTP request
    fields into the internal MimoMtpFastpathParams model. When
    ``mimo_mtp_fastpath`` is False (the default), returns None so normal
    autoregressive generation stays unchanged. When True, returns a
    MimoMtpFastpathParams instance with enabled=True and the remaining
    fields mapped 1:1.

    Field mapping contract:
      - mimo_mtp_fastpath (bool) → gates None vs MimoMtpFastpathParams
      - mimo_mtp_depth (int | None) → depth (int | None)
      - mimo_mtp_sidecar_path (str | None) → sidecar_path (str | None)
      - mimo_mtp_fail_closed (bool) → fail_closed (bool)
    """
    if not fields.mimo_mtp_fastpath:
        return None
    return MimoMtpFastpathParams(
        enabled=True,
        depth=fields.mimo_mtp_depth,
        sidecar_path=fields.mimo_mtp_sidecar_path,
        fail_closed=fields.mimo_mtp_fail_closed,
    )


def default_mimo_mtp_fastpath_params(
    default_sidecar_path: str | None,
) -> MimoMtpFastpathParams | None:
    """Server-side MTP default for requests that carry no MTP fields.

    When the deployment sets a default sidecar path (the
    EXO_MIMO_MTP_DEFAULT_SIDECAR_PATH environment variable, read at the
    API layer), requests without explicit MTP intent default to the
    fastpath at the worker-resolved depth (1). fail_closed is forced
    False so the default can never break a request: an incompatible
    model, missing sidecar, or any fastpath rejection silently falls
    back to plain autoregressive generation. Explicit request fields
    always win — this is only consulted when
    map_request_mtp_fields_to_fastpath_params returned None.
    """
    if not default_sidecar_path:
        return None
    return MimoMtpFastpathParams(
        enabled=True,
        depth=None,
        sidecar_path=default_sidecar_path,
        fail_closed=False,
    )


def assign_mimo_mtp_fastpath_params(
    task_params: "TextGenerationTaskParams",
    mtp_params: MimoMtpFastpathParams | None,
) -> "TextGenerationTaskParams":
    """Assign MimoMtpFastpathParams into TextGenerationTaskParams.

    This is the canonical merge/assignment function for propagating guarded
    MTP intent from the API boundary into the internal task params. It is a
    pure function that returns a new TextGenerationTaskParams with the MTP
    field set or cleared, preserving all other fields unchanged.

    Merge strategy:
    - When mtp_params is None, returns task_params with
      mimo_mtp_fastpath_params set to None (default autoregressive generation).
    - When mtp_params is a valid MimoMtpFastpathParams, returns task_params
      with mimo_mtp_fastpath_params set to the provided instance.
    - Uses model_copy to preserve immutability; no in-place mutation.
    - The caller is responsible for validation (e.g., depth range, model
      eligibility) before calling this function.
    """
    return task_params.model_copy(
        update={"mimo_mtp_fastpath_params": mtp_params},
    )


class TextGenerationTaskParams(BaseModel, frozen=True):
    """Canonical internal task params for text generation.

    Every API adapter converts its wire type into this before handing
    off to the master/worker pipeline.
    """

    model_config = ConfigDict(populate_by_name=True)

    model: ModelId
    input: list[InputMessage]
    instructions: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    bench: bool = False
    top_k: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    chat_template_messages: list[dict[str, Any]] | None = None
    reasoning_effort: ReasoningEffort | None = None
    enable_thinking: bool | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    repetition_context_size: int | None = None
    images: list[str] = Field(default_factory=list)
    videos: list[str] = Field(default_factory=list)
    video_sources: list[VideoSource] = Field(default_factory=list)
    video_urls: list[str] = Field(default_factory=list)
    image_hashes: dict[int, str] = Field(default_factory=dict)
    total_input_chunks: int = 0
    image_count: int = 0
    mimo_mtp_fastpath_params: MimoMtpFastpathParams | None = Field(
        default=None,
        validation_alias=AliasChoices("mimo_mtp_fastpath_params", "mimo_mtp_fastpath"),
        serialization_alias="mimo_mtp_fastpath",
    )

    @property
    def mimo_mtp_fastpath(self) -> MimoMtpFastpathParams | None:
        return self.mimo_mtp_fastpath_params

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        if update is not None and "mimo_mtp_fastpath" in update:
            normalized_update: dict[str, Any] = dict(update)
            mimo_mtp_fastpath_update = cast(
                object, normalized_update.pop("mimo_mtp_fastpath")
            )
            if "mimo_mtp_fastpath_params" not in normalized_update:
                normalized_update["mimo_mtp_fastpath_params"] = mimo_mtp_fastpath_update
            update = normalized_update
        return super().model_copy(update=update, deep=deep)

    @model_serializer(mode="wrap")
    def _serialize_without_disabled_mtp(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        raw_serialized = cast(object, handler(self))
        assert isinstance(raw_serialized, dict)
        serialized: dict[str, Any] = dict(cast(Mapping[str, Any], raw_serialized))
        serialized.pop("mimo_mtp_fastpath_params", None)
        if self.mimo_mtp_fastpath_params is None or not self.mimo_mtp_fastpath_params.enabled:
            serialized.pop("mimo_mtp_fastpath", None)
        else:
            serialized["mimo_mtp_fastpath"] = self.mimo_mtp_fastpath_params.model_dump()
        return serialized
