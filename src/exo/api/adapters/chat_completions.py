"""OpenAI Chat Completions API adapter for converting requests/responses."""

import base64
import binascii
import ipaddress
import os
import re
import time
from collections.abc import AsyncGenerator, Mapping
from typing import Any, cast
from urllib.parse import urlparse

from fastapi import HTTPException

from exo.api.types import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionMessageImageUrl,
    ChatCompletionMessageText,
    ChatCompletionMessageVideoUrl,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorInfo,
    ErrorResponse,
    FinishReason,
    Logprobs,
    LogprobsContentItem,
    StreamingChoiceResponse,
    ToolCall,
    Usage,
    VideoErrorInfo,
    VideoErrorResponse,
)
from exo.download.download_utils import create_http_session
from exo.shared.constants import allow_kimi_video, allow_kimi_video_data_urls
from exo.shared.models.model_cards import MIMO_V25_PRO_MODEL_IDS
from exo.shared.types.chunks import (
    ErrorChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.common import CommandId
from exo.shared.types.text_generation import (
    InputMessage,
    MimoMtpFastpathParams,
    TextGenerationTaskParams,
    VideoSource,
    resolve_reasoning_params,
)
from exo.shared.types.video_errors import (
    VIDEO_ERROR_CODE_DECODE_FAILED as VIDEO_ERROR_CODE_DECODE_FAILED,
)
from exo.shared.types.video_errors import (
    VIDEO_ERROR_CODE_FETCH_FAILED as VIDEO_ERROR_CODE_FETCH_FAILED,
)
from exo.shared.types.video_errors import (
    VIDEO_ERROR_CODE_INVALID_VIDEO_URL as VIDEO_ERROR_CODE_INVALID_VIDEO_URL,
)
from exo.shared.types.video_errors import (
    VIDEO_ERROR_CODE_MULTIPLE_VIDEOS as VIDEO_ERROR_CODE_MULTIPLE_VIDEOS,
)
from exo.shared.types.video_errors import (
    VIDEO_ERROR_CODE_STREAMING_UNSUPPORTED as VIDEO_ERROR_CODE_STREAMING_UNSUPPORTED,
)
from exo.shared.types.video_errors import (
    VIDEO_ERROR_CODE_TOO_LARGE as VIDEO_ERROR_CODE_TOO_LARGE,
)
from exo.shared.types.video_errors import (
    VIDEO_ERROR_CODE_UNSUPPORTED_FORMAT as VIDEO_ERROR_CODE_UNSUPPORTED_FORMAT,
)
from exo.shared.types.video_errors import (
    VIDEO_ERROR_CODE_URL_BLOCKED as VIDEO_ERROR_CODE_URL_BLOCKED,
)
from exo.worker.engines.mlx.mimo_mtp_fast.sidecar_contract import (
    probe_mimo_mtp_sidecar,
)

DEFAULT_MAX_VIDEO_PAYLOAD_BYTES = 100 * 1024 * 1024
_MISSING_VIDEO_URL_FIELD = object()
_DATA_VIDEO_MP4_BASE64_RE = re.compile(
    r"\Adata:video/mp4;base64,(?P<payload>.*)\Z",
    re.IGNORECASE | re.DOTALL,
)
_DATA_URL_HEADER_RE = re.compile(
    r"\Adata:(?P<header>[^,]*)(?:,|\Z)", re.IGNORECASE | re.DOTALL
)

# Codecs explicitly listed by Moonshot's chat-with-video documentation.
# Indexed by lowercase MIME subtype (the bit after `video/`). Mirrored as
# file extensions for HTTP(S) URL path-suffix sniffing.
SUPPORTED_VIDEO_MIME_TYPES: frozenset[str] = frozenset(
    {
        "video/mp4",
        "video/mpeg",
        "video/quicktime",
        "video/x-msvideo",
        "video/x-flv",
        "video/mpg",
        "video/webm",
        "video/x-ms-wmv",
        "video/3gpp",
    }
)
SUPPORTED_VIDEO_MIME_SUBTYPES: frozenset[str] = frozenset(
    {mime_type.split("/", maxsplit=1)[1] for mime_type in SUPPORTED_VIDEO_MIME_TYPES}
)
SUPPORTED_VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".mpeg", ".mpg", ".mov", ".avi", ".flv", ".webm", ".wmv", ".3gp", ".3gpp"}
)


class VideoValidationError(HTTPException):
    """HTTPException subclass that carries an OpenAI-style structured error
    envelope for video request validation failures. The API exception
    handler detects this subclass and emits VideoErrorResponse JSON instead
    of the legacy ErrorResponse shape.
    """

    def __init__(
        self,
        *,
        status_code: int,
        message: str,
        code: str,
        param: str | None = "messages.content[].video_url.url",
    ) -> None:
        super().__init__(status_code=status_code, detail=message)
        self.error_code: str = code
        self.error_message: str = message
        self.error_param: str | None = param

    def to_response(self) -> VideoErrorResponse:
        return VideoErrorResponse(
            error=VideoErrorInfo(
                message=self.error_message,
                code=self.error_code,
                param=self.error_param,
            )
        )


def get_max_video_payload_bytes() -> int:
    raw = os.getenv("EXO_KIMI_VIDEO_MAX_BYTES")
    if raw is None or raw == "":
        raw = os.getenv("EXO_MAX_VIDEO_PAYLOAD_BYTES")
    if raw is None or raw == "":
        return DEFAULT_MAX_VIDEO_PAYLOAD_BYTES
    return int(raw)


def allow_video_data_urls() -> bool:
    """Return whether inline base64 video data URLs are enabled."""
    return allow_kimi_video_data_urls()


def validate_video_payload_size(byte_count: int, max_bytes: int | None = None) -> None:
    resolved_max_bytes = (
        get_max_video_payload_bytes() if max_bytes is None else max_bytes
    )
    if byte_count <= resolved_max_bytes:
        return

    actual_mib = byte_count / 1024 / 1024
    max_mib = resolved_max_bytes / 1024 / 1024
    raise VideoValidationError(
        status_code=413,
        code=VIDEO_ERROR_CODE_TOO_LARGE,
        message=(
            f"video payload is too large: {actual_mib:.2f} MiB exceeds "
            f"the configured limit of {max_mib:.2f} MiB. "
            "Use a shorter/lower-bitrate clip or raise EXO_KIMI_VIDEO_MAX_BYTES."
        ),
    )


def validate_video_base64_payload_size(
    video_b64: str, max_bytes: int | None = None
) -> None:
    stripped = video_b64.rstrip("=")
    estimated_byte_count = (len(stripped) * 3) // 4
    validate_video_payload_size(estimated_byte_count, max_bytes=max_bytes)


def is_data_video_mp4_base64_url(url: str) -> bool:
    """Return True for explicit Moonshot-style inline MP4 video data URLs."""
    return _DATA_VIDEO_MP4_BASE64_RE.match(url) is not None


def _invalid_data_video_url_error(message: str) -> VideoValidationError:
    return VideoValidationError(
        status_code=400,
        code=VIDEO_ERROR_CODE_INVALID_VIDEO_URL,
        message=message,
    )


def _validate_data_video_mp4_base64_payload(payload: str) -> None:
    if payload == "":
        raise _invalid_data_video_url_error(
            "video_url.url data:video/mp4;base64 payload is empty."
        )
    validate_video_base64_payload_size(payload)
    try:
        decoded_payload = base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise _invalid_data_video_url_error(
            "video_url.url has an invalid base64 video payload."
        ) from exc
    validate_video_payload_size(len(decoded_payload))


def extract_data_video_mp4_base64_payload(data_url: str) -> str:
    """Extract and validate the payload from data:video/mp4;base64,<payload>."""
    match = _DATA_VIDEO_MP4_BASE64_RE.match(data_url)
    if match is None:
        header_match = _DATA_URL_HEADER_RE.match(data_url)
        observed_header = (
            header_match.group("header") if header_match is not None else ""
        )
        observed = f"data:{observed_header}" if observed_header else "data URL"
        raise _invalid_data_video_url_error(
            "Malformed video_url.url data URL header. Expected exactly "
            f"data:video/mp4;base64,<base64 payload>; received {observed!r}."
        )
    payload = match.group("payload")
    _validate_data_video_mp4_base64_payload(payload)
    return payload


def _normalized_base64_mp4_video_source(video_b64: str) -> VideoSource:
    """Build the normalized source object after base64 payload validation."""
    decoded_payload = base64.b64decode(video_b64, validate=True)
    return VideoSource(
        type="base64",
        media_type="video/mp4",
        data=video_b64,
        byte_count=len(decoded_payload),
    )


def extract_base64_from_data_url(data_url: str) -> str:
    match = re.match(r"data:[^;]+;base64,(.+)", data_url)
    if match:
        return match.group(1)
    return data_url


def _extract_url_path_extension(url: str) -> str | None:
    """Return the lowercase path extension (with leading dot) for an HTTP URL."""
    parsed = urlparse(url)
    path = parsed.path or ""
    dot = path.rfind(".")
    if dot == -1:
        return None
    return path[dot:].lower()


def _looks_like_local_video_path(url: str) -> bool:
    """Return True for v1-unsupported local file video_url values.

    Detect these in request validation so they fail closed before any media
    loading or accidental base64/data-url fallback runs. HTTP(S) and data URLs
    are handled separately by validate_video_url_format.
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme == "file":
        return True
    if re.match(r"\A[a-zA-Z]:[\\/]", url):
        return True
    if scheme:
        return False

    stripped_url = url.strip()
    if stripped_url.startswith("//"):
        return False
    if stripped_url.startswith(("/", "./", "../", "~/")):
        return True
    if "/" in stripped_url or "\\" in stripped_url:
        return True

    path_extension = _extract_url_path_extension(stripped_url)
    return path_extension in SUPPORTED_VIDEO_EXTENSIONS


def _normalize_declared_video_mime_type(raw_mime_type: str) -> str:
    return raw_mime_type.split(";", maxsplit=1)[0].strip().lower()


def _validate_declared_video_mime_type(
    video_url_mapping: Mapping[object, object],
) -> None:
    for field_name in ("mime_type", "media_type", "content_type"):
        raw_mime_type = video_url_mapping.get(field_name)
        if raw_mime_type is None:
            continue
        param = f"messages.content[].video_url.{field_name}"
        if not isinstance(raw_mime_type, str) or raw_mime_type.strip() == "":
            raise VideoValidationError(
                status_code=400,
                code=VIDEO_ERROR_CODE_INVALID_VIDEO_URL,
                param=param,
                message=f"video_url.{field_name} must be a non-empty string when provided.",
            )
        mime_type = _normalize_declared_video_mime_type(raw_mime_type)
        if mime_type in SUPPORTED_VIDEO_MIME_TYPES:
            continue
        raise VideoValidationError(
            status_code=415,
            code=VIDEO_ERROR_CODE_UNSUPPORTED_FORMAT,
            param=param,
            message=(
                f"video MIME type '{mime_type}' is not supported for Kimi video_url v1. "
                f"Supported MIME types: {sorted(SUPPORTED_VIDEO_MIME_TYPES)}."
            ),
        )


def _raise_local_video_path_error(url: str) -> None:
    raise VideoValidationError(
        status_code=400,
        code=VIDEO_ERROR_CODE_INVALID_VIDEO_URL,
        message=(
            "Local file paths are not supported for Kimi video_url requests in v1: "
            f"'{url}'. Use a bounded http://, https://, or enabled "
            "data:video/mp4;base64 URL that is safe for the selected exo cluster."
        ),
    )


def validate_video_url_format(url: str) -> None:
    """Reject video URLs whose scheme, MIME type, or extension is unsupported.

    V1 accepts HTTP(S) URLs and, when explicitly enabled, bounded
    data:video/mp4;base64 URLs. HTTP(S) URLs are sniffed via path extension.
    Data URLs are strict: the header must be exactly video/mp4 plus base64
    encoding, and the payload must be valid base64 before runner dispatch.
    HTTP(S) URLs with no recognizable extension are accepted here and
    re-checked downstream by the worker-side decoder.
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()

    if _looks_like_local_video_path(url):
        _raise_local_video_path_error(url)

    if scheme in {"http", "https"}:
        ext = _extract_url_path_extension(url)
        if ext is None:
            # No extension to sniff; defer to downstream decoder.
            return
        if ext not in SUPPORTED_VIDEO_EXTENSIONS:
            raise VideoValidationError(
                status_code=415,
                code=VIDEO_ERROR_CODE_UNSUPPORTED_FORMAT,
                message=(
                    f"video extension '{ext}' is not in the supported codec "
                    f"list. Supported extensions: "
                    f"{sorted(SUPPORTED_VIDEO_EXTENSIONS)}."
                ),
            )
        return

    if scheme == "data":
        if not allow_video_data_urls():
            raise VideoValidationError(
                status_code=400,
                code=VIDEO_ERROR_CODE_INVALID_VIDEO_URL,
                message=(
                    "video data URL base64 payloads are disabled by default. "
                    "Set EXO_KIMI_VIDEO_ALLOW_DATA_URLS=1 only for bounded "
                    "Kimi video smokes after payload-size limits are verified."
                ),
            )
        extract_data_video_mp4_base64_payload(url)
        return

    rejected_scheme = scheme or "missing"
    raise VideoValidationError(
        status_code=400,
        code=VIDEO_ERROR_CODE_INVALID_VIDEO_URL,
        message=(
            f"unsupported video_url scheme '{rejected_scheme}'. "
            "Use http://, https://, or data:video URLs for Kimi video requests. "
            "Local paths and other remote schemes are out of scope for v1."
        ),
    )


def allow_local_video_urls() -> bool:
    # Keep the pre-candidate override as a compatibility alias so existing
    # single-node smoke scripts do not silently lose their intentional local-URL
    # bypass when the Kimi-specific gate is introduced.
    return (
        os.getenv("EXO_KIMI_VIDEO_ALLOW_LOCAL_URLS")
        or os.getenv("EXO_ALLOW_LOCAL_VIDEO_URLS")
        or ""
    ).lower() in {
        "1",
        "true",
        "yes",
    }


def _is_blocked_node_local_video_host(host: str) -> bool:
    """Return True for URL hosts that should fail closed for cluster video fetch.

    V1 worker-side video fetch runs on the selected Kimi runner. Literal
    loopback, link-local, private LAN/ULA, unspecified, multicast, and mDNS
    hosts are unsafe by default because they may resolve differently per Studio
    node or be reachable only from the API node. DNS hostnames are otherwise
    left unresolved here so validation stays cheap and avoids network side
    effects in the API/coordinator path.
    """
    normalized_host = host.rstrip(".").lower()
    local_hostnames = {"localhost", "localhost.localdomain"}
    if normalized_host in local_hostnames or normalized_host.endswith(".localhost"):
        return True
    if normalized_host.endswith(".local"):
        return True

    try:
        ip = ipaddress.ip_address(normalized_host)
    except ValueError:
        return False

    return (
        ip.is_loopback
        or ip.is_link_local
        or ip.is_private
        or ip.is_unspecified
        or ip.is_multicast
        or ip.is_reserved
    )


def validate_distributed_video_url(
    url: str, *, allow_node_local_video_urls: bool = False
) -> None:
    """Reject URLs that are likely reachable only from one node in a cluster."""
    if allow_node_local_video_urls or allow_local_video_urls():
        return

    parsed = urlparse(url)
    hostname = parsed.hostname
    if hostname is None:
        return

    if not _is_blocked_node_local_video_host(hostname):
        return

    raise VideoValidationError(
        status_code=400,
        code=VIDEO_ERROR_CODE_URL_BLOCKED,
        message=(
            "video_url must be reachable from every exo node before it can be "
            "used by distributed Kimi video requests. "
            f"Non-cluster-reachable video URL '{url}' is only safe for "
            "single-node testing; serve the video on a routable address such "
            "as the Studio Tailscale IP, set EXO_KIMI_VIDEO_ALLOW_LOCAL_URLS=1, "
            "or pass kimi_video_allow_local_urls=true for an intentional "
            "single-node smoke override."
        ),
    )


async def fetch_image_url(url: str) -> str:
    headers = {"User-Agent": "exo/1.0"}
    async with (
        create_http_session(timeout_profile="short") as session,
        session.get(url, headers=headers) as resp,
    ):
        resp.raise_for_status()
        data = await resp.read()
        return base64.b64encode(data).decode("ascii")


async def fetch_video_url(url: str) -> str:
    headers = {"User-Agent": "exo/1.0"}
    async with (
        create_http_session(timeout_profile="long") as session,
        session.get(url, headers=headers) as resp,
    ):
        resp.raise_for_status()
        content_length = resp.headers.get("Content-Length")
        if content_length is not None:
            validate_video_payload_size(int(content_length))
        data = await resp.read()
        validate_video_payload_size(len(data))
        return base64.b64encode(data).decode("ascii")


def _is_raw_frame_array_video_payload(part: object) -> bool:
    if not isinstance(part, Mapping):
        return False

    part_mapping = cast(Mapping[object, object], part)
    frames = part_mapping.get("frames")
    has_top_level_frames = isinstance(frames, list)

    raw_type = part_mapping.get("type")
    if isinstance(raw_type, str):
        normalized_raw_type = raw_type.lower()
        if has_top_level_frames and normalized_raw_type in {
            "video",
            "input_video",
            "video_frames",
            "frame_array",
        }:
            return True
        nested_payload = part_mapping.get(normalized_raw_type)
        if isinstance(nested_payload, Mapping):
            nested_payload_mapping = cast(Mapping[object, object], nested_payload)
            if isinstance(nested_payload_mapping.get("frames"), list):
                return True

    if not has_top_level_frames:
        return False

    # Some clients incorrectly put raw decoded/extracted frame arrays inside
    # the video_url object instead of the supported {"url": "..."} shape.
    return "url" not in part_mapping


def _raise_raw_frame_array_video_payload_error(*, param: str) -> None:
    raise VideoValidationError(
        status_code=400,
        code=VIDEO_ERROR_CODE_UNSUPPORTED_FORMAT,
        param=param,
        message=(
            "Raw frame-array video payloads are not supported for Kimi video "
            "chat completions v1. Use exactly one mixed content part with a "
            "video_url.url string: {'type': 'video_url', 'video_url': {'url': "
            "'<http(s) URL or enabled data:video/mp4;base64 URL>'}} so the selected runner can "
            "fetch/decode/preprocess the video through Kimi's native media-token path."
        ),
    )


def _require_video_url_string(video_url: object) -> str:
    """Validate OpenAI/Moonshot video_url part shape and return its URL.

    Pydantic intentionally keeps ChatCompletionMessageVideoUrl.video_url loose so
    malformed client payloads can be rejected here with the same structured
    OpenAI-style envelope used by other video validation failures, instead of a
    generic FastAPI/Pydantic 422.
    """
    if not isinstance(video_url, Mapping):
        raise VideoValidationError(
            status_code=400,
            code=VIDEO_ERROR_CODE_INVALID_VIDEO_URL,
            param="messages.content[].video_url",
            message="video_url must be an object containing a string url field.",
        )

    video_url_mapping = cast(Mapping[object, object], video_url)
    _validate_declared_video_mime_type(video_url_mapping)
    if _is_raw_frame_array_video_payload(video_url_mapping):
        _raise_raw_frame_array_video_payload_error(param="messages.content[].video_url")

    if "url" not in video_url_mapping:
        raise VideoValidationError(
            status_code=400,
            code=VIDEO_ERROR_CODE_INVALID_VIDEO_URL,
            param="messages.content[].video_url.url",
            message="video_url is missing required url field.",
        )

    url = video_url_mapping["url"]
    if not isinstance(url, str):
        raise VideoValidationError(
            status_code=400,
            code=VIDEO_ERROR_CODE_INVALID_VIDEO_URL,
            param="messages.content[].video_url.url",
            message="video_url.url must be a string.",
        )

    if url.strip() == "":
        raise VideoValidationError(
            status_code=400,
            code=VIDEO_ERROR_CODE_INVALID_VIDEO_URL,
            param="messages.content[].video_url.url",
            message="video_url.url must not be empty.",
        )

    return url


def _message_contains_video_url(content: object) -> bool:
    if isinstance(content, ChatCompletionMessageVideoUrl):
        return True
    if isinstance(content, list):
        parts = cast(list[object], content)
        return any(
            isinstance(part, ChatCompletionMessageVideoUrl)
            or _is_raw_frame_array_video_payload(part)
            for part in parts
        )
    return False


def validate_video_url_content_array_shape(request: ChatCompletionRequest) -> None:
    """Validate OpenAI/Moonshot mixed content-array shape for video_url.

    Kimi video support intentionally accepts video_url only as one part inside a
    mixed content array. Rejecting top-level/non-array video_url and video-only
    arrays before URL validation avoids assertion failures and keeps malformed
    media requests from being dispatched to workers.
    """
    for message_index, message in enumerate(request.messages):
        message_content_param = f"messages[{message_index}].content"
        content = message.content
        if not _message_contains_video_url(content):
            continue

        if not isinstance(content, list):
            raise VideoValidationError(
                status_code=400,
                code=VIDEO_ERROR_CODE_INVALID_VIDEO_URL,
                param=message_content_param,
                message=(
                    "content containing video_url must be a mixed content array "
                    "with at least one text part and one video_url part."
                ),
            )

        parts = cast(list[object], content)
        if any(_is_raw_frame_array_video_payload(part) for part in parts):
            _raise_raw_frame_array_video_payload_error(param="messages.content[].video")

        invalid_parts = [
            part
            for part in parts
            if not isinstance(
                part,
                (
                    ChatCompletionMessageText,
                    ChatCompletionMessageImageUrl,
                    ChatCompletionMessageVideoUrl,
                ),
            )
        ]
        if invalid_parts:
            raise VideoValidationError(
                status_code=400,
                code=VIDEO_ERROR_CODE_INVALID_VIDEO_URL,
                param="messages.content[]",
                message=(
                    "content arrays containing video_url may only include text, "
                    "image_url, or video_url parts."
                ),
            )

        text_parts = [
            part for part in parts if isinstance(part, ChatCompletionMessageText)
        ]
        if not text_parts:
            raise VideoValidationError(
                status_code=400,
                code=VIDEO_ERROR_CODE_INVALID_VIDEO_URL,
                param=message_content_param,
                message=(
                    "content containing video_url must include at least one text "
                    "part in the same mixed content array."
                ),
            )

        has_non_empty_text_context = any(part.text.strip() for part in text_parts)
        if not has_non_empty_text_context:
            same_message_scope = (
                "same user message" if message.role == "user" else "same message"
            )
            raise VideoValidationError(
                status_code=400,
                code=VIDEO_ERROR_CODE_INVALID_VIDEO_URL,
                param=message_content_param,
                message=(
                    f"{message.role} messages containing video_url must include "
                    f"non-empty text context in the {same_message_scope}."
                ),
            )


def _count_video_url_parts_in_message(message: ChatCompletionMessage) -> int:
    if isinstance(message.content, ChatCompletionMessageVideoUrl):
        return 1
    if isinstance(message.content, list):
        return sum(
            1
            for part in message.content
            if isinstance(part, ChatCompletionMessageVideoUrl)
        )
    return 0


def validate_single_video_url_part_per_request(request: ChatCompletionRequest) -> None:
    total_video_url_part_count = 0
    for message in request.messages:
        message_video_url_part_count = _count_video_url_parts_in_message(message)
        if message_video_url_part_count > 1:
            raise VideoValidationError(
                status_code=400,
                code=VIDEO_ERROR_CODE_MULTIPLE_VIDEOS,
                param="messages.content[].video_url",
                message=(
                    "chat completions accept at most one video_url part in a single "
                    f"message, received {message_video_url_part_count}. Submit additional "
                    "videos as separate requests."
                ),
            )
        total_video_url_part_count += message_video_url_part_count

    if total_video_url_part_count > 1:
        raise VideoValidationError(
            status_code=400,
            code=VIDEO_ERROR_CODE_MULTIPLE_VIDEOS,
            param="messages.content[].video_url",
            message=(
                "chat completions accept at most one video_url part in a single "
                f"chat completion request, received {total_video_url_part_count}. "
                "Submit additional videos as separate requests."
            ),
        )


def _request_contains_video_url(request: ChatCompletionRequest) -> bool:
    return any(
        _count_video_url_parts_in_message(message) for message in request.messages
    )


def validate_video_streaming_not_requested(request: ChatCompletionRequest) -> None:
    if not request.stream:
        return
    if not _request_contains_video_url(request):
        return
    raise VideoValidationError(
        status_code=400,
        code=VIDEO_ERROR_CODE_STREAMING_UNSUPPORTED,
        param="stream",
        message=(
            "Streaming video chat completions are out of scope for v1. "
            "Set stream=false and use a short video fixture so preprocessing "
            "latency is measured separately from generation throughput."
        ),
    )


def validate_kimi_video_feature_enabled(request: ChatCompletionRequest) -> None:
    if not _request_contains_video_url(request) or allow_kimi_video():
        return
    raise VideoValidationError(
        status_code=400,
        code=VIDEO_ERROR_CODE_URL_BLOCKED,
        param="messages.content[].video_url",
        message=(
            "Kimi video_url chat completions are disabled. Set "
            "EXO_KIMI_VIDEO_ENABLED=1 only after local smoke checks pass."
        ),
    )


def _is_mimo_v25_pro_request(request: ChatCompletionRequest) -> bool:
    return request.model in MIMO_V25_PRO_MODEL_IDS


def _mapping_media_marker(content_part: Mapping[object, object]) -> str | None:
    raw_type = content_part.get("type")
    normalized_type = raw_type.lower() if isinstance(raw_type, str) else ""
    if normalized_type in {
        "audio",
        "image",
        "image_url",
        "input_audio",
        "input_image",
        "input_video",
        "speech",
        "video",
        "video_url",
    }:
        return normalized_type
    for marker in ("audio", "image", "image_url", "input_audio", "video", "video_url"):
        if marker in content_part:
            return marker
    if _is_raw_frame_array_video_payload(content_part):
        return "video"
    return None


def _content_part_media_marker(content_part: object) -> str | None:
    if isinstance(content_part, ChatCompletionMessageImageUrl):
        return "image_url"
    if isinstance(content_part, ChatCompletionMessageVideoUrl):
        return "video_url"
    if isinstance(content_part, Mapping):
        return _mapping_media_marker(cast(Mapping[object, object], content_part))
    return None


def _message_media_marker(message: ChatCompletionMessage) -> str | None:
    content = message.content
    if isinstance(content, list):
        for part in content:
            marker = _content_part_media_marker(part)
            if marker is not None:
                return marker
        return None
    return _content_part_media_marker(content)


def validate_mimo_mtp_fastpath_eligibility(request: ChatCompletionRequest) -> None:
    """Fail closed when an explicit guarded MiMo MTP fastpath request is ineligible."""
    if not request.mimo_mtp_fastpath:
        return

    if not _is_mimo_v25_pro_request(request):
        raise HTTPException(
            status_code=400,
            detail=(
                "MiMo MTP fastpath is only supported for MiMo V2.5 Pro models; "
                f"model {request.model} is not eligible. "
                "Disable mimo_mtp_fastpath or use a MiMo V2.5 Pro model."
            ),
        )

    if not request.mimo_mtp_fail_closed:
        return

    if request.mimo_mtp_sidecar_path is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "MiMo MTP fastpath is disabled: required sidecar path was not "
                "provided. disable_reason=missing_sidecar. Provide a valid "
                "model_mtp.safetensors sidecar or disable mimo_mtp_fastpath."
            ),
        )

    sidecar_probe = probe_mimo_mtp_sidecar(request.mimo_mtp_sidecar_path)
    if sidecar_probe.status == "missing":
        raise HTTPException(
            status_code=400,
            detail=(
                "MiMo MTP fastpath is disabled: required sidecar is missing at "
                f"{sidecar_probe.path}. disable_reason=missing_sidecar. "
                "Provide a valid model_mtp.safetensors sidecar or disable "
                "mimo_mtp_fastpath."
            ),
        )


def validate_mimo_v25_pro_text_only_request(request: ChatCompletionRequest) -> None:
    """Fail closed for MiMo Pro media requests before media fetch or dispatch."""
    if not _is_mimo_v25_pro_request(request):
        return

    for message_index, message in enumerate(request.messages):
        marker = _message_media_marker(message)
        if marker is None:
            continue
        raise VideoValidationError(
            status_code=400,
            code=VIDEO_ERROR_CODE_UNSUPPORTED_FORMAT,
            param=f"messages[{message_index}].content",
            message=(
                f"{request.model} is text-only in Track A and does not "
                f"support media content ({marker}). Submit a text-only request; "
                "image, video, audio, speech, multimodal, and omnimodal inputs "
                "are rejected before inference or media download."
            ),
        )


async def chat_request_to_text_generation(
    request: ChatCompletionRequest,
) -> TextGenerationTaskParams:
    validate_mimo_mtp_fastpath_eligibility(request)
    validate_mimo_v25_pro_text_only_request(request)
    validate_video_url_content_array_shape(request)
    validate_single_video_url_part_per_request(request)
    validate_video_streaming_not_requested(request)
    validate_kimi_video_feature_enabled(request)

    instructions: str | None = None
    input_messages: list[InputMessage] = []
    chat_template_messages: list[dict[str, Any]] = []
    images: list[str] = []
    videos: list[str] = []
    video_sources: list[VideoSource] = []
    video_urls: list[str] = []
    allow_node_local_video_urls = request.kimi_video_allow_local_urls is True

    for msg in request.messages:
        # Normalize content to string
        content: str
        has_images = False
        has_videos = False
        if msg.content is None:
            content = ""
        elif isinstance(msg.content, str):
            content = msg.content
        elif isinstance(msg.content, ChatCompletionMessageText):
            content = msg.content.text
        elif isinstance(msg.content, ChatCompletionMessageImageUrl):
            url = msg.content.image_url.get("url", "")
            if url:
                if url.startswith(("http://", "https://")):
                    images.append(await fetch_image_url(url))
                else:
                    images.append(extract_base64_from_data_url(url))
                has_images = True
            content = ""
        elif isinstance(msg.content, ChatCompletionMessageVideoUrl):
            url = _require_video_url_string(
                getattr(msg.content, "video_url", _MISSING_VIDEO_URL_FIELD)
            )
            validate_video_url_format(url)
            if url.startswith(("http://", "https://")):
                validate_distributed_video_url(
                    url,
                    allow_node_local_video_urls=allow_node_local_video_urls,
                )
                video_urls.append(url)
            else:
                if is_data_video_mp4_base64_url(url):
                    video_b64 = extract_data_video_mp4_base64_payload(url)
                else:
                    video_b64 = extract_base64_from_data_url(url)
                validate_video_base64_payload_size(video_b64)
                video_sources.append(_normalized_base64_mp4_video_source(video_b64))
            has_videos = True
            content = ""
        else:
            video_url_part_count = sum(
                1
                for part in msg.content
                if isinstance(part, ChatCompletionMessageVideoUrl)
            )
            if video_url_part_count > 1:
                raise VideoValidationError(
                    status_code=400,
                    code=VIDEO_ERROR_CODE_MULTIPLE_VIDEOS,
                    param="messages.content[].video_url",
                    message=(
                        "chat completions accept at most one video_url part in a single "
                        f"message, received {video_url_part_count}. Submit additional "
                        "videos as separate requests."
                    ),
                )

            text_parts: list[str] = []
            for part in msg.content:
                if isinstance(part, ChatCompletionMessageText):
                    text_parts.append(part.text)
                elif isinstance(part, ChatCompletionMessageVideoUrl):
                    url = _require_video_url_string(
                        getattr(part, "video_url", _MISSING_VIDEO_URL_FIELD)
                    )
                    validate_video_url_format(url)
                    if url.startswith(("http://", "https://")):
                        validate_distributed_video_url(
                            url,
                            allow_node_local_video_urls=allow_node_local_video_urls,
                        )
                        video_urls.append(url)
                    else:
                        if is_data_video_mp4_base64_url(url):
                            video_b64 = extract_data_video_mp4_base64_payload(url)
                        else:
                            video_b64 = extract_base64_from_data_url(url)
                        validate_video_base64_payload_size(video_b64)
                        video_sources.append(
                            _normalized_base64_mp4_video_source(video_b64)
                        )
                    has_videos = True
                elif isinstance(part, ChatCompletionMessageImageUrl):
                    url = part.image_url.get("url", "")
                    if url:
                        if url.startswith(("http://", "https://")):
                            images.append(await fetch_image_url(url))
                        else:
                            images.append(extract_base64_from_data_url(url))
                        has_images = True
                else:
                    raise VideoValidationError(
                        status_code=400,
                        code=VIDEO_ERROR_CODE_INVALID_VIDEO_URL,
                        param="messages.content[]",
                        message=(
                            "content arrays may only include text, image_url, or "
                            "supported video_url parts. Raw frame-array video payloads "
                            "must not be used."
                        ),
                    )
            content = "\n".join(text_parts)

        # Extract system message as instructions
        if msg.role == "system":
            if instructions is None:
                instructions = content
            else:
                # Append additional system messages
                instructions = f"{instructions}\n{content}"
            chat_template_messages.append({"role": "system", "content": content})
        else:
            # Skip messages with no meaningful content
            if (
                msg.content is None
                and msg.reasoning_content is None
                and msg.tool_calls is None
            ):
                continue

            if msg.role in ("user", "assistant", "developer"):
                input_messages.append(InputMessage(role=msg.role, content=content))

            # Build full message dict for chat template (preserves tool_calls etc.)
            # Normalize content for model_dump
            if has_images or has_videos:
                multimodal_content: list[dict[str, Any]] = []
                assert isinstance(msg.content, list)
                for part in msg.content:
                    if isinstance(part, ChatCompletionMessageText):
                        multimodal_content.append({"type": "text", "text": part.text})
                    elif isinstance(part, ChatCompletionMessageVideoUrl):
                        multimodal_content.append(
                            {"type": "video_url", "video_url": part.video_url}
                        )
                    else:
                        multimodal_content.append({"type": "image"})
                chat_template_messages.append(
                    {"role": msg.role, "content": multimodal_content}
                )
                continue
            msg_copy = msg.model_copy(update={"content": content})

            dumped: dict[str, Any] = msg_copy.model_dump(exclude_none=True)
            chat_template_messages.append(dumped)

    total_video_count = len(videos) + len(video_sources) + len(video_urls)
    if total_video_count > 1:
        raise VideoValidationError(
            status_code=400,
            code=VIDEO_ERROR_CODE_MULTIPLE_VIDEOS,
            message=(
                f"chat completions accept at most one video per request, "
                f"received {total_video_count}. "
                "Submit additional videos as separate requests."
            ),
        )

    resolved_effort, resolved_thinking = resolve_reasoning_params(
        request.reasoning_effort, request.enable_thinking
    )
    mimo_mtp_fastpath = (
        MimoMtpFastpathParams(
            enabled=True,
            depth=request.mimo_mtp_depth,
            sidecar_path=request.mimo_mtp_sidecar_path,
            fail_closed=request.mimo_mtp_fail_closed,
        )
        if request.mimo_mtp_fastpath
        else None
    )

    return TextGenerationTaskParams(
        model=request.model,
        input=input_messages
        if input_messages
        else [InputMessage(role="user", content="")],
        instructions=instructions,
        max_output_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop=request.stop,
        seed=request.seed,
        stream=request.stream,
        tools=request.tools,
        reasoning_effort=resolved_effort,
        enable_thinking=resolved_thinking,
        chat_template_messages=chat_template_messages
        if chat_template_messages
        else None,
        logprobs=request.logprobs or False,
        top_logprobs=request.top_logprobs,
        min_p=request.min_p,
        repetition_penalty=request.repetition_penalty,
        repetition_context_size=request.repetition_context_size,
        images=images,
        videos=videos,
        video_sources=video_sources,
        video_urls=video_urls,
        mimo_mtp_fastpath=mimo_mtp_fastpath,
    )


def chunk_to_response(
    chunk: TokenChunk, command_id: CommandId
) -> ChatCompletionResponse:
    """Convert a TokenChunk to a streaming ChatCompletionResponse."""
    # Build logprobs if available
    logprobs: Logprobs | None = None
    if chunk.logprob is not None:
        logprobs = Logprobs(
            content=[
                LogprobsContentItem(
                    token=chunk.text,
                    logprob=chunk.logprob,
                    top_logprobs=chunk.top_logprobs or [],
                )
            ]
        )

    if chunk.is_thinking:
        delta = ChatCompletionMessage(role="assistant", reasoning_content=chunk.text)
    else:
        delta = ChatCompletionMessage(role="assistant", content=chunk.text)

    return ChatCompletionResponse(
        id=command_id,
        created=int(time.time()),
        model=chunk.model,
        choices=[
            StreamingChoiceResponse(
                index=0,
                delta=delta,
                logprobs=logprobs,
                finish_reason=chunk.finish_reason,
            )
        ],
    )


async def generate_chat_stream(
    command_id: CommandId,
    chunk_stream: AsyncGenerator[
        PrefillProgressChunk | ErrorChunk | ToolCallChunk | TokenChunk, None
    ],
) -> AsyncGenerator[str, None]:
    """Generate Chat Completions API streaming events from chunks."""
    last_usage: Usage | None = None

    async for chunk in chunk_stream:
        match chunk:
            case PrefillProgressChunk():
                # Use SSE comment so third-party clients ignore it
                yield f": prefill_progress {chunk.model_dump_json()}\n\n"

            case ErrorChunk():
                if chunk.error_code is not None:
                    video_error_response = VideoErrorResponse(
                        error=VideoErrorInfo(
                            message=chunk.error_message or "Internal server error",
                            code=chunk.error_code,
                        )
                    )
                    yield f"data: {video_error_response.model_dump_json()}\n\n"
                else:
                    error_response = ErrorResponse(
                        error=ErrorInfo(
                            message=chunk.error_message or "Internal server error",
                            type="InternalServerError",
                            code=500,
                        )
                    )
                    yield f"data: {error_response.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                return

            case ToolCallChunk():
                last_usage = chunk.usage or last_usage

                tool_call_deltas = [
                    ToolCall(
                        id=tool.id,
                        index=i,
                        function=tool,
                    )
                    for i, tool in enumerate(chunk.tool_calls)
                ]
                tool_response = ChatCompletionResponse(
                    id=command_id,
                    created=int(time.time()),
                    model=chunk.model,
                    choices=[
                        StreamingChoiceResponse(
                            index=0,
                            delta=ChatCompletionMessage(
                                role="assistant",
                                tool_calls=tool_call_deltas,
                            ),
                            finish_reason="tool_calls",
                        )
                    ],
                    usage=last_usage,
                )
                yield f"data: {tool_response.model_dump_json()}\n\n"
                if chunk.stats is not None:
                    yield f": generation_stats {chunk.stats.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                return

            case TokenChunk():
                last_usage = chunk.usage or last_usage

                chunk_response = chunk_to_response(chunk, command_id)
                if chunk.finish_reason is not None:
                    chunk_response = chunk_response.model_copy(
                        update={"usage": last_usage}
                    )
                yield f"data: {chunk_response.model_dump_json()}\n\n"

                if chunk.finish_reason is not None:
                    if chunk.stats is not None:
                        yield f": generation_stats {chunk.stats.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"
                    return


async def collect_chat_response(
    command_id: CommandId,
    chunk_stream: AsyncGenerator[
        ErrorChunk | ToolCallChunk | TokenChunk | PrefillProgressChunk, None
    ],
) -> ChatCompletionResponse | ErrorResponse | VideoErrorResponse:
    """Collect all token chunks and return a single ChatCompletionResponse."""
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    logprobs_content: list[LogprobsContentItem] = []
    model: str | None = None
    finish_reason: FinishReason | None = None
    error_message: str | None = None
    error_code: str | None = None
    last_usage: Usage | None = None

    async for chunk in chunk_stream:
        match chunk:
            case PrefillProgressChunk():
                continue

            case ErrorChunk():
                error_message = chunk.error_message or "Internal server error"
                error_code = chunk.error_code
                break

            case TokenChunk():
                if model is None:
                    model = chunk.model
                last_usage = chunk.usage or last_usage
                if chunk.is_thinking:
                    thinking_parts.append(chunk.text)
                else:
                    text_parts.append(chunk.text)
                if chunk.logprob is not None:
                    logprobs_content.append(
                        LogprobsContentItem(
                            token=chunk.text,
                            logprob=chunk.logprob,
                            top_logprobs=chunk.top_logprobs or [],
                        )
                    )
                if chunk.finish_reason is not None:
                    finish_reason = chunk.finish_reason

            case ToolCallChunk():
                if model is None:
                    model = chunk.model
                last_usage = chunk.usage or last_usage
                tool_calls.extend(
                    ToolCall(
                        id=tool.id,
                        index=i,
                        function=tool,
                    )
                    for i, tool in enumerate(chunk.tool_calls)
                )
                finish_reason = chunk.finish_reason

    if error_message is not None:
        if error_code is not None:
            return VideoErrorResponse(
                error=VideoErrorInfo(
                    message=error_message,
                    code=error_code,
                )
            )
        return ErrorResponse(
            error=ErrorInfo(
                message=error_message,
                type="InternalServerError",
                code=500,
            )
        )

    combined_text = "".join(text_parts)
    combined_thinking = "".join(thinking_parts) if thinking_parts else None
    if model is None:
        return ErrorResponse(
            error=ErrorInfo(
                message="No response chunks were received from the model",
                type="InternalServerError",
                code=500,
            )
        )

    return ChatCompletionResponse(
        id=command_id,
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=combined_text,
                    reasoning_content=combined_thinking,
                    tool_calls=tool_calls if tool_calls else None,
                ),
                logprobs=Logprobs(content=logprobs_content)
                if logprobs_content
                else None,
                finish_reason=finish_reason,
            )
        ],
        usage=last_usage,
    )
