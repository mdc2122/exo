# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportUnnecessaryTypeIgnoreComment=false
import base64
import contextlib
import hashlib
import importlib
import inspect
import io
import json
import os
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

import httpx

if TYPE_CHECKING:
    from mlx_vlm.utils import ImageProcessor
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_vlm.prompt_utils import get_message_json
from mlx_vlm.utils import load_image_processor
from PIL import Image
from safetensors import safe_open
from transformers import AutoImageProcessor

from exo.download.download_utils import build_model_path
from exo.shared.models.model_cards import VisionCardConfig
from exo.shared.types.common import ModelId
from exo.shared.types.mlx import Model
from exo.shared.types.text_generation import TextGenerationTaskParams, VideoSource
from exo.shared.types.video_errors import (
    VIDEO_ERROR_CODE_DECODE_FAILED,
    VIDEO_ERROR_CODE_DURATION_UNSUPPORTED,
    VIDEO_ERROR_CODE_FETCH_FAILED,
    VIDEO_ERROR_CODE_INVALID_VIDEO_URL,
    VIDEO_ERROR_CODE_TOO_LARGE,
    VIDEO_ERROR_CODE_UNSUPPORTED_FORMAT,
    VisionPreprocessingError,
)
from exo.worker.engines.mlx.cache import encode_prompt
from exo.worker.engines.mlx.utils_mlx import (
    fix_unmatched_think_end_tokens,
    render_chat_template,
)
from exo.worker.runner.bootstrap import logger


def _filter_config(cls: type, d: dict[str, Any]) -> dict[str, Any]:
    valid = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
    return {k: v for k, v in d.items() if k in valid}  # type: ignore


DEFAULT_MAX_VIDEO_PAYLOAD_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_VIDEO_DURATION_SECONDS = 10.0
DEFAULT_KIMI_VIDEO_FETCH_CONNECT_TIMEOUT_SECONDS = 5.0
DEFAULT_KIMI_VIDEO_FETCH_READ_TIMEOUT_SECONDS = 15.0
DEFAULT_KIMI_VIDEO_FETCH_TOTAL_TIMEOUT_SECONDS = 30.0
KIMI_VIDEO_FETCH_CHUNK_BYTES = 64 * 1024
DEFAULT_KIMI_VIDEO_ALLOWED_CODECS = ("h264",)
# PyAV/ffprobe report ISO BMFF MP4 as this comma-separated format family.
DEFAULT_KIMI_VIDEO_ALLOWED_CONTAINERS = ("mov", "mp4", "m4a", "3gp", "3g2", "mj2")


def _get_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _get_max_video_payload_bytes() -> int:
    raw = os.getenv("EXO_KIMI_VIDEO_MAX_BYTES")
    if raw is None or raw == "":
        raw = os.getenv("EXO_MAX_VIDEO_PAYLOAD_BYTES")
    if raw is None or raw == "":
        return DEFAULT_MAX_VIDEO_PAYLOAD_BYTES
    return int(raw)


def _get_max_video_duration_seconds() -> float:
    raw = os.getenv("EXO_KIMI_VIDEO_MAX_DURATION_SECONDS")
    if raw is None or raw == "":
        return DEFAULT_MAX_VIDEO_DURATION_SECONDS
    return float(raw)


def _get_csv_env_values(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    values = tuple(value.strip().lower() for value in raw.split(",") if value.strip())
    return values or default


def _split_container_names(format_name: str) -> tuple[str, ...]:
    return tuple(
        part.strip().lower() for part in format_name.split(",") if part.strip()
    )


def _probe_stream_codec_name(stream: object) -> str | None:
    codec_context = getattr(stream, "codec_context", None)
    codec_context_name = getattr(codec_context, "name", None)
    if isinstance(codec_context_name, str) and codec_context_name.strip():
        return codec_context_name.strip().lower()

    codec = getattr(stream, "codec", None)
    codec_name = getattr(codec, "name", None)
    if isinstance(codec_name, str) and codec_name.strip():
        return codec_name.strip().lower()

    direct_codec_name = getattr(stream, "codec_name", None)
    if isinstance(direct_codec_name, str) and direct_codec_name.strip():
        return direct_codec_name.strip().lower()

    return None


def _probe_container_names(container: object) -> tuple[str, ...]:
    container_format = getattr(container, "format", None)
    format_name = getattr(container_format, "name", None)
    if not isinstance(format_name, str):
        return ()
    return _split_container_names(format_name)


def _validate_video_metadata_before_decode(container: object, stream: object) -> None:
    """Fail closed on unsupported/unprobeable metadata before frame decode.

    V1 intentionally keeps support narrow to MP4/H.264 because broad codec and
    container coverage is out of scope and risky for the live MLX cluster. This
    inspection uses only demuxer metadata already available after `av.open()`;
    it does not decode frames or place large frame tensors into process state.
    """

    codec_name = _probe_stream_codec_name(stream)
    if codec_name is None:
        raise VisionPreprocessingError(
            "unprobeable video codec: the selected runner could not safely "
            "identify the video stream codec before decode. Kimi video_url v1 "
            "supports only bounded MP4/H.264 inputs.",
            code=VIDEO_ERROR_CODE_UNSUPPORTED_FORMAT,
        )

    allowed_codecs = _get_csv_env_values(
        "EXO_KIMI_VIDEO_ALLOWED_CODECS", DEFAULT_KIMI_VIDEO_ALLOWED_CODECS
    )
    if codec_name not in allowed_codecs:
        raise VisionPreprocessingError(
            f"unsupported video codec '{codec_name}' for Kimi video_url v1. "
            f"Supported codecs: {sorted(allowed_codecs)}.",
            code=VIDEO_ERROR_CODE_UNSUPPORTED_FORMAT,
        )

    container_names = _probe_container_names(container)
    if not container_names:
        raise VisionPreprocessingError(
            "unprobeable video container: the selected runner could not safely "
            "identify the video container before decode. Kimi video_url v1 "
            "supports only bounded MP4/H.264 inputs.",
            code=VIDEO_ERROR_CODE_UNSUPPORTED_FORMAT,
        )

    allowed_containers = _get_csv_env_values(
        "EXO_KIMI_VIDEO_ALLOWED_CONTAINERS", DEFAULT_KIMI_VIDEO_ALLOWED_CONTAINERS
    )
    if not any(
        container_name in allowed_containers for container_name in container_names
    ):
        raise VisionPreprocessingError(
            f"unsupported video container '{','.join(container_names)}' for Kimi "
            f"video_url v1. Supported containers: {sorted(allowed_containers)}.",
            code=VIDEO_ERROR_CODE_UNSUPPORTED_FORMAT,
        )


def _validate_video_duration_seconds(
    duration_seconds: float, max_seconds: float | None = None
) -> None:
    resolved_max_seconds = (
        _get_max_video_duration_seconds() if max_seconds is None else max_seconds
    )
    if duration_seconds <= resolved_max_seconds:
        return
    raise VisionPreprocessingError(
        "Long videos are out of scope for v1: "
        f"{duration_seconds:.2f}s exceeds the configured limit of "
        f"{resolved_max_seconds:.2f}s. Use a short video fixture or raise "
        "EXO_KIMI_VIDEO_MAX_DURATION_SECONDS after throughput smokes pass.",
        code=VIDEO_ERROR_CODE_DURATION_UNSUPPORTED,
    )


def _validate_video_payload_size(byte_count: int, max_bytes: int | None = None) -> None:
    resolved_max_bytes = (
        _get_max_video_payload_bytes() if max_bytes is None else max_bytes
    )
    if byte_count <= resolved_max_bytes:
        return
    actual_mib = byte_count / 1024 / 1024
    max_mib = resolved_max_bytes / 1024 / 1024
    # Kept as ValueError for backward-compat with existing tests; callers
    # that need a structured code wrap this in VisionPreprocessingError.
    raise ValueError(
        f"video payload is too large: {actual_mib:.2f} MiB exceeds "
        f"the configured limit of {max_mib:.2f} MiB"
    )


def _fetch_video_url_as_base64(url: str, max_bytes: int | None = None) -> str:
    resolved_max_bytes = (
        _get_max_video_payload_bytes() if max_bytes is None else max_bytes
    )
    parsed_url = urlsplit(url)
    if parsed_url.scheme not in {"http", "https"}:
        raise VisionPreprocessingError(
            "unsupported video URL scheme: only http and https video URLs "
            f"can be fetched by the worker, got {parsed_url.scheme!r}",
            code=VIDEO_ERROR_CODE_INVALID_VIDEO_URL,
        )
    if parsed_url.netloc == "":
        raise VisionPreprocessingError(
            f"invalid video URL {url!r}: missing network location",
            code=VIDEO_ERROR_CODE_INVALID_VIDEO_URL,
        )

    connect_timeout_seconds = _get_float_env(
        "EXO_KIMI_VIDEO_FETCH_CONNECT_TIMEOUT_SECONDS",
        DEFAULT_KIMI_VIDEO_FETCH_CONNECT_TIMEOUT_SECONDS,
    )
    read_timeout_seconds = _get_float_env(
        "EXO_KIMI_VIDEO_FETCH_READ_TIMEOUT_SECONDS",
        DEFAULT_KIMI_VIDEO_FETCH_READ_TIMEOUT_SECONDS,
    )
    total_timeout_seconds = _get_float_env(
        "EXO_KIMI_VIDEO_FETCH_TIMEOUT_SECONDS",
        DEFAULT_KIMI_VIDEO_FETCH_TOTAL_TIMEOUT_SECONDS,
    )
    http_timeout = httpx.Timeout(
        timeout=total_timeout_seconds,
        connect=connect_timeout_seconds,
        read=read_timeout_seconds,
        write=connect_timeout_seconds,
        pool=connect_timeout_seconds,
    )
    started_at = time.monotonic()
    data = bytearray()

    try:
        with (
            httpx.Client(timeout=http_timeout, follow_redirects=False) as client,
            client.stream("GET", url, headers={"User-Agent": "exo/1.0"}) as response,
        ):
            response.raise_for_status()
            content_length = cast(str | None, response.headers.get("Content-Length"))
            if content_length is not None:
                _validate_video_payload_size(
                    int(content_length), max_bytes=resolved_max_bytes
                )
            for chunk in response.iter_bytes(chunk_size=KIMI_VIDEO_FETCH_CHUNK_BYTES):
                if time.monotonic() - started_at > total_timeout_seconds:
                    raise VisionPreprocessingError(
                        f"timed out while fetching video URL {url!r} after "
                        f"{total_timeout_seconds:.2f}s",
                        code=VIDEO_ERROR_CODE_FETCH_FAILED,
                    )
                data.extend(chunk)
                _validate_video_payload_size(len(data), max_bytes=resolved_max_bytes)
    except VisionPreprocessingError:
        raise
    except httpx.HTTPError as exc:
        raise VisionPreprocessingError(
            f"failed to fetch video URL {url!r}: {exc}",
            code=VIDEO_ERROR_CODE_FETCH_FAILED,
        ) from exc

    return base64.b64encode(bytes(data)).decode("ascii")


def _wrap_fetch_video_url_for_structured_errors(url: str) -> str:
    """Call _fetch_video_url_as_base64 but translate raw exceptions into
    VisionPreprocessingError with stable codes. Network/IO failures map to
    video_fetch_failed; size-limit ValueErrors map to video_too_large.
    Anything else maps to video_fetch_failed as a safe default.
    """
    try:
        return _fetch_video_url_as_base64(url)
    except ValueError as exc:
        # _validate_video_payload_size raises ValueError when the response
        # exceeds the configured size limit.
        raise VisionPreprocessingError(
            f"video at {url} exceeds the configured size limit: {exc}",
            code=VIDEO_ERROR_CODE_TOO_LARGE,
        ) from exc
    except (httpx.HTTPError, OSError) as exc:
        raise VisionPreprocessingError(
            f"failed to fetch video URL {url!r}: {exc}",
            code=VIDEO_ERROR_CODE_FETCH_FAILED,
        ) from exc


_video_processor_patched = False


def _patch_video_processor() -> None:
    """Patch so we don't crash horribly when torch vision isn't installed"""
    # TODO: Update if we add torch vision.
    global _video_processor_patched
    if _video_processor_patched:
        return
    try:
        from transformers.processing_utils import MODALITY_TO_AUTOPROCESSOR_MAPPING

        mapping = MODALITY_TO_AUTOPROCESSOR_MAPPING._MAPPING_NAMES  # type: ignore
        mapping.pop("video_processor", None)
    except (ImportError, AttributeError):
        pass
    _video_processor_patched = True


def decode_base64_image(b64_data: str) -> Image.Image:
    raw = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(raw))
    return img.convert("RGB")


def extract_video_frames(
    video_data: str,
    sample_fps: float = 2.0,
    temporal_chunk_size: int = 4,
) -> list[list[Image.Image]]:
    import av

    raw = base64.b64decode(video_data)
    try:
        container = av.open(io.BytesIO(raw))
    except Exception as exc:
        raise VisionPreprocessingError(
            "unprobeable video container: the selected runner could not safely "
            "inspect video metadata before decode. Kimi video_url v1 supports "
            "only bounded MP4/H.264 inputs.",
            code=VIDEO_ERROR_CODE_UNSUPPORTED_FORMAT,
        ) from exc
    try:
        video_streams = list(getattr(container.streams, "video", []))
        if not video_streams:
            raise VisionPreprocessingError(
                "unprobeable video stream: no video stream was found during "
                "safe metadata inspection before decode. Kimi video_url v1 "
                "supports only bounded MP4/H.264 inputs.",
                code=VIDEO_ERROR_CODE_UNSUPPORTED_FORMAT,
            )
        stream = video_streams[0]
        _validate_video_metadata_before_decode(container, stream)
        duration_seconds: float | None = None
        if stream.duration is not None and stream.time_base is not None:
            duration_seconds = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration_seconds = float(container.duration) / 1_000_000.0
        if duration_seconds is not None:
            _validate_video_duration_seconds(duration_seconds)
        total_frames = stream.frames or 0
        video_fps = float(stream.average_rate or 30)

        effective_fps = min(sample_fps, video_fps)
        sampled_count = max(round(total_frames * effective_fps / video_fps), 1)
        indices = (
            np.linspace(0, total_frames - 1, sampled_count).round().astype(int).tolist()
        )
        index_set = set(indices)

        all_frames: list[Image.Image] = []
        for i, frame in enumerate(container.decode(video=0)):
            if i in index_set:
                all_frames.append(frame.to_image().convert("RGB"))
            if len(all_frames) >= len(indices):
                break
    finally:
        with contextlib.suppress(Exception):
            container.close()

    chunks: list[list[Image.Image]] = []
    for start in range(0, len(all_frames), temporal_chunk_size):
        chunk = all_frames[start : start + temporal_chunk_size]
        while len(chunk) < temporal_chunk_size:
            chunk.append(chunk[-1])
        chunks.append(chunk)

    return chunks


def preprocess_video_chunk(
    frames: list[Image.Image],
    patch_size: int = 14,
    merge_kernel_size: int = 2,
    in_patch_limit_each_frame: int = 4096,
) -> dict[str, np.ndarray]:
    import math

    frame_count = len(frames)
    w, h = frames[0].size

    factor = merge_kernel_size * patch_size
    s1 = math.sqrt(
        in_patch_limit_each_frame
        / (max(1.0, w // patch_size) * max(1.0, h // patch_size))
    )
    scale = min(1.0, s1)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    pad_w = (factor - new_w % factor) % factor
    pad_h = (factor - new_h % factor) % factor

    pixel_arrays = []
    for frame in frames:
        arr = np.array(
            frame.resize((new_w, new_h), Image.Resampling.BICUBIC), dtype=np.float32
        )
        if pad_h > 0 or pad_w > 0:
            arr = np.pad(
                arr,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        arr = (arr / 255.0 - 0.5) / 0.5
        pixel_arrays.append(arr)

    pixel_values = np.stack(pixel_arrays, axis=0)

    _, height, width, channels = pixel_values.shape
    patches = pixel_values.reshape(
        frame_count,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
        channels,
    )
    patches = patches.transpose(0, 1, 3, 5, 2, 4)
    patches = patches.reshape(-1, channels, patch_size, patch_size)
    grid_thw = np.array([frame_count, height // patch_size, width // patch_size])

    return {"pixel_values": patches, "grid_thw": grid_thw}


def preprocess_kimi_images(
    images: list[Image.Image],
    patch_size: int = 14,
    merge_kernel_size: int = 2,
    in_patch_limit_each_frame: int = 4096,
) -> dict[str, np.ndarray]:
    pixel_values_parts: list[np.ndarray] = []
    grid_thw_parts: list[np.ndarray] = []

    for image in images:
        processed = preprocess_video_chunk(
            [image.convert("RGB")],
            patch_size=patch_size,
            merge_kernel_size=merge_kernel_size,
            in_patch_limit_each_frame=in_patch_limit_each_frame,
        )
        pixel_values_parts.append(processed["pixel_values"])
        grid_thw_parts.append(processed["grid_thw"])

    return {
        "pixel_values": np.concatenate(pixel_values_parts, axis=0),
        "grid_thw": np.stack(grid_thw_parts, axis=0),
    }


def make_video_chunk_prompts(
    num_chunks: int,
    fps: float,
    chunk_size: int,
) -> list[str]:
    prompts = []
    for i in range(num_chunks):
        timestamp_sec = i * chunk_size / fps
        hours = int(timestamp_sec // 3600)
        minutes = int((timestamp_sec % 3600) // 60)
        seconds = int(timestamp_sec % 60)
        millis = int((timestamp_sec % 1) * 1000)
        ts = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
        prompts.append(
            f"{ts}<|media_begin|>video<|media_content|><|media_pad|><|media_end|>"
        )
    return prompts


def expand_video_placeholder(
    text: str,
    chunk_prompts: list[str],
    image_token: str | None = None,
    n_tokens_per_chunk: list[int] | None = None,
) -> str:
    placeholder = "<|kimi_k25_video_placeholder|>"
    if placeholder not in text:
        return text

    expanded_chunks: list[str] = []
    for i, chunk_prompt in enumerate(chunk_prompts):
        n_tokens = (
            n_tokens_per_chunk[i]
            if n_tokens_per_chunk is not None and i < len(n_tokens_per_chunk)
            else 1
        )
        if image_token is not None and n_tokens > 1:
            chunk_prompt = chunk_prompt.replace(image_token, image_token * n_tokens, 1)
        expanded_chunks.append(chunk_prompt)

    replacement = "".join(expanded_chunks)
    return text.replace(placeholder, replacement, 1)


def _format_vlm_messages(
    messages: list[dict[str, Any]],
    model_type: str,
) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for msg in messages:
        role: str = str(msg.get("role", "user"))  # type: ignore
        content: Any = msg.get("content")
        if not isinstance(content, list):
            formatted.append(msg)
            continue
        parts: list[dict[str, Any]] = content  # type: ignore
        text_parts = [str(p["text"]) for p in parts if p.get("type") == "text"]  # type: ignore
        n_images = sum(1 for p in parts if p.get("type") in ("image", "image_url"))
        n_videos = sum(1 for p in parts if p.get("type") in ("video", "video_url"))
        text_with_placeholders = " ".join(text_parts)
        if n_videos > 0:
            text_with_placeholders += " " + " ".join(
                "<|kimi_k25_video_placeholder|>" for _ in range(n_videos)
            )
        result: dict[str, Any] = get_message_json(
            model_type, text_with_placeholders, role, num_images=n_images
        )
        formatted.append(result)
    return formatted


def build_vision_prompt(
    tokenizer: TokenizerWrapper,
    chat_template_messages: list[dict[str, Any]],
    n_tokens_per_image: list[int],
    image_token: str,
    task_params: TextGenerationTaskParams,
) -> str:
    prompt = render_chat_template(tokenizer, chat_template_messages, task_params)

    image_idx = 0
    result: list[str] = []
    i = 0
    pad_len = len(image_token)
    while i < len(prompt):
        if prompt[i : i + pad_len] == image_token:
            n = (
                n_tokens_per_image[image_idx]
                if image_idx < len(n_tokens_per_image)
                else 1
            )
            result.append(image_token * n)
            image_idx += 1
            i += pad_len
        else:
            result.append(prompt[i])
            i += 1

    return "".join(result)


@dataclass
class MediaRegion:
    content_hash: str
    start_pos: int
    end_pos: int


@dataclass
class VisionResult:
    prompt: str
    prompt_tokens: mx.array
    embeddings: mx.array
    media_regions: list[MediaRegion]


class VisionEncoder:
    def __init__(self, config: VisionCardConfig, model_id: ModelId):
        self._config = config
        self._main_model_path = build_model_path(model_id)
        self._model_path = build_model_path(ModelId(config.weights_repo))
        self._vision_tower: nn.Module | None = None
        self._projector: nn.Module | None = None
        self._processor: "ImageProcessor | None" = None
        self._spatial_merge_size: int = 2
        self._merge_kernel_size: list[int] | None = None
        self._needs_nhwc: bool = False
        self._loaded = False

    def _load_config_json(self) -> dict[str, Any]:
        for candidate in (self._main_model_path, self._model_path):
            path = candidate / "config.json"
            if path.exists():
                with open(path) as f:
                    return json.load(f)  # type: ignore
        return {}

    def _import_mlx_vlm(self, *submodules: str) -> Any:  # type: ignore
        mt = self._config.model_type
        results: list[Any] = []
        for sub in submodules:
            name = f"mlx_vlm.models.{mt}.{sub}"
            results.append(importlib.import_module(name))
        return results[0] if len(results) == 1 else tuple(results)

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._load_weights()
        self._loaded = True

    def _load_weights(self) -> None:
        _patch_video_processor()
        logger.info(f"Loading vision weights from {self._model_path}")
        config = self._load_config_json()
        if not config:
            raise FileNotFoundError(f"config.json not found in {self._model_path}")

        vision_cfg = config.get("vision_config", {})  # type: ignore

        config_mod, vision_mod = self._import_mlx_vlm("config", "vision")  # type: ignore
        vision_config_cls = config_mod.VisionConfig  # type: ignore
        vision_model_cls = vision_mod.VisionModel  # type: ignore

        vision_config = vision_config_cls(  # type: ignore
            **_filter_config(vision_config_cls, vision_cfg)  # type: ignore
        )
        self._spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2)  # type: ignore
        self._vision_tower = vision_model_cls(vision_config)
        model_mod: Any = None
        with contextlib.suppress(ImportError):
            model_mod = self._import_mlx_vlm(self._config.model_type)  # type: ignore

        projector_cls = None
        if model_mod is not None:
            for attr_name in dir(model_mod):  # type: ignore
                obj = getattr(model_mod, attr_name)  # type: ignore
                if (
                    isinstance(obj, type)
                    and issubclass(obj, nn.Module)
                    and "Projector" in attr_name
                ):
                    projector_cls = obj
                    break

        if projector_cls is not None:
            text_config = config_mod.TextConfig(  # type: ignore
                **_filter_config(config_mod.TextConfig, config.get("text_config", {}))  # type: ignore
            )
            extra = {
                k: v
                for k, v in config.items()  # type: ignore
                if k not in ("text_config", "vision_config")
            }
            extra.setdefault("model_type", self._config.model_type)
            model_config = config_mod.ModelConfig(  # type: ignore
                text_config=text_config,
                vision_config=vision_config,
                **_filter_config(config_mod.ModelConfig, extra),  # type: ignore
            )
            self._projector = projector_cls(model_config)  # type: ignore

        processor_repo = self._config.processor_repo
        if processor_repo:
            self._load_weights_from_separate_repo()
        else:
            self._load_weights_from_model_repo()

        repo = processor_repo or str(self._model_path)
        if processor_repo:
            self._merge_kernel_size = vision_cfg.get("merge_kernel_size", [2, 2])  # type: ignore
            self._needs_nhwc = True
            logger.info(
                "Skipping HF image processor load for kimi_vl; using local preprocess"
            )
        else:
            image_proc = load_image_processor(repo)
            if image_proc is not None:
                self._processor = image_proc
            else:
                self._processor = AutoImageProcessor.from_pretrained(  # type: ignore
                    repo, trust_remote_code=True
                )
            logger.info(f"HF image processor loaded from {repo}")

    def _load_weights_from_separate_repo(self) -> None:
        safetensors_files = list(self._model_path.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {self._model_path}")

        weights: dict[str, mx.array] = {}
        for sf_path in safetensors_files:
            with safe_open(str(sf_path), framework="pt") as f:
                keys = f.keys()
                for key in keys:
                    tensor = f.get_tensor(key)  # type: ignore
                    np_tensor = tensor.float().numpy()  # type: ignore
                    weights[key] = mx.array(np_tensor, dtype=mx.bfloat16)  # type: ignore

        vision_weights: dict[str, mx.array] = {}
        projector_weights: dict[str, mx.array] = {}
        for key, val in weights.items():
            if key.startswith("vision_tower."):
                short_key = key[len("vision_tower.") :]
                if short_key.startswith("encoder."):
                    short_key = short_key[len("encoder.") :]
                m = re.match(r"^(blocks\.\d+)\.(wqkv|wo)\.(weight|bias)$", short_key)
                if m:
                    short_key = f"{m.group(1)}.attn.{m.group(2)}.{m.group(3)}"
                if short_key == "patch_embed.proj.weight" and val.ndim == 4:
                    val = val.transpose(0, 2, 3, 1)
                vision_weights[short_key] = val
            elif key.startswith(("mm_projector.", "multi_modal_projector.")):
                if key.startswith("multi_modal_projector."):
                    short_key = key[len("multi_modal_projector.") :]
                    if short_key.startswith("mm_projector."):
                        short_key = short_key[len("mm_projector.") :]
                else:
                    short_key = key[len("mm_projector.") :]
                short_key = short_key.replace("proj.0.", "linear_1.").replace(
                    "proj.2.", "linear_2."
                )
                projector_weights[short_key] = val

        assert self._vision_tower is not None
        self._vision_tower.load_weights(list(vision_weights.items()))
        mx.eval(self._vision_tower.parameters())

        if self._projector is not None and projector_weights:
            self._projector.load_weights(list(projector_weights.items()))
            mx.eval(self._projector.parameters())

        n_vision = sum(v.size for _, v in vision_weights.items())
        n_proj = sum(v.size for _, v in projector_weights.items())
        logger.info(
            f"Vision encoder loaded: {n_vision / 1e6:.1f}M params"
            + (f", projector: {n_proj / 1e6:.1f}M params" if n_proj else "")
        )

    def _load_weights_from_model_repo(self) -> None:
        safetensors_files = sorted(self._model_path.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {self._model_path}")

        vision_prefixes = ["vision_tower.", "model.visual."]
        vision_weights: dict[str, mx.array] = {}
        found_raw_prefix = False
        for sf_path in safetensors_files:
            file_weights: dict[str, mx.array] = mx.load(str(sf_path))  # type: ignore
            for key, val in file_weights.items():
                for prefix in vision_prefixes:
                    if key.startswith(prefix):
                        short_key = key[len(prefix) :]
                        vision_weights[short_key] = val
                        if prefix == "model.visual.":
                            found_raw_prefix = True
                        break

        if not vision_weights:
            raise ValueError(
                f"No vision weights found with prefixes {vision_prefixes} in {self._model_path}. "
                "Ensure the model repo contains bundled vision weights."
            )

        assert self._vision_tower is not None
        if found_raw_prefix and hasattr(self._vision_tower, "sanitize"):
            vision_weights = self._vision_tower.sanitize(vision_weights)  # type: ignore

        self._vision_tower.load_weights(list(vision_weights.items()))  # type: ignore
        mx.eval(self._vision_tower.parameters())

        n_vision = sum(v.size for _, v in vision_weights.items())  # type: ignore
        logger.info(f"Vision encoder loaded: {n_vision / 1e6:.1f}M params")

    def _flatten_feature_batch(self, feature_batch: mx.array) -> mx.array:
        if feature_batch.ndim <= 2:
            return feature_batch
        return feature_batch.reshape(-1, feature_batch.shape[-1])

    def _flatten_feature_batches(self, feature_batches: list[mx.array]) -> mx.array:
        flattened = [self._flatten_feature_batch(batch) for batch in feature_batches]
        if not flattened:
            raise ValueError("Vision tower returned no image features")
        if len(flattened) == 1:
            return flattened[0]
        return mx.concatenate(flattened, axis=0)

    def _validate_projected_batch_sizes(
        self,
        feature_batches: list[mx.array],
        n_tokens_per_image: list[int],
    ) -> None:
        if len(feature_batches) != len(n_tokens_per_image):
            raise ValueError(
                "Projected image batch count does not match expected token groups: "
                f"{len(feature_batches)} != {len(n_tokens_per_image)}"
            )

        for index, (feature_batch, expected_tokens) in enumerate(
            zip(feature_batches, n_tokens_per_image, strict=True)
        ):
            actual_tokens = int(feature_batch.shape[0])
            if actual_tokens != expected_tokens:
                raise ValueError(
                    "Projected image feature count does not match grid_thw-derived token count "
                    f"for image {index}: {actual_tokens} != {expected_tokens}"
                )

    def _project_image_hidden_states(
        self,
        hidden_states: mx.array | list[mx.array],
        n_tokens_per_image: list[int] | None = None,
    ) -> mx.array:
        if isinstance(hidden_states, list):
            logger.info(
                "encode_images: projector start hidden_state_batches={} first_shape={}",
                len(hidden_states),
                hidden_states[0].shape if hidden_states else None,
            )
            if self._projector is not None:
                projected_batches = [self._projector(batch) for batch in hidden_states]
                if n_tokens_per_image is not None:
                    self._validate_projected_batch_sizes(
                        projected_batches, n_tokens_per_image
                    )
                return self._flatten_feature_batches(projected_batches)
            if n_tokens_per_image is not None:
                raise ValueError(
                    "Kimi vision image path requires a projector to align merged vision features"
                )
            return self._flatten_feature_batches(hidden_states)

        if self._projector is not None:
            logger.info(
                "encode_images: projector start hidden_states_shape={}",
                hidden_states.shape,
            )
            return self._flatten_feature_batch(self._projector(hidden_states))

        return self._flatten_feature_batch(hidden_states)

    def encode_images(self, images: list[str]) -> tuple[mx.array, list[int]]:
        self.ensure_loaded()
        assert self._vision_tower is not None

        logger.info("encode_images: decoding {} image(s)", len(images))
        pil_images = [decode_base64_image(b64) for b64 in images]
        for idx, img in enumerate(pil_images):
            logger.info(f"Image {idx}: {img.width}x{img.height} mode={img.mode}")

        if self._config.processor_repo:
            logger.info("encode_images: local kimi preprocess start")
            processed = preprocess_kimi_images(
                pil_images,
                merge_kernel_size=self._merge_kernel_size[0]
                if self._merge_kernel_size is not None
                else 2,
                in_patch_limit_each_frame=self._config.in_patch_limit_each_frame,
            )
            logger.info("encode_images: local kimi preprocess done")
            pixel_values = mx.array(processed["pixel_values"])
            grid_thw = mx.array(processed["grid_thw"])
            assert self._merge_kernel_size is not None
            merge_length = int(np.prod(self._merge_kernel_size))
            n_tokens_per_image = [
                int(mx.prod(grid_thw[i]).item()) // merge_length
                for i in range(grid_thw.shape[0])
            ]
        else:
            assert self._processor is not None
            logger.info("encode_images: processor call start")
            processed = self._processor(
                images=pil_images,
                return_tensors="np",
            )
            logger.info("encode_images: processor call done")
            pixel_values = mx.array(processed["pixel_values"])  # type: ignore
            grid_thw = mx.array(processed["image_grid_thw"])  # type: ignore
            merge_unit = self._spatial_merge_size**2
            n_tokens_per_image = [
                int(
                    grid_thw[i, 0].item()
                    * grid_thw[i, 1].item()
                    * grid_thw[i, 2].item()
                )
                // merge_unit
                for i in range(grid_thw.shape[0])
            ]

        if self._needs_nhwc:
            logger.info(
                "encode_images: vision tower forward start needs_nhwc=True pixel_values_shape={} grid_thw_shape={}",
                pixel_values.shape,
                grid_thw.shape,
            )
            grid_hw = grid_thw[:, 1:] if grid_thw.shape[-1] == 3 else grid_thw
            result = self._vision_tower(
                pixel_values.transpose(0, 2, 3, 1),
                output_hidden_states=True,
                grid_thw=grid_hw,
            )
            logger.info("encode_images: vision tower forward done")
            hidden_states = result[0] if isinstance(result, tuple) else result
        else:
            logger.info(
                "encode_images: vision tower forward start needs_nhwc=False pixel_values_shape={} grid_thw_shape={}",
                pixel_values.shape,
                grid_thw.shape,
            )
            result = self._vision_tower(pixel_values, grid_thw)
            logger.info("encode_images: vision tower forward done")
            hidden_states = result[0] if isinstance(result, tuple) else result

        image_features = self._project_image_hidden_states(
            hidden_states, n_tokens_per_image
        )
        if image_features.ndim != 2:
            raise ValueError(
                f"Expected 2D image features after projection, got {image_features.shape}"
            )
        logger.info(
            "encode_images: projector done image_features_shape={}",
            image_features.shape,
        )

        return image_features, n_tokens_per_image

    def encode_video_chunk(
        self, pixel_values: mx.array, grid_thw: mx.array
    ) -> tuple[mx.array, list[int]]:
        self.ensure_loaded()
        assert self._vision_tower is not None

        from exo.worker.engines.mlx.kimi_vl_temporal import (
            VisionModel as TemporalVisionModel,
        )

        if not hasattr(self, "_temporal_tower"):
            self._temporal_tower = TemporalVisionModel(self._vision_tower.config)  # type: ignore
            base_weights = list(tree_flatten(self._vision_tower.parameters()))
            self._temporal_tower.load_weights(base_weights, strict=False)
            mx.eval(self._temporal_tower.parameters())

        if self._needs_nhwc:
            pixel_values = pixel_values.transpose(0, 2, 3, 1)

        hidden_states_list: list[mx.array] = self._temporal_tower(
            pixel_values, grid_thw
        )

        if self._projector is not None:
            features_list = [self._projector(hs) for hs in hidden_states_list]
        else:
            features_list = hidden_states_list

        all_features = mx.concatenate(
            [f.reshape(-1, f.shape[-1]) for f in features_list], axis=0
        )
        n_tokens_per_chunk = [
            f.shape[0] * f.shape[1] if f.ndim == 3 else f.shape[0]
            for f in features_list
        ]

        return all_features, n_tokens_per_chunk


def get_inner_model(model: nn.Module) -> Any:  # type: ignore
    for candidate in (
        getattr(model, "model", None),
        getattr(getattr(model, "language_model", None), "model", None),
    ):
        if candidate is not None and hasattr(candidate, "embed_tokens"):  # type: ignore
            return candidate  # type: ignore

    raise ValueError(
        f"Could not find inner transformer (embed_tokens) in {type(model).__name__}. "
        "Add a new pattern to _get_inner_model() for this architecture."
    )


def create_vision_embeddings(
    model: Model,
    prompt_tokens: mx.array,
    image_features: mx.array,
    image_token_id: int,
) -> mx.array:
    inner = get_inner_model(model)  # type: ignore
    embed_tokens = inner.embed_tokens  # type: ignore

    input_embeddings: mx.array = embed_tokens(prompt_tokens[None])  # type: ignore

    is_image: mx.array = mx.equal(prompt_tokens, image_token_id)
    n_placeholders = int(mx.sum(is_image).item())

    if n_placeholders > 0:
        if n_placeholders != image_features.shape[0]:
            logger.warning(
                f"Placeholder count ({n_placeholders}) != image features "
                f"({image_features.shape[0]}). Using min of both."
            )
            n = min(n_placeholders, image_features.shape[0])
            image_features = image_features[:n]

        image_indices = mx.cumsum(is_image.astype(mx.int32)) - 1
        image_indices = mx.clip(image_indices, 0, image_features.shape[0] - 1)

        gathered = image_features[image_indices].astype(input_embeddings.dtype)
        result = mx.where(is_image[:, None], gathered, input_embeddings[0])
        input_embeddings = result[None]

    logger.info(
        "create_vision_embeddings: prompt_tokens_shape={} image_features_shape={} output_shape={} placeholders={}",
        prompt_tokens.shape,
        image_features.shape,
        input_embeddings.shape,
        n_placeholders,
    )

    return input_embeddings


def _build_media_region_payloads(
    images: list[str],
    videos: list[str],
    video_chunk_counts: list[int],
) -> list[str]:
    """Return payloads aligned to contiguous image-token regions in the prompt.

    Image inputs produce one media-token region each. Kimi video inputs expand one
    logical video placeholder into one timestamped media-token region per
    extracted chunk, so the same video payload must be repeated once per chunk
    for prefix-cache content hashing.
    """
    payloads = list(images)
    for index, video_payload in enumerate(videos):
        chunk_count = (
            video_chunk_counts[index] if index < len(video_chunk_counts) else 1
        )
        payloads.extend([video_payload] * chunk_count)
    return payloads


def _find_media_regions(
    prompt_tokens: mx.array,
    images: list[str],
    image_token_id: int,
) -> list[MediaRegion]:
    tokens_np = np.array(prompt_tokens)
    is_pad = tokens_np == image_token_id  # type: ignore

    regions: list[MediaRegion] = []
    in_run = False
    run_start = 0
    for pos, pad in enumerate(is_pad):  # type: ignore
        if pad and not in_run:
            run_start = pos
            in_run = True
        elif not pad and in_run:
            regions.append(
                MediaRegion(content_hash="", start_pos=run_start, end_pos=pos)
            )
            in_run = False
    if in_run:
        regions.append(
            MediaRegion(content_hash="", start_pos=run_start, end_pos=len(tokens_np))
        )

    for i, region in enumerate(regions):
        if i < len(images):
            payload = images[i]
            with contextlib.suppress(Exception):
                img = decode_base64_image(payload)
                region.content_hash = hashlib.sha256(img.tobytes()).hexdigest()
                continue
            region.content_hash = hashlib.sha256(base64.b64decode(payload)).hexdigest()
        else:
            logger.warning(f"Media region {i} has no corresponding image")

    return regions


def _resolve_video_payloads_for_preprocessing(
    videos: list[str],
    video_sources: list[VideoSource],
    video_urls: list[str],
) -> list[str]:
    """Resolve local base64 sources and remote URLs into Kimi video payloads.

    Normalized base64 VideoSource payloads are already validated by the API
    adapter and should flow directly into local Kimi frame extraction. HTTP(S)
    video_url values intentionally keep the existing worker-side fetch wrapper
    so the remote URL path remains unchanged.
    """
    resolved_videos = [source.as_pipeline_payload() for source in video_sources]
    resolved_videos.extend(videos)
    for video_url in video_urls:
        logger.debug("Fetching Kimi video URL on worker")
        resolved_videos.append(_wrap_fetch_video_url_for_structured_errors(video_url))
    return resolved_videos


class VisionProcessor:
    """
    Pipeline for vision models:
    1. Encode images into features (or grab from cache)
    2. Replace image placeholders with the features
    3. Build vision prompt
    4. Provide media regions for prefix caching
    """

    def __init__(self, config: VisionCardConfig, model_id: ModelId):
        self.vision_config = config
        self._encoder = VisionEncoder(config, model_id)
        self._feature_cache: dict[str, tuple[mx.array, list[int]]] = {}
        self._feature_cache_max = 32

    def load(self) -> None:
        self._encoder.ensure_loaded()

    def _image_cache_key(self, images: list[str]) -> str:
        h = hashlib.sha256()
        for img in images:
            h.update(img.encode("ascii"))
        return h.hexdigest()

    def process(
        self,
        images: list[str],
        chat_template_messages: list[dict[str, Any]],
        tokenizer: TokenizerWrapper,
        model: Model,
        task_params: TextGenerationTaskParams,
        videos: list[str] | None = None,
        video_sources: list[VideoSource] | None = None,
        video_urls: list[str] | None = None,
    ) -> VisionResult:
        videos = videos or []
        video_sources = video_sources or []
        video_urls = video_urls or []
        logger.info(
            f"Vision pipeline: {len(images)} image(s), "
            f"{len(videos)} inline video(s), {len(video_sources)} normalized video source(s), "
            f"{len(video_urls)} video URL(s)"
        )

        all_features_parts: list[mx.array] = []
        all_n_tokens: list[int] = []

        if images:
            cache_key = self._image_cache_key(images)
            cached = self._feature_cache.pop(cache_key, None)
            if cached is not None:
                self._feature_cache[cache_key] = cached
                image_features, n_tokens_per_image = cached
            else:
                image_features, n_tokens_per_image = self._encoder.encode_images(images)
                self._feature_cache[cache_key] = (image_features, n_tokens_per_image)
                while len(self._feature_cache) > self._feature_cache_max:
                    del self._feature_cache[next(iter(self._feature_cache))]
            all_features_parts.append(image_features)
            all_n_tokens.extend(n_tokens_per_image)
            logger.info(
                f"Image features: {image_features.shape} "
                f"({image_features.shape[0]} tokens, per-image: {n_tokens_per_image})"
            )

        video_chunk_prompts: list[str] = []
        video_chunk_token_counts: list[int] = []
        video_chunk_counts: list[int] = []
        resolved_videos = _resolve_video_payloads_for_preprocessing(
            videos=videos,
            video_sources=video_sources,
            video_urls=video_urls,
        )
        if resolved_videos:
            sample_fps = self.vision_config.sample_fps
            chunk_size = self.vision_config.temporal_merge_kernel_size
            for video_b64 in resolved_videos:
                try:
                    chunks = extract_video_frames(
                        video_b64,
                        sample_fps=sample_fps,
                        temporal_chunk_size=chunk_size,
                    )
                except VisionPreprocessingError:
                    raise
                except Exception as exc:
                    raise VisionPreprocessingError(
                        f"failed to decode video: {exc}",
                        code=VIDEO_ERROR_CODE_DECODE_FAILED,
                    ) from exc
                video_chunk_counts.append(len(chunks))
                logger.info(
                    f"Video extracted: {len(chunks)} chunks of {chunk_size} frames"
                )
                prompts = make_video_chunk_prompts(len(chunks), sample_fps, chunk_size)
                video_chunk_prompts.extend(prompts)
                for chunk in chunks:
                    preprocessed = preprocess_video_chunk(
                        chunk,
                        in_patch_limit_each_frame=self.vision_config.in_patch_limit_each_frame,
                    )
                    pixel_values = mx.array(preprocessed["pixel_values"])
                    grid_thw = mx.array(preprocessed["grid_thw"]).reshape(1, 3)
                    chunk_features, chunk_n_tokens = self._encoder.encode_video_chunk(
                        pixel_values, grid_thw
                    )
                    all_features_parts.append(chunk_features)
                    all_n_tokens.extend(chunk_n_tokens)
                    video_chunk_token_counts.extend(chunk_n_tokens)

        if all_features_parts:
            combined_features = mx.concatenate(all_features_parts, axis=0)
        else:
            combined_features = mx.zeros((0, 1))

        image_token = self.vision_config.image_token
        if image_token is None:
            image_token = tokenizer.decode([self.vision_config.image_token_id])

        formatted_messages = _format_vlm_messages(
            chat_template_messages, self.vision_config.model_type
        )

        prompt = build_vision_prompt(
            tokenizer,
            formatted_messages,
            all_n_tokens,
            image_token,
            task_params,
        )

        if video_chunk_prompts:
            prompt = expand_video_placeholder(
                prompt,
                video_chunk_prompts,
                image_token=image_token,
                n_tokens_per_chunk=video_chunk_token_counts,
            )

        logger.info(
            f"Expanded prompt has {prompt.count(image_token)} media_token occurrences, total len={len(prompt)}"
        )

        prompt_tokens: mx.array = encode_prompt(tokenizer, prompt)
        prompt_tokens = fix_unmatched_think_end_tokens(prompt_tokens, tokenizer)
        n_media_tokens = int(
            mx.sum(mx.equal(prompt_tokens, self.vision_config.image_token_id)).item()
        )
        logger.info(
            f"Encoded prompt: {len(prompt_tokens)} tokens, {n_media_tokens} media pad tokens"
        )

        embeddings = create_vision_embeddings(
            model,
            prompt_tokens,
            combined_features,
            self.vision_config.image_token_id,
        )
        mx.eval(embeddings)

        media_payloads = _build_media_region_payloads(
            images,
            resolved_videos,
            video_chunk_counts,
        )
        media_regions = _find_media_regions(
            prompt_tokens,
            media_payloads,
            self.vision_config.image_token_id,
        )

        return VisionResult(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            embeddings=embeddings,
            media_regions=media_regions,
        )


def prepare_vision(
    images: list[str] | None,
    chat_template_messages: list[dict[str, Any]] | None,
    vision_processor: VisionProcessor,
    tokenizer: TokenizerWrapper,
    model: Model,
    model_id: ModelId,
    task_params: TextGenerationTaskParams,
    videos: list[str] | None = None,
    video_sources: list[VideoSource] | None = None,
    video_urls: list[str] | None = None,
) -> VisionResult | None:
    if not images and not videos and not video_sources and not video_urls:
        return None
    if chat_template_messages is None:
        logger.warning("Vision request missing chat_template_messages — ignoring media")
        return None

    return vision_processor.process(
        images=images or [],
        chat_template_messages=chat_template_messages,
        tokenizer=tokenizer,
        model=model,
        task_params=task_params,
        videos=videos or [],
        video_sources=video_sources or [],
        video_urls=video_urls or [],
    )
