"""Shared structured error codes and exception types for video chat
requests. The API adapter and the worker both need access to the same
stable string codes so worker-side preprocessing failures can surface
through ErrorChunk and end up in an OpenAI-style envelope on the wire.

Keeping this in shared/types avoids a worker→api dependency cycle.
"""

from typing import Final

# Stable OpenAI-style structured error codes. These are slugs, not HTTP
# status codes, so clients can branch on them without parsing prose.
VIDEO_ERROR_CODE_TOO_LARGE: Final[str] = "video_too_large"
VIDEO_ERROR_CODE_URL_BLOCKED: Final[str] = "video_url_blocked"
VIDEO_ERROR_CODE_UNSUPPORTED_FORMAT: Final[str] = "unsupported_video_format"
VIDEO_ERROR_CODE_FETCH_FAILED: Final[str] = "video_fetch_failed"
VIDEO_ERROR_CODE_DECODE_FAILED: Final[str] = "video_decode_failed"
VIDEO_ERROR_CODE_MULTIPLE_VIDEOS: Final[str] = "multiple_videos_unsupported"
VIDEO_ERROR_CODE_INVALID_VIDEO_URL: Final[str] = "invalid_video_url"
VIDEO_ERROR_CODE_STREAMING_UNSUPPORTED: Final[str] = "video_streaming_unsupported"
VIDEO_ERROR_CODE_DURATION_UNSUPPORTED: Final[str] = "video_duration_unsupported"


class VisionPreprocessingError(Exception):
    """Raised when worker-side vision preprocessing fails in a way that
    must surface to the API client as a structured error rather than a
    silent text-only fallback. Carries a stable string `code` so the API
    adapter can populate the OpenAI-style envelope.
    """

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code: str = code
