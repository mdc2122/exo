# pyright: reportAny=false, reportPrivateUsage=false
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api.adapters.chat_completions import get_max_video_payload_bytes
from exo.api.main import API
from exo.master.video_store import VideoStore
from exo.shared.constants import EXO_VIDEO_CACHE_DIR
from exo.shared.types.video_errors import VIDEO_ERROR_CODE_TOO_LARGE


def _client(tmp_path: Path, monkeypatch: Any) -> TestClient:
    monkeypatch.setattr("exo.api.main.EXO_VIDEO_CACHE_DIR", tmp_path)
    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api.port = 52415
    api._video_store = VideoStore(tmp_path)
    api._setup_exception_handlers()
    api._setup_routes()
    return TestClient(app)


def test_default_video_guard_allows_100_mib(monkeypatch: Any) -> None:
    monkeypatch.delenv("EXO_KIMI_VIDEO_MAX_BYTES", raising=False)
    monkeypatch.delenv("EXO_MAX_VIDEO_PAYLOAD_BYTES", raising=False)

    assert get_max_video_payload_bytes() == 100 * 1024 * 1024


def test_video_upload_returns_public_reachable_url(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setenv("EXO_MEDIA_PUBLIC_BASE_URL", "http://100.93.190.120:52415")
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/v1/videos",
        files={"file": ("clip.mp4", b"fake mp4 bytes", "video/mp4")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["content_type"] == "video/mp4"
    assert data["bytes"] == len(b"fake mp4 bytes")
    assert data["url"].startswith("http://100.93.190.120:52415/v1/videos/")

    served = client.get(data["url"].replace("http://100.93.190.120:52415", ""))
    assert served.status_code == 200
    assert served.content == b"fake mp4 bytes"
    assert served.headers["content-type"].startswith("video/mp4")


def test_video_upload_rejects_non_mp4(tmp_path: Path, monkeypatch: Any) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/v1/videos",
        files={"file": ("clip.mov", b"fake mov bytes", "video/quicktime")},
    )

    assert response.status_code == 400
    data = response.json()
    assert data["error"]["message"] == "Only H.264/MP4 uploads are supported."
    assert data["error"]["param"] == "file"


def test_video_upload_uses_configured_size_guard(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setenv("EXO_KIMI_VIDEO_MAX_BYTES", "5")
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/v1/videos",
        files={"file": ("clip.mp4", b"123456", "video/mp4")},
    )

    assert response.status_code == 413
    data = response.json()
    assert data["error"]["code"] == VIDEO_ERROR_CODE_TOO_LARGE
    assert data["error"]["param"] == "file"


def test_video_cache_dir_is_separate_from_images() -> None:
    assert EXO_VIDEO_CACHE_DIR.name == "videos"
