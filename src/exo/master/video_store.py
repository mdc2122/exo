import time
from pathlib import Path

from pydantic import BaseModel

from exo.shared.types.common import Id


class StoredVideo(BaseModel, frozen=True):
    video_id: Id
    file_path: Path
    content_type: str
    byte_count: int
    expires_at: float


class VideoStore:
    def __init__(self, storage_dir: Path, default_expiry_seconds: int = 3600) -> None:
        self._storage_dir = storage_dir
        self._default_expiry_seconds = default_expiry_seconds
        self._videos: dict[Id, StoredVideo] = {}
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def store(self, video_bytes: bytes, content_type: str) -> StoredVideo:
        video_id = Id()
        file_path = self._storage_dir / f"{video_id}.mp4"
        file_path.write_bytes(video_bytes)

        stored = StoredVideo(
            video_id=video_id,
            file_path=file_path,
            content_type=content_type,
            byte_count=len(video_bytes),
            expires_at=time.time() + self._default_expiry_seconds,
        )
        self._videos[video_id] = stored
        return stored

    def get(self, video_id: Id) -> StoredVideo | None:
        stored = self._videos.get(video_id)
        if stored is None:
            return None

        if time.time() > stored.expires_at:
            self._remove(video_id)
            return None

        return stored

    def cleanup_expired(self) -> int:
        now = time.time()
        expired_ids = [
            video_id
            for video_id, stored in self._videos.items()
            if now > stored.expires_at
        ]

        for video_id in expired_ids:
            self._remove(video_id)

        return len(expired_ids)

    def _remove(self, video_id: Id) -> None:
        stored = self._videos.pop(video_id, None)
        if stored is not None and stored.file_path.exists():
            stored.file_path.unlink()
