from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

from pytest import MonkeyPatch

from exo.worker.engines.mlx.generator import batch_generate as batch_generate_module

BatchCompatibleCacheFn: TypeAlias = Callable[[object], object]
CacheForBatchEngineFn: TypeAlias = Callable[[Sequence[object]], list[object]]


class FakeCacheList:
    def __init__(self, *caches: object) -> None:
        self.caches = caches


@dataclass
class FakeLayerCache:
    native_cache: object | None = None

    @property
    def _native_cache(self) -> object | None:
        return self.native_cache


def _batch_compatible_cache(layer_cache: object) -> object:
    helper = cast(
        BatchCompatibleCacheFn,
        batch_generate_module._batch_compatible_cache,  # pyright: ignore[reportPrivateUsage]
    )
    return helper(layer_cache)


def _cache_for_batch_engine(cache: Sequence[object]) -> list[object]:
    helper = cast(
        CacheForBatchEngineFn,
        batch_generate_module._cache_for_batch_engine,  # pyright: ignore[reportPrivateUsage]
    )
    return helper(cache)


class TestBatchCompatibleCache:
    def test_returns_original_cache_when_fallback_disabled(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.delenv("EXO_TURBOQUANT_NATIVE_BATCH_FALLBACK", raising=False)
        cache = FakeLayerCache(native_cache="native")

        result = _batch_compatible_cache(cache)

        assert result is cache

    def test_uses_native_cache_when_fallback_enabled(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXO_TURBOQUANT_NATIVE_BATCH_FALLBACK", "1")
        cache = FakeLayerCache(native_cache="native")

        result = _batch_compatible_cache(cache)

        assert result == "native"

    def test_recursively_unwraps_cache_lists_when_fallback_enabled(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXO_TURBOQUANT_NATIVE_BATCH_FALLBACK", "1")
        monkeypatch.setattr(batch_generate_module, "CacheList", FakeCacheList)
        nested = FakeCacheList(
            FakeLayerCache(native_cache="left"),
            FakeLayerCache(native_cache="right"),
        )

        result = _batch_compatible_cache(nested)

        assert isinstance(result, FakeCacheList)
        assert result.caches == ("left", "right")

    def test_preserves_cache_without_native_fallback(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXO_TURBOQUANT_NATIVE_BATCH_FALLBACK", "1")
        cache = FakeLayerCache(native_cache=None)

        result = _batch_compatible_cache(cache)

        assert result is cache


class TestCacheForBatchEngine:
    def test_converts_each_layer_cache(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("EXO_TURBOQUANT_NATIVE_BATCH_FALLBACK", "1")
        caches = [
            FakeLayerCache(native_cache="first"),
            FakeLayerCache(native_cache="second"),
        ]

        result = _cache_for_batch_engine(caches)

        assert result == ["first", "second"]
