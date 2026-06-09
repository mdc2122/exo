from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx

from exo.worker.engines.mlx.mimo_mtp_fast.mtp_generate import (
    _forward_hidden_and_logits,  # pyright: ignore[reportPrivateUsage]
)


@dataclass
class _RecordingCache:
    committed_token_ids: list[int] = field(default_factory=list)


class _TargetModelWithSeparateTopLevelCall:
    def __init__(self) -> None:
        self.top_level_calls: list[int] = []
        self.inner_calls: list[int] = []
        self.lm_head_calls = 0
        self.model = self._inner_forward

    def __call__(self, token_ids: mx.array, cache: _RecordingCache) -> mx.array:
        token_id = int(token_ids.reshape(-1)[0].item())
        self.top_level_calls.append(token_id)
        cache.committed_token_ids.append(token_id)
        return mx.zeros((1, 1, 8), dtype=mx.float32)

    def _inner_forward(
        self, token_ids: mx.array, cache: _RecordingCache
    ) -> mx.array:
        token_id = int(token_ids.reshape(-1)[0].item())
        self.inner_calls.append(token_id)
        cache.committed_token_ids.append(token_id)
        return mx.ones((1, 1, 4), dtype=mx.float32) * float(token_id)

    def lm_head(self, hidden: mx.array) -> mx.array:
        self.lm_head_calls += 1
        logits = mx.zeros((1, 1, 128), dtype=mx.float32)
        token_id = int(hidden[0, 0, 0].item())
        logits[:, :, token_id] = 1_000.0
        return logits


def test_forward_hidden_and_logits_commits_cache_once_per_token() -> None:
    model = _TargetModelWithSeparateTopLevelCall()
    cache = _RecordingCache()

    hidden, logits = _forward_hidden_and_logits(
        model, mx.array([[42]], dtype=mx.int32), cache  # type: ignore[arg-type]
    )

    assert hidden.shape == (1, 1, 4)
    assert int(mx.argmax(logits[0, 0]).item()) == 42
    assert cache.committed_token_ids == [42]
    assert model.inner_calls == [42]
    assert model.top_level_calls == []
    assert model.lm_head_calls == 1


def test_forward_hidden_and_logits_preserves_primary_correction_sequence_order() -> None:
    model = _TargetModelWithSeparateTopLevelCall()
    cache = _RecordingCache()

    for token_id in (10, 99):
        _forward_hidden_and_logits(
            model, mx.array([[token_id]], dtype=mx.int32), cache  # type: ignore[arg-type]
        )

    assert cache.committed_token_ids == [10, 99]
    assert model.inner_calls == [10, 99]
    assert model.top_level_calls == []
    assert model.lm_head_calls == 2
