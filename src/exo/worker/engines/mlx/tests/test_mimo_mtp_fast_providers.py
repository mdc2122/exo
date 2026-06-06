from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.mimo_mtp_fast.one_cycle import MimoMtpFastpathError
from exo.worker.engines.mlx.mimo_mtp_fast.providers import (
    make_mtp_one_cycle_runner,
    make_mtp_proposal_provider,
    make_replay_hidden_state_provider,
    make_replay_mtp_one_cycle_runner,
    make_replay_target_verifier_provider,
    make_target_verifier_provider,
    mx_token_tuple,
)


class _FakeStack:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def propose(
        self,
        *,
        previous_hidden_state: mx.array,
        latest_token_ids: mx.array,
        max_draft_tokens: int,
        sampler: object,
        cache: object | None = None,
    ) -> tuple[mx.array, list[mx.array]]:
        self.calls.append(
            {
                "previous_hidden_state_shape": previous_hidden_state.shape,
                "latest_token_ids": mx_token_tuple(latest_token_ids),
                "max_draft_tokens": max_draft_tokens,
                "cache": cache,
                "sampler": sampler,
            }
        )
        return mx.array([[10, 11, 12]], dtype=mx.int32), []


class _FakeTargetModel:
    def __init__(self, verifier_tokens: tuple[int, ...]) -> None:
        self.verifier_tokens = verifier_tokens
        self.forwarded_tokens: list[int] = []
        self._index = 0

    def model(self, input_tokens: mx.array, cache: object) -> mx.array:
        self.forwarded_tokens.append(int(input_tokens[0, 0].item()))
        return mx.array([[[float(self._index + 1)]]], dtype=mx.float32)

    def lm_head(self, hidden: mx.array) -> mx.array:
        del hidden
        token = self.verifier_tokens[self._index]
        self._index += 1
        logits = mx.zeros((1, 1, 128), dtype=mx.float32)
        logits[:, :, token] = 1_000.0
        return logits


@dataclass(frozen=True)
class _HiddenProvider:
    hidden: mx.array
    calls: list[tuple[int, ...]]

    def __call__(self, token_history: tuple[int, ...]) -> mx.array:
        self.calls.append(token_history)
        return self.hidden


def test_mx_token_tuple_flattens_batch_tokens() -> None:
    assert mx_token_tuple(mx.array([[1, 2, 3]], dtype=mx.int32)) == (1, 2, 3)
    assert mx_token_tuple(mx.array([4, 5], dtype=mx.int32)) == (4, 5)


def test_mtp_proposal_provider_calls_stack_with_latest_token_depth_sampler_and_cache() -> (
    None
):
    stack = _FakeStack()
    hidden_calls: list[tuple[int, ...]] = []
    hidden_provider = _HiddenProvider(
        hidden=mx.ones((1, 1, 4), dtype=mx.float32),
        calls=hidden_calls,
    )
    mtp_cache = {"kind": "mtp-cache"}

    def sampler(logits: mx.array) -> mx.array:
        return mx.argmax(logits, axis=-1).astype(mx.int32)

    proposal_provider = make_mtp_proposal_provider(
        stack=stack,
        hidden_state_provider=hidden_provider,
        token_dtype=mx.int32,
        mtp_cache=mtp_cache,
        sampler=sampler,
    )

    proposal = proposal_provider((101, 102), 2)

    assert proposal == (10, 11)
    assert hidden_calls == [(101, 102)]
    assert stack.calls == [
        {
            "previous_hidden_state_shape": (1, 1, 4),
            "latest_token_ids": (102,),
            "max_draft_tokens": 2,
            "cache": mtp_cache,
            "sampler": sampler,
        }
    ]


def test_target_verifier_provider_forwards_boundary_then_draft_prefix() -> None:
    model = _FakeTargetModel(verifier_tokens=(10, 11, 99))
    verifier_provider = make_target_verifier_provider(
        model=model,
        cache={"kind": "target-cache"},
        token_dtype=mx.int32,
    )

    verifier_tokens = verifier_provider((101, 102), (10, 11, 12))

    assert verifier_tokens == (10, 11, 99)
    assert model.forwarded_tokens == [102, 10, 11]


def test_make_mtp_one_cycle_runner_composes_real_proposal_and_verifier_providers() -> (
    None
):
    stack = _FakeStack()
    target_model = _FakeTargetModel(verifier_tokens=(10, 99))
    hidden_calls: list[tuple[int, ...]] = []
    one_cycle = make_mtp_one_cycle_runner(
        stack=stack,
        target_model=target_model,
        target_cache={},
        hidden_state_provider=_HiddenProvider(
            hidden=mx.ones((1, 1, 4), dtype=mx.float32),
            calls=hidden_calls,
        ),
        token_dtype=mx.int32,
        mtp_cache=None,
    )

    result = one_cycle((7, 8), 2)

    assert result.proposed_token_ids == (10, 11)
    assert result.accepted_token_ids == (10,)
    assert result.fallback_token_id == 99
    assert result.attempted_depth == 2
    assert result.accepted_depth == 1
    assert hidden_calls == [(7, 8)]
    assert target_model.forwarded_tokens == [8, 10]


def test_d1_d2_d3_one_cycle_runner_is_deterministic_under_fake_providers() -> None:
    for depth, expected in ((1, (10,)), (2, (10, 11)), (3, (10, 11, 12))):
        stack = _FakeStack()
        target_model = _FakeTargetModel(verifier_tokens=(10, 11, 12))
        hidden_calls: list[tuple[int, ...]] = []
        one_cycle = make_mtp_one_cycle_runner(
            stack=stack,
            target_model=target_model,
            target_cache={},
            hidden_state_provider=_HiddenProvider(
                hidden=mx.ones((1, 1, 4), dtype=mx.float32),
                calls=hidden_calls,
            ),
            token_dtype=mx.int32,
            mtp_cache=None,
        )

        result = one_cycle((7, 8), depth)

        assert result.proposed_token_ids == expected
        assert result.accepted_token_ids == expected
        assert result.fallback_token_id is None
        assert result.attempted_depth == depth
        assert result.accepted_depth == depth
        assert hidden_calls == [(7, 8)]


def test_proposal_provider_rejects_non_integer_token_array() -> None:
    class FloatTokenStack(_FakeStack):
        def propose(
            self,
            *,
            previous_hidden_state: mx.array,
            latest_token_ids: mx.array,
            max_draft_tokens: int,
            sampler: object,
            cache: object | None = None,
        ) -> tuple[mx.array, list[mx.array]]:
            del (
                previous_hidden_state,
                latest_token_ids,
                max_draft_tokens,
                sampler,
                cache,
            )
            return mx.array([[1.25]], dtype=mx.float32), []

    proposal_provider = make_mtp_proposal_provider(
        stack=FloatTokenStack(),
        hidden_state_provider=_HiddenProvider(
            hidden=mx.ones((1, 1, 4), dtype=mx.float32),
            calls=[],
        ),
        token_dtype=mx.int32,
        mtp_cache=None,
    )

    with pytest.raises(MimoMtpFastpathError) as exc_info:
        proposal_provider((1, 2), 1)

    assert "MTP proposal provider returned non-integer token dtype" in str(
        exc_info.value
    )


def test_target_verifier_provider_rejects_invalid_hidden_shape_with_fastpath_error() -> (
    None
):
    class BadHiddenModel(_FakeTargetModel):
        def model(self, input_tokens: mx.array, cache: object) -> mx.array:
            del input_tokens, cache
            return mx.ones((4,), dtype=mx.float32)

    verifier_provider = make_target_verifier_provider(
        model=BadHiddenModel(verifier_tokens=(10,)),
        cache={},
        token_dtype=mx.int32,
    )

    with pytest.raises(MimoMtpFastpathError) as exc_info:
        verifier_provider((1, 2), (10,))

    assert "target verifier model returned invalid hidden shape" in str(exc_info.value)


def test_proposal_provider_rejects_invalid_hidden_state_shape() -> None:
    proposal_provider = make_mtp_proposal_provider(
        stack=_FakeStack(),
        hidden_state_provider=_HiddenProvider(
            hidden=mx.ones((4,), dtype=mx.float32),
            calls=[],
        ),
        token_dtype=mx.int32,
        mtp_cache=None,
    )

    with pytest.raises(MimoMtpFastpathError) as exc_info:
        proposal_provider((1, 2), 1)

    assert "MTP hidden-state provider returned invalid hidden shape" in str(
        exc_info.value
    )


def test_target_verifier_provider_rejects_integer_logits_dtype() -> None:
    class IntegerLogitsModel(_FakeTargetModel):
        def lm_head(self, hidden: mx.array) -> mx.array:
            del hidden
            return mx.zeros((1, 1, 128), dtype=mx.int32)

    verifier_provider = make_target_verifier_provider(
        model=IntegerLogitsModel(verifier_tokens=(10,)),
        cache={},
        token_dtype=mx.int32,
    )

    with pytest.raises(MimoMtpFastpathError) as exc_info:
        verifier_provider((1, 2), (10,))

    assert "target verifier lm_head returned non-floating logits dtype" in str(
        exc_info.value
    )


class _TracingReplayModel:
    def __init__(self, verifier_tokens: tuple[int, ...]) -> None:
        self.verifier_tokens = verifier_tokens
        self.forwarded: list[tuple[str, int]] = []
        self._index = 0

    def model(self, input_tokens: mx.array, cache: object) -> mx.array:
        cache_name = cache.name if isinstance(cache, _NamedCache) else "unknown"
        self.forwarded.append((cache_name, int(input_tokens[0, 0].item())))
        return mx.array([[[float(len(self.forwarded))]]], dtype=mx.float32)

    def lm_head(self, hidden: mx.array) -> mx.array:
        del hidden
        token = self.verifier_tokens[self._index]
        self._index += 1
        logits = mx.zeros((1, 1, 128), dtype=mx.float32)
        logits[:, :, token] = 1_000.0
        return logits


@dataclass(frozen=True)
class _NamedCache:
    name: str


def _named_cache_factory() -> tuple[Callable[[], object], list[_NamedCache]]:
    caches: list[_NamedCache] = []

    def cache_factory() -> _NamedCache:
        cache = _NamedCache(name=f"cache-{len(caches)}")
        caches.append(cache)
        return cache

    return cache_factory, caches


def test_replay_hidden_state_provider_replays_history_into_fresh_cache() -> None:
    model = _TracingReplayModel(verifier_tokens=())
    cache_factory, caches = _named_cache_factory()
    hidden_provider = make_replay_hidden_state_provider(
        model=model,
        cache_factory=cache_factory,
        token_dtype=mx.int32,
    )

    hidden = hidden_provider((1, 2, 3))

    assert hidden.shape == (1, 1, 1)
    assert len(caches) == 1
    assert model.forwarded == [("cache-0", 1), ("cache-0", 2), ("cache-0", 3)]


def test_replay_target_verifier_provider_prefills_prefix_then_boundary_and_drafts() -> (
    None
):
    model = _TracingReplayModel(verifier_tokens=(10, 11, 99))
    cache_factory, caches = _named_cache_factory()
    verifier_provider = make_replay_target_verifier_provider(
        model=model,
        cache_factory=cache_factory,
        token_dtype=mx.int32,
    )

    verifier_tokens = verifier_provider((1, 2, 3), (10, 11, 12))

    assert verifier_tokens == (10, 11, 99)
    assert len(caches) == 1
    assert model.forwarded == [
        ("cache-0", 1),
        ("cache-0", 2),
        ("cache-0", 3),
        ("cache-0", 10),
        ("cache-0", 11),
    ]


def test_replay_mtp_one_cycle_runner_uses_replay_verifier_provider() -> None:
    stack = _FakeStack()
    target_model = _TracingReplayModel(verifier_tokens=(10, 99))
    cache_factory, caches = _named_cache_factory()
    one_cycle = make_replay_mtp_one_cycle_runner(
        stack=stack,
        target_model=target_model,
        target_cache_factory=cache_factory,
        token_dtype=mx.int32,
    )

    result = one_cycle((7, 8), 2)

    assert result.accepted_token_ids == (10,)
    assert result.fallback_token_id == 99
    assert [cache.name for cache in caches] == ["cache-0", "cache-1", "cache-2"]
    assert target_model.forwarded == [
        ("cache-1", 7),
        ("cache-1", 8),
        ("cache-2", 7),
        ("cache-2", 8),
        ("cache-2", 10),
    ]
