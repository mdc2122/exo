from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from exo.worker.engines.mlx.mimo_mtp_fast.providers import (
    make_mtp_one_cycle_runner,
    make_mtp_proposal_provider,
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


def test_mtp_proposal_provider_calls_stack_with_latest_token_and_depth() -> None:
    stack = _FakeStack()
    hidden_calls: list[tuple[int, ...]] = []
    hidden_provider = _HiddenProvider(
        hidden=mx.ones((1, 1, 4), dtype=mx.float32),
        calls=hidden_calls,
    )
    mtp_cache = {"kind": "mtp-cache"}

    proposal_provider = make_mtp_proposal_provider(
        stack=stack,
        hidden_state_provider=hidden_provider,
        token_dtype=mx.int32,
        mtp_cache=mtp_cache,
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


def test_make_mtp_one_cycle_runner_composes_real_proposal_and_verifier_providers() -> None:
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
