from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import mlx.core as mx

from exo.worker.engines.mlx.mimo_mtp_fast.one_cycle import (
    MimoMtpOneCycleResult,
    run_mimo_mtp_one_cycle,
)


class MtpProposalStack(Protocol):
    def propose(
        self,
        *,
        previous_hidden_state: mx.array,
        latest_token_ids: mx.array,
        max_draft_tokens: int,
        sampler: Sampler,
        cache: object | None = None,
    ) -> tuple[mx.array, list[mx.array]]: ...


class TargetVerifierModel(Protocol):
    def model(self, input_tokens: mx.array, cache: object) -> mx.array: ...

    def lm_head(self, hidden: mx.array) -> mx.array: ...


Sampler = Callable[[mx.array], mx.array | int]
HiddenStateProvider = Callable[[tuple[int, ...]], mx.array]
CacheFactory = Callable[[], object]
MtpOneCycleFn = Callable[[tuple[int, ...], int], MimoMtpOneCycleResult]


def mx_token_tuple(tokens: mx.array) -> tuple[int, ...]:
    flattened = tokens.reshape(-1)
    return tuple(int(flattened[index].item()) for index in range(int(flattened.size)))


def _greedy_sampler(logits: mx.array) -> mx.array:
    token_ids = mx.argmax(logits, axis=-1)
    if token_ids.ndim == 1:
        token_ids = token_ids[:, None]
    return token_ids.astype(mx.int32)


def _last_token_batch(token_id: int, *, token_dtype: mx.Dtype) -> mx.array:
    return mx.array([[int(token_id)]], dtype=token_dtype)


def make_mtp_proposal_provider(
    *,
    stack: MtpProposalStack,
    hidden_state_provider: HiddenStateProvider,
    token_dtype: mx.Dtype,
    mtp_cache: object | None,
    sampler: Sampler = _greedy_sampler,
) -> Callable[[tuple[int, ...], int], tuple[int, ...]]:
    def proposal_provider(token_history: tuple[int, ...], requested_depth: int) -> tuple[int, ...]:
        if not token_history or requested_depth <= 0:
            return ()
        previous_hidden_state = hidden_state_provider(token_history)
        latest_token_ids = _last_token_batch(token_history[-1], token_dtype=token_dtype)
        proposed_tokens, _per_layer_logits = stack.propose(
            previous_hidden_state=previous_hidden_state,
            latest_token_ids=latest_token_ids,
            max_draft_tokens=requested_depth,
            sampler=sampler,
            cache=mtp_cache,
        )
        return mx_token_tuple(proposed_tokens)[:requested_depth]

    return proposal_provider


def _last_logits(logits: mx.array) -> mx.array:
    if logits.ndim == 1:
        return logits
    if logits.ndim == 2:
        return logits[-1, :]
    return logits[:, -1, :][0]


def _greedy_token_from_logits(logits: mx.array) -> int:
    return int(mx.argmax(_last_logits(logits), axis=-1).item())


def _last_hidden_step(hidden: mx.array) -> mx.array:
    if hidden.ndim == 3:
        return hidden[:, -1:, :]
    if hidden.ndim == 2:
        return hidden[None, -1:, :]
    raise ValueError(f"expected hidden rank 2 or 3; got shape={tuple(hidden.shape)}")


def make_replay_hidden_state_provider(
    *,
    model: TargetVerifierModel,
    cache_factory: CacheFactory,
    token_dtype: mx.Dtype,
) -> HiddenStateProvider:
    def hidden_state_provider(token_history: tuple[int, ...]) -> mx.array:
        if not token_history:
            raise ValueError("token_history must be non-empty for MTP hidden replay")
        cache = cache_factory()
        hidden: mx.array | None = None
        for token_id in token_history:
            hidden = model.model(_last_token_batch(token_id, token_dtype=token_dtype), cache)
        assert hidden is not None
        return _last_hidden_step(hidden)

    return hidden_state_provider


def make_replay_target_verifier_provider(
    *,
    model: TargetVerifierModel,
    cache_factory: CacheFactory,
    token_dtype: mx.Dtype,
) -> Callable[[tuple[int, ...], tuple[int, ...]], tuple[int, ...]]:
    def verifier_provider(
        token_history: tuple[int, ...], proposed_token_ids: tuple[int, ...]
    ) -> tuple[int, ...]:
        if not token_history or not proposed_token_ids:
            return ()
        cache = cache_factory()
        for context_token_id in token_history[:-1]:
            model.model(_last_token_batch(context_token_id, token_dtype=token_dtype), cache)
        verifier_inputs = (token_history[-1], *proposed_token_ids[:-1])
        verified_tokens: list[int] = []
        for token_id in verifier_inputs:
            hidden = model.model(_last_token_batch(token_id, token_dtype=token_dtype), cache)
            logits = model.lm_head(hidden)
            verified_tokens.append(_greedy_token_from_logits(logits))
        return tuple(verified_tokens)

    return verifier_provider


def make_replay_mtp_one_cycle_runner(
    *,
    stack: MtpProposalStack,
    target_model: TargetVerifierModel,
    target_cache_factory: CacheFactory,
    token_dtype: mx.Dtype,
    sampler: Sampler = _greedy_sampler,
) -> MtpOneCycleFn:
    return make_mtp_one_cycle_runner(
        stack=stack,
        target_model=target_model,
        target_cache=target_cache_factory(),
        hidden_state_provider=make_replay_hidden_state_provider(
            model=target_model,
            cache_factory=target_cache_factory,
            token_dtype=token_dtype,
        ),
        token_dtype=token_dtype,
        mtp_cache=None,
        sampler=sampler,
        verifier_provider=make_replay_target_verifier_provider(
            model=target_model,
            cache_factory=target_cache_factory,
            token_dtype=token_dtype,
        ),
    )


def make_target_verifier_provider(
    *,
    model: TargetVerifierModel,
    cache: object,
    token_dtype: mx.Dtype,
) -> Callable[[tuple[int, ...], tuple[int, ...]], tuple[int, ...]]:
    def verifier_provider(
        token_history: tuple[int, ...], proposed_token_ids: tuple[int, ...]
    ) -> tuple[int, ...]:
        if not token_history or not proposed_token_ids:
            return ()
        verifier_inputs = (token_history[-1], *proposed_token_ids[:-1])
        verified_tokens: list[int] = []
        for token_id in verifier_inputs:
            hidden = model.model(_last_token_batch(token_id, token_dtype=token_dtype), cache)
            logits = model.lm_head(hidden)
            verified_tokens.append(_greedy_token_from_logits(logits))
        return tuple(verified_tokens)

    return verifier_provider


def make_mtp_one_cycle_runner(
    *,
    stack: MtpProposalStack,
    target_model: TargetVerifierModel,
    target_cache: object,
    hidden_state_provider: HiddenStateProvider,
    token_dtype: mx.Dtype,
    mtp_cache: object | None,
    sampler: Sampler = _greedy_sampler,
    verifier_provider: Callable[[tuple[int, ...], tuple[int, ...]], tuple[int, ...]] | None = None,
) -> MtpOneCycleFn:
    proposal_provider = make_mtp_proposal_provider(
        stack=stack,
        hidden_state_provider=hidden_state_provider,
        token_dtype=token_dtype,
        mtp_cache=mtp_cache,
        sampler=sampler,
    )
    resolved_verifier_provider = (
        make_target_verifier_provider(
            model=target_model,
            cache=target_cache,
            token_dtype=token_dtype,
        )
        if verifier_provider is None
        else verifier_provider
    )

    def one_cycle(token_history: tuple[int, ...], requested_depth: int) -> MimoMtpOneCycleResult:
        return run_mimo_mtp_one_cycle(
            token_history=token_history,
            requested_depth=requested_depth,
            proposal_provider=proposal_provider,
            verifier_provider=resolved_verifier_provider,
        )

    return one_cycle
