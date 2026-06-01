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
        sampler: object,
        cache: object | None = None,
    ) -> tuple[mx.array, list[mx.array]]: ...


class TargetVerifierModel(Protocol):
    def model(self, input_tokens: mx.array, cache: object) -> mx.array: ...

    def lm_head(self, hidden: mx.array) -> mx.array: ...


HiddenStateProvider = Callable[[tuple[int, ...]], mx.array]
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
    sampler: object = _greedy_sampler,
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
    sampler: object = _greedy_sampler,
) -> MtpOneCycleFn:
    proposal_provider = make_mtp_proposal_provider(
        stack=stack,
        hidden_state_provider=hidden_state_provider,
        token_dtype=token_dtype,
        mtp_cache=mtp_cache,
        sampler=sampler,
    )
    verifier_provider = make_target_verifier_provider(
        model=target_model,
        cache=target_cache,
        token_dtype=token_dtype,
    )

    def one_cycle(token_history: tuple[int, ...], requested_depth: int) -> MimoMtpOneCycleResult:
        return run_mimo_mtp_one_cycle(
            token_history=token_history,
            requested_depth=requested_depth,
            proposal_provider=proposal_provider,
            verifier_provider=verifier_provider,
        )

    return one_cycle
