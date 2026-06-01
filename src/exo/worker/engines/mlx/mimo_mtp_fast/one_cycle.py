from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MimoMtpOneCycleResult:
    proposed_token_ids: tuple[int, ...]
    accepted_token_ids: tuple[int, ...]
    fallback_token_id: int | None
    attempted_depth: int
    accepted_depth: int
    elapsed_seconds: float


ProposalProvider = Callable[[tuple[int, ...], int], tuple[int, ...]]
VerifierProvider = Callable[[tuple[int, ...], tuple[int, ...]], tuple[int, ...]]


def _requested_depth(requested_depth: int) -> int:
    return max(0, int(requested_depth))


def _accepted_prefix(
    proposed_token_ids: tuple[int, ...], verifier_token_ids: tuple[int, ...]
) -> tuple[int, ...]:
    accepted: list[int] = []
    for proposed_token_id, verifier_token_id in zip(
        proposed_token_ids, verifier_token_ids, strict=False
    ):
        if proposed_token_id != verifier_token_id:
            break
        accepted.append(proposed_token_id)
    return tuple(accepted)


def run_mimo_mtp_one_cycle(
    *,
    token_history: tuple[int, ...],
    requested_depth: int,
    proposal_provider: ProposalProvider,
    verifier_provider: VerifierProvider,
) -> MimoMtpOneCycleResult:
    start = time.perf_counter()
    desired_depth = _requested_depth(requested_depth)
    if desired_depth == 0:
        return MimoMtpOneCycleResult(
            proposed_token_ids=(),
            accepted_token_ids=(),
            fallback_token_id=None,
            attempted_depth=0,
            accepted_depth=0,
            elapsed_seconds=time.perf_counter() - start,
        )

    proposed_token_ids = tuple(proposal_provider(token_history, desired_depth))[
        :desired_depth
    ]
    attempted_depth = len(proposed_token_ids)
    if attempted_depth == 0:
        return MimoMtpOneCycleResult(
            proposed_token_ids=(),
            accepted_token_ids=(),
            fallback_token_id=None,
            attempted_depth=0,
            accepted_depth=0,
            elapsed_seconds=time.perf_counter() - start,
        )

    verifier_token_ids = tuple(verifier_provider(token_history, proposed_token_ids))
    accepted_token_ids = _accepted_prefix(proposed_token_ids, verifier_token_ids)
    accepted_depth = len(accepted_token_ids)
    fallback_token_id = (
        None
        if accepted_depth == attempted_depth
        else verifier_token_ids[accepted_depth]
        if accepted_depth < len(verifier_token_ids)
        else None
    )
    return MimoMtpOneCycleResult(
        proposed_token_ids=proposed_token_ids,
        accepted_token_ids=accepted_token_ids,
        fallback_token_id=fallback_token_id,
        attempted_depth=attempted_depth,
        accepted_depth=accepted_depth,
        elapsed_seconds=time.perf_counter() - start,
    )
