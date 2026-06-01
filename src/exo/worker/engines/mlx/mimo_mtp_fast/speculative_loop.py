from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from exo.worker.engines.mlx.mimo_mtp_fast.one_cycle import MimoMtpOneCycleResult


@dataclass(frozen=True, slots=True)
class MimoMtpStreamEvent:
    token_id: int
    accepted_depth: int
    attempted_depth: int
    used_fallback: bool


OneCycleFn = Callable[[tuple[int, ...], int], MimoMtpOneCycleResult]


def _event_tokens(result: MimoMtpOneCycleResult) -> tuple[int, ...]:
    if result.accepted_token_ids:
        return result.accepted_token_ids
    if result.fallback_token_id is not None:
        return (result.fallback_token_id,)
    return ()


def stream_mimo_mtp_fast(
    *,
    token_history: tuple[int, ...],
    max_tokens: int,
    requested_depth: int,
    one_cycle: OneCycleFn,
) -> Iterator[MimoMtpStreamEvent]:
    history = list(token_history)
    emitted = 0
    token_budget = max(0, int(max_tokens))
    while emitted < token_budget:
        result = one_cycle(tuple(history), requested_depth)
        output_tokens = _event_tokens(result)
        if not output_tokens:
            break

        for token_id in output_tokens:
            history.append(token_id)
            emitted += 1
            yield MimoMtpStreamEvent(
                token_id=token_id,
                accepted_depth=result.accepted_depth,
                attempted_depth=result.attempted_depth,
                used_fallback=(
                    result.accepted_depth == 0 and result.fallback_token_id == token_id
                ),
            )
            if emitted >= token_budget:
                break
