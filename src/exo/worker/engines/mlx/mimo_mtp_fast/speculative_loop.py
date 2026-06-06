from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum

from exo.worker.engines.mlx.mimo_mtp_fast.one_cycle import (
    MimoMtpCycleTiming,
    MimoMtpFastpathError,
    MimoMtpOneCycleResult,
)


class MimoMtpFailureKind(StrEnum):
    STRUCTURAL = "structural"
    RUNTIME_PROVIDER = "runtime_provider"


@dataclass(frozen=True, slots=True)
class MimoMtpStreamEvent:
    token_id: int
    accepted_depth: int
    attempted_depth: int
    used_fallback: bool


@dataclass(frozen=True, slots=True)
class MimoMtpCycleTrace:
    cycle_index: int
    history_length_before: int
    proposed_token_ids: tuple[int, ...]
    accepted_token_ids: tuple[int, ...]
    fallback_token_id: int | None
    attempted_depth: int
    accepted_depth: int
    emitted_token_ids: tuple[int, ...]
    failure_kind: MimoMtpFailureKind | None = None
    failure_message: str | None = None
    elapsed_seconds: float = 0.0
    timing: MimoMtpCycleTiming = MimoMtpCycleTiming()


OneCycleFn = Callable[[tuple[int, ...], int], MimoMtpOneCycleResult]
TraceCollector = Callable[[MimoMtpCycleTrace], None]
FallbackTokenProvider = Callable[[tuple[int, ...]], int]
ResetMtpCache = Callable[[object], None]


def _event_tokens(result: MimoMtpOneCycleResult) -> tuple[int, ...]:
    if result.accepted_token_ids:
        return result.accepted_token_ids
    if result.fallback_token_id is not None:
        return (result.fallback_token_id,)
    return ()


def _collect_trace(
    trace_collector: TraceCollector | None,
    trace: MimoMtpCycleTrace,
) -> None:
    if trace_collector is not None:
        trace_collector(trace)


def stream_mimo_mtp_fast(
    *,
    token_history: tuple[int, ...],
    max_tokens: int,
    requested_depth: int,
    one_cycle: OneCycleFn,
    trace_collector: TraceCollector | None = None,
    fallback_token_provider: FallbackTokenProvider | None = None,
    reset_mtp_cache: ResetMtpCache | None = None,
    mtp_cache: object | None = None,
) -> Iterator[MimoMtpStreamEvent]:
    history = list(token_history)
    emitted = 0
    cycle_index = 0
    token_budget = max(0, int(max_tokens))
    while emitted < token_budget:
        history_before = tuple(history)
        try:
            result = one_cycle(history_before, requested_depth)
        except MimoMtpFastpathError as exc:
            if reset_mtp_cache is not None and mtp_cache is not None:
                reset_mtp_cache(mtp_cache)
            _collect_trace(
                trace_collector,
                MimoMtpCycleTrace(
                    cycle_index=cycle_index,
                    history_length_before=len(history_before),
                    proposed_token_ids=(),
                    accepted_token_ids=(),
                    fallback_token_id=None,
                    attempted_depth=0,
                    accepted_depth=0,
                    emitted_token_ids=(),
                    failure_kind=MimoMtpFailureKind.STRUCTURAL,
                    failure_message=str(exc),
                ),
            )
            raise
        except Exception as exc:
            if fallback_token_provider is None:
                _collect_trace(
                    trace_collector,
                    MimoMtpCycleTrace(
                        cycle_index=cycle_index,
                        history_length_before=len(history_before),
                        proposed_token_ids=(),
                        accepted_token_ids=(),
                        fallback_token_id=None,
                        attempted_depth=0,
                        accepted_depth=0,
                        emitted_token_ids=(),
                        failure_kind=MimoMtpFailureKind.RUNTIME_PROVIDER,
                        failure_message=str(exc),
                    ),
                )
                raise
            fallback_token = int(fallback_token_provider(history_before))
            history.append(fallback_token)
            emitted += 1
            _collect_trace(
                trace_collector,
                MimoMtpCycleTrace(
                    cycle_index=cycle_index,
                    history_length_before=len(history_before),
                    proposed_token_ids=(),
                    accepted_token_ids=(),
                    fallback_token_id=fallback_token,
                    attempted_depth=0,
                    accepted_depth=0,
                    emitted_token_ids=(fallback_token,),
                    failure_kind=MimoMtpFailureKind.RUNTIME_PROVIDER,
                    failure_message=str(exc),
                ),
            )
            yield MimoMtpStreamEvent(
                token_id=fallback_token,
                accepted_depth=0,
                attempted_depth=0,
                used_fallback=True,
            )
            cycle_index += 1
            continue

        output_tokens = _event_tokens(result)
        if not output_tokens:
            _collect_trace(
                trace_collector,
                MimoMtpCycleTrace(
                    cycle_index=cycle_index,
                    history_length_before=len(history_before),
                    proposed_token_ids=result.proposed_token_ids,
                    accepted_token_ids=result.accepted_token_ids,
                    fallback_token_id=result.fallback_token_id,
                    attempted_depth=result.attempted_depth,
                    accepted_depth=result.accepted_depth,
                    emitted_token_ids=(),
                    elapsed_seconds=result.elapsed_seconds,
                    timing=result.timing,
                ),
            )
            break

        emitted_tokens: list[int] = []
        for token_id in output_tokens:
            history.append(token_id)
            emitted += 1
            emitted_tokens.append(token_id)
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
        _collect_trace(
            trace_collector,
            MimoMtpCycleTrace(
                cycle_index=cycle_index,
                history_length_before=len(history_before),
                proposed_token_ids=result.proposed_token_ids,
                accepted_token_ids=result.accepted_token_ids,
                fallback_token_id=result.fallback_token_id,
                attempted_depth=result.attempted_depth,
                accepted_depth=result.accepted_depth,
                emitted_token_ids=tuple(emitted_tokens),
                elapsed_seconds=result.elapsed_seconds,
                timing=result.timing,
            ),
        )
        cycle_index += 1
