from __future__ import annotations

from exo.worker.engines.mlx.mimo_mtp_fast.one_cycle import MimoMtpOneCycleResult
from exo.worker.engines.mlx.mimo_mtp_fast.speculative_loop import (
    MimoMtpStreamEvent,
    stream_mimo_mtp_fast,
)


def _cycle_result(
    *,
    proposed: tuple[int, ...],
    accepted: tuple[int, ...],
    fallback: int | None,
    attempted_depth: int,
) -> MimoMtpOneCycleResult:
    return MimoMtpOneCycleResult(
        proposed_token_ids=proposed,
        accepted_token_ids=accepted,
        fallback_token_id=fallback,
        attempted_depth=attempted_depth,
        accepted_depth=len(accepted),
        elapsed_seconds=0.001,
    )


def test_stream_emits_accepted_tokens_and_updates_history_between_cycles() -> None:
    observed_histories: list[tuple[int, ...]] = []

    def one_cycle(history: tuple[int, ...], requested_depth: int) -> MimoMtpOneCycleResult:
        observed_histories.append(history)
        if len(observed_histories) == 1:
            assert requested_depth == 3
            return _cycle_result(
                proposed=(10, 11, 12),
                accepted=(10, 11),
                fallback=99,
                attempted_depth=3,
            )
        return _cycle_result(
            proposed=(12, 13),
            accepted=(12, 13),
            fallback=None,
            attempted_depth=2,
        )

    events = list(
        stream_mimo_mtp_fast(
            token_history=(1, 2),
            max_tokens=4,
            requested_depth=3,
            one_cycle=one_cycle,
        )
    )

    assert [event.token_id for event in events] == [10, 11, 12, 13]
    assert [event.accepted_depth for event in events] == [2, 2, 2, 2]
    assert [event.attempted_depth for event in events] == [3, 3, 2, 2]
    assert [event.used_fallback for event in events] == [False, False, False, False]
    assert observed_histories == [(1, 2), (1, 2, 10, 11)]


def test_stream_emits_fallback_token_when_no_proposal_is_accepted() -> None:
    def one_cycle(_history: tuple[int, ...], _requested_depth: int) -> MimoMtpOneCycleResult:
        return _cycle_result(
            proposed=(5, 6, 7),
            accepted=(),
            fallback=42,
            attempted_depth=3,
        )

    events = list(
        stream_mimo_mtp_fast(
            token_history=(1,),
            max_tokens=1,
            requested_depth=3,
            one_cycle=one_cycle,
        )
    )

    assert events == [
        MimoMtpStreamEvent(
            token_id=42,
            accepted_depth=0,
            attempted_depth=3,
            used_fallback=True,
        )
    ]


def test_stream_stops_when_one_cycle_returns_no_output_tokens() -> None:
    calls = 0

    def one_cycle(_history: tuple[int, ...], _requested_depth: int) -> MimoMtpOneCycleResult:
        nonlocal calls
        calls += 1
        return _cycle_result(proposed=(), accepted=(), fallback=None, attempted_depth=0)

    events = list(
        stream_mimo_mtp_fast(
            token_history=(1,),
            max_tokens=3,
            requested_depth=3,
            one_cycle=one_cycle,
        )
    )

    assert events == []
    assert calls == 1


def test_stream_respects_max_tokens_when_accepted_prefix_is_longer_than_budget() -> None:
    def one_cycle(_history: tuple[int, ...], _requested_depth: int) -> MimoMtpOneCycleResult:
        return _cycle_result(
            proposed=(20, 21, 22),
            accepted=(20, 21, 22),
            fallback=None,
            attempted_depth=3,
        )

    events = list(
        stream_mimo_mtp_fast(
            token_history=(1,),
            max_tokens=2,
            requested_depth=3,
            one_cycle=one_cycle,
        )
    )

    assert [event.token_id for event in events] == [20, 21]
    assert all(not event.used_fallback for event in events)


def test_stream_does_not_mutate_input_history_tuple() -> None:
    history = (8, 9)

    def one_cycle(_history: tuple[int, ...], _requested_depth: int) -> MimoMtpOneCycleResult:
        return _cycle_result(proposed=(30,), accepted=(30,), fallback=None, attempted_depth=1)

    list(
        stream_mimo_mtp_fast(
            token_history=history,
            max_tokens=1,
            requested_depth=1,
            one_cycle=one_cycle,
        )
    )

    assert history == (8, 9)
