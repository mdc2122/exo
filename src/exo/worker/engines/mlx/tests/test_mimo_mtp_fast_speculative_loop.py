from __future__ import annotations

import pytest

from exo.worker.engines.mlx.mimo_mtp_fast.one_cycle import (
    MimoMtpFastpathError,
    MimoMtpOneCycleResult,
)
from exo.worker.engines.mlx.mimo_mtp_fast.speculative_loop import (
    MimoMtpCycleTrace,
    MimoMtpFailureKind,
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

    def one_cycle(
        history: tuple[int, ...], requested_depth: int
    ) -> MimoMtpOneCycleResult:
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
    def one_cycle(
        _history: tuple[int, ...], _requested_depth: int
    ) -> MimoMtpOneCycleResult:
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

    def one_cycle(
        _history: tuple[int, ...], _requested_depth: int
    ) -> MimoMtpOneCycleResult:
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


def test_stream_respects_max_tokens_when_accepted_prefix_is_longer_than_budget() -> (
    None
):
    def one_cycle(
        _history: tuple[int, ...], _requested_depth: int
    ) -> MimoMtpOneCycleResult:
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

    def one_cycle(
        _history: tuple[int, ...], _requested_depth: int
    ) -> MimoMtpOneCycleResult:
        return _cycle_result(
            proposed=(30,), accepted=(30,), fallback=None, attempted_depth=1
        )

    list(
        stream_mimo_mtp_fast(
            token_history=history,
            max_tokens=1,
            requested_depth=1,
            one_cycle=one_cycle,
        )
    )

    assert history == (8, 9)


def test_stream_collects_per_cycle_trace_without_ralph_machinery() -> None:
    trace: list[MimoMtpCycleTrace] = []

    def one_cycle(
        history: tuple[int, ...], _requested_depth: int
    ) -> MimoMtpOneCycleResult:
        if history == (1,):
            return _cycle_result(
                proposed=(10, 11), accepted=(10,), fallback=99, attempted_depth=2
            )
        return _cycle_result(
            proposed=(20,), accepted=(), fallback=42, attempted_depth=1
        )

    events = list(
        stream_mimo_mtp_fast(
            token_history=(1,),
            max_tokens=2,
            requested_depth=2,
            one_cycle=one_cycle,
            trace_collector=trace.append,
        )
    )

    assert [event.token_id for event in events] == [10, 42]
    assert len(trace) == 2
    assert trace[0].cycle_index == 0
    assert trace[0].history_length_before == 1
    assert trace[0].proposed_token_ids == (10, 11)
    assert trace[0].accepted_token_ids == (10,)
    assert trace[0].fallback_token_id == 99
    assert trace[0].attempted_depth == 2
    assert trace[0].accepted_depth == 1
    assert trace[0].emitted_token_ids == (10,)
    assert trace[0].failure_kind is None
    assert trace[1].cycle_index == 1
    assert trace[1].history_length_before == 2
    assert trace[1].emitted_token_ids == (42,)
    assert trace[1].failure_kind is None


def test_stream_falls_back_on_runtime_provider_failure_and_records_trace() -> None:
    trace: list[MimoMtpCycleTrace] = []
    fallback_calls: list[tuple[int, ...]] = []

    def one_cycle(
        _history: tuple[int, ...], _requested_depth: int
    ) -> MimoMtpOneCycleResult:
        raise RuntimeError("temporary target cache miss")

    def fallback_token_provider(history: tuple[int, ...]) -> int:
        fallback_calls.append(history)
        return 77

    events = list(
        stream_mimo_mtp_fast(
            token_history=(1, 2),
            max_tokens=1,
            requested_depth=3,
            one_cycle=one_cycle,
            fallback_token_provider=fallback_token_provider,
            trace_collector=trace.append,
        )
    )

    assert events == [
        MimoMtpStreamEvent(
            token_id=77,
            accepted_depth=0,
            attempted_depth=0,
            used_fallback=True,
        )
    ]
    assert fallback_calls == [(1, 2)]
    assert trace[0].failure_kind == MimoMtpFailureKind.RUNTIME_PROVIDER
    assert trace[0].failure_message == "temporary target cache miss"
    assert trace[0].emitted_token_ids == (77,)


def test_stream_fails_closed_on_structural_fastpath_error_and_resets_stale_cache() -> (
    None
):
    trace: list[MimoMtpCycleTrace] = []
    resets: list[object] = []

    def one_cycle(
        _history: tuple[int, ...], _requested_depth: int
    ) -> MimoMtpOneCycleResult:
        raise MimoMtpFastpathError("invalid MTP proposal shape")

    with pytest.raises(MimoMtpFastpathError) as exc_info:
        list(
            stream_mimo_mtp_fast(
                token_history=(1, 2),
                max_tokens=2,
                requested_depth=3,
                one_cycle=one_cycle,
                reset_mtp_cache=resets.append,
                mtp_cache="cache-1",
                trace_collector=trace.append,
            )
        )

    assert "invalid MTP proposal shape" in str(exc_info.value)
    assert resets == ["cache-1"]
    assert len(trace) == 1
    assert trace[0].failure_kind == MimoMtpFailureKind.STRUCTURAL
    assert trace[0].failure_message == "invalid MTP proposal shape"
    assert trace[0].emitted_token_ids == ()


def test_stream_fails_closed_when_runtime_failure_has_no_fallback_provider() -> None:
    def one_cycle(
        _history: tuple[int, ...], _requested_depth: int
    ) -> MimoMtpOneCycleResult:
        raise RuntimeError("no fallback available")

    with pytest.raises(RuntimeError) as exc_info:
        list(
            stream_mimo_mtp_fast(
                token_history=(1,),
                max_tokens=1,
                requested_depth=1,
                one_cycle=one_cycle,
            )
        )

    assert "no fallback available" in str(exc_info.value)
