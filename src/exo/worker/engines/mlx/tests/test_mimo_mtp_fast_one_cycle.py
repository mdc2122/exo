from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest

from exo.worker.engines.mlx.mimo_mtp_fast.one_cycle import (
    MimoMtpFastpathError,
    run_mimo_mtp_one_cycle,
)

ApproxFn = Callable[[float], object]
APPROX = cast(ApproxFn, pytest.approx)


def test_one_cycle_accepts_longest_verified_prefix_without_mutating_history() -> None:
    token_history = (101, 102)
    external_cache_marker = {"cache_tokens": [101, 102]}
    proposal_calls: list[tuple[tuple[int, ...], int]] = []
    verifier_calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def proposal_provider(history: tuple[int, ...], depth: int) -> tuple[int, ...]:
        proposal_calls.append((history, depth))
        return (10, 11, 12)

    def verifier_provider(
        history: tuple[int, ...], proposed_token_ids: tuple[int, ...]
    ) -> tuple[int, ...]:
        verifier_calls.append((history, proposed_token_ids))
        return (10, 11, 99)

    result = run_mimo_mtp_one_cycle(
        token_history=token_history,
        requested_depth=3,
        proposal_provider=proposal_provider,
        verifier_provider=verifier_provider,
    )

    assert result.proposed_token_ids == (10, 11, 12)
    assert result.accepted_token_ids == (10, 11)
    assert result.fallback_token_id == 99
    assert result.attempted_depth == 3
    assert result.accepted_depth == 2
    assert result.elapsed_seconds >= 0.0
    assert proposal_calls == [((101, 102), 3)]
    assert verifier_calls == [((101, 102), (10, 11, 12))]
    assert token_history == (101, 102)
    assert external_cache_marker == {"cache_tokens": [101, 102]}


def test_one_cycle_clamps_attempted_depth_to_available_proposals() -> None:
    def proposal_provider(_history: tuple[int, ...], _depth: int) -> tuple[int, ...]:
        return (20, 21)

    def verifier_provider(
        _history: tuple[int, ...], _proposed_token_ids: tuple[int, ...]
    ) -> tuple[int, ...]:
        return (20, 21)

    result = run_mimo_mtp_one_cycle(
        token_history=(1,),
        requested_depth=4,
        proposal_provider=proposal_provider,
        verifier_provider=verifier_provider,
    )

    assert result.proposed_token_ids == (20, 21)
    assert result.accepted_token_ids == (20, 21)
    assert result.fallback_token_id is None
    assert result.attempted_depth == 2
    assert result.accepted_depth == 2


def test_one_cycle_uses_first_verifier_token_as_fallback_when_first_proposal_rejected() -> (
    None
):
    def proposal_provider(_history: tuple[int, ...], _depth: int) -> tuple[int, ...]:
        return (5, 6, 7)

    def verifier_provider(
        _history: tuple[int, ...], _proposed_token_ids: tuple[int, ...]
    ) -> tuple[int, ...]:
        return (42, 6, 7)

    result = run_mimo_mtp_one_cycle(
        token_history=(1, 2, 3),
        requested_depth=3,
        proposal_provider=proposal_provider,
        verifier_provider=verifier_provider,
    )

    assert result.proposed_token_ids == (5, 6, 7)
    assert result.accepted_token_ids == ()
    assert result.fallback_token_id == 42
    assert result.attempted_depth == 3
    assert result.accepted_depth == 0


def test_one_cycle_returns_no_tokens_when_proposal_provider_has_no_proposal() -> None:
    verifier_called = False

    def proposal_provider(_history: tuple[int, ...], _depth: int) -> tuple[int, ...]:
        return ()

    def verifier_provider(
        _history: tuple[int, ...], _proposed_token_ids: tuple[int, ...]
    ) -> tuple[int, ...]:
        nonlocal verifier_called
        verifier_called = True
        return (99,)

    result = run_mimo_mtp_one_cycle(
        token_history=(1,),
        requested_depth=3,
        proposal_provider=proposal_provider,
        verifier_provider=verifier_provider,
    )

    assert result.proposed_token_ids == ()
    assert result.accepted_token_ids == ()
    assert result.fallback_token_id is None
    assert result.attempted_depth == 0
    assert result.accepted_depth == 0
    assert not verifier_called


def test_one_cycle_accepts_all_proposed_tokens() -> None:
    result = run_mimo_mtp_one_cycle(
        token_history=(1,),
        requested_depth=3,
        proposal_provider=lambda _history, _depth: (20, 21, 22),
        verifier_provider=lambda _history, _proposal: (20, 21, 22),
    )

    assert result.proposed_token_ids == (20, 21, 22)
    assert result.accepted_token_ids == (20, 21, 22)
    assert result.fallback_token_id is None
    assert result.attempted_depth == 3
    assert result.accepted_depth == 3


def test_one_cycle_handles_verifier_short_output_without_index_error() -> None:
    result = run_mimo_mtp_one_cycle(
        token_history=(1,),
        requested_depth=3,
        proposal_provider=lambda _history, _depth: (30, 31, 32),
        verifier_provider=lambda _history, _proposal: (30,),
    )

    assert result.proposed_token_ids == (30, 31, 32)
    assert result.accepted_token_ids == (30,)
    assert result.fallback_token_id is None
    assert result.attempted_depth == 3
    assert result.accepted_depth == 1


def test_one_cycle_records_timing_breakdown_for_proposal_verification_and_acceptance() -> (
    None
):
    times = iter((1.0, 1.2, 1.7, 1.75))

    result = run_mimo_mtp_one_cycle(
        token_history=(1,),
        requested_depth=2,
        proposal_provider=lambda _history, _depth: (40, 41),
        verifier_provider=lambda _history, _proposal: (40, 99),
        timer=lambda: next(times),
    )

    assert result.elapsed_seconds == APPROX(0.75)
    assert result.timing.proposal_seconds == APPROX(0.2)
    assert result.timing.verification_seconds == APPROX(0.5)
    assert result.timing.acceptance_seconds == APPROX(0.05)
    assert result.timing.fallback_seconds == 0.0


def test_one_cycle_rejects_invalid_proposal_token_output_with_fastpath_error() -> None:
    with pytest.raises(MimoMtpFastpathError) as exc_info:
        run_mimo_mtp_one_cycle(
            token_history=(1,),
            requested_depth=2,
            proposal_provider=lambda _history, _depth: (10, "bad"),
            verifier_provider=lambda _history, _proposal: (10, 11),
        )

    assert "proposal_provider returned non-int token at index 1" in str(exc_info.value)
