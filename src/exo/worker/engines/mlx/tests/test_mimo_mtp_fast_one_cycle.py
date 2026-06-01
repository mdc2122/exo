from __future__ import annotations

from exo.worker.engines.mlx.mimo_mtp_fast.one_cycle import run_mimo_mtp_one_cycle


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


def test_one_cycle_uses_first_verifier_token_as_fallback_when_first_proposal_rejected() -> None:
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
