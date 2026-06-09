from __future__ import annotations

from collections.abc import Callable
from typing import cast

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.mimo_mtp_fast.mtp_generate import (
    _sample_residual_correction_token,  # pyright: ignore[reportPrivateUsage]
    _speculative_acceptance_probability,  # pyright: ignore[reportPrivateUsage]
)

ApproxFn = Callable[[float], object]
APPROX = cast(ApproxFn, pytest.approx)


def test_speculative_acceptance_probability_uses_min_one_target_over_draft() -> None:
    target_logits = mx.array([[0.0, 2.0]], dtype=mx.float32)
    draft_logits = mx.array([[2.0, 0.0]], dtype=mx.float32)

    probability = _speculative_acceptance_probability(
        target_logits=target_logits,
        draft_logits=draft_logits,
        draft_token=1,
        temperature=1.0,
        top_p=1.0,
        min_p=0.0,
        top_k=0,
    )

    assert float(probability) == APPROX(1.0)


def test_speculative_acceptance_probability_reduces_when_draft_overestimates_token() -> None:
    target_logits = mx.array([[2.0, 0.0]], dtype=mx.float32)
    draft_logits = mx.array([[0.0, 2.0]], dtype=mx.float32)

    probability = _speculative_acceptance_probability(
        target_logits=target_logits,
        draft_logits=draft_logits,
        draft_token=1,
        temperature=1.0,
        top_p=1.0,
        min_p=0.0,
        top_k=0,
    )

    assert float(probability) == APPROX(0.13533528)


def test_residual_correction_samples_from_positive_target_minus_draft_mass() -> None:
    # Target strongly prefers token 0 while draft strongly over-proposes token 1.
    target_logits = mx.array([[2.0, 0.0]], dtype=mx.float32)
    draft_logits = mx.array([[0.0, 2.0]], dtype=mx.float32)

    token = _sample_residual_correction_token(
        target_logits=target_logits,
        draft_logits=draft_logits,
        temperature=1.0,
        top_p=1.0,
        min_p=0.0,
        top_k=0,
    )

    assert token == 0
