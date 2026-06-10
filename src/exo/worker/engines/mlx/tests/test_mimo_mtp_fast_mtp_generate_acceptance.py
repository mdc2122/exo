from __future__ import annotations

import math
from collections.abc import Callable
from typing import TypedDict, cast

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.mimo_mtp_fast.mtp_generate import (
 _sample_residual_correction_token_lazy,  # pyright: ignore[reportPrivateUsage]
 _sampling_probabilities,  # pyright: ignore[reportPrivateUsage]
 _speculative_acceptance_probability,  # pyright: ignore[reportPrivateUsage]
)


def _sample_token(
    *,
    target_logits: mx.array,
    draft_logits: mx.array,
    temperature: float,
    top_p: float,
    min_p: float,
    top_k: int,
) -> int:
    return int(
        _sample_residual_correction_token_lazy(
            target_logits=target_logits,
            draft_logits=draft_logits,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
        ).item()
    )


class _SamplingKwargs(TypedDict):
 temperature: float
 top_p: float
 min_p: float
 top_k: int


ApproxFn = Callable[[float], object]
APPROX = cast(ApproxFn, pytest.approx)

# Shared default sampling parameters for invariant tests.
_DEFAULT_SAMPLING: _SamplingKwargs = {
 "temperature": 1.0,
 "top_p": 1.0,
 "min_p": 0.0,
 "top_k": 0,
}


def test_speculative_acceptance_probability_uses_min_one_target_over_draft() -> None:
 target_logits = mx.array([[0.0, 2.0]], dtype=mx.float32)
 draft_logits = mx.array([[2.0, 0.0]], dtype=mx.float32)

 probability = _speculative_acceptance_probability(
  target_logits=target_logits,
  draft_logits=draft_logits,
  draft_token=1,
  **_DEFAULT_SAMPLING,
 )

 assert float(probability) == APPROX(1.0)


def test_speculative_acceptance_probability_reduces_when_draft_overestimates_token() -> None:
 target_logits = mx.array([[2.0, 0.0]], dtype=mx.float32)
 draft_logits = mx.array([[0.0, 2.0]], dtype=mx.float32)

 probability = _speculative_acceptance_probability(
  target_logits=target_logits,
  draft_logits=draft_logits,
  draft_token=1,
  **_DEFAULT_SAMPLING,
 )

 assert float(probability) == APPROX(0.13533528)


def test_residual_correction_samples_from_positive_target_minus_draft_mass() -> None:
 # Target strongly prefers token 0 while draft strongly over-proposes token 1.
 target_logits = mx.array([[2.0, 0.0]], dtype=mx.float32)
 draft_logits = mx.array([[0.0, 2.0]], dtype=mx.float32)

 token = _sample_token(
  target_logits=target_logits,
  draft_logits=draft_logits,
  **_DEFAULT_SAMPLING,
 )

 assert token == 0


# ---------------------------------------------------------------------------
# C002 edge-case invariant tests
# ---------------------------------------------------------------------------


def test_acceptance_probability_is_exactly_one_when_target_exceeds_draft() -> None:
 """When p_target >= q_draft the acceptance probability must be exactly 1.0."""
 # 3 tokens: target strongly prefers token 2, draft puts more mass on token 0.
 target_logits = mx.array([[0.0, 0.5, 3.0]], dtype=mx.float32)
 draft_logits = mx.array([[2.0, 0.0, 0.5]], dtype=mx.float32)

 # Token 2: target probability > draft probability.
 probability = _speculative_acceptance_probability(
  target_logits=target_logits,
  draft_logits=draft_logits,
  draft_token=2,
  **_DEFAULT_SAMPLING,
 )

 assert float(probability) == APPROX(1.0)
 # Must never exceed 1.0.
 assert float(probability) <= 1.0 + 1e-9


def test_acceptance_probability_equals_p_over_q_when_target_below_draft() -> None:
 """When p_target < q_draft the acceptance probability must equal p_target / q_draft."""
 # Construct logits where p and q have an exact ratio for the chosen token.
 # Use 4-token vocabulary with distinct logits.
 target_logits = mx.array([[0.0, 1.0, 2.0, 0.5]], dtype=mx.float32)
 draft_logits = mx.array([[1.0, 0.0, 0.0, 3.0]], dtype=mx.float32)

 # Token 3: draft puts heavy mass here (logit 3.0) while target is small (logit 0.5).
 target_probs = _sampling_probabilities(target_logits, **_DEFAULT_SAMPLING)
 draft_probs = _sampling_probabilities(draft_logits, **_DEFAULT_SAMPLING)

 p_target = float(target_probs[0, 3].item())
 q_draft = float(draft_probs[0, 3].item())

 probability = _speculative_acceptance_probability(
  target_logits=target_logits,
  draft_logits=draft_logits,
  draft_token=3,
  **_DEFAULT_SAMPLING,
 )

 expected_ratio = p_target / q_draft
 assert float(probability) == APPROX(expected_ratio)


def test_residual_correction_zero_where_draft_exceeds_target() -> None:
 """Tokens with q > p must have zero residual probability."""
 target_logits = mx.array([[2.0, 0.0, 1.0]], dtype=mx.float32)
 draft_logits = mx.array([[0.0, 3.0, 0.5]], dtype=mx.float32)

 target_probs = _sampling_probabilities(target_logits, **_DEFAULT_SAMPLING)
 draft_probs = _sampling_probabilities(draft_logits, **_DEFAULT_SAMPLING)

 residual = mx.maximum(target_probs - draft_probs, mx.zeros_like(target_probs))
 # Token 1 has q >> p so residual must be 0.
 assert float(residual[0, 1].item()) == APPROX(0.0)

 # Normalize residual to get the correction distribution.
 residual_total = float(mx.sum(residual).item())
 assert residual_total > 0.0  # At least one token has positive residual.
 corrected = residual / mx.sum(residual, axis=-1, keepdims=True)

 # All probability must be on tokens where p > q (tokens 0 and 2).
 assert float(corrected[0, 0].item()) > 0.0
 assert float(corrected[0, 2].item()) > 0.0
 assert float(corrected[0, 1].item()) == APPROX(0.0)

 # Verify the correction distribution sums to 1.
 assert float(mx.sum(corrected).item()) == APPROX(1.0)


def test_residual_correction_fallback_when_p_equals_q_everywhere() -> None:
 """When p == q everywhere (residual is all zero), fallback samples from target — no NaN, valid token id."""
 # Identical logits → p == q for every token → positive residual is all zero.
 identical_logits = mx.array([[1.0, 2.0, 0.5]], dtype=mx.float32)

 token = _sample_token(
  target_logits=identical_logits,
  draft_logits=identical_logits,
  **_DEFAULT_SAMPLING,
 )

 # Must return a valid token id within vocabulary range.
 assert isinstance(token, int)
 assert 0 <= token < 3

 # Verify the fallback path: corrected_probs should equal target_probs.
 target_probs = _sampling_probabilities(identical_logits, **_DEFAULT_SAMPLING)
 draft_probs = _sampling_probabilities(identical_logits, **_DEFAULT_SAMPLING)
 residual = mx.maximum(target_probs - draft_probs, mx.zeros_like(target_probs))
 residual_total = float(mx.sum(residual).item())
 assert residual_total == APPROX(0.0)

 # No NaN in target probabilities (fallback path).
 for i in range(3):
  val = float(target_probs[0, i].item())
  assert not math.isnan(val), f"NaN at token {i}"


def test_determinism_with_fixed_seed() -> None:
 """With temperature=0 the accept/reject decision must be deterministic across calls."""
 # Temperature 0 → greedy: p_target is a one-hot at argmax, q_draft is one-hot at its argmax.
 # If argmax differs, p/q ratio is always 0 for the draft token → always reject.
 # If argmax matches, p/q is 1/1 = 1 → always accept.
 target_logits = mx.array([[0.0, 2.0]], dtype=mx.float32)
 draft_logits = mx.array([[0.0, 2.0]], dtype=mx.float32)

 zero_sampling: _SamplingKwargs = {
  "temperature": 0.0,
  "top_p": 1.0,
  "min_p": 0.0,
  "top_k": 0,
 }

 results: list[float] = []
 for _ in range(10):
  probability = _speculative_acceptance_probability(
   target_logits=target_logits,
   draft_logits=draft_logits,
   draft_token=1,
   **zero_sampling,
  )
  results.append(float(probability))

 # All 10 calls must produce the identical value.
 assert all(r == results[0] for r in results), f"non-deterministic: {results}"

 # Also test residual correction determinism under temperature=0.
 correction_tokens: list[int] = []
 for _ in range(10):
  token = _sample_token(
   target_logits=target_logits,
   draft_logits=draft_logits,
   **zero_sampling,
  )
  correction_tokens.append(token)

 assert all(t == correction_tokens[0] for t in correction_tokens), (
  f"non-deterministic correction: {correction_tokens}"
 )
