"""Model-neutral draft-model contract for MTP speculative decoding.

The decode engine in mtp_generate.py (batched verify, chained leftmost-prefix
acceptance, cache rewind, persistent draft context) is model-agnostic. What
varies per model family is exactly three things: which tensors form the draft
layers (sidecar loader), how those layers are configured (rope theta, head
geometry, eh_proj concat order), and how the target cache is laid out. This
module defines the seam: any model exposing a DraftModel can be served by the
engine. MimoMtpStack (sidecar_module.py) is the first adapter; a GLM 5.1
adapter is next.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeAlias

import mlx.core as mx

Sampler: TypeAlias = Callable[[mx.array], mx.array | int]


class DraftCache(Protocol):
    """Per-request mutable draft-layer cache; reset drops all entries."""

    def reset(self) -> None: ...


class DraftModel(Protocol):
    """What the MTP decode engine requires of a draft model.

    Implementations draft up to num_draft_layers tokens per call by chaining
    their layers, returning the sampled draft tokens alongside the raw logits
    each token was sampled from (the q distributions for p/q acceptance).
    """

    @property
    def num_draft_layers(self) -> int: ...

    def make_cache(self, *, window_size: int | None = None) -> DraftCache: ...

    def reset_cache_on_fallback(self, cache: object) -> None: ...

    def propose(
        self,
        *,
        previous_hidden_state: mx.array,
        latest_token_ids: mx.array,
        max_draft_tokens: int,
        sampler: Sampler,
        cache: object | None = None,
    ) -> tuple[mx.array, list[mx.array]]: ...


@dataclass(frozen=True)
class DraftLayerConfig:
    """Attention/projection geometry for one family's MTP draft layers.

    Adapters derive this from the base model's args; the derivation is where
    family-specific traps live (e.g. MiMo's MTP layers are SWA-style, so
    rope_theta here must be swa_rope_theta, not the full-attention theta).
    """

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    value_head_dim: int
    layernorm_epsilon: float
    rope_theta: float
    partial_rotary_factor: float
