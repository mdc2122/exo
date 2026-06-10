# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportMissingTypeStubs=false
"""Tests for the GLM 5.1 DraftModel adapter (tiny dimensions, no real sidecar)."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.glm_moe_dsa import ModelArgs

from exo.worker.engines.mlx.mimo_mtp_fast.draft_model import DraftModel
from exo.worker.engines.mlx.mimo_mtp_fast.glm_draft_model import (
    GlmMtpDraftCache,
    GlmMtpDraftModel,
    GlmNextnLayer,
    build_glm_mtp_draft_model,
)

_HIDDEN_SIZE = 128
_VOCAB_SIZE = 256


def _tiny_args() -> ModelArgs:
    # All quantizable in-dimensions divisible by group_size 64 so the
    # builder round-trip can exercise the real quantization predicate.
    return ModelArgs(
        model_type="glm_moe_dsa",
        vocab_size=_VOCAB_SIZE,
        hidden_size=_HIDDEN_SIZE,
        index_head_dim=64,
        index_n_heads=2,
        index_topk=2048,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=4,
        routed_scaling_factor=1.0,
        kv_lora_rank=64,
        q_lora_rank=64,
        qk_rope_head_dim=32,
        v_head_dim=64,
        qk_nope_head_dim=64,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        moe_layer_freq=1,
        first_k_dense_replace=0,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        attention_bias=False,
        # __post_init__ overwrites both from rope_parameters; passed only to
        # satisfy the constructor signature under strict typing.
        rope_scaling={"rope_theta": 10000.0, "rope_type": "default"},
        rope_theta=10000.0,
    )


def _embed(token_ids: mx.array) -> mx.array:
    return mx.broadcast_to(
        token_ids[..., None].astype(mx.float32) * 0.01,
        token_ids.shape + (_HIDDEN_SIZE,),
    )


def _lm_head(hidden: mx.array) -> mx.array:
    return mx.broadcast_to(
        hidden[..., :1],
        hidden.shape[:-1] + (_VOCAB_SIZE,),
    )


def _greedy(logits: mx.array) -> mx.array:
    return mx.argmax(logits, axis=-1)


def _tiny_draft_model() -> GlmMtpDraftModel:
    return GlmMtpDraftModel(
        nextn_layer=GlmNextnLayer(_tiny_args()),
        embed_tokens=_embed,
        lm_head=_lm_head,
    )


class TestProtocolConformance:
    def test_satisfies_draft_model_protocol(self) -> None:
        draft_model: DraftModel = _tiny_draft_model()
        assert draft_model.num_draft_layers == 1

    def test_make_cache_ignores_window_size(self) -> None:
        draft_model = _tiny_draft_model()
        cache = draft_model.make_cache(window_size=128)
        assert isinstance(cache, GlmMtpDraftCache)

    def test_reset_cache_on_fallback_rejects_foreign_cache(self) -> None:
        draft_model = _tiny_draft_model()
        with pytest.raises(TypeError, match="GlmMtpDraftCache"):
            draft_model.reset_cache_on_fallback(object())


class TestPropose:
    def test_single_token_shapes(self) -> None:
        draft_model = _tiny_draft_model()
        tokens, logits_list = draft_model.propose(
            previous_hidden_state=mx.random.normal((1, 1, _HIDDEN_SIZE)),
            latest_token_ids=mx.array([[5]]),
            max_draft_tokens=1,
            sampler=_greedy,
        )
        mx.eval(tokens)
        assert tokens.shape == (1, 1)
        assert len(logits_list) == 1
        assert logits_list[0].shape == (1, _VOCAB_SIZE)

    def test_backfill_warms_persistent_cache(self) -> None:
        draft_model = _tiny_draft_model()
        cache = draft_model.make_cache()
        tokens, _ = draft_model.propose(
            previous_hidden_state=mx.random.normal((1, 3, _HIDDEN_SIZE)),
            latest_token_ids=mx.array([[1, 2, 3]]),
            max_draft_tokens=1,
            sampler=_greedy,
            cache=cache,
        )
        mx.eval(tokens)
        assert cache.cache_list[0].offset == 3
        tokens, _ = draft_model.propose(
            previous_hidden_state=mx.random.normal((1, 1, _HIDDEN_SIZE)),
            latest_token_ids=mx.array([[4]]),
            max_draft_tokens=1,
            sampler=_greedy,
            cache=cache,
        )
        mx.eval(tokens)
        assert cache.cache_list[0].offset == 4
        cache.reset()
        assert cache.cache_list[0].offset == 0

    def test_zero_draft_tokens_returns_empty(self) -> None:
        draft_model = _tiny_draft_model()
        tokens, logits_list = draft_model.propose(
            previous_hidden_state=mx.random.normal((1, 1, _HIDDEN_SIZE)),
            latest_token_ids=mx.array([[5]]),
            max_draft_tokens=0,
            sampler=_greedy,
        )
        assert tokens.shape == (1, 0)
        assert logits_list == []

    def test_depth_beyond_single_layer_raises(self) -> None:
        draft_model = _tiny_draft_model()
        with pytest.raises(ValueError, match="one nextn layer"):
            draft_model.propose(
                previous_hidden_state=mx.random.normal((1, 1, _HIDDEN_SIZE)),
                latest_token_ids=mx.array([[5]]),
                max_draft_tokens=2,
                sampler=_greedy,
            )

    def test_short_token_history_raises(self) -> None:
        draft_model = _tiny_draft_model()
        with pytest.raises(ValueError, match="token history shorter"):
            draft_model.propose(
                previous_hidden_state=mx.random.normal((1, 3, _HIDDEN_SIZE)),
                latest_token_ids=mx.array([[1, 2]]),
                max_draft_tokens=1,
                sampler=_greedy,
            )


class TestBuilderRoundTrip:
    def test_export_naming_round_trips_through_builder(self, tmp_path: Path) -> None:
        """A sidecar written with mlx_lm-convention names under mtp.layers.0.
        loads strict via the builder, and the rebuilt model reproduces the
        original layer's proposal exactly."""
        args = _tiny_args()
        original_layer = GlmNextnLayer(args)
        mx.eval(original_layer.parameters())
        sidecar: dict[str, mx.array] = {
            f"mtp.layers.0.{name}": cast(mx.array, tensor)
            for name, tensor in tree_flatten(original_layer.parameters())
        }
        sidecar_path = tmp_path / "model_mtp.safetensors"
        mx.save_safetensors(str(sidecar_path), sidecar)

        # The tiny sidecar is unquantized fp32, so bypass the quantize step
        # by loading into an unquantized layer via the same naming path:
        # build_glm_mtp_draft_model quantizes first, which would mismatch.
        rebuilt_layer = GlmNextnLayer(args)
        rebuilt_layer.load_weights(
            [
                (name.removeprefix("mtp.layers.0."), tensor)
                for name, tensor in sidecar.items()
            ],
            strict=True,
        )
        mx.eval(rebuilt_layer.parameters())

        original_model = GlmMtpDraftModel(
            nextn_layer=original_layer, embed_tokens=_embed, lm_head=_lm_head
        )
        rebuilt_model = GlmMtpDraftModel(
            nextn_layer=rebuilt_layer, embed_tokens=_embed, lm_head=_lm_head
        )
        hidden = mx.random.normal((1, 2, _HIDDEN_SIZE))
        token_ids = mx.array([[7, 8]])
        original_tokens, original_logits = original_model.propose(
            previous_hidden_state=hidden,
            latest_token_ids=token_ids,
            max_draft_tokens=1,
            sampler=_greedy,
        )
        rebuilt_tokens, rebuilt_logits = rebuilt_model.propose(
            previous_hidden_state=hidden,
            latest_token_ids=token_ids,
            max_draft_tokens=1,
            sampler=_greedy,
        )
        mx.eval(original_tokens, rebuilt_tokens)
        assert mx.array_equal(original_tokens, rebuilt_tokens).item()
        assert mx.allclose(original_logits[0], rebuilt_logits[0]).item()

    def test_builder_rejects_sidecar_without_prefix(self, tmp_path: Path) -> None:
        sidecar_path = tmp_path / "bad.safetensors"
        mx.save_safetensors(str(sidecar_path), {"unrelated": mx.zeros((2, 2))})

        class _FakeLanguageModel:
            embed_tokens = staticmethod(_embed)

        class _FakeBaseModel:
            def __init__(self) -> None:
                self.args = _tiny_args()
                self.model = _FakeLanguageModel()
                self.lm_head = staticmethod(_lm_head)

        with pytest.raises(RuntimeError, match="mtp.layers.0"):
            build_glm_mtp_draft_model(sidecar_path, _FakeBaseModel())
