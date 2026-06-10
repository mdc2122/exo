# Unknown-type rules are suppressed file-wide at the untyped mlx_lm boundary
# (same precedent as mtp_generate.py: per-site ignores oscillate between
# required and unnecessary across basedpyright runs).
# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false
"""GLM 5.1 adapter for the MTP DraftModel protocol.

GLM 5.1 (glm_moe_dsa = deepseek_v32 in mlx_lm) declares one nextn predict
layer: a full DSA decoder layer (MLA attention + indexer, 256-expert MoE)
wrapped in the DeepSeek-V3 MTP head pattern — hnorm/enorm/eh_proj in front,
shared_head.norm behind, embedding and lm_head shared with the base model.

Unlike MiMo's hand-written SWA draft layer, this adapter reuses mlx_lm's
DeepseekV32DecoderLayer so attention semantics (MLA absorption, indexer
top-k, rope scaling) stay exactly the base model's. The sidecar safetensors
uses mlx_lm naming (.scales/.biases siblings) under the prefix
`mtp.layers.0.` — see scripts/glm51_mtp_sidecar_export.py.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import CacheList, KVCache
from mlx_lm.models.deepseek_v32 import DeepseekV32DecoderLayer
from mlx_lm.models.deepseek_v32 import ModelArgs as Dsv32ModelArgs
from mlx_lm.models.glm_moe_dsa import ModelArgs

from exo.worker.engines.mlx.mimo_mtp_fast.draft_model import Sampler

_GLM_SIDECAR_PREFIX = "mtp.layers.0."
_GLM_QUANT_GROUP_SIZE = 64
# Mirror of the base export's per-tensor rules (see exporter docstring).
_GLM_8BIT_MODULES = frozenset(
    {
        "self_attn.q_a_proj",
        "self_attn.q_b_proj",
        "self_attn.kv_a_proj_with_mqa",
        "self_attn.o_proj",
        "self_attn.embed_q",
        "self_attn.unembed_out",
        "self_attn.indexer.wq_b",
        "self_attn.indexer.wk",
        "self_attn.indexer.weights_proj",
        "mlp.switch_mlp.gate_proj",
        "mlp.shared_experts.gate_proj",
        "mlp.shared_experts.up_proj",
        "mlp.shared_experts.down_proj",
    }
)
_GLM_6BIT_MODULES = frozenset(
    {
        "mlp.switch_mlp.up_proj",
        "mlp.switch_mlp.down_proj",
        "eh_proj",
    }
)


class _TokenEmbedding(Protocol):
    def __call__(self, token_ids: mx.array) -> mx.array: ...


class _LmHead(Protocol):
    def __call__(self, hidden: mx.array) -> mx.array: ...


class _GlmBaseLanguageModel(Protocol):
    embed_tokens: _TokenEmbedding


class _GlmBaseModel(Protocol):
    args: ModelArgs
    model: _GlmBaseLanguageModel
    lm_head: _LmHead


class GlmMtpDraftCache:
    """DraftCache for the GLM nextn layer: MLA KV latent + indexer cache."""

    def __init__(self) -> None:
        self.cache_list: CacheList = CacheList(KVCache(), KVCache())

    def reset(self) -> None:
        self.cache_list = CacheList(KVCache(), KVCache())


class _SharedHead(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)


class GlmNextnLayer(DeepseekV32DecoderLayer):
    """DSV32 decoder layer plus the DeepSeek-V3 MTP head extras.

    Attribute names line up with the sidecar's relative tensor names so
    load_weights(strict=True) validates the full inventory.
    """

    def __init__(self, args: ModelArgs) -> None:
        # layer_idx >= first_k_dense_replace selects the MoE MLP variant,
        # matching the nextn layer's weights. glm_moe_dsa ModelArgs is not a
        # subclass of deepseek_v32's, but __post_init__ fills rope_theta /
        # rope_scaling so it is attribute-compatible.
        super().__init__(
            cast(Dsv32ModelArgs, cast(object, args)),
            layer_idx=int(args.num_hidden_layers),
        )
        hidden_size = int(args.hidden_size)
        eps = float(args.rms_norm_eps)
        self.enorm = nn.RMSNorm(hidden_size, eps=eps)
        self.hnorm = nn.RMSNorm(hidden_size, eps=eps)
        self.eh_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.shared_head = _SharedHead(hidden_size, eps)


class GlmMtpDraftModel(nn.Module):
    """Implements the DraftModel protocol for GLM 5.1's single nextn layer."""

    def __init__(
        self,
        *,
        nextn_layer: GlmNextnLayer,
        embed_tokens: _TokenEmbedding,
        lm_head: _LmHead,
    ) -> None:
        super().__init__()
        self.nextn_layer = nextn_layer
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head

    @property
    def num_draft_layers(self) -> int:
        return 1

    def make_cache(self, *, window_size: int | None = None) -> GlmMtpDraftCache:
        # The nextn layer is full-attention MLA; no windowing needed. The
        # parameter is accepted for protocol compatibility and ignored.
        del window_size
        return GlmMtpDraftCache()

    def reset_cache_on_fallback(self, cache: object) -> None:
        if not isinstance(cache, GlmMtpDraftCache):
            raise TypeError(f"expected GlmMtpDraftCache, got {type(cache).__name__}")
        cache.reset()

    def propose(
        self,
        *,
        previous_hidden_state: mx.array,
        latest_token_ids: mx.array,
        max_draft_tokens: int,
        sampler: Sampler,
        cache: object | None = None,
    ) -> tuple[mx.array, list[mx.array]]:
        if max_draft_tokens < 0:
            raise ValueError("max_draft_tokens must be non-negative")
        if max_draft_tokens > 1:
            raise ValueError(
                f"GLM 5.1 has one nextn layer; max_draft_tokens={max_draft_tokens}"
            )
        if cache is not None and not isinstance(cache, GlmMtpDraftCache):
            raise TypeError(f"expected GlmMtpDraftCache, got {type(cache).__name__}")
        token_history = _as_token_history(latest_token_ids)
        backfill_length = int(previous_hidden_state.shape[1])
        if int(token_history.shape[1]) < backfill_length:
            raise ValueError(
                "token history shorter than hidden-state backfill: "
                f"{int(token_history.shape[1])} < {backfill_length}"
            )
        if max_draft_tokens == 0:
            return mx.zeros(
                (int(token_history.shape[0]), 0), dtype=token_history.dtype
            ), []

        draft_cache = cache if cache is not None else GlmMtpDraftCache()
        # Same input contract as MiMo: hidden state at position t pairs with
        # the embedding of token t+1 over the trailing token window.
        current_token_ids = token_history[:, -backfill_length:]
        token_embedding = self.embed_tokens(current_token_ids)
        embedding_norm = self.nextn_layer.enorm(token_embedding)
        hidden_norm = self.nextn_layer.hnorm(previous_hidden_state)
        # DeepSeek-V3 eh_proj order is [embed, hidden]; GLM 5.1 follows the
        # DSV3 lineage. Env toggle kept for the catalogued A/B trap.
        if os.environ.get("EXO_GLM_MTP_CONCAT_ORDER", "") == "hidden_first":
            eh_proj_input = mx.concatenate([hidden_norm, embedding_norm], axis=-1)
        else:
            eh_proj_input = mx.concatenate([embedding_norm, hidden_norm], axis=-1)
        hidden_state = self.nextn_layer.eh_proj(eh_proj_input)

        sequence_length = int(hidden_state.shape[1])
        attention_mask = (
            create_attention_mask(
                hidden_state, draft_cache.cache_list[0], return_array=True
            )
            if sequence_length > 1
            else None
        )
        layer_output = self.nextn_layer(
            hidden_state, attention_mask, draft_cache.cache_list
        )
        final_hidden = self.nextn_layer.shared_head.norm(layer_output[:, -1:, :])
        logits = self.lm_head(final_hidden)[:, -1, :]
        sampled_token = _as_token_column(sampler(logits))
        return sampled_token, [logits]


def build_glm_mtp_draft_model(
    sidecar_path: Path | str,
    base_model: object,
) -> GlmMtpDraftModel:
    typed_base_model = cast(_GlmBaseModel, base_model)
    nextn_layer = GlmNextnLayer(typed_base_model.args)

    def _quantization_predicate(path: str, module: nn.Module) -> bool | dict[str, int]:
        if not hasattr(module, "to_quantized"):
            return False
        if path in _GLM_8BIT_MODULES:
            return {"bits": 8, "group_size": _GLM_QUANT_GROUP_SIZE}
        if path in _GLM_6BIT_MODULES:
            return {"bits": 6, "group_size": _GLM_QUANT_GROUP_SIZE}
        return False

    nn.quantize(
        nextn_layer,
        group_size=_GLM_QUANT_GROUP_SIZE,
        bits=6,
        class_predicate=_quantization_predicate,
    )

    raw_weights = mx.load(str(sidecar_path))
    if not isinstance(raw_weights, dict):
        raise TypeError(f"sidecar load returned {type(raw_weights).__name__}")
    layer_weights = [
        (name.removeprefix(_GLM_SIDECAR_PREFIX), tensor)
        for name, tensor in raw_weights.items()
        if name.startswith(_GLM_SIDECAR_PREFIX)
    ]
    if not layer_weights:
        raise RuntimeError(f"no {_GLM_SIDECAR_PREFIX}* tensors in {sidecar_path}")
    nextn_layer.load_weights(layer_weights, strict=True)
    mx.eval(nextn_layer.parameters())

    return GlmMtpDraftModel(
        nextn_layer=nextn_layer,
        embed_tokens=typed_base_model.model.embed_tokens,
        lm_head=typed_base_model.lm_head,
    )


def _as_token_history(token_ids: mx.array) -> mx.array:
    if token_ids.ndim == 1:
        return token_ids[:, None]
    if token_ids.ndim == 2:
        return token_ids
    raise ValueError(
        f"latest_token_ids must be rank 1 or 2, got rank {int(token_ids.ndim)}"
    )


def _as_token_column(token_ids: mx.array | int) -> mx.array:
    if isinstance(token_ids, int):
        return mx.array([[token_ids]], dtype=mx.int32)
    if token_ids.ndim == 1:
        return token_ids[:, None]
    if token_ids.ndim == 2 and int(token_ids.shape[1]) == 1:
        return token_ids
    raise ValueError(
        f"sampled token ids must be rank 1 or single-column rank 2, got {token_ids.shape}"
    )
