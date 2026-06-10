from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias, cast

import mlx.core as mx
import mlx.nn as nn

from exo.worker.engines.mlx.mimo_mtp_fast.sidecar_loader import (
    MimoMtpLayerTensors,
    MimoMtpSidecarTensors,
)

MX_FAST_SCALED_DOT_PRODUCT_ATTENTION = cast(
    Callable[..., mx.array],
    mx.fast.scaled_dot_product_attention,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
)
MX_FROM_FP8 = cast(
    Callable[[mx.array, mx.Dtype], mx.array],
    mx.from_fp8,  # pyright: ignore[reportAttributeAccessIssue]
)
MX_QUANTIZED_MATMUL = cast(
    Callable[..., mx.array],
    mx.quantized_matmul,
)

_FP8_BLOCK_SIZE = 128
_MIMO_MTP_QUANTIZED_GROUP_SIZE = 64
_MIMO_MTP_QUANTIZED_BITS = 6
_MIMO_MTP_QUANTIZED_MODE = "affine"


class _TokenEmbedding(Protocol):
    def __call__(self, token_ids: mx.array) -> mx.array: ...


class _LanguageModel(Protocol):
    embed_tokens: _TokenEmbedding


class _LayerArgs(Protocol):
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    v_head_dim: int
    layernorm_epsilon: float
    rope_theta: float
    partial_rotary_factor: float


class _BaseModel(Protocol):
    args: _LayerArgs
    model: _LanguageModel
    lm_head: Callable[[mx.array], mx.array]


class _LinearLike(Protocol):
    def __call__(self, x: mx.array) -> mx.array: ...


Sampler: TypeAlias = Callable[[mx.array], mx.array | int]


@dataclass
class MimoMtpLayerCache:
    keys: mx.array | None = None
    values: mx.array | None = None
    offset: int = 0

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        self.offset += int(keys.shape[2])
        if self.keys is None or self.values is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def reset(self) -> None:
        self.keys = None
        self.values = None
        self.offset = 0


@dataclass
class MimoMtpCache:
    num_layers: int
    layers: list[MimoMtpLayerCache] = field(init=False)

    def __post_init__(self) -> None:
        self.layers = [MimoMtpLayerCache() for _ in range(self.num_layers)]

    def reset(self) -> None:
        for layer_cache in self.layers:
            layer_cache.reset()


class DenseLinear(nn.Module):
    def __init__(self, weight: mx.array) -> None:
        super().__init__()
        self.weight = weight

    def __call__(self, x: mx.array) -> mx.array:
        return mx.matmul(x.astype(self.weight.dtype), self.weight.T)


class Fp8BlockLinear(nn.Module):
    def __init__(self, *, weight: mx.array, weight_scale_inv: mx.array) -> None:
        super().__init__()
        self.weight = _dequantize_fp8_block_weight(weight, weight_scale_inv)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.matmul(x.astype(self.weight.dtype), self.weight.T)


class QuantizedAffineLinear(nn.Module):
    def __init__(
        self,
        *,
        weight: mx.array,
        scales: mx.array,
        biases: mx.array,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.scales = scales
        self.biases = biases

    def __call__(self, x: mx.array) -> mx.array:
        return MX_QUANTIZED_MATMUL(
            x,
            self.weight,
            scales=self.scales,
            biases=self.biases,
            transpose=True,
            group_size=_MIMO_MTP_QUANTIZED_GROUP_SIZE,
            bits=_MIMO_MTP_QUANTIZED_BITS,
            mode=_MIMO_MTP_QUANTIZED_MODE,
        )


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _expanded_block_scales(weight: mx.array, weight_scale_inv: mx.array) -> mx.array:
    rows = int(weight.shape[0])
    cols = int(weight.shape[1])
    row_blocks = _ceil_div(rows, _FP8_BLOCK_SIZE)
    col_blocks = _ceil_div(cols, _FP8_BLOCK_SIZE)
    scale_rows = int(weight_scale_inv.shape[0])
    if scale_rows > row_blocks:
        # Per-row-group scale layout (MiMo fused qkv: one 128-block grid per
        # KV-head group whose stride is not a multiple of 128). Top-aligned
        # expansion mis-scales everything past the first group boundary.
        group_count = next(
            (
                candidate
                for candidate in range(2, scale_rows + 1)
                if rows % candidate == 0
                and candidate * _ceil_div(rows // candidate, _FP8_BLOCK_SIZE)
                == scale_rows
            ),
            None,
        )
        if group_count is None:
            raise ValueError(
                "FP8 scale rows do not match a uniform or grouped 128-block "
                f"layout: scales={tuple(weight_scale_inv.shape)} "
                f"weight={tuple(weight.shape)}"
            )
        rows_per_group = rows // group_count
        scale_rows_per_group = scale_rows // group_count
        row_expanded = mx.concatenate(
            [
                mx.repeat(
                    weight_scale_inv[
                        group * scale_rows_per_group : (group + 1)
                        * scale_rows_per_group,
                        :col_blocks,
                    ],
                    _FP8_BLOCK_SIZE,
                    axis=0,
                )[:rows_per_group]
                for group in range(group_count)
            ],
            axis=0,
        )
    else:
        row_expanded = mx.repeat(
            weight_scale_inv[:row_blocks, :col_blocks], _FP8_BLOCK_SIZE, axis=0
        )[:rows]
    expanded = mx.repeat(row_expanded, _FP8_BLOCK_SIZE, axis=1)
    return expanded[:rows, :cols]


def _dequantize_fp8_block_weight(
    weight: mx.array, weight_scale_inv: mx.array
) -> mx.array:
    dequantized = MX_FROM_FP8(weight.astype(mx.uint8), mx.bfloat16)
    scales = _expanded_block_scales(weight, weight_scale_inv).astype(mx.bfloat16)
    return dequantized * scales


def _split_qkv(
    qkv: mx.array,
    *,
    q_size: int,
    k_size: int,
    v_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    value_head_dim: int,
) -> tuple[mx.array, mx.array, mx.array]:
    grouped_width = q_size + k_size + v_size
    if int(qkv.shape[-1]) != grouped_width:
        raise ValueError(
            "MTP qkv projection shape does not match configured head sizes: "
            f"expected {grouped_width}, got {int(qkv.shape[-1])}"
        )

    # MiMo V2.5 Pro official MTP stores qkv grouped by KV head:
    # [q heads for group, one k head, one v head] * num_key_value_heads.
    if num_attention_heads % num_key_value_heads == 0:
        batch_size, sequence_length, _width = qkv.shape
        q_heads_per_kv = num_attention_heads // num_key_value_heads
        q_group_size = q_heads_per_kv * head_dim
        group_size = q_group_size + head_dim + value_head_dim
        if group_size * num_key_value_heads == grouped_width:
            grouped = qkv.reshape(
                batch_size, sequence_length, num_key_value_heads, group_size
            )
            queries = grouped[..., :q_group_size].reshape(
                batch_size, sequence_length, q_size
            )
            keys = grouped[..., q_group_size : q_group_size + head_dim].reshape(
                batch_size, sequence_length, k_size
            )
            values = grouped[..., q_group_size + head_dim :].reshape(
                batch_size, sequence_length, v_size
            )
            return queries, keys, values

    queries, keys, values = cast(
        tuple[mx.array, mx.array, mx.array],
        cast(object, mx.split(qkv, [q_size, q_size + k_size], axis=-1)),
    )
    return queries, keys, values


class MimoMtpAttention(nn.Module):
    def __init__(
        self,
        *,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        value_head_dim: int,
        rope_theta: float,
        partial_rotary_factor: float,
        qkv_proj: _LinearLike,
        o_proj: _LinearLike,
        attention_sink_bias: mx.array | None,
    ) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.value_head_dim = value_head_dim
        self.scale: float = 1.0 / (float(head_dim) ** 0.5)
        self.qkv_proj = qkv_proj
        self.o_proj = o_proj
        self.attention_sink_bias = attention_sink_bias
        self.rope = nn.RoPE(
            int(partial_rotary_factor * head_dim),
            traditional=False,
            base=rope_theta,
        )

    @property
    def q_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def k_size(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def v_size(self) -> int:
        return self.num_key_value_heads * self.value_head_dim

    def __call__(
        self, x: mx.array, *, cache: MimoMtpLayerCache | None = None
    ) -> mx.array:
        batch_size, sequence_length, _hidden_width = x.shape
        qkv = self.qkv_proj(x)
        queries, keys, values = _split_qkv(
            qkv,
            q_size=self.q_size,
            k_size=self.k_size,
            v_size=self.v_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            value_head_dim=self.value_head_dim,
        )
        queries = queries.reshape(
            batch_size, sequence_length, self.num_attention_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        keys = keys.reshape(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch_size, sequence_length, self.num_key_value_heads, self.value_head_dim
        ).transpose(0, 2, 1, 3)
        if cache is None:
            queries = self.rope(queries)
            keys = self.rope(keys)
            full_keys = keys
            full_values = values
        else:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            full_keys, full_values = cache.update_and_fetch(keys, values)

        attention_output = MX_FAST_SCALED_DOT_PRODUCT_ATTENTION(
            queries,
            full_keys,
            full_values,
            scale=self.scale,
            sinks=self.attention_sink_bias,
        )
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(
            batch_size,
            sequence_length,
            self.num_attention_heads * self.value_head_dim,
        )
        return self.o_proj(attention_output)


class MimoMtpMLP(nn.Module):
    def __init__(
        self, *, gate_proj: _LinearLike, up_proj: _LinearLike, down_proj: _LinearLike
    ) -> None:
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.gate_proj(x)
        return self.down_proj((gate * mx.sigmoid(gate)) * self.up_proj(x))


class MimoMtpLayer(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        hnorm: nn.RMSNorm,
        enorm: nn.RMSNorm,
        eh_proj: _LinearLike,
        self_attn: MimoMtpAttention,
        input_layernorm: nn.RMSNorm,
        pre_mlp_layernorm: nn.RMSNorm,
        mlp: MimoMtpMLP,
        final_layernorm: nn.RMSNorm,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.hnorm = hnorm
        self.enorm = enorm
        self.eh_proj = eh_proj
        self.self_attn = self_attn
        self.input_layernorm = input_layernorm
        self.pre_mlp_layernorm = pre_mlp_layernorm
        self.mlp = mlp
        self.final_layernorm = final_layernorm

    def __call__(
        self,
        previous_hidden_state: mx.array,
        token_embedding: mx.array,
        *,
        cache: MimoMtpLayerCache | None = None,
    ) -> mx.array:
        hidden_norm = self.hnorm(previous_hidden_state)
        embedding_norm = self.enorm(token_embedding)
        # MiMo-V2.5's eh_proj consumes [embed, hidden] (DeepSeek-V3 order) —
        # verified live 2026-06-09: 28/36 depth-1 acceptances vs 0/63 with the
        # [hidden, embed] order MTPLX uses for MiMo-7B. Env toggle kept for A/B.
        if os.environ.get("EXO_MIMO_MTP_CONCAT_ORDER", "") == "hidden_first":
            eh_proj_input = mx.concatenate([hidden_norm, embedding_norm], axis=-1)
        else:
            eh_proj_input = mx.concatenate([embedding_norm, hidden_norm], axis=-1)
        hidden_state = self.eh_proj(eh_proj_input)
        hidden_state = hidden_state + self.self_attn(
            self.input_layernorm(hidden_state), cache=cache
        )
        hidden_state = hidden_state + self.mlp(self.pre_mlp_layernorm(hidden_state))
        output = self.final_layernorm(hidden_state)
        if int(output.shape[-1]) != self.hidden_size:
            raise ValueError(
                "MTP layer output width mismatch: "
                f"expected {self.hidden_size}, got {int(output.shape[-1])}"
            )
        return output.astype(previous_hidden_state.dtype)


class MimoMtpStack(nn.Module):
    def __init__(
        self,
        *,
        layers: Sequence[MimoMtpLayer],
        embed_tokens: _TokenEmbedding,
        lm_head: Callable[[mx.array], mx.array],
    ) -> None:
        super().__init__()
        self.layers = list(layers)
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head

    def make_cache(self) -> MimoMtpCache:
        return MimoMtpCache(num_layers=len(self.layers))

    def reset_cache_on_fallback(self, cache: object) -> None:
        if not isinstance(cache, MimoMtpCache):
            raise TypeError(f"expected MimoMtpCache, got {type(cache).__name__}")
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
        if max_draft_tokens > len(self.layers):
            raise ValueError(
                f"max_draft_tokens exceeds available MTP layers: {max_draft_tokens} > {len(self.layers)}"
            )
        if cache is not None and not isinstance(cache, MimoMtpCache):
            raise TypeError(f"expected MimoMtpCache, got {type(cache).__name__}")
        mtp_cache = self.make_cache() if cache is None else cache
        token_history = _as_token_history(latest_token_ids)
        current_token_ids = token_history[:, -1:]
        current_hidden_state = previous_hidden_state
        sampled_tokens: list[mx.array] = []
        per_layer_logits: list[mx.array] = []

        for layer_index, layer in enumerate(self.layers[:max_draft_tokens]):
            token_embedding = self.embed_tokens(current_token_ids)
            current_hidden_state = layer(
                current_hidden_state,
                token_embedding,
                cache=mtp_cache.layers[layer_index],
            )
            logits = self.lm_head(current_hidden_state)[:, -1, :]
            sampled_token = _as_token_column(sampler(logits))
            sampled_tokens.append(sampled_token)
            per_layer_logits.append(logits)
            token_history = mx.concatenate([token_history, sampled_token], axis=1)
            current_token_ids = sampled_token

        if not sampled_tokens:
            return mx.zeros(
                (int(token_history.shape[0]), 0), dtype=token_history.dtype
            ), []
        return mx.concatenate(sampled_tokens, axis=1), per_layer_logits


def build_mimo_mtp_stack(
    sidecar: MimoMtpSidecarTensors,
    base_model: object,
) -> MimoMtpStack:
    typed_base_model = cast(_BaseModel, base_model)
    args = typed_base_model.args
    layers = tuple(
        _build_layer(layer_tensors=layer_tensors, args=args)
        for layer_tensors in sidecar.layers
    )
    return MimoMtpStack(
        layers=layers,
        embed_tokens=typed_base_model.model.embed_tokens,
        lm_head=typed_base_model.lm_head,
    )


def _build_layer(
    *, layer_tensors: MimoMtpLayerTensors, args: _LayerArgs
) -> MimoMtpLayer:
    hidden_size = int(args.hidden_size)
    return MimoMtpLayer(
        hidden_size=hidden_size,
        hnorm=_make_rmsnorm(
            layer_tensors.get("hnorm.weight"),
            hidden_size,
            float(args.layernorm_epsilon),
        ),
        enorm=_make_rmsnorm(
            layer_tensors.get("enorm.weight"),
            hidden_size,
            float(args.layernorm_epsilon),
        ),
        eh_proj=_make_quantized_linear(layer_tensors, "eh_proj"),
        self_attn=MimoMtpAttention(
            num_attention_heads=int(args.num_attention_heads),
            num_key_value_heads=int(args.num_key_value_heads),
            head_dim=int(args.head_dim),
            value_head_dim=int(args.v_head_dim),
            rope_theta=float(args.rope_theta),
            partial_rotary_factor=float(args.partial_rotary_factor),
            qkv_proj=_make_quantized_linear(layer_tensors, "self_attn.qkv_proj"),
            o_proj=_make_quantized_linear(layer_tensors, "self_attn.o_proj"),
            attention_sink_bias=layer_tensors.get("self_attn.attention_sink_bias"),
        ),
        input_layernorm=_make_rmsnorm(
            layer_tensors.get("input_layernorm.weight"),
            hidden_size,
            float(args.layernorm_epsilon),
        ),
        pre_mlp_layernorm=_make_rmsnorm(
            layer_tensors.get("pre_mlp_layernorm.weight"),
            hidden_size,
            float(args.layernorm_epsilon),
        ),
        mlp=MimoMtpMLP(
            gate_proj=_make_quantized_linear(layer_tensors, "mlp.gate_proj"),
            up_proj=_make_quantized_linear(layer_tensors, "mlp.up_proj"),
            down_proj=_make_quantized_linear(layer_tensors, "mlp.down_proj"),
        ),
        final_layernorm=_make_rmsnorm(
            layer_tensors.get("final_layernorm.weight"),
            hidden_size,
            float(args.layernorm_epsilon),
        ),
    )


def _make_quantized_linear(
    layer_tensors: MimoMtpLayerTensors, prefix: str
) -> _LinearLike:
    weight, scales, biases = layer_tensors.quantized_linear_tensors(prefix)
    if weight.dtype != mx.uint32:
        return DenseLinear(weight)
    return QuantizedAffineLinear(weight=weight, scales=scales, biases=biases)


def _make_rmsnorm(weight: mx.array, hidden_size: int, eps: float) -> nn.RMSNorm:
    norm = nn.RMSNorm(hidden_size, eps=eps)
    norm.weight = weight
    return norm


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
