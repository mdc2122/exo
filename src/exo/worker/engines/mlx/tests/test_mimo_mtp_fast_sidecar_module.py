from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import mlx.core as mx

from exo.worker.engines.mlx.mimo_mtp_fast.sidecar_loader import (
    MimoMtpLayerTensors,
    MimoMtpSidecarTensors,
)
from exo.worker.engines.mlx.mimo_mtp_fast.sidecar_module import (
    Fp8BlockLinear,
    MimoMtpCache,
    build_mimo_mtp_stack,
)

MX_TO_FP8 = cast(
    Callable[[mx.array], mx.array],
    mx.to_fp8,  # pyright: ignore[reportAttributeAccessIssue]
)


@dataclass(frozen=True)
class _TinyArgs:
    hidden_size: int = 2
    intermediate_size: int = 4
    num_attention_heads: int = 1
    num_key_value_heads: int = 1
    head_dim: int = 2
    v_head_dim: int = 2
    layernorm_epsilon: float = 1e-5
    rope_theta: float = 10_000.0
    partial_rotary_factor: float = 1.0


class _TinyEmbedding:
    def __call__(self, token_ids: mx.array) -> mx.array:
        values = token_ids.astype(mx.float32)
        return mx.stack([values, values + 1.0], axis=-1)


@dataclass(frozen=True)
class _TinyInnerModel:
    embed_tokens: _TinyEmbedding


class _TinyBaseModel:
    def __init__(self) -> None:
        self.args = _TinyArgs()
        self.model = _TinyInnerModel(embed_tokens=_TinyEmbedding())
        self.lm_head_calls = 0

    def lm_head(self, hidden: mx.array) -> mx.array:
        del hidden
        self.lm_head_calls += 1
        logits = mx.zeros((1, 1, 16), dtype=mx.float32)
        logits[:, :, 3 + self.lm_head_calls] = 1_000.0
        return logits


def _fp8_weight(shape: tuple[int, int], value: float = 0.0) -> mx.array:
    return MX_TO_FP8(mx.full(shape, value, dtype=mx.float32))


def _tiny_layer(layer_index: int) -> MimoMtpLayerTensors:
    del layer_index
    tensors = {
        "eh_proj.weight": mx.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=mx.bfloat16),
        "enorm.weight": mx.ones((2,), dtype=mx.bfloat16),
        "hnorm.weight": mx.ones((2,), dtype=mx.bfloat16),
        "final_layernorm.weight": mx.ones((2,), dtype=mx.bfloat16),
        "input_layernorm.weight": mx.ones((2,), dtype=mx.bfloat16),
        "pre_mlp_layernorm.weight": mx.ones((2,), dtype=mx.bfloat16),
        "self_attn.attention_sink_bias": mx.zeros((1,), dtype=mx.bfloat16),
        "self_attn.o_proj.weight": mx.zeros((2, 2), dtype=mx.bfloat16),
        "self_attn.qkv_proj.weight": _fp8_weight((6, 2)),
        "self_attn.qkv_proj.weight_scale_inv": mx.ones((1, 1), dtype=mx.float32),
        "mlp.down_proj.weight": _fp8_weight((2, 4)),
        "mlp.down_proj.weight_scale_inv": mx.ones((1, 1), dtype=mx.float32),
        "mlp.gate_proj.weight": _fp8_weight((4, 2)),
        "mlp.gate_proj.weight_scale_inv": mx.ones((1, 1), dtype=mx.float32),
        "mlp.up_proj.weight": _fp8_weight((4, 2)),
        "mlp.up_proj.weight_scale_inv": mx.ones((1, 1), dtype=mx.float32),
    }
    return MimoMtpLayerTensors(layer_index=0, tensors=tensors)


def test_fp8_block_linear_expands_official_scale_inv_blocks() -> None:
    layer = Fp8BlockLinear(
        weight=MX_TO_FP8(mx.ones((2, 3), dtype=mx.float32)),
        weight_scale_inv=mx.array([[2.0]], dtype=mx.float32),
    )

    output = layer(mx.ones((1, 1, 3), dtype=mx.float32))

    assert output.shape == (1, 1, 2)
    assert mx.allclose(output, mx.array([[[6.0, 6.0]]]), atol=0.01)


def test_build_mimo_mtp_stack_proposes_tokens_with_tiny_sidecar() -> None:
    sidecar = MimoMtpSidecarTensors(
        path=Path("model_mtp.safetensors"),
        layers=(_tiny_layer(0), _tiny_layer(1), _tiny_layer(2)),
    )
    base_model = _TinyBaseModel()
    stack = build_mimo_mtp_stack(sidecar, base_model)

    tokens, logits = stack.propose(
        previous_hidden_state=mx.ones((1, 1, 2), dtype=mx.float32),
        latest_token_ids=mx.array([[1]], dtype=mx.int32),
        max_draft_tokens=2,
        sampler=lambda row: mx.argmax(row, axis=-1),
        cache=stack.make_cache(),
    )

    assert tokens.shape == (1, 2)
    assert tuple(int(tokens[0, index].item()) for index in range(2)) == (4, 5)
    assert len(logits) == 2
    assert base_model.lm_head_calls == 2


def test_mimo_mtp_cache_reset_clears_layer_offsets() -> None:
    cache = MimoMtpCache(num_layers=2)
    keys = mx.zeros((1, 1, 1, 2), dtype=mx.float32)
    values = mx.zeros((1, 1, 1, 2), dtype=mx.float32)
    cache.layers[0].update_and_fetch(keys, values)

    cache.reset()

    assert cache.layers[0].offset == 0
    assert cache.layers[0].keys is None
    assert cache.layers[0].values is None
