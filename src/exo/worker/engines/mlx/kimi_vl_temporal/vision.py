# pyright: basic, reportMissingImports=false, reportMissingModuleSource=false

from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn

from .merger import tpool_patch_merger
from .pos_embed import Learnable2DInterpPosEmb, TemporalPosEmbed


class VisionConfigLike(Protocol):
    model_type: str
    depth: int
    embed_dim: int
    hidden_size: int
    num_heads: int
    patch_size: int
    num_channels: int
    intermediate_size: int
    init_pos_emb_height: int
    spatial_merge_size: int
    merge_kernel_size: Sequence[int]


def _as_thw_shapes(grid_thws) -> List[Tuple[int, int, int]]:
    raw_shapes = grid_thws.tolist() if hasattr(grid_thws, "tolist") else grid_thws
    parsed: List[Tuple[int, int, int]] = []
    for shape in raw_shapes:
        if len(shape) == 3:
            parsed.append((int(shape[0]), int(shape[1]), int(shape[2])))
        else:
            parsed.append((1, int(shape[0]), int(shape[1])))
    return parsed


def make_block_attention_mask(cu_seqlens: mx.array, seq_length: int) -> mx.array:
    attention_mask = mx.zeros((seq_length, seq_length), dtype=mx.bool_)
    for i in range(1, len(cu_seqlens)):
        start = int(cu_seqlens[i - 1])
        end = int(cu_seqlens[i])
        attention_mask[start:end, start:end] = True
    return attention_mask


def check_array_shape(arr):
    shape = arr.shape

    if len(shape) != 4:
        return False

    out_channels, kernel_height, kernel_width, _ = shape
    return (
        out_channels >= kernel_height
        and out_channels >= kernel_width
        and kernel_height == kernel_width
    )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(tensor, freqs) -> mx.array:
    orig_dtype = tensor.dtype

    cos = mx.cos(freqs)
    sin = mx.sin(freqs)

    cos = mx.expand_dims(cos, axis=1)
    cos = mx.tile(cos, (1, 1, 2))
    cos = mx.expand_dims(cos, axis=0)

    sin = mx.expand_dims(sin, axis=1)
    sin = mx.tile(sin, (1, 1, 2))
    sin = mx.expand_dims(sin, axis=0)

    output = (tensor * cos) + (rotate_half(tensor) * sin)
    return output.astype(orig_dtype)


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def __call__(self, seqlen: int) -> mx.array:
        inv_freq = 1.0 / (
            self.theta ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )
        seq = mx.arange(int(seqlen), dtype=inv_freq.dtype)
        freqs = mx.outer(seq, inv_freq)
        return freqs


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        num_channels: int = 3,
        embed_dim: int = 1152,
        init_pos_emb_height: int = 64,
        num_frames: int = 4,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.init_pos_emb_height = init_pos_emb_height

        self.proj = nn.Conv2d(
            num_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        self.pos_emb = TemporalPosEmbed(
            Learnable2DInterpPosEmb(
                height=init_pos_emb_height,
                width=init_pos_emb_height,
                dim=embed_dim,
            ),
            num_frames=num_frames,
        )

    def __call__(self, hidden_states: mx.array, grid_thw: mx.array) -> mx.array:
        hidden_states = self.proj(hidden_states).swapaxes(1, 3)
        hidden_states = hidden_states.reshape(hidden_states.shape[0], -1)
        hidden_states = self.pos_emb(hidden_states, grid_thw)
        return hidden_states


def _apply_rope_input_validation(x, freqs_cis):
    assert x.ndim == freqs_cis.ndim + 1, (x.shape, freqs_cis.shape)
    assert x.shape[:-2] == freqs_cis.shape[:-1], (x.shape, freqs_cis.shape)
    assert x.shape[-1] == 2 * freqs_cis.shape[-1], (x.shape, freqs_cis.shape)
    assert freqs_cis.dtype == mx.complex64, freqs_cis.dtype


def view_as_complex(x):
    real, imag = x[..., 0], x[..., 1]
    return real + 1j * imag


def view_as_real(x):
    real = mx.real(x)
    imag = mx.imag(x)
    return mx.stack([real, imag], axis=-1)


def apply_rope(
    q: mx.array, k: mx.array, freqs_cis: mx.array
) -> tuple[mx.array, mx.array]:
    _apply_rope_input_validation(q, freqs_cis)
    _apply_rope_input_validation(k, freqs_cis)

    freqs_cis = mx.expand_dims(freqs_cis, axis=-2)
    q_ = view_as_complex(q.astype(mx.float32).reshape(*q.shape[:-1], -1, 2))
    k_ = view_as_complex(k.astype(mx.float32).reshape(*k.shape[:-1], -1, 2))
    q_out = view_as_real(q_ * freqs_cis).flatten(-2)
    k_out = view_as_real(k_ * freqs_cis).flatten(-2)
    return q_out.astype(q.dtype), k_out.astype(k.dtype)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.wqkv = nn.Linear(dim, dim * 3, bias=True)
        self.wo = nn.Linear(dim, dim, bias=True)

    def __call__(
        self,
        x: mx.array,
        cu_seqlens: Optional[mx.array] = None,
        rotary_pos_emb: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        seq_length = x.shape[0]
        qkv = (
            self.wqkv(x)
            .reshape(seq_length, 3, self.num_heads, self.head_dim)
            .transpose(1, 0, 2, 3)
        )
        q, k, v = mx.split(qkv, 3)
        q = q.squeeze(0)
        k = k.squeeze(0)
        v = v.squeeze(0)

        if rotary_pos_emb is None:
            raise ValueError("rotary_pos_emb must be provided")
        q, k = apply_rope(q, k, rotary_pos_emb)

        if attention_mask is None:
            if cu_seqlens is None:
                raise ValueError(
                    "Either attention_mask or cu_seqlens must be provided."
                )
            attention_mask = make_block_attention_mask(cu_seqlens, seq_length)

        q = q.transpose(1, 0, 2)[None, ...]
        k = k.transpose(1, 0, 2)[None, ...]
        v = v.transpose(1, 0, 2)[None, ...]

        attn_output = mx.fast.scaled_dot_product_attention(  # pyright: ignore[reportAttributeAccessIssue]
            q, k, v, scale=self.scale, mask=attention_mask
        )
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(seq_length, -1)
        return self.wo(attn_output)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.activation_fn = nn.GELU()
        self.fc0 = nn.Linear(dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.activation_fn(self.fc0(x))
        x = self.fc1(x)
        return x


class Qwen2VLVisionBlock(nn.Module):
    def __init__(self, config: VisionConfigLike) -> None:
        super().__init__()
        self.norm0 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=1e-6)

        self.attn = Attention(dim=config.embed_dim, num_heads=config.num_heads)
        self.mlp = MLP(dim=config.embed_dim, hidden_dim=config.intermediate_size)

    def __call__(
        self, hidden_states, cu_seqlens, rotary_pos_emb, attention_mask: mx.array
    ) -> mx.array:
        hidden_states = hidden_states + self.attn(
            self.norm0(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + self.mlp(self.norm1(hidden_states))
        return hidden_states


class Rope2DPosEmb(nn.Module):
    def __init__(self, dim: int, max_height: int, max_width: int, theta_base=10000):
        super().__init__()
        self.dim = dim
        assert self.dim % 4 == 0, "dim must be divisible by 4"
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

        self._freqs_cis = None
        self._shape_freqs_cache: Dict[Tuple[int, int, int], mx.array] = {}

    def extra_repr(self):
        return f"dim={self.dim}, max_height={self.max_height}, max_width={self.max_width}, theta_base={self.theta_base}"

    def _precompute_freqs_cis(self) -> mx.array:
        flat_pos = mx.arange(0, self.max_height * self.max_width, dtype=mx.float32)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = mx.arange(0, self.dim, 4)[: (self.dim // 4)].astype(mx.float32)
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = mx.outer(x_pos, freqs)
        y_freqs = mx.outer(y_pos, freqs)

        x_cis = mx.cos(x_freqs) + 1j * mx.sin(x_freqs)
        y_cis = mx.cos(y_freqs) + 1j * mx.sin(y_freqs)
        freqs_cis = mx.stack([x_cis, y_cis], axis=-1)
        return freqs_cis.reshape(self.max_height, self.max_width, -1)

    def get_freqs_cis(self, grid_thws: mx.array) -> mx.array:
        if self._freqs_cis is None:
            self._freqs_cis = self._precompute_freqs_cis()

        shapes = _as_thw_shapes(grid_thws)
        assert all(
            1 <= height <= self.max_height and 1 <= width <= self.max_width
            for _, height, width in shapes
        ), (shapes, self.max_height, self.max_width)

        freqs_cis_list = []
        for shape in shapes:
            cached = self._shape_freqs_cache.get(shape)
            if cached is None:
                time, height, width = shape
                cached = self._freqs_cis[:height, :width].reshape(-1, self.dim // 2)
                if time > 1:
                    cached = mx.tile(cached, (time, 1))
                self._shape_freqs_cache[shape] = cached
            freqs_cis_list.append(cached)

        return mx.concatenate(freqs_cis_list, axis=0)


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfigLike) -> None:
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        if self.model_type not in ["qwen2_vl", "moonvit"]:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        self.spatial_merge_size = config.spatial_merge_size
        self.merge_kernel_size = config.merge_kernel_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            embed_dim=config.embed_dim,
            init_pos_emb_height=config.init_pos_emb_height,
        )

        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = [Qwen2VLVisionBlock(config) for _ in range(config.depth)]
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.rope_pos_emb = Rope2DPosEmb(head_dim, 512, 512)

    def __call__(
        self,
        hidden_states: mx.array,
        grid_thw: mx.array,
        output_hidden_states: Optional[bool] = None,
    ) -> List[mx.array]:
        hidden_states = self.patch_embed(hidden_states, grid_thw)
        rotary_pos_emb = self.rope_pos_emb.get_freqs_cis(grid_thw)

        seq_lengths = mx.array(
            [time * height * width for time, height, width in _as_thw_shapes(grid_thw)],
            dtype=mx.int32,
        )
        lengths = mx.concatenate([mx.zeros((1,), dtype=mx.int32), seq_lengths])
        cu_seqlens = mx.cumsum(lengths, axis=0)
        attention_mask = make_block_attention_mask(cu_seqlens, hidden_states.shape[0])

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                attention_mask=attention_mask,
            )

        hidden_states = self.final_layernorm(hidden_states)
        merged_hidden_states = tpool_patch_merger(
            hidden_states, grid_thw, merge_kernel_size=self.merge_kernel_size
        )
        return merged_hidden_states

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                continue
            elif "patch_embed.proj.weight" in k:
                if check_array_shape(v):
                    sanitized_weights[k] = v
                else:
                    sanitized_weights[k] = v.transpose(0, 2, 3, 1)
            elif "vision_tower.blocks" in k:
                if "attn" not in k and ("wqkv" in k or "wo" in k):
                    new_key = k.replace("wqkv", "attn.wqkv").replace("wo", "attn.wo")
                    sanitized_weights[new_key] = v
                else:
                    sanitized_weights[k] = v
            else:
                sanitized_weights[k] = v

        return sanitized_weights
