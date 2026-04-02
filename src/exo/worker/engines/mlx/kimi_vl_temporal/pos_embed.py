# pyright: basic, reportMissingImports=false, reportMissingModuleSource=false

from typing import Any, Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm.models.kernels import bicubic_interpolate


def _as_thw_shapes(grid_thws) -> List[Tuple[int, int, int]]:
    raw_shapes = grid_thws.tolist() if hasattr(grid_thws, "tolist") else grid_thws
    parsed: List[Tuple[int, int, int]] = []
    for shape in raw_shapes:
        if len(shape) == 3:
            parsed.append((int(shape[0]), int(shape[1]), int(shape[2])))
        else:
            parsed.append((1, int(shape[0]), int(shape[1])))
    return parsed


def get_1d_sincos_pos_embed(embed_dim: int, length: int) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")

    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = np.arange(length, dtype=np.float32)
    out = np.einsum("m,d->md", pos, omega)
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    return emb


class Learnable2DInterpPosEmb(nn.Module):
    def __init__(
        self, height: int, width: int, dim: int, interpolation_mode: str = "bicubic"
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.interpolation_mode = interpolation_mode
        self.weight = mx.ones((height, width, dim))
        self._interp_cache: Dict[Tuple[int, int], mx.array] = {}

    def _get_pos_emb(self, shape: Tuple[int, int]) -> mx.array:
        cached = self._interp_cache.get(shape)
        if cached is not None:
            return cached

        if shape == self.weight.shape[:-1]:
            cached = self.weight.flatten(end_axis=1)
        else:
            cached = (
                bicubic_interpolate(
                    mx.expand_dims(self.weight.transpose(2, 0, 1), axis=0),
                    size=shape,
                )
                .squeeze(0)
                .transpose(1, 2, 0)
                .flatten(end_axis=1)
            )

        self._interp_cache[shape] = cached
        return cached

    def __call__(self, x: mx.array, grid_thws: mx.array) -> mx.array:
        pos_embs = [self._get_pos_emb((h, w)) for _, h, w in _as_thw_shapes(grid_thws)]
        return x + mx.concatenate(pos_embs, axis=0).astype(x.dtype)


class TemporalPosEmbed(nn.Module):
    def __init__(self, base_pos_emb: Any, num_frames: int = 4) -> None:
        super().__init__()
        self.height = base_pos_emb.height
        self.width = base_pos_emb.width
        self.interpolation_mode = base_pos_emb.interpolation_mode
        self.weight = base_pos_emb.weight
        self._interp_cache: Dict[Tuple[int, int], mx.array] = base_pos_emb._interp_cache
        self.num_frames = num_frames

        dim = int(self.weight.shape[-1])
        time_weight_np = get_1d_sincos_pos_embed(dim, num_frames).astype(np.float32)
        self.time_weight: np.ndarray = time_weight_np[:, None, :]

    def _get_pos_emb(self, shape: Tuple[int, int]) -> mx.array:
        cached = self._interp_cache.get(shape)
        if cached is not None:
            return cached

        if shape == self.weight.shape[:-1]:
            cached = self.weight.flatten(end_axis=1)
        else:
            cached = (
                bicubic_interpolate(
                    mx.expand_dims(self.weight.transpose(2, 0, 1), axis=0),
                    size=shape,
                )
                .squeeze(0)
                .transpose(1, 2, 0)
                .flatten(end_axis=1)
            )

        self._interp_cache[shape] = cached
        return cached

    def __call__(self, x: mx.array, grid_thws: mx.array) -> mx.array:
        pos_embs = []
        for t, h, w in _as_thw_shapes(grid_thws):
            if t > self.num_frames:
                raise ValueError(
                    f"TemporalPosEmbed only supports up to {self.num_frames} frames, got {t}"
                )

            pos_emb_2d = self._get_pos_emb((h, w))
            if t == 1:
                pos_embs.append(pos_emb_2d)
                continue

            spatial_repeated = mx.broadcast_to(
                pos_emb_2d[None],
                (t, h * w, pos_emb_2d.shape[-1]),
            )
            temporal = mx.array(self.time_weight[:t])
            pos_emb_3d = spatial_repeated + temporal
            pos_embs.append(pos_emb_3d.reshape(-1, pos_emb_2d.shape[-1]))

        return x + mx.concatenate(pos_embs, axis=0).astype(x.dtype)
