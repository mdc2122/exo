# pyright: basic, reportMissingImports=false, reportMissingModuleSource=false

from typing import List, Sequence, Tuple

import mlx.core as mx


def _as_thw_shapes(grid_thws) -> List[Tuple[int, int, int]]:
    raw_shapes = grid_thws.tolist() if hasattr(grid_thws, "tolist") else grid_thws
    parsed: List[Tuple[int, int, int]] = []
    for shape in raw_shapes:
        if len(shape) == 3:
            parsed.append((int(shape[0]), int(shape[1]), int(shape[2])))
        else:
            parsed.append((1, int(shape[0]), int(shape[1])))
    return parsed


def tpool_patch_merger(
    x: mx.array,
    grid_thws: mx.array,
    merge_kernel_size: Sequence[int] = (2, 2),
) -> List[mx.array]:
    kernel_height, kernel_width = merge_kernel_size
    shapes = _as_thw_shapes(grid_thws)
    lengths = [t * height * width for t, height, width in shapes]

    split_points = []
    running = 0
    for length in lengths[:-1]:
        running += length
        split_points.append(running)

    sequences = mx.split(x, split_points, axis=0) if split_points else [x]
    outputs = []
    for seq, (time, height, width) in zip(sequences, shapes, strict=True):
        new_height = height // kernel_height
        new_width = width // kernel_width
        reshaped_seq = seq.reshape(
            time,
            new_height,
            kernel_height,
            new_width,
            kernel_width,
            -1,
        )
        reshaped_seq = mx.mean(reshaped_seq, axis=0)
        reshaped_seq = mx.transpose(reshaped_seq, (0, 2, 1, 3, 4))
        padded_seq = reshaped_seq.reshape(
            new_height * new_width,
            kernel_height * kernel_width,
            -1,
        )
        outputs.append(padded_seq)

    return outputs
