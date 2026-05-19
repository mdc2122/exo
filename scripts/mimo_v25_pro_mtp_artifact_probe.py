#!/usr/bin/env python3
"""Inspect MiMo V2.5 Pro MTP safetensors without loading the full model."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, TypedDict, cast

from safetensors import safe_open

DEFAULT_MTP_SHARD: Final[Path] = Path(
    "/Volumes/GLM5-NVMe/exo/mimo-v25-pro/hf/XiaomiMiMo--MiMo-V2.5-Pro/model_mtp.safetensors"
)
EXPECTED_MTP_LAYERS: Final[tuple[int, ...]] = (0, 1, 2)
MTP_LAYER_RE: Final[re.Pattern[str]] = re.compile(r"^model\.mtp\.layers\.(\d+)\.")


class TensorReport(TypedDict):
    dtype: str
    shape: list[int]


class MtpReport(TypedDict):
    source_file: str
    tensor_count: int
    layers: list[int]
    missing_expected_layers: list[int]
    complete_expected_layers: bool
    tensors: dict[str, TensorReport]


@dataclass(frozen=True)
class CliArgs:
    path: Path
    json_out: Path | None


class _SafeSlice(Protocol):
    def get_dtype(self) -> str: ...

    def get_shape(self) -> Sequence[int]: ...


class _SafeOpenHandle(Protocol):
    def keys(self) -> Sequence[str]: ...

    def get_slice(self, name: str) -> _SafeSlice: ...


def _mtp_layer(tensor_name: str) -> int | None:
    match = MTP_LAYER_RE.match(tensor_name)
    return int(match.group(1)) if match else None


def probe_mtp_shard(path: Path) -> MtpReport:
    if not path.is_file():
        raise FileNotFoundError(f"MTP shard not found: {path}")

    tensors: dict[str, TensorReport] = {}
    layers: set[int] = set()
    with safe_open(str(path), framework="pt") as raw_handle:
        handle = cast(_SafeOpenHandle, cast(object, raw_handle))
        for name in sorted(handle.keys()):
            layer = _mtp_layer(name)
            if layer is None:
                continue
            tensor_slice = handle.get_slice(name)
            layers.add(layer)
            tensors[name] = {
                "dtype": tensor_slice.get_dtype(),
                "shape": [int(dim) for dim in tensor_slice.get_shape()],
            }

    missing_expected_layers = [
        layer for layer in EXPECTED_MTP_LAYERS if layer not in layers
    ]
    return {
        "source_file": str(path),
        "tensor_count": len(tensors),
        "layers": sorted(layers),
        "missing_expected_layers": missing_expected_layers,
        "complete_expected_layers": not missing_expected_layers,
        "tensors": tensors,
    }


def _parse_args(argv: Sequence[str]) -> CliArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_MTP_SHARD)
    parser.add_argument("--json-out", type=Path, default=None)
    namespace = parser.parse_args(argv)
    path = cast(object, namespace.path)
    json_out = cast(object, namespace.json_out)

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    if json_out is not None and not isinstance(json_out, Path):
        raise TypeError("--json-out must be a pathlib.Path when provided")

    return CliArgs(path=path, json_out=json_out)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    report = probe_mtp_shard(args.path)
    text = json.dumps(cast(object, report), indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
