from __future__ import annotations

import json
from pathlib import Path

from exo.worker.engines.mlx.mimo_mtp_fast.benchmark import (
    MimoMtpBenchmarkMode,
    build_contract_probe_row,
    build_metric_row,
    parse_benchmark_modes,
    render_json_line,
)


def test_contract_probe_row_reports_missing_sidecar_without_loading_model(
    tmp_path: Path,
) -> None:
    missing_sidecar = tmp_path / "model_mtp.safetensors"

    row = build_contract_probe_row(
        sidecar_path=missing_sidecar,
        model_path=None,
    )

    assert row["kind"] == "contract_probe"
    assert row["ready"] is False
    assert row["sidecar_status"] == "missing"
    assert row["sidecar_path"] == str(missing_sidecar)
    assert row["model_path"] is None
    assert row["next_step"] == "provide a valid official-layout MiMo model_mtp.safetensors sidecar"


def test_parse_benchmark_modes_preserves_requested_order() -> None:
    modes = parse_benchmark_modes("ar,d1,d2,d3,auto")

    assert modes == (
        MimoMtpBenchmarkMode.AR,
        MimoMtpBenchmarkMode.D1,
        MimoMtpBenchmarkMode.D2,
        MimoMtpBenchmarkMode.D3,
        MimoMtpBenchmarkMode.AUTO,
    )


def test_parse_benchmark_modes_rejects_unknown_mode() -> None:
    try:
        parse_benchmark_modes("ar,d4")
    except ValueError as exc:
        assert "Unsupported MiMo MTP benchmark mode" in str(exc)
        assert "d4" in str(exc)
    else:
        raise AssertionError("parse_benchmark_modes should reject d4")


def test_metric_row_computes_decode_tokens_per_second_and_conclusion() -> None:
    row = build_metric_row(
        mode=MimoMtpBenchmarkMode.D3,
        generated_tokens=96,
        decode_seconds=3.0,
        attempted_depth_counts={3: 40},
        accepted_depth_counts={0: 4, 3: 36},
        ar_baseline_tok_s=22.0,
    )

    assert row["kind"] == "benchmark_metric"
    assert row["mode"] == "d3"
    assert row["generated_tokens"] == 96
    assert row["decode_seconds"] == 3.0
    assert row["decode_tok_s"] == 32.0
    assert row["attempted_depth_counts"] == {"3": 40}
    assert row["accepted_depth_counts"] == {"0": 4, "3": 36}
    assert row["next_step"] == "MTP mode d3 beats AR baseline; proceed to guarded exo integration"


def test_render_json_line_is_single_line_stable_json() -> None:
    line = render_json_line({"b": 2, "a": 1})

    assert line == '{"a":1,"b":2}'
    assert "\n" not in line
    assert json.loads(line) == {"a": 1, "b": 2}
