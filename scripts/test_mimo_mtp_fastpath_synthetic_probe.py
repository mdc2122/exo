from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from scripts import mimo_mtp_fastpath_synthetic_probe as probe


def test_synthetic_probe_reports_timing_acceptance_and_no_full_model_load() -> None:
    report = probe.run_synthetic_probe(max_tokens=4, requested_depth=3)

    assert report["kind"] == "synthetic_mtp_fastpath_probe"
    assert report["full_model_loaded"] is False
    assert report["generated_tokens"] == 4
    assert report["requested_depth"] == 3
    assert report["attempted_depth_counts"] == {"1": 1, "3": 3}
    assert report["accepted_depth_counts"] == {"0": 1, "1": 3}
    assert report["acceptance_rate"] == 3 / 10
    assert report["timing_breakdown_seconds"] == {
        "proposal": 0.004,
        "verification": 0.008,
        "acceptance": 0.001,
        "fallback": 0.002,
    }
    assert report["notes"] == [
        "synthetic/tiny providers only; no MiMo model materialized",
        "diagnostics are for code-path readiness, not speedup claims",
    ]


def test_synthetic_probe_cli_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "synthetic_probe.json"

    exit_code = probe.main(
        ["--max-tokens", "2", "--requested-depth", "2", "--json-out", str(output)]
    )

    assert exit_code == 0
    data = cast(dict[str, object], json.loads(output.read_text()))
    assert data["kind"] == "synthetic_mtp_fastpath_probe"
    assert data["generated_tokens"] == 2
    assert data["requested_depth"] == 2
    assert data["full_model_loaded"] is False
