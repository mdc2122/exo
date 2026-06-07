# Sub-AC 5 Evidence: User-Facing MiMo MTP Budget Calculator Entry Point

Timestamp: 2026-06-06 21:34 local

## Scope

Sub-AC 5 requested a user-facing script or documented calculator entry point that consumes AR/MTP JSONL files and prints or writes computed budget metrics, classifications, and bottlenecks in a stable format, with runnable tests using fixture JSONL files including AR-only and AR+MTP cases.

This slice adds a CLI calculator around the existing cluster benchmark ingestion and speedup budget helpers. It does not collect live benchmark rows and does not enable MTP by default.

## Files Changed

Created for this Sub-AC:

- `scripts/mimo_mtp_budget_calculator.py`
  - User-facing CLI entry point:
    - `uv run python3 scripts/mimo_mtp_budget_calculator.py <rows.jsonl>` prints a stable JSON report.
    - `uv run python3 scripts/mimo_mtp_budget_calculator.py <rows.jsonl> --output <report.json>` writes the report.
  - Exposes `build_budget_report(...)` for scripted consumption.
  - Consumes one or more JSONL files from `scripts/bench_mimo_mtp_cluster.py`.
  - Reuses `scripts/mimo_mtp_benchmark_ingest.py` with low-acceptance-rate classification enabled.
  - Reuses `scripts/bench_mimo_mtp_cluster.py::calculate_mtp_speedup_budget(...)` and throughput threshold constants/classifier.
  - Emits stable JSON with:
    - `kind`: `mimo_mtp_budget_report`
    - `schema_version`: `1`
    - `source_files`
    - `ingestion` counts
    - `budget` metrics: AR median tok/s, AR ms/token, MTP median tok/s, 30/40 tok/s gaps, MTP-vs-AR speedup ratio, status, next step
    - `classifications`: same-cluster evidence status, MTP throughput threshold, Slice 5 gate, and explicit claim-allowed booleans
    - `bottlenecks`
    - `next_step`

- `scripts/test_mimo_mtp_budget_calculator.py`
  - Fixture-backed tests for:
    - AR-only JSONL: computes AR budget, keeps MTP/speedup/target claims blocked, emits `missing_live_mtp_rows`.
    - AR+MTP JSONL: computes MTP budget, threshold classification, guarded review gate, and `acceptance_rate_low` bottleneck.
    - CLI stdout JSON output.
    - CLI `--output` file writing.
    - CLI `--help` user-facing documentation.

- `scripts/fixtures/mimo_mtp_budget_ar_only.jsonl`
  - Two same-cluster AR rows.

- `scripts/fixtures/mimo_mtp_budget_ar_plus_mtp.jsonl`
  - Two same-cluster AR rows and two guarded live-MTP rows with acceptance/timing telemetry.

## TDD / RED Evidence

Tests were written before the calculator entry point existed.

Command:

```bash
cd .. && uv run pytest scripts/test_mimo_mtp_budget_calculator.py -q
```

Observed RED result:

```text
ERROR scripts/test_mimo_mtp_budget_calculator.py
ImportError: cannot import name 'mimo_mtp_budget_calculator' from 'scripts' (unknown location)
1 error in 0.05s
```

This failed for the expected reason: the user-facing calculator module/CLI did not exist yet.

## GREEN / Verification Evidence

Initial implementation made the pure report-builder tests pass, then direct CLI execution exposed an import-path issue. That was fixed while preserving CLI execution by file path. A later typecheck failure was traced to fallback imports rebinding imported `Final` constants; the fix was to import helper modules as aliases and access constants/functions through those module aliases.

Final focused verification command:

```bash
cd .. && uv run ruff format scripts/mimo_mtp_budget_calculator.py scripts/test_mimo_mtp_budget_calculator.py && \
uv run pytest scripts/test_mimo_mtp_budget_calculator.py scripts/test_mimo_mtp_benchmark_ingest.py scripts/test_bench_mimo_mtp_cluster.py -q && \
uv run ruff check scripts/mimo_mtp_budget_calculator.py scripts/test_mimo_mtp_budget_calculator.py scripts/mimo_mtp_benchmark_ingest.py scripts/test_mimo_mtp_benchmark_ingest.py scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py && \
uv run basedpyright scripts/mimo_mtp_budget_calculator.py scripts/test_mimo_mtp_budget_calculator.py scripts/mimo_mtp_benchmark_ingest.py scripts/test_mimo_mtp_benchmark_ingest.py scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py && \
uv run ruff format --check scripts/mimo_mtp_budget_calculator.py scripts/test_mimo_mtp_budget_calculator.py scripts/mimo_mtp_benchmark_ingest.py scripts/test_mimo_mtp_benchmark_ingest.py scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py && \
git diff --check -- scripts/mimo_mtp_budget_calculator.py scripts/test_mimo_mtp_budget_calculator.py scripts/fixtures/mimo_mtp_budget_ar_only.jsonl scripts/fixtures/mimo_mtp_budget_ar_plus_mtp.jsonl
```

Observed result:

```text
1 file reformatted, 1 file left unchanged
37 passed in 0.41s
All checks passed!
0 errors, 0 warnings, 0 notes
6 files already formatted
# git diff --check produced no output
```

## CLI Sample Evidence

AR-only fixture command:

```bash
cd .. && uv run python3 scripts/mimo_mtp_budget_calculator.py scripts/fixtures/mimo_mtp_budget_ar_only.jsonl
```

Observed key fields:

```json
{
  "kind": "mimo_mtp_budget_report",
  "budget": {
    "ar_median_tok_s": 21.0,
    "ar_ms_per_token": 47.619048,
    "mtp_median_tok_s": null,
    "mtp_vs_ar_speedup_ratio": null,
    "status": "blocked_missing_mtp_rows"
  },
  "classifications": {
    "same_cluster_evidence": "blocked_missing_mtp_rows",
    "mtp_throughput_threshold": null,
    "slice_5_gate": "blocked_missing_mtp_rows",
    "speedup_claim_allowed": false,
    "target_30_tok_s_claim_allowed": false,
    "target_40_tok_s_claim_allowed": false
  },
  "bottlenecks": ["missing_live_mtp_rows"]
}
```

AR+MTP fixture command:

```bash
cd .. && uv run python3 scripts/mimo_mtp_budget_calculator.py scripts/fixtures/mimo_mtp_budget_ar_plus_mtp.jsonl
```

Observed key fields from synthetic fixture data:

```json
{
  "kind": "mimo_mtp_budget_report",
  "budget": {
    "ar_median_tok_s": 21.0,
    "ar_ms_per_token": 47.619048,
    "mtp_median_tok_s": 35.0,
    "mtp_target_gap_to_30_tok_s": 0.0,
    "mtp_target_gap_to_40_tok_s": 5.0,
    "mtp_vs_ar_speedup_ratio": 1.666667,
    "status": "same_cluster_budget_ready"
  },
  "classifications": {
    "same_cluster_evidence": "same_cluster_budget_ready",
    "mtp_throughput_threshold": "at_least_30_tok_s",
    "slice_5_gate": "eligible_for_guarded_review",
    "speedup_claim_allowed": true,
    "target_30_tok_s_claim_allowed": true,
    "target_40_tok_s_claim_allowed": false
  },
  "bottlenecks": ["acceptance_rate_low"]
}
```

Output-file smoke command:

```bash
cd .. && tmp_report=$(mktemp) && \
uv run python3 scripts/mimo_mtp_budget_calculator.py scripts/fixtures/mimo_mtp_budget_ar_only.jsonl --output "$tmp_report" && \
python3 - <<'PY' "$tmp_report"
import json, sys
path = sys.argv[1]
with open(path, encoding='utf-8') as f:
    report = json.load(f)
print(report['kind'])
print(report['classifications']['slice_5_gate'])
PY
rm -f "$tmp_report"
```

Observed result:

```text
mimo_mtp_budget_report
blocked_missing_mtp_rows
```

## Benchmark Rows / Blocker

No live cluster benchmark rows were collected for this Sub-AC. The AR+MTP fixture rows are synthetic test fixtures only and are not evidence for a real 30+ tok/s result, 40+ tok/s result, MTP speedup, or Slice 5 production/default enablement.

Live evidence remains blocked until an exo cluster API and guarded MTP path produce same-cluster `/bench/chat/completions` AR-vs-MTP rows.

Runnable calculator commands for future live rows:

```bash
uv run python3 scripts/mimo_mtp_budget_calculator.py /path/to/ar.jsonl /path/to/mtp.jsonl
uv run python3 scripts/mimo_mtp_budget_calculator.py /path/to/ar-and-mtp.jsonl --output /path/to/mimo-mtp-budget-report.json
```

Runnable same-cluster collection commands remain provided by the cluster harness, for example:

```bash
uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 16 --repeats 1 --mode-label ar --list-models
uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 64 --repeats 1 --mode-label ar --list-models
```

When the guarded MTP cluster request path is available, collect MTP rows with explicit guarded payload JSON, then feed AR and MTP JSONL files to `scripts/mimo_mtp_budget_calculator.py`.

## Remaining Risk

- The calculator is only as accurate as the ingested benchmark rows; malformed or missing live telemetry keeps claims blocked/null rather than guessed.
- Tests use fixture JSONL rows, not live cluster rows.
- `eligible_for_guarded_review` in fixture output is a stable-format classification for synthetic test data, not a live production/default MTP approval.
- Slice 5 remains blocked for real rollout until same-cluster AR-vs-MTP benchmark rows prove a real MTP speed win without fallback/correctness concerns.
