# Sub-AC 3 Evidence: MTP Throughput Threshold Classification

Timestamp: 2026-06-06 21:27 local

## Scope

Sub-AC 3 requested throughput threshold classification for computed MTP metrics, distinguishing:

- below 30 tok/s
- at least 30 tok/s
- at least 40 tok/s

The implementation is benchmark-telemetry only. It does not enable MTP by default and does not alter default AR generation behavior.

## Files Changed

- `scripts/bench_mimo_mtp_cluster.py`
  - Adds `MtpThroughputThreshold` with explicit labels:
    - `below_30_tok_s`
    - `at_least_30_tok_s`
    - `at_least_40_tok_s`
  - Adds `MTP_TARGET_TOKENS_PER_SECOND = 30.0` and `MTP_PREFERRED_TOKENS_PER_SECOND = 40.0` constants.
  - Adds `classify_mtp_throughput_threshold(...)` for computed MTP `generation_tps` values.
  - Adds per-row MTP budget telemetry via `build_mtp_speedup_budget(...)`:
    - `generation_tps`
    - `threshold_30_tok_s_met`
    - `threshold_40_tok_s_met`
    - `tok_s_gap_to_30`
    - `tok_s_gap_to_40`
  - Wires the classification into `build_cluster_metric_row(...)` only for non-AR/MTP-labeled rows with numeric `generation_tps`.
  - Leaves AR rows without `mtp_throughput_threshold` or `mtp_speedup_budget` fields.
  - Adds/keeps aggregate same-cluster budget support through `calculate_mtp_speedup_budget(...)` so median AR-vs-MTP rows remain evidence-gated and missing telemetry stays nullable.

- `scripts/test_bench_mimo_mtp_cluster.py`
  - Adds regression tests for:
    - below-30 MTP row classification
    - exactly/at-least-30 MTP row classification
    - exactly/at-least-40 MTP row classification
    - AR rows not emitting MTP threshold telemetry
  - Existing aggregate speedup budget tests verify no speedup/target metrics are fabricated when telemetry is absent.

Note: these files also include neighboring uncommitted benchmark-harness work from earlier ACs in this worktree. This Sub-AC claims only the throughput threshold classification and directly related budget telemetry described above.

## TDD / RED Evidence

The threshold tests were added before the implementation and run against the pre-threshold row builder:

```bash
cd .. && uv run pytest \
  scripts/test_bench_mimo_mtp_cluster.py::test_mtp_metric_row_classifies_generation_tps_below_30_tok_s \
  scripts/test_bench_mimo_mtp_cluster.py::test_mtp_metric_row_classifies_generation_tps_at_least_30_tok_s \
  scripts/test_bench_mimo_mtp_cluster.py::test_mtp_metric_row_classifies_generation_tps_at_least_40_tok_s \
  scripts/test_bench_mimo_mtp_cluster.py::test_ar_metric_row_does_not_emit_mtp_throughput_classification \
  -q
```

Observed RED result:

```text
FFF.                                                                     [100%]
3 failed, 1 passed in 0.09s
```

The three MTP tests failed with:

```text
KeyError: 'mtp_throughput_threshold'
```

The AR safety test passed during RED, confirming AR rows did not already emit MTP threshold telemetry.

## GREEN / Verification Evidence

Focused threshold and aggregate budget tests:

```bash
cd .. && uv run pytest \
  scripts/test_bench_mimo_mtp_cluster.py::test_mtp_metric_row_classifies_generation_tps_below_30_tok_s \
  scripts/test_bench_mimo_mtp_cluster.py::test_mtp_metric_row_classifies_generation_tps_at_least_30_tok_s \
  scripts/test_bench_mimo_mtp_cluster.py::test_mtp_metric_row_classifies_generation_tps_at_least_40_tok_s \
  scripts/test_bench_mimo_mtp_cluster.py::test_ar_metric_row_does_not_emit_mtp_throughput_classification \
  scripts/test_bench_mimo_mtp_cluster.py::test_calculate_mtp_speedup_budget_for_ar_only_rows \
  scripts/test_bench_mimo_mtp_cluster.py::test_calculate_mtp_speedup_budget_for_ar_and_mtp_rows \
  scripts/test_bench_mimo_mtp_cluster.py::test_calculate_mtp_speedup_budget_marks_incomplete_telemetry_without_fabricating_metrics \
  -q
```

Observed result:

```text
7 passed in 0.05s
```

Full cluster benchmark harness tests:

```bash
cd .. && uv run pytest scripts/test_bench_mimo_mtp_cluster.py -q
```

Observed result:

```text
25 passed in 0.13s
```

Focused lint:

```bash
cd .. && uv run ruff check scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py
```

Observed result:

```text
All checks passed!
```

Focused typecheck:

```bash
cd .. && uv run basedpyright scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py
```

Observed result:

```text
0 errors, 0 warnings, 0 notes
```

Focused format check:

```bash
cd .. && uv run ruff format --check scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py
```

Observed result:

```text
2 files already formatted
```

Whitespace diff check:

```bash
cd .. && git diff --check -- scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py
```

Observed result: exit code 0, no whitespace errors printed.

Fresh final verification command also located the implemented symbols/tests:

```bash
cd .. && uv run pytest scripts/test_bench_mimo_mtp_cluster.py -q && \
uv run ruff check scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py && \
uv run basedpyright scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py && \
uv run ruff format --check scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py && \
git diff --check -- scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py && \
rg -n "def classify_mtp_throughput_threshold|def build_mtp_speedup_budget|def calculate_mtp_speedup_budget|def _median_float|def _cluster_generation_tps_values|mtp_throughput_threshold" scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py
```

Observed result included:

```text
25 passed in 0.13s
All checks passed!
0 errors, 0 warnings, 0 notes
2 files already formatted
scripts/test_bench_mimo_mtp_cluster.py:377:    assert row["mtp_throughput_threshold"] == "below_30_tok_s"
scripts/test_bench_mimo_mtp_cluster.py:405:    assert row["mtp_throughput_threshold"] == "at_least_30_tok_s"
scripts/test_bench_mimo_mtp_cluster.py:433:    assert row["mtp_throughput_threshold"] == "at_least_40_tok_s"
scripts/test_bench_mimo_mtp_cluster.py:460:    assert "mtp_throughput_threshold" not in row
scripts/bench_mimo_mtp_cluster.py:251:def classify_mtp_throughput_threshold(
scripts/bench_mimo_mtp_cluster.py:262:def build_mtp_speedup_budget(generation_tps: object) -> JsonObject | None:
scripts/bench_mimo_mtp_cluster.py:363:            row["mtp_throughput_threshold"] = classify_mtp_throughput_threshold(
scripts/bench_mimo_mtp_cluster.py:461:def _median_float(values: Iterable[float]) -> float | None:
scripts/bench_mimo_mtp_cluster.py:493:def _cluster_generation_tps_values(
scripts/bench_mimo_mtp_cluster.py:511:def calculate_mtp_speedup_budget(rows: Iterable[Mapping[str, object]]) -> JsonObject:
```

## Benchmark Rows / Blocker

No live cluster benchmark rows were collected for this Sub-AC. This Sub-AC is limited to computed metric classification and unit-level harness verification.

No claim is made that MTP reaches 30 tok/s, reaches 40 tok/s, or beats AR. Those claims remain blocked until same-cluster `/bench/chat/completions` AR-vs-MTP rows exist.

Runnable same-cluster AR baseline commands remain preserved by the harness:

```bash
uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 16 --repeats 1 --mode-label ar --list-models
uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 64 --repeats 1 --mode-label ar --list-models
```

When a guarded MTP request contract is available and safe, collect same-cluster MTP rows with the appropriate explicit payload extra JSON and compare only against same-cluster AR rows.

## Remaining Risk

- The classification is based on normalized `generation_stats.generation_tps`; if live cluster responses omit or misreport that field, MTP threshold fields are not emitted and aggregate budgets remain blocked/nullable.
- Unit tests use injected fake responses and do not prove live cluster performance.
- Slice 5 remains ineligible until same-cluster AR-vs-MTP evidence rows show a real MTP speed win without fallback/correctness concerns.
- Other uncommitted changes from earlier ACs are present in the worktree; this report only covers the Sub-AC 3 threshold classification slice.
