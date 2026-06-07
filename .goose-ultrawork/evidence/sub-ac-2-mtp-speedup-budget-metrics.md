# Sub-AC 2 Evidence: MTP Speedup Budget Metrics

Timestamp: 2026-06-06 21:27 local

## Scope

Sub-AC 2 requested a budget metric calculation module/function that computes, from parsed benchmark rows:

- AR ms/token
- MTP tok/s
- target gaps for 30 tok/s and 40 tok/s
- MTP-versus-AR speedup ratio

The implementation must handle AR-only, AR+MTP, and incomplete telemetry inputs without fabricating performance claims.

## Files Changed

- `scripts/bench_mimo_mtp_cluster.py`
  - Added `calculate_mtp_speedup_budget(rows)` as a pure aggregate helper for parsed `/bench/chat/completions` benchmark rows.
  - Computes median AR tok/s, AR ms/token, median guarded-MTP tok/s, MTP gaps to 30 and 40 tok/s, and MTP-vs-AR speedup ratio.
  - Counts only exo cluster metric rows from `cluster_path == "exo_api_bench_chat_completions"`.
  - Counts MTP rows only when `accepted_execution_path == "mimo_mtp_fastpath"`, keeping fallback/unknown rows out of speedup evidence.
  - Leaves metrics nullable and emits blocked statuses when telemetry is incomplete or one side of the same-cluster comparison is missing.

- `scripts/test_bench_mimo_mtp_cluster.py`
  - Added runnable tests for:
    - AR-only rows: AR ms/token is computed, MTP metrics/speedup remain nullable, status blocks speedup claims.
    - AR+MTP rows: MTP median tok/s, 30/40 tok/s gaps, and speedup ratio are computed.
    - Incomplete telemetry: no metrics are fabricated, row counts are zero, and status remains blocked.

## TDD / RED Evidence

A targeted RED run was executed before the final aggregate helper existed. The new tests failed as expected with missing-symbol errors:

```text
uv run pytest scripts/test_bench_mimo_mtp_cluster.py -q
...
FAILED scripts/test_bench_mimo_mtp_cluster.py::test_calculate_mtp_speedup_budget_for_ar_only_rows
FAILED scripts/test_bench_mimo_mtp_cluster.py::test_calculate_mtp_speedup_budget_for_ar_and_mtp_rows
FAILED scripts/test_bench_mimo_mtp_cluster.py::test_calculate_mtp_speedup_budget_marks_incomplete_telemetry_without_fabricating_metrics
AttributeError: module 'scripts.bench_mimo_mtp_cluster' has no attribute 'calculate_mtp_speedup_budget'
```

During cleanup, duplicate helper definitions were detected by lint (`F811`) and consolidated to one canonical `calculate_mtp_speedup_budget` definition.

## GREEN / Verification Evidence

Fresh focused symbol check and verification:

```text
rg -n "def calculate_mtp_speedup_budget|def _cluster_generation_tps_values|def _median_float|def test_calculate_mtp_speedup_budget" scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py
scripts/test_bench_mimo_mtp_cluster.py:725:def test_calculate_mtp_speedup_budget_for_ar_only_rows() -> None:
scripts/test_bench_mimo_mtp_cluster.py:760:def test_calculate_mtp_speedup_budget_for_ar_and_mtp_rows() -> None:
scripts/test_bench_mimo_mtp_cluster.py:811:def test_calculate_mtp_speedup_budget_marks_incomplete_telemetry_without_fabricating_metrics() -> (
scripts/bench_mimo_mtp_cluster.py:461:def _median_float(values: Iterable[float]) -> float | None:
scripts/bench_mimo_mtp_cluster.py:493:def _cluster_generation_tps_values(
scripts/bench_mimo_mtp_cluster.py:511:def calculate_mtp_speedup_budget(rows: Iterable[Mapping[str, object]]) -> JsonObject:
```

Focused tests:

```text
uv run pytest scripts/test_bench_mimo_mtp_cluster.py -q
.........................                                                [100%]
25 passed in 0.13s
```

Focused lint:

```text
uv run ruff check scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py
All checks passed!
```

Focused typecheck:

```text
uv run basedpyright scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py
0 errors, 0 warnings, 0 notes
```

Focused format check:

```text
uv run ruff format --check scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py
2 files already formatted
```

Focused whitespace diff check:

```text
git diff --check -- scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py
# no output
```

## Benchmark Rows / Blocker

No live cluster benchmark rows were collected for this Sub-AC. This task implements the budget metric calculation from parsed rows only. Live AR-vs-MTP evidence remains required before any throughput, 30+ tok/s, 40+ tok/s, or MTP speedup claim.

When the cluster API is available, rerun same-cluster AR and guarded MTP commands through `scripts/bench_mimo_mtp_cluster.py`, then feed the parsed metric rows to `calculate_mtp_speedup_budget(rows)`.

## Remaining Risk

- Tests use synthetic parsed rows; they verify budget calculation semantics but do not prove live cluster telemetry availability.
- MTP rows are intentionally counted only when `accepted_execution_path == "mimo_mtp_fastpath"`; fallback or missing-path rows will keep speedup evidence blocked until telemetry is wired and complete.
- Slice 5 remains blocked until same-cluster AR-vs-MTP benchmark rows prove a real MTP win without fallback/correctness concerns.
