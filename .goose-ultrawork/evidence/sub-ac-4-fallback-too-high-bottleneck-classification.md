# Sub-AC 4 — `fallback_too_high` bottleneck classification

## Acceptance criterion

Implement `fallback_too_high` bottleneck classification in the bottleneck classifier module/function, emitting it only when fallback-rate telemetry exists and crosses the configured high-fallback threshold, with runnable positive and missing-telemetry suppression tests.

## Files changed for this slice

- `src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py`
  - Added typed `BottleneckClassification` values.
  - Added `classify_bottlenecks(...)` with `fallback_too_high` gated on both fallback-rate telemetry presence (`fallback_rate is not None`) and the configured `high_fallback_threshold`.
- `src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py`
  - Added a positive test for high fallback-rate telemetry.
  - Added a suppression test proving missing fallback-rate telemetry does not emit `fallback_too_high`.

Note: these files already contained adjacent uncommitted changes from prior acceptance-criterion work in this governed worktree. This evidence claims only the classifier/test additions above.

## TDD evidence

### RED

Command:

```bash
uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py -q
```

Expected failing result before implementation:

```text
ImportError: cannot import name 'classify_bottlenecks' from 'exo.worker.engines.mlx.mimo_mtp_fast.benchmark'
```

### GREEN / verification

Fresh command run from repo root (`/Users/studio2/exo/.worktrees/mimo-v25-pro-mtp-fastpath-single-stream-20260601`):

```bash
uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py -q && \
uv run ruff check src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py && \
uv run basedpyright src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py && \
git diff --check && \
git diff --name-only -- src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py
```

Result:

```text
19 passed in 0.05s
All checks passed!
0 errors, 0 warnings, 0 notes
src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py
src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py
```

## Benchmark rows / speed claims

No live AR-vs-MTP benchmark rows were collected for this Sub-AC. This slice is bottleneck-classifier unit coverage only. No >=30 tok/s, >=40 tok/s, MTP speedup, or Slice 5 eligibility claim is made.

## Remaining risk

- `classify_bottlenecks(...)` now provides the guarded `fallback_too_high` classification path, but broader AC-P2 budget-calculator integration may still need to call this classifier from live JSONL ingestion rows once fallback-rate telemetry is available there.
- Threshold selection remains caller-configured; this slice verifies only that the configured threshold is honored and missing telemetry suppresses the classification.
