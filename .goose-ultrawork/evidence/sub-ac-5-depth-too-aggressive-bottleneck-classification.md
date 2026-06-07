# Sub-AC 5 — `depth_too_aggressive` bottleneck classification

Timestamp: 2026-06-06 21:28 local

## Acceptance criterion

Implement `depth_too_aggressive` bottleneck classification in the bottleneck classifier module/function, emitting it only when rollout-depth telemetry exists and crosses the configured aggressive-depth threshold, with runnable positive and missing-telemetry suppression tests.

## Files changed for this slice

- `src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py`
  - Extended existing `classify_bottlenecks(...)` with optional rollout-depth telemetry:
    - `rollout_depth: int | None = None`
    - `aggressive_depth_threshold: int | None = None`
  - Emits `depth_too_aggressive` only for non-AR modes when both rollout-depth telemetry and a configured threshold are present and `rollout_depth >= aggressive_depth_threshold`.
  - Keeps default AR behavior unchanged and suppresses the classification when telemetry is absent.
- `src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py`
  - Added positive coverage for aggressive rollout depth crossing the configured threshold.
  - Added missing-telemetry suppression coverage.
  - Added below-threshold suppression coverage to prove the configured threshold is honored.

Note: these files already contained adjacent uncommitted changes from prior governed acceptance-criterion work in this worktree, including the earlier `fallback_too_high` classifier slice and MTP depth eligibility checks. This evidence claims only the `depth_too_aggressive` classifier/test additions above.

## TDD evidence

### RED

Command run from repo root (`/Users/studio2/exo/.worktrees/mimo-v25-pro-mtp-fastpath-single-stream-20260601`):

```bash
uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py -q
```

Observed failing result before implementation:

```text
FAILED src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py::test_bottleneck_classifier_emits_depth_too_aggressive_when_rollout_depth_crosses_threshold
FAILED src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py::test_bottleneck_classifier_suppresses_depth_too_aggressive_without_rollout_depth_telemetry
TypeError: classify_bottlenecks() got an unexpected keyword argument 'rollout_depth'
2 failed, 19 passed in 0.07s
```

This proved the tests exercised missing rollout-depth classifier support rather than an already-implemented behavior.

### GREEN / verification

Fresh command run from repo root after implementation:

```bash
uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py -q && \
uv run ruff check src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py && \
uv run basedpyright src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py && \
git diff --check -- src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py
```

Observed result:

```text
......................                                                   [100%]
22 passed in 0.05s
All checks passed!
0 errors, 0 warnings, 0 notes
```

`git diff --check` completed with exit code 0 and no whitespace errors.

## Benchmark rows / speed claims

No live AR-vs-MTP cluster benchmark rows were collected for this Sub-AC. This slice is bottleneck-classifier unit coverage only. No >=30 tok/s, >=40 tok/s, MTP speedup, or Slice 5 eligibility claim is made.

## Remaining risk

- `depth_too_aggressive` is available in the in-engine benchmark classifier API, but live JSONL ingestion or cluster benchmark callers still need to pass rollout-depth telemetry and a configured aggressive-depth threshold for real row classification.
- The threshold value is intentionally caller-configured by this slice; operational calibration still requires same-cluster benchmark telemetry.
- The wider worktree contains additional governed rollout changes from other AC slices; this artifact scopes verification to the classifier/test surfaces listed above.
