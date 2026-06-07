# Sub-AC 3 — verifier_too_slow bottleneck classification

Timestamp: 2026-06-06 21:25 local

## Scope

Implemented a focused bottleneck classifier module for MiMo MTP benchmark rows with `verifier_too_slow` support.

The classifier emits `verifier_too_slow` only when verifier telemetry is present and crosses the configured slow-verifier threshold:

- verifier latency path: `timing_breakdown_seconds.verification` exists and verifier time share is `>= BottleneckThresholds.slow_verifier_time_share`;
- verifier throughput path: explicit or derived verifier throughput exists and is `<= BottleneckThresholds.slow_verifier_tokens_per_second` when that optional threshold is configured;
- missing verifier telemetry suppresses `verifier_too_slow`.

During verification, same-file RED coverage for `proposal_too_slow` was also present, so the minimal compatible proposal-throughput classifier was implemented without changing the Sub-AC's verifier gating semantics.

## Files changed

- `scripts/mimo_mtp_bottleneck_classifier.py`
  - Added `BottleneckThresholds` with configured slow verifier and proposal thresholds.
  - Added `classify_bottlenecks(...)` returning bottleneck labels.
  - Added guarded `verifier_too_slow` classification from timing/throughput telemetry.
  - Added missing-telemetry suppression paths.
- `scripts/test_mimo_mtp_bottleneck_classifier.py`
  - Positive `verifier_too_slow` test for verifier time-share crossing threshold.
  - Missing verifier telemetry suppression test.
  - Existing/same-file proposal throughput positive and missing-telemetry suppression coverage also passes.
- `.goose-ultrawork/reports/sub_ac_3_verifier_too_slow_bottleneck_classifier.md`
  - This evidence report.

## TDD evidence

RED command:

```bash
cd /Users/studio2/exo/.worktrees/mimo-v25-pro-mtp-fastpath-single-stream-20260601 && uv run pytest scripts/test_mimo_mtp_bottleneck_classifier.py -q
```

Initial RED result:

```text
ModuleNotFoundError: No module named 'scripts.mimo_mtp_bottleneck_classifier'
```

GREEN/focused verification command:

```bash
cd /Users/studio2/exo/.worktrees/mimo-v25-pro-mtp-fastpath-single-stream-20260601 && uv run pytest scripts/test_mimo_mtp_bottleneck_classifier.py -q && uv run ruff check scripts/mimo_mtp_bottleneck_classifier.py scripts/test_mimo_mtp_bottleneck_classifier.py && uv run ruff format --check scripts/mimo_mtp_bottleneck_classifier.py scripts/test_mimo_mtp_bottleneck_classifier.py && uv run basedpyright scripts/mimo_mtp_bottleneck_classifier.py scripts/test_mimo_mtp_bottleneck_classifier.py && git diff --check -- scripts/mimo_mtp_bottleneck_classifier.py scripts/test_mimo_mtp_bottleneck_classifier.py
```

Fresh verification result:

```text
....                                                                     [100%]
4 passed in 0.01s
All checks passed!
2 files already formatted
0 errors, 0 warnings, 0 notes
```

`git diff --check` produced no output and exited 0.

## Benchmark rows

No live cluster benchmark rows were required or collected for this Sub-AC. This change is classifier/test plumbing for later benchmark row analysis and does not assert MTP speedup, 30+ tok/s, or 40+ tok/s.

## Remaining risk

- The classifier is currently a focused script module and is not yet wired into a broader AC-P2 budget-report CLI or AC-P8 next-optimization recommender.
- Verifier throughput field naming may need extension if future cluster/API telemetry uses names other than `verification_tps`, `generation_tokens`, or `timing_breakdown_seconds.verification`.
- Slice 5 remains blocked until same-cluster AR-vs-MTP evidence proves a real MTP speed win.
