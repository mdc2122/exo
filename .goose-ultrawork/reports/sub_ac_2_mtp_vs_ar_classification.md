# Sub-AC 2 Evidence: MTP versus AR baseline classification

Timestamp: 2026-06-06 21:27 local

## Scope

Implemented and tested classification for guarded MiMo MTP rows against same-cluster AR baseline rows. The classifier covers:

- MTP beats AR: `classification=pass`, includes `mtp_beats_ar` plus target-band labels such as `mtp_reaches_30` / `mtp_reaches_40` when thresholds are met.
- MTP does not beat AR: `classification=fail`, with bottleneck labels from available telemetry such as `acceptance_rate_low`.
- Absent MTP telemetry: `classification=ambiguous`, `bottlenecks=["absent_mtp_telemetry"]`, and no fabricated speedup/target metrics.

This preserves the Seed constraint that no 30+ tok/s, 40+ tok/s, or MTP speedup claim is made without same-cluster AR-vs-MTP evidence rows.

## Files changed for this Sub-AC

- `scripts/bench_mimo_mtp_cluster.py`
  - Adds `classify_mtp_vs_ar_baseline(...)` for one AR row vs one guarded MTP row.
  - Computes AR/MTP ms per token, MTP target gaps for 30 and 40 tok/s, speedup ratio, pass/fail/ambiguous classification, and bottleneck labels.
  - Adds aggregate `calculate_mtp_speedup_budget(...)` support for row sets, preserving blocked status when AR or MTP rows/telemetry are absent.
  - Keeps same-cluster requirements explicit through `kind=cluster_benchmark_metric`, `cluster_path=exo_api_bench_chat_completions`, AR mode, non-AR MTP mode, and matching model.

- `scripts/test_bench_mimo_mtp_cluster.py`
  - Adds regression coverage for MTP beating AR.
  - Adds regression coverage for MTP not beating AR.
  - Adds/keeps coverage for absent MTP telemetry and aggregate budget outcomes.

Note: this worktree contains neighboring rollout changes from other ACs. This report claims only the MTP-vs-AR classification and focused budget-classification surface above.

## TDD / RED evidence

The two requested behavior tests were written before the classifier implementation and failed as expected because the production API did not exist yet:

```bash
cd .. && uv run pytest \
  scripts/test_bench_mimo_mtp_cluster.py::test_classify_mtp_vs_ar_baseline_marks_mtp_speed_win \
  scripts/test_bench_mimo_mtp_cluster.py::test_classify_mtp_vs_ar_baseline_marks_mtp_non_win \
  -q
```

Observed RED result:

```text
FF                                                                       [100%]
AttributeError: module 'scripts.bench_mimo_mtp_cluster' has no attribute 'classify_mtp_vs_ar_baseline'
2 failed in 0.08s
```

## GREEN / verification evidence

Targeted classifier/budget tests:

```bash
cd .. && uv run pytest \
  scripts/test_bench_mimo_mtp_cluster.py::test_mtp_metric_row_classifies_generation_tps_below_30_tok_s \
  scripts/test_bench_mimo_mtp_cluster.py::test_mtp_metric_row_classifies_generation_tps_at_least_30_tok_s \
  scripts/test_bench_mimo_mtp_cluster.py::test_mtp_metric_row_classifies_generation_tps_at_least_40_tok_s \
  scripts/test_bench_mimo_mtp_cluster.py::test_classify_mtp_vs_ar_baseline_marks_mtp_speed_win \
  scripts/test_bench_mimo_mtp_cluster.py::test_classify_mtp_vs_ar_baseline_marks_mtp_non_win \
  scripts/test_bench_mimo_mtp_cluster.py::test_classify_mtp_vs_ar_baseline_marks_absent_mtp_telemetry_ambiguous \
  scripts/test_bench_mimo_mtp_cluster.py::test_calculate_mtp_speedup_budget_for_ar_only_rows \
  scripts/test_bench_mimo_mtp_cluster.py::test_calculate_mtp_speedup_budget_for_ar_and_mtp_rows \
  scripts/test_bench_mimo_mtp_cluster.py::test_calculate_mtp_speedup_budget_marks_incomplete_telemetry_without_fabricating_metrics \
  -q
```

Observed result:

```text
.........                                                                [100%]
9 passed in 0.05s
```

Fresh focused verification before closeout:

```bash
cd .. && uv run pytest scripts/test_bench_mimo_mtp_cluster.py -q && \
uv run ruff check scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py && \
uv run ruff format --check scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py && \
uv run basedpyright scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py && \
git diff --check -- scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py
```

Observed result:

```text
.........................                                                [100%]
25 passed in 0.13s
All checks passed!
2 files already formatted
0 errors, 0 warnings, 0 notes
```

`git diff --check` completed with exit code 0 and no output.

## Benchmark rows / blocker status

No live cluster benchmark rows were collected for this Sub-AC. This task implemented classification logic and unit regression tests only. No >=30 tok/s, >=40 tok/s, or MTP speedup claim is made.

Live same-cluster evidence remains blocked pending an explicitly available/safe exo cluster API and guarded MTP execution path. Rerun commands once the cluster is available include:

```bash
uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 16 --repeats 1 --mode-label ar --list-models
uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 64 --repeats 1 --mode-label ar --list-models
uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 16 --repeats 3 --mode-label mtp-d1 --payload-extra-json '{"mimo_mtp_fastpath":true,"mimo_mtp_depth":1,"mimo_mtp_fail_closed":true}'
uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 64 --repeats 3 --mode-label mtp-d1 --payload-extra-json '{"mimo_mtp_fastpath":true,"mimo_mtp_depth":1,"mimo_mtp_fail_closed":true}'
```

## Remaining risk

- Classification tests use synthetic rows; they prove the budget/classification semantics but do not prove live cluster performance.
- Same-cluster median AR-vs-MTP rows are still required before any speedup, 30+ tok/s, 40+ tok/s, or Slice 5 eligibility claim.
- Bottleneck labels are limited to telemetry present in rows; richer proposal/verifier/fallback labels depend on later MTP telemetry surfacing.
