# Sub-AC 3.3.1 Evidence: absent/insufficient MTP telemetry classifies as ambiguous

Status: complete

## Scope

Implemented and tested benchmark classification behavior for absent or insufficient guarded MTP telemetry. When an MTP comparison row lacks live fastpath `generation_tps` telemetry, the classifier now returns an explicit ambiguous outcome instead of pass or fail.

## Files changed for this Sub-AC

- `scripts/test_bench_mimo_mtp_cluster.py`
  - Added regression coverage for absent MTP telemetry in `classify_mtp_vs_ar_baseline(...)`.
  - Expected outcome is `classification="ambiguous"` with `bottlenecks=["absent_mtp_telemetry"]` and no speedup/target claim.
- `scripts/bench_mimo_mtp_cluster.py`
  - Refined `classify_mtp_vs_ar_baseline(...)` ambiguous branch to distinguish missing/zero MTP `generation_tps` telemetry from generic invalid comparison inputs.
  - Preserves fail-closed/no-claim guidance by requiring same-cluster guarded MTP rows with `accepted_execution_path=mimo_mtp_fastpath` and `generation_tps` before speedup or target claims.
  - Added/retained the same-cluster `calculate_mtp_speedup_budget(...)` helper required by the existing budget tests so absent rows remain blocked/incomplete rather than fabricated.

Note: the target worktree contains prior/parallel uncommitted changes in these files. This evidence is scoped to the ambiguous absent-MTP-telemetry classification and adjacent focused budget helper behavior verified below.

## TDD evidence

RED command:

```bash
uv run pytest scripts/test_bench_mimo_mtp_cluster.py::test_classify_mtp_vs_ar_baseline_marks_absent_mtp_telemetry_ambiguous -q
```

Observed RED failure before implementation:

```text
FAILED scripts/test_bench_mimo_mtp_cluster.py::test_classify_mtp_vs_ar_baseline_marks_absent_mtp_telemetry_ambiguous
Differing items:
{'bottlenecks': []} != {'bottlenecks': ['absent_mtp_telemetry']}
{'reason': 'classification requires positive same-cluster AR and MTP generation_tps rows'} != {'reason': 'MTP row lacks live fastpath generation_tps telemetry; cannot classify pass or fail'}
```

GREEN command after implementation:

```bash
uv run pytest scripts/test_bench_mimo_mtp_cluster.py::test_classify_mtp_vs_ar_baseline_marks_absent_mtp_telemetry_ambiguous -q
```

Observed result:

```text
1 passed in 0.05s
```

## Verification commands run

Focused benchmark harness tests:

```bash
uv run pytest scripts/test_bench_mimo_mtp_cluster.py -q
```

Observed result:

```text
25 passed in 0.13s
```

Lint:

```bash
uv run ruff check scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py
```

Observed result:

```text
All checks passed!
```

Format check:

```bash
uv run ruff format --check scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py
```

Observed result:

```text
2 files already formatted
```

Type check:

```bash
uv run basedpyright scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py
```

Observed result:

```text
0 errors, 0 warnings, 0 notes
```

Whitespace check:

```bash
git diff --check -- scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py
```

Observed result: exit code 0 with no output.

## Benchmark/live rows

No live cluster benchmark rows were collected for this Sub-AC. This is a classification/telemetry regression slice only. No 30+ tok/s, 40+ tok/s, MTP speedup, or Slice 5 eligibility claim is made.

## Blockers

None for this Sub-AC.

## Remaining risk

- Live guarded MTP execution remains separately evidence-gated; ambiguous classification only prevents absent/insufficient telemetry from being misreported as pass/fail.
- Worktree contains prior/parallel uncommitted changes, so final integration should review the combined diff before merge.
