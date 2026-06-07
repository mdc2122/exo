# MiMo MTP Optimized Rollout Ultrawork Stabilization

Date: 2026-06-06

## Run outcome

The Goose-Ouroboros ultrawork wave was launched from .goose-ultrawork/seed.yaml.
The raw gou-run process ended with RUN_EXIT=141, so the orchestrator status is not treated as a clean completion.
The generated implementation was manually reviewed, repaired, formatted, and verified after the run.

## Accepted stabilized output

- Optimized rollout seed/control files under .goose-ultrawork/.
- Cluster benchmark harness enhancements in scripts/bench_mimo_mtp_cluster.py.
- Matrix command rendering for same-cluster AR and guarded MTP D1/D2/D3 rows at max_tokens=16 and max_tokens=64.
- MTP benchmark row ingestion, speedup budget calculator, and bottleneck classifier scripts.
- Explicit experimental MiMo MTP request contract fields.
- Internal MimoMtpFastpathParams propagation through TextGenerationTaskParams.
- Fail-closed/fail-open handling for requested-but-unwired MTP execution.
- Benchmark-grade MTP telemetry surfaces and tests.
- Module validation helpers for the MiMo MTP fastpath.
- Docs/runlog/TODO updates preserving the strict Slice 5 gate.

## Important limitation

This commit does not claim that distributed MTP execution is fully wired or faster than AR.
Explicit MTP requests are guarded and the unwired execution backend is reported honestly.
Default exo generation remains AR.
Slice 5 remains blocked until same-cluster AR-vs-MTP rows show a real speed win.

## Verification

- Focused pytest: 146 passed in 3.63s.
- Ruff check on touched surfaces: passed.
- Ruff format check on touched surfaces: 36 files already formatted.
- Basedpyright on touched surfaces: 0 errors, 0 warnings, 0 notes.
- git diff --check: clean.
- Matrix command smoke emitted 8 rows.
- Budget analyzer fixture smoke returned kind=mimo_mtp_budget_report, status=same_cluster_budget_ready, mtp_throughput_threshold=at_least_30_tok_s on fixture data.

## Next best step

Run the matrix command generator with the real sidecar path and collect actual AR baseline rows through a running exo tensor-parallel cluster.
Then complete the worker/generator distributed MTP backend so guarded MTP rows can be collected on the same cluster.
