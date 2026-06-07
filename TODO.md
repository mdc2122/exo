3. Task cancellation. When API http request gets cancelled, it should cancel corresponding task.
4. I'd like to see profiled network latency / bandwidth.
5. I'd like to see how much bandwidth each link is using.
7. Solve the problem of in continuous batching when a new prompt comes in, it will block decode of the current batch until the prefill is complete.
8. We want people to be able to copy models over to a new device without ever connecting EXO to the internet. Right now EXO require internet connection once to cache some files to check if a download is complete. Instead, we should simply check if there is a non-empty model folder locally with no .partial files. This indicates it's a fully downloaded model that can be loaded.
13. Memory pressure instead of memory used.
14. Show the type of each connection (TB5, Ethernet, etc.) in the UI. Refer to old exo: https://github.com/exo-explore/exo/blob/56f783b38dc6b08ce606b07a5386dc40dae00330/exo/helpers.py#L251
15. Prioritise certain connection types (or by latency). TB5 > Ethernet > WiFi. Refer to old exo: https://github.com/exo-explore/exo/blob/56f783b38dc6b08ce606b07a5386dc40dae00330/exo/helpers.py#L251
16. Dynamically switch to higher priority connection when it becomes available. Probably bring back InstanceReplacedAtomically.
17. Faster model loads by streaming model from other devices in cluster.
18. Add support for specifying the type of network connection to use in a test. Depends on 15/16.
25. Rethink retry logic
27. Log cleanup - per-module log filters and default to DEBUG log levels
28. Validate RDMA connections with ibv_devinfo in the info gatherer

## MiMo MTP fastpath readiness ledger — 2026-06-06

- Code-readiness wave completed for isolated `mimo_mtp_fast` surfaces using tiny/synthetic tests only; no exo startup, no live MiMo cluster, no full-model load, and no live AR-vs-MTP benchmark.
- Verification before ledger update: focused fastpath/script pytest `61 passed in 2.40s`; ruff on touched fastpath/script surfaces `All checks passed!`; ruff format check `19 files already formatted`; basedpyright on touched fastpath/script surfaces `0 errors, 0 warnings, 0 notes`; `git diff --check` clean.
- Beads notes updated: `.5ge.4` records benchmark/code-readiness and absent live rows; `.5ge.5` records Slice 5 blocked/default AR unchanged.
- Blocker/gate: no speedup or >=30 tok/s claim without same-model/same-hardware AR-vs-MTP rows; production/default MTP enablement remains blocked.

- Stabilization note: full repo pytest remains red with four deferred TurboQuant failures outside touched MiMo MTP fastpath surfaces; Nix formatter path was attempted but blocked by local disk exhaustion while building formatter dependencies, so touched Python surfaces were formatted/checked with ruff.

## MiMo MTP benchmark-survival ledger - 2026-06-06

- AC1-AC4 executed: benchmark CLI now supports --preflight-only and --load-only, emits structured stage/memory diagnostics, and focused fastpath/script checks remain green: 63 passed, ruff, ruff-format check, basedpyright, diff-check.
- Official local preflight succeeded with the quantized MiMo model path and official sidecar: model path validated, sidecar contract validated, benchmark_preflight.ready=true.
- Official local load-only failed with exit 137; last structured stage was base_model_materialization started. This confirms the local kill happens during full base-model materialization before generation.
- Minimal AR row blocked by load-only exit 137; minimal MTP D1 row skipped because AR did not load.
- Slice 5 remains blocked: no same-model/same-hardware AR-vs-MTP rows, no >=30 tok/s claim, no production/default MTP integration.

## MiMo MTP cluster benchmark correction - 2026-06-06

- Correction: local single-process exit 137 is not a model viability blocker for exo. The cluster can load MiMo via tensor parallelization; live rows must use the exo cluster API rather than a one-Studio local model load.
- Added scripts/bench_mimo_mtp_cluster.py to collect distributed /bench/chat/completions rows without loading the model inside the benchmark process.
- Next: start or point to the exo cluster API and collect AR baseline rows with the cluster harness; then add/enable a guarded cluster MTP request path so MTP rows are measured on the same cluster.

## MiMo MTP optimized rollout ultrawork — 2026-06-06

- Seed mirror: `.goose-ultrawork/seed.yaml` defines the performance-first optimized rollout tree.
- Goal: Implement the optimized MiMo V2.5 Pro MTP rollout strategy: a guarded exo-cluster MTP vertical slice with benchmark-grade telemetry, cluster AR baseline support, bottleneck analysis, and a strict Slice 5 gate for 30+ tok/s preferably 40+ tok/s.
- One-shot target: guarded exo-cluster MTP vertical slice with benchmark-grade telemetry, AR baseline support, bottleneck analysis, and strict Slice 5 gate.
- Probability optimization: baseline first, full vertical slice over partial plumbing, fail-closed honesty, sidecar lifecycle guard, acceptance-rate telemetry, auto-depth policy, and no Slice 5 without same-cluster speedup evidence.
- Slice 5 remains blocked until same-cluster AR-vs-MTP rows prove a real win; 30+ tok/s target, 40+ tok/s preferred.
- Mirrored acceptance criteria:
  - AC-P0 Optimized rollout seed and repo context are validated:
  - AC-P1 Cluster AR baseline harness is canonical and budget-ready:
  - AC-P2 MTP speedup budget model exists before optimization claims:
  - AC-P3 Explicit guarded MTP request contract is added and fail-closed:
  - AC-P4 MTP intent propagates through immutable internal task params:
  - AC-P5 Worker/generator guarded vertical slice is production-shaped but disabled by default:
  - AC-P6 Benchmark-grade MTP telemetry is defined and carried to cluster rows:
  - AC-P7 Same-cluster AR-vs-MTP benchmark matrix is runnable and evidence-gated:
  - AC-P8 Measurement-driven optimization loop is encoded:
  - AC-P9 Slice 5 gate remains strict:
  - AC-P10 Verification and closeout:

## MiMo MTP optimized rollout stabilization — 2026-06-06

- Goose-Ouroboros optimized rollout wave ended with raw RUN_EXIT=141, but generated changes were manually reviewed, repaired, formatted, and stabilized.
- Stabilized outputs: explicit experimental MTP request contract, internal MTP params propagation, fail-closed/fail-open unwired execution guard, cluster benchmark matrix command generation, benchmark row ingestion, 30/40 tok/s budget calculator, bottleneck classifier, MTP telemetry surfaces, module validation helpers, and AC evidence/report files.
- Verification after stabilization: focused pytest 146 passed; ruff check passed; ruff format check passed; basedpyright 0 errors; git diff --check clean; matrix command smoke emitted 8 rows; fixture budget analyzer returned same_cluster_budget_ready / at_least_30_tok_s.
- No live same-cluster AR-vs-MTP rows were collected; no >=30 or >=40 tok/s claim is made; Slice 5 production/default integration remains blocked.
- Next: run matrix commands against a running exo tensor-parallel cluster and then wire the distributed MTP execution backend so guarded MTP rows can be measured honestly.
