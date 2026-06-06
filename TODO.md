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
