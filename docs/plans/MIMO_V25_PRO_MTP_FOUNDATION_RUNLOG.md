# MiMo V2.5 Pro MTP Foundation Runlog

Date: 2026-05-19

## Known-Good Baseline

- Baseline branch: `backup/mimo-v25-pro-6bit-working-baseline-20260519`
- Baseline commit: `1114ff855fa8f5d897388b50ccc6011c205dbf39`
- Runtime branch at planning start: `con-75-mimo-pro-live-integration-20260504-200136`
- Current MTP design spec: `docs/superpowers/specs/2026-05-19-mimo-v25-pro-mtp-design.md`

## Runtime Facts

- Model id: `kernelpool/MiMo-V2.5-Pro-6bit`
- Main artifact: `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit`
- Main artifact size: about `778G` on disk
- Safetensors metadata size: `835299199488` bytes
- Source MTP shard: `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/hf/XiaomiMiMo--MiMo-V2.5-Pro/model_mtp.safetensors`
- Source MTP shard size: about `2.3G`
- Preferred performance placement: tensor/JACCL over TB5

## Throughput Baseline

- Single request: about `22 tok/s`
- Batched aggregate: about `40 tok/s`
- Benchmark artifacts:
  - machine-local evidence on Studio2:
  - `/Users/studio2/Documents/Codex/2026-05-19/superpowers-can-we-determine-whether-22/mimo_v25_pro_sustain_probe_20260519.json`
  - `/Users/studio2/Documents/Codex/2026-05-19/superpowers-can-we-determine-whether-22/mimo_v25_pro_concurrency_probe_20260519.json`
  - `/Users/studio2/Documents/Codex/2026-05-19/superpowers-can-we-determine-whether-22/mimo_v25_pro_concurrency8_probe_20260519.json`

## Dirty Work Preserved

- Patch artifact: `docs/plans/artifacts/mimo-v25-pro-kv-cache-wip-20260519.patch`
- This is a raw preservation patch and may retain source whitespace exactly; this is intentional.
- Patch scope:
  - `src/exo/worker/engines/mlx/cache.py`
  - `src/exo/worker/engines/mlx/generator/batch_generate.py`
  - `src/exo/worker/engines/mlx/generator/generate.py`
  - `src/exo/worker/tests/unittests/test_mlx/test_kv_prefix_cache.py`

## Observed State

- `git status --short --branch` showed branch `con-75-mimo-pro-live-integration-20260504-200136` ahead of `fork/con-75-mimo-pro-live-integration-20260504-200136` by `3`, with the four KV/cache runtime files dirty.
- `git rev-parse HEAD` returned `69e60f26e5cf602f64e71101cfd888628909404b`.
- `git rev-parse fork/con-75-mimo-pro-live-integration-20260504-200136` returned `1114ff855fa8f5d897388b50ccc6011c205dbf39`.
- `git rev-parse backup/mimo-v25-pro-6bit-working-baseline-20260519` returned `1114ff855fa8f5d897388b50ccc6011c205dbf39`.
- `test -s docs/plans/artifacts/mimo-v25-pro-kv-cache-wip-20260519.patch` passed.

## Recovery Recipe

To return to the known-good baseline code, start from a clean worktree or a fresh
clone/worktree. Do not switch branches in a dirty recovery workspace unless you
have first preserved unrelated edits with `git stash push` or by copying them
elsewhere.

From that clean baseline recovery workspace, use a self-healing switch that
re-creates the backup branch locally if it is missing:

```bash
git switch backup/mimo-v25-pro-6bit-working-baseline-20260519 || git switch -c backup/mimo-v25-pro-6bit-working-baseline-20260519 1114ff855fa8f5d897388b50ccc6011c205dbf39
```

Only if you intentionally want to restore the preserved KV/cache WIP after
recovering the baseline, apply the patch artifact in that clean workspace:

```bash
git apply docs/plans/artifacts/mimo-v25-pro-kv-cache-wip-20260519.patch
```

## Safety Rule

Do not start a full MiMo V2.5 Pro model if any MiMo instance is active, loading,
warming, stale-but-resident, or if either Studio node has not recovered memory.
Duplicate full MiMo startup can crash the Mac Studios.

## Foundation Verification

- Focused tests: `uv run pytest scripts/test_mimo_v25_pro_runtime_guard.py scripts/test_mimo_v25_pro_mtp_artifact_probe.py -q`
  - Result: `13 passed in 0.53s`
- Lint: `uv run ruff check scripts/mimo_v25_pro_runtime_guard.py scripts/test_mimo_v25_pro_runtime_guard.py scripts/mimo_v25_pro_mtp_artifact_probe.py scripts/test_mimo_v25_pro_mtp_artifact_probe.py`
  - Result: `All checks passed!`
- Typecheck: `uv run basedpyright scripts/mimo_v25_pro_runtime_guard.py scripts/test_mimo_v25_pro_runtime_guard.py scripts/mimo_v25_pro_mtp_artifact_probe.py scripts/test_mimo_v25_pro_mtp_artifact_probe.py`
  - Result: `0 errors, 0 warnings, 0 notes`
- Real MTP shape report: `docs/plans/artifacts/mimo-v25-pro-mtp-shape-report-20260519.json`
  - Result: `tensor_count=48`, `layers=[0, 1, 2]`, `complete_expected_layers=true`
- Live duplicate-start guard: `uv run python scripts/mimo_v25_pro_runtime_guard.py --exo-url http://127.0.0.1:52415`
  - Result: `safe=false`; reasons included `active MiMo instance 465e23f7-c776-4e9d-8760-cf768db9c56c`, four loading runners, one resident MiMo-related telemetry process match, and both nodes below the memory floor
- Staged-state check: `git diff --cached --name-only`
  - Result: `empty`
- Working-tree check: `git status --short`
  - Result: only the four pre-existing dirty KV/cache runtime files remained unstaged; the verification commit was doc-only despite the intentionally dirty working tree

## 2026-06-06 Fastpath Readiness Wave

Scope: guarded Ouroboros AC-tree code-readiness work for the isolated native MiMo
V2.5 Pro MTP fastpath. This wave did not start exo, did not launch a MiMo
cluster, did not materialize/load the full MiMo model, did not run a live
AR-vs-MTP benchmark, and did not enable MTP in default generation.

Implemented/verified readiness surfaces:

- Official sidecar contract diagnostics: required tensor role/dtype/rank checks
  now report actionable fastpath errors with key, role, dtype, and shape.
- Sidecar module semantics: FP8 block dequantization remains constructor-time,
  with focused coverage, and grouped QKV split behavior is tested with tiny MLX
  fixtures only.
- One-cycle proposal/verification semantics: proposal/verifier ordering,
  longest-prefix acceptance, verifier-short output, D1/D2/D3 deterministic fake
  behavior, and invalid provider outputs are covered.
- Streaming loop observability/failure policy: per-cycle traces, attempted and
  accepted depth counts, exact `max_tokens`, runtime fallback classification,
  structural fail-closed classification, and stale MTP cache reset hooks are
  covered without Ralph machinery.
- Provider seam hardening: MLX-shaped proposal/verifier protocols now validate
  token dtype/shape, hidden-state shape, and logits dtype/shape with
  fastpath-specific errors. Replay one-cycle runner remains the benchmark
  correctness reference.
- Benchmark readiness: JSON metric rows include acceptance-rate and timing
  breakdown diagnostics. `scripts/bench_mimo_mtp_fastpath.py` keeps the existing
  dry-run sidecar contract path and now emits clear pre-load validation JSON for
  missing model paths and invalid sidecars.
- Synthetic probe: `scripts/mimo_mtp_fastpath_synthetic_probe.py` exercises the
  tiny/synthetic fastpath timing and acceptance diagnostics without loading a
  full MiMo model.

Verification evidence before ledger update:

- Focused fastpath/script tests: `uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast*.py scripts/test_bench_mimo_mtp_fastpath.py scripts/test_mimo_mtp_fastpath_synthetic_probe.py -q`
  - Result: `61 passed in 2.55s`
- Lint: `uv run ruff check src/exo/worker/engines/mlx/mimo_mtp_fast src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_one_cycle.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_providers.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_sidecar_contract.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_sidecar_module.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_speculative_loop.py scripts/bench_mimo_mtp_fastpath.py scripts/test_bench_mimo_mtp_fastpath.py scripts/mimo_mtp_fastpath_synthetic_probe.py scripts/test_mimo_mtp_fastpath_synthetic_probe.py`
  - Result: `All checks passed!`
- Typecheck: `uv run basedpyright src/exo/worker/engines/mlx/mimo_mtp_fast src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_one_cycle.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_providers.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_sidecar_contract.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_sidecar_module.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_speculative_loop.py scripts/bench_mimo_mtp_fastpath.py scripts/test_bench_mimo_mtp_fastpath.py scripts/mimo_mtp_fastpath_synthetic_probe.py scripts/test_mimo_mtp_fastpath_synthetic_probe.py`
  - Result: `0 errors, 0 warnings, 0 notes`
- Diff whitespace: `git diff --check`
  - Result: clean / no output

Beads updates:

- Added a code-readiness note to `mimo-v25-pro-mtp-d1-live-preflight-20260601-5ge.4`.
- Added a Slice 5 blocked/default-AR note to `mimo-v25-pro-mtp-d1-live-preflight-20260601-5ge.5`.

Commit IDs: none in this execution at ledger-update time.

Live benchmark blocker / performance gate:

- No live AR-vs-MTP rows were produced in this workflow by design and constraint.
- No >=30 tok/s or MTP speedup claim is made by this readiness wave.
- Slice 5 production/default integration remains blocked. Default generation is
  still normal AR, and this diff touches no generator/API/master/model-card or
  inference-card integration path.
- Next external gate remains same-model/same-hardware AR-vs-MTP measurement with
  real rows showing MTP clearly beats AR before any guarded integration review.


## 2026-06-06 Manual Stabilization Follow-up

After the Ouroboros/Goose readiness execution, manual stabilization was run in
`/Users/studio2/exo/.worktrees/mimo-v25-pro-mtp-fastpath-single-stream-20260601`.
The Ouroboros runtime completed successfully, but its post-execution QA verdict
was `revise` because repository-wide canonical checks include unrelated/deferred
TurboQuant failures and the Nix formatter/build path encountered local disk
pressure. The fastpath-specific code-readiness surface was then stabilized
manually.

Manual verification after formatting:

- Focused fastpath/script tests: `uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast*.py scripts/test_bench_mimo_mtp_fastpath.py scripts/test_mimo_mtp_fastpath_synthetic_probe.py -q`
  - Result: `61 passed in 2.40s`
- Lint: `uv run ruff check` on touched fastpath/script surfaces
  - Result: `All checks passed!`
- Python formatting: `uv run ruff format --check` on touched fastpath/script surfaces
  - Result: `19 files already formatted`
- Typecheck: `uv run basedpyright` on touched fastpath/script surfaces
  - Result: `0 errors, 0 warnings, 0 notes`
- Diff whitespace: `git diff --check`
  - Result: clean / no output
- Official sidecar dry-run: `uv run python scripts/bench_mimo_mtp_fastpath.py --sidecar-path /Volumes/GLM5-NVMe/exo/mimo-v25-pro/hf/XiaomiMiMo--MiMo-V2.5-Pro/model_mtp.safetensors --dry-run-contract-only`
  - Result: JSON row with `"ready": true`, `"sidecar_status": "ready"`, and `"sidecar_layer_count": 3`
- Synthetic probe smoke: `uv run python scripts/mimo_mtp_fastpath_synthetic_probe.py --max-tokens 4 --requested-depth 3`
  - Result: JSON row with `"full_model_loaded": false`, timing breakdowns, and acceptance diagnostics
- Missing-model validation smoke: `uv run python scripts/bench_mimo_mtp_fastpath.py --model-path /tmp/definitely-missing-mimo-model --sidecar-path /tmp/definitely-missing-mtp-sidecar.safetensors`
  - Result: exit code `2` with JSON `validation_error` for `model_path`, before model or sidecar materialization

Known non-fastpath blockers observed during stabilization:

- Full `uv run pytest -q` remains red with four existing/deferred TurboQuant failures outside the touched MiMo MTP fastpath surfaces:
  - `test_make_kv_cache_uses_turboquant_for_glm_moe_dsa`
  - `test_make_kv_cache_sets_fused_flag_and_applies_patch`
  - `test_warmup_inference_skips_generation_when_turboquant_active`
  - `test_builder_keeps_batch_generator_for_single_node_turboquant`
- `nix --extra-experimental-features 'nix-command flakes' fmt` was attempted but failed while building formatter dependencies because the local system volume ran out of space. Nix GC recovered several GiB, but the full Nix formatter path was not completed in this stabilization.

Gate remains unchanged: no live AR-vs-MTP rows exist, no >=30 tok/s claim is
made, and Slice 5 production/default integration remains blocked.

## 2026-06-06 Benchmark Survival Diagnostics Follow-up

Executed the AC1-AC7 benchmark-survival wave after the fastpath readiness commit.
The benchmark CLI now has --preflight-only and --load-only modes, and emits
structured JSON-lines benchmark_stage diagnostics with elapsed time and memory
snapshots at major stages.

Focused verification after the change:

- Focused fastpath/script tests: uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast*.py scripts/test_bench_mimo_mtp_fastpath.py scripts/test_mimo_mtp_fastpath_synthetic_probe.py -q
  - Result: 63 passed in 2.47s
- Lint: uv run ruff check on touched fastpath/script surfaces
  - Result: All checks passed!
- Python formatting: uv run ruff format --check on touched fastpath/script surfaces
  - Result: 12 files already formatted
- Typecheck: uv run basedpyright on touched fastpath/script surfaces
  - Result: 0 errors, 0 warnings, 0 notes
- Diff whitespace: git diff --check
  - Result: clean / no output

Official local artifact preflight:

- Command: uv run python scripts/bench_mimo_mtp_fastpath.py --model-path /Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit --sidecar-path /Volumes/GLM5-NVMe/exo/mimo-v25-pro/hf/XiaomiMiMo--MiMo-V2.5-Pro/model_mtp.safetensors --modes ar,d1 --max-tokens 1 --preflight-only
  - Result: exit 0, benchmark_preflight.ready=true; completed stages were mode validation, model path validation, and sidecar contract validation.

Official local artifact load-only:

- Command: uv run python scripts/bench_mimo_mtp_fastpath.py --model-path /Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit --sidecar-path /Volumes/GLM5-NVMe/exo/mimo-v25-pro/hf/XiaomiMiMo--MiMo-V2.5-Pro/model_mtp.safetensors --modes ar --max-tokens 1 --load-only
  - Result: exit 137; last structured row was base_model_materialization with status=started. No materialization-completed row, tokenizer-load row, sidecar tensor-load row, prompt-tokenization row, AR metric row, or MTP metric row was produced.

Because load-only failed, AC5 minimal AR generation was recorded as blocked and
AC6 minimal MTP D1 generation was skipped. AC7 remains unchanged: no live
AR-vs-MTP rows exist, no speedup claim is made, and Slice 5 production/default
integration remains blocked.

## 2026-06-06 Cluster Benchmark Direction Correction

The single-process benchmark load-only exit 137 is not evidence that the exo cluster cannot run MiMo. It only proves that one process on one Studio cannot materialize the full quantized model locally. The correct live benchmark path for MiMo V2.5 Pro is the exo cluster path, where tensor parallelization distributes the model across nodes.

Added scripts/bench_mimo_mtp_cluster.py as the cluster-facing benchmark harness. It posts to an already-running exo API /bench/chat/completions endpoint and emits JSON-lines rows with generation_stats.generation_tps, token counts, power usage, repeat index, and mode labels. This harness does not load model weights in the benchmark process.

Current implication:

- Use the local sidecar/fastpath CLI for contract, synthetic, and provider diagnostics only.
- Use scripts/bench_mimo_mtp_cluster.py for live distributed AR rows.
- MTP distributed rows still require a guarded cluster request path or worker integration flag before they can be honestly collected through the cluster.
- Slice 5 production/default enablement remains blocked until same-cluster AR and guarded-MTP rows show a real MTP speed win.

## 2026-06-07 Sub-AC 4 Exact Cluster AR Baseline Commands

The canonical live MiMo V2.5 Pro benchmark harness path is the exo cluster API
endpoint `/bench/chat/completions`, driven by `scripts/bench_mimo_mtp_cluster.py`.
The benchmark process must not materialize the full model locally; it posts to an
already-running tensor-parallel exo API.

Run these exact AR baseline commands before making any MTP speedup, >=30 tok/s,
>=40 tok/s, or Slice 5 eligibility claim. These commands intentionally use
`--mode-label ar`, include `--list-models`, and provide no `--payload-extra-json`,
so default generation remains AR and `mtp_enabled=false` in the resulting rows.

```bash
uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 16 --repeats 1 --mode-label ar --list-models
```

```bash
uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 64 --repeats 1 --mode-label ar --list-models
```

Guarded MTP opt-in is separate and must be explicit. A row may not be treated as
successful MTP unless the request includes guarded experimental fields such as
`mimo_mtp_fastpath=true`, `mimo_mtp_fail_closed=true`, a matching
`mimo_mtp_depth`, and the execution telemetry proves the `mimo_mtp_fastpath`
path was attempted/accepted. AR behavior with MTP-shaped fields must be labeled
as disabled, fail-closed, or fail-open fallback rather than successful MTP.

## 2026-06-06 Optimized Rollout Ultrawork Seed

Created or refreshed `.goose-ultrawork/seed.yaml` as the source of truth for a performance-first MiMo MTP rollout AC tree.
The tree optimizes for a one-shot guarded cluster vertical slice rather than a broad production rollout. The critical path is:

1. collect or require same-cluster AR baseline rows through `/bench/chat/completions`,
2. compute 30+/40+ tok/s budget and bottleneck classification,
3. add explicit guarded MTP request/task contract,
4. route compatible requests through production-shaped MTP scaffolding or fail/fallback with honest telemetry,
5. preserve benchmark-grade MTP telemetry in cluster rows,
6. run AR-vs-MTP matrix when cluster and guarded MTP path are available,
7. keep Slice 5 blocked unless same-cluster AR-vs-MTP rows prove a real speed win.

This seed explicitly corrects the local single-Studio load framing: live MiMo rows must use the exo tensor-parallel cluster path, while local scripts remain for sidecar, synthetic, and provider diagnostics.

Mirrored acceptance criteria:
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

## 2026-06-06 Optimized Rollout Ultrawork Stabilization

The optimized Goose-Ouroboros ultrawork wave was launched from .goose-ultrawork/seed.yaml. The raw gou-run ended with RUN_EXIT=141, so the orchestrator result is recorded as failed rather than a clean AC completion. However, the generated changes were manually reviewed and stabilized.

Accepted stabilized outputs:

- Experimental MiMo MTP request fields and internal immutable MimoMtpFastpathParams propagation.
- Fail-closed/fail-open handling for explicit MTP requests while the distributed execution backend remains unwired. Requested MTP is not silently reported as AR.
- Cluster benchmark harness support for preserving telemetry and rendering the same-cluster AR/MTP matrix commands.
- Benchmark ingestion, speedup budget, and bottleneck classifier scripts for 30+/40+ tok/s evidence analysis.
- Module validation helpers for MTP stack semantics.
- Docs/TODO/ultrawork evidence updates preserving the Slice 5 gate.

Verification after stabilization:

- Focused pytest across scripts/API/MTP fastpath tests: 146 passed in 3.63s
- uv run ruff check on touched surfaces: passed
- uv run ruff format --check on touched surfaces: passed; 36 files already formatted
- uv run basedpyright on touched surfaces: 0 errors, 0 warnings, 0 notes
- git diff --check: clean
- Matrix command smoke emitted 8 rows for AR plus MTP D1/D2/D3 at max_tokens 16 and 64.
- Budget analyzer fixture smoke returned same_cluster_budget_ready and at_least_30_tok_s for fixture data.

Gate status remains unchanged: no live same-cluster AR-vs-MTP rows were collected, no >=30 tok/s or >=40 tok/s claim is made, and Slice 5 production/default integration remains blocked.

### 2026-06-06T22:51:00 Sub-AC 3 runlog_mirror_sink

- Summary: Implemented append-only runlog bookkeeping sink for benchmark/rollout AC evidence without TODO or Beads parameters.
- Evidence:
  - files changed: scripts/mirror_mimo_mtp_rollout_seed.py, scripts/test_mirror_mimo_mtp_rollout_seed.py
  - RED: uv run pytest scripts/test_mirror_mimo_mtp_rollout_seed.py::test_append_runlog_bookkeeping_record_appends_without_rewriting_existing_runlog scripts/test_mirror_mimo_mtp_rollout_seed.py::test_append_runlog_bookkeeping_record_does_not_touch_todo_or_beads -q -> 2 failed with missing append_runlog_bookkeeping_record
  - GREEN: same targeted pytest -> 2 passed in 0.01s
  - Fresh scoped verification: targeted/full mirror seed tests + rollout_seed_mirroring tests -> 12 passed in 0.02s; ruff check passed; ruff format --check already formatted; basedpyright 0 errors; git diff --check passed
  - benchmark rows: not applicable; this Sub-AC is bookkeeping sink plumbing and does not claim tok/s, MTP speedup, or Slice 5 eligibility
- Remaining risk: Adjacent Beads CLI guard tests in scripts/test_mirror_rollout_seed_to_beads.py remain outside this runlog-sink slice; Slice 5 remains blocked until same-cluster AR-vs-MTP evidence proves a real speed win.

## 2026-06-07 AC-P9 Strict Slice 5 Gate

AC-P9 tightened the Slice 5 budget gate without enabling MTP by default.
Default exo generation remains AR; guarded MiMo MTP remains an explicit
experimental request/runtime path only.

Slice 5 status is **blocked** in this worktree. No live same-cluster AR-vs-guarded-MTP
benchmark rows were collected, and no >=30 tok/s, >=40 tok/s, MTP speedup, or
production/default integration claim is made.

The calculator now reports Slice 5 review eligibility separately from raw speedup
claims. Guarded production integration review is allowed only when all of these
are true:

1. same-cluster AR and live guarded MTP rows are present,
2. guarded MTP median tok/s beats AR and either reaches at least 1.10x (10%; 10-15% is the review band) or reaches the explicit >=30 tok/s target,
3. 40+ tok/s remains the preferred target,
4. no correctness/fallback concern is present; rows with `fallback_too_high` keep Slice 5 blocked.

Focused AC-P9 tests cover small-speedup blocking and fallback-concern blocking.
The existing default AR behavior was not changed.

## 2026-06-07 AC-P10 Verification and Closeout

AC-P10 closed the optimized MiMo V2.5 Pro MTP rollout slice with fresh focused
verification across the touched scripts/API/shared/worker surfaces and local
bookkeeping artifacts.

Closeout artifacts updated:

- `.goose-ultrawork/evidence/ac-p10-verification-closeout.md`
- `.goose-ultrawork/reports/final_stabilization.md`
- `.goose-ultrawork/evidence/sub-ac-4-beads-mirror-sink.md`
- `TODO.md`
- this runlog

Verification evidence:

- Focused pytest across touched rollout surfaces: `225 passed in 4.28s`.
- Ruff check over touched code/test surfaces: `All checks passed!`.
- Ruff format check initially reported `Would reformat: scripts/test_git_cleanliness.py`; after `uv run ruff format scripts/test_git_cleanliness.py`, rerun reported `41 files already formatted`.
- Basedpyright over touched code/test surfaces: `0 errors, 0 warnings, 0 notes`.
- `git diff --check`: clean / no output.

AC status summary:

- AC-P0 through AC-P10 have local evidence artifacts and are complete for the guarded rollout slice.
- Live same-cluster benchmark collection remains blocked by the lack of an available cluster API in this execution environment.
- Slice 5 remains blocked. No live same-cluster AR-vs-guarded-MTP rows exist, no >=30 tok/s or >=40 tok/s claim is made, no MTP speedup claim is made, and MTP is not enabled as production/default generation.

Beads status:

- No live Beads rows were mutated during AC-P10.
- Beads mirroring remains explicitly opt-in through `uv run python3 scripts/mirror_rollout_seed_to_beads.py .goose-ultrawork/seed.yaml --enable-beads-mirror`.
- The Beads-facing closeout status is recorded in `.goose-ultrawork/evidence/sub-ac-4-beads-mirror-sink.md` for guarded mirroring.

Next best step: run the matrix commands from `.goose-ultrawork/evidence/ac-p7-benchmark-matrix-commands.jsonl` against a healthy exo tensor-parallel cluster API with the official MiMo MTP sidecar, then analyze the same-cluster rows with the budget and bottleneck scripts before any Slice 5 production/default review.

AC-P10 final post-format pytest rerun: after `scripts/test_git_cleanliness.py`
was formatted, the same focused touched-surface pytest command was rerun and
reported `225 passed in 4.23s`.

### MiMo MTP rollout tracking: sub-ac-3-runlog-artifact-synchronization

- Last updated: 2026-06-07T01:51:00Z
- Acceptance criterion: Sub-AC 3
- Summary: Implemented canonical-payload runlog artifact synchronization with append/update semantics, timestamped gate status, and bottleneck-analysis fields while preserving prior runlog history.
- Benchmark rows:
  - same_cluster_ar_rows: 0
  - same_cluster_mtp_rows: 0
  - live_rows_status: not_collected_for_runlog_sync_only
- Gate status:
  - default_generation: ar
  - slice_5_gate: blocked_missing_same_cluster_ar_vs_mtp_evidence
  - speedup_claim_allowed: False
- Bottleneck analysis:
  - recommendation_status: bookkeeping_sync_verified_no_live_bottleneck_claim
  - bottlenecks: missing_same_cluster_ar_vs_mtp_rows
  - next_step: Use the synced runlog tracking section for future canonical payload updates after live cluster rows are collected.
- Evidence:
  - scripts/mirror_mimo_mtp_rollout_seed.py
  - scripts/test_mirror_mimo_mtp_rollout_seed.py
  - uv run pytest scripts/test_mirror_mimo_mtp_rollout_seed.py -q
  - uv run ruff check scripts/mirror_mimo_mtp_rollout_seed.py scripts/test_mirror_mimo_mtp_rollout_seed.py
  - uv run ruff format --check scripts/mirror_mimo_mtp_rollout_seed.py scripts/test_mirror_mimo_mtp_rollout_seed.py
  - uv run basedpyright scripts/mirror_mimo_mtp_rollout_seed.py scripts/test_mirror_mimo_mtp_rollout_seed.py
  - git diff --check -- scripts/mirror_mimo_mtp_rollout_seed.py scripts/test_mirror_mimo_mtp_rollout_seed.py
- Remaining risk: This Sub-AC validates runlog artifact synchronization only; Slice 5 remains blocked until same-cluster AR-vs-MTP rows prove a real guarded MTP speed win.

## 2026-06-07 AC-P10 Final Verification Refresh

A final closeout refresh was run in
`/Users/studio2/exo/.worktrees/mimo-v25-pro-mtp-fastpath-single-stream-20260601`
on branch `feature/mimo-v25-pro-mtp-fastpath-single-stream-20260601` at HEAD
`6911a9f545f07276516e29b16872622e3357a68a`.

Fresh evidence prefix: `.goose-ultrawork/evidence/ac-p10-final-20260607T080246Z-*`.

Verification evidence:

- Focused pytest across touched rollout surfaces: `317 passed in 5.18s`.
- Ruff check over touched code/test surfaces: `All checks passed!`.
- Ruff format check over touched code/test surfaces: `49 files already formatted`.
- Basedpyright over touched code/test surfaces: `0 errors, 0 warnings, 0 notes`.
- `git diff --check`: no output / exit 0.
- Matrix command generation emitted the canonical 8-row AR + guarded MTP D1/D2/D3 matrix for max_tokens 16 and 64.
- Budget fixture and bottleneck classifier fixture completed with exit 0.

Live row blocker:

- The local cluster probe against `http://127.0.0.1:52415` failed with
  `<urlopen error [Errno 61] Connection refused>` and emitted
  `status=blocked_with_command` plus an exact rerun command in
  `.goose-ultrawork/evidence/ac-p10-final-20260607T080246Z-cluster-probe.log`.

AC status and gate:

- AC-P0 through AC-P10 are evidenced for the guarded rollout slice.
- No live same-cluster AR-vs-guarded-MTP rows exist in this worktree.
- No >=30 tok/s, >=40 tok/s, MTP speedup, or Slice 5 production/default
  eligibility claim is made.
- Default generation remains AR; MiMo MTP remains explicit/experimental only.
- Next best step remains running the saved matrix commands against a healthy exo
  tensor-parallel cluster API with the official MiMo MTP sidecar, then analyzing
  those same-cluster rows with the budget and bottleneck scripts before any
  Slice 5 review.
