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
