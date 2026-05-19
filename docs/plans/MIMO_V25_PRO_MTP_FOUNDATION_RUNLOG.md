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
