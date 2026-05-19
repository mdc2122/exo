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
  - `/Users/studio2/Documents/Codex/2026-05-19/superpowers-can-we-determine-whether-22/mimo_v25_pro_sustain_probe_20260519.json`
  - `/Users/studio2/Documents/Codex/2026-05-19/superpowers-can-we-determine-whether-22/mimo_v25_pro_concurrency_probe_20260519.json`
  - `/Users/studio2/Documents/Codex/2026-05-19/superpowers-can-we-determine-whether-22/mimo_v25_pro_concurrency8_probe_20260519.json`

## Dirty Work Preserved

- Patch artifact: `docs/plans/artifacts/mimo-v25-pro-kv-cache-wip-20260519.patch`
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

To return to the known-good baseline code:

```bash
git switch backup/mimo-v25-pro-6bit-working-baseline-20260519
```

To reapply the preserved KV/cache work from the current branch:

```bash
git apply docs/plans/artifacts/mimo-v25-pro-kv-cache-wip-20260519.patch
```

## Safety Rule

Do not start a full MiMo V2.5 Pro model if any MiMo instance is active, loading,
warming, stale-but-resident, or if either Studio node has not recovered memory.
Duplicate full MiMo startup can crash the Mac Studios.
