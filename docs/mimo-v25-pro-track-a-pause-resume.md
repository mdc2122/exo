# CON-75 — MiMo V2.5-Pro Track A pause/resume handoff

Paused: 2026-05-06T17:30:54Z

## Status

Track A for `XiaomiMiMo/MiMo-V2.5-Pro-6bit-MLX` is intentionally paused due to memory pressure during Studio1 rank-1 shard loading. Do not restart live Track A placement/generation until a bounded memory-safe resume plan is selected.

## What is preserved in git

Branch: `con-75-mimo-pro-live-integration-20260504-200136`

Key commits:

- `b578fb72` — `CON-75: add MiMo Pro loader fixes and rank1 probe seed`
- `f59b9bb9` — `CON-75: add MiMo rank1 load probe artifact`
- `425c334b` — `CON-75: add MiMo rank1 bisect probe seed`
- `9c079b10` — `CON-75: add MiMo rank1 bisect probe artifact`

Tracked resume artifacts:

- `src/exo/worker/engines/mlx/utils_mlx.py`
  - MiMo `mimo_v2 -> mimo_v2_flash` remap preserved.
  - Custom 6-bit sanitize hook splits fused `qkv_proj` and normalizes `*.weight.scales` / `*.weight.biases` names.
  - Verbose model object logging reduced.
- `src/exo/worker/engines/mlx/tests/test_mimo_mlx_runtime.py`
  - Regression for fused-qkv split and quantized key normalization.
- `scripts/mimo_track_a_rank1_load_probe.py`
  - Studio1-only standalone rank-1 load probe; no JACCL/libp2p/API/generation.
- `scripts/test_mimo_track_a_rank1_load_probe.py`
- `scripts/mimo_track_a_rank1_bisect_probe.py`
  - Safe dry-run default; heavy probe requires `--execute`.
  - Candidate ranges include `[36,60)`, `[36,62)`, `[36,65)`, `[36,67)`, `[36,70)`.
  - Interprets shell-style `137` as `SIGKILL`.
- `scripts/test_mimo_track_a_rank1_bisect_probe.py`
- `mimo-track-a-rank1-load-probe.seed.yaml`
- `mimo-track-a-rank1-bisect-probe.seed.yaml`

## Verification already run

- MiMo/topology targeted suite: `11 passed`
- Rank-1 load probe tests: local `4 passed`, Studio1 candidate `4 passed`
- Rank-1 bisect probe tests: local `5 passed`, Studio1 candidate `5 passed`
- `ruff check` on new probe scripts: passed locally and remotely
- `py_compile` on new probe scripts: passed locally and remotely

## Live runtime state at pause

Track A runtime/probe processes were stopped before this handoff.

Expected inactive ports:

- Studio2 API: `52415`
- Studio2 libp2p: `9010`
- Studio1 libp2p: `9011`

Re-check before resuming:

```bash
lsof -nP -iTCP:52415 -sTCP:LISTEN || true
lsof -nP -iTCP:9010 -sTCP:LISTEN || true
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null studio1@Studio1s-Mac-Studio.local 'lsof -nP -iTCP:9011 -sTCP:LISTEN || true'
```

## Blocker evidence

Main blocker: Studio1 rank-1 shard `[36,70)` hard-kills during load/eval from memory pressure.

Evidence roots:

- `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/investigations/signal9-20260506T162535Z/summary.md`
- `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/investigations/rank1-load-probe/20260506T170118Z/remote-copy/probe.jsonl`

Key observed values from standalone rank-1 probe:

- Exit code: `137` (`SIGKILL`)
- Last recorded global layer: `66`
- Rank-1 shard interval: `[36,70)`
- MLX active: about `320.50 GiB`
- MLX peak: about `330.62 GiB`
- MLX cache: about `10.13 GiB`
- Process RSS: about `14.59 GiB`
- Free pages at finish: about `44,597` pages (`~0.68 GiB`)
- Compressor occupied: about `10,063,094` pages (`~153.55 GiB`)
- Swap usage remained low, so pressure is unified-memory/compressor/file-cache dominated rather than normal swap exhaustion.

## GLM 5.1 comparison notes

GLM 5.1 did not hit the same file-cache/compressed-memory cliff as severely despite comparable/larger on-disk size because its implementation and artifact layout are materially friendlier to the current exo/MLX path:

- GLM uses a dedicated `GlmMoeDsaShardingStrategy`; MiMo currently relies on generic loading plus a compatibility sanitize hook.
- GLM artifact layout is layer-local: `152` safetensors shards totaling about `623.62 GiB`, commonly ~`3–5 GiB` each, with each shard touching one or two adjacent layers.
- MiMo artifact layout is expert-parallel/global: about `34` indexed safetensors files under the active symlink, with many ~`25.47 GiB` files; each expert shard spans almost every layer (`1..69`). A single rank/layer slice therefore touches many huge files.
- MiMo has far denser key fanout: about `3,420` weight-map keys/layer and ~`32.5` files/layer vs GLM about `52.6` keys/layer and ~`2.9` files/layer.
- MiMo custom 6-bit weights require fused `qkv_proj` splitting and quantized key-name normalization before loading. That is now patched, but it adds a less mature loader path than GLM's native route.
- MiMo architecture is heavier in the relevant dimensions: `384` routed experts, `128` attention heads, `head_dim=192`, no shared expert; GLM has `256` routed experts, `64` heads, and a dedicated DSA strategy with mixed quantization overrides.

Working hypothesis: the fatal memory pressure is not simply total model size. It is the combination of MiMo's global expert-shard file layout, high per-layer key/file fanout, generic/sanitize load path, and the current rank split `[36,70)` causing many large mapped files plus MLX/Metal materialization/compressor pressure to accumulate.

## Safe resume plan

1. Keep Track A stopped unless explicitly resuming.
2. Do not immediately restart 2-node live exo placement.
3. First run only the dry-run bisect manifest:

```bash
cd /Users/studio2/exo
uv run python scripts/mimo_track_a_rank1_bisect_probe.py \
  --dry-run \
  --model-path /Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/XiaomiMiMo--MiMo-V2.5-Pro-6bit-MLX
```

4. If the user accepts risk, execute bounded Studio1 ranges from smallest to largest, not `[36,70)` first:
   - `[36,60)`
   - `[36,62)`
   - `[36,65)`
   - `[36,67)`
   - `[36,70)` only if prior ranges pass with headroom
5. After each range, capture `vm_stat`, `sysctl vm.swapusage kern.memorystatus_level`, process state, and probe JSONL.
6. If reduced ranges pass, adjust placement/rank split or load strategy before live retry.
7. If reduced ranges still SIGKILL, treat as broader MLX/Metal allocator or artifact-layout issue; consider re-sharding/converting MiMo artifact into more layer-local files before another live Track A attempt.

## Do not touch during pause

- Do not restart Kimi as part of this handoff.
- Do not run MiMo live cluster/generation until explicitly requested.
- Do not delete the evidence bundles; they are resume-critical.
