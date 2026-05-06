# CON-75 — MiMo V2.5-Pro Track A pause/resume handoff

Paused: 2026-05-06T17:30:54Z

## Status

Track A for `XiaomiMiMo/MiMo-V2.5-Pro-6bit-MLX` was resumed after Studio2 reboot using the bounded Studio1 standalone rank-1 probe plan. The safe per-rank split is now narrowed: `[36,68)` completes, while `[36,69)` and `[36,70)` SIGKILL after evaluating global layer `67` and before recording layer `68`. Do not restart live Track A placement/generation with the old rank-1 `[36,70)` split; use a safer split such as rank-1 ending at `68` or implement a different placement/load strategy first.

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

Main blocker: Studio1 rank-1 shard `[36,69)` / `[36,70)` hard-kills during layer evaluation from memory pressure. After the Studio2 reboot and bounded resume probes, `[36,68)` completes; the failure threshold is the next layer interval.

Evidence roots:

- `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/investigations/signal9-20260506T162535Z/summary.md`
- `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/investigations/rank1-load-probe/20260506T170118Z/remote-copy/probe.jsonl`
- `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/investigations/rank1-load-probe/mimo-track-a-rank1-bisect-manifest-20260506T223409Z.jsonl` — `[36,60)` passed
- `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/investigations/rank1-load-probe/mimo-track-a-rank1-bisect-manifest-20260506T223521Z.jsonl` — `[36,62)` passed
- `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/investigations/rank1-load-probe/mimo-track-a-rank1-bisect-manifest-20260506T223613Z.jsonl` — `[36,65)` passed
- `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/investigations/rank1-load-probe/mimo-track-a-rank1-bisect-manifest-20260506T223730Z.jsonl` — `[36,67)` passed
- `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/investigations/rank1-load-probe/mimo-track-a-rank1-bisect-manifest-20260506T224047Z.jsonl` — `[36,68)` passed
- `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/investigations/rank1-load-probe/mimo-track-a-rank1-bisect-manifest-20260506T224217Z.jsonl` — `[36,69)` SIGKILL
- `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/investigations/rank1-load-probe/mimo-track-a-rank1-bisect-manifest-20260506T223851Z.jsonl` — `[36,70)` SIGKILL

Key observed values from the post-reboot bounded Studio1 probes:

- `[36,60)`: exit `0`, active `248.13 GiB`, peak `258.25 GiB`, cache `10.13 GiB`.
- `[36,62)`: exit `0`, active `268.81 GiB`, peak `278.93 GiB`, cache `10.13 GiB`.
- `[36,65)`: exit `0`, active `299.82 GiB`, peak `309.95 GiB`, cache `10.13 GiB`.
- `[36,67)`: exit `0`, active `320.50 GiB`, peak `330.62 GiB`, cache `10.13 GiB`.
- `[36,68)`: exit `0`, active `330.84 GiB`, peak `340.96 GiB`, cache `10.13 GiB`.
- `[36,69)`: exit `-9` (`SIGKILL`) after recording `after-layer-eval` for global layer `67`; last active `330.84 GiB`, peak `340.96 GiB`, cache `10.13 GiB`.
- `[36,70)`: exit `-9` (`SIGKILL`) after recording `after-layer-eval` for global layer `67`; last active `330.84 GiB`, peak `340.96 GiB`, cache `10.13 GiB`.
- Post-probe Studio1 state recovered to `kern.memorystatus_level: 98`, swap used about `44.94 MiB`, and no rank-1 probe processes remained.

Earlier key observed values from standalone rank-1 probe:

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

1. Keep old Track A `[36,70)` live placement stopped.
2. Do not restart 2-node live exo placement with the old rank split.
3. Verified post-reboot bounded standalone probes on Studio1:
   - `[36,60)`, `[36,62)`, `[36,65)`, `[36,67)`, `[36,68)` passed.
   - `[36,69)` and `[36,70)` SIGKILL after global layer `67`.
4. Next implementation step: adjust placement/rank split so Studio1 rank-1 ends at `68` or below, then run live Track A only after verifying the new split does not assign layer `68+` to the same rank-1 shard.
5. If full model quality requires layers `68..69` on Studio1, implement a different load strategy first (for example more granular/pipeline split, layer-local re-sharding/conversion, or allocator/cache cleanup between layer materializations) before retrying live.
6. After any live retry, capture `vm_stat`, `sysctl vm.swapusage kern.memorystatus_level`, process state, probe/placement logs, API health, and generation evidence.
7. If reduced live split still SIGKILLs, treat as broader MLX/Metal allocator or artifact-layout issue; consider re-sharding/converting MiMo artifact into more layer-local files before another live Track A attempt.

## Do not touch unless explicitly part of Track A resume

- Do not restart Kimi as part of this handoff.
- Do not run MiMo live cluster/generation with the old `[36,70)` rank-1 split.
- Do not delete the evidence bundles; they are resume-critical.
