# MiMo V2.5-Pro Custom MLX 6-bit Quantization Runlog

## Scope

Continue the dirty Ouroboros Goose worktree custom 6-bit MLX quantization work for MiMo V2.5-Pro Track A text/agentic/long-context only.

This run intentionally avoids Kimi deployment scripts, Kimi processes, Kimi-specific model paths, TurboQuant, media requests, and any source safetensors mutation.

## Branch and Worktree

- Worktree: `/Users/studio2/.ouroboros/worktrees/exo/orch_goose_42e7de630503`
- Branch: `ooo/orch_goose_42e7de630503`
- Starting HEAD observed during finalization: `b357af110bfeb9951ddc4b81f77277d30b668ac9`
- Code/config/docs commit created by this run: `93a8a01633b3d3e5f7acfca449ddaab918343124` (`CON-75 prepare MiMo Pro MLX 6bit quantization`).
- Runlog hash finalization commit: this document is finalized by the subsequent `CON-75 finalize MiMo Pro 6bit quantization runlog` commit.

## Source and Output Paths

- Preserved source checkpoint: `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/hf/XiaomiMiMo--MiMo-V2.5-Pro`
- Source index: `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/hf/XiaomiMiMo--MiMo-V2.5-Pro/model.safetensors.index.json`
- Approved output root: `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/MiMo-V2.5-Pro-6bit`
- Dry-run manifest: `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/MiMo-V2.5-Pro-6bit/quantization_manifest.json`
- Output index from dry-run: `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/MiMo-V2.5-Pro-6bit/model.safetensors.index.json`
- Logs directory: `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/logs`
- Dry-run log: `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/logs/mimo-v25-pro-6bit-dry-run-20260505T183544Z.log`

## Source Snapshot

Static source inspection found:

```json
{
  "source_metadata": {
    "save_format": "fp8",
    "total_size": 1033369538304,
    "tp_size": 8
  },
  "source_shard_count": 34,
  "source_weight_count": 159581
}
```

The source checkpoint was only read for metadata/dry-run planning and was not rewritten or deleted.

## Quantization Harness

Added `scripts/mimo_v25_pro_6bit_quantize.py`:

- Guards source path to exactly `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/hf/XiaomiMiMo--MiMo-V2.5-Pro`.
- Guards output path to `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/MiMo-V2.5-Pro-6bit` or a child.
- Uses MLX native affine quantization with `bits=6`, `group_size=64`, `mode=affine`.
- Does not use TurboQuant.
- Converts shard-by-shard with manifest status tracking.
- Writes hidden temp shard files and atomically replaces complete output shards.
- Supports resumability by preserving already complete shards during a subsequent dry-run/conversion.
- Writes:
  - `quantization_manifest.json`
  - rewritten `config.json` with `quantization_config.quant_method = mlx-affine`
  - `model.safetensors.index.json`
  - support tokenizer/code files
  - README identity note
- Provides `--cleanup-incomplete` to remove temp shards and mark incomplete manifest entries.

## Focused Tests and Validation

### Focused pytest / ruff / basedpyright

Command:

```bash
uv run pytest \
  scripts/test_mimo_v25_pro_6bit_quantize.py \
  src/exo/shared/tests/test_mimo_v25_pro_6bit_model_card.py \
  src/exo/shared/tests/test_mimo_v25_pro_model_card.py \
  -q && \
uv run ruff check \
  scripts/mimo_v25_pro_6bit_quantize.py \
  scripts/test_mimo_v25_pro_6bit_quantize.py \
  src/exo/shared/models/model_cards.py \
  src/exo/shared/tests/test_mimo_v25_pro_6bit_model_card.py \
  src/exo/shared/tests/test_mimo_v25_pro_model_card.py && \
uv run basedpyright \
  src/exo/shared/models/model_cards.py \
  src/exo/shared/tests/test_mimo_v25_pro_6bit_model_card.py \
  src/exo/shared/tests/test_mimo_v25_pro_model_card.py
```

Result:

```text
31 passed in 1.18s
All checks passed!
0 errors, 0 warnings, 0 notes
```

The tests cover:

- source/output path guards
- MLX affine 6-bit dry-run sizing
- FP8 paired-scale planning logic
- config metadata rewrite from official FP8 metadata to custom MLX affine metadata
- manifest round-trip and output index generation
- resumability preservation of completed shard manifest entries
- incomplete temp cleanup
- tiny local pilot conversion dtype/metadata/manifest cleanup
- separate custom model-card identity
- official FP8 card remains separate

### Tiny Safe Pilot Conversion

A synthetic local fixture was generated under `/tmp/mimo_6bit_pilot/source` using `uv run python`. The production full-model source was not loaded.

Pilot conversion was run through the harness module with test-only path constants pointing at `/tmp/mimo_6bit_pilot/source` and `/tmp/mimo_6bit_pilot/output`.

Result summary:

```json
{
  "disk_manifest_model_id": "XiaomiMiMo/MiMo-V2.5-Pro-6bit-MLX",
  "dtypes": {
    "layer.bias": "mlx.core.float16",
    "layer.weight": "mlx.core.uint32",
    "layer.weight.biases": "mlx.core.float16",
    "layer.weight.scales": "mlx.core.float16"
  },
  "expected_output_size_bytes": 116,
  "index_save_format": "mlx-affine-6bit",
  "index_total_size": 116,
  "manifest": "/tmp/mimo_6bit_pilot/output/quantization_manifest.json",
  "output": "/tmp/mimo_6bit_pilot/output",
  "shard_status": "complete",
  "tmp_files": []
}
```

## Official Dry-run Sizing

Command:

```bash
mkdir -p /Volumes/GLM5-NVMe/exo/mimo-v25-pro/logs
LOG=/Volumes/GLM5-NVMe/exo/mimo-v25-pro/logs/mimo-v25-pro-6bit-dry-run-$(date -u +%Y%m%dT%H%M%SZ).log
uv run python scripts/mimo_v25_pro_6bit_quantize.py --dry-run 2>&1 | tee "$LOG"
```

Result:

```json
{
  "manifest": "/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/MiMo-V2.5-Pro-6bit/quantization_manifest.json",
  "expected_output_size_bytes": 895286156544
}
```

Disk/sizing summary after dry-run:

```json
{
  "expected_output_gib": 833.8,
  "expected_output_size_bytes": 895286156544,
  "free_bytes_after_dry_run": 1285925904384,
  "free_gib_after_dry_run": 1197.612,
  "headroom_bytes": 268435456000,
  "required_with_headroom_bytes": 1163721612544,
  "required_with_headroom_gib": 1083.8,
  "shard_count": 34,
  "status_counts": {
    "dry-run": 34
  }
}
```

Dry-run wrote only support/config/manifest/index files under the approved output root. No output `.safetensors` model shard exists yet from the full conversion.

## Full Resumable Conversion Command

Safety prerequisites now satisfied before launch:

- focused tests pass
- tiny pilot conversion pass
- official dry-run manifest present
- source checkpoint path guard present
- output path guard present
- disk headroom currently sufficient for estimate plus 250 GiB headroom
- resumability manifest handling validated

Prepared full conversion command:

```bash
cd /Users/studio2/.ouroboros/worktrees/exo/orch_goose_42e7de630503
mkdir -p /Volumes/GLM5-NVMe/exo/mimo-v25-pro/logs
LOG=/Volumes/GLM5-NVMe/exo/mimo-v25-pro/logs/mimo-v25-pro-6bit-convert-$(date -u +%Y%m%dT%H%M%SZ).log
nohup uv run python scripts/mimo_v25_pro_6bit_quantize.py --convert >"$LOG" 2>&1 &
echo "PID=$! LOG=$LOG"
```

Monitor command:

```bash
tail -f /Volumes/GLM5-NVMe/exo/mimo-v25-pro/logs/mimo-v25-pro-6bit-convert-*.log
```

Progress/manifest command:

```bash
uv run python - <<'PY'
import json
from pathlib import Path
manifest = json.loads(Path('/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/MiMo-V2.5-Pro-6bit/quantization_manifest.json').read_text())
counts = {}
for shard in manifest.get('shards', {}).values():
    counts[shard.get('status', 'missing')] = counts.get(shard.get('status', 'missing'), 0) + 1
print(json.dumps(counts, indent=2, sort_keys=True))
PY
```

Abort command:

```bash
pkill -f 'scripts/mimo_v25_pro_6bit_quantize.py --convert'
```

Cleanup incomplete command:

```bash
cd /Users/studio2/.ouroboros/worktrees/exo/orch_goose_42e7de630503
uv run python scripts/mimo_v25_pro_6bit_quantize.py --cleanup-incomplete
```

The full conversion was **not launched by this run** despite satisfying the safety prerequisites, because the remaining work was to finalize/commit the harness/card/runlog first and avoid leaving a multi-hundred-GB mutation running while code/docs were still uncommitted. The command above is operator-ready.

## Model Identity / Card Integration

Added separate built-in model card:

- `resources/inference_model_cards/XiaomiMiMo--MiMo-V2.5-Pro-6bit-MLX.toml`
- Model id: `XiaomiMiMo/MiMo-V2.5-Pro-6bit-MLX`
- Base model: `XiaomiMiMo/MiMo-V2.5-Pro`
- Quantization: `6bit-mlx-affine`
- Storage size: `895286156544` bytes, matching official dry-run estimate
- Tasks: `TextGeneration`
- Capabilities: `text`, `agentic`, `long-context`
- No `vision` config and no media capabilities

The official FP8 card remains present and separate:

- `resources/inference_model_cards/XiaomiMiMo--MiMo-V2.5-Pro.toml`
- Model id: `XiaomiMiMo/MiMo-V2.5-Pro`
- Quantization: `fp8`
- Storage size: `1033369538304` bytes

`src/exo/shared/models/model_cards.py` now applies MiMo V2.5-Pro text-only validation to both the official FP8 identity and the custom MLX 6-bit identity.

## Deployment Feasibility Validation

### Static model-card/API check

Command loaded both cards through `ModelCard.load(ModelId(...))`.

Result summary:

```json
{
  "custom_6bit": {
    "base_model": "XiaomiMiMo/MiMo-V2.5-Pro",
    "capabilities": ["text", "agentic", "long-context"],
    "model_id": "XiaomiMiMo/MiMo-V2.5-Pro-6bit-MLX",
    "quantization": "6bit-mlx-affine",
    "storage_size": 895286156544,
    "tasks": ["TextGeneration"],
    "vision": false
  },
  "official": {
    "capabilities": ["text", "agentic", "long-context"],
    "model_id": "XiaomiMiMo/MiMo-V2.5-Pro",
    "quantization": "fp8",
    "storage_size": 1033369538304,
    "tasks": ["TextGeneration"],
    "vision": false
  }
}
```

### Non-persistent local API smoke

First attempt with `HOME=/tmp/...` was blocked because `uv` tried to rebuild dependencies in a fresh cache and failed during MLX build. Recovery used the repo uv environment and isolated only exo runtime state with `EXO_HOME`.

Dashboard build was required because `dashboard/build` did not exist in this worktree:

```bash
cd dashboard && npm install && npm run build
```

Result: build succeeded; npm reported existing dependency audit warnings and Svelte accessibility/state warnings unrelated to this MiMo quantization change.

Safe API smoke command shape:

```bash
EXO_HOME=.exo-mimo-6bit-api-smoke-<pid> \
EXO_OFFLINE=true \
EXO_DASHBOARD_DIR="$PWD/dashboard/build" \
uv run exo --offline --no-worker --no-downloads --api-port 52575 --libp2p-port 0
```

Queried:

- `GET /models`
- `GET /v1/models`
- `GET /instance/previews?model_id=XiaomiMiMo%2FMiMo-V2.5-Pro-6bit-MLX`

Result summary:

```json
{
  "custom_6bit": {
    "base_model": "XiaomiMiMo/MiMo-V2.5-Pro",
    "capabilities": ["text", "agentic", "long-context"],
    "id": "XiaomiMiMo/MiMo-V2.5-Pro-6bit-MLX",
    "quantization": "6bit-mlx-affine",
    "storage_size_megabytes": 853811,
    "tasks": ["TextGeneration"]
  },
  "models_count": 88,
  "official": {
    "capabilities": ["text", "agentic", "long-context"],
    "id": "XiaomiMiMo/MiMo-V2.5-Pro",
    "quantization": "fp8",
    "tasks": ["TextGeneration"]
  },
  "previews": {
    "previews": []
  },
  "v1_models_count": 88
}
```

No worker was started, no downloads were enabled, no placement was created, no MiMo weights were loaded, and no text generation was attempted.

Tiny text-only generation was **not run** because there is no converted full 6-bit artifact yet and no valid placement/load for the custom 6-bit model in this isolated smoke.

## Generated Artifacts and Tracking

Generated artifacts intentionally left outside git:

- `/tmp/mimo_6bit_pilot/source`
- `/tmp/mimo_6bit_pilot/output`
- `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/MiMo-V2.5-Pro-6bit/*`
- `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/logs/*`
- `dashboard/build/`
- `dashboard/node_modules/`
- `dashboard/.svelte-kit/`

Only code/config/docs/test changes should be committed.

## Current Status

- Quantization harness: ready for operator-run full conversion.
- Validation status: focused static and pilot validation passed.
- Custom model identity/card: integrated and API-visible as separate 6-bit identity.
- Full conversion: not launched; dry-run manifest only for full source.
- Deployment: static/API list/previews validated; generation blocked pending full conversion and valid placement/load.

## Blockers / Risks

- Full conversion is expected to write about `895,286,156,544` bytes (~833.8 GiB) and may take substantial time.
- Conversion path dequantizes FP8 tensors shard-by-shard through CPU/torch to MLX; performance and peak memory behavior on real largest shards is not yet proven beyond metadata dry-run and tiny pilot.
- The custom 6-bit artifact may require MLX-LM/runtime compatibility validation after full conversion before serving.
- `/instance/previews` returned an empty preview list in isolated `--no-worker` smoke because no topology nodes were available; this is expected and avoids placement/load.
- No text generation has been run for the 6-bit custom identity.

## Exact Next Operator Action

If ready to consume the disk/time budget, run:

```bash
cd /Users/studio2/.ouroboros/worktrees/exo/orch_goose_42e7de630503
mkdir -p /Volumes/GLM5-NVMe/exo/mimo-v25-pro/logs
LOG=/Volumes/GLM5-NVMe/exo/mimo-v25-pro/logs/mimo-v25-pro-6bit-convert-$(date -u +%Y%m%dT%H%M%SZ).log
nohup uv run python scripts/mimo_v25_pro_6bit_quantize.py --convert >"$LOG" 2>&1 &
echo "PID=$! LOG=$LOG"
```

After completion:

1. Verify all manifest shard statuses are `complete`.
2. Verify `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/MiMo-V2.5-Pro-6bit/model.safetensors.index.json` maps expected quantized keys.
3. Configure exo model search/read-only path to include `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized`.
4. Start a real multi-node exo cluster with enough visible memory for ~895 GB model storage/RAM budget.
5. Check `/instance/previews?model_id=XiaomiMiMo%2FMiMo-V2.5-Pro-6bit-MLX` for valid placement.
6. Only after valid placement/load, run one tiny text-only generation request.
7. Do not send media requests for this Pro Track A model.
8. Do not use TurboQuant for this path.

## 2026-05-05 CON-75 real-shard FP8 scale broadcast repair

### Failure reproduced

A focused diagnostic loaded only `model_mtp.safetensors` and the paired MTP QKV tensors:

- `model.mtp.layers.0.self_attn.qkv_proj.weight`: shape `[27136, 6144]`, dtype `torch.float8_e4m3fn`
- `model.mtp.layers.0.self_attn.qkv_proj.weight_scale_inv`: shape `[216, 48]`, dtype `torch.float32`

The previous broadcast helper inferred block sizes with integer division:

- row block: `27136 // 216 = 125`
- column block: `6144 // 48 = 128`
- expanded scale shape: `[27000, 6144]`

Multiplying `[27136, 6144]` by `[27000, 6144]` reproduced the real failure:

```text
RuntimeError: The size of tensor a (27136) must match the size of tensor b (27000) at non-singleton dimension 0
```

Correct semantics for these FP8 source tensors are fixed 128x128 block scales. `scale_inv` may contain extra scale rows/columns for partial final blocks. For the MTP QKV tensor, `[216,48]` expands to `[27648,6144]` and must then be cropped to the target weight shape `[27136,6144]`.

### Harness fix

Updated `scripts/mimo_v25_pro_6bit_quantize.py`:

- Added explicit `FP8_BLOCK_SIZE = 128`.
- Validates target and scale tensors are positive rank-2 shapes.
- Computes required scale shape with ceil-div over 128x128 source blocks.
- Fails if the scale tensor does not cover the target.
- Expands each scale value over 128 rows/columns and crops to exactly the target tensor shape.

Regression tests added in `scripts/test_mimo_v25_pro_6bit_quantize.py`:

- Exact MTP QKV case: target `[27136,6144]`, scale `[216,48]`, verifying crop from the extra FP8 block rows.
- Normal divisible case: target `[256,256]`, scale `[2,2]`, verifying standard 128x128 block expansion.

### Focused verification commands and exact results

```bash
uv run pytest scripts/test_mimo_v25_pro_6bit_quantize.py -q
# 14 passed in 1.08s
# exit 0

uv run ruff check scripts/mimo_v25_pro_6bit_quantize.py scripts/test_mimo_v25_pro_6bit_quantize.py
# All checks passed!
# exit 0

uv run basedpyright scripts/mimo_v25_pro_6bit_quantize.py scripts/test_mimo_v25_pro_6bit_quantize.py
# 83 errors, 0 warnings, 0 notes
# exit 1
```

`basedpyright` failures are pre-existing strict typing issues in this standalone script/test path (JSON `Any`, safetensors stub gaps, total=False TypedDict required-key access, argparse `Any`, and private test helper usage). The FP8 repair did not introduce a new runtime regression in the focused pytest/ruff checks. An earlier attempted `ruff check ... docs/plans/MIMO_V25_PRO_CUSTOM_6BIT_QUANT_RUNLOG.md` was an invalid target because Ruff parsed Markdown as Python and produced syntax errors; Ruff was rerun on the changed Python files only and passed.

### Failed manifest cleanup / resume state

Before cleanup, the official output manifest at `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/MiMo-V2.5-Pro-6bit/quantization_manifest.json` had:

```text
{'failed': 1, 'dry-run': 33}
model_mtp.safetensors: failed RuntimeError('The size of tensor a (27136) must match the size of tensor b (27000) at non-singleton dimension 0')
tmp files: 0
```

Ran:

```bash
uv run python scripts/mimo_v25_pro_6bit_quantize.py --cleanup-incomplete
uv run python scripts/mimo_v25_pro_6bit_quantize.py --dry-run
```

After cleanup and dry-run regeneration:

```text
{'dry-run': 34}
model_mtp.safetensors status: dry-run
model_mtp.safetensors error: None
tmp_count: 0
expected_output_size_bytes: 895286156544
```

### Real-shard pilot

A real-shard pilot converted only `model_mtp.safetensors` into an approved child output:

```bash
OUT=/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/MiMo-V2.5-Pro-6bit/pilot-model-mtp-20260505T184640Z
LOG=/Volumes/GLM5-NVMe/exo/mimo-v25-pro/logs/mimo-v25-pro-6bit-model-mtp-pilot-20260505T184640Z.log
uv run python scripts/mimo_v25_pro_6bit_quantize.py \
  --output "$OUT" \
  --include-shard model_mtp.safetensors \
  --convert 2>&1 | tee "$LOG"
```

Pilot result:

```text
counts {'complete': 1}
model_mtp.safetensors status: complete
output_size: 1659958379
error: None
tmp_count: 0
sample source_scale_inv: model.mtp.layers.0.self_attn.qkv_proj.weight_scale_inv
```

Disk check after the pilot showed about `1.2Ti` available on `/Volumes/GLM5-NVMe`.

### Current full conversion status

Because the real-shard pilot passed and disk remained sufficient, full conversion was relaunched in the background:

```bash
LOG=/Volumes/GLM5-NVMe/exo/mimo-v25-pro/logs/mimo-v25-pro-6bit-convert-resume-20260505T184658Z.log
PIDFILE=/Volumes/GLM5-NVMe/exo/mimo-v25-pro/logs/mimo-v25-pro-6bit-convert-resume-20260505T184658Z.pid
nohup uv run python scripts/mimo_v25_pro_6bit_quantize.py --convert >"$LOG" 2>&1 &
echo 54533 > "$PIDFILE"
```

Monitoring commands:

```bash
ps -p 54533 -o pid=,stat=,etime=,command=
tail -f /Volumes/GLM5-NVMe/exo/mimo-v25-pro/logs/mimo-v25-pro-6bit-convert-resume-20260505T184658Z.log
uv run python - <<'PY'
import json
from pathlib import Path
manifest = json.loads(Path('/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/MiMo-V2.5-Pro-6bit/quantization_manifest.json').read_text())
counts = {}
for shard in manifest.get('shards', {}).values():
    counts[shard.get('status')] = counts.get(shard.get('status'), 0) + 1
print(counts)
PY
```

Observed immediately after launch:

```text
PID 54533 running: uv run python scripts/mimo_v25_pro_6bit_quantize.py --convert
status_counts {'complete': 1, 'running': 1, 'dry-run': 32}
model_mtp.safetensors: complete
model_pp0_ep0_shard0.safetensors: running
```

The official source checkpoint under `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/hf/XiaomiMiMo--MiMo-V2.5-Pro` was not modified.
