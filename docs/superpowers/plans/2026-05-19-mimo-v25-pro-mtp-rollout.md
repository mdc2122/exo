# MiMo V2.5 Pro MTP Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement, validate, and roll out a gated MTP runtime path for `kernelpool/MiMo-V2.5-Pro-6bit` that raises single-request effective output throughput on the Studio1/Studio2 TB5 tensor cluster while preserving the known-good non-MTP fallback.

**Architecture:** Add an MTP sidecar runtime beside the existing MLX MiMo model. The sidecar loads the experimental quantized MTP artifact, proposes future tokens from MiMo hidden states, verifies proposals with the main model, emits the longest valid accepted prefix, and falls back to normal one-token decode on rejection. The path starts opt-in behind fail-closed environment config, proves itself with tiny fixtures and offline probes, then advances through single-node and Studio1/Studio2 tensor/JACCL gates before it can become the default single-stream MiMo decode path. Batch generation remains on the existing `MlxBatchGenerator` path during this rollout.

**Tech Stack:** Python 3.13, MLX, MLX-LM `mimo_v2_flash`, exo MLX generator, tensor/JACCL over TB5, pytest, basedpyright, ruff, `bench/exo_bench.py`, safetensors, git.

---

## Scope And Gates

This is one end-to-end rollout plan. It includes module loading, decode, exo integration, single-node validation, Studio1/Studio2 cluster validation, and default rollout. Execution must stop at the first failed hard gate.

Hard gates:

1. Offline artifact evidence passes: the experimental MTP artifact exists, is complete, and the existing MTP-only module probe loads it without touching the production 6-bit artifact.
2. Tiny MTP controller tests pass: propose, verify, accept, reject, fallback, streaming order, stop-token handling, and telemetry are deterministic on a synthetic model.
3. Standalone MLX MTP stack tests pass: the sidecar loads quantized MTP tensors and produces expected hidden and logits shapes without loading the full main model.
4. Exo integration tests pass: MTP stays disabled for batch, vision, unsupported models, missing artifact, logprobs, and prefix-cache paths until explicit support is proven.
5. Runtime guard passes before any full MiMo startup: no active, loading, warming, or resident MiMo process and node memory above the configured floor.
6. Single-node live run passes bounded correctness, crash-free repeatability, fallback switch, and no duplicate full-model start.
7. Studio1/Studio2 TB5 tensor run passes bounded correctness, crash-free repeatability, and throughput evidence.
8. Promotion gate passes: repeated TB5 tensor runs beat the 22 tok/s single-request baseline and the non-MTP fallback remains available through a runtime kill switch.

Rollout success means MTP becomes the default single-stream decode path for MiMo V2.5 Pro 6-bit only after Gate 8. The batch generator remains the aggregate-throughput path unless a later verified batch-MTP path is added.

## Safety Rules

- Never start a second full MiMo V2.5 Pro instance while one is active, loading, warming, crashed but resident, or still consuming memory.
- Stop the current exo cluster before full-model single-node or TB5 tests.
- Run `scripts/mimo_v25_pro_runtime_guard.py` before every full-model startup.
- Treat any non-empty guard `reasons` list as a hard stop.
- Keep the production artifact at `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit` read-only.
- Load the sidecar artifact from `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental`.
- Keep `EXO_MIMO_MTP=0` as a kill switch after rollout.
- Do not promote MTP to default on theoretical speed alone. Promotion requires measured correctness and throughput evidence.
- Warmup must run with MTP disabled unless a specific warmup-MTP test is added and passes. Runtime warmup exists to establish runner health, not to benchmark MTP.

## Current Evidence

- Existing foundation plan: `docs/superpowers/plans/2026-05-19-mimo-v25-pro-mtp-foundation.md`
- Existing artifact plan: `docs/superpowers/plans/2026-05-19-mimo-v25-pro-mtp-artifact.md`
- Design spec: `docs/superpowers/specs/2026-05-19-mimo-v25-pro-mtp-design.md`
- Artifact runlog: `docs/plans/MIMO_V25_PRO_MTP_ARTIFACT_RUNLOG.md`
- Artifact summary: `docs/plans/artifacts/mimo-v25-pro-mtp-artifact-summary-20260519.json`
- Module probe: `docs/plans/artifacts/mimo-v25-pro-mtp-module-probe-20260519.json`
- MTP sidecar model id: `kernelpool/MiMo-V2.5-Pro-6bit-mtp-experimental`
- MTP sidecar shard: `model_mtp-00001-of-00001.safetensors`
- MTP sidecar shard SHA256: `2bd374402cf14a6c70f9f730677be75c566c42890240d95f21bfb89c3dd18b81`
- MTP sidecar tensor count: `72`
- MTP source layers: `[0, 1, 2]`
- Baseline single-stream speed: about `22 tok/s`
- Target stretch speed: `40 tok/s`, subject to measured acceptance rate and verification overhead

## Files To Create

- `src/exo/worker/engines/mlx/mimo_mtp/__init__.py`
- `src/exo/worker/engines/mlx/mimo_mtp/config.py`
- `src/exo/worker/engines/mlx/mimo_mtp/artifact.py`
- `src/exo/worker/engines/mlx/mimo_mtp/modules.py`
- `src/exo/worker/engines/mlx/mimo_mtp/controller.py`
- `src/exo/worker/engines/mlx/mimo_mtp/generation.py`
- `src/exo/worker/engines/mlx/mimo_mtp/sharding.py`
- `src/exo/worker/engines/mlx/tests/test_mimo_mtp_config.py`
- `src/exo/worker/engines/mlx/tests/test_mimo_mtp_artifact.py`
- `src/exo/worker/engines/mlx/tests/test_mimo_mtp_modules.py`
- `src/exo/worker/engines/mlx/tests/test_mimo_mtp_controller.py`
- `src/exo/worker/engines/mlx/tests/test_mimo_mtp_generation.py`
- `src/exo/worker/engines/mlx/tests/test_mimo_mtp_integration.py`
- `src/exo/worker/engines/mlx/tests/test_mimo_mtp_sharding.py`
- `scripts/mimo_v25_pro_mtp_rollout_guard.py`
- `scripts/test_mimo_v25_pro_mtp_rollout_guard.py`
- `scripts/mimo_v25_pro_mtp_bench_report.py`
- `scripts/test_mimo_v25_pro_mtp_bench_report.py`
- `docs/plans/MIMO_V25_PRO_MTP_ROLLOUT_RUNLOG.md`
- `resources/inference_model_cards/kernelpool--MiMo-V2.5-Pro-6bit-mtp.toml`
- `src/exo/shared/tests/test_mimo_v25_pro_mtp_model_card.py`

## Files To Modify

- `src/exo/worker/engines/mlx/generator/generate.py`
- `src/exo/worker/engines/mlx/generator/batch_generate.py`
- `src/exo/worker/engines/mlx/utils_mlx.py`
- `src/exo/worker/engines/mlx/auto_parallel.py`
- `src/exo/shared/models/model_cards.py`
- `src/exo/shared/tests/test_mimo_v25_pro_model_card.py`
- `src/exo/shared/tests/test_mimo_v25_pro_6bit_model_card.py`

## Files To Read During Integration

- `src/exo/worker/runner/llm_inference/batch_generator.py`
- `src/exo/worker/engines/mlx/cache.py`
- `src/exo/master/placement.py`
- `src/exo/api/types/api.py`

The pre-existing dirty files named below must be handled carefully. Read them before editing and preserve user work:

- `src/exo/worker/engines/mlx/cache.py`
- `src/exo/worker/engines/mlx/generator/batch_generate.py`
- `src/exo/worker/engines/mlx/generator/generate.py`
- `src/exo/worker/tests/unittests/test_mlx/test_kv_prefix_cache.py`

---

### Task 1: Re-verify Foundation And Artifact Gates

**Files:**
- Read: `docs/plans/MIMO_V25_PRO_MTP_FOUNDATION_RUNLOG.md`
- Read: `docs/plans/MIMO_V25_PRO_MTP_ARTIFACT_RUNLOG.md`
- Read: `docs/plans/artifacts/mimo-v25-pro-mtp-artifact-summary-20260519.json`
- Read: `docs/plans/artifacts/mimo-v25-pro-mtp-module-probe-20260519.json`

- [ ] **Step 1: Confirm git recovery and dirty state**

Run:

```bash
git status --short --branch
git rev-parse backup/mimo-v25-pro-6bit-working-baseline-20260519
git diff --name-only -- src/exo/worker/engines/mlx/cache.py src/exo/worker/engines/mlx/generator/batch_generate.py src/exo/worker/engines/mlx/generator/generate.py src/exo/worker/tests/unittests/test_mlx/test_kv_prefix_cache.py
```

Expected:

- Branch is `con-75-mimo-pro-live-integration-20260504-200136`.
- Backup branch resolves to `1114ff855fa8f5d897388b50ccc6011c205dbf39`.
- The dirty file list is understood before any edit touches those files.

- [ ] **Step 2: Re-run existing MTP artifact tests**

Run:

```bash
uv run pytest scripts/test_mimo_v25_pro_runtime_guard.py scripts/test_mimo_v25_pro_mtp_artifact_probe.py scripts/test_mimo_v25_pro_mtp_quantize.py scripts/test_mimo_v25_pro_mtp_module_probe.py -q
uv run ruff check scripts/mimo_v25_pro_runtime_guard.py scripts/mimo_v25_pro_mtp_artifact_probe.py scripts/mimo_v25_pro_mtp_quantize.py scripts/mimo_v25_pro_mtp_module_probe.py scripts/test_mimo_v25_pro_runtime_guard.py scripts/test_mimo_v25_pro_mtp_artifact_probe.py scripts/test_mimo_v25_pro_mtp_quantize.py scripts/test_mimo_v25_pro_mtp_module_probe.py
uv run basedpyright scripts/mimo_v25_pro_runtime_guard.py scripts/mimo_v25_pro_mtp_artifact_probe.py scripts/mimo_v25_pro_mtp_quantize.py scripts/mimo_v25_pro_mtp_module_probe.py scripts/test_mimo_v25_pro_runtime_guard.py scripts/test_mimo_v25_pro_mtp_artifact_probe.py scripts/test_mimo_v25_pro_mtp_quantize.py scripts/test_mimo_v25_pro_mtp_module_probe.py
```

Expected:

- `pytest` reports `20 passed` or higher when new tests already exist.
- `ruff` reports `All checks passed!`.
- `basedpyright` reports `0 errors, 0 warnings, 0 notes`.

- [ ] **Step 3: Assert artifact and probe JSON values**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("docs/plans/artifacts/mimo-v25-pro-mtp-artifact-summary-20260519.json").read_text())
probe = json.loads(Path("docs/plans/artifacts/mimo-v25-pro-mtp-module-probe-20260519.json").read_text())

assert summary["artifact_kind"] == "mimo-v25-pro-mtp-only"
assert summary["complete"] is True
assert summary["tensor_count"] == 72
assert summary["output_shard_sha256"] == "2bd374402cf14a6c70f9f730677be75c566c42890240d95f21bfb89c3dd18b81"
assert probe["artifact_kind"] == "mimo-v25-pro-mtp-only"
assert probe["loaded_tensor_count"] == 72
assert probe["synthetic_forwards"]["layer_0_eh_proj"] == [1, 1, 6144]
assert probe["synthetic_forwards"]["layer_0_qkv_proj"] == [1, 1, 27136]
print("mtp-foundation-ok")
PY
```

Expected:

```text
mtp-foundation-ok
```

- [ ] **Step 4: Commit no code in this task**

Run:

```bash
git diff --cached --name-only
```

Expected:

- No staged files.

Gate: stop unless every Step 1-3 command passes.

---

### Task 2: Add MTP Runtime Config And Telemetry Types

**Files:**
- Create: `src/exo/worker/engines/mlx/mimo_mtp/__init__.py`
- Create: `src/exo/worker/engines/mlx/mimo_mtp/config.py`
- Create: `src/exo/worker/engines/mlx/tests/test_mimo_mtp_config.py`

- [ ] **Step 1: Write failing config tests**

Create `src/exo/worker/engines/mlx/tests/test_mimo_mtp_config.py` with tests that require:

- Default config disables MTP.
- `EXO_MIMO_MTP=1` enables opt-in runtime.
- `EXO_MIMO_MTP=0` disables MTP even after default rollout.
- Artifact root defaults to the experimental MTP artifact path.
- `max_draft_tokens` defaults to `3` and rejects values outside `1..3`.
- Missing artifact root returns an ineligible reason rather than raising during request handling.
- Unsupported model ids are ineligible.
- Vision, batch, warmup, logprobs, and prefix-cache requests are ineligible for the first rollout.
- Telemetry initializes with zero proposed, accepted, emitted, fallback, and rejected tokens.

Run:

```bash
uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_config.py -q
```

Expected:

- Tests fail because `exo.worker.engines.mlx.mimo_mtp.config` does not exist.

- [ ] **Step 2: Implement config**

Create `src/exo/worker/engines/mlx/mimo_mtp/config.py` with:

- `MIMO_MTP_EXPERIMENTAL_ARTIFACT_ROOT`
- `MIMO_MTP_BASE_MODEL_ID`
- `MIMO_MTP_EXPERIMENTAL_MODEL_ID`
- `MIMO_MTP_PROMOTED_MODEL_ID`
- `MimoMtpRuntimeConfig`
- `MimoMtpEligibility`
- `MimoMtpTelemetry`
- `load_mimo_mtp_runtime_config(environ: Mapping[str, str] | None = None) -> MimoMtpRuntimeConfig`
- `is_mimo_mtp_eligible(config: MimoMtpRuntimeConfig, model_id: ModelId, artifact_index_exists: bool, request_features: MimoMtpRequestFeatures) -> MimoMtpEligibility`

Use `@dataclass(frozen=True)` for immutable config records. Parse these env vars:

- `EXO_MIMO_MTP`: `1`, `true`, `on`, `yes` enable; `0`, `false`, `off`, `no` disable.
- `EXO_MIMO_MTP_DISABLE`: any truthy value disables the path.
- `EXO_MIMO_MTP_ARTIFACT_ROOT`: sidecar artifact directory.
- `EXO_MIMO_MTP_MAX_DRAFT_TOKENS`: integer `1..3`.
- `EXO_MIMO_MTP_FAIL_CLOSED`: default `1`.
- `EXO_MIMO_MTP_DEFAULT_ON`: default `0` until Task 12.

Eligibility rules:

- Eligible model ids: `kernelpool/MiMo-V2.5-Pro-6bit`, `kernelpool/MiMo-V2.5-Pro-6bit-mtp`, and `kernelpool/MiMo-V2.5-Pro-6bit-mtp-experimental`.
- Ineligible when the feature is disabled.
- Ineligible when the artifact index file is missing.
- Ineligible for vision requests.
- Ineligible for batch generation in this rollout.
- Ineligible for warmup generation in this rollout.
- Ineligible for logprobs in this rollout.
- Ineligible when KV prefix-cache reuse is active in this rollout.
- Ineligible when `group` is present and MTP sharding has not been initialized.

- [ ] **Step 3: Run focused config verification**

Run:

```bash
uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_config.py -q
uv run ruff check src/exo/worker/engines/mlx/mimo_mtp src/exo/worker/engines/mlx/tests/test_mimo_mtp_config.py
uv run basedpyright src/exo/worker/engines/mlx/mimo_mtp/config.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_config.py
```

Expected:

- `pytest` passes all config tests.
- `ruff` reports `All checks passed!`.
- `basedpyright` reports `0 errors, 0 warnings, 0 notes`.

- [ ] **Step 4: Commit config**

Run:

```bash
git add src/exo/worker/engines/mlx/mimo_mtp/__init__.py src/exo/worker/engines/mlx/mimo_mtp/config.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_config.py
git commit -m "Gate MiMo MTP runtime config" \
  -m "Introduce fail-closed configuration and telemetry records before touching decode." \
  -m "Constraint: MTP must remain opt-in until live correctness and throughput gates pass" \
  -m "Rejected: Enable MTP by model config alone | the current MLX path strips model.mtp weights and needs explicit sidecar wiring" \
  -m "Confidence: high" \
  -m "Scope-risk: narrow" \
  -m "Tested: uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_config.py -q; uv run ruff check src/exo/worker/engines/mlx/mimo_mtp src/exo/worker/engines/mlx/tests/test_mimo_mtp_config.py; uv run basedpyright src/exo/worker/engines/mlx/mimo_mtp/config.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_config.py" \
  -m "Not-tested: No decode integration yet"
```

Gate: stop unless config tests, lint, and typecheck pass.

---

### Task 3: Load The MTP Sidecar Artifact Without The Main Model

**Files:**
- Create: `src/exo/worker/engines/mlx/mimo_mtp/artifact.py`
- Create: `src/exo/worker/engines/mlx/tests/test_mimo_mtp_artifact.py`
- Reuse: `scripts/mimo_v25_pro_mtp_module_probe.py`

- [ ] **Step 1: Write failing artifact loader tests**

Create tests that require:

- Production artifact paths are rejected.
- Relative traversal is rejected.
- The experimental artifact root is accepted.
- `model.safetensors.index.json` must contain `artifact_kind=mimo-v25-pro-mtp-only`.
- Required keys for layers `0`, `1`, and `2` are present.
- The loader returns an immutable `MimoMtpArtifact` with `layers=[0, 1, 2]`, `hidden_size=6144`, `eh_input_size=12288`, and `qkv_output_size=27136`.
- The loader does not import or call `mlx_lm.utils.load_model`.
- `load_mtp_tensors()` loads only `model_mtp-00001-of-00001.safetensors`.

Run:

```bash
uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_artifact.py -q
```

Expected:

- Tests fail because `artifact.py` does not exist.

- [ ] **Step 2: Implement artifact loader**

Create `src/exo/worker/engines/mlx/mimo_mtp/artifact.py` with:

- `MimoMtpArtifact`
- `MimoMtpTensorStore`
- `guard_mimo_mtp_artifact_path(path: Path) -> None`
- `load_mimo_mtp_artifact(path: Path) -> MimoMtpArtifact`
- `load_mimo_mtp_tensors(artifact: MimoMtpArtifact) -> MimoMtpTensorStore`
- `required_mimo_mtp_keys(layers: Sequence[int]) -> list[str]`

The key set must include, for every MTP layer:

- `eh_proj.weight`, `eh_proj.weight.scales`, `eh_proj.weight.biases`
- `enorm.weight`
- `hnorm.weight`
- `input_layernorm.weight`
- `self_attn.qkv_proj.weight`, `self_attn.qkv_proj.weight.scales`, `self_attn.qkv_proj.weight.biases`
- `self_attn.o_proj.weight`
- `pre_mlp_layernorm.weight`
- `mlp.gate_proj.weight`, `mlp.gate_proj.weight.scales`, `mlp.gate_proj.weight.biases`
- `mlp.up_proj.weight`, `mlp.up_proj.weight.scales`, `mlp.up_proj.weight.biases`
- `mlp.down_proj.weight`, `mlp.down_proj.weight.scales`, `mlp.down_proj.weight.biases`
- `final_layernorm.weight`

- [ ] **Step 3: Verify artifact loader against the real sidecar**

Run:

```bash
uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_artifact.py -q
uv run python scripts/mimo_v25_pro_mtp_module_probe.py --artifact /Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental --json-out docs/plans/artifacts/mimo-v25-pro-mtp-module-probe-20260519.json
uv run ruff check src/exo/worker/engines/mlx/mimo_mtp/artifact.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_artifact.py
uv run basedpyright src/exo/worker/engines/mlx/mimo_mtp/artifact.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_artifact.py
```

Expected:

- Artifact tests pass.
- Probe JSON still reports `loaded_tensor_count=72`.
- `ruff` reports `All checks passed!`.
- `basedpyright` reports `0 errors, 0 warnings, 0 notes`.

- [ ] **Step 4: Commit artifact loader**

Run:

```bash
git add src/exo/worker/engines/mlx/mimo_mtp/artifact.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_artifact.py docs/plans/artifacts/mimo-v25-pro-mtp-module-probe-20260519.json
git commit -m "Load MiMo MTP sidecar artifact" \
  -m "Add a runtime artifact reader that accepts only the experimental MTP sidecar and never loads the full MiMo base model." \
  -m "Constraint: The production 6-bit artifact remains read-only" \
  -m "Rejected: Reuse MLX-LM sanitize for model.mtp weights | the pinned mimo_v2_flash sanitize removes those keys" \
  -m "Confidence: high" \
  -m "Scope-risk: narrow" \
  -m "Tested: uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_artifact.py -q; uv run python scripts/mimo_v25_pro_mtp_module_probe.py --artifact /Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental --json-out docs/plans/artifacts/mimo-v25-pro-mtp-module-probe-20260519.json; uv run ruff check src/exo/worker/engines/mlx/mimo_mtp/artifact.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_artifact.py; uv run basedpyright src/exo/worker/engines/mlx/mimo_mtp/artifact.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_artifact.py" \
  -m "Not-tested: MTP decode is still unimplemented"
```

Gate: stop unless the sidecar loader passes real artifact validation.

---

### Task 4: Implement Standalone MTP Modules

**Files:**
- Create: `src/exo/worker/engines/mlx/mimo_mtp/modules.py`
- Create: `src/exo/worker/engines/mlx/tests/test_mimo_mtp_modules.py`

- [ ] **Step 1: Write failing module tests**

Tests must use tiny in-memory tensors for fast CPU/MLX execution and require:

- Quantized affine linear calls `mx.quantized_matmul` with `bits=6`, `group_size=64`, `mode="affine"`, and `transpose=True`.
- `MimoMtpLayer` accepts a previous hidden state and a token embedding, concatenates normalized inputs to width `2 * hidden_size`, and produces width `hidden_size`.
- `MimoMtpStack.propose()` returns exactly `max_draft_tokens` candidate token ids plus per-layer logits.
- The module can share the base model token embedding and `lm_head` without copying the full base weights.
- The module can load the real sidecar tensor keys and run layer `0` synthetic projections with shapes `[1, 1, 6144]` and `[1, 1, 27136]`.

Run:

```bash
uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_modules.py -q
```

Expected:

- Tests fail because `modules.py` does not exist.

- [ ] **Step 2: Implement module stack**

Create:

- `QuantizedAffineLinear`
- `MimoMtpAttention`
- `MimoMtpMLP`
- `MimoMtpLayer`
- `MimoMtpStack`
- `build_mimo_mtp_stack(artifact: MimoMtpArtifact, tensor_store: MimoMtpTensorStore, base_model: nn.Module) -> MimoMtpStack`

Forward semantics:

1. Read the latest base hidden state from the main MiMo forward.
2. Embed the latest accepted token through `base_model.model.embed_tokens`.
3. Apply `hnorm` to the hidden state and `enorm` to the token embedding.
4. Concatenate normalized embedding and hidden state on the last axis.
5. Apply `eh_proj`.
6. Apply the MTP self-attention and MLP sublayer with the MTP layer norms.
7. Apply `final_layernorm`.
8. Project through the shared base `lm_head`.
9. Sample a draft token using the same sampler and logits processors contract as normal decode.
10. Feed the draft token embedding into the next MTP layer.

For the first rollout, implement MTP caches as a separate `MimoMtpCache` object and reset them on fallback. Do not store MTP cache state in the base model KV cache.

- [ ] **Step 3: Verify module stack**

Run:

```bash
uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_modules.py -q
uv run ruff check src/exo/worker/engines/mlx/mimo_mtp/modules.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_modules.py
uv run basedpyright src/exo/worker/engines/mlx/mimo_mtp/modules.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_modules.py
```

Expected:

- Module tests pass.
- `ruff` reports `All checks passed!`.
- `basedpyright` reports `0 errors, 0 warnings, 0 notes`.

- [ ] **Step 4: Commit module stack**

Run:

```bash
git add src/exo/worker/engines/mlx/mimo_mtp/modules.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_modules.py
git commit -m "Run MiMo MTP sidecar modules" \
  -m "Build the MTP sidecar module stack around the quantized artifact and shared base MiMo embedding/head surfaces." \
  -m "Constraint: MTP weights are loaded from the sidecar artifact and not from the base model sanitize path" \
  -m "Rejected: Copy base embedding and lm_head weights | sharing avoids extra full-model memory pressure" \
  -m "Confidence: medium" \
  -m "Scope-risk: moderate" \
  -m "Tested: uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_modules.py -q; uv run ruff check src/exo/worker/engines/mlx/mimo_mtp/modules.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_modules.py; uv run basedpyright src/exo/worker/engines/mlx/mimo_mtp/modules.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_modules.py" \
  -m "Not-tested: Full MiMo main-model runtime remains gated"
```

Gate: stop unless tiny module tests and real sidecar projection checks pass.

---

### Task 5: Build The Propose, Verify, Accept Controller

**Files:**
- Create: `src/exo/worker/engines/mlx/mimo_mtp/controller.py`
- Create: `src/exo/worker/engines/mlx/tests/test_mimo_mtp_controller.py`

- [ ] **Step 1: Write failing controller tests**

Tests must use a deterministic synthetic model and cover:

- No proposals: emits the normal next token and records fallback.
- All proposals accepted: emits the normal next token plus all accepted draft tokens.
- First proposal rejected: emits the normal next token plus the main-model correction token.
- Middle proposal rejected: emits the normal next token, the accepted prefix, and the correction token.
- Stop token in accepted prefix stops emission at the stop token.
- Max output token count is never exceeded.
- Logits processors receive the same token history as normal decode.
- Telemetry counts `proposed`, `accepted`, `emitted`, `rejected`, `fallback_steps`, and `verification_steps`.
- Controller returns cache trim instructions that match the MLX-LM speculative decode pattern.

Run:

```bash
uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_controller.py -q
```

Expected:

- Tests fail because `controller.py` does not exist.

- [ ] **Step 2: Implement controller**

Create:

- `MimoMtpProposal`
- `MimoMtpVerification`
- `MimoMtpEmission`
- `MimoMtpDecodeController`
- `accept_longest_prefix(draft_tokens: Sequence[int], verified_tokens: Sequence[int]) -> int`

Verification semantics:

1. The main model produces the normal next token `t0` and its logprobs.
2. The MTP stack proposes draft tokens `[d1, d2, d3]`.
3. The verifier runs the main model once on `[t0, d1, d2, d3]` using a trim-capable verification cache.
4. Verified sampled tokens are compared against the draft tokens in order.
5. Accepted prefix length is the longest prefix where `verified[i] == draft[i]`.
6. Emission contains `t0`, the accepted draft prefix, and the first correction token when a rejection occurs before the draft list is exhausted.
7. The base cache is trimmed using the same rollback count as MLX-LM speculative decoding.
8. When verification fails structurally, emit only `t0`, reset MTP cache, and record fallback.

- [ ] **Step 3: Verify controller**

Run:

```bash
uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_controller.py -q
uv run ruff check src/exo/worker/engines/mlx/mimo_mtp/controller.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_controller.py
uv run basedpyright src/exo/worker/engines/mlx/mimo_mtp/controller.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_controller.py
```

Expected:

- Controller tests pass.
- `ruff` reports `All checks passed!`.
- `basedpyright` reports `0 errors, 0 warnings, 0 notes`.

- [ ] **Step 4: Commit controller**

Run:

```bash
git add src/exo/worker/engines/mlx/mimo_mtp/controller.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_controller.py
git commit -m "Control MiMo MTP proposal verification" \
  -m "Add deterministic propose-verify-accept semantics before wiring the path into exo generation." \
  -m "Constraint: Correct fallback behavior must be proven before speed work" \
  -m "Rejected: Emit unverified MTP tokens | accepted-token divergence would corrupt user-visible output" \
  -m "Confidence: high" \
  -m "Scope-risk: moderate" \
  -m "Tested: uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_controller.py -q; uv run ruff check src/exo/worker/engines/mlx/mimo_mtp/controller.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_controller.py; uv run basedpyright src/exo/worker/engines/mlx/mimo_mtp/controller.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_controller.py" \
  -m "Not-tested: Full MLX generator integration"
```

Gate: stop unless every controller rejection and fallback case is covered.

---

### Task 6: Add An MTP Stream Generator For Single-Stream MiMo

**Files:**
- Create: `src/exo/worker/engines/mlx/mimo_mtp/generation.py`
- Create: `src/exo/worker/engines/mlx/tests/test_mimo_mtp_generation.py`

- [ ] **Step 1: Write failing generation tests**

Tests must use tiny fake MLX modules and require:

- `mimo_forward_logits_and_hidden()` calls `model.model(input_tokens, cache)` and `model.lm_head(hidden)`.
- `mimo_mtp_generate_step()` emits the same first token as normal decode when MTP is enabled.
- Accepted MTP tokens are emitted in the same order as normal autoregressive verification.
- The generator calls `cache.trim_prompt_cache` with the expected trim count after rejection.
- EOS token terminates the stream.
- `max_tokens` caps visible emitted tokens, not verification calls.
- Generation telemetry includes `effective_generation_tps`, `accepted_per_verify_step`, and `normal_generation_tps_estimate`.
- Setting `EXO_MIMO_MTP=0` returns a disabled decision and never builds the sidecar stack.

Run:

```bash
uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_generation.py -q
```

Expected:

- Tests fail because `generation.py` does not exist.

- [ ] **Step 2: Implement generator**

Create:

- `mimo_forward_logits_and_hidden(model: nn.Module, input_tokens: mx.array, cache: object) -> tuple[mx.array, mx.array]`
- `mimo_mtp_generate_step(model: nn.Module, state: MimoMtpGenerationState) -> MimoMtpEmission`
- `stream_mimo_mtp_generate(model: nn.Module, tokenizer: TokenizerWrapper, prompt: mx.array, state: MimoMtpGenerationState) -> Generator[GenerationResponse, None, None]`
- `MimoMtpGenerationState`

Implementation requirements:

- Mirror the MLX-LM `generate_step` and `speculative_generate_step` structure instead of patching MLX-LM source.
- Use the existing `generation_stream`.
- Use the same sampler and logits processors that `mlx_generate()` already creates.
- Use `cache.trim_prompt_cache` for verification rollback.
- Update `MimoMtpTelemetry` after every decode cycle.
- Call `mx.async_eval` for the next cycle only after cache state is valid.
- Fall back to normal single-token generation when the sidecar raises a controlled `MimoMtpUnavailable` exception.
- Do not integrate with batch generation in this task.

- [ ] **Step 3: Verify generation module**

Run:

```bash
uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_generation.py -q
uv run ruff check src/exo/worker/engines/mlx/mimo_mtp/generation.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_generation.py
uv run basedpyright src/exo/worker/engines/mlx/mimo_mtp/generation.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_generation.py
```

Expected:

- Generation tests pass.
- `ruff` reports `All checks passed!`.
- `basedpyright` reports `0 errors, 0 warnings, 0 notes`.

- [ ] **Step 4: Commit generator**

Run:

```bash
git add src/exo/worker/engines/mlx/mimo_mtp/generation.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_generation.py
git commit -m "Generate MiMo tokens through MTP verification" \
  -m "Add a single-stream MTP generator that follows MLX-LM speculative cache rollback while preserving normal decode fallback." \
  -m "Constraint: Accepted tokens must be verified by the base model before emission" \
  -m "Rejected: Patch mlx_lm.generate directly | local wrapper keeps the change scoped to exo MiMo runtime" \
  -m "Confidence: medium" \
  -m "Scope-risk: moderate" \
  -m "Tested: uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_generation.py -q; uv run ruff check src/exo/worker/engines/mlx/mimo_mtp/generation.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_generation.py; uv run basedpyright src/exo/worker/engines/mlx/mimo_mtp/generation.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_generation.py" \
  -m "Not-tested: Exo request integration and distributed sharding"
```

Gate: stop unless synthetic MTP generation matches normal autoregressive verification.

---

### Task 7: Wire Opt-In MTP Into Exo Single-Stream Generation

**Files:**
- Modify: `src/exo/worker/engines/mlx/generator/generate.py`
- Modify: `src/exo/worker/engines/mlx/generator/batch_generate.py`
- Modify: `src/exo/worker/engines/mlx/utils_mlx.py`
- Create: `src/exo/worker/engines/mlx/tests/test_mimo_mtp_integration.py`

- [ ] **Step 1: Read dirty files before edits**

Run:

```bash
git diff -- src/exo/worker/engines/mlx/generator/generate.py src/exo/worker/engines/mlx/generator/batch_generate.py src/exo/worker/engines/mlx/cache.py src/exo/worker/tests/unittests/test_mlx/test_kv_prefix_cache.py
```

Expected:

- The existing dirty KV/cache work is understood and not reverted.

- [ ] **Step 2: Write failing integration tests**

Tests must require:

- `mlx_generate()` selects `stream_mimo_mtp_generate()` only when config and eligibility pass.
- MTP is not selected when `vision_processor` is present.
- MTP is not selected during `warmup_inference()`.
- MTP is not selected when `task.logprobs` is true.
- MTP is not selected when KV prefix-cache has a hit.
- MTP is not selected for non-MiMo model ids.
- MTP is not selected when artifact root is missing.
- `batch_generate.py` logs a disabled reason and uses the existing `MlxBatchGenerator`.
- MTP telemetry is logged at request completion.
- `CompletionTokensDetails.accepted_prediction_tokens` and `CompletionTokensDetails.rejected_prediction_tokens` are filled from MTP telemetry for completed MTP requests.
- `EXO_MIMO_MTP=0` disables MTP even for the promoted model id.

Run:

```bash
uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_integration.py -q
```

Expected:

- Tests fail because integration is not wired.

- [ ] **Step 3: Modify single-stream generation**

In `src/exo/worker/engines/mlx/generator/generate.py`:

- Import `load_mimo_mtp_runtime_config` and `is_mimo_mtp_eligible`.
- Import `stream_mimo_mtp_generate`.
- Add `mimo_mtp_allowed: bool = True` to `mlx_generate()`.
- Pass `mimo_mtp_allowed=False` from `warmup_inference()`.
- After prefill and before the existing `stream_generate` loop, compute eligibility.
- Use MTP stream only when eligible.
- Preserve the existing `stream_generate` loop for all ineligible paths.
- Preserve stop-sequence trimming, thinking token accounting, usage, stats, logprobs fallback, and prefix-cache update semantics.
- Set `Usage.completion_tokens_details.accepted_prediction_tokens` and `Usage.completion_tokens_details.rejected_prediction_tokens` from the MTP telemetry when MTP ran.
- Log one structured summary at completion:
  - `model`
  - `enabled`
  - `disabled_reason`
  - `proposed_tokens`
  - `accepted_tokens`
  - `emitted_tokens`
  - `fallback_steps`
  - `verification_steps`
  - `acceptance_rate`
  - `effective_generation_tps`

In `src/exo/worker/engines/mlx/generator/batch_generate.py`:

- Import `load_mimo_mtp_runtime_config`.
- Log `mimo_mtp=batch-disabled` when the config is on.
- Keep `MlxBatchGenerator` as the only batch path for this rollout.

In `src/exo/worker/engines/mlx/utils_mlx.py`:

- Keep existing MiMo alias and qkv split behavior.
- Do not make upstream sanitize preserve `model.mtp.*` for the base model.
- Add a small helper that reports whether a loaded model object is MLX-LM MiMo and has the required `model` and `lm_head` attributes.

- [ ] **Step 4: Verify integration**

Run:

```bash
uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_config.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_artifact.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_modules.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_controller.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_generation.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_integration.py -q
uv run ruff check src/exo/worker/engines/mlx/mimo_mtp src/exo/worker/engines/mlx/generator/generate.py src/exo/worker/engines/mlx/generator/batch_generate.py src/exo/worker/engines/mlx/utils_mlx.py
uv run basedpyright src/exo/worker/engines/mlx/mimo_mtp src/exo/worker/engines/mlx/generator/generate.py src/exo/worker/engines/mlx/generator/batch_generate.py src/exo/worker/engines/mlx/utils_mlx.py
```

Expected:

- Focused MTP tests pass.
- `ruff` reports `All checks passed!`.
- `basedpyright` reports `0 errors, 0 warnings, 0 notes`.

- [ ] **Step 5: Commit opt-in integration**

Run:

```bash
git add src/exo/worker/engines/mlx/mimo_mtp src/exo/worker/engines/mlx/generator/generate.py src/exo/worker/engines/mlx/generator/batch_generate.py src/exo/worker/engines/mlx/utils_mlx.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_integration.py
git commit -m "Wire opt-in MiMo MTP decode" \
  -m "Route eligible single-stream MiMo requests through the verified MTP generator while preserving the existing MLX and batch fallbacks." \
  -m "Constraint: Batch, vision, logprobs, prefix-cache, and unsupported models remain on the existing path" \
  -m "Rejected: Enable MTP from model id alone | rollout needs explicit fail-closed config until live gates pass" \
  -m "Confidence: medium" \
  -m "Scope-risk: broad" \
  -m "Tested: uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_config.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_artifact.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_modules.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_controller.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_generation.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_integration.py -q; uv run ruff check src/exo/worker/engines/mlx/mimo_mtp src/exo/worker/engines/mlx/generator/generate.py src/exo/worker/engines/mlx/generator/batch_generate.py src/exo/worker/engines/mlx/utils_mlx.py; uv run basedpyright src/exo/worker/engines/mlx/mimo_mtp src/exo/worker/engines/mlx/generator/generate.py src/exo/worker/engines/mlx/generator/batch_generate.py src/exo/worker/engines/mlx/utils_mlx.py" \
  -m "Not-tested: Full model startup remains guarded"
```

Gate: stop unless opt-in integration proves all unsupported request types remain on the known-good path.

---

### Task 8: Add Tensor/JACCL Sharding For The MTP Sidecar

**Files:**
- Create: `src/exo/worker/engines/mlx/mimo_mtp/sharding.py`
- Create: `src/exo/worker/engines/mlx/tests/test_mimo_mtp_sharding.py`
- Modify: `src/exo/worker/engines/mlx/auto_parallel.py`
- Modify: `src/exo/worker/engines/mlx/utils_mlx.py`

- [ ] **Step 1: Write failing sharding tests**

Tests must require:

- MTP attention q, k, v projections are split from fused `qkv_proj` using MiMo V2 shapes.
- MTP attention head counts are divided by tensor world size.
- `attention_sink_bias` is sliced by rank.
- MTP MLP gate and up projections use all-to-sharded sharding.
- MTP MLP down projection uses sharded-to-all sharding.
- `eh_proj` uses all-to-sharded sharding and preserves output width per rank.
- The sidecar stack is sharded after base model tensor sharding and before generation.
- Pipeline sharding rejects MTP in this rollout with a clear disabled reason.

Run:

```bash
uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_sharding.py -q
```

Expected:

- Tests fail because MTP sharding is not implemented.

- [ ] **Step 2: Implement sharding**

In `src/exo/worker/engines/mlx/mimo_mtp/sharding.py` create:

- `shard_mimo_mtp_stack(stack: MimoMtpStack, group: mx.distributed.Group, helpers: MimoMtpShardHelpers) -> MimoMtpStack`
- `MimoMtpShardHelpers`
- `split_mimo_mtp_qkv(layer: MimoMtpLayer, args: object) -> None`

In `src/exo/worker/engines/mlx/auto_parallel.py`:

- Factor reusable MiMo linear sharding helpers from `MimoV2ShardingStrategy`.
- Add a public helper that can shard an already-built MTP sidecar stack with the same tensor group.
- Keep base model sharding behavior unchanged.

In `src/exo/worker/engines/mlx/utils_mlx.py`:

- After base MiMo tensor sharding, allow the MTP sidecar to be built and sharded only when config is enabled and the shard metadata is `TensorShardMetadata`.
- Do not load or shard the sidecar for `PipelineShardMetadata`.
- Do not evaluate the full MTP stack before it is sliced by rank.
- Attach the sharded sidecar to the loaded MiMo model as `_exo_mimo_mtp_stack` after successful base model tensor sharding.
- Let single-node generation lazily build the sidecar when `_exo_mimo_mtp_stack` is absent and eligibility passes.

- [ ] **Step 3: Verify sharding**

Run:

```bash
uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_sharding.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_integration.py -q
uv run ruff check src/exo/worker/engines/mlx/mimo_mtp/sharding.py src/exo/worker/engines/mlx/auto_parallel.py src/exo/worker/engines/mlx/utils_mlx.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_sharding.py
uv run basedpyright src/exo/worker/engines/mlx/mimo_mtp/sharding.py src/exo/worker/engines/mlx/auto_parallel.py src/exo/worker/engines/mlx/utils_mlx.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_sharding.py
```

Expected:

- Sharding and integration tests pass.
- `ruff` reports `All checks passed!`.
- `basedpyright` reports `0 errors, 0 warnings, 0 notes`.

- [ ] **Step 4: Commit sharding**

Run:

```bash
git add src/exo/worker/engines/mlx/mimo_mtp/sharding.py src/exo/worker/engines/mlx/auto_parallel.py src/exo/worker/engines/mlx/utils_mlx.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_sharding.py
git commit -m "Shard MiMo MTP sidecar for tensor runtime" \
  -m "Apply the existing MiMo tensor/JACCL sharding strategy to the MTP sidecar after base-model sharding and before decode." \
  -m "Constraint: The Studio1/Studio2 TB5 tensor path is the primary performance target" \
  -m "Rejected: Replicate full MTP weights on every rank | sharding keeps memory pressure bounded" \
  -m "Confidence: medium" \
  -m "Scope-risk: broad" \
  -m "Tested: uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_sharding.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_integration.py -q; uv run ruff check src/exo/worker/engines/mlx/mimo_mtp/sharding.py src/exo/worker/engines/mlx/auto_parallel.py src/exo/worker/engines/mlx/utils_mlx.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_sharding.py; uv run basedpyright src/exo/worker/engines/mlx/mimo_mtp/sharding.py src/exo/worker/engines/mlx/auto_parallel.py src/exo/worker/engines/mlx/utils_mlx.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_sharding.py" \
  -m "Not-tested: Live tensor startup remains gated by the rollout guard"
```

Gate: stop unless MTP sidecar sharding is proven without materializing full unsliced MTP weights per rank.

---

### Task 9: Add Rollout Guard And Benchmark Report Scripts

**Files:**
- Create: `scripts/mimo_v25_pro_mtp_rollout_guard.py`
- Create: `scripts/test_mimo_v25_pro_mtp_rollout_guard.py`
- Create: `scripts/mimo_v25_pro_mtp_bench_report.py`
- Create: `scripts/test_mimo_v25_pro_mtp_bench_report.py`

- [ ] **Step 1: Write failing rollout guard tests**

Tests must require:

- `preflight` wraps `mimo_v25_pro_runtime_guard.evaluate_state`.
- Unsafe guard verdict exits non-zero and prints JSON.
- Safe guard verdict exits zero and prints JSON.
- Existing MiMo resident process is unsafe.
- Existing runner loading or warming is unsafe.
- Node below memory floor is unsafe.
- `assert-no-duplicate-start` checks current process list immediately before launch.
- The script never calls `kill`, `pkill`, or `subprocess.Popen` in tests.

Run:

```bash
uv run pytest scripts/test_mimo_v25_pro_mtp_rollout_guard.py -q
```

Expected:

- Tests fail because the rollout guard script does not exist.

- [ ] **Step 2: Implement rollout guard**

Create CLI modes:

- `preflight`
- `assert-no-duplicate-start`
- `wait-safe`
- `record-state`

The script must call the existing runtime guard and write JSON reports under `docs/plans/artifacts/` when `--json-out` is supplied. It must not terminate processes directly. Process stopping remains an explicit shell action in the live runbook.

- [ ] **Step 3: Write failing benchmark report tests**

Tests must require:

- Baseline and MTP bench JSON files are parsed.
- Report includes `baseline_generation_tps`, `mtp_generation_tps`, `speedup_ratio`, `accepted_tokens`, `proposed_tokens`, `acceptance_rate`, and `promotion_passed`.
- Promotion passes only when MTP generation TPS is greater than `22.0` and at least `1.10x` the measured non-MTP baseline for the same placement.
- Stretch status reports whether MTP reaches `40.0 tok/s`.
- Missing telemetry fails closed.

Run:

```bash
uv run pytest scripts/test_mimo_v25_pro_mtp_bench_report.py -q
```

Expected:

- Tests fail because the benchmark report script does not exist.

- [ ] **Step 4: Implement benchmark report**

Create CLI:

```bash
uv run python scripts/mimo_v25_pro_mtp_bench_report.py \
  --baseline docs/plans/artifacts/mimo-v25-pro-mtp-baseline-tb5-20260519.json \
  --mtp docs/plans/artifacts/mimo-v25-pro-mtp-enabled-tb5-20260519.json \
  --telemetry docs/plans/artifacts/mimo-v25-pro-mtp-telemetry-tb5-20260519.jsonl \
  --json-out docs/plans/artifacts/mimo-v25-pro-mtp-rollout-report-20260519.json
```

The script exits `0` only when the promotion gate passes. It exits `2` when evidence is valid but promotion fails. It exits `3` when evidence is malformed.

- [ ] **Step 5: Verify scripts**

Run:

```bash
uv run pytest scripts/test_mimo_v25_pro_mtp_rollout_guard.py scripts/test_mimo_v25_pro_mtp_bench_report.py -q
uv run ruff check scripts/mimo_v25_pro_mtp_rollout_guard.py scripts/test_mimo_v25_pro_mtp_rollout_guard.py scripts/mimo_v25_pro_mtp_bench_report.py scripts/test_mimo_v25_pro_mtp_bench_report.py
uv run basedpyright scripts/mimo_v25_pro_mtp_rollout_guard.py scripts/test_mimo_v25_pro_mtp_rollout_guard.py scripts/mimo_v25_pro_mtp_bench_report.py scripts/test_mimo_v25_pro_mtp_bench_report.py
```

Expected:

- Script tests pass.
- `ruff` reports `All checks passed!`.
- `basedpyright` reports `0 errors, 0 warnings, 0 notes`.

- [ ] **Step 6: Commit scripts**

Run:

```bash
git add scripts/mimo_v25_pro_mtp_rollout_guard.py scripts/test_mimo_v25_pro_mtp_rollout_guard.py scripts/mimo_v25_pro_mtp_bench_report.py scripts/test_mimo_v25_pro_mtp_bench_report.py
git commit -m "Guard MiMo MTP live rollout evidence" \
  -m "Add fail-closed preflight and benchmark report scripts so live rollout cannot promote without safety, correctness, and speed evidence." \
  -m "Constraint: Duplicate MiMo startup can crash the Studio nodes" \
  -m "Rejected: Promote from raw bench output alone | telemetry and baseline comparison are required" \
  -m "Confidence: high" \
  -m "Scope-risk: narrow" \
  -m "Tested: uv run pytest scripts/test_mimo_v25_pro_mtp_rollout_guard.py scripts/test_mimo_v25_pro_mtp_bench_report.py -q; uv run ruff check scripts/mimo_v25_pro_mtp_rollout_guard.py scripts/test_mimo_v25_pro_mtp_rollout_guard.py scripts/mimo_v25_pro_mtp_bench_report.py scripts/test_mimo_v25_pro_mtp_bench_report.py; uv run basedpyright scripts/mimo_v25_pro_mtp_rollout_guard.py scripts/test_mimo_v25_pro_mtp_rollout_guard.py scripts/mimo_v25_pro_mtp_bench_report.py scripts/test_mimo_v25_pro_mtp_bench_report.py" \
  -m "Not-tested: Live exo lifecycle commands remain gated"
```

Gate: stop unless rollout scripts fail closed on incomplete or unsafe evidence.

---

### Task 10: Run Full Pre-Live Verification

**Files:**
- All code touched in Tasks 2-9.

- [ ] **Step 1: Run focused MTP test suite**

Run:

```bash
uv run pytest \
  scripts/test_mimo_v25_pro_runtime_guard.py \
  scripts/test_mimo_v25_pro_mtp_artifact_probe.py \
  scripts/test_mimo_v25_pro_mtp_quantize.py \
  scripts/test_mimo_v25_pro_mtp_module_probe.py \
  scripts/test_mimo_v25_pro_mtp_rollout_guard.py \
  scripts/test_mimo_v25_pro_mtp_bench_report.py \
  src/exo/worker/engines/mlx/tests/test_mimo_mtp_config.py \
  src/exo/worker/engines/mlx/tests/test_mimo_mtp_artifact.py \
  src/exo/worker/engines/mlx/tests/test_mimo_mtp_modules.py \
  src/exo/worker/engines/mlx/tests/test_mimo_mtp_controller.py \
  src/exo/worker/engines/mlx/tests/test_mimo_mtp_generation.py \
  src/exo/worker/engines/mlx/tests/test_mimo_mtp_integration.py \
  src/exo/worker/engines/mlx/tests/test_mimo_mtp_sharding.py \
  -q
```

Expected:

- All focused tests pass.

- [ ] **Step 2: Run repo checks for touched surfaces**

Run:

```bash
uv run ruff check scripts/mimo_v25_pro_runtime_guard.py scripts/mimo_v25_pro_mtp_artifact_probe.py scripts/mimo_v25_pro_mtp_quantize.py scripts/mimo_v25_pro_mtp_module_probe.py scripts/mimo_v25_pro_mtp_rollout_guard.py scripts/mimo_v25_pro_mtp_bench_report.py src/exo/worker/engines/mlx/mimo_mtp src/exo/worker/engines/mlx/generator/generate.py src/exo/worker/engines/mlx/generator/batch_generate.py src/exo/worker/engines/mlx/utils_mlx.py src/exo/worker/engines/mlx/auto_parallel.py
uv run basedpyright scripts/mimo_v25_pro_runtime_guard.py scripts/mimo_v25_pro_mtp_artifact_probe.py scripts/mimo_v25_pro_mtp_quantize.py scripts/mimo_v25_pro_mtp_module_probe.py scripts/mimo_v25_pro_mtp_rollout_guard.py scripts/mimo_v25_pro_mtp_bench_report.py src/exo/worker/engines/mlx/mimo_mtp src/exo/worker/engines/mlx/generator/generate.py src/exo/worker/engines/mlx/generator/batch_generate.py src/exo/worker/engines/mlx/utils_mlx.py src/exo/worker/engines/mlx/auto_parallel.py
```

Expected:

- `ruff` reports `All checks passed!`.
- `basedpyright` reports `0 errors, 0 warnings, 0 notes`.

- [ ] **Step 3: Run runtime guard in non-launch mode**

Run:

```bash
uv run python scripts/mimo_v25_pro_mtp_rollout_guard.py preflight --exo-url http://127.0.0.1:52415 --json-out docs/plans/artifacts/mimo-v25-pro-mtp-prelive-guard-20260519.json
```

Expected:

- If an existing exo cluster is up, the command may exit `2` with unsafe reasons. That is an expected hard stop for live startup, not a test failure.
- If no cluster is up and memory is recovered, it exits `0`.
- The JSON output is committed either way as pre-live state evidence.

- [ ] **Step 4: Commit pre-live evidence**

Run:

```bash
git add docs/plans/artifacts/mimo-v25-pro-mtp-prelive-guard-20260519.json
git commit -m "Record MiMo MTP pre-live gate" \
  -m "Capture guard evidence before any full-model startup attempt." \
  -m "Constraint: Live MiMo startup requires a clean duplicate-start and memory preflight" \
  -m "Confidence: high" \
  -m "Scope-risk: narrow" \
  -m "Tested: focused MTP pytest suite; touched-surface ruff; touched-surface basedpyright; rollout preflight guard" \
  -m "Not-tested: Full MiMo model not started in this gate"
```

Gate: stop before live testing unless focused tests, lint, typecheck, and guard evidence are complete.

---

### Task 11: Single-Node Full-Model Validation

**Files:**
- Generate evidence under: `docs/plans/artifacts/`
- Update: `docs/plans/MIMO_V25_PRO_MTP_ROLLOUT_RUNLOG.md`

This task loads the full MiMo model. Run it only during a downtime window after the existing exo cluster is stopped and memory is recovered.

- [ ] **Step 1: Stop the existing exo cluster deliberately**

On each Studio node that is running exo, send `SIGTERM` to the known exo process from the terminal that launched it. When the launch terminal is unavailable, run:

```bash
pgrep -af 'uv run exo|python.*exo|/exo '
```

Then record and terminate only the matching exo process identifiers:

```bash
pgrep -af 'uv run exo|python.*exo|/exo ' | awk '{print $1}' \
  | tee docs/plans/artifacts/mimo-v25-pro-mtp-exo-pids-to-stop-20260519.txt
while read -r process_id; do
  test -n "$process_id" && kill -TERM "$process_id"
done < docs/plans/artifacts/mimo-v25-pro-mtp-exo-pids-to-stop-20260519.txt
```

Expected:

- `curl http://127.0.0.1:52415/state` fails to connect or returns no active MiMo instance.
- `ps axo command= | rg 'MiMo-V2.5-Pro|kernelpool/MiMo-V2.5-Pro-6bit'` returns no resident model process.

- [ ] **Step 2: Wait for local guard success**

Run:

```bash
uv run python scripts/mimo_v25_pro_mtp_rollout_guard.py wait-safe --exo-url http://127.0.0.1:52415 --min-available-gib 260 --json-out docs/plans/artifacts/mimo-v25-pro-mtp-single-node-guard-20260519.json
```

Expected:

- Exit code `0`.
- JSON contains `"safe": true` and `"reasons": []`.

- [ ] **Step 3: Start one local experimental exo process**

Run from `/Users/studio2/exo`:

```bash
EXO_MIMO_MTP=1 \
EXO_MIMO_MTP_FAIL_CLOSED=1 \
EXO_MIMO_MTP_ARTIFACT_ROOT=/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental \
EXO_MIMO_MTP_MAX_DRAFT_TOKENS=3 \
EXO_LIBP2P_NAMESPACE=mimo-mtp-single-node-20260519 \
EXO_OFFLINE=true \
uv run exo -vv
```

Expected:

- Exactly one local exo process starts.
- No second full MiMo instance is launched.
- Dashboard/API is available at `http://127.0.0.1:52415`.

- [ ] **Step 4: Create one bounded MiMo instance**

Run:

```bash
curl -sS "http://127.0.0.1:52415/instance/previews?model_id=kernelpool/MiMo-V2.5-Pro-6bit&sharding=tensor" \
  | tee docs/plans/artifacts/mimo-v25-pro-mtp-single-node-preview-20260519.json
```

Select the first preview with no error and create it:

```bash
python3 - <<'PY' > /tmp/mimo-mtp-single-node-instance.json
import json
from pathlib import Path

payload = json.loads(Path("docs/plans/artifacts/mimo-v25-pro-mtp-single-node-preview-20260519.json").read_text())
for preview in payload["previews"]:
    if preview.get("error") is None:
        print(json.dumps(preview["instance"]))
        raise SystemExit(0)
raise SystemExit("no safe preview")
PY
curl -sS -X POST http://127.0.0.1:52415/instance \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/mimo-mtp-single-node-instance.json \
  | tee docs/plans/artifacts/mimo-v25-pro-mtp-single-node-instance-20260519.json
```

Expected:

- Instance creation returns an instance id.
- The server does not create a second full MiMo instance.

- [ ] **Step 5: Run bounded correctness request**

Run:

```bash
curl -sS -N -X POST http://127.0.0.1:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
    "messages": [{"role": "user", "content": "Write exactly one sentence about why cache correctness matters."}],
    "temperature": 0,
    "max_tokens": 64,
    "stream": false
  }' \
  | tee docs/plans/artifacts/mimo-v25-pro-mtp-single-node-response-20260519.json
```

Expected:

- Request completes without crash.
- Response contains text.
- Logs contain one MTP telemetry summary.
- No accepted-token divergence is logged.

- [ ] **Step 6: Run single-node fallback request**

Restart the local process with `EXO_MIMO_MTP=0` and run the same bounded request:

```bash
curl -sS -N -X POST http://127.0.0.1:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
    "messages": [{"role": "user", "content": "Write exactly one sentence about why cache correctness matters."}],
    "temperature": 0,
    "max_tokens": 64,
    "stream": false
  }' \
  | tee docs/plans/artifacts/mimo-v25-pro-mtp-single-node-fallback-response-20260519.json
```

Expected:

- Request completes without crash.
- Logs show MTP disabled.
- Fallback output is valid text.

- [ ] **Step 7: Commit single-node evidence**

Run:

```bash
git add docs/plans/artifacts/mimo-v25-pro-mtp-single-node-guard-20260519.json docs/plans/artifacts/mimo-v25-pro-mtp-single-node-preview-20260519.json docs/plans/artifacts/mimo-v25-pro-mtp-single-node-instance-20260519.json docs/plans/artifacts/mimo-v25-pro-mtp-single-node-response-20260519.json docs/plans/artifacts/mimo-v25-pro-mtp-single-node-fallback-response-20260519.json docs/plans/MIMO_V25_PRO_MTP_ROLLOUT_RUNLOG.md
git commit -m "Validate MiMo MTP single-node startup" \
  -m "Record guarded single-node full-model evidence before moving to the TB5 tensor cluster." \
  -m "Constraint: Only one full MiMo instance may be resident during validation" \
  -m "Confidence: medium" \
  -m "Scope-risk: narrow" \
  -m "Tested: runtime guard; bounded MTP request; bounded non-MTP fallback request" \
  -m "Not-tested: Studio1/Studio2 tensor rollout remains gated"
```

Gate: stop unless the single-node MTP request and fallback request both complete crash-free.

---

### Task 12: Studio1/Studio2 TB5 Tensor Validation

**Files:**
- Generate evidence under: `docs/plans/artifacts/`
- Update: `docs/plans/MIMO_V25_PRO_MTP_ROLLOUT_RUNLOG.md`

This task is the primary performance gate. It starts exactly one distributed MiMo instance across Studio1 and Studio2.

- [ ] **Step 1: Confirm both nodes are safe**

On Studio2:

```bash
uv run python scripts/mimo_v25_pro_mtp_rollout_guard.py wait-safe --exo-url http://127.0.0.1:52415 --min-available-gib 260 --json-out docs/plans/artifacts/mimo-v25-pro-mtp-studio2-guard-20260519.json
```

On Studio1 from its exo checkout:

```bash
uv run python scripts/mimo_v25_pro_mtp_rollout_guard.py wait-safe --exo-url http://127.0.0.1:52415 --min-available-gib 260 --json-out docs/plans/artifacts/mimo-v25-pro-mtp-studio1-guard-20260519.json
```

Expected:

- Both commands exit `0`.
- Both JSON files contain `"safe": true` and `"reasons": []`.

- [ ] **Step 2: Start exo on both nodes with MTP opt-in**

Run on Studio1 and Studio2, from each node's exo checkout:

```bash
EXO_MIMO_MTP=1 \
EXO_MIMO_MTP_FAIL_CLOSED=1 \
EXO_MIMO_MTP_ARTIFACT_ROOT=/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental \
EXO_MIMO_MTP_MAX_DRAFT_TOKENS=3 \
EXO_LIBP2P_NAMESPACE=mimo-mtp-tb5-20260519 \
EXO_FAST_SYNCH=1 \
EXO_OFFLINE=true \
uv run exo -vv
```

Expected:

- Each node starts one exo process.
- The cluster sees two nodes.
- No node starts a second MiMo model instance.

- [ ] **Step 3: Confirm tensor preview**

Run from Studio2:

```bash
curl -sS "http://127.0.0.1:52415/instance/previews?model_id=kernelpool/MiMo-V2.5-Pro-6bit&sharding=tensor&max_nodes=2" \
  | tee docs/plans/artifacts/mimo-v25-pro-mtp-tb5-preview-20260519.json
```

Expected:

- At least one preview has no error.
- Chosen preview uses tensor sharding across Studio1 and Studio2.

- [ ] **Step 4: Create one distributed instance**

Run:

```bash
python3 - <<'PY' > /tmp/mimo-mtp-tb5-instance.json
import json
from pathlib import Path

payload = json.loads(Path("docs/plans/artifacts/mimo-v25-pro-mtp-tb5-preview-20260519.json").read_text())
for preview in payload["previews"]:
    if preview.get("error") is None:
        instance = preview["instance"]
        assignments = instance["shardAssignments"]["runnerToShard"]
        if len(assignments) == 2:
            print(json.dumps(instance))
            raise SystemExit(0)
raise SystemExit("no two-node tensor preview")
PY
curl -sS -X POST http://127.0.0.1:52415/instance \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/mimo-mtp-tb5-instance.json \
  | tee docs/plans/artifacts/mimo-v25-pro-mtp-tb5-instance-20260519.json
```

Expected:

- Instance creation returns one instance id.
- State shows one MiMo instance across two runners.

- [ ] **Step 5: Run bounded MTP correctness requests**

Run:

```bash
for prompt in \
  "Give one sentence on Thunderbolt tensor parallelism." \
  "Give one sentence on speculative verification." \
  "Give one sentence on safe rollback."; do
  curl -sS -N -X POST http://127.0.0.1:52415/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"kernelpool/MiMo-V2.5-Pro-6bit\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"temperature\":0,\"max_tokens\":64,\"stream\":false}" \
    >> docs/plans/artifacts/mimo-v25-pro-mtp-tb5-correctness-20260519.jsonl
  printf '\n' >> docs/plans/artifacts/mimo-v25-pro-mtp-tb5-correctness-20260519.jsonl
done
```

Expected:

- All three requests complete without crash.
- No accepted-token divergence is logged.
- MTP telemetry summary exists for each request.

- [ ] **Step 6: Run non-MTP TB5 baseline benchmark**

Restart both nodes with `EXO_MIMO_MTP=0` and the same namespace. Recreate one tensor instance. Then run:

```bash
uv run bench/exo_bench.py \
  --model kernelpool/MiMo-V2.5-Pro-6bit \
  --pp 128 \
  --tg 128 \
  --max-nodes 2 \
  --sharding tensor \
  --repeat 3 \
  --warmup 1 \
  --json-out docs/plans/artifacts/mimo-v25-pro-mtp-baseline-tb5-20260519.json
```

Expected:

- Benchmark completes.
- Reported generation throughput is close to the known `22 tok/s` baseline or the current measured non-MTP equivalent.

- [ ] **Step 7: Run MTP-enabled TB5 benchmark**

Restart both nodes with `EXO_MIMO_MTP=1`, recreate one tensor instance, and run:

```bash
uv run bench/exo_bench.py \
  --model kernelpool/MiMo-V2.5-Pro-6bit \
  --pp 128 \
  --tg 128 \
  --max-nodes 2 \
  --sharding tensor \
  --repeat 3 \
  --warmup 1 \
  --json-out docs/plans/artifacts/mimo-v25-pro-mtp-enabled-tb5-20260519.json
```

Capture MTP telemetry logs into:

```text
docs/plans/artifacts/mimo-v25-pro-mtp-telemetry-tb5-20260519.jsonl
```

Expected:

- Benchmark completes without crash.
- Telemetry includes proposed, accepted, emitted, fallback, verification, and effective TPS for every run.

- [ ] **Step 8: Generate rollout benchmark report**

Run:

```bash
uv run python scripts/mimo_v25_pro_mtp_bench_report.py \
  --baseline docs/plans/artifacts/mimo-v25-pro-mtp-baseline-tb5-20260519.json \
  --mtp docs/plans/artifacts/mimo-v25-pro-mtp-enabled-tb5-20260519.json \
  --telemetry docs/plans/artifacts/mimo-v25-pro-mtp-telemetry-tb5-20260519.jsonl \
  --json-out docs/plans/artifacts/mimo-v25-pro-mtp-rollout-report-20260519.json
```

Expected for promotion:

- Exit code `0`.
- JSON contains `"promotion_passed": true`.
- `mtp_generation_tps > 22.0`.
- `speedup_ratio >= 1.10`.
- `accepted_token_divergence_count == 0`.

Expected when MTP is correct but not fast enough:

- Exit code `2`.
- JSON contains `"promotion_passed": false`.
- Do not continue to Task 13.

- [ ] **Step 9: Commit TB5 evidence**

Run:

```bash
git add docs/plans/artifacts/mimo-v25-pro-mtp-studio2-guard-20260519.json docs/plans/artifacts/mimo-v25-pro-mtp-studio1-guard-20260519.json docs/plans/artifacts/mimo-v25-pro-mtp-tb5-preview-20260519.json docs/plans/artifacts/mimo-v25-pro-mtp-tb5-instance-20260519.json docs/plans/artifacts/mimo-v25-pro-mtp-tb5-correctness-20260519.jsonl docs/plans/artifacts/mimo-v25-pro-mtp-baseline-tb5-20260519.json docs/plans/artifacts/mimo-v25-pro-mtp-enabled-tb5-20260519.json docs/plans/artifacts/mimo-v25-pro-mtp-telemetry-tb5-20260519.jsonl docs/plans/artifacts/mimo-v25-pro-mtp-rollout-report-20260519.json docs/plans/MIMO_V25_PRO_MTP_ROLLOUT_RUNLOG.md
git commit -m "Validate MiMo MTP on TB5 tensor cluster" \
  -m "Record Studio1/Studio2 correctness, baseline, MTP-enabled throughput, and promotion evidence for the gated rollout." \
  -m "Constraint: Promotion requires MTP to beat both the 22 tok/s baseline and the measured non-MTP TB5 run" \
  -m "Confidence: medium" \
  -m "Scope-risk: moderate" \
  -m "Tested: two-node guard; tensor preview; bounded correctness requests; non-MTP bench; MTP bench; rollout report" \
  -m "Not-tested: Default-on behavior remains gated until the promotion report passes"
```

Gate: continue to Task 13 only when `mimo-v25-pro-mtp-rollout-report-20260519.json` has `"promotion_passed": true`.

---

### Task 13: Promote MTP To Default Single-Stream MiMo Runtime

**Files:**
- Modify: `src/exo/worker/engines/mlx/mimo_mtp/config.py`
- Create: `resources/inference_model_cards/kernelpool--MiMo-V2.5-Pro-6bit-mtp.toml`
- Modify: `src/exo/shared/models/model_cards.py`
- Create: `src/exo/shared/tests/test_mimo_v25_pro_mtp_model_card.py`
- Modify: `src/exo/worker/engines/mlx/tests/test_mimo_mtp_config.py`

- [ ] **Step 1: Write failing promotion tests**

Tests must require:

- `kernelpool/MiMo-V2.5-Pro-6bit-mtp` model card loads from the built-in registry.
- MTP card is text-only, supports tensor, and declares `quantization = "6bit-mlx-affine-mtp"`.
- MTP card storage size is `836959157867` bytes.
- MTP card capabilities include `mtp` and do not include media capabilities.
- `kernelpool/MiMo-V2.5-Pro-6bit` uses MTP by default after this task flips the code default, the sidecar artifact exists, and `EXO_MIMO_MTP_DISABLE` is not set.
- `EXO_MIMO_MTP=0` disables MTP for both base and promoted model ids.
- Missing sidecar artifact falls back to non-MTP when `EXO_MIMO_MTP_FAIL_CLOSED=0` and rejects the request when `EXO_MIMO_MTP_FAIL_CLOSED=1`.

Run:

```bash
uv run pytest src/exo/shared/tests/test_mimo_v25_pro_mtp_model_card.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_config.py -q
```

Expected:

- Tests fail because the promoted card and config are not implemented.

- [ ] **Step 2: Add promoted model card**

Create `resources/inference_model_cards/kernelpool--MiMo-V2.5-Pro-6bit-mtp.toml`:

```toml
# Kernelpool MLX affine 6-bit MiMo V2.5 Pro artifact with verified MTP sidecar.
model_id = "kernelpool/MiMo-V2.5-Pro-6bit-mtp"
n_layers = 70
hidden_size = 6144
num_key_value_heads = 8
architecture = "MiMoV2ForCausalLM"
supports_tensor = true
tasks = ["TextGeneration"]
family = "mimo"
quantization = "6bit-mlx-affine-mtp"
base_model = "XiaomiMiMo/MiMo-V2.5-Pro"
capabilities = [
  "text",
  "thinking",
  "thinking_toggle",
  "agentic",
  "long-context",
  "code",
  "mtp",
]

context_length = 1048576
trust_remote_code = true

[storage_size]
in_bytes = 836959157867
```

- [ ] **Step 3: Add model-card registry support**

In `src/exo/shared/models/model_cards.py`:

- Add `MIMO_V25_PRO_KERNELPOOL_6BIT_MTP_MODEL_ID`.
- Add it to `MIMO_V25_PRO_MODEL_IDS`.
- Keep text-only validation active for the MTP card.
- Allow `quantization = "6bit-mlx-affine-mtp"` for the promoted card.

- [ ] **Step 4: Flip default-on after the promotion report passes**

In `src/exo/worker/engines/mlx/mimo_mtp/config.py`:

- Keep env kill switch precedence: `EXO_MIMO_MTP=0` and `EXO_MIMO_MTP_DISABLE=1` disable MTP.
- Enable default-on for `kernelpool/MiMo-V2.5-Pro-6bit` in code only after Task 12 records `"promotion_passed": true`.
- Enable default-on for `kernelpool/MiMo-V2.5-Pro-6bit-mtp` when the sidecar exists and kill switch is absent.
- Preserve `EXO_MIMO_MTP=1` as an explicit opt-in override before promotion.
- Preserve `EXO_MIMO_MTP_FAIL_CLOSED=1` as the startup-safe default.

- [ ] **Step 5: Verify promotion tests**

Run:

```bash
uv run pytest src/exo/shared/tests/test_mimo_v25_pro_mtp_model_card.py src/exo/shared/tests/test_mimo_v25_pro_model_card.py src/exo/shared/tests/test_mimo_v25_pro_6bit_model_card.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_config.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_integration.py -q
uv run ruff check src/exo/shared/models/model_cards.py src/exo/shared/tests/test_mimo_v25_pro_mtp_model_card.py src/exo/worker/engines/mlx/mimo_mtp/config.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_config.py
uv run basedpyright src/exo/shared/models/model_cards.py src/exo/shared/tests/test_mimo_v25_pro_mtp_model_card.py src/exo/worker/engines/mlx/mimo_mtp/config.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_config.py
```

Expected:

- Promotion tests pass.
- Existing MiMo model-card tests continue to pass.
- `ruff` reports `All checks passed!`.
- `basedpyright` reports `0 errors, 0 warnings, 0 notes`.

- [ ] **Step 6: Commit promotion**

Run:

```bash
git add resources/inference_model_cards/kernelpool--MiMo-V2.5-Pro-6bit-mtp.toml src/exo/shared/models/model_cards.py src/exo/shared/tests/test_mimo_v25_pro_mtp_model_card.py src/exo/worker/engines/mlx/mimo_mtp/config.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_config.py
git commit -m "Promote verified MiMo MTP runtime" \
  -m "Make the verified MTP sidecar the default single-stream MiMo path after promotion evidence passes while retaining the non-MTP kill switch." \
  -m "Constraint: Default-on is gated by recorded TB5 correctness and throughput evidence" \
  -m "Rejected: Remove non-MTP fallback | rollback must be available without code changes" \
  -m "Confidence: medium" \
  -m "Scope-risk: broad" \
  -m "Tested: uv run pytest src/exo/shared/tests/test_mimo_v25_pro_mtp_model_card.py src/exo/shared/tests/test_mimo_v25_pro_model_card.py src/exo/shared/tests/test_mimo_v25_pro_6bit_model_card.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_config.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_integration.py -q; uv run ruff check src/exo/shared/models/model_cards.py src/exo/shared/tests/test_mimo_v25_pro_mtp_model_card.py src/exo/worker/engines/mlx/mimo_mtp/config.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_config.py; uv run basedpyright src/exo/shared/models/model_cards.py src/exo/shared/tests/test_mimo_v25_pro_mtp_model_card.py src/exo/worker/engines/mlx/mimo_mtp/config.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_config.py" \
  -m "Not-tested: Post-promotion live smoke remains next"
```

Gate: stop unless promotion tests pass and the promotion report already passed.

---

### Task 14: Post-Promotion Live Smoke And Rollback Proof

**Files:**
- Generate evidence under: `docs/plans/artifacts/`
- Update: `docs/plans/MIMO_V25_PRO_MTP_ROLLOUT_RUNLOG.md`

- [ ] **Step 1: Start promoted default-on TB5 runtime**

Run on Studio1 and Studio2:

```bash
EXO_MIMO_MTP_FAIL_CLOSED=1 \
EXO_MIMO_MTP_ARTIFACT_ROOT=/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental \
EXO_LIBP2P_NAMESPACE=mimo-mtp-promoted-20260519 \
EXO_FAST_SYNCH=1 \
EXO_OFFLINE=true \
uv run exo -vv
```

Expected:

- The base model id `kernelpool/MiMo-V2.5-Pro-6bit` selects MTP by default because promotion evidence passed.
- Only one distributed MiMo instance is created.

- [ ] **Step 2: Smoke base model id**

Run:

```bash
curl -sS -N -X POST http://127.0.0.1:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
    "messages": [{"role": "user", "content": "Give one sentence about safe default rollout."}],
    "temperature": 0,
    "max_tokens": 64,
    "stream": false
  }' \
  | tee docs/plans/artifacts/mimo-v25-pro-mtp-promoted-base-smoke-20260519.json
```

Expected:

- Response completes.
- Logs show `enabled=true`.
- No divergence is logged.

- [ ] **Step 3: Smoke promoted model id**

Run:

```bash
curl -sS -N -X POST http://127.0.0.1:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "kernelpool/MiMo-V2.5-Pro-6bit-mtp",
    "messages": [{"role": "user", "content": "Give one sentence about verified draft tokens."}],
    "temperature": 0,
    "max_tokens": 64,
    "stream": false
  }' \
  | tee docs/plans/artifacts/mimo-v25-pro-mtp-promoted-id-smoke-20260519.json
```

Expected:

- Response completes.
- Logs show `enabled=true`.
- No divergence is logged.

- [ ] **Step 4: Prove kill-switch rollback**

Restart both nodes with:

```bash
EXO_MIMO_MTP=0 \
EXO_MIMO_MTP_ARTIFACT_ROOT=/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental \
EXO_LIBP2P_NAMESPACE=mimo-mtp-rollback-20260519 \
EXO_FAST_SYNCH=1 \
EXO_OFFLINE=true \
uv run exo -vv
```

Run:

```bash
curl -sS -N -X POST http://127.0.0.1:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "kernelpool/MiMo-V2.5-Pro-6bit",
    "messages": [{"role": "user", "content": "Give one sentence about rollback."}],
    "temperature": 0,
    "max_tokens": 64,
    "stream": false
  }' \
  | tee docs/plans/artifacts/mimo-v25-pro-mtp-rollback-smoke-20260519.json
```

Expected:

- Response completes.
- Logs show MTP disabled by env.
- Existing non-MTP path remains functional.

- [ ] **Step 5: Commit smoke evidence**

Run:

```bash
git add docs/plans/artifacts/mimo-v25-pro-mtp-promoted-base-smoke-20260519.json docs/plans/artifacts/mimo-v25-pro-mtp-promoted-id-smoke-20260519.json docs/plans/artifacts/mimo-v25-pro-mtp-rollback-smoke-20260519.json docs/plans/MIMO_V25_PRO_MTP_ROLLOUT_RUNLOG.md
git commit -m "Smoke MiMo MTP default rollout" \
  -m "Verify promoted default-on MTP behavior and kill-switch rollback on the Studio1/Studio2 TB5 runtime." \
  -m "Constraint: Rollback must work without code changes or artifact mutation" \
  -m "Confidence: medium" \
  -m "Scope-risk: broad" \
  -m "Tested: default base model smoke; promoted model id smoke; EXO_MIMO_MTP=0 rollback smoke" \
  -m "Not-tested: Long-duration production soak"
```

Gate: rollout is complete only after default smoke and rollback smoke both pass.

---

### Task 15: Final Verification, Soak Decision, And Closeout

**Files:**
- Update: `docs/plans/MIMO_V25_PRO_MTP_ROLLOUT_RUNLOG.md`
- Write closeout: `/Users/studio2/AgentOutbox/2026-05-19-mimo-mtp-rollout-codex.md`

- [ ] **Step 1: Run final focused verification**

Run:

```bash
uv run pytest \
  scripts/test_mimo_v25_pro_runtime_guard.py \
  scripts/test_mimo_v25_pro_mtp_artifact_probe.py \
  scripts/test_mimo_v25_pro_mtp_quantize.py \
  scripts/test_mimo_v25_pro_mtp_module_probe.py \
  scripts/test_mimo_v25_pro_mtp_rollout_guard.py \
  scripts/test_mimo_v25_pro_mtp_bench_report.py \
  src/exo/shared/tests/test_mimo_v25_pro_model_card.py \
  src/exo/shared/tests/test_mimo_v25_pro_6bit_model_card.py \
  src/exo/shared/tests/test_mimo_v25_pro_mtp_model_card.py \
  src/exo/worker/engines/mlx/tests/test_mimo_mtp_config.py \
  src/exo/worker/engines/mlx/tests/test_mimo_mtp_artifact.py \
  src/exo/worker/engines/mlx/tests/test_mimo_mtp_modules.py \
  src/exo/worker/engines/mlx/tests/test_mimo_mtp_controller.py \
  src/exo/worker/engines/mlx/tests/test_mimo_mtp_generation.py \
  src/exo/worker/engines/mlx/tests/test_mimo_mtp_integration.py \
  src/exo/worker/engines/mlx/tests/test_mimo_mtp_sharding.py \
  -q
uv run ruff check scripts src/exo/shared/models/model_cards.py src/exo/shared/tests/test_mimo_v25_pro_mtp_model_card.py src/exo/worker/engines/mlx/mimo_mtp src/exo/worker/engines/mlx/generator/generate.py src/exo/worker/engines/mlx/generator/batch_generate.py src/exo/worker/engines/mlx/utils_mlx.py src/exo/worker/engines/mlx/auto_parallel.py
uv run basedpyright scripts/mimo_v25_pro_runtime_guard.py scripts/mimo_v25_pro_mtp_artifact_probe.py scripts/mimo_v25_pro_mtp_quantize.py scripts/mimo_v25_pro_mtp_module_probe.py scripts/mimo_v25_pro_mtp_rollout_guard.py scripts/mimo_v25_pro_mtp_bench_report.py src/exo/shared/models/model_cards.py src/exo/shared/tests/test_mimo_v25_pro_mtp_model_card.py src/exo/worker/engines/mlx/mimo_mtp src/exo/worker/engines/mlx/generator/generate.py src/exo/worker/engines/mlx/generator/batch_generate.py src/exo/worker/engines/mlx/utils_mlx.py src/exo/worker/engines/mlx/auto_parallel.py
```

Expected:

- Focused pytest passes.
- `ruff` reports `All checks passed!`.
- `basedpyright` reports `0 errors, 0 warnings, 0 notes`.

- [ ] **Step 2: Run final guard**

Run:

```bash
uv run python scripts/mimo_v25_pro_mtp_rollout_guard.py record-state --exo-url http://127.0.0.1:52415 --json-out docs/plans/artifacts/mimo-v25-pro-mtp-final-guard-20260519.json
```

Expected:

- JSON records final cluster state.
- Any unsafe state is documented in the runlog before closeout.

- [ ] **Step 3: Write runlog close**

Update `docs/plans/MIMO_V25_PRO_MTP_ROLLOUT_RUNLOG.md` with:

- Final commit list.
- Artifact paths.
- Guard evidence paths.
- Baseline and MTP benchmark report paths.
- Promotion result.
- Stretch result relative to `40 tok/s`.
- Rollback command:

```bash
EXO_MIMO_MTP=0 EXO_OFFLINE=true uv run exo -vv
```

- Soak recommendation:
  - `continue default rollout` when all gates pass and no instability occurred.
  - `keep opt-in only` when correctness passes but promotion report fails.
  - `disable MTP` when correctness, guard, or crash-free gates fail.

- [ ] **Step 4: Commit final evidence**

Run:

```bash
git add docs/plans/MIMO_V25_PRO_MTP_ROLLOUT_RUNLOG.md docs/plans/artifacts/mimo-v25-pro-mtp-final-guard-20260519.json
git commit -m "Close MiMo MTP rollout evidence" \
  -m "Record final verification, guard state, throughput result, and rollback instructions after the gated MTP rollout." \
  -m "Constraint: Future default changes must preserve EXO_MIMO_MTP=0 rollback" \
  -m "Confidence: medium" \
  -m "Scope-risk: narrow" \
  -m "Tested: final focused pytest; ruff; basedpyright; final rollout guard" \
  -m "Not-tested: Long-duration production soak remains a follow-up monitoring activity"
```

- [ ] **Step 5: Write durable closeout packet**

Create `/Users/studio2/AgentOutbox/2026-05-19-mimo-mtp-rollout-codex.md` with:

```md
# Closeout: MiMo MTP Rollout - gated implementation and rollout

- Date: 2026-05-19
- Agent: Codex Desktop/CLI on studio2
- Repo: /Users/studio2/exo
- Branch: con-75-mimo-pro-live-integration-20260504-200136
- GitHub PR:
- GitHub CI:
- Commit(s):
- Linear:
- Slack thread or Telegram DM:
- Obsidian project/context note:
- Status:

## Summary

Implemented and rolled out the gated MiMo V2.5 Pro MTP runtime path, or stopped at the first failed gate.

## Evidence

## Verification

## Decisions / Context

## Blockers

## Next Action

## Suggested Linear Comment

## Suggested Slack/Telegram Update
```

Fill every section with the final evidence gathered during execution.

Gate: do not claim rollout complete unless all tasks through Task 14 pass. When any gate fails, close out with the stopped gate, evidence, rollback state, and next action.
