# MiMo V2.5 Pro MTP Fastpath Single-Stream Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real exo/MLX native MiMo V2.5 Pro MTP fast path that can prove or disprove single-stream decode >=30 tok/s against the existing AR baseline.

**Architecture:** Keep the current exo MiMo/kernelpool AR path intact and add a guarded `mimo_mtp_fast` runtime beside it. The first slices load and validate the official/converted MTP sidecar, then build a one-cycle proposal/verify smoke before any broad exo API integration. Only after the hot path materially beats AR do we wire it into `mlx_generate` behind `EXO_MIMO_MTP=1`.

**Tech Stack:** Python 3.13, MLX, mlx-lm, safetensors, pytest, basedpyright, ruff, Beads recovery ledger.

**Worktree:** `/Users/studio2/exo/.worktrees/mimo-v25-pro-mtp-fastpath-single-stream-20260601`

**Beads recovery ledger:** Epic `mimo-v25-pro-mtp-d1-live-preflight-20260601-5ge`; tasks `.5ge.1` through `.5ge.5`.

---

## File Structure

Create a focused package:

- `src/exo/worker/engines/mlx/mimo_mtp_fast/__init__.py` — exports the fastpath public API.
- `src/exo/worker/engines/mlx/mimo_mtp_fast/sidecar_contract.py` — sidecar path resolution, safetensors header inspection, required key/dtype/shape contract, fail-closed probe result.
- `src/exo/worker/engines/mlx/mimo_mtp_fast/sidecar_loader.py` — materialized tensor loading and mapping from official `model.mtp.layers.{0,1,2}.*` names to local layer records.
- `src/exo/worker/engines/mlx/mimo_mtp_fast/sidecar_module.py` — minimal MTP layer/module forward path. This may reuse selected implementation from the parity worktree `mimo_mtp/modules.py`, but not the old controller.
- `src/exo/worker/engines/mlx/mimo_mtp_fast/one_cycle.py` — one-cycle D1/D2/D3 proposal/verify smoke that calls target model verification once for the candidate block.
- `src/exo/worker/engines/mlx/mimo_mtp_fast/speculative_loop.py` — streaming single-request loop after the one-cycle smoke proves executable.
- `src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py` — AR vs D1/D2/D3/auto benchmark runner helpers.

Modify existing code only at the guarded seams:

- `src/exo/worker/engines/mlx/generator/generate.py` — call the fastpath only when `EXO_MIMO_MTP=1`, model is MiMo, no vision media is active, and sidecar probe passes.
- `src/exo/worker/engines/mlx/utils_mlx.py` — if needed, attach model path/config metadata to loaded model object in a non-invasive way for sidecar discovery.

Tests live beside existing MLX tests:

- `src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_sidecar_contract.py`
- `src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_sidecar_loader.py`
- `src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_one_cycle.py`
- `src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_generator_guard.py`

Use tiny synthetic safetensors files for loader/contract tests. Do not require the 835GB/1TB model for unit tests.

---

## Task 1: Sidecar contract probe

**Beads:** `mimo-v25-pro-mtp-d1-live-preflight-20260601-5ge.1`

**Files:**
- Create: `src/exo/worker/engines/mlx/mimo_mtp_fast/__init__.py`
- Create: `src/exo/worker/engines/mlx/mimo_mtp_fast/sidecar_contract.py`
- Test: `src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_sidecar_contract.py`

- [ ] **Step 1: Write the failing contract tests**

Create `src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_sidecar_contract.py` with tests for three cases: missing file fails closed, a synthetic official-layout sidecar passes, and missing a required tensor reports the exact missing key.

The synthetic sidecar should create required tensors for 3 layers using small shapes that preserve rank and suffix names. Use `safetensors.numpy.save_file`; do not import MLX for this test.

- [ ] **Step 2: Run the contract tests and confirm failure**

Run:

```bash
uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_sidecar_contract.py -q
```

Expected before implementation: import failure for `exo.worker.engines.mlx.mimo_mtp_fast.sidecar_contract`.

- [ ] **Step 3: Implement `sidecar_contract.py`**

Implement these concrete API objects:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

MIMO_MTP_LAYER_COUNT: Final[int] = 3
MIMO_MTP_REQUIRED_SUFFIXES: Final[tuple[str, ...]] = (
    "eh_proj.weight",
    "enorm.weight",
    "hnorm.weight",
    "final_layernorm.weight",
    "input_layernorm.weight",
    "pre_mlp_layernorm.weight",
    "self_attn.attention_sink_bias",
    "self_attn.o_proj.weight",
    "self_attn.qkv_proj.weight",
    "self_attn.qkv_proj.weight_scale_inv",
    "mlp.down_proj.weight",
    "mlp.down_proj.weight_scale_inv",
    "mlp.gate_proj.weight",
    "mlp.gate_proj.weight_scale_inv",
    "mlp.up_proj.weight",
    "mlp.up_proj.weight_scale_inv",
)

MimoMtpSidecarStatus = Literal["ready", "missing", "invalid"]

@dataclass(frozen=True, slots=True)
class MimoMtpTensorSpec:
    key: str
    dtype: str
    shape: tuple[int, ...]

@dataclass(frozen=True, slots=True)
class MimoMtpSidecarProbe:
    path: Path
    status: MimoMtpSidecarStatus
    layer_count: int
    tensors: tuple[MimoMtpTensorSpec, ...]
    missing_keys: tuple[str, ...]
    error: str | None

    @property
    def ready(self) -> bool:
        return self.status == "ready"


def official_mimo_mtp_key(layer_index: int, suffix: str) -> str:
    return f"model.mtp.layers.{layer_index}.{suffix}"


def required_mimo_mtp_keys() -> tuple[str, ...]:
    return tuple(
        official_mimo_mtp_key(layer_index, suffix)
        for layer_index in range(MIMO_MTP_LAYER_COUNT)
        for suffix in MIMO_MTP_REQUIRED_SUFFIXES
    )


def probe_mimo_mtp_sidecar(path: str | Path) -> MimoMtpSidecarProbe:
    sidecar_path = Path(path).expanduser()
    if not sidecar_path.exists():
        return MimoMtpSidecarProbe(
            path=sidecar_path,
            status="missing",
            layer_count=0,
            tensors=(),
            missing_keys=required_mimo_mtp_keys(),
            error=f"MiMo MTP sidecar does not exist: {sidecar_path}",
        )

    try:
        from safetensors import safe_open

        tensor_specs: list[MimoMtpTensorSpec] = []
        with safe_open(str(sidecar_path), framework="np") as handle:
            keys = tuple(handle.keys())
            key_set = set(keys)
            for key in keys:
                tensor_slice = handle.get_slice(key)
                tensor_specs.append(
                    MimoMtpTensorSpec(
                        key=key,
                        dtype=str(tensor_slice.get_dtype()),
                        shape=tuple(int(dim) for dim in tensor_slice.get_shape()),
                    )
                )
    except Exception as exc:
        return MimoMtpSidecarProbe(
            path=sidecar_path,
            status="invalid",
            layer_count=0,
            tensors=(),
            missing_keys=required_mimo_mtp_keys(),
            error=f"Failed to inspect MiMo MTP sidecar {sidecar_path}: {exc}",
        )

    required_keys = required_mimo_mtp_keys()
    missing_keys = tuple(key for key in required_keys if key not in key_set)
    layer_count = sum(
        1
        for layer_index in range(MIMO_MTP_LAYER_COUNT)
        if any(key.startswith(f"model.mtp.layers.{layer_index}.") for key in key_set)
    )
    status: MimoMtpSidecarStatus = "ready" if not missing_keys and layer_count == MIMO_MTP_LAYER_COUNT else "invalid"
    return MimoMtpSidecarProbe(
        path=sidecar_path,
        status=status,
        layer_count=layer_count,
        tensors=tuple(sorted(tensor_specs, key=lambda spec: spec.key)),
        missing_keys=missing_keys,
        error=None if status == "ready" else "MiMo MTP sidecar is missing required official-layout tensors",
    )
```

`probe_mimo_mtp_sidecar` must read only safetensors metadata/header, never materialize the full 2.46GB sidecar in memory. Use `safetensors.safe_open(str(path), framework="np")` to enumerate keys, dtypes, and shapes.

- [ ] **Step 4: Run contract tests**

Run:

```bash
uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_sidecar_contract.py -q
```

Expected: tests pass.

- [ ] **Step 5: Run focused static checks**

Run:

```bash
uv run ruff check src/exo/worker/engines/mlx/mimo_mtp_fast/sidecar_contract.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_sidecar_contract.py
uv run basedpyright src/exo/worker/engines/mlx/mimo_mtp_fast/sidecar_contract.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_sidecar_contract.py
```

Expected: ruff passes; basedpyright has 0 errors for these files.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/exo/worker/engines/mlx/mimo_mtp_fast/__init__.py src/exo/worker/engines/mlx/mimo_mtp_fast/sidecar_contract.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_sidecar_contract.py
git commit -m "feat: probe MiMo MTP sidecar contract"
```

Update Beads task `.5ge.1` with the current implementation milestone and next coding step.

---

## Task 2: Sidecar tensor loader

**Beads:** `mimo-v25-pro-mtp-d1-live-preflight-20260601-5ge.1`

**Files:**
- Create: `src/exo/worker/engines/mlx/mimo_mtp_fast/sidecar_loader.py`
- Test: `src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_sidecar_loader.py`

- [ ] **Step 1: Write failing loader tests**

Create tests that write a tiny 3-layer official-layout safetensors file and assert that `load_mimo_mtp_sidecar_tensors(path)` returns three layer records, each with named tensor attributes for `eh_proj_weight`, `qkv_proj_weight`, `qkv_proj_scale_inv`, `gate_proj_weight`, `up_proj_weight`, and `down_proj_weight`.

- [ ] **Step 2: Run loader tests and confirm failure**

```bash
uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_sidecar_loader.py -q
```

Expected before implementation: import failure for `sidecar_loader`.

- [ ] **Step 3: Implement `sidecar_loader.py`**

Implement frozen dataclasses:

```python
@dataclass(frozen=True, slots=True)
class MimoMtpLayerTensors:
    layer_index: int
    tensors: Mapping[str, mx.array]

    def get(self, suffix: str) -> mx.array:
        try:
            return self.tensors[suffix]
        except KeyError as exc:
            raise KeyError(f"MiMo MTP layer {self.layer_index} missing tensor suffix {suffix!r}") from exc

@dataclass(frozen=True, slots=True)
class MimoMtpSidecarTensors:
    path: Path
    layers: tuple[MimoMtpLayerTensors, ...]
```

Implement:

```python
class MimoMtpSidecarLoadError(RuntimeError):
    pass


def _local_suffix_for_key(layer_index: int, key: str) -> str:
    prefix = f"model.mtp.layers.{layer_index}."
    if not key.startswith(prefix):
        raise ValueError(f"Expected key for layer {layer_index}, got {key!r}")
    return key.removeprefix(prefix)


def load_mimo_mtp_sidecar_tensors(path: str | Path) -> MimoMtpSidecarTensors:
    probe = probe_mimo_mtp_sidecar(path)
    if not probe.ready:
        raise MimoMtpSidecarLoadError(
            f"MiMo MTP sidecar is not ready: status={probe.status} missing={list(probe.missing_keys[:5])} error={probe.error}"
        )

    sidecar_path = Path(path).expanduser()
    loaded = mx.load(str(sidecar_path))
    layers: list[MimoMtpLayerTensors] = []
    for layer_index in range(MIMO_MTP_LAYER_COUNT):
        layer_tensors: dict[str, mx.array] = {}
        for suffix in MIMO_MTP_REQUIRED_SUFFIXES:
            official_key = official_mimo_mtp_key(layer_index, suffix)
            tensor = loaded[official_key]
            layer_tensors[_local_suffix_for_key(layer_index, official_key)] = tensor
        layers.append(MimoMtpLayerTensors(layer_index=layer_index, tensors=layer_tensors))
    return MimoMtpSidecarTensors(path=sidecar_path, layers=tuple(layers))
```

The loader may materialize the sidecar because this is runtime loading. It must refuse missing/invalid probes before `mx.load`.

- [ ] **Step 4: Run loader tests and checks**

```bash
uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_sidecar_loader.py -q
uv run ruff check src/exo/worker/engines/mlx/mimo_mtp_fast/sidecar_loader.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_sidecar_loader.py
uv run basedpyright src/exo/worker/engines/mlx/mimo_mtp_fast/sidecar_loader.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_sidecar_loader.py
```

Expected: tests and checks pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/exo/worker/engines/mlx/mimo_mtp_fast/sidecar_loader.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_sidecar_loader.py
git commit -m "feat: load MiMo MTP sidecar tensors"
```

---

## Task 3: One-cycle proposal/verify smoke

**Beads:** `mimo-v25-pro-mtp-d1-live-preflight-20260601-5ge.2`

**Files:**
- Create: `src/exo/worker/engines/mlx/mimo_mtp_fast/sidecar_module.py`
- Create: `src/exo/worker/engines/mlx/mimo_mtp_fast/one_cycle.py`
- Test: `src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_one_cycle.py`

- [ ] **Step 1: Port only the necessary module pieces from parity code**

Use the parity worktree file:

```text
/Users/studio2/exo/.worktrees/mimo-v25-pro-mtp-d1-parity-20260531/src/exo/worker/engines/mlx/mimo_mtp/modules.py
```

Port only:

- quantized affine linear helper
- MTP attention shape splitting
- MTP MLP
- MTP layer forward
- MTP stack `propose` equivalent

Do not port:

- `MimoMtpDecodeController`
- Ralph trace hooks
- projection audit/repair hooks
- parity rollback machinery
- broad diagnostics paths

- [ ] **Step 2: Write the fake-model one-cycle tests**

Create a fake target model and fake MTP stack that operate on tiny logits. Test that `run_mimo_mtp_one_cycle`:

- receives the current token/history,
- asks the MTP stack for up to requested depth proposals,
- verifies candidate block with target model logits,
- returns accepted prefix length and next token,
- does not mutate external cache objects in the fake test.

- [ ] **Step 3: Implement `one_cycle.py` API**

Implement:

```python
@dataclass(frozen=True, slots=True)
class MimoMtpOneCycleResult:
    proposed_token_ids: tuple[int, ...]
    accepted_token_ids: tuple[int, ...]
    fallback_token_id: int | None
    attempted_depth: int
    accepted_depth: int
    elapsed_seconds: float


def run_mimo_mtp_one_cycle(
    *,
    token_history: tuple[int, ...],
    requested_depth: int,
    proposal_token_ids: tuple[int, ...],
    verifier_token_ids: tuple[int, ...],
) -> MimoMtpOneCycleResult:
    start = time.perf_counter()
    attempted_depth = max(1, min(requested_depth, len(proposal_token_ids)))
    proposed = proposal_token_ids[:attempted_depth]
    accepted: list[int] = []
    for proposed_token, verifier_token in zip(proposed, verifier_token_ids, strict=False):
        if proposed_token != verifier_token:
            break
        accepted.append(proposed_token)
    fallback = None if len(accepted) == attempted_depth else verifier_token_ids[len(accepted)]
    return MimoMtpOneCycleResult(
        proposed_token_ids=proposed,
        accepted_token_ids=tuple(accepted),
        fallback_token_id=fallback,
        attempted_depth=attempted_depth,
        accepted_depth=len(accepted),
        elapsed_seconds=time.perf_counter() - start,
    )
```

The plan uses this fake-token implementation only for the first tests. The production implementation must replace `proposal_token_ids` and `verifier_token_ids` with callables that run the MTP stack and target verifier against MLX caches. The first production implementation may support greedy/top-1 acceptance only. Sampling probability-ratio acceptance is deferred until greedy fastpath is executable.

- [ ] **Step 4: Run one-cycle tests and checks**

```bash
uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_one_cycle.py -q
uv run ruff check src/exo/worker/engines/mlx/mimo_mtp_fast src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_one_cycle.py
uv run basedpyright src/exo/worker/engines/mlx/mimo_mtp_fast src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_one_cycle.py
```

Expected: focused tests/checks pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/exo/worker/engines/mlx/mimo_mtp_fast/sidecar_module.py src/exo/worker/engines/mlx/mimo_mtp_fast/one_cycle.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_one_cycle.py
git commit -m "feat: add MiMo MTP one-cycle proposal verifier"
```

Update Beads `.5ge.2` with whether the old parity controller was avoided and what the next concrete hot-path coding step is.

---

## Task 4: Streaming speculative loop

**Beads:** `mimo-v25-pro-mtp-d1-live-preflight-20260601-5ge.3`

**Files:**
- Create: `src/exo/worker/engines/mlx/mimo_mtp_fast/speculative_loop.py`
- Test: extend `src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_one_cycle.py` or create `test_mimo_mtp_fast_speculative_loop.py`

- [ ] **Step 1: Write fake-model streaming tests**

Test that the loop can generate a requested number of tokens using a fake target verifier and fake MTP stack, falls back to AR when proposals are rejected, and does not enter the old parity controller path.

- [ ] **Step 2: Implement loop API**

Implement:

```python
@dataclass(frozen=True, slots=True)
class MimoMtpStreamEvent:
    token_id: int
    accepted_depth: int
    attempted_depth: int
    used_fallback: bool


def stream_mimo_mtp_fast(
    *,
    token_history: tuple[int, ...],
    max_tokens: int,
    requested_depth: int,
    one_cycle: Callable[[tuple[int, ...], int], MimoMtpOneCycleResult],
) -> Iterator[MimoMtpStreamEvent]:
    history = list(token_history)
    emitted = 0
    while emitted < max_tokens:
        result = one_cycle(tuple(history), requested_depth)
        output_tokens = result.accepted_token_ids
        if not output_tokens and result.fallback_token_id is not None:
            output_tokens = (result.fallback_token_id,)
        if not output_tokens:
            break
        for token_id in output_tokens:
            history.append(token_id)
            emitted += 1
            yield MimoMtpStreamEvent(
                token_id=token_id,
                accepted_depth=result.accepted_depth,
                attempted_depth=result.attempted_depth,
                used_fallback=result.fallback_token_id == token_id and result.accepted_depth == 0,
            )
            if emitted >= max_tokens:
                break
```

The plan uses this callable-driven loop for fake-model tests. The production loop must pass a callable backed by the MLX MTP proposal path and target verifier. Keep the first loop single-stream only. Reject batch, vision, and missing-sidecar inputs by returning `None` from a guard function rather than throwing inside normal AR generation.

- [ ] **Step 3: Run loop tests and checks**

```bash
uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_speculative_loop.py -q
uv run ruff check src/exo/worker/engines/mlx/mimo_mtp_fast src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_speculative_loop.py
uv run basedpyright src/exo/worker/engines/mlx/mimo_mtp_fast src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_speculative_loop.py
```

- [ ] **Step 4: Commit Task 4**

```bash
git add src/exo/worker/engines/mlx/mimo_mtp_fast/speculative_loop.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_speculative_loop.py
git commit -m "feat: stream MiMo MTP fastpath tokens"
```

---

## Task 5: Local AR vs MTP benchmark harness

**Beads:** `mimo-v25-pro-mtp-d1-live-preflight-20260601-5ge.4`

**Files:**
- Create: `src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py`
- Create: `scripts/bench_mimo_mtp_fastpath.py`

- [ ] **Step 1: Implement benchmark runner**

The script must accept explicit model and sidecar paths. Use environment variables so a real operator supplies local snapshot paths rather than hard-coding machine-specific locations in the script:

```bash
MIMO_MODEL_PATH=/Users/studio2/.cache/huggingface/hub/models--kernelpool--MiMo-V2.5-Pro-6bit/snapshots/local-snapshot \
MIMO_MTP_SIDECAR=/Users/studio2/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.5-Pro/snapshots/local-snapshot/model_mtp.safetensors \
uv run python3 scripts/bench_mimo_mtp_fastpath.py \
  --model-path "$MIMO_MODEL_PATH" \
  --sidecar-path "$MIMO_MTP_SIDECAR" \
  --prompt "Write a Python function that parses JSON lines." \
  --max-tokens 128 \
  --modes ar,d1,d2,d3,auto
```

It must print JSON lines with mode, generated token count, decode seconds, decode tok/s, attempted depth counts, accepted depth counts, and the next implementation conclusion string.

- [ ] **Step 2: Add no-model dry-run mode**

Add `--dry-run-contract-only` to probe the sidecar and print model/sidecar readiness without loading the full model.

- [ ] **Step 3: Run dry-run benchmark probe**

```bash
MIMO_MTP_SIDECAR=/Users/studio2/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.5-Pro/snapshots/local-snapshot/model_mtp.safetensors \
uv run python3 scripts/bench_mimo_mtp_fastpath.py --sidecar-path "$MIMO_MTP_SIDECAR" --dry-run-contract-only
```

Expected with a real sidecar: JSON `ready: true`. Expected without a sidecar: JSON `ready: false` with missing-file error. Do not fake readiness.

- [ ] **Step 4: Commit Task 5**

```bash
git add src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py scripts/bench_mimo_mtp_fastpath.py
git commit -m "feat: benchmark MiMo MTP fastpath modes"
```

Update Beads `.5ge.4` with the next coding move indicated by the benchmark, not just the number.

---

## Task 6: Guarded exo generator integration

**Beads:** `mimo-v25-pro-mtp-d1-live-preflight-20260601-5ge.5`

**Files:**
- Modify: `src/exo/worker/engines/mlx/generator/generate.py`
- Possibly modify: `src/exo/worker/engines/mlx/utils_mlx.py`
- Test: `src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_generator_guard.py`

- [ ] **Step 1: Write guard tests**

Test that the fastpath is not used unless all of these are true:

- `EXO_MIMO_MTP=1`
- model id or config indicates MiMo V2.5 Pro
- sidecar probe is ready
- no vision media regions are active
- generation is single-stream

Test that AR fallback remains the default.

- [ ] **Step 2: Add guard function**

Implement a small function in `generate.py` or a new guard module:

```python
def should_use_mimo_mtp_fastpath(
    *,
    model_id: ModelId,
    task_params: TextGenerationTaskParams,
    sidecar_probe: MimoMtpSidecarProbe | None,
    media_regions: Sequence[MediaRegion],
) -> bool:
    env_enabled = os.environ.get("EXO_MIMO_MTP", "").strip().lower() in {"1", "true", "yes", "on"}
    if not env_enabled:
        return False
    model_text = str(model_id).lower()
    if "mimo-v2.5-pro" not in model_text and "mimo" not in model_text:
        return False
    if sidecar_probe is None or not sidecar_probe.ready:
        return False
    if media_regions:
        return False
    if task_params.image_count or task_params.videos or task_params.video_urls:
        return False
    return True
```

The function must return `False` for any uncertain condition.

- [ ] **Step 3: Wire fastpath into `mlx_generate`**

At the seam after prefill and before `stream_generate(...)`, route to `stream_mimo_mtp_fast(...)` only when the guard returns true. Keep existing `stream_generate` untouched as fallback.

- [ ] **Step 4: Run focused generator guard tests**

```bash
uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_generator_guard.py -q
uv run ruff check src/exo/worker/engines/mlx/generator/generate.py src/exo/worker/engines/mlx/mimo_mtp_fast src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_generator_guard.py
uv run basedpyright src/exo/worker/engines/mlx/generator/generate.py src/exo/worker/engines/mlx/mimo_mtp_fast src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_generator_guard.py
```

- [ ] **Step 5: Commit Task 6**

```bash
git add src/exo/worker/engines/mlx/generator/generate.py src/exo/worker/engines/mlx/mimo_mtp_fast src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_generator_guard.py
git commit -m "feat: guard MiMo MTP fastpath in mlx generation"
```

---

## Final verification for this plan

Run these before claiming completion of the implementation branch:

```bash
uv run ruff check src/exo/worker/engines/mlx/mimo_mtp_fast scripts/bench_mimo_mtp_fastpath.py
uv run basedpyright src/exo/worker/engines/mlx/mimo_mtp_fast scripts/bench_mimo_mtp_fastpath.py
uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_sidecar_contract.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_sidecar_loader.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_one_cycle.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_speculative_loop.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_generator_guard.py -q
git diff --check
```

Live/full-model benchmark command is allowed only when the sidecar/model paths are known and the user wants to run the large model:

```bash
MIMO_MODEL_PATH=/Users/studio2/.cache/huggingface/hub/models--kernelpool--MiMo-V2.5-Pro-6bit/snapshots/local-snapshot \
MIMO_MTP_SIDECAR=/Users/studio2/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.5-Pro/snapshots/local-snapshot/model_mtp.safetensors \
uv run python3 scripts/bench_mimo_mtp_fastpath.py \
  --model-path "$MIMO_MODEL_PATH" \
  --sidecar-path "$MIMO_MTP_SIDECAR" \
  --prompt "Write a Python function that parses JSON lines." \
  --max-tokens 128 \
  --modes ar,d1,d2,d3,auto
```

Completion criterion is not merely tests passing. The branch must either:

1. demonstrate a real single-stream MTP speed win and proceed to guarded exo integration, or
2. identify the concrete implementation/architecture blocker preventing MiMo MTP from beating AR.
