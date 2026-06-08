# MiMo MTP Sidecar Lifecycle

This document describes the lifecycle of the MiMo V2.5 Pro MTP sidecar within exo,
covering setup, caching, teardown expectations, and the invariants that tests must verify.

## Overview

The MiMo MTP sidecar is an auxiliary safetensors file (`model_mtp.safetensors`) that
contains the multi-token-prediction (MTP) head layers for the MiMo V2.5 Pro model. It
enables speculative decoding: the MTP stack proposes draft tokens and the base model
verifies them, potentially emitting multiple tokens per generation cycle.

## Lifecycle Phases

### 1. Probe (contract validation, no tensor materialization)

**Component:** `probe_mimo_mtp_sidecar` (`sidecar_contract.py`)

- Reads the safetensors file metadata without loading tensors into MLX memory.
- Validates: file existence, required key layout (`model.mtp.layers.{0,1,2}.*`),
  tensor dtype/shape contract for each required key.
- Returns `MimoMtpSidecarProbe` with `status` in `{"ready", "missing", "invalid"}`.
- **Cost:** Cheap. Only reads metadata, no GPU/memory allocation.
- **When called:** On every `evaluate_mimo_mtp_worker_fastpath` invocation (the fastpath
  decision gate probes before loading). This is acceptable because probing is O(keys)
  with no tensor materialization.

### 2. Load (tensor materialization, must be cached)

**Component:** `load_mimo_mtp_sidecar_tensors` (`sidecar_loader.py`)

- Loads all sidecar tensors from the safetensors file into MLX arrays via `mx.load`.
- Validates readiness via `probe_mimo_mtp_sidecar` first; raises
  `MimoMtpSidecarLoadError` if the sidecar is not ready.
- Returns `MimoMtpSidecarTensors` containing per-layer tensor mappings.
- **Cost:** Expensive. Reads ~2.3 GB from disk, allocates MLX memory.
- **Cache invariant:** The sidecar must be loaded **at most once per sidecar path**
  during a runner process lifetime. The `MimoMtpWorkerFastpathCache` enforces this.

### 3. Build (MTP stack construction, must be cached)

**Component:** `build_mimo_mtp_stack` (`sidecar_module.py`)

- Constructs `MimoMtpStack` from `MimoMtpSidecarTensors` and the base model.
- Builds `MimoMtpLayer` instances with dequantized FP8 weights, RMSNorm layers,
  attention modules, and MLP modules.
- The dequantization in `Fp8BlockLinear.__init__` happens once at construction time;
  subsequent `__call__` invocations use the cached dequantized weight.
- **Cost:** Moderate. Involves MLX module construction and FP8→BF16 dequantization.
- **Cache invariant:** The MTP stack should be built once per model load, not per
  generation call or per token. Currently the stack is not separately cached beyond
  the sidecar tensors; wiring the stack build into the fastpath cache is a future
  integration point.

### 4. Generation (per-stream MTP cache, reset between streams)

**Component:** `MimoMtpCache` / `MimoMtpStack.make_cache()` (`sidecar_module.py`)

- Each MTP generation stream creates its own `MimoMtpCache` via `stack.make_cache()`.
- The `MimoMtpCache` contains per-layer KV caches (`MimoMtpLayerCache`) that
  accumulate keys/values across MTP cycles within a single stream.
- `MimoMtpCache.reset()` clears all layer caches; it is called on fallback events
  via `MimoMtpStack.reset_cache_on_fallback`.
- **Cost:** Lightweight per-stream allocation; no disk I/O.
- **Lifecycle:** Created fresh per stream, reset on fallback, discarded after stream
  ends. Never reused across streams.

### 5. Teardown (runner process exit)

- The `MimoMtpWorkerFastpathCache` holds sidecar tensors for the runner's lifetime.
- Tensors are released when the runner process exits (Python GC + MLX memory
  management).
- No explicit teardown step is required; the sidecar cache does not hold external
  resources (file handles, locks, etc.) beyond MLX arrays.

## Caching Invariants

These invariants are the core of the sidecar lifecycle contract and must be verified
by tests:

| Invariant | Scope | Verification |
|-----------|-------|-------------|
| Sidecar load count ≤ 1 per path | Runner process lifetime | `MimoMtpWorkerFastpathCache.load_count_for_path(path) == 1` after multiple `evaluate_mimo_mtp_worker_fastpath` calls |
| Sidecar tensors are identity-equal across calls | Runner process lifetime | `decision.sidecar is first_decision.sidecar` |
| FP8 dequantization happens once per `Fp8BlockLinear` | Module construction | `MX_FROM_FP8` call count == 1 per `Fp8BlockLinear` instance, regardless of `__call__` count |
| MTP cache is fresh per stream | Generation stream | `stack.make_cache()` returns a new `MimoMtpCache` each call |
| MTP cache is reset on fallback | Generation stream | `reset_cache_on_fallback` clears all layer offsets/entries |

## Per-Phase Call Frequency

```
Phase         Called per              Disk I/O    MLX alloc    Must cache
─────────────────────────────────────────────────────────────────────────
probe         fastpath eval           metadata    none         no (cheap)
load          fastpath eval (once)    ~2.3 GB     yes          YES
build         model load (once)       none        moderate     recommended
make_cache    per stream              none        lightweight  no (per-stream)
reset_cache   on fallback             none        none         n/a
```

## Unsupported Lifecycle States and Telemetry

When the sidecar lifecycle encounters an unsupported or invalid state, the system
emits structured telemetry via the `MimoMtpWorkerFastpathDecision.telemetry` mapping
and/or raises typed errors:

| State | Telemetry / Error | Signal |
|-------|-------------------|--------|
| `missing` sidecar | `sidecar_status="missing"`, `disable_reason="missing_sidecar"` | Sidecar file does not exist on disk |
| `invalid` sidecar | `sidecar_status="invalid"`, `disable_reason="invalid_sidecar"`, `contract_errors=[...]` | Sidecar exists but fails tensor contract |
| `unsupported_model` | `disable_reason="unsupported_model"` | Requested model is not in `MIMO_V25_PRO_MODEL_IDS` |
| `native_runtime_disabled` | `disable_reason="mimo_mtp_native_runtime_disabled"` | `EXO_MIMO_MTP_NATIVE_RUNTIME` env var is not set |
| `unwired_execution` | `disable_reason="mimo_mtp_distributed_generator_unwired"` | Generator seam not yet wired for MTP |
| `load_error` | `MimoMtpSidecarLoadError` raised, caught → `disable_reason="invalid_sidecar"` | Sidecar file corrupt or unreadable |
| `fail_closed` | `accepted_execution_path="rejected"`, `reject_reason=<disable_reason>` | Request explicitly asked for fail-closed behavior |
| `fail_open` | `accepted_execution_path="ar"`, `fallback_reason="fail_open_<reason>"` | Request allows fallback to AR |

All telemetry is logged via `_log_mimo_mtp_worker_decision` in `batch_generator.py`
when a disable reason is present or the execution path is not default AR.

## Architecture Diagram

```
                    ┌──────────────────────────┐
                    │  API Request             │
                    │  (mimo_mtp_fastpath      │
                    │   params optional)       │
                    └──────────┬───────────────┘
                               │
                    ┌──────────▼───────────────┐
                    │  evaluate_mimo_mtp_      │
                    │  worker_fastpath()       │
                    │                          │
                    │  1. Probe sidecar        │
                    │  2. Check model/runtime  │
                    │  3. Load sidecar (cache) │
                    └──────────┬───────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
     ┌────────▼──────┐ ┌──────▼───────┐ ┌──────▼──────┐
     │  AR default   │ │  Rejected    │ │  MTP path   │
     │  (no sidecar) │ │  (fail-      │ │  (sidecar   │
     │               │ │   closed)    │ │   loaded)   │
     └───────────────┘ └──────────────┘ └──────┬──────┘
                                            │
                                 ┌──────────▼──────────┐
                                 │  build_mimo_mtp_    │
                                 │  stack()            │
                                 │  (once per model)   │
                                 └──────────┬──────────┘
                                            │
                                 ┌──────────▼──────────┐
                                 │  MTP generation     │
                                 │  stack.make_cache() │
                                 │  stack.propose()    │
                                 │  verifier matches   │
                                 └─────────────────────┘
```

## Test Coverage Requirements

The following test scenarios must exist for the sidecar lifecycle:

1. **Cache-once probe:** Multiple `evaluate_mimo_mtp_worker_fastpath` calls with the
   same sidecar path must result in `load_count_for_path == 1`.
2. **Identity-equality:** Sidecar tensors returned from the cache must be the same
   object (`is`) across multiple fastpath evaluations.
3. **FP8 dequantization once:** `Fp8BlockLinear` must dequantize exactly once at
   construction, not per forward call. (Already covered by
   `test_fp8_block_linear_dequantizes_once_at_construction`.)
4. **Per-stream MTP cache:** `stack.make_cache()` must return distinct objects per
   call; the cache must not be shared across streams.
5. **Unsupported state telemetry:** Each unsupported lifecycle state must emit
   structured telemetry with `mtp_execution_state`, `mtp_disable_reason`, and/or
   `mtp_fallback_reason` fields.
6. **Build-count invariant:** `build_mimo_mtp_stack` must be called at most once per
   model load, not per generation cycle. (This invariant should be tested when the
   build step is integrated into the fastpath cache.)

## Future Integration Notes

- The `build_mimo_mtp_stack` step is not yet cached in `MimoMtpWorkerFastpathCache`.
  When the distributed MTP generator seam is wired, the stack should be built once
  and cached alongside the sidecar tensors.
- The `MimoMtpCache` per-stream lifecycle must be respected when wiring the MTP
  fastpath into the batch generator: each stream gets its own cache, and fallback
  events reset the cache for that stream only.
