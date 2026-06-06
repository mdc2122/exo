# MiMo MTP Fastpath Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden MiMo V2.5 Pro native MTP fastpath semantics, diagnostics, optimization readiness, and benchmark readiness without live full-model loading or production/default enablement.

**Architecture:** Extend the existing isolated `src/exo/worker/engines/mlx/mimo_mtp_fast/` modules with fastpath-specific validation, timing diagnostics, trace events, and synthetic benchmark probes. Keep provider seams callable against production-shaped MLX/MimoMtpStack APIs while tests use fake/tiny providers only.

**Tech Stack:** Python 3.12+, pytest, MLX, safetensors, ruff, basedpyright.

---

### Task 1: Sidecar contract guards

**Files:**
- Modify: `src/exo/worker/engines/mlx/mimo_mtp_fast/sidecar_contract.py`
- Modify: `src/exo/worker/engines/mlx/mimo_mtp_fast/sidecar_loader.py`
- Modify/Test: `src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_sidecar_contract.py`
- Modify/Test: `src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_sidecar_loader.py`
- Modify/Test: `src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_sidecar_module.py`

- [x] Add failing tests for malformed required tensor shape/dtype messages and grouped-QKV split behavior.
- [x] Run focused tests and confirm the new tests fail for missing validation/export.
- [x] Add lightweight contract validation that reports exact tensor key, expected role, actual dtype, and shape.
- [x] Run sidecar contract/loader/module tests.

### Task 2: One-cycle and provider seam validation

**Files:**
- Modify: `src/exo/worker/engines/mlx/mimo_mtp_fast/one_cycle.py`
- Modify: `src/exo/worker/engines/mlx/mimo_mtp_fast/providers.py`
- Modify/Test: `src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_one_cycle.py`
- Modify/Test: `src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_providers.py`

- [x] Add failing tests for all-accepted, partial, none, empty, verifier-short output, D1/D2/D3 determinism, and invalid provider outputs.
- [x] Run focused tests and confirm failures for new validation/diagnostic fields where absent.
- [x] Add fastpath-specific errors and timing breakdown fields while preserving existing call order.
- [x] Run focused one-cycle/provider tests.

### Task 3: Streaming loop trace and failure semantics

**Files:**
- Modify: `src/exo/worker/engines/mlx/mimo_mtp_fast/speculative_loop.py`
- Modify/Test: `src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_speculative_loop.py`

- [x] Add failing tests for trace collector, max token exactness, fallback classification, fail-closed classification, history immutability, and stale MTP cache reset after structural failure.
- [x] Add trace dataclass, failure policy, provider failure classification, and optional cache reset hook.
- [x] Run focused speculative-loop tests.

### Task 4: Benchmark diagnostics and synthetic probe

**Files:**
- Modify: `src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py`
- Modify: `scripts/bench_mimo_mtp_fastpath.py`
- Create: `scripts/mimo_mtp_fastpath_synthetic_probe.py`
- Create/Test: `scripts/test_mimo_mtp_fastpath_synthetic_probe.py`
- Modify/Test: `src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_benchmark.py`

- [x] Add failing tests for JSON timing/acceptance summaries and clear CLI validation messages.
- [x] Add timing breakdown propagation and acceptance-rate helper fields.
- [x] Add synthetic/tiny probe that reports JSON diagnostics without full model load.
- [x] Run focused benchmark and script tests.

### Task 5: Verification and ledger closeout

**Files:**
- Modify: `docs/plans/MIMO_V25_PRO_MTP_FOUNDATION_RUNLOG.md` or `docs/plans/MIMO_V25_PRO_MTP_ARTIFACT_RUNLOG.md`
- Modify: `.ouroboros/seeds/mimo-v25-pro-mtp-fastpath-readiness-20260606.yaml` only if it already records run status and not the immutable Seed body.

- [x] Run `uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast*.py scripts/test_mimo_mtp_fastpath_synthetic_probe.py`.
- [x] Run `uv run ruff check` on touched fastpath/script surfaces.
- [x] Run `uv run basedpyright` on touched fastpath/script surfaces where supported; otherwise record full command blocker.
- [x] Run `git diff --check`.
- [x] Update run ledger with tests, blockers, and explicit performance gate status: benchmark-ready code only; no speedup claim and no Slice 5 production/default enablement.


## Stabilization Result

- Ouroboros/Goose execution completed successfully at runtime level for session `orch_aeef7b6b4c7c`, but post-execution QA returned `revise` because canonical full `uv run pytest` still has unrelated/deferred TurboQuant failures and Nix build tooling was not available/complete in that QA pass.
- Manual focused stabilization passed after formatting: `uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast*.py scripts/test_bench_mimo_mtp_fastpath.py scripts/test_mimo_mtp_fastpath_synthetic_probe.py -q` -> `61 passed in 2.40s`.
- Manual lint/type/format checks passed on touched fastpath/script surfaces: `uv run ruff check ...` -> all checks passed; `uv run ruff format --check ...` -> 19 files already formatted; `uv run basedpyright ...` -> 0 errors, 0 warnings, 0 notes; `git diff --check` -> clean.
- Smoke checks passed: official sidecar dry-run returned `ready=true`; synthetic probe emitted JSON with `full_model_loaded=false`; missing model validation returned JSON with exit code `2` before sidecar/model load.
- Full `uv run pytest -q` was intentionally not made a blocker for this fastpath-readiness stabilization because it fails in four existing/deferred TurboQuant tests outside the touched MiMo MTP fastpath surfaces.
- `nix --extra-experimental-features 'nix-command flakes' fmt` was attempted twice; it failed while building formatter dependencies due local disk exhaustion (`No space left on device`). After Nix GC, focused Python formatting was completed with `ruff format` and verified with `ruff format --check`.
- No live full-model load, exo startup, MiMo cluster startup, AR-vs-MTP benchmark, or Slice 5 production/default integration was performed.
