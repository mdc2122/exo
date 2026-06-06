# MiMo MTP benchmark survival diagnostics - 2026-06-06

This note records execution of the benchmark-survival acceptance criteria for the
MiMo V2.5 Pro native MTP fastpath. It is intentionally not a speed claim: the
local machine still cannot materialize the full quantized MiMo model before the
OS kills the process.

## Acceptance criteria status

- AC1 benchmark CLI exposes --preflight-only and --load-only modes: done.
- AC2 benchmark emits structured stage/timing/memory diagnostics: done. The CLI
  emits JSON-lines benchmark_stage rows for mode validation, model path
  validation, sidecar contract validation, base model materialization, tokenizer
  load, sidecar tensor load, sidecar stack build, prompt tokenization, and
  generation.
- AC3 sidecar contract and synthetic probe remain green: done in focused checks.
- AC4 missing model/sidecar validation remains fail-fast: done in focused checks.
- AC5 smallest AR live row attempted and recorded: blocked by load-only model
  materialization kill before generation could start.
- AC6 smallest MTP live row attempted if AR load succeeds: skipped because AR
  load did not succeed.
- AC7 Slice 5 remains blocked unless real AR-vs-MTP win exists: still blocked.

## Local artifact paths

- Model: /Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit
- Sidecar: /Volumes/GLM5-NVMe/exo/mimo-v25-pro/hf/XiaomiMiMo--MiMo-V2.5-Pro/model_mtp.safetensors

## Evidence

Preflight-only command:

uv run python scripts/bench_mimo_mtp_fastpath.py \
  --model-path /Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit \
  --sidecar-path /Volumes/GLM5-NVMe/exo/mimo-v25-pro/hf/XiaomiMiMo--MiMo-V2.5-Pro/model_mtp.safetensors \
  --modes ar,d1 \
  --max-tokens 1 \
  --preflight-only

Result: exit 0. Completed stages were modes_validated,
model_path_validated, and sidecar_contract_validated. Final row was
benchmark_preflight with ready=true.

Load-only command:

uv run python scripts/bench_mimo_mtp_fastpath.py \
  --model-path /Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit \
  --sidecar-path /Volumes/GLM5-NVMe/exo/mimo-v25-pro/hf/XiaomiMiMo--MiMo-V2.5-Pro/model_mtp.safetensors \
  --modes ar \
  --max-tokens 1 \
  --load-only

Result: exit 137. Last emitted structured stage was
base_model_materialization with status=started; no base_model_materialization
completion row was emitted. This confirms the kill happens during full model
materialization, before tokenizer load, sidecar tensor load, prompt
tokenization, AR generation, or MTP generation.

Minimal AR row: skipped because load-only failed with exit 137.

Minimal MTP D1 row: skipped because the AR prerequisite failed.

Machine memory diagnostics in stage rows reported process RSS near 372 MB before
model materialization; macOS system-available memory is not populated by the
portable Linux /proc/meminfo probe and is therefore null on this host.

## Gate conclusion

No live AR-vs-MTP metric rows exist from this host. No >=30 tok/s claim is made.
Slice 5 production/default integration remains blocked.

The next implementation move should be either:

1. run the same benchmark-survival commands on hardware that can materialize the
   full model, or
2. implement a distributed/load-reduced benchmark path that can produce real
   same-model/same-hardware AR-vs-MTP rows without local exit 137.
