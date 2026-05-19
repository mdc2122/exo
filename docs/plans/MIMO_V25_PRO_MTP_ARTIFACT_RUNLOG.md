# MiMo V2.5 Pro MTP Artifact Runlog

Date: 2026-05-19

## Scope

This run generated the experimental MTP-only artifact for MiMo V2.5 Pro. It did
not start exo, load the full main model, alter distributed decode, or modify the
production 6-bit artifact.

## Artifact Identity

- Experimental model id: `kernelpool/MiMo-V2.5-Pro-6bit-mtp-experimental`
- Artifact kind: `mimo-v25-pro-mtp-only`
- Base runtime artifact: `kernelpool/MiMo-V2.5-Pro-6bit`
- Source MTP shard: `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/hf/XiaomiMiMo--MiMo-V2.5-Pro/model_mtp.safetensors`
- Output directory: `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental`
- Output shard: `model_mtp-00001-of-00001.safetensors`

## Conversion Evidence

- Dry run: `uv run python -m scripts.mimo_v25_pro_mtp_quantize --dry-run`
  - Result: `model_id=kernelpool/MiMo-V2.5-Pro-6bit-mtp-experimental`, `status=dry-run`, `source_tensor_count=48`, `artifact_kind=mimo-v25-pro-mtp-only`, `mtp_layers=[0, 1, 2]`
- Conversion: `uv run python -m scripts.mimo_v25_pro_mtp_quantize --convert --summary-json docs/plans/artifacts/mimo-v25-pro-mtp-artifact-summary-20260519.json`
  - Result: complete; summary JSON written
- Summary: `docs/plans/artifacts/mimo-v25-pro-mtp-artifact-summary-20260519.json`
  - Result: `output_shard_size=1659958379`, `tensor_count=72`, `complete=true`

## Safety Evidence

- Production artifact check: `test ! -e /Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit/model_mtp-00001-of-00001.safetensors`
  - Result: passed
- Staged state before commit: `git diff --cached --name-only`
  - Result: empty
- Working tree note: only pre-existing dirty KV/cache runtime files remained unstaged.

## Next Gate

Run the standalone MTP-only module probe before any runtime decode plan.

## Standalone MTP-Only Module Probe

- Probe: `uv run python scripts/mimo_v25_pro_mtp_module_probe.py --artifact /Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit-mtp-experimental --json-out docs/plans/artifacts/mimo-v25-pro-mtp-module-probe-20260519.json`
  - Result: probe completed; JSON report written
- Report: `docs/plans/artifacts/mimo-v25-pro-mtp-module-probe-20260519.json`
  - Result: `loaded_tensor_count=72`, `layer_0_eh_proj=[1, 1, 6144]`, `layer_0_qkv_proj=[1, 1, 27136]`

## Decode Boundary

The MTP-only artifact and standalone module probe passed. Full distributed MTP
decode remains blocked until a separate decode-controller plan covers
propose/verify/accept semantics and fallback behavior.
