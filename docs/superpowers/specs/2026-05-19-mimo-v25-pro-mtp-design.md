# MiMo V2.5 Pro 6-bit MTP Acceleration Design

Date: 2026-05-19

## Purpose

Raise single-request effective output throughput for `kernelpool/MiMo-V2.5-Pro-6bit`
on the Studio1/Studio2 two-node M3 Ultra cluster while preserving the known-good
non-MTP runtime path.

Current measured baseline:

- Single request: about 22 tokens/second sustained decode.
- Batched aggregate: about 40 tokens/second across concurrent requests.
- Hardware: two Mac Studio M3 Ultra machines, each with 512 GB RAM, connected via
  the current exo tensor/JACCL setup.
- Loaded model: `kernelpool/MiMo-V2.5-Pro-6bit`.
- Artifact size: about 778 GiB on disk, with safetensors metadata reporting
  835,299,199,488 bytes.
- Source MTP shard: `model_mtp.safetensors`, about 2.3 GiB, present in the
  original `XiaomiMiMo/MiMo-V2.5-Pro` checkpoint.

The target is to discover the true safe single-stream ceiling with MTP enabled.
Reaching 40 tokens/second is a stretch goal, not a promise.

## Non-Negotiable Safety Rules

Never start a second full MiMo V2.5 Pro instance while one is already resident,
loading, warming, or partially failed but still consuming memory. The Mac Studios
can crash if duplicate MiMo instances are started.

Any full-model experiment must fail closed unless preflight confirms:

- No active, loading, warming, or stale MiMo instance exists.
- No resident MiMo runner process remains from a prior run.
- Studio1 and Studio2 both report recovered memory above a conservative floor.
- The operator intentionally stopped the current exo cluster for the experiment.

Metadata-only and MTP-shard-only probes may run while the production exo cluster
is up. Anything that loads the full MiMo main model requires a downtime window.

## Git Backup And Recovery

Before MTP work begins, create durable recovery points:

- A backup branch or tag at the current pushed baseline commit `1114ff85`.
- A separate checkpoint for the current dirty KV/cache/batch-generation work, if
  those changes are intentional.
- A recovery note recording the known-good branch, commit, exo startup
  environment, model artifact path, instance id, and benchmark artifacts.

MTP work must happen on a separate feature branch, for example:

`feature/mimo-v25-pro-mtp-single-stream-20260519`

The known-good 6-bit non-MTP runtime remains rollbackable and must not be mutated
in place.

## Approach

Use a hybrid research path:

1. Protect the current working MiMo implementation with git recovery points.
2. Probe MTP artifacts offline.
3. Build MTP module loading outside the live exo cluster.
4. Prove propose/verify/accept logic with a tiny fixture or synthetic model.
5. Only then integrate an opt-in MTP path into distributed exo generation.

This avoids treating MTP as a simple config toggle. The current
`mimo_v2_flash` path drops `model.mtp.*` weights during sanitize, and the active
6-bit config has no enabled `mtp_config`, `use_mtp`, or
`num_nextn_predict_layers` fields.

## Architecture

### MTP Artifact Layer

Own discovery and conversion of `model_mtp.safetensors`.

Responsibilities:

- Detect whether a MiMo artifact has MTP weights.
- Validate expected `model.mtp.layers.0..2` tensor coverage.
- Quantize MTP tensors using the same MLX affine 6-bit rules as the main
  artifact.
- Write a separate experimental artifact identity, never the working artifact.
- Produce a manifest with tensor names, shapes, source dtypes, output dtypes, and
  conversion decisions such as FP8 scale broadcast/crop handling.

### MTP Module Layer

Own loading and running the MTP modules.

Responsibilities:

- Extend or wrap `mimo_v2_flash` only when an experimental model id or feature
  flag is active.
- Make the existing `model.mtp.*` sanitize drop conditional.
- Load MTP tensors without loading the full main MiMo model during offline
  probes.
- Expose a narrow proposal API that can be exercised with synthetic hidden
  states before full-model runtime testing.

### MTP Decode Controller

Own proposal, verification, and acceptance.

Normal decode emits one visible token per main-model step. MTP decode should:

1. Run the main model for the next token.
2. Use MTP modules to propose additional future tokens.
3. Verify proposals against the main model.
4. Accept the longest valid prefix.
5. Fall back safely to normal one-token decode when proposals are rejected.

Telemetry must report proposed tokens per step, accepted tokens per step,
acceptance rate, fallback rate, effective output tokens/second, and any
divergence reason.

### Exo Runtime Integration

Only after offline probes pass, add opt-in runtime support:

- Preserve the current `BatchGenerator`/MLX path as the default.
- Enable MTP only through an explicit experimental model identity or feature
  flag.
- Run the duplicate-MiMo preflight before any full-model experiment.
- Start exactly one experimental MiMo instance across Studio1/Studio2.
- Verify with tiny bounded requests before repeated or longer throughput tests.

## Staged Delivery

### Stage 1: Artifact And Shape Probe

Inspect source and quantized artifacts offline. Confirm the MTP shard exists,
enumerate `model.mtp.layers.0..2`, and generate a JSON shape report. This stage
does not start exo, JACCL, or full model loading.

### Stage 2: Quantized MTP Artifact

Extend the existing quantization script to convert `model_mtp.safetensors`.
Output goes to an experimental artifact directory only.

### Stage 3: Standalone MTP Module Probe

Instantiate and load only the MTP module surface, using synthetic hidden states
and the 2.3 GiB MTP shard. Do not load the full 778 GiB MiMo main artifact.

### Stage 4: Tiny Fixture Decode Prototype

Validate propose/verify/accept semantics on a tiny fixture or synthetic model.
Correctness comes before speed. The fallback path must produce valid normal
autoregressive output when no proposals are accepted.

### Stage 5: Distributed Full-Model Experiment

Stop the existing exo cluster, wait for memory recovery, start exactly one
experimental MiMo instance, then run tiny bounded requests. Collect acceptance
rate, single-request effective tokens/second, memory, power, and crash-free
repeatability evidence.

## Success Criteria

Backup success:

- Known-good baseline branch/tag exists.
- Dirty KV/cache work is separately preserved or explicitly excluded.
- Recovery note identifies branch, commit, startup env, artifact paths, and
  benchmark artifacts.

Artifact success:

- MTP source shard is detected.
- All expected MTP tensors are accounted for.
- Quantized MTP output is written only to an experimental artifact.

Module success:

- MTP modules instantiate and load without full MiMo main-model load.
- Synthetic forward produces expected shapes.

Decode success:

- Propose/verify/accept is proven on a tiny fixture.
- Rejected proposals fall back to normal decode.
- Telemetry distinguishes proposed, accepted, and emitted tokens.

Distributed runtime success:

- Preflight refuses unsafe duplicate-MiMo startup.
- One bounded distributed request completes without crash.
- MTP-disabled known-good path still works.
- Single-request effective tokens/second improves over the 22 tokens/second
  baseline.

Stretch success:

- Single-request effective output approaches or exceeds 40 tokens/second if MTP
  acceptance rate supports it.

## Risks

- MTP weights may not map cleanly onto the pinned `mimo_v2_flash` MLX-LM
  implementation.
- MTP acceptance rate may be too low to approach 40 tokens/second.
- Verification may add enough overhead that speed gains are smaller than theory.
- Distributed tensor/JACCL synchronization may limit single-stream speed even
  with MTP.
- Full-model experiments are memory dangerous and require strict duplicate-run
  prevention.

## Out Of Scope

- Mutating the current working 6-bit artifact in place.
- Making MTP the default runtime path.
- Starting full MiMo while another MiMo instance is resident.
- Chasing multimodal MiMo support.
- Replacing the existing batching path for aggregate throughput.

