# MTPLX reference for adapting MiMo V2.5 Pro MTP into exo

Source repo: https://github.com/youssofal/MTPLX
Local clone: /Users/studio2/reference-repos/MTPLX
HEAD: 0ad700c Prepare v0.3.7 release

Use MTPLX as a reference implementation, not as code to copy wholesale. Adapt it for exo's distributed MLX worker/generator path and MiMo V2.5 Pro model constraints.

## Key reference files

- `mtplx/mimo_mtp_patch.py`: MiMo-specific runtime MTP injection.
  - Detects MiMo MTP config when `model_type == "mimo"` and `num_nextn_predict_layers`/`mtp_num_hidden_layers` > 0.
  - Locates MTP weights via expected MTP artifact, safetensors index, or model shard scan.
  - Rewrites raw MiMo MTP weight keys from `model.mtp_layers.*` / appended `model.layers.<start+i>.*` into local `layers.<i>.*` module keys.
  - Builds `_MiMOMTPLayer` with token RMSNorm, hidden RMSNorm, `input_proj([previous_hidden, token_embedding])`, MiMo `TransformerBlock`, final RMSNorm.
  - Adds `model.mtp_forward(hidden_states, next_token_ids, mtp_cache=...)` and `model.make_mtp_cache()`.
  - Important limitation in MTPLX: MiMo MTP currently supports `mtp_depth=1` only (`mtp_forward` rejects depth > 1). Do not assume D2/D3 are correct for MiMo without implementing them.

- `mtplx/generation.py`: reference AR and MTP generation semantics.
  - `generate_mtp1` is the relevant one-token speculative path.
  - Algorithm shape:
    1. Prefill target model with prompt and return both logits and hidden state.
    2. Sample a primary target token from current target logits; append it.
    3. Call `draft_mtp(hidden, [[primary]], mtp_cache=make_mtp_cache())` to propose the next token distribution.
    4. Sample draft token from draft logits using the draft sampler.
    5. Verify with target model over `[primary, draft_token]` using the target AR path and current KV cache.
    6. Compare target distribution for the draft position against draft distribution.
    7. Accept draft token with probability `min(1, p_target(token) / q_draft(token))`; if rejected, sample correction from residual `max(p-q,0)` distribution.
    8. Update emitted token stream and KV/cache state correctly:
       - accepted draft: commit target verify results for both primary and draft; next logits/hidden are from the draft position.
       - rejected greedy: keep/repair only the primary target token state.
       - rejected stochastic correction: rollback verify side effects, then forward `[primary, correction]` (or equivalent) to keep target KV consistent.
    9. EOS/stop/max_tokens behavior must match AR semantics.
  - MTPLX includes optimized/capture variants, but exo should first implement a clear correct path before performance tuning.

- `mtplx/sampling.py`: correctness oracle for speculative sampling.
  - `acceptance_probability(p, q, token) = min(1, p/q)` with q<=0 handling.
  - `residual_distribution(p, q) = normalize(max(p-q,0))`, fallback to target if residual empty.
  - `verify_one_token` and `speculative_output_marginal` are useful references for tests.

## Adaptation guidance for exo

- Do not treat MTPLX benchmarks, UI, profiles, graphbank, capture, thermal, server, or evidence machinery as part of this task.
- The exo implementation should focus on actual code correctness in:
  - model-specific MTP sidecar/module loading,
  - exo MLX worker/generator decode loop,
  - target model verify pass,
  - accepted/rejected token emission,
  - KV/cache rollback/repair/commit semantics,
  - preserving default AR behavior.
- A correct but simple MiMo depth=1 implementation is better than broad D1/D2/D3 plumbing that is not semantically correct.
- If MiMo D2/D3 is attempted, first establish the exact MiMo nextN architecture semantics; MTPLX explicitly says its MiMo backend currently only supports depth=1.
