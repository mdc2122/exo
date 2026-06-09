# Direct MiMo V2.5 Pro MTP Implementation Directive

Objective: get real MiMo V2.5 Pro MTP working in exo's MLX/tensor-parallel cluster path, then reach 30+ tok/s, preferably 40+ tok/s.

Hard constraints:
- Do not add trace modules, mapping docs, benchmark frameworks, broad scaffolding, or process evidence as deliverables.
- Do not fake MTP by reporting AR as MTP.
- Do not change default AR behavior unless guarded MiMo MTP is explicitly requested.
- Add only minimal targeted tests/probes that protect implementation invariants.

Immediate implementation focus:
1. Fix `src/exo/worker/engines/mlx/mimo_mtp_fast/mtp_generate.py` so acceptance follows MTPLX depth=1 p/q speculative sampling semantics, not argmax equality.
2. Verify reject path samples/corrects from residual `normalize(max(p_target - q_draft, 0))` with safe fallback to target distribution.
3. Verify target KV/cache commit/rollback/repair semantics for primary, accepted draft, and rejected draft cases.
4. Ensure `batch_generator.py` routes guarded MiMo MTP requests into the real MLX MTP path where safe and fail-closes to AR/rejects explicitly where not safe.
5. Run a minimal local correctness smoke, then a real two-node exo cluster smoke. Only then optimize toward 30+/40+ tok/s.

Reference:
- `/Users/studio2/reference-repos/MTPLX/mtplx/generation.py::generate_mtp1`
- `/Users/studio2/reference-repos/MTPLX/mtplx/sampling.py`
- `/Users/studio2/reference-repos/MTPLX/mtplx/mimo_mtp_patch.py`

Previous drifted checkpoint preserved at:
- `/Users/studio2/.ouroboros/worktrees/mimo-v25-pro-mtp-code-correctness-20260609T041233Z/orch_98fe76d2baac`
- commit `a30e9ed1 checkpoint: preserve pre-recenter MiMo MTP candidate`
