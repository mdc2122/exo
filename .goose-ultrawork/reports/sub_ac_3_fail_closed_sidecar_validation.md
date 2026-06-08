# Sub-AC 3 Evidence — fail_closed sidecar validation

Timestamp: 2026-06-06 23:03 local

## Scope

Implemented and verified request-level fail-closed validation for explicit guarded MiMo V2.5 Pro MTP fastpath requests when the required MTP sidecar is missing.

Default requests remain autoregressive because the MTP guard returns immediately unless `mimo_mtp_fastpath=true` is supplied. Missing-sidecar requests with `mimo_mtp_fail_closed=true` now return an explicit rejected/not-executed telemetry object instead of any response shape that could be mistaken for MTP execution.

## Files changed for this Sub-AC

- `src/exo/api/adapters/chat_completions.py`
  - Uses `probe_mimo_mtp_sidecar(...)` at the API boundary for explicit guarded MTP requests.
  - Rejects explicit MTP requests before dispatch when `mimo_mtp_fail_closed=true` and the sidecar path is absent or the sidecar file is missing.
  - Returns structured disable telemetry for missing sidecars:
    - `error="mimo_mtp_sidecar_missing"`
    - `mtp_enabled=false`
    - `accepted_execution_path="rejected"`
    - `mtp_depth=null`
    - `mtp_sidecar_status="missing"`
    - `mtp_disable_reason="missing_sidecar"`
  - Preserves fail-open behavior for `mimo_mtp_fail_closed=false`, leaving later unwired-backend/fallback handling to existing guarded paths.
- `src/exo/api/tests/test_mimo_mtp_request_schema.py`
  - Adds/updates regression coverage for `mimo_mtp_fastpath=true`, `mimo_mtp_fail_closed=true`, and a missing `model_mtp.safetensors` sidecar.
  - Asserts HTTP 400 and the explicit rejected/not-executed telemetry object.

Note: the worktree contains additional neighboring rollout changes from other AC workers. This evidence is scoped to the fail-closed missing-sidecar validation slice above.

## TDD evidence

RED command:

```bash
cd .. && uv run pytest src/exo/api/tests/test_mimo_mtp_request_schema.py::test_mimo_mtp_fastpath_fail_closed_rejects_missing_sidecar -q
```

Observed RED failure:

```text
FAILED src/exo/api/tests/test_mimo_mtp_request_schema.py::test_mimo_mtp_fastpath_fail_closed_rejects_missing_sidecar
AssertionError: assert False
 + where False = isinstance('MiMo MTP fastpath is disabled: required sidecar is missing at ... disable_reason=missing_sidecar. Provide a valid model_mtp.safetensors sidecar or disable mimo_mtp_fastpath.', dict)
```

This proved the missing-sidecar fail-closed path returned a string detail rather than benchmark-grade rejected/not-executed telemetry.

GREEN targeted command:

```bash
cd .. && uv run pytest src/exo/api/tests/test_mimo_mtp_request_schema.py::test_mimo_mtp_fastpath_fail_closed_rejects_missing_sidecar -q
```

Observed GREEN output:

```text
.                                                                        [100%]
1 passed in 0.27s
```

## Final focused verification

Command:

```bash
cd .. && uv run pytest src/exo/api/tests/test_mimo_mtp_request_schema.py src/exo/api/tests/test_mimo_mtp_normal_execution_path.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_sidecar_contract.py::test_missing_sidecar_fails_closed -q && uv run ruff check src/exo/api/adapters/chat_completions.py src/exo/api/tests/test_mimo_mtp_request_schema.py src/exo/api/tests/test_mimo_mtp_normal_execution_path.py src/exo/worker/engines/mlx/mimo_mtp_fast/sidecar_contract.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_sidecar_contract.py && uv run ruff format --check src/exo/api/adapters/chat_completions.py src/exo/api/tests/test_mimo_mtp_request_schema.py src/exo/api/tests/test_mimo_mtp_normal_execution_path.py src/exo/worker/engines/mlx/mimo_mtp_fast/sidecar_contract.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_sidecar_contract.py && uv run basedpyright src/exo/api/adapters/chat_completions.py src/exo/api/tests/test_mimo_mtp_request_schema.py src/exo/api/tests/test_mimo_mtp_normal_execution_path.py src/exo/worker/engines/mlx/mimo_mtp_fast/sidecar_contract.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_sidecar_contract.py && git diff --check -- src/exo/api/adapters/chat_completions.py src/exo/api/tests/test_mimo_mtp_request_schema.py src/exo/api/tests/test_mimo_mtp_normal_execution_path.py src/exo/worker/engines/mlx/mimo_mtp_fast/sidecar_contract.py src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_sidecar_contract.py
```

Observed output:

```text
............                                                             [100%]
12 passed in 0.52s
All checks passed!
5 files already formatted
0 errors, 0 warnings, 0 notes
```

`git diff --check` produced no output and completed successfully as part of the chained verification command.

## Benchmark/live rows

Not applicable for this Sub-AC. No cluster API or live benchmark rows were required or fabricated. This AC validates request eligibility and fail-closed missing-sidecar error telemetry before dispatch.

## Remaining risk

- This Sub-AC validates missing sidecar handling for `fail_closed=true`; invalid-but-present sidecar behavior remains governed by the sidecar contract/loader and later runtime telemetry paths.
- The MTP execution backend remains guarded/unwired elsewhere, so no MTP throughput or speedup claim is made here.
- The worktree has unrelated modified/untracked files from neighboring rollout ACs; this evidence is scoped to the files and commands listed above.
