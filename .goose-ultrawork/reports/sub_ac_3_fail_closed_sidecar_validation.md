# Sub-AC 3 Evidence — fail_closed sidecar validation

Timestamp: 2026-06-06 21:13 local

## Scope

Implemented request-level fail-closed validation for explicit guarded MiMo MTP fastpath requests when the required sidecar is missing.

Default requests remain autoregressive because the guard returns immediately unless `mimo_mtp_fastpath=true` is supplied.

## Files changed for this Sub-AC

- `src/exo/api/adapters/chat_completions.py`
  - Imports `probe_mimo_mtp_sidecar`.
  - Adds `validate_mimo_mtp_fastpath_eligibility(...)` for explicit MTP request eligibility.
  - Rejects non-MiMo V2.5 Pro requests with the existing clear unsupported-model reason.
  - For `mimo_mtp_fastpath=true` and `mimo_mtp_fail_closed=true`, rejects:
    - missing sidecar path with `disable_reason=missing_sidecar`
    - missing sidecar file with `disable_reason=missing_sidecar`
  - Calls the validation before request conversion/dispatch.
- `src/exo/api/tests/test_mimo_mtp_request_schema.py`
  - Adds `test_mimo_mtp_fastpath_fail_closed_rejects_missing_sidecar` asserting HTTP 400 and the exact missing-sidecar disable reason.

Note: the worktree contains additional neighboring rollout/AC changes from other workers. This report only claims the fail-closed missing-sidecar validation slice above.

## TDD evidence

RED command:

```bash
uv run pytest src/exo/api/tests/test_mimo_mtp_request_schema.py::test_mimo_mtp_fastpath_fail_closed_rejects_missing_sidecar -q
```

Observed RED failure after using the allowlisted MiMo V2.5 Pro model constant:

```text
Failed: DID NOT RAISE <class 'fastapi.exceptions.HTTPException'>
```

This proved that an explicit `mimo_mtp_fastpath=true`, `mimo_mtp_fail_closed=true` request with a missing sidecar path was not rejected before the production change.

GREEN/final focused verification:

```bash
uv run pytest src/exo/api/tests/test_mimo_mtp_request_schema.py::test_mimo_mtp_fastpath_fail_closed_rejects_missing_sidecar -q
uv run pytest src/exo/api/tests/test_mimo_mtp_request_schema.py -q
uv run pytest src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_sidecar_contract.py::test_missing_sidecar_fails_closed -q
uv run ruff check src/exo/api/adapters/chat_completions.py src/exo/api/tests/test_mimo_mtp_request_schema.py
uv run ruff format --check src/exo/api/adapters/chat_completions.py src/exo/api/tests/test_mimo_mtp_request_schema.py
uv run basedpyright src/exo/api/adapters/chat_completions.py src/exo/api/tests/test_mimo_mtp_request_schema.py
git diff --check -- src/exo/api/adapters/chat_completions.py src/exo/api/tests/test_mimo_mtp_request_schema.py
```

Observed final output:

- Targeted missing-sidecar regression: `1 passed in 0.28s`
- Full MTP request-schema file: `5 passed in 0.32s`
- Existing sidecar contract missing-sidecar probe test: `1 passed in 0.04s`
- Ruff check: `All checks passed!`
- Ruff format check: `2 files already formatted`
- Basedpyright: `0 errors, 0 warnings, 0 notes`
- `git diff --check`: exit 0 with no output

## Benchmark/live rows

Not applicable for this Sub-AC. No cluster API or live benchmark rows were required or fabricated. This AC validates request eligibility and the missing-sidecar disable/error path.

## Remaining risk

- This Sub-AC only rejects a missing sidecar for `fail_closed=true`. Invalid-but-present sidecar behavior remains governed by the existing sidecar probe/loader and later telemetry/worker ACs.
- The worktree has unrelated modified/untracked files from neighboring rollout ACs; this evidence is scoped to the files listed above.
