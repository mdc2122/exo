# Sub-AC 5.2 — MTP request field documentation validation

## Requirement

Document every added MTP request field as experimental and disabled by default, with a runnable documentation validation check that fails if any added field is missing or not marked experimental/disabled-by-default.

## Files changed for this Sub-AC

- `docs/api.md`
  - Added `Experimental MiMo V2.5 Pro MTP request fields` section under chat completions.
  - Documents all schema-discovered MTP request fields as `Experimental` and `Disabled by default`:
    - `mimo_mtp_fastpath`
    - `mimo_mtp_depth`
    - `mimo_mtp_sidecar_path`
    - `mimo_mtp_fail_closed`
  - Includes the runnable validation command:
    - `uv run python3 scripts/validate_mimo_mtp_request_docs.py`
- `scripts/validate_mimo_mtp_request_docs.py`
  - Adds a runnable validation check.
  - Discovers `mimo_mtp_*` fields from `ChatCompletionRequest.model_fields` so future added MTP request fields must be documented.
  - Fails if a field has no docs row or if that row lacks `Experimental` or `Disabled by default`.
- `scripts/test_validate_mimo_mtp_request_docs.py`
  - Adds TDD coverage for schema discovery, missing field rows, missing markers, and passing docs.

## TDD RED evidence

Initial unit check failed before the validator existed:

```bash
uv run pytest scripts/test_validate_mimo_mtp_request_docs.py -q
```

Observed failure:

```text
ModuleNotFoundError: No module named 'scripts.validate_mimo_mtp_request_docs'
```

After the validator existed but before docs were updated, the real docs check failed as intended:

```bash
uv run python3 scripts/validate_mimo_mtp_request_docs.py
```

Observed failure:

```text
MiMo MTP request documentation validation failed:
- mimo_mtp_fastpath: missing documentation row
- mimo_mtp_depth: missing documentation row
- mimo_mtp_sidecar_path: missing documentation row
- mimo_mtp_fail_closed: missing documentation row
```

A later schema-discovery hardening test also failed before implementation:

```text
ImportError: cannot import name 'discover_mimo_mtp_request_fields'
```

## Passing verification evidence

Final focused checks run from repo root (`/Users/studio2/exo/.worktrees/mimo-v25-pro-mtp-fastpath-single-stream-20260601`):

```bash
uv run python3 scripts/validate_mimo_mtp_request_docs.py
uv run pytest scripts/test_validate_mimo_mtp_request_docs.py -q
uv run ruff check scripts/validate_mimo_mtp_request_docs.py scripts/test_validate_mimo_mtp_request_docs.py
uv run basedpyright scripts/validate_mimo_mtp_request_docs.py scripts/test_validate_mimo_mtp_request_docs.py
```

Observed passing output:

```text
MiMo MTP request documentation validation passed: docs/api.md
8 passed in 0.17s
All checks passed!
0 errors, 0 warnings, 0 notes
```

## Benchmark rows

Not applicable for Sub-AC 5.2. This criterion is documentation validation only. No tok/s, 30+ tok/s, 40+ tok/s, or MTP speedup claim is made.

## Remaining risk

- The validator checks exact per-field documentation rows for the required markers; it does not semantically prove that every sentence is correct.
- The check is runnable and evidence-backed, but it is not yet wired into a broader CI job by this Sub-AC.
