# Sub-AC 4.4.1 Evidence — MTP unwired backend fail-closed guard

## Scope

Implemented and tested a fail-closed guard that detects explicit MiMo MTP requests when no exo-cluster MTP execution backend/path is wired, preventing silent AR-as-MTP execution before dispatch starts.

## Files changed for this Sub-AC surface

- `src/exo/shared/types/text_generation.py`
  - Added immutable `MimoMtpFastpathParams` internal request intent.
  - Added optional `mimo_mtp_fastpath` to `TextGenerationTaskParams`.
  - Serializer omits disabled/default MTP intent so default AR task dumps remain unchanged.
- `src/exo/api/adapters/chat_completions.py`
  - Maps explicit API MTP extension fields into internal task params.
  - Validates explicit MTP requests fail closed for non-MiMo model and missing sidecar when `mimo_mtp_fail_closed=true`.
- `src/exo/api/main.py`
  - Adds pre-dispatch `_guard_unwired_mimo_mtp_request()` for chat and bench completions.
  - Rejects fail-closed explicit MTP requests before `_send` when no MTP execution backend is wired.
  - Preserves fail-open benchmark telemetry path for explicit `mimo_mtp_fail_closed=false` without claiming MTP ran.
- `src/exo/api/types/api.py`
  - Adds experimental MTP request fields and benchmark telemetry response fields.
- `src/exo/api/tests/test_mimo_mtp_request_schema.py`
  - Tests extension defaults/schema and missing-sidecar fail-closed behavior.
- `src/exo/api/tests/test_mimo_mtp_normal_execution_path.py`
  - Tests normal chat request remains non-MTP/default AR and dispatches normally.
- `src/exo/api/tests/test_mimo_v25_pro_chat_adapter.py`
  - Tests normal MiMo request task params remain equivalent to pre-MTP golden behavior.
- `src/exo/api/tests/test_mimo_v25_pro_preview_path.py`
  - Tests bench response execution-path telemetry.
  - Tests explicit fail-closed MTP request is rejected before `_send` when backend is unwired.
  - Tests explicit fail-open request reports AR fallback/disable telemetry instead of pretending MTP ran.

## TDD RED evidence

Command:

```bash
uv run pytest src/exo/api/tests/test_mimo_v25_pro_preview_path.py::test_bench_mtp_request_fails_before_send_when_backend_is_unwired -q
```

Initial result: failed because `_send` was reached and raised `AssertionError("MTP fail-closed guard must reject before _send")`, proving the missing guard.

## Verification commands run

Focused tests:

```bash
uv run pytest src/exo/api/tests/test_mimo_mtp_request_schema.py src/exo/api/tests/test_mimo_v25_pro_chat_adapter.py src/exo/api/tests/test_mimo_mtp_normal_execution_path.py src/exo/api/tests/test_mimo_v25_pro_preview_path.py::test_bench_mtp_request_fails_before_send_when_backend_is_unwired src/exo/api/tests/test_mimo_v25_pro_preview_path.py::test_bench_mtp_request_fail_open_reports_ar_disabled_telemetry -q
```

Result: `22 passed in 2.54s` in final verification sequence.

Lint:

```bash
uv run ruff check src/exo/shared/types/text_generation.py src/exo/api/adapters/chat_completions.py src/exo/api/main.py src/exo/api/types/api.py src/exo/api/tests/test_mimo_mtp_request_schema.py src/exo/api/tests/test_mimo_mtp_normal_execution_path.py src/exo/api/tests/test_mimo_v25_pro_chat_adapter.py src/exo/api/tests/test_mimo_v25_pro_preview_path.py
```

Result: `All checks passed!` in final verification sequence.

Type check:

```bash
uv run basedpyright src/exo/shared/types/text_generation.py src/exo/api/adapters/chat_completions.py src/exo/api/main.py src/exo/api/types/api.py src/exo/api/tests/test_mimo_mtp_request_schema.py src/exo/api/tests/test_mimo_mtp_normal_execution_path.py src/exo/api/tests/test_mimo_v25_pro_chat_adapter.py src/exo/api/tests/test_mimo_v25_pro_preview_path.py
```

Result: `0 errors, 0 warnings, 0 notes`.

Whitespace/diff check:

```bash
git diff --check -- src/exo/shared/types/text_generation.py src/exo/api/adapters/chat_completions.py src/exo/api/main.py src/exo/api/types/api.py src/exo/api/tests/test_mimo_mtp_request_schema.py src/exo/api/tests/test_mimo_mtp_normal_execution_path.py src/exo/api/tests/test_mimo_v25_pro_chat_adapter.py src/exo/api/tests/test_mimo_v25_pro_preview_path.py
```

Result: exit code 0, no output.

## Benchmark rows

No live AR-vs-MTP benchmark rows were collected for this Sub-AC. This criterion is a guard/dispatch safety slice, not a speed claim. No >=30 tok/s, >=40 tok/s, or MTP speedup claim is made.

## Remaining risk

- Full distributed MTP execution is still intentionally unwired; explicit fail-closed requests reject before execution starts.
- Fail-open benchmark requests can continue through AR only with explicit fallback/disable telemetry and must not be interpreted as MTP performance evidence.
- Broader AC-P5/P6/P7 work remains needed before any Slice 5 eligibility or performance claim.
