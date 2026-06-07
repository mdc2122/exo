# Sub-AC 4.4.2 Evidence — Requested-but-unwired MTP reason propagation

Timestamp: 2026-06-06 21:17 local

## Scope

Implemented and tested propagation of a clear requested-but-unwired MiMo MTP reason through caller-visible API/benchmark telemetry. Explicit guarded MiMo MTP requests no longer silently dispatch as AR when fail-closed is enabled and the exo cluster MTP execution backend is not wired.

## Files changed in scoped/touched surfaces

- `src/exo/shared/types/text_generation.py`
  - Added immutable `MimoMtpFastpathParams` internal request-intent model.
  - Added optional `mimo_mtp_fastpath` to `TextGenerationTaskParams`.
  - Kept default AR serialization behavior by omitting disabled/`None` MTP params from `model_dump()`.
- `src/exo/api/adapters/chat_completions.py`
  - Maps explicit guarded request fields into `MimoMtpFastpathParams`.
  - Validates non-MiMo requests and missing required fail-closed sidecar with clear disable reasons.
- `src/exo/api/types/api.py`
  - Added benchmark response telemetry fields: `accepted_execution_path`, `mtp_enabled`, `mtp_depth`, `mtp_sidecar_status`, `mtp_disable_reason`, `mtp_fallback_reason`, and `requested_mtp_depth`.
- `src/exo/api/main.py`
  - Added fail-closed guard that raises structured `mimo_mtp_execution_backend_unwired` detail before dispatch.
  - Added fail-open bench telemetry fields showing AR fallback with `mimo_mtp_execution_backend_unwired` disable reason.
  - Preserves default AR behavior for requests without explicit MTP params.
- `src/exo/api/tests/test_mimo_v25_pro_preview_path.py`
  - Added/verified tests for fail-closed unwired error propagation and fail-open AR disabled telemetry.
- Related rollout/benchmark surfaces present from parallel AC work and covered by focused checks:
  - `scripts/bench_mimo_mtp_cluster.py`
  - `scripts/test_bench_mimo_mtp_cluster.py`
  - `src/exo/api/tests/test_mimo_v25_pro_chat_adapter.py`

## TDD evidence

RED command:

```bash
uv run pytest src/exo/api/tests/test_mimo_v25_pro_preview_path.py::test_bench_mtp_request_fails_before_send_when_backend_is_unwired -q
```

Observed RED failure before implementation:

```text
FAILED src/exo/api/tests/test_mimo_v25_pro_preview_path.py::test_bench_mtp_request_fails_before_send_when_backend_is_unwired
AssertionError: MTP fail-closed guard must reject before _send
```

This proved the explicit MTP request reached `_send` and would have been dispatched without the requested-but-unwired reason.

GREEN targeted command after implementation:

```bash
uv run pytest src/exo/api/tests/test_mimo_v25_pro_preview_path.py::test_bench_mtp_request_fails_before_send_when_backend_is_unwired src/exo/api/tests/test_mimo_v25_pro_preview_path.py::test_bench_mtp_request_fail_open_reports_ar_disabled_telemetry -q
```

Observed result:

```text
..                                                                       [100%]
2 passed in 2.42s
```

## Verification commands run

Focused API/schema/default-AR/cluster benchmark tests:

```bash
uv run pytest src/exo/api/tests/test_mimo_v25_pro_preview_path.py src/exo/api/tests/test_mimo_mtp_request_schema.py src/exo/api/tests/test_mimo_v25_pro_chat_adapter.py src/exo/api/tests/test_mimo_mtp_normal_execution_path.py scripts/test_bench_mimo_mtp_cluster.py -q
```

Observed result:

```text
............................................                             [100%]
44 passed in 2.69s
```

Ruff:

```bash
uv run ruff check src/exo/shared/types/text_generation.py src/exo/api/adapters/chat_completions.py src/exo/api/main.py src/exo/api/types/api.py src/exo/api/tests/test_mimo_v25_pro_preview_path.py src/exo/api/tests/test_mimo_mtp_request_schema.py src/exo/api/tests/test_mimo_v25_pro_chat_adapter.py src/exo/api/tests/test_mimo_mtp_normal_execution_path.py scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py
```

Observed result:

```text
All checks passed!
```

Basedpyright:

```bash
uv run basedpyright src/exo/shared/types/text_generation.py src/exo/api/adapters/chat_completions.py src/exo/api/main.py src/exo/api/types/api.py src/exo/api/tests/test_mimo_v25_pro_preview_path.py src/exo/api/tests/test_mimo_mtp_request_schema.py src/exo/api/tests/test_mimo_v25_pro_chat_adapter.py src/exo/api/tests/test_mimo_mtp_normal_execution_path.py scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py
```

Observed result:

```text
0 errors, 0 warnings, 0 notes
```

Diff whitespace check:

```bash
git diff --check -- src/exo/shared/types/text_generation.py src/exo/api/adapters/chat_completions.py src/exo/api/main.py src/exo/api/types/api.py src/exo/api/tests/test_mimo_v25_pro_preview_path.py src/exo/api/tests/test_mimo_mtp_request_schema.py src/exo/api/tests/test_mimo_v25_pro_chat_adapter.py src/exo/api/tests/test_mimo_mtp_normal_execution_path.py scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py
```

Observed result:

```text
git diff --check: OK
```

## Blockers

None for Sub-AC 4.4.2.

## Remaining risk

- Full distributed MTP execution remains intentionally unwired in this slice; fail-closed requests surface `mimo_mtp_execution_backend_unwired`, and fail-open bench requests report AR fallback telemetry rather than pretending MTP ran.
- No live cluster benchmark rows were collected for this Sub-AC; this AC was scoped to propagation of the requested-but-unwired disable/error reason.
- Working tree contains additional governed rollout changes from parallel AC workers; this evidence scopes verification to the touched API/task/benchmark telemetry surfaces listed above.
