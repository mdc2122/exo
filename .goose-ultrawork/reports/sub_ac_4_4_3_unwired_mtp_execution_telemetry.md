# Sub-AC 4.4.3 Evidence: requested-but-unwired MTP never reports MTP success

Status: complete

## Scope

Implemented and tested the inference/API execution guard so an explicit MiMo MTP request cannot be recorded or reported as successful MTP execution when the cluster generation path is not wired to an MTP backend.

## Files changed for this Sub-AC

- `src/exo/api/adapters/chat_completions.py`
  - Validates explicit guarded MTP request eligibility.
  - Carries explicit MiMo MTP request intent into internal task params only when requested.
  - Fails closed for ineligible/non-MiMo and missing sidecar preconditions.
- `src/exo/shared/types/text_generation.py`
  - Adds immutable `MimoMtpFastpathParams` and optional `TextGenerationTaskParams.mimo_mtp_fastpath`.
  - Keeps default AR task param serialization free of MTP intent when no MTP request is present.
- `src/exo/api/types/api.py`
  - Adds experimental MTP request fields with strict disabled defaults.
  - Adds benchmark response telemetry fields for accepted execution path and disabled/unwired MTP status.
- `src/exo/api/main.py`
  - Rejects explicit fail-closed MTP requests before dispatch when the execution backend is unwired.
  - For explicit fail-open/unwired benchmark requests, allows AR dispatch but returns explicit disabled telemetry: `accepted_execution_path="ar"`, `mtp_enabled=false`, `mtp_disable_reason="mimo_mtp_execution_backend_unwired"`, `mtp_fallback_reason="fail_open_backend_unwired"`.
  - Adds execution-path metadata for benchmark responses so same-cluster rows can distinguish tensor-parallel exo cluster path.
- `src/exo/api/tests/test_mimo_v25_pro_preview_path.py`
  - Adds regression coverage for fail-closed unwired MTP rejecting before `_send`.
  - Adds regression coverage for fail-open unwired MTP reporting AR/disabled telemetry rather than MTP success.

Note: the target worktree contains other parallel/previous AC changes and untracked files. This evidence is scoped only to the files above.

## TDD evidence

RED command:

```bash
uv run pytest src/exo/api/tests/test_mimo_v25_pro_preview_path.py::test_bench_mtp_request_fails_before_send_when_backend_is_unwired -q
```

Observed RED failure before implementation:

```text
FAILED ... AssertionError: MTP fail-closed guard must reject before _send
```

This proved an explicit MTP request reached `_send` as ordinary `TextGenerationTaskParams`, i.e. the requested-but-unwired path could silently dispatch AR without honest MTP disabled status.

Additional RED for fail-open telemetry:

```bash
uv run pytest src/exo/api/tests/test_mimo_v25_pro_preview_path.py::test_bench_mtp_request_fail_open_reports_ar_disabled_telemetry src/exo/api/tests/test_mimo_v25_pro_preview_path.py::test_bench_mtp_request_fails_before_send_when_backend_is_unwired -q
```

Observed fail-open RED failure before adding response telemetry:

```text
AttributeError: 'BenchChatCompletionResponse' object has no attribute 'accepted_execution_path'
```

## Verification commands run

Focused behavioral tests:

```bash
uv run pytest \
  src/exo/api/tests/test_mimo_v25_pro_preview_path.py::test_bench_mtp_request_fail_open_reports_ar_disabled_telemetry \
  src/exo/api/tests/test_mimo_v25_pro_preview_path.py::test_bench_mtp_request_fails_before_send_when_backend_is_unwired \
  src/exo/api/tests/test_mimo_mtp_request_schema.py \
  src/exo/api/tests/test_mimo_mtp_normal_execution_path.py \
  -q
```

Observed result:

```text
8 passed in 2.57s
```

Lint and format:

```bash
uv run ruff check \
  src/exo/api/main.py \
  src/exo/api/types/api.py \
  src/exo/api/adapters/chat_completions.py \
  src/exo/shared/types/text_generation.py \
  src/exo/api/tests/test_mimo_v25_pro_preview_path.py \
  src/exo/api/tests/test_mimo_mtp_request_schema.py \
  src/exo/api/tests/test_mimo_mtp_normal_execution_path.py

uv run ruff format --check \
  src/exo/api/main.py \
  src/exo/api/types/api.py \
  src/exo/api/adapters/chat_completions.py \
  src/exo/shared/types/text_generation.py \
  src/exo/api/tests/test_mimo_v25_pro_preview_path.py \
  src/exo/api/tests/test_mimo_mtp_request_schema.py \
  src/exo/api/tests/test_mimo_mtp_normal_execution_path.py
```

Observed results:

```text
All checks passed!
7 files already formatted
```

Type checking and whitespace:

```bash
uv run basedpyright \
  src/exo/api/main.py \
  src/exo/api/types/api.py \
  src/exo/api/adapters/chat_completions.py \
  src/exo/shared/types/text_generation.py \
  src/exo/api/tests/test_mimo_v25_pro_preview_path.py \
  src/exo/api/tests/test_mimo_mtp_request_schema.py \
  src/exo/api/tests/test_mimo_mtp_normal_execution_path.py

git diff --check -- \
  src/exo/api/main.py \
  src/exo/api/types/api.py \
  src/exo/api/adapters/chat_completions.py \
  src/exo/shared/types/text_generation.py \
  src/exo/api/tests/test_mimo_v25_pro_preview_path.py \
  src/exo/api/tests/test_mimo_mtp_request_schema.py \
  src/exo/api/tests/test_mimo_mtp_normal_execution_path.py
```

Observed results:

```text
0 errors, 0 warnings, 0 notes
```

`git diff --check` completed with exit code 0 and no output.

## Benchmark/live rows

No live cluster benchmark rows were collected for this Sub-AC. This Sub-AC is a guarded execution/telemetry regression slice. No 30+ tok/s, 40+ tok/s, or MTP speedup claim is made.

## Blockers

None for this Sub-AC.

## Remaining risk

- Full distributed MTP execution remains unwired in this slice; the implemented behavior intentionally fails closed by default or fail-opens as AR with explicit disabled telemetry.
- Broader AC-P5/P6 work still needs actual worker/generator MTP execution telemetry once the native sidecar-backed backend is wired.
- Worktree contains other parallel AC changes; this report does not claim those changes.
