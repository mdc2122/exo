# Sub-AC 3 Evidence: Benchmark harness AR mode labeling

## Scope

Ensure benchmark requests/results emitted by the cluster benchmark harness include AR mode labeling, with a runnable regression test verifying the AR label is present and correct.

## Files changed for this Sub-AC

- `scripts/bench_mimo_mtp_cluster.py`
  - Adds `benchmark_mode_label` to each emitted `/bench/chat/completions` request payload using the requested `--mode-label` value.
  - Existing result rows continue to emit `mode`, and AR rows resolve `accepted_execution_path` to `ar` when response telemetry does not override it.
- `scripts/test_bench_mimo_mtp_cluster.py`
  - Adds `test_run_cluster_benchmark_labels_ar_requests_and_results`, which verifies:
    - posted request payload contains `benchmark_mode_label == "ar"`
    - emitted result row contains `mode == "ar"`
    - AR result row contains `accepted_execution_path == "ar"`
  - Existing request-payload expectations include the explicit AR label.

Note: the same files also contain neighboring benchmark-harness changes from the broader AC-P1/P0 work in this worktree. This report only claims the AR mode labeling slice above.

## TDD evidence

1. Added `test_run_cluster_benchmark_labels_ar_requests_and_results` before implementation.
2. Verified RED with:

```bash
cd .. && uv run pytest scripts/test_bench_mimo_mtp_cluster.py::test_run_cluster_benchmark_labels_ar_requests_and_results -q
```

Observed failure:

```text
KeyError: 'benchmark_mode_label'
```

3. Implemented the minimal harness change: set `request_payload["benchmark_mode_label"] = args.mode_label` before posting to `/bench/chat/completions`.
4. Verified GREEN with:

```bash
cd .. && uv run pytest scripts/test_bench_mimo_mtp_cluster.py::test_run_cluster_benchmark_labels_ar_requests_and_results -q
```

Observed result:

```text
1 passed in 0.05s
```

## Verification commands run

```bash
cd .. && uv run pytest scripts/test_bench_mimo_mtp_cluster.py -q
```

Observed result:

```text
15 passed in 0.15s
```

```bash
cd .. && uv run ruff check scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py
```

Observed result:

```text
All checks passed!
```

```bash
cd .. && uv run basedpyright scripts/bench_mimo_mtp_cluster.py
```

Observed result:

```text
0 errors, 0 warnings, 0 notes
```

```bash
cd .. && git diff --check -- scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py
```

Observed result: exit code 0, no whitespace errors printed.

## Benchmark rows / blocker status

- No live cluster benchmark rows were collected for this Sub-AC.
- This Sub-AC is a harness request/result labeling test slice; live cluster rows remain dependent on an available exo cluster API.
- Runnable AR baseline command preserved by the harness/tests:

```bash
uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 16 --repeats 1 --mode-label ar --list-models
```

## Remaining risk

- FastAPI/Pydantic request models may ignore unknown benchmark-only request payload fields unless later ACs add explicit typed request contract fields; this Sub-AC only guarantees the harness emits the AR label and preserves AR result labeling in harness rows.
- Same-cluster AR-vs-MTP performance evidence is not claimed here. Slice 5 remains blocked until same-cluster benchmark rows prove a real MTP win.
