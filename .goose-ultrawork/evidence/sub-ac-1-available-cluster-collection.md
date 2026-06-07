# Sub-AC 1 Evidence — Available-cluster collection success path

Timestamp: 2026-06-06 21:04 local

## Scope

Added/updated runnable test coverage for the canonical exo-cluster benchmark harness success path. The test verifies that when the cluster API responds successfully through the injectable HTTP seams, expected telemetry/result rows are collected and returned by `run_cluster_benchmark(...)`.

## Files changed in scope

- `scripts/test_bench_mimo_mtp_cluster.py`
  - Updated `test_run_cluster_benchmark_emits_models_probe_and_metric_row` to assert successful collection returns:
    - `/v1/models` probe row,
    - `/bench/chat/completions` metric row,
    - `generation_tps`,
    - `generation_tokens`,
    - top-level `prompt_tps`,
    - `power_usage`,
    - `payload_extra_keys`,
    - repeat/mode request telemetry in the posted payload.
  - Formatted with `ruff format`.
- `scripts/bench_mimo_mtp_cluster.py`
  - Added top-level `prompt_tps` propagation from successful `generation_stats` into returned `cluster_benchmark_metric` rows.

Note: this governed worktree also contains adjacent AC-P1 changes in the same benchmark harness, including command rendering, blocked-with-command unavailable-cluster rows, and execution-path classification. This evidence claims only the available-cluster success-path collection slice above.

## TDD evidence

RED command:

```bash
uv run pytest scripts/test_bench_mimo_mtp_cluster.py::test_run_cluster_benchmark_emits_models_probe_and_metric_row -q
```

Observed RED failure before implementation:

```text
KeyError: 'prompt_tps'
```

This showed the success-path test detected that the collected metric row did not yet expose required top-level `prompt_tps` telemetry.

GREEN/focused command after implementation:

```bash
uv run pytest scripts/test_bench_mimo_mtp_cluster.py::test_run_cluster_benchmark_emits_models_probe_and_metric_row -q
```

Observed result:

```text
1 passed in 0.05s
```

## Verification commands run

```bash
uv run pytest scripts/test_bench_mimo_mtp_cluster.py::test_run_cluster_benchmark_emits_models_probe_and_metric_row -q
uv run pytest scripts/test_bench_mimo_mtp_cluster.py -q
uv run ruff check scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py
uv run ruff format --check scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py
uv run basedpyright scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py
git diff --check -- scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py
```

Observed final results before writing this report:

```text
1 passed in 0.05s
15 passed in 0.15s
All checks passed!
2 files already formatted
0 errors, 0 warnings, 0 notes
```

`git diff --check` completed with exit code 0 and no output for the touched harness files.

## Benchmark/live rows

No live cluster rows were collected in this Sub-AC. The test uses injected fake HTTP functions to verify the available-cluster success path deterministically without fabricating benchmark performance evidence.

Runnable live collection commands preserved by the harness for later AC-P1 evidence include:

```bash
uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 16 --repeats 1 --mode-label ar --list-models
uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 64 --repeats 1 --mode-label ar --list-models
```

## Blockers

None for this test slice.

## Remaining risk

- This Sub-AC proves success-path row collection via injected HTTP seams, not live cluster availability.
- No tok/s target, MTP speedup, or Slice 5 eligibility claim is made; those require same-cluster AR-vs-MTP benchmark rows.
- Broader benchmark harness changes from adjacent AC-P1 work remain in the worktree and should be reviewed in their own AC evidence.
