# Sub-AC 2 Evidence: AR Baseline Row Parsing Schema

Timestamp: 2026-06-06 21:05 local

## Scope

Sub-AC 2 requested the AR baseline row parsing/normalization unit and runnable tests so emitted cluster benchmark rows include:

- `generation_tps`
- `generation_tokens`
- `prompt_tps`
- `power_usage`
- model id (`model`)
- mode label (`mode`)
- repeat index (`repeat_index`)
- propagated payload extra keys (`payload_extra_keys`)

## Files Changed

- `scripts/bench_mimo_mtp_cluster.py`
  - Exposes `build_cluster_metric_row(...)` as the row-normalization unit.
  - Normalizes `/bench/chat/completions` response stats into top-level row fields: `generation_tps`, `generation_tokens`, `prompt_tps`, and `power_usage`.
  - Preserves `model`, `mode`, `repeat_index`, and sorted `payload_extra_keys` on emitted metric rows.
  - Propagates benchmark request metadata (`benchmark_mode_label`, `benchmark_repeat_index`) into each request payload.
  - Adds runnable command renderers for same-cluster AR baseline collection.
  - Normalizes unavailable-cluster rows with `status=blocked_with_command`, `success_row_count`, and `rerun_command`.

- `scripts/test_bench_mimo_mtp_cluster.py`
  - Adds/updates runnable harness and row-normalization tests.
  - Verifies AR metric rows include required schema fields.
  - Verifies non-empty payload extra keys are propagated to emitted rows.
  - Verifies repeat index propagation and unavailable-cluster blocked-row behavior.
  - Verifies exact max_tokens=16 and max_tokens=64 AR baseline commands.

## TDD / RED Evidence

Targeted tests were run before the final implementation was complete and failed as expected:

```text
uv run pytest scripts/test_bench_mimo_mtp_cluster.py -q
..F.........F..
2 failed, 13 passed in 0.16s
```

Failures proved missing row/request normalization behavior:

- `KeyError: 'benchmark_mode_label'` for request mode propagation.
- Missing unavailable-cluster blocked-row metadata (`status`, `success_row_count`, `rerun_command`).

A later intermediate RED run after partial wiring showed all remaining failures were stale/unwired error-row schema call sites:

```text
uv run pytest scripts/test_bench_mimo_mtp_cluster.py -q
............FFF
3 failed, 12 passed in 0.16s
```

## GREEN / Verification Evidence

Fresh targeted pytest:

```text
uv run pytest scripts/test_bench_mimo_mtp_cluster.py -q
...............
15 passed in 0.15s
```

Focused lint:

```text
uv run ruff check scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py
All checks passed!
```

Focused typecheck:

```text
uv run basedpyright scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py
0 errors, 0 warnings, 0 notes
```

Focused whitespace diff check:

```text
git diff --check -- scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py
# no output
```

Focused format check:

```text
uv run ruff format --check scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py
2 files already formatted
```

## Runnable AR Baseline Commands Preserved

The harness renders exact commands for same-cluster AR rows:

```bash
uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 16 --repeats 1 --mode-label ar --list-models
uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 64 --repeats 1 --mode-label ar --list-models
```

## Benchmark Rows / Blocker

No live cluster benchmark row was collected in this Sub-AC execution. This task was limited to row parsing/normalization schema and runnable unit tests. To avoid fabricating performance evidence or accidentally triggering MiMo cluster generation outside this Sub-AC, live AR row collection remains blocked pending an explicitly available/safe exo cluster API. Use the commands above when the cluster is confirmed safe and available.

## Remaining Risk

- Schema tests use injected fake HTTP responses; they verify parsing/normalization behavior but do not prove a live cluster currently returns all telemetry fields.
- Live `/bench/chat/completions` rows are still required before any throughput, 30+ tok/s, 40+ tok/s, or speedup claim.
- `model_path_classification` depends on live `response.execution_path` telemetry when available; absent telemetry is marked `unknown` rather than assumed tensor-parallel.
