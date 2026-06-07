# Sub-AC 4 Evidence: Exact AR Baseline Command Generation

Timestamp: 2026-06-06 21:05:00

## Acceptance criterion

Add/update a command-rendering or documentation-backed testable unit that provides exact runnable commands for `max_tokens=16` and `max_tokens=64` AR baseline rows using `scripts/bench_mimo_mtp_cluster.py`.

## Files changed

- `../scripts/bench_mimo_mtp_cluster.py`
  - Added `AR_BASELINE_MAX_TOKENS = (16, 64)`.
  - Added `render_cluster_benchmark_command(...)` for exact runnable cluster benchmark command rendering.
  - Added `render_ar_baseline_commands(...)` for the required same-cluster AR baseline command pair.
  - Existing unavailable-cluster rows now include blocked rerun-command evidence through the same benchmark command path.
- `../scripts/test_bench_mimo_mtp_cluster.py`
  - Added `test_render_ar_baseline_commands_includes_exact_16_and_64_token_rows`.
  - Kept/updated unavailable-cluster expectations so blocked rows carry a runnable rerun command rather than implying live success.

## TDD evidence

RED command:

```bash
cd .. && uv run pytest scripts/test_bench_mimo_mtp_cluster.py::test_render_ar_baseline_commands_includes_exact_16_and_64_token_rows -q
```

RED result:

```text
FAILED scripts/test_bench_mimo_mtp_cluster.py::test_render_ar_baseline_commands_includes_exact_16_and_64_token_rows
AttributeError: module 'scripts.bench_mimo_mtp_cluster' has no attribute 'render_ar_baseline_commands'
```

GREEN targeted command:

```bash
cd .. && uv run pytest scripts/test_bench_mimo_mtp_cluster.py::test_render_ar_baseline_commands_includes_exact_16_and_64_token_rows -q
```

GREEN targeted result:

```text
1 passed in 0.05s
```

## Fresh verification commands

```bash
cd .. && uv run pytest scripts/test_bench_mimo_mtp_cluster.py -q && uv run ruff check scripts/bench_mimo_mtp_cluster.py scripts/test_bench_mimo_mtp_cluster.py && uv run basedpyright scripts/bench_mimo_mtp_cluster.py
```

Result:

```text
15 passed in 0.13s
All checks passed!
0 errors, 0 warnings, 0 notes
```

## Generated exact AR baseline commands

```bash
uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 16 --repeats 1 --mode-label ar --list-models
uv run python3 scripts/bench_mimo_mtp_cluster.py --api-base http://127.0.0.1:52415 --model-id kernelpool/MiMo-V2.5-Pro-6bit --max-tokens 64 --repeats 1 --mode-label ar --list-models
```

## Benchmark rows / blocker

No live AR benchmark rows were collected in this Sub-AC because the local cluster API was unavailable.

Probe command:

```bash
python3 - <<'PY'
import json
import urllib.request
url = 'http://127.0.0.1:52415/v1/models'
try:
    with urllib.request.urlopen(url, timeout=1.0) as response:
        print(json.dumps({'status': 'available', 'url': url, 'http_status': response.status}, sort_keys=True))
except Exception as exc:
    print(json.dumps({'status': 'blocked_cluster_api_unavailable', 'url': url, 'error': str(exc)}, sort_keys=True))
PY
```

Probe result:

```json
{"error": "<urlopen error [Errno 61] Connection refused>", "status": "blocked_cluster_api_unavailable", "url": "http://127.0.0.1:52415/v1/models"}
```

## Remaining risk

- The command renderer is verified and runnable, but same-cluster AR baseline evidence rows remain blocked until an exo cluster API is running at the selected `--api-base`.
- No MTP speedup, 30+ tok/s, or 40+ tok/s claim is made by this Sub-AC.
- Slice 5 remains blocked until same-cluster AR-vs-MTP rows prove a real MTP speed win.
