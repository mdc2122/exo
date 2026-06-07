# Sub-AC 1 Evidence: AR/MTP JSONL Benchmark Ingestion

Timestamp: 2026-06-06 21:25 local

## Scope

Added an AR/MTP JSONL ingestion surface for benchmark rows emitted by the guarded exo cluster `/bench/chat/completions` harness. The ingestion unit:

- Parses JSONL content and JSONL files.
- Distinguishes AR telemetry from MTP telemetry using the row `mode` label.
- Marks live MTP rows only when `accepted_execution_path == "mimo_mtp_fastpath"`.
- Validates required benchmark fields.
- Tolerates missing optional fields such as `prompt_tps`, `power_usage`, `payload_extra_keys`, `live_execution_path`, and MTP acceptance telemetry.
- Reports malformed rows without dropping other valid rows.
- Preserves a safe `blocked_no_live_mtp_rows` status when a dataset has MTP-labelled rows but no response-confirmed live MTP rows.
- Adds optional bottleneck classification for low MTP acceptance rate when acceptance telemetry and a classifier config are supplied.

## Files Changed

- `scripts/mimo_mtp_benchmark_ingest.py`
  - New `ingest_benchmark_jsonl(...)` function.
  - New `ingest_benchmark_jsonl_file(...)` function.
  - New immutable dataclasses for ingestion errors, ingested rows, result summaries, and bottleneck classifier config.
  - Validates required fields: `kind`, `cluster_path`, `mode`, `model`, `repeat_index`, `generation_tps`, and `generation_tokens`.
  - Accepts only cluster metric rows from `cluster_path == "exo_api_bench_chat_completions"`.
  - Distinguishes `telemetry_mode == "ar"` for `mode == "ar"`; all non-AR labels are treated as MTP candidates.
  - Computes summaries including AR row count, MTP row count, live MTP row count, max generation tok/s, AR median generation tok/s, live-MTP median generation tok/s, and comparison readiness status.
  - Derives optional `optional_acceptance_rate` from top-level `acceptance_rate`, nested `generation_stats.acceptance_rate`, or nested `generation_stats.mimo_mtp_attempted_tokens` / `generation_stats.mimo_mtp_accepted_tokens`.
  - Emits `acceptance_rate_low` only when configured and when acceptance telemetry is present and below threshold.

- `scripts/test_mimo_mtp_benchmark_ingest.py`
  - Tests valid AR and live MTP rows.
  - Tests malformed JSON and malformed required fields while preserving later valid rows.
  - Tests missing optional fields and datasets with no live MTP rows.
  - Tests JSONL file ingestion.
  - Tests low-acceptance bottleneck classification from derived attempted/accepted token telemetry.
  - Tests explicit top-level acceptance-rate telemetry.
  - Tests classifier suppression when acceptance telemetry is absent.

## TDD / RED Evidence

Initial test-first RED run before creating the ingestion module:

```text
uv run pytest scripts/test_mimo_mtp_benchmark_ingest.py -q
...
ImportError: cannot import name 'mimo_mtp_benchmark_ingest' from 'scripts' (unknown location)
1 error in 0.05s
```

Later RED after adjacent bottleneck-classification expectations appeared in the same test file:

```text
uv run pytest scripts/test_mimo_mtp_benchmark_ingest.py -q
...FF.
AttributeError: module 'scripts.mimo_mtp_benchmark_ingest' has no attribute 'BottleneckClassifierConfig'
2 failed, 4 passed in 0.03s
```

## GREEN / Verification Evidence

Fresh focused verification after implementation and formatting:

```text
uv run pytest scripts/test_mimo_mtp_benchmark_ingest.py -q
.......                                                                  [100%]
7 passed in 0.01s
```

```text
uv run ruff check scripts/mimo_mtp_benchmark_ingest.py scripts/test_mimo_mtp_benchmark_ingest.py
All checks passed!
```

```text
uv run ruff format --check scripts/mimo_mtp_benchmark_ingest.py scripts/test_mimo_mtp_benchmark_ingest.py
2 files already formatted
```

```text
uv run basedpyright scripts/mimo_mtp_benchmark_ingest.py scripts/test_mimo_mtp_benchmark_ingest.py
0 errors, 0 warnings, 0 notes
```

```text
git diff --check -- scripts/mimo_mtp_benchmark_ingest.py scripts/test_mimo_mtp_benchmark_ingest.py
# no output
```

## Benchmark Rows / Blocker Status

No live cluster AR/MTP benchmark rows were collected for this Sub-AC. This Sub-AC implements the ingestion and validation layer only. It does not claim 30+ tok/s, 40+ tok/s, or MTP speedup. Same-cluster AR-vs-MTP rows remain required before any Slice 5 production/default MTP decision.

Runnable ingestion usage once JSONL rows are available:

```python
from scripts.mimo_mtp_benchmark_ingest import ingest_benchmark_jsonl_file

result = ingest_benchmark_jsonl_file("path/to/cluster-benchmark.jsonl")
print(result.summary)
```

## Remaining Risk

- The parser is intentionally strict about accepting only `cluster_benchmark_metric` rows from `exo_api_bench_chat_completions`; if future harnesses rename fields, ingestion tests and parser validation should be updated together.
- Rows with non-`ar` mode labels are classified as MTP candidates, but they are not treated as live MTP unless the response explicitly reports `accepted_execution_path == "mimo_mtp_fastpath"`.
- Live cluster behavior is still unproven here; this ingestion module is ready to consume evidence but does not fabricate benchmark rows.
