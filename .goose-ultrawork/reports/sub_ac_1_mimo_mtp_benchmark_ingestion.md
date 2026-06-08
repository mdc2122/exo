# Sub-AC 1 Evidence: AR/MTP JSONL Benchmark Ingestion

Timestamp: 2026-06-06 23:12 local

## Scope

Added and verified an AR/MTP JSONL ingestion surface for benchmark rows emitted by the guarded exo cluster `/bench/chat/completions` harness. The ingestion unit:

- Parses JSONL content and JSONL files.
- Distinguishes AR telemetry from MTP telemetry using the row `mode` label.
- Separates parsed telemetry through `BenchmarkIngestionResult.ar_rows`, `mtp_rows`, and `live_mtp_rows`.
- Marks live MTP rows only when `accepted_execution_path == "mimo_mtp_fastpath"`.
- Validates required benchmark fields.
- Tolerates missing optional fields such as `prompt_tps`, `power_usage`, `payload_extra_keys`, `live_execution_path`, and MTP acceptance telemetry.
- Tolerates AR-only JSONL rows where no MTP rows are present; these rows remain valid and produce `status == "blocked_no_mtp_rows"` instead of an ingestion error.
- Reports MTP-labelled rows with no response-confirmed live MTP as `status == "blocked_no_live_mtp_rows"`.
- Reports malformed rows without dropping other valid rows.
- Preserves metadata aliases (`mode_label`, `model_id`) and normalized `payload_extra_keys` in raw rows for downstream budget analysis.
- Adds optional bottleneck classification for low MTP acceptance rate when acceptance telemetry and a classifier config are supplied.
- Computes parsed AR-vs-live-MTP speedup budget metrics without making Slice 5 claims when either side is missing.

## Files Changed

- `scripts/mimo_mtp_benchmark_ingest.py`
  - `ingest_benchmark_jsonl(...)` parses JSONL text into validated benchmark rows and errors.
  - `ingest_benchmark_jsonl_file(...)` reads a JSONL path and delegates to the parser.
  - Immutable dataclasses model ingestion errors, ingested rows, result summaries, and bottleneck classifier config.
  - Required field validation covers `kind`, `cluster_path`, `mode`, `model`, `repeat_index`, `generation_tps`, and `generation_tokens`.
  - Parser accepts only `kind == "cluster_benchmark_metric"` rows from `cluster_path == "exo_api_bench_chat_completions"`.
  - `mode == "ar"` maps to AR telemetry; non-AR mode labels map to MTP candidates.
  - `accepted_execution_path == "mimo_mtp_fastpath"` is required before an MTP candidate is counted as live MTP.
  - Summary includes AR row count, MTP row count, live MTP row count, max generation tok/s, AR median generation tok/s, live-MTP median generation tok/s, and comparison status.
  - Comparison status now distinguishes:
    - `ready_for_same_cluster_comparison` when live MTP rows exist,
    - `blocked_no_live_mtp_rows` when MTP-labelled rows exist but none are live,
    - `blocked_no_mtp_rows` when JSONL contains AR rows only.
  - Acceptance rate can be read from top-level `acceptance_rate`, nested `generation_stats.acceptance_rate`, or derived from `generation_stats.mimo_mtp_attempted_tokens` and `generation_stats.mimo_mtp_accepted_tokens`.
  - `calculate_parsed_mtp_speedup_budget(...)` computes parsed median AR/MTP tok/s, ms/token, target gaps to 30 and 40 tok/s, speedup ratio when both sides exist, and blocked statuses when rows are incomplete.

- `scripts/test_mimo_mtp_benchmark_ingest.py`
  - Tests valid mixed AR and live MTP rows.
  - Tests AR-only JSONL rows with no MTP telemetry and verifies they are valid with `blocked_no_mtp_rows`.
  - Tests MTP-labelled but not live rows and verifies `blocked_no_live_mtp_rows`.
  - Tests malformed JSON and malformed required fields while preserving later valid rows.
  - Tests missing optional fields.
  - Tests JSONL file ingestion.
  - Tests metadata aliases and normalized `payload_extra_keys` preservation.
  - Tests low-acceptance bottleneck classification from derived attempted/accepted token telemetry.
  - Tests explicit top-level acceptance-rate telemetry.
  - Tests classifier suppression when acceptance telemetry is absent.
  - Tests parsed budget metrics for complete AR/MTP rows and AR-only rows.

## Commands / Tests Run

Repository and file discovery from `.goose-ultrawork`:

```text
pwd && find . -maxdepth 2 -type f | sort | sed 's#^./##'
# confirmed working directory: /Users/studio2/exo/.worktrees/mimo-v25-pro-mtp-fastpath-single-stream-20260601/.goose-ultrawork
```

```text
git rev-parse --show-toplevel && git status --short
# confirmed repo root: /Users/studio2/exo/.worktrees/mimo-v25-pro-mtp-fastpath-single-stream-20260601
# confirmed existing modified files from prior ACs before this Sub-AC finalization
```

Inspected existing ingestion/test/evidence surfaces:

```text
sed -n '1,240p' scripts/mimo_mtp_benchmark_ingest.py
sed -n '1,260p' scripts/test_mimo_mtp_benchmark_ingest.py
sed -n '1,180p' .goose-ultrawork/reports/sub_ac_1_mimo_mtp_benchmark_ingestion.md
```

TDD RED for the explicit AR-only/missing-MTP acceptance case:

```text
uv run pytest scripts/test_mimo_mtp_benchmark_ingest.py::test_ingest_ar_only_rows_are_valid_when_mtp_rows_are_missing -q
F                                                                        [100%]
E       AssertionError: assert 'blocked_no_live_mtp_rows' == 'blocked_no_mtp_rows'
1 failed in 0.04s
```

GREEN and targeted verification after the minimal `_comparison_status(...)` implementation:

```text
uv run pytest scripts/test_mimo_mtp_benchmark_ingest.py::test_ingest_ar_only_rows_are_valid_when_mtp_rows_are_missing -q
.                                                                        [100%]
1 passed in 0.01s
```

```text
uv run pytest scripts/test_mimo_mtp_benchmark_ingest.py -q
............                                                             [100%]
12 passed in 0.01s
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

No new live cluster AR/MTP benchmark rows were collected for this Sub-AC. This Sub-AC implements and verifies the ingestion and validation layer only. It does not claim 30+ tok/s, 40+ tok/s, or MTP speedup.

Same-cluster AR-vs-MTP evidence rows remain required before any Slice 5 production/default MTP decision.

Runnable ingestion usage once JSONL rows are available:

```python
from scripts.mimo_mtp_benchmark_ingest import ingest_benchmark_jsonl_file

result = ingest_benchmark_jsonl_file("path/to/cluster-benchmark.jsonl")
print(result.summary)
```

Runnable targeted verification command:

```bash
cd /Users/studio2/exo/.worktrees/mimo-v25-pro-mtp-fastpath-single-stream-20260601
uv run pytest scripts/test_mimo_mtp_benchmark_ingest.py -q
```

## Remaining Risk

- The parser is intentionally strict about accepting only `cluster_benchmark_metric` rows from `exo_api_bench_chat_completions`; future harness field renames require parser and test updates together.
- Rows with non-`ar` mode labels are classified as MTP candidates, but they are not treated as live MTP unless the response explicitly reports `accepted_execution_path == "mimo_mtp_fastpath"`.
- AR-only ingestion is intentionally non-fatal and blocked for speedup comparison; downstream gates must keep treating `blocked_no_mtp_rows` as insufficient evidence for 30+ tok/s, 40+ tok/s, or MTP speedup claims.
- Live cluster behavior remains unproven here; this module is ready to consume evidence but does not fabricate benchmark rows.
