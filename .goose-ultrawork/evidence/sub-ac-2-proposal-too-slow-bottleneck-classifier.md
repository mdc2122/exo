# Sub-AC 2 Evidence: `proposal_too_slow` Bottleneck Classification

## Scope

Implemented focused bottleneck classification support for `proposal_too_slow` in the MiMo MTP bottleneck classifier.

The classifier now emits `proposal_too_slow` only when:

- a configured slow-proposal threshold exists (`slow_proposal_tokens_per_second`), and
- proposal throughput telemetry can be derived from proposal timing plus proposed-token telemetry, or from explicit `proposal_tps`, and
- derived/explicit proposal throughput is less than or equal to the configured threshold.

Missing proposal timing/throughput telemetry suppresses `proposal_too_slow` rather than guessing.

## Files Changed

- `scripts/mimo_mtp_bottleneck_classifier.py`
  - Added `slow_proposal_tokens_per_second` to `BottleneckThresholds`.
  - Added proposal telemetry parsing for explicit `proposal_tps` and proposed-token/timing-derived throughput.
  - Added `proposal_too_slow` emission before verifier classification.
- `scripts/test_mimo_mtp_bottleneck_classifier.py`
  - Added positive test for slow proposal throughput crossing the configured threshold.
  - Added missing-proposal-telemetry suppression test.

## TDD Evidence

### RED

Command:

```bash
uv run pytest scripts/test_mimo_mtp_bottleneck_classifier.py
```

Observed result before implementation:

- 4 tests collected.
- Existing verifier tests passed.
- Proposal tests failed with:
  - `TypeError: BottleneckThresholds.__init__() got an unexpected keyword argument 'slow_proposal_tokens_per_second'`

### GREEN / Verification

Command:

```bash
uv run pytest scripts/test_mimo_mtp_bottleneck_classifier.py && uv run ruff check scripts/mimo_mtp_bottleneck_classifier.py scripts/test_mimo_mtp_bottleneck_classifier.py && uv run basedpyright scripts/mimo_mtp_bottleneck_classifier.py scripts/test_mimo_mtp_bottleneck_classifier.py
```

Observed result after implementation:

- `pytest`: `4 passed in 0.01s`
- `ruff`: `All checks passed!`
- `basedpyright`: `0 errors, 0 warnings, 0 notes`

## Benchmark / Speed Claims

No live cluster benchmark rows were collected for this Sub-AC. This Sub-AC only implements a bottleneck-classifier behavior and its unit coverage.

No `>=30 tok/s`, `>=40 tok/s`, MTP speedup, or Slice 5 eligibility claim is made.

## Remaining Risk

- Classifier currently covers `proposal_too_slow` and existing `verifier_too_slow` behavior only. Other AC-P2 labels (`acceptance_rate_low`, `fallback_too_high`, `depth_too_aggressive`, `mtp_beats_ar`, `mtp_reaches_30`, `mtp_reaches_40`) remain outside this Sub-AC unless already handled by adjacent work.
- Live usefulness depends on cluster/API rows carrying proposal timing plus proposed-token telemetry or explicit `proposal_tps`.
