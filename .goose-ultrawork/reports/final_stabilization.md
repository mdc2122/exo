# MiMo MTP Optimized Rollout Ultrawork Final Closeout

Date: 2026-06-07

## Run outcome

The optimized rollout work was executed in the isolated repository worktree:

`/Users/studio2/exo/.worktrees/mimo-v25-pro-mtp-fastpath-single-stream-20260601`

The earlier Goose-Ouroboros run ended with raw `RUN_EXIT=141`, so the orchestrator
itself is not recorded as a clean completion. The generated implementation was
manually reviewed, repaired, formatted, verified, and closed out through AC-P10.

No nested Ouroboros/auto run was started during closeout.

## Completed ACs

- AC-P0: optimized rollout seed and repo context validated.
- AC-P1: cluster AR baseline harness canonical and budget-ready; live rows blocked when cluster API unavailable.
- AC-P2: speedup budget model exists before optimization claims.
- AC-P3: explicit guarded MTP request contract added and fail-closed.
- AC-P4: MTP intent propagates through immutable internal task params.
- AC-P5: worker/generator guarded vertical slice production-shaped but disabled by default.
- AC-P6: benchmark-grade MTP telemetry defined and carried to cluster rows.
- AC-P7: same-cluster AR-vs-MTP benchmark matrix runnable and evidence-gated.
- AC-P8: measurement-driven optimization/bottleneck loop encoded.
- AC-P9: strict Slice 5 gate preserved; blocked without live same-cluster evidence.
- AC-P10: focused verification and closeout artifacts recorded.

## Remaining ACs

No acceptance criteria remain unrecorded for this guarded rollout slice. The
remaining work is external evidence collection and possible implementation follow-up,
not a completed Slice 5 enablement claim.

## Worktrees reviewed/merged

- Reviewed/modified worktree: `feature/mimo-v25-pro-mtp-fastpath-single-stream-20260601` at closeout HEAD `6911a9f5`.
- Worktrees merged by this closeout: none.
- Commits created by this closeout: none.

## Accepted stabilized output

- Optimized rollout seed/control files under `.goose-ultrawork/`.
- Cluster benchmark harness enhancements in `scripts/bench_mimo_mtp_cluster.py`.
- Matrix command rendering for same-cluster AR and guarded MTP D1/D2/D3 rows.
- MTP benchmark row ingestion, speedup budget calculator, and bottleneck classifier scripts.
- Explicit experimental MiMo MTP request contract fields.
- Internal `MimoMtpFastpathParams` propagation through text generation task params.
- Fail-closed/fail-open reporting for requested-but-unwired MTP execution.
- Benchmark-grade MTP telemetry surfaces and tests.
- Module validation and guarded worker fastpath helper surfaces.
- Docs/runlog/TODO/Beads-facing evidence preserving the strict Slice 5 gate.

## Verification run during AC-P10

Focused pytest command covered touched scripts/API/shared/worker tests and observed:

```text
225 passed in 4.28s
```

Ruff check over touched code/test surfaces observed:

```text
All checks passed!
```

Ruff format check initially observed:

```text
Would reformat: scripts/test_git_cleanliness.py
1 file would be reformatted, 40 files already formatted
```

After `uv run ruff format scripts/test_git_cleanliness.py`, rerun observed:

```text
41 files already formatted
```

Basedpyright over touched code/test surfaces observed:

```text
0 errors, 0 warnings, 0 notes
```

Diff whitespace check:

```text
git diff --check: clean / no output
```

## Live benchmark and Slice 5 status

No live same-cluster AR-vs-guarded-MTP rows were collected in AC-P10. Earlier live
collection attempts remain blocked by unavailable cluster API responses. The
canonical runnable matrix remains recorded in
`.goose-ultrawork/evidence/ac-p7-benchmark-matrix-commands.jsonl`.

This closeout makes no `>=30 tok/s`, no `>=40 tok/s`, no MTP speedup, and no Slice
5 production/default eligibility claim. Default exo generation remains AR unless
an explicit experimental MTP guard is enabled.

## Beads status

No live Beads rows were mutated during AC-P10. Beads mirroring remains explicitly
opt-in and guarded by:

```bash
uv run python3 scripts/mirror_rollout_seed_to_beads.py .goose-ultrawork/seed.yaml --enable-beads-mirror
```

The Beads-facing closeout status is recorded in
`.goose-ultrawork/evidence/sub-ac-4-beads-mirror-sink.md`.

## Next best step

Start or point to a healthy exo tensor-parallel cluster API, use the official
MiMo MTP sidecar path, run the commands in
`.goose-ultrawork/evidence/ac-p7-benchmark-matrix-commands.jsonl`, then analyze
the resulting same-cluster rows with `scripts/mimo_mtp_budget_calculator.py` and
`scripts/mimo_mtp_bottleneck_classifier.py`. Only after those rows prove guarded
MTP beats AR without fallback/correctness concern should Slice 5 guarded
production review be considered.

## Final post-format pytest rerun

After `scripts/test_git_cleanliness.py` was reformatted during AC-P10, the same
focused touched-surface pytest command was rerun and observed:

```text
225 passed in 4.23s
```

## QA revise evidence pass -- 2026-06-07T00:48 local

A focused QA revise pass added concrete command transcripts, AC evidence mapping, git dirty-state explanation, benchmark blocked-with-command evidence, request/task propagation snippets, fail-closed/fail-open telemetry snippets, and an explicit Slice 5 gate decision.

Evidence packet: .goose-ultrawork/evidence/ac-p10-qa-revise-evidence.md

Command logs:
- focused-pytest: .goose-ultrawork/evidence/qa-revise-20260607T004800Z-focused-pytest.log -- EXIT_CODE=0
- ruff-check: .goose-ultrawork/evidence/qa-revise-20260607T004800Z-ruff-check.log -- EXIT_CODE=0
- ruff-format-check: .goose-ultrawork/evidence/qa-revise-20260607T004800Z-ruff-format-check.log -- EXIT_CODE=0
- basedpyright: .goose-ultrawork/evidence/qa-revise-20260607T004800Z-basedpyright.log -- EXIT_CODE=0
- diff-check: .goose-ultrawork/evidence/qa-revise-20260607T004800Z-diff-check.log -- EXIT_CODE=0
- matrix-commands: .goose-ultrawork/evidence/qa-revise-20260607T004800Z-matrix-commands.log -- EXIT_CODE=0
- unavailable-cluster-probe: .goose-ultrawork/evidence/qa-revise-20260607T004800Z-unavailable-cluster-probe.log -- EXIT_CODE=1
- budget-fixture: .goose-ultrawork/evidence/qa-revise-20260607T004800Z-budget-fixture.log -- EXIT_CODE=0
- classifier-fixture: .goose-ultrawork/evidence/qa-revise-20260607T004800Z-classifier-fixture.log -- EXIT_CODE=0
- git-context: .goose-ultrawork/evidence/qa-revise-20260607T004800Z-git-context.log -- EXIT_CODE=0

Closeout wording is scoped: evidence-recorded guarded scaffold and benchmark readiness, not completed production/default MTP deployment. Slice 5 remains blocked until same-cluster AR-vs-MTP rows prove a real guarded MTP win.

## Final AC-P10 refresh -- 2026-06-07T03:02 local

A final closeout refresh covered the expanded touched rollout helper surface and
resolved the last local verification findings:

- `scripts/test_mimo_mtp_benchmark_ingest.py` and
  `scripts/test_mimo_mtp_model_path_evidence.py` were formatted with `ruff format`.
- `scripts/validate_mimo_mtp_request_docs.py` now validates/casts generated schema
  mappings before reading MTP defaults, keeping strict basedpyright clean.

Fresh evidence prefix: `.goose-ultrawork/evidence/ac-p10-final-20260607T080246Z-*`.

Command results:

- Focused pytest: `317 passed in 5.18s`, `EXIT_CODE=0`.
- Ruff check: `All checks passed!`, `EXIT_CODE=0`.
- Ruff format check: `49 files already formatted`, `EXIT_CODE=0`.
- Basedpyright: `0 errors, 0 warnings, 0 notes`, `EXIT_CODE=0`.
- `git diff --check`: no output, `EXIT_CODE=0`.
- Matrix command generation: canonical 8-row AR + guarded MTP D1/D2/D3 matrix emitted, `EXIT_CODE=0`.
- Budget fixture: completed, `EXIT_CODE=0`.
- Bottleneck classifier fixture: completed, `EXIT_CODE=0`.
- Local live cluster probe: blocked with `Connection refused`, `EXIT_CODE=1`, with rerun command recorded in the log.

No live same-cluster AR-vs-guarded-MTP benchmark rows were collected. No `>=30
tok/s`, no `>=40 tok/s`, no MTP speedup, and no Slice 5 production/default
eligibility claim is made. Default generation remains AR; guarded MTP remains
explicit/experimental only.
