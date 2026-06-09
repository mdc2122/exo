# MiMo V2.5 Pro MTP Light-Factory Contract

> Purpose: replace the last 24 hours of brittle scaffold-heavy agent activity with a small, well-lit, contract-based workflow that can actually ship MiMo V2.5 Pro MTP in exo and then optimize toward 30+ tok/s, preferably 40+ tok/s, on the Studio1+Studio2 tensor-parallel cluster.

## 0. Why this exists

The previous approach failed operationally even though individual agents produced useful commits. Failure modes observed from the last 24h transcript and live worktrees:

- Long-running Ouroboros sessions accumulated context and drifted into trace/test scaffolding.
- Kanban cards were used as a ledger and launch mechanism, but dependencies/statuses caused claim failures and stale blocked cards.
- Some worker profiles lacked required skills and crashed immediately (`Unknown skill(s): kanban-worker`).
- Parallelism was increased before the bottleneck was identified; workers produced branches faster than integration could absorb.
- The active Goose job ran for >45m with no direct-branch git output.
- The main exo checkout and several old MiMo worktrees contain large amounts of stale/untracked evidence noise.
- The user’s actual goal is not “more process”; it is a working, correct, fast MTP path.

This contract is the new source of truth for MiMo MTP agent work until replaced.

## 1. Principle: fresh session per contract

No open-ended “go build MTP” sessions.

Every worker gets exactly one contract with:

1. Tiny scope.
2. Exact allowed files.
3. Exact forbidden work.
4. Deterministic completion gate.
5. A clean worktree.
6. Required final evidence.

When the contract is done, the session ends. New work means a new contract and a fresh context.

## 2. Roles

### Human / Morley

- Owns product acceptance and risk decisions.
- Judges final artifacts and live cluster behavior.
- Does not need to manage individual implementation details.

### Hermes Mayor / Conductor

- Owns the board, contracts, worktree hygiene, and stop/go decisions.
- Spawns only contract workers, not vague task workers.
- Stops stale jobs proactively.
- Integrates only reviewed commits.
- Keeps chat updates concise and evidence-backed.

### Implementer workers

- Touch code only inside their contract.
- Do not create broad scaffolding, trace modules, dashboards, evidence frameworks, or benchmark infrastructure.
- Commit local changes on their lane branch.
- End by reporting commit hash, tests run, remaining blocker.

### Reviewer workers

- Review a specific branch/diff or merge gate.
- Try to disprove readiness neutrally.
- Do not rewrite large features.
- Produce pass/fail with exact evidence.

### Integrator workers

- Merge a small set of already-reviewed commits onto the integration branch.
- Resolve conflicts only within stated files.
- Run the contract’s local gate.
- Commit the integration result.

## 3. Hard rules for MiMo MTP

- No `mtp_trace/` or trace-only modules.
- No broad test/evidence scaffolding.
- No fake MTP by reporting AR as MTP.
- Default AR must remain unchanged unless guarded MTP is explicitly requested.
- Depth=1 first because it is the smallest real correctness slice; depth=2/3 only after live depth=1 semantics are correct.
- Tests are allowed only when they directly protect a load-bearing invariant:
  - p/q accept probability
  - residual correction on reject
  - no unverified draft token emission
  - KV/cache commit/rollback/repair
  - sidecar tensor load correctness
  - guarded routing fail-closed behavior

## 4. Current canonical branches/worktrees

### Integration base

- Branch: `integration/mimo-mtp-direct-20260609T1500Z`
- Worktree: `/Users/studio2/exo/.worktrees/mimo-mtp-integration-20260609T1500Z`
- Current HEAD: `6eb25ca7 MTP: fix target KV cache commit semantics`

### Candidate lane commits

Merge-order reviewer gate: `/Users/studio2/exo/.worktrees/mimo-mtp-int-review-20260609T1500Z/INTEGRATION_G_MERGE_GATE.md`

- A: `41a62bbe MiMo MTP: implement p/q speculative acceptance` — blocked on basedpyright errors.
- B: `cef5af85 MIMO-MTP: repair minimal decode invariant probes` — should follow A.
- C: `d41db9a5 t_c072bca9: port MTP hot-path token history` — after A+B, inspect `mtp_generate.py` semantics.
- D: `eefa72a0 MTP: load 6-bit quantized sidecar tensors` — sidecar fix.
- E: `a86072b4 MTP: fail closed unwired batch routing` — routing guard.
- F: `f4624c32 t_7d17767c: harden live two-node smoke gate` — smoke harness, last.

### Known dirty/noisy worktrees to quarantine

Do not launch new workers from these without an explicit rescue contract:

- `/Users/studio2/exo` — 1140 dirty entries, mostly historical evidence artifacts.
- `/Users/studio2/exo/.worktrees/mimo-v25-pro-mtp-fastpath-single-stream-20260601` — 179 dirty entries.
- `/Users/studio2/exo/.worktrees/mimo-v25-pro-mtp-d1-live-preflight-20260601` — 53 dirty entries.
- `/Users/studio2/exo/.worktrees/codex-mimo-v25-pro-tp-20260527` — 8 dirty entries.
- `/Users/studio2/.ouroboros/worktrees/mimo-v25-pro-mtp-direct-20260609T142051Z/orch_581683d5b196` — 2 dirty entries from cancelled Goose run; salvage only if explicitly contracted.

## 5. Workflow topology: local Gas-City-like “light factory”

We will not install a new factory stack until this project is stable. We will emulate the useful topology with existing Hermes/Kanban/Git primitives.

```text
Human goal
  ↓
Hermes Mayor / Conductor
  ↓ creates one-page CONTRACT.md per unit
Contract queue / Kanban board
  ↓
Fresh implementer worker(s) on isolated worktrees
  ↓ local commit + evidence
Reviewer worker attempts to disprove readiness
  ↓
Integrator worker merges reviewed commits into integration branch
  ↓ local deterministic gate
Hermes Mayor decides next contract or live smoke
  ↓
Live Studio1+Studio2 exo cluster smoke
  ↓
Throughput optimization contracts toward 30+/40+ tok/s
```

## 6. Immediate next contracts

### Contract 1 — Fix A type gate and merge p/q acceptance

Allowed files:

- `src/exo/worker/engines/mlx/mimo_mtp_fast/mtp_generate.py`
- `src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_mtp_generate_acceptance.py`

Inputs:

- Base branch: `integration/mimo-mtp-direct-20260609T1500Z`
- Candidate commit: `41a62bbe`
- Blocker: `basedpyright` reports Unknown propagation from `tokenizer.decode(...)` into `accumulated_text`.

Done when:

```bash
uv run pytest -q \
  src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_cache_semantics.py \
  src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_mtp_generate_acceptance.py

uv run ruff check \
  src/exo/worker/engines/mlx/mimo_mtp_fast/mtp_generate.py \
  src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_mtp_generate_acceptance.py

uv run basedpyright \
  src/exo/worker/engines/mlx/mimo_mtp_fast/mtp_generate.py \
  src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_mtp_generate_acceptance.py
```

Expected: all pass. Commit on integration branch.

Forbidden:

- Do not touch sidecar/routing/perf/smoke files.
- Do not add trace docs.
- Do not broaden tests beyond this gate.

### Contract 2 — Merge B invariants after Contract 1

Allowed files:

- `src/exo/worker/engines/mlx/tests/test_mimo_mtp_fast_decode_invariants.py`
- `src/exo/worker/engines/mlx/mimo_mtp_fast/mtp_generate.py` only if a small hook is required.

Done when p/q residual and rejected-cache invariants pass after A.

### Contract 3 — Port C hot-path history optimization

Allowed file:

- `src/exo/worker/engines/mlx/mimo_mtp_fast/mtp_generate.py`

Done when generated-token-ID history coexists with p/q acceptance and cache semantics.

### Contract 4 — Merge D/E/F late gates

D sidecar, E routing, F smoke harness should be separate contracts after decode correctness passes.

## 7. Monitoring policy

- No long-running coding job may run >20 minutes without cursor or git movement unless explicitly justified.
- No worker pool may be considered active until process count and `kanban list --status running` confirm it.
- If a worker blocks with review-required, do not immediately spawn another worker; first integrate or reject its output.
- If three or more cards block on review, switch to integration/review swarm, not more implementation fanout.
- Stale watchers are paused when their jobs are intentionally cancelled.

## 8. Completion definition for the project

The project is not “done” when tests pass. It is done when:

1. Local correctness gates pass on the integration branch.
2. Code is synced to both cluster nodes intentionally.
3. Live two-node cluster smoke shows a request enters the real MTP fastpath.
4. Runner completion, QA verdict, and acceptance/live evidence are reported separately.
5. Throughput is >=25 tok/s minimum, target >=30 tok/s, stretch >=40 tok/s.
6. Default AR path remains safe and unchanged.

## 9. What Hermes should do next

1. Stop stale jobs/watchers. Done for `job_b7e23460b288` and watcher `1cf674929d21`.
2. Create Contract 1 as a single Kanban card with no parent dependency.
3. Dispatch exactly one implementer for Contract 1.
4. Dispatch exactly one reviewer after Contract 1 commits.
5. Only then continue Contract 2.

No more broad swarms until the first contract is green.
