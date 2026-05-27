# MiMo V2.5 Pro MTP Ralphathon Design

## Purpose

Create a Ralphathon-worthy autonomous workflow for getting MiMo V2.5 Pro MTP from the current blocked state to a verified live deployment. The hard success gate is **30+ output tokens/second on any MTP setting with correctness intact**. The stretch gate is **40+ output tokens/second on any MTP setting**, attempted only after the 30 tok/s proof is preserved.

The workflow intentionally uses two Ralph loops instead of one long loop. The prior run made real progress but drifted across local fixes, sync issues, launcher failures, live attempts, and performance triage. This design separates those concerns so local correctness and residency fixes earn permission for guarded live deployment.

## Scope

In scope:

- Design two Ralph task folders for `/Users/studio2/exo`.
- Define loop boundaries, evidence commands, required outputs, guardrails, and completion gates.
- Require progress memory and blocker classification to prevent repeated inconclusive iterations.
- Keep the live deployment loop safe around Studio1/Studio2 synchronization and duplicate full-MiMo residency.

Out of scope for this spec:

- Implementing the RALPH.md files.
- Editing MTP/cache/runtime code.
- Running D2 or D3 live attempts.
- Syncing Studio1 or changing remote state.

Those actions belong to the implementation plan and then the Ralph loops.

## Proposed File Layout

```text
ralphathons/mimo-v25-pro-mtp-ralphathon/
  README.md
  01-local-residency-fix/
    RALPH.md
    RALPH_PROGRESS.md
    OPEN_QUESTIONS.md
    HANDOFF_TO_LIVE.md
  02-guarded-live-deployment/
    RALPH.md
    RALPH_PROGRESS.md
    OPEN_QUESTIONS.md
    LIVE_THROUGHPUT_REPORT.md
    ROLLBACK.md
```

The first loop produces the handoff that authorizes the second loop. The second loop consumes that handoff and focuses on guarded deployment and benchmarking.

## Loop 1: Local Residency Fix

### Goal

Diagnose and fix the local root cause for poor MTP speed, currently suspected to involve accepted-prefix residency not advancing because rollback/cache behavior leaves `resident_token_count` at zero after accepted MTP tokens.

### Boundaries

Loop 1 may:

- Read existing MTP rollout docs, traces, tests, and runtime code.
- Add or update local regression tests.
- Add focused instrumentation needed to prove accepted-prefix residency and rollback behavior.
- Edit MTP controller/cache/runtime code when tests justify the change.
- Commit small verified local changes.
- Produce `HANDOFF_TO_LIVE.md` when local evidence justifies deployment.

Loop 1 must not:

- Start full live MiMo on Studio1/Studio2.
- Sync Studio1.
- Run D2 or D3 live deployment attempts.
- Declare deployment success.

### Required Evidence

Before completing, Loop 1 must produce evidence that:

1. Accepted MTP tokens advance accepted-prefix or resident-token state in the relevant cache/controller path.
2. Rejected draft tokens roll back without corrupting the verifier cache.
3. Existing prefix-cache, logprob, stop-sequence, and non-MTP fallback behavior remain protected by tests or explicit unchanged evidence.
4. A deterministic local probe or focused test shows nonzero residency progress under an MTP-eligible path.
5. `HANDOFF_TO_LIVE.md` names the exact commit, test commands, remaining risk, and recommended live MTP settings to try first.

### Suggested Ralph Commands

The implementation plan should verify exact command names, but Loop 1 should include commands shaped like:

- focused MTP controller tests;
- focused KV/prefix-cache tests;
- focused MiMo MTP runtime or probe tests;
- `git status --short --branch`;
- `git log --oneline -10`.

Any long test output should be summarized by the loop through context-mode or equivalent output filtering.

### Completion Gate

Loop 1 should use:

- `completion_promise: DONE`
- `completion_gate: required`
- required outputs:
  - `HANDOFF_TO_LIVE.md`
  - `OPEN_QUESTIONS.md`

`OPEN_QUESTIONS.md` must contain no unresolved P0/P1 questions before completion.

## Loop 2: Guarded Live Deployment

### Goal

Use Loop 1's handoff to run safe Studio1/Studio2 live attempts and produce a verified deployment result. Hard success is **30+ tok/s on any MTP setting with correctness intact**. Stretch success is **40+ tok/s on any MTP setting**, only after the 30 tok/s proof is captured.

### Boundaries

Loop 2 may:

- Read Loop 1 handoff and local/live artifacts.
- Run read-only preflights.
- Verify Studio1 and Studio2 commit alignment.
- Verify launcher and venv readiness.
- Run duplicate full-MiMo process guards.
- Run one guarded live attempt at a time after preflight passes.
- Tune MTP settings to seek 30+ tok/s, then optionally 40+ tok/s.
- Record artifacts and rollback instructions.

Loop 2 must not:

- Start full MiMo if duplicate process guard fails.
- Run a live attempt when Studio1/Studio2 commits are mismatched.
- Hide or overwrite failed live evidence.
- Treat 40+ tok/s as required for completion.
- Perform broad source-code exploration unless Loop 1's handoff is proven stale or false.

If source fixes are needed, Loop 2 should either make the smallest safe fix with evidence or stop with a blocker that routes back to Loop 1.

### Required Evidence

For a successful 30 tok/s completion, Loop 2 must record:

1. Studio1 and Studio2 commit hashes.
2. Launch command or wrapper identity.
3. MTP settings used.
4. Prompt and response sample sufficient to judge correctness.
5. Measured output tok/s.
6. Accepted/rejected token telemetry when available.
7. Accepted-prefix or resident-token evidence when available.
8. Artifact paths for logs/traces.
9. Rollback instructions.

For stretch 40+ tok/s, the report should add a separate section so the 30 tok/s proof remains valid even if stretch tuning fails.

### Suggested Ralph Commands

The implementation plan should verify exact command names and paths, but Loop 2 should include commands shaped like:

- Loop 1 handoff summary;
- Studio2 branch/status/commit check;
- Studio1 remote commit check;
- duplicate MiMo process guard;
- launcher/venv sanity check;
- latest live artifact summary;
- optional acceptance command that verifies `LIVE_THROUGHPUT_REPORT.md`, `ROLLBACK.md`, and `OPEN_QUESTIONS.md` are ready.

### Completion Gate

Loop 2 should use:

- `completion_promise: DONE`
- `completion_gate: required`
- required outputs:
  - `LIVE_THROUGHPUT_REPORT.md`
  - `ROLLBACK.md`
  - `OPEN_QUESTIONS.md`

`OPEN_QUESTIONS.md` must contain no unresolved P0/P1 questions before completion. The acceptance gate should fail if the throughput report does not contain a measured 30+ tok/s result with correctness evidence.

## Blocker Classification

Every unresolved blocker must be classified as one of:

- local test failure;
- MTP disabled or not selected;
- low draft-token acceptance;
- accepted-prefix or resident-token state still zero;
- verifier/rollback/cache corruption risk;
- verification overhead too high;
- tensor placement or sharding issue;
- Studio1/Studio2 sync mismatch;
- launcher or venv issue;
- duplicate resident MiMo process;
- unknown with exact evidence and next diagnostic.

The loops must not repeatedly report generic "deployment blocked" status. A blocker either becomes a local task, a live preflight task, or a P0/P1 open question.

## Safety Rules

Both loops must block dangerous operations such as `git push`, broad destructive `rm -rf`, package publishing, and secret-bearing file reads. Loop 2 must also protect live cluster safety:

- one full MiMo live attempt at a time;
- no launch while duplicate process guard fails;
- no launch with Studio1/Studio2 commit mismatch;
- rollback instructions must exist before declaring success;
- failed artifacts are preserved rather than overwritten.

## Progress Memory

Each loop should maintain `RALPH_PROGRESS.md` with:

- completed iterations and commits;
- current top hypothesis;
- current blocker classification;
- next concrete action;
- artifacts worth preserving.

This prevents fresh Ralph iterations from rediscovering old evidence or repeating stale sync/launcher mistakes.

## Launch Model

The generated task folders are launched with Pi slash commands, not shell execution:

```text
/ralph --path ./ralphathons/mimo-v25-pro-mtp-ralphathon/01-local-residency-fix
/ralph --path ./ralphathons/mimo-v25-pro-mtp-ralphathon/02-guarded-live-deployment
```

Loop 2 should not be launched until Loop 1 has completed and `HANDOFF_TO_LIVE.md` is reviewed.

## Testing The Ralphathon Setup

The implementation plan should include verification that:

- each `RALPH.md` frontmatter parses;
- command names referenced in the body match frontmatter command names;
- required output paths are relative and valid;
- guardrail patterns block dangerous commands;
- completion gates require the expected files;
- the live acceptance check rejects reports below 30 tok/s;
- launch instructions use `/ralph --path ...` and never run `/ralph` through bash.
