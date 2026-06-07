# Sub-AC 4 Evidence — AC mirroring orchestration boundary

Timestamp: 2026-06-06 20:50 local

## Scope

Implemented and tested the AC mirroring orchestration boundary so it invokes only mirroring dependencies and explicitly does not call unrelated repo validation logic.

## Files changed

- `scripts/rollout_seed_mirroring.py`
  - Added typed `AcMirroringBoundaryDependencies` and `AcMirroringBoundaryResult` dataclasses.
  - Added `orchestrate_ac_mirroring_boundary(...)`, which calls only:
    1. `plan_mirror()`
    2. `persist_files(plan)`
    3. `persist_beads(plan)`
  - Kept `validate_repo_root` and `discover_verification_commands` as injected sentinel dependencies for boundary tests, but the orchestrator does not invoke them.
- `scripts/test_rollout_seed_mirroring.py`
  - Added `test_ac_mirroring_boundary_invokes_only_mirroring_dependencies` proving call order and non-invocation of unrelated validation logic.

## TDD evidence

RED command:

```bash
uv run pytest scripts/test_rollout_seed_mirroring.py::test_ac_mirroring_boundary_invokes_only_mirroring_dependencies
```

Observed RED failure before implementation:

```text
ImportError: cannot import name 'AcMirroringBoundaryDependencies' from 'scripts.rollout_seed_mirroring'
```

GREEN/focused command after implementation:

```bash
uv run pytest scripts/test_rollout_seed_mirroring.py::test_ac_mirroring_boundary_invokes_only_mirroring_dependencies
```

Observed result:

```text
1 passed in 0.01s
```

## Verification commands run

```bash
uv run pytest scripts/test_rollout_seed_mirroring.py::test_ac_mirroring_boundary_invokes_only_mirroring_dependencies
uv run pytest scripts/test_rollout_seed_mirroring.py
uv run ruff check scripts/rollout_seed_mirroring.py scripts/test_rollout_seed_mirroring.py
git diff --check -- scripts/rollout_seed_mirroring.py scripts/test_rollout_seed_mirroring.py
```

Observed final results:

```text
scripts/test_rollout_seed_mirroring.py . [100%]
1 passed in 0.01s

scripts/test_rollout_seed_mirroring.py .... [100%]
4 passed in 0.01s

All checks passed!
```

`git diff --check` completed with exit code 0 and no output.

## Blockers

None for this Sub-AC.

## Remaining risk

- This Sub-AC only proves the orchestration boundary contract. Broader Beads CLI/file persistence integration remains covered by separate mirroring modules/tests and by later AC closeout.
- The working tree contains other governed rollout changes from parallel AC workers; this evidence is scoped only to `scripts/rollout_seed_mirroring.py` and `scripts/test_rollout_seed_mirroring.py`.
