# Sub-AC 3 Evidence: Worker verification-command discovery

Status: complete

## Scope

Implemented and tested verification-command discovery for worker-related changes on the shared rollout discovery surface.

## Files changed for this Sub-AC

- `scripts/verification_command_discovery.py`
  - Adds `WORKER_VERIFICATION_COMMANDS`.
  - Routes worker-only changed paths to focused runnable worker checks only.
  - Preserves existing benchmark and cross-layer cluster seam discovery behavior.
- `scripts/test_verification_command_discovery.py`
  - Adds `test_worker_related_changes_return_focused_runnable_worker_checks_only`.
  - Asserts worker paths return only worker pytest/ruff/basedpyright commands and exclude API/benchmark checks.

Note: the target worktree also contains other concurrent/untracked Sub-AC files and pre-existing modified files. This report only claims the worker verification-command discovery slice above.

## TDD evidence

RED:

```bash
uv run pytest scripts/test_verification_command_discovery.py -q
```

Observed failure before implementation:

- `test_worker_related_changes_return_focused_runnable_worker_checks_only` failed because worker paths returned cluster benchmark checks:
  - `uv run pytest scripts/test_bench_mimo_mtp_cluster.py -q`
  - cluster live probe command
- This proved the test detected the missing worker-focused discovery behavior.

GREEN/final verification:

```bash
uv run pytest scripts/test_verification_command_discovery.py -q
uv run ruff check scripts/verification_command_discovery.py scripts/test_verification_command_discovery.py
uv run ruff format --check scripts/verification_command_discovery.py scripts/test_verification_command_discovery.py
uv run basedpyright scripts/verification_command_discovery.py scripts/test_verification_command_discovery.py
uv run python3 - <<'PY'
from scripts.verification_command_discovery import discover_verification_commands
for command in discover_verification_commands([
    'src/exo/worker/main.py',
    'src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py',
    'docs/worker.md',
]):
    print(command)
PY
git diff --check
```

Observed final output:

- Pytest: `5 passed in 0.01s`
- Ruff check: `All checks passed!`
- Ruff format check: `2 files already formatted`
- Basedpyright: `0 errors, 0 warnings, 0 notes`
- Direct worker discovery smoke returned exactly:
  - `uv run pytest src/exo/worker/tests src/exo/worker/engines/mlx/tests -q`
  - `uv run ruff check src/exo/worker`
  - `uv run basedpyright src/exo/worker`
- `git diff --check`: clean/no output

## Benchmark/live rows

Not applicable for this verification-command discovery Sub-AC. No cluster API or live benchmark rows were invoked or fabricated.

## Remaining risk

- The worker commands are intentionally focused and runnable discovery output; the full worker pytest suite itself was not executed in this Sub-AC to avoid broad unrelated surface/runtime cost.
- The discovery module is untracked in the current worktree because neighboring Sub-ACs also introduced the shared verification discovery surface concurrently.
