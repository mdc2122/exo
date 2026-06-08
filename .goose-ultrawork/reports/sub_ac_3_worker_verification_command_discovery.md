# Sub-AC 3 Evidence: Worker verification-command discovery

Status: complete

## Scope

Implemented and tested worker-change verification command discovery that maps worker execution related changed files to focused runnable worker test commands.

This Sub-AC specifically covers changed-file command discovery for:

- worker implementation files under `src/exo/worker/`
- worker execution contract files under `src/exo/shared/types/tasks.py` and `src/exo/shared/types/text_generation.py`
- `.goose-ultrawork`-relative paths such as `../src/exo/api/main.py`

## Files changed for this Sub-AC

- `scripts/verification_command_discovery.py`
  - Adds normalization for `./` and `.goose-ultrawork`-relative `../` changed paths.
  - Keeps worker-only changes mapped to focused worker commands:
    - `uv run pytest src/exo/worker/tests src/exo/worker/engines/mlx/tests -q`
    - `uv run ruff check src/exo/worker`
    - `uv run basedpyright src/exo/worker`
  - Adds focused worker execution contract routing for `src/exo/shared/types/tasks.py` and `src/exo/shared/types/text_generation.py` when changed without API/worker cross-layer files:
    - `uv run pytest src/exo/worker/tests src/exo/worker/engines/mlx/tests -q`
    - `uv run ruff check src/exo/worker <contract paths>`
    - `uv run basedpyright src/exo/worker <contract paths>`
  - Preserves cluster benchmark routing for explicit benchmark harness changes and cross-layer API/shared/worker cluster seam changes.
- `scripts/test_verification_command_discovery.py`
  - Adds `test_worker_execution_contract_changes_return_focused_worker_checks` for shared worker execution contracts.
  - Keeps/extends coverage for task type model changes, `.goose-ultrawork` relative API paths, and benchmark telemetry discovery.

Note: the target worktree contains additional modified/untracked files owned by sibling AC work. This report only claims the verification-command discovery files listed above.

## TDD evidence

RED command:

```bash
cd .. && uv run pytest scripts/test_verification_command_discovery.py -q
```

Observed failure before implementation:

- `test_worker_execution_contract_changes_return_focused_worker_checks` failed because shared worker execution contract paths:
  - `./src/exo/shared/types/tasks.py`
  - `src/exo/shared/types/text_generation.py`
- returned cluster benchmark commands instead of focused worker commands.
- The unexpected output included:
  - `uv run pytest scripts/test_bench_mimo_mtp_cluster.py -q`
  - the live cluster benchmark smoke command with `--list-models`

This proved the test detected the missing worker-focused discovery behavior.

GREEN/final verification:

```bash
cd .. && uv run pytest scripts/test_verification_command_discovery.py -q && \
  uv run ruff check scripts/verification_command_discovery.py scripts/test_verification_command_discovery.py && \
  uv run ruff format --check scripts/verification_command_discovery.py scripts/test_verification_command_discovery.py && \
  uv run basedpyright scripts/verification_command_discovery.py scripts/test_verification_command_discovery.py && \
  uv run python3 - <<'PY'
from scripts.verification_command_discovery import discover_verification_commands
for command in discover_verification_commands([
    './src/exo/shared/types/tasks.py',
    'src/exo/shared/types/text_generation.py',
    'README.md',
]):
    print(command)
PY

git diff --check -- scripts/verification_command_discovery.py scripts/test_verification_command_discovery.py
```

Observed final output:

- Pytest: `9 passed in 0.01s`
- Ruff check: `All checks passed!`
- Ruff format check: `2 files already formatted`
- Basedpyright: `0 errors, 0 warnings, 0 notes`
- Direct worker execution contract discovery smoke returned exactly:
  - `uv run pytest src/exo/worker/tests src/exo/worker/engines/mlx/tests -q`
  - `uv run ruff check src/exo/worker src/exo/shared/types/tasks.py src/exo/shared/types/text_generation.py`
  - `uv run basedpyright src/exo/worker src/exo/shared/types/tasks.py src/exo/shared/types/text_generation.py`
- `git diff --check -- scripts/verification_command_discovery.py scripts/test_verification_command_discovery.py`: clean/no output

## Benchmark/live rows

Not applicable for this verification-command discovery Sub-AC. No cluster API or live benchmark rows were invoked or fabricated.

## Remaining risk

- The discovery output is intentionally focused and runnable; the full worker command suite itself was not executed in this Sub-AC to avoid broad unrelated runtime cost.
- Sibling AC work has additional dirty files in the worktree, so only the two discovery files above are claimed here.
