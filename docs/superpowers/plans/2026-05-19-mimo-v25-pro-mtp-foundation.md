# MiMo V2.5 Pro MTP Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the safe foundation for MiMo V2.5 Pro MTP work: git recovery points, duplicate-start guards, and offline MTP artifact evidence without loading a second full MiMo instance.

**Architecture:** This first plan deliberately stops before full-model MTP runtime work. It adds operator-facing recovery docs, a reusable MiMo runtime safety guard, and an offline MTP artifact probe that can run while the current tensor/JACCL TB5 exo cluster is up. Later plans will cover MTP quantized artifact generation, MTP module loading, propose/verify/accept decoding, and distributed promotion to default.

**Tech Stack:** Python 3.13, pytest, safetensors, MLX, exo FastAPI state endpoints, git.

---

## Scope Boundary

This plan implements only the safe foundation from the approved design:

- Backup and recovery records.
- Duplicate-MiMo startup prevention utility.
- Offline MTP source-shard shape probe.
- Tests for the new utility and probe.

It intentionally does not:

- Stop or restart the existing exo cluster.
- Load the full 778 GiB MiMo V2.5 Pro 6-bit model.
- Convert `model_mtp.safetensors` into the production artifact.
- Change `mimo_v2_flash` loading or decode behavior.
- Promote MTP to default runtime.

Those require separate plans after this foundation passes.

## File Structure

- Create `docs/plans/MIMO_V25_PRO_MTP_FOUNDATION_RUNLOG.md`
  - Human recovery and baseline note with branches, commits, runtime state, artifact paths, and benchmark files.
- Create `scripts/mimo_v25_pro_runtime_guard.py`
  - Pure-Python guard used before any future full MiMo startup. It checks exo state JSON, process lists, and node memory snapshots and returns a structured verdict.
- Create `scripts/test_mimo_v25_pro_runtime_guard.py`
  - Unit tests for duplicate active/loading/stale MiMo detection and memory floor behavior.
- Create `scripts/mimo_v25_pro_mtp_artifact_probe.py`
  - Offline MTP shard inspector. It reads safetensors metadata and emits a JSON report. It must not import exo runtime, initialize MLX distributed, or load the full main model.
- Create `scripts/test_mimo_v25_pro_mtp_artifact_probe.py`
  - Unit tests using tiny safetensors fixtures that include `model.mtp.layers.*` tensors.

---

### Task 1: Create Git Recovery Points And Runlog

**Files:**
- Create: `docs/plans/MIMO_V25_PRO_MTP_FOUNDATION_RUNLOG.md`

- [ ] **Step 1: Record current state before creating branches**

Run:

```bash
git status --short --branch
git rev-parse HEAD
git rev-parse fork/con-75-mimo-pro-live-integration-20260504-200136
```

Expected:

- Branch is `con-75-mimo-pro-live-integration-20260504-200136`.
- HEAD is the latest planning commit.
- Worktree may still show existing dirty KV/cache files.

- [ ] **Step 2: Create a baseline backup branch at the known-good pushed commit**

Run:

```bash
git branch backup/mimo-v25-pro-6bit-working-baseline-20260519 1114ff855fa8f5d897388b50ccc6011c205dbf39
git rev-parse backup/mimo-v25-pro-6bit-working-baseline-20260519
```

Expected:

```text
1114ff855fa8f5d897388b50ccc6011c205dbf39
```

- [ ] **Step 3: Save the current dirty KV/cache work as a named patch**

Run:

```bash
mkdir -p docs/plans/artifacts
git diff -- src/exo/worker/engines/mlx/cache.py \
  src/exo/worker/engines/mlx/generator/batch_generate.py \
  src/exo/worker/engines/mlx/generator/generate.py \
  src/exo/worker/tests/unittests/test_mlx/test_kv_prefix_cache.py \
  > docs/plans/artifacts/mimo-v25-pro-kv-cache-wip-20260519.patch
test -s docs/plans/artifacts/mimo-v25-pro-kv-cache-wip-20260519.patch
```

Expected:

- Patch file exists and is non-empty.
- No source code is changed by this step.

- [ ] **Step 4: Write the recovery runlog**

Create `docs/plans/MIMO_V25_PRO_MTP_FOUNDATION_RUNLOG.md` with this content, updating only the observed command outputs from Steps 1-3:

```markdown
# MiMo V2.5 Pro MTP Foundation Runlog

Date: 2026-05-19

## Known-Good Baseline

- Baseline branch: `backup/mimo-v25-pro-6bit-working-baseline-20260519`
- Baseline commit: `1114ff855fa8f5d897388b50ccc6011c205dbf39`
- Runtime branch at planning start: `con-75-mimo-pro-live-integration-20260504-200136`
- Current MTP design spec: `docs/superpowers/specs/2026-05-19-mimo-v25-pro-mtp-design.md`

## Runtime Facts

- Model id: `kernelpool/MiMo-V2.5-Pro-6bit`
- Main artifact: `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/quantized/kernelpool--MiMo-V2.5-Pro-6bit`
- Main artifact size: about `778G` on disk
- Safetensors metadata size: `835299199488` bytes
- Source MTP shard: `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/hf/XiaomiMiMo--MiMo-V2.5-Pro/model_mtp.safetensors`
- Source MTP shard size: about `2.3G`
- Preferred performance placement: tensor/JACCL over TB5

## Throughput Baseline

- Single request: about `22 tok/s`
- Batched aggregate: about `40 tok/s`
- Benchmark artifacts:
  - `/Users/studio2/Documents/Codex/2026-05-19/superpowers-can-we-determine-whether-22/mimo_v25_pro_sustain_probe_20260519.json`
  - `/Users/studio2/Documents/Codex/2026-05-19/superpowers-can-we-determine-whether-22/mimo_v25_pro_concurrency_probe_20260519.json`
  - `/Users/studio2/Documents/Codex/2026-05-19/superpowers-can-we-determine-whether-22/mimo_v25_pro_concurrency8_probe_20260519.json`

## Dirty Work Preserved

- Patch artifact: `docs/plans/artifacts/mimo-v25-pro-kv-cache-wip-20260519.patch`
- Patch scope:
  - `src/exo/worker/engines/mlx/cache.py`
  - `src/exo/worker/engines/mlx/generator/batch_generate.py`
  - `src/exo/worker/engines/mlx/generator/generate.py`
  - `src/exo/worker/tests/unittests/test_mlx/test_kv_prefix_cache.py`

## Recovery Recipe

To return to the known-good baseline code:

```bash
git switch backup/mimo-v25-pro-6bit-working-baseline-20260519
```

To reapply the preserved KV/cache work from the current branch:

```bash
git apply docs/plans/artifacts/mimo-v25-pro-kv-cache-wip-20260519.patch
```

## Safety Rule

Do not start a full MiMo V2.5 Pro model if any MiMo instance is active, loading,
warming, stale-but-resident, or if either Studio node has not recovered memory.
Duplicate full MiMo startup can crash the Mac Studios.
```

- [ ] **Step 5: Commit recovery artifacts**

Run:

```bash
git add docs/plans/MIMO_V25_PRO_MTP_FOUNDATION_RUNLOG.md \
  docs/plans/artifacts/mimo-v25-pro-kv-cache-wip-20260519.patch
git commit -m "Preserve MiMo MTP recovery baseline" \
  -m "Record the known-good MiMo V2.5 Pro 6-bit baseline, benchmark evidence, and the current dirty KV/cache patch before starting MTP foundation work." \
  -m "Constraint: Full MiMo startup is memory-dangerous on the 512 GB Studio nodes" \
  -m "Confidence: high" \
  -m "Scope-risk: narrow" \
  -m "Tested: test -s docs/plans/artifacts/mimo-v25-pro-kv-cache-wip-20260519.patch" \
  -m "Not-tested: No runtime code changed"
```

Expected:

- Commit succeeds.
- Source runtime files remain dirty if they were dirty before this task.

---

### Task 2: Add Duplicate-MiMo Runtime Guard

**Files:**
- Create: `scripts/mimo_v25_pro_runtime_guard.py`
- Create: `scripts/test_mimo_v25_pro_runtime_guard.py`

- [ ] **Step 1: Write failing tests for unsafe active/loading MiMo states**

Create `scripts/test_mimo_v25_pro_runtime_guard.py`:

```python
from scripts import mimo_v25_pro_runtime_guard as guard


def _bytes(gib: int) -> dict[str, dict[str, int]]:
    return {"inBytes": gib * 1024**3}


def test_detects_active_mimo_instance_as_unsafe() -> None:
    state = {
        "instances": {
            "abc": {
                "MlxJacclInstance": {
                    "shardAssignments": {"modelId": guard.MIMO_MODEL_ID}
                }
            }
        },
        "runners": {},
        "nodeMemory": {},
    }

    verdict = guard.evaluate_state(state, processes=[], min_available_bytes=200)

    assert verdict.safe is False
    assert "active MiMo instance abc" in verdict.reasons


def test_detects_loading_mimo_runner_as_unsafe() -> None:
    state = {
        "instances": {},
        "runners": {
            "runner-a": {"RunnerLoading": {"layersLoaded": 10, "totalLayers": 70}}
        },
        "nodeMemory": {},
    }

    verdict = guard.evaluate_state(state, processes=[], min_available_bytes=200)

    assert verdict.safe is False
    assert "runner runner-a is loading" in verdict.reasons


def test_detects_resident_mimo_process_as_unsafe() -> None:
    verdict = guard.evaluate_state(
        {"instances": {}, "runners": {}, "nodeMemory": {}},
        processes=[
            "/Users/studio2/exo/.venv/bin/python3 -m exo "
            "kernelpool/MiMo-V2.5-Pro-6bit"
        ],
        min_available_bytes=200,
    )

    assert verdict.safe is False
    assert "resident MiMo process" in verdict.reasons[0]


def test_detects_low_node_memory_as_unsafe() -> None:
    state = {
        "instances": {},
        "runners": {},
        "nodeMemory": {
            "studio1": {"ramAvailable": _bytes(100)},
            "studio2": {"ramAvailable": _bytes(260)},
        },
    }

    verdict = guard.evaluate_state(
        state,
        processes=[],
        min_available_bytes=200 * 1024**3,
    )

    assert verdict.safe is False
    assert "node studio1 below memory floor" in verdict.reasons


def test_clean_state_is_safe() -> None:
    state = {
        "instances": {},
        "runners": {},
        "nodeMemory": {
            "studio1": {"ramAvailable": _bytes(300)},
            "studio2": {"ramAvailable": _bytes(300)},
        },
    }

    verdict = guard.evaluate_state(
        state,
        processes=[],
        min_available_bytes=200 * 1024**3,
    )

    assert verdict.safe is True
    assert verdict.reasons == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest scripts/test_mimo_v25_pro_runtime_guard.py -q
```

Expected:

- FAIL because `scripts/mimo_v25_pro_runtime_guard.py` does not exist.

- [ ] **Step 3: Implement the guard module**

Create `scripts/mimo_v25_pro_runtime_guard.py`:

```python
#!/usr/bin/env python3
"""Fail-closed safety guard before full MiMo V2.5 Pro startup."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any, Final, Sequence
from urllib import request

MIMO_MODEL_ID: Final[str] = "kernelpool/MiMo-V2.5-Pro-6bit"
MIMO_MODEL_MARKERS: Final[tuple[str, ...]] = (
    "MiMo-V2.5-Pro",
    "mimo-v25-pro",
    "kernelpool/MiMo-V2.5-Pro-6bit",
    "XiaomiMiMo/MiMo-V2.5-Pro",
)
DEFAULT_EXO_URL: Final[str] = "http://127.0.0.1:52415"
DEFAULT_MIN_AVAILABLE_GIB: Final[int] = 200


@dataclass(frozen=True)
class GuardVerdict:
    safe: bool
    reasons: list[str]


def _unwrap_instance(instance: dict[str, Any]) -> dict[str, Any]:
    if len(instance) != 1:
        return {}
    value = next(iter(instance.values()))
    return value if isinstance(value, dict) else {}


def _model_id_from_instance(instance: dict[str, Any]) -> str | None:
    inner = _unwrap_instance(instance)
    shard_assignments = inner.get("shardAssignments")
    if isinstance(shard_assignments, dict):
        model_id = shard_assignments.get("modelId")
        return str(model_id) if model_id is not None else None
    return None


def _is_mimo_text(value: str) -> bool:
    return any(marker in value for marker in MIMO_MODEL_MARKERS)


def evaluate_state(
    state: dict[str, Any],
    *,
    processes: Sequence[str],
    min_available_bytes: int,
) -> GuardVerdict:
    reasons: list[str] = []

    instances = state.get("instances", {})
    if isinstance(instances, dict):
        for instance_id, instance in instances.items():
            if isinstance(instance, dict):
                model_id = _model_id_from_instance(instance)
                if model_id is not None and _is_mimo_text(model_id):
                    reasons.append(f"active MiMo instance {instance_id}")

    runners = state.get("runners", {})
    if isinstance(runners, dict):
        for runner_id, runner in runners.items():
            if isinstance(runner, dict):
                if "RunnerLoading" in runner:
                    reasons.append(f"runner {runner_id} is loading")
                if "RunnerWarming" in runner:
                    reasons.append(f"runner {runner_id} is warming")

    for proc in processes:
        if _is_mimo_text(proc) and "rg " not in proc:
            reasons.append(f"resident MiMo process: {proc[:180]}")

    node_memory = state.get("nodeMemory", {})
    if isinstance(node_memory, dict):
        for node_id, memory in node_memory.items():
            if not isinstance(memory, dict):
                continue
            available = (
                memory.get("ramAvailable", {}).get("inBytes", 0)
                if isinstance(memory.get("ramAvailable"), dict)
                else 0
            )
            if int(available) < min_available_bytes:
                reasons.append(f"node {node_id} below memory floor")

    return GuardVerdict(safe=not reasons, reasons=reasons)


def fetch_exo_state(base_url: str) -> dict[str, Any]:
    with request.urlopen(f"{base_url.rstrip('/')}/state", timeout=10) as response:
        data = json.loads(response.read().decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("exo /state did not return a JSON object")
    return data


def collect_processes() -> list[str]:
    output = subprocess.check_output(["ps", "axo", "command="], text=True)
    return [line.strip() for line in output.splitlines() if line.strip()]


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exo-url", default=DEFAULT_EXO_URL)
    parser.add_argument("--state-json", type=str, default=None)
    parser.add_argument(
        "--min-available-gib",
        type=int,
        default=DEFAULT_MIN_AVAILABLE_GIB,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    state = (
        json.loads(args.state_json)
        if args.state_json
        else fetch_exo_state(args.exo_url)
    )
    verdict = evaluate_state(
        state,
        processes=collect_processes(),
        min_available_bytes=args.min_available_gib * 1024**3,
    )
    print(json.dumps(asdict(verdict), indent=2))
    return 0 if verdict.safe else 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run pytest scripts/test_mimo_v25_pro_runtime_guard.py -q
```

Expected:

```text
5 passed
```

- [ ] **Step 5: Run static checks for the new script**

Run:

```bash
uv run ruff check scripts/mimo_v25_pro_runtime_guard.py scripts/test_mimo_v25_pro_runtime_guard.py
```

Expected:

```text
All checks passed!
```

- [ ] **Step 6: Commit the guard**

Run:

```bash
git add scripts/mimo_v25_pro_runtime_guard.py scripts/test_mimo_v25_pro_runtime_guard.py
git commit -m "Guard against duplicate MiMo runtime startup" \
  -m "Add a fail-closed preflight utility for future full-model MiMo experiments so duplicate active/loading/resident instances and low-memory nodes are detected before startup." \
  -m "Constraint: Starting a second full MiMo instance can crash the Studio nodes" \
  -m "Confidence: high" \
  -m "Scope-risk: narrow" \
  -m "Tested: uv run pytest scripts/test_mimo_v25_pro_runtime_guard.py -q; uv run ruff check scripts/mimo_v25_pro_runtime_guard.py scripts/test_mimo_v25_pro_runtime_guard.py" \
  -m "Not-tested: Live cluster shutdown/startup behavior"
```

---

### Task 3: Add Offline MTP Artifact Probe

**Files:**
- Create: `scripts/mimo_v25_pro_mtp_artifact_probe.py`
- Create: `scripts/test_mimo_v25_pro_mtp_artifact_probe.py`

- [ ] **Step 1: Write failing tests for MTP tensor coverage**

Create `scripts/test_mimo_v25_pro_mtp_artifact_probe.py`:

```python
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from scripts import mimo_v25_pro_mtp_artifact_probe as probe


def _write_fixture(path: Path) -> None:
    save_file(
        {
            "model.mtp.layers.0.self_attn.qkv_proj.weight": torch.zeros(
                (4, 4), dtype=torch.float16
            ),
            "model.mtp.layers.0.self_attn.qkv_proj.weight_scale_inv": torch.ones(
                (1, 1), dtype=torch.float32
            ),
            "model.mtp.layers.1.self_attn.qkv_proj.weight": torch.zeros(
                (4, 4), dtype=torch.float16
            ),
            "model.mtp.layers.2.mlp.down_proj.weight": torch.zeros(
                (4, 4), dtype=torch.float16
            ),
            "model.layers.0.mlp.down_proj.weight": torch.zeros(
                (4, 4), dtype=torch.float16
            ),
        },
        path,
    )


def test_probe_reports_only_mtp_tensors(tmp_path: Path) -> None:
    shard = tmp_path / "model_mtp.safetensors"
    _write_fixture(shard)

    report = probe.probe_mtp_shard(shard)

    assert report["source_file"] == str(shard)
    assert report["tensor_count"] == 4
    assert report["layers"] == [0, 1, 2]
    assert "model.layers.0.mlp.down_proj.weight" not in report["tensors"]
    assert report["tensors"]["model.mtp.layers.0.self_attn.qkv_proj.weight"][
        "shape"
    ] == [4, 4]


def test_probe_marks_missing_expected_layers(tmp_path: Path) -> None:
    shard = tmp_path / "model_mtp.safetensors"
    save_file(
        {
            "model.mtp.layers.0.self_attn.qkv_proj.weight": torch.zeros(
                (4, 4), dtype=torch.float16
            )
        },
        shard,
    )

    report = probe.probe_mtp_shard(shard)

    assert report["layers"] == [0]
    assert report["missing_expected_layers"] == [1, 2]
    assert report["complete_expected_layers"] is False


def test_probe_cli_writes_json(tmp_path: Path) -> None:
    shard = tmp_path / "model_mtp.safetensors"
    output = tmp_path / "report.json"
    _write_fixture(shard)

    exit_code = probe.main([str(shard), "--json-out", str(output)])

    assert exit_code == 0
    data = json.loads(output.read_text())
    assert data["complete_expected_layers"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest scripts/test_mimo_v25_pro_mtp_artifact_probe.py -q
```

Expected:

- FAIL because `scripts/mimo_v25_pro_mtp_artifact_probe.py` does not exist.

- [ ] **Step 3: Implement offline artifact probe**

Create `scripts/mimo_v25_pro_mtp_artifact_probe.py`:

```python
#!/usr/bin/env python3
"""Inspect MiMo V2.5 Pro MTP safetensors without loading the full model."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Final, Sequence, TypedDict, cast

from safetensors import safe_open

DEFAULT_MTP_SHARD: Final[Path] = Path(
    "/Volumes/GLM5-NVMe/exo/mimo-v25-pro/hf/XiaomiMiMo--MiMo-V2.5-Pro/model_mtp.safetensors"
)
EXPECTED_MTP_LAYERS: Final[tuple[int, ...]] = (0, 1, 2)
MTP_LAYER_RE: Final[re.Pattern[str]] = re.compile(
    r"^model\\.mtp\\.layers\\.(\\d+)\\."
)


class TensorReport(TypedDict):
    dtype: str
    shape: list[int]


class MtpReport(TypedDict):
    source_file: str
    tensor_count: int
    layers: list[int]
    missing_expected_layers: list[int]
    complete_expected_layers: bool
    tensors: dict[str, TensorReport]


def _mtp_layer(tensor_name: str) -> int | None:
    match = MTP_LAYER_RE.match(tensor_name)
    return int(match.group(1)) if match else None


def probe_mtp_shard(path: Path) -> MtpReport:
    if not path.is_file():
        raise FileNotFoundError(f"MTP shard not found: {path}")

    tensors: dict[str, TensorReport] = {}
    layers: set[int] = set()
    with safe_open(path, framework="pt", device="cpu") as handle:
        for name in sorted(handle.keys()):
            layer = _mtp_layer(name)
            if layer is None:
                continue
            tensor_slice = handle.get_slice(name)
            layers.add(layer)
            tensors[name] = {
                "dtype": tensor_slice.get_dtype(),
                "shape": [int(dim) for dim in tensor_slice.get_shape()],
            }

    sorted_layers = sorted(layers)
    missing = [layer for layer in EXPECTED_MTP_LAYERS if layer not in layers]
    return {
        "source_file": str(path),
        "tensor_count": len(tensors),
        "layers": sorted_layers,
        "missing_expected_layers": missing,
        "complete_expected_layers": not missing,
        "tensors": tensors,
    }


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_MTP_SHARD)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    report = probe_mtp_shard(cast(Path, args.path))
    text = json.dumps(cast(dict[str, Any], report), indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run pytest scripts/test_mimo_v25_pro_mtp_artifact_probe.py -q
```

Expected:

```text
3 passed
```

- [ ] **Step 5: Run the probe against the real MTP shard**

Run:

```bash
uv run python scripts/mimo_v25_pro_mtp_artifact_probe.py \
  /Volumes/GLM5-NVMe/exo/mimo-v25-pro/hf/XiaomiMiMo--MiMo-V2.5-Pro/model_mtp.safetensors \
  --json-out docs/plans/artifacts/mimo-v25-pro-mtp-shape-report-20260519.json
python3 - <<'PY'
import json
from pathlib import Path
data = json.loads(Path("docs/plans/artifacts/mimo-v25-pro-mtp-shape-report-20260519.json").read_text())
print(data["tensor_count"])
print(data["layers"])
print(data["complete_expected_layers"])
PY
```

Expected:

- `complete_expected_layers` prints `True`.
- `layers` prints `[0, 1, 2]`.
- `tensor_count` is greater than `0`.

- [ ] **Step 6: Run static checks**

Run:

```bash
uv run ruff check scripts/mimo_v25_pro_mtp_artifact_probe.py scripts/test_mimo_v25_pro_mtp_artifact_probe.py
```

Expected:

```text
All checks passed!
```

- [ ] **Step 7: Commit the MTP probe**

Run:

```bash
git add scripts/mimo_v25_pro_mtp_artifact_probe.py \
  scripts/test_mimo_v25_pro_mtp_artifact_probe.py \
  docs/plans/artifacts/mimo-v25-pro-mtp-shape-report-20260519.json
git commit -m "Probe MiMo MTP artifact safely offline" \
  -m "Add an offline safetensors inspector for model_mtp.safetensors so MTP layer coverage and tensor shapes can be verified without starting exo or loading the full MiMo model." \
  -m "Constraint: The current exo cluster may remain up because this probe reads only safetensors metadata" \
  -m "Confidence: high" \
  -m "Scope-risk: narrow" \
  -m "Tested: uv run pytest scripts/test_mimo_v25_pro_mtp_artifact_probe.py -q; uv run ruff check scripts/mimo_v25_pro_mtp_artifact_probe.py scripts/test_mimo_v25_pro_mtp_artifact_probe.py; real MTP shard probe" \
  -m "Not-tested: MTP quantization or runtime decode"
```

---

### Task 4: Run Foundation Verification

**Files:**
- Verify: `docs/plans/MIMO_V25_PRO_MTP_FOUNDATION_RUNLOG.md`
- Verify: `scripts/mimo_v25_pro_runtime_guard.py`
- Verify: `scripts/mimo_v25_pro_mtp_artifact_probe.py`

- [ ] **Step 1: Run focused tests**

Run:

```bash
uv run pytest \
  scripts/test_mimo_v25_pro_runtime_guard.py \
  scripts/test_mimo_v25_pro_mtp_artifact_probe.py \
  -q
```

Expected:

```text
8 passed
```

- [ ] **Step 2: Run lint**

Run:

```bash
uv run ruff check \
  scripts/mimo_v25_pro_runtime_guard.py \
  scripts/test_mimo_v25_pro_runtime_guard.py \
  scripts/mimo_v25_pro_mtp_artifact_probe.py \
  scripts/test_mimo_v25_pro_mtp_artifact_probe.py
```

Expected:

```text
All checks passed!
```

- [ ] **Step 3: Run the guard against the current live state**

Run:

```bash
uv run python scripts/mimo_v25_pro_runtime_guard.py --exo-url http://127.0.0.1:52415
```

Expected while the current MiMo cluster is up:

```json
{
  "safe": false,
  "reasons": [
    "active MiMo instance 465e23f7-c776-4e9d-8760-cf768db9c56c"
  ]
}
```

This is a pass condition. Additional reasons are acceptable when they identify
loading runners, resident MiMo processes, or low memory. The required assertion
is that `safe` is `false` and at least one reason starts with
`active MiMo instance`.

- [ ] **Step 4: Confirm no runtime source files were accidentally staged**

Run:

```bash
git diff --cached --name-only
git status --short
```

Expected:

- No staged files unless preparing the final verification commit.
- Existing unrelated dirty KV/cache runtime files remain visible and unchanged from before this plan unless a previous task intentionally committed the patch artifact only.

- [ ] **Step 5: Append foundation verification evidence**

Run:

```bash
cat >> docs/plans/MIMO_V25_PRO_MTP_FOUNDATION_RUNLOG.md <<'MD'

## Foundation Verification

- Focused tests: `uv run pytest scripts/test_mimo_v25_pro_runtime_guard.py scripts/test_mimo_v25_pro_mtp_artifact_probe.py -q`
- Lint: `uv run ruff check scripts/mimo_v25_pro_runtime_guard.py scripts/test_mimo_v25_pro_runtime_guard.py scripts/mimo_v25_pro_mtp_artifact_probe.py scripts/test_mimo_v25_pro_mtp_artifact_probe.py`
- Real MTP shape report: `docs/plans/artifacts/mimo-v25-pro-mtp-shape-report-20260519.json`
- Live duplicate-start guard: refused startup while current MiMo instance was resident
MD
```

Expected:

- `docs/plans/MIMO_V25_PRO_MTP_FOUNDATION_RUNLOG.md` has a `Foundation Verification` section.

- [ ] **Step 6: Commit foundation verification evidence**

```bash
git add docs/plans/MIMO_V25_PRO_MTP_FOUNDATION_RUNLOG.md
git commit -m "Record MiMo MTP foundation verification" \
  -m "Record focused test, lint, real MTP shard probe, and live guard evidence for the MTP foundation phase." \
  -m "Confidence: high" \
  -m "Scope-risk: narrow" \
  -m "Tested: focused pytest, ruff, real MTP shard probe, live duplicate-start guard" \
  -m "Not-tested: Full-model MTP runtime"
```

---

## Handoff To Next Plan

After this plan passes, write a second plan for MTP quantized artifact generation. That next plan should cover:

- Experimental artifact identity and model card.
- `model_mtp.safetensors` conversion into MLX affine 6-bit output.
- Quantization manifest/index updates.
- Standalone MTP-only module loading probe.

Do not plan full distributed MTP decode until the experimental MTP artifact and MTP-only module probe both pass.
