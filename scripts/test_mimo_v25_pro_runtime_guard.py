import json
from typing import cast

import pytest

from scripts import mimo_v25_pro_runtime_guard as guard


def _bytes(gib: int) -> dict[str, int]:
    return {"inBytes": gib * 1024**3}


def _parse_json_object(raw_json: str) -> dict[str, object]:
    decoded = cast(object, json.loads(raw_json))
    if not isinstance(decoded, dict):
        raise AssertionError("expected JSON object")
    candidate = cast(dict[object, object], decoded)
    if not all(isinstance(key, str) for key in candidate):
        raise AssertionError("expected JSON object with string keys")
    return cast(dict[str, object], candidate)


def _reason_at(verdict: dict[str, object], index: int) -> str:
    reasons = verdict.get("reasons")
    if not isinstance(reasons, list):
        raise AssertionError("expected reasons list")
    reason = cast(object, reasons[index])
    if not isinstance(reason, str):
        raise AssertionError("expected string reason")
    return reason


def _empty_state(base_url: str) -> dict[str, object]:
    del base_url
    return {"instances": {}, "runners": {}, "nodeMemory": {}}


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


def test_detects_warming_mimo_runner_as_unsafe() -> None:
    state = {
        "instances": {},
        "runners": {
            "runner-a": {"RunnerWarmingUp": {"layersLoaded": 10, "totalLayers": 70}}
        },
        "nodeMemory": {},
    }

    verdict = guard.evaluate_state(state, processes=[], min_available_bytes=200)

    assert verdict.safe is False
    assert "runner runner-a is warming" in verdict.reasons


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


def test_main_fail_closes_on_malformed_state_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = guard.main(["--state-json", "{not-json"])

    captured = capsys.readouterr()
    verdict = _parse_json_object(captured.out)

    assert exit_code == 2
    assert verdict["safe"] is False
    assert "failed to parse --state-json" in _reason_at(verdict, 0)
    assert captured.err == ""


def test_main_fail_closes_on_invalid_memory_value_in_state(
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = {
        "instances": {},
        "runners": {},
        "nodeMemory": {
            "studio1": {"ramAvailable": {"inBytes": "not-a-number"}},
        },
    }

    exit_code = guard.main(["--state-json", json.dumps(state)])

    captured = capsys.readouterr()
    verdict = _parse_json_object(captured.out)

    assert exit_code == 2
    assert verdict["safe"] is False
    assert (
        "failed to evaluate state" in _reason_at(verdict, 0)
        or "invalid memory value" in _reason_at(verdict, 0)
    )
    assert captured.err == ""


def test_main_fail_closes_on_state_fetch_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _raise_fetch_error(base_url: str) -> dict[str, object]:
        raise RuntimeError(f"boom from {base_url}")

    monkeypatch.setattr(guard, "fetch_exo_state", _raise_fetch_error)

    exit_code = guard.main([])

    captured = capsys.readouterr()
    verdict = _parse_json_object(captured.out)

    assert exit_code == 2
    assert verdict["safe"] is False
    assert "failed to fetch exo state" in _reason_at(verdict, 0)
    assert captured.err == ""


def test_main_fail_closes_on_process_collection_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        guard,
        "fetch_exo_state",
        _empty_state,
    )

    def _raise_process_error() -> list[str]:
        raise RuntimeError("ps unavailable")

    monkeypatch.setattr(guard, "collect_processes", _raise_process_error)

    exit_code = guard.main([])

    captured = capsys.readouterr()
    verdict = _parse_json_object(captured.out)

    assert exit_code == 2
    assert verdict["safe"] is False
    assert "failed to collect processes" in _reason_at(verdict, 0)
    assert captured.err == ""
