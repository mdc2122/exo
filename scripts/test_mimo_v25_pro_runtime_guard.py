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
