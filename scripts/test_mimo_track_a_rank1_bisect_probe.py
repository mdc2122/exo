import json
import signal
from pathlib import Path
from typing import Any

from scripts import mimo_track_a_rank1_bisect_probe as probe


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_parse_args_defaults_are_safe_track_a_rank1() -> None:
    config = probe.parse_args([])

    assert config.model_path == probe.DEFAULT_MODEL_PATH
    assert config.start_layer == 36
    assert config.min_end_layer == 60
    assert config.max_end_layer == 70
    assert config.device_rank == 1
    assert config.world_size == 2
    assert config.plan_style == "binary"
    assert config.output_dir == probe.DEFAULT_INVESTIGATION_DIR
    assert config.manifest_path is None
    assert config.execute is False
    assert probe.parse_args(["--dry-run"]).execute is False
    assert config.probe_script == probe.PROBE_SCRIPT


def test_planned_ranges_default_binary_and_linear() -> None:
    binary_config = probe.parse_args([])
    assert probe.planned_end_layers(binary_config) == [60, 62, 65, 67, 70]
    assert [
        (attempt.start_layer, attempt.end_layer)
        for attempt in probe.make_attempts(binary_config, Path("manifest.jsonl"))
    ] == [(36, 60), (36, 62), (36, 65), (36, 67), (36, 70)]

    linear_config = probe.parse_args(["--plan-style", "linear"])
    assert probe.planned_end_layers(linear_config) == list(range(60, 71))


def test_fake_subprocess_result_parsing_writes_logs_and_last_record(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    config = probe.parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--manifest-path",
            str(manifest_path),
            "--execute",
        ]
    )
    attempt = probe.make_attempts(config, manifest_path)[0]
    attempt.jsonl_path.write_text(
        json.dumps({"stage": "before-load-model", "global_layer_index": None})
        + "\n"
        + "not-json\n"
        + json.dumps({"stage": "after-layer-eval", "global_layer_index": 59})
        + "\n",
        encoding="utf-8",
    )

    captured_command: list[str] = []

    def fake_runner(command: list[str], **_kwargs: object) -> probe.CompletedProcessLike:
        captured_command.extend(command)
        return probe.CompletedProcessLike(
            returncode=-signal.SIGKILL,
            stdout="stdout text",
            stderr="stderr text",
        )

    record = probe.run_attempt(config, attempt, runner=fake_runner)

    assert captured_command == record["command"]
    assert record["exit_code"] == -signal.SIGKILL
    assert record["signal"] == "SIGKILL"
    assert probe.infer_signal(137) == "SIGKILL"
    assert probe.infer_signal(0) is None
    assert record["stdout_log_path"] == str(attempt.stdout_path)
    assert record["stderr_log_path"] == str(attempt.stderr_path)
    assert record["jsonl_path"] == str(attempt.jsonl_path)
    assert record["last_parsed_diagnostic_record"] == {
        "stage": "after-layer-eval",
        "global_layer_index": 59,
    }
    assert attempt.stdout_path.read_text(encoding="utf-8") == "stdout text"
    assert attempt.stderr_path.read_text(encoding="utf-8") == "stderr text"


def test_manifest_writing_includes_header_and_dry_run_attempts(tmp_path: Path) -> None:
    manifest_path = tmp_path / "bisect-manifest.jsonl"

    result = probe.main(
        [
            "--output-dir",
            str(tmp_path),
            "--manifest-path",
            str(manifest_path),
            "--min-end-layer",
            "68",
            "--max-end-layer",
            "70",
        ]
    )

    assert result == 0
    records = _read_jsonl(manifest_path)
    assert records[0]["type"] == "manifest-header"
    assert records[0]["execute"] is False
    assert records[0]["safety_boundary"].startswith("No exo restart")
    assert [record["end_layer"] for record in records[1:]] == [68, 69, 70]
    assert all(record["mode"] == "dry-run" for record in records[1:])
    assert all(record["exit_code"] is None for record in records[1:])
    assert all("mimo_track_a_rank1_load_probe.py" in record["command"][1] for record in records[1:])


def test_no_heavy_load_default_does_not_run_attempt(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "safe-default.jsonl"

    def fail_if_execute_path_runs(*_args: object, **_kwargs: object) -> dict[str, Any]:
        raise AssertionError("default dry-run must not launch the standalone load probe")

    monkeypatch.setattr(probe, "run_attempt", fail_if_execute_path_runs)

    result = probe.main(
        [
            "--output-dir",
            str(tmp_path),
            "--manifest-path",
            str(manifest_path),
            "--min-end-layer",
            "70",
            "--max-end-layer",
            "70",
        ]
    )

    assert result == 0
    records = _read_jsonl(manifest_path)
    assert records[0]["execute"] is False
    assert records[1]["mode"] == "dry-run"
    assert not (tmp_path / "safe-default-layers-36-70.stdout.log").exists()
    assert not (tmp_path / "safe-default-layers-36-70.stderr.log").exists()
