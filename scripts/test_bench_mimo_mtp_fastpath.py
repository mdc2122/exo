from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from scripts import bench_mimo_mtp_fastpath as bench


def _json_lines(output: str) -> list[dict[str, object]]:
    return [cast(dict[str, object], json.loads(line)) for line in output.splitlines()]


def test_bench_dry_run_contract_only_accepts_missing_sidecar_without_model_load(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_sidecar = tmp_path / "model_mtp.safetensors"

    exit_code = bench.main(
        [
            "--sidecar-path",
            str(missing_sidecar),
            "--dry-run-contract-only",
        ]
    )

    assert exit_code == 0
    row = cast(dict[str, object], json.loads(capsys.readouterr().out))
    assert row["kind"] == "contract_probe"
    assert row["ready"] is False
    assert row["sidecar_status"] == "missing"
    assert row["sidecar_path"] == str(missing_sidecar)


def test_bench_non_dry_run_reports_clear_missing_model_path_without_loading_sidecar(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_model = tmp_path / "missing-model"
    missing_sidecar = tmp_path / "missing-sidecar.safetensors"

    exit_code = bench.main(
        [
            "--model-path",
            str(missing_model),
            "--sidecar-path",
            str(missing_sidecar),
        ]
    )

    assert exit_code == 2
    rows = _json_lines(capsys.readouterr().out)
    assert rows[0]["kind"] == "benchmark_stage"
    assert rows[0]["stage"] == "modes_validated"
    assert rows[1]["kind"] == "benchmark_stage"
    assert rows[1]["stage"] == "model_path_validated"
    assert rows[1]["status"] == "failed"
    assert rows[-1] == {
        "kind": "validation_error",
        "field": "model_path",
        "path": str(missing_model),
        "error": "MiMo MTP benchmark model path does not exist",
        "next_step": "provide an existing local MiMo model snapshot path or use --dry-run-contract-only",
    }


def test_bench_non_dry_run_reports_clear_invalid_sidecar_before_model_load(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_path = tmp_path / "model"
    missing_sidecar = tmp_path / "missing-sidecar.safetensors"
    model_path.mkdir()

    exit_code = bench.main(
        [
            "--model-path",
            str(model_path),
            "--sidecar-path",
            str(missing_sidecar),
        ]
    )

    assert exit_code == 2
    rows = _json_lines(capsys.readouterr().out)
    assert rows[0]["stage"] == "modes_validated"
    assert rows[1]["stage"] == "model_path_validated"
    assert rows[1]["status"] == "completed"
    assert rows[2]["stage"] == "sidecar_contract_validated"
    assert rows[2]["status"] == "failed"
    row = rows[-1]
    assert row["kind"] == "validation_error"
    assert row["field"] == "sidecar_path"
    assert row["path"] == str(missing_sidecar)
    assert row["error"] == "MiMo MTP sidecar is not ready"
    assert row["sidecar_status"] == "missing"
    assert (
        row["next_step"]
        == "run --dry-run-contract-only or provide a valid official-layout sidecar"
    )


def test_bench_preflight_only_validates_without_loading_model(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model"
    sidecar_path = tmp_path / "model_mtp.safetensors"
    model_path.mkdir()

    class FakeProbe:
        ready = True
        path = sidecar_path
        status = "ready"
        layer_count = 3

    def fake_probe(_path: object) -> FakeProbe:
        return FakeProbe()

    def fail_load_model(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("preflight-only should not load the model")

    monkeypatch.setattr(bench, "probe_mimo_mtp_sidecar", fake_probe)
    monkeypatch.setattr(bench, "load_model", fail_load_model)

    exit_code = bench.main(
        [
            "--model-path",
            str(model_path),
            "--sidecar-path",
            str(sidecar_path),
            "--modes",
            "ar,d1",
            "--preflight-only",
        ]
    )

    assert exit_code == 0
    rows = _json_lines(capsys.readouterr().out)
    assert [row["stage"] for row in rows if row["kind"] == "benchmark_stage"] == [
        "modes_validated",
        "model_path_validated",
        "sidecar_contract_validated",
    ]
    assert rows[-1] == {
        "kind": "benchmark_preflight",
        "ready": True,
        "model_path": str(model_path),
        "sidecar_path": str(sidecar_path),
        "modes": ["ar", "d1"],
        "next_step": "run --load-only to test model materialization before generation",
    }


def test_bench_load_only_materializes_dependencies_without_generation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model"
    sidecar_path = tmp_path / "model_mtp.safetensors"
    model_path.mkdir()

    class FakeProbe:
        ready = True
        path = sidecar_path
        status = "ready"
        layer_count = 3

    class FakeToken:
        def __init__(self, value: int) -> None:
            self._value = value

        def item(self) -> int:
            return self._value

    class FakePromptTokens:
        size = 2

        def reshape(self, *_shape: int) -> FakePromptTokens:
            return self

        def __getitem__(self, index: int) -> FakeToken:
            return FakeToken(index + 10)

    class FakeMx:
        int32 = object()

        @staticmethod
        def eval(_value: object) -> None:
            return None

    def fake_probe(_path: object) -> FakeProbe:
        return FakeProbe()

    def fake_load_model(
        *_args: object, **_kwargs: object
    ) -> tuple[object, dict[str, object]]:
        return object(), {}

    def fake_load_tokenizer_for_model_id(*_args: object) -> object:
        return object()

    def fake_load_mimo_mtp_sidecar_tensors(_path: object) -> object:
        return object()

    def fake_build_mimo_mtp_stack(*_args: object) -> object:
        return object()

    def fake_apply_chat_template(*_args: object) -> str:
        return "templated prompt"

    def fake_encode_prompt(*_args: object) -> FakePromptTokens:
        return FakePromptTokens()

    def fail_run_benchmark_modes(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("load-only should exit before generation")

    monkeypatch.setattr(bench, "probe_mimo_mtp_sidecar", fake_probe)
    monkeypatch.setattr(bench, "load_model", fake_load_model)
    monkeypatch.setattr(bench, "mx", FakeMx)
    monkeypatch.setattr(
        bench, "load_tokenizer_for_model_id", fake_load_tokenizer_for_model_id
    )
    monkeypatch.setattr(
        bench,
        "load_mimo_mtp_sidecar_tensors",
        fake_load_mimo_mtp_sidecar_tensors,
    )
    monkeypatch.setattr(bench, "build_mimo_mtp_stack", fake_build_mimo_mtp_stack)
    monkeypatch.setattr(bench, "apply_chat_template", fake_apply_chat_template)
    monkeypatch.setattr(bench, "encode_prompt", fake_encode_prompt)
    monkeypatch.setattr(bench, "run_benchmark_modes", fail_run_benchmark_modes)

    exit_code = bench.main(
        [
            "--model-path",
            str(model_path),
            "--sidecar-path",
            str(sidecar_path),
            "--modes",
            "ar",
            "--load-only",
        ]
    )

    assert exit_code == 0
    rows = _json_lines(capsys.readouterr().out)
    completed_stages = [
        row["stage"]
        for row in rows
        if row["kind"] == "benchmark_stage" and row["status"] == "completed"
    ]
    assert completed_stages == [
        "modes_validated",
        "model_path_validated",
        "sidecar_contract_validated",
        "base_model_materialization",
        "tokenizer_load",
        "sidecar_tensor_load",
        "sidecar_stack_build",
        "prompt_tokenization",
    ]
    assert rows[-1] == {
        "kind": "benchmark_load",
        "ready": True,
        "model_path": str(model_path),
        "sidecar_path": str(sidecar_path),
        "prompt_token_count": 2,
        "next_step": "run a minimal live AR row with --modes ar --max-tokens 1",
    }
