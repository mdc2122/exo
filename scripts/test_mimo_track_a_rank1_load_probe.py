import json
from pathlib import Path

from scripts import mimo_track_a_rank1_load_probe as probe


def _write_minimal_mimo_config(model_path: Path) -> None:
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["MiMoV2ForCausalLM"],
                "model_type": "mimo_v2",
                "num_hidden_layers": 70,
            }
        )
    )


def test_parse_args_defaults_target_rank1_failing_shard() -> None:
    config = probe.parse_args([])

    assert config.model_path == probe.DEFAULT_MODEL_PATH
    assert config.start_layer == 36
    assert config.end_layer == 70
    assert config.device_rank == 1
    assert config.world_size == 2
    assert config.output_path is None
    assert config.dry_run is False
    assert config.no_eval is False


def test_dry_run_validates_metadata_prints_plan_and_does_not_load_weights(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    model_path = tmp_path / "XiaomiMiMo--MiMo-V2.5-Pro-6bit-MLX"
    _write_minimal_mimo_config(model_path)
    output_path = tmp_path / "probe.jsonl"

    def fail_if_probe_loads_weights(*_args, **_kwargs) -> None:
        raise AssertionError("dry-run must not load model weights")

    monkeypatch.setattr(probe, "run_probe", fail_if_probe_loads_weights)

    result = probe.main(
        [
            "--model-path",
            str(model_path),
            "--output-path",
            str(output_path),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "MiMo Track A rank-1 load probe plan" in captured.out
    assert "global layers [36, 70)" in captured.out
    assert "planned_layers: 36..69 (34 layers)" in captured.out
    assert "distributed/JACCL/libp2p/API/generation: disabled" in captured.out
    assert "dry-run: model weights were not loaded or evaluated" in captured.out
    assert not output_path.exists()


def test_validate_shard_range_rejects_invalid_rank(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    _write_minimal_mimo_config(model_path)
    metadata = probe.validate_model_metadata(model_path)
    config = probe.ProbeConfig(
        model_path=model_path,
        start_layer=36,
        end_layer=70,
        device_rank=2,
        world_size=2,
        output_path=None,
        dry_run=True,
        no_eval=True,
    )

    try:
        probe.validate_shard_range(config, metadata)
    except ValueError as exc:
        assert "device-rank must be less than world-size" in str(exc)
    else:  # pragma: no cover - explicit assertion branch
        raise AssertionError("expected invalid rank to be rejected")


def test_default_output_path_is_timestamped_rank1_probe_jsonl(monkeypatch) -> None:
    class FixedDateTime:
        @staticmethod
        def now(_tz):
            from datetime import datetime, timezone

            return datetime(2026, 5, 6, 12, 0, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(probe, "datetime", FixedDateTime)

    assert probe.default_output_path() == (
        probe.DEFAULT_INVESTIGATION_DIR
        / "mimo-track-a-rank1-load-probe-20260506T120000Z.jsonl"
    )
