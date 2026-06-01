from __future__ import annotations

import json
from pathlib import Path

from exo.worker.engines.mlx.mimo_mtp_fast.benchmark import (
    MimoMtpBenchmarkMode,
    build_contract_probe_row,
    build_metric_row,
    parse_benchmark_modes,
    render_json_line,
)


def test_contract_probe_row_reports_missing_sidecar_without_loading_model(
    tmp_path: Path,
) -> None:
    missing_sidecar = tmp_path / "model_mtp.safetensors"

    row = build_contract_probe_row(
        sidecar_path=missing_sidecar,
        model_path=None,
    )

    assert row["kind"] == "contract_probe"
    assert row["ready"] is False
    assert row["sidecar_status"] == "missing"
    assert row["sidecar_path"] == str(missing_sidecar)
    assert row["model_path"] is None
    assert row["next_step"] == "provide a valid official-layout MiMo model_mtp.safetensors sidecar"


def test_parse_benchmark_modes_preserves_requested_order() -> None:
    modes = parse_benchmark_modes("ar,d1,d2,d3,auto")

    assert modes == (
        MimoMtpBenchmarkMode.AR,
        MimoMtpBenchmarkMode.D1,
        MimoMtpBenchmarkMode.D2,
        MimoMtpBenchmarkMode.D3,
        MimoMtpBenchmarkMode.AUTO,
    )


def test_parse_benchmark_modes_rejects_unknown_mode() -> None:
    try:
        parse_benchmark_modes("ar,d4")
    except ValueError as exc:
        assert "Unsupported MiMo MTP benchmark mode" in str(exc)
        assert "d4" in str(exc)
    else:
        raise AssertionError("parse_benchmark_modes should reject d4")


def test_metric_row_computes_decode_tokens_per_second_and_conclusion() -> None:
    row = build_metric_row(
        mode=MimoMtpBenchmarkMode.D3,
        generated_tokens=96,
        decode_seconds=3.0,
        attempted_depth_counts={3: 40},
        accepted_depth_counts={0: 4, 3: 36},
        ar_baseline_tok_s=22.0,
    )

    assert row["kind"] == "benchmark_metric"
    assert row["mode"] == "d3"
    assert row["generated_tokens"] == 96
    assert row["decode_seconds"] == 3.0
    assert row["decode_tok_s"] == 32.0
    assert row["attempted_depth_counts"] == {"3": 40}
    assert row["accepted_depth_counts"] == {"0": 4, "3": 36}
    assert row["next_step"] == "MTP mode d3 beats AR baseline; proceed to guarded exo integration"


def test_render_json_line_is_single_line_stable_json() -> None:
    line = render_json_line({"b": 2, "a": 1})

    assert line == '{"a":1,"b":2}'
    assert "\n" not in line
    assert json.loads(line) == {"a": 1, "b": 2}


def test_run_benchmark_modes_calls_runner_and_threads_ar_baseline() -> None:
    from exo.worker.engines.mlx.mimo_mtp_fast.benchmark import (
        BenchmarkRunResult,
        run_benchmark_modes,
    )

    calls: list[MimoMtpBenchmarkMode] = []

    def fake_runner(mode: MimoMtpBenchmarkMode) -> BenchmarkRunResult:
        calls.append(mode)
        if mode == MimoMtpBenchmarkMode.AR:
            return BenchmarkRunResult(
                mode=mode,
                generated_tokens=44,
                decode_seconds=2.0,
                attempted_depth_counts={},
                accepted_depth_counts={},
            )
        return BenchmarkRunResult(
            mode=mode,
            generated_tokens=96,
            decode_seconds=3.0,
            attempted_depth_counts={3: 40},
            accepted_depth_counts={3: 32, 0: 8},
        )

    rows = run_benchmark_modes(
        modes=(MimoMtpBenchmarkMode.AR, MimoMtpBenchmarkMode.D3),
        runner=fake_runner,
    )

    assert calls == [MimoMtpBenchmarkMode.AR, MimoMtpBenchmarkMode.D3]
    assert rows[0]["mode"] == "ar"
    assert rows[0]["decode_tok_s"] == 22.0
    assert rows[0]["ar_baseline_tok_s"] is None
    assert rows[1]["mode"] == "d3"
    assert rows[1]["decode_tok_s"] == 32.0
    assert rows[1]["ar_baseline_tok_s"] == 22.0
    assert rows[1]["next_step"] == "MTP mode d3 beats AR baseline; proceed to guarded exo integration"


def test_run_benchmark_modes_uses_none_baseline_when_ar_not_requested_first() -> None:
    from exo.worker.engines.mlx.mimo_mtp_fast.benchmark import (
        BenchmarkRunResult,
        run_benchmark_modes,
    )

    def fake_runner(mode: MimoMtpBenchmarkMode) -> BenchmarkRunResult:
        return BenchmarkRunResult(
            mode=mode,
            generated_tokens=40,
            decode_seconds=2.0,
            attempted_depth_counts={1: 20},
            accepted_depth_counts={1: 20},
        )

    rows = run_benchmark_modes(modes=(MimoMtpBenchmarkMode.D1,), runner=fake_runner)

    assert rows == [
        build_metric_row(
            mode=MimoMtpBenchmarkMode.D1,
            generated_tokens=40,
            decode_seconds=2.0,
            attempted_depth_counts={1: 20},
            accepted_depth_counts={1: 20},
            ar_baseline_tok_s=None,
        )
    ]


def test_run_ar_benchmark_invokes_generation_dependencies_and_counts_tokens() -> None:
    from exo.worker.engines.mlx.mimo_mtp_fast.benchmark import (
        ArBenchmarkRequest,
        run_ar_benchmark,
    )

    calls: dict[str, object] = {}
    prompt_builder_task: object | None = None

    def fake_prompt_builder(tokenizer: object, task_params: object) -> str:
        nonlocal prompt_builder_task
        calls["prompt_builder_tokenizer"] = tokenizer
        prompt_builder_task = task_params
        return "templated prompt"

    def fake_generate(
        *,
        model: object,
        tokenizer: object,
        task: object,
        prompt: str,
        kv_prefix_cache: object | None,
        group: object | None,
    ):
        calls["generate"] = {
            "model": model,
            "tokenizer": tokenizer,
            "task": task,
            "prompt": prompt,
            "kv_prefix_cache": kv_prefix_cache,
            "group": group,
        }
        yield object()
        yield object()
        yield object()

    request = ArBenchmarkRequest(
        model=object(),
        tokenizer=object(),
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        prompt="hello",
        max_tokens=3,
    )

    result = run_ar_benchmark(
        request,
        prompt_builder=fake_prompt_builder,
        generate=fake_generate,
        timer=lambda: 10.0,
        elapsed_seconds_override=1.5,
    )

    assert result.mode == MimoMtpBenchmarkMode.AR
    assert result.generated_tokens == 3
    assert result.decode_seconds == 1.5
    assert result.attempted_depth_counts == {}
    assert result.accepted_depth_counts == {}
    assert calls["generate"] == {
        "model": request.model,
        "tokenizer": request.tokenizer,
        "task": prompt_builder_task,
        "prompt": "templated prompt",
        "kv_prefix_cache": None,
        "group": None,
    }


def test_run_ar_benchmark_uses_timer_when_no_elapsed_override() -> None:
    from exo.worker.engines.mlx.mimo_mtp_fast.benchmark import (
        ArBenchmarkRequest,
        run_ar_benchmark,
    )

    timer_values = iter((2.0, 5.5))

    def fake_generate(**_kwargs: object):
        yield object()
        yield object()

    request = ArBenchmarkRequest(
        model=object(),
        tokenizer=object(),
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        prompt="hello",
        max_tokens=2,
    )

    result = run_ar_benchmark(
        request,
        prompt_builder=lambda _tokenizer, _task_params: "prompt",
        generate=fake_generate,
        timer=lambda: next(timer_values),
    )

    assert result.generated_tokens == 2
    assert result.decode_seconds == 3.5


def test_run_mtp_benchmark_streams_events_and_counts_depths() -> None:
    from exo.worker.engines.mlx.mimo_mtp_fast.benchmark import (
        MtpBenchmarkRequest,
        run_mtp_benchmark,
    )
    from exo.worker.engines.mlx.mimo_mtp_fast.one_cycle import MimoMtpOneCycleResult

    observed_histories: list[tuple[int, ...]] = []

    def one_cycle(history: tuple[int, ...], requested_depth: int) -> MimoMtpOneCycleResult:
        observed_histories.append(history)
        if len(observed_histories) == 1:
            return MimoMtpOneCycleResult(
                proposed_token_ids=(10, 11, 12),
                accepted_token_ids=(10, 11),
                fallback_token_id=99,
                attempted_depth=requested_depth,
                accepted_depth=2,
                elapsed_seconds=0.001,
            )
        return MimoMtpOneCycleResult(
            proposed_token_ids=(12,),
            accepted_token_ids=(),
            fallback_token_id=77,
            attempted_depth=1,
            accepted_depth=0,
            elapsed_seconds=0.001,
        )

    request = MtpBenchmarkRequest(
        mode=MimoMtpBenchmarkMode.D3,
        token_history=(1, 2),
        max_tokens=3,
        requested_depth=3,
    )

    result = run_mtp_benchmark(
        request,
        one_cycle=one_cycle,
        timer=lambda: 100.0,
        elapsed_seconds_override=2.0,
    )

    assert result.mode == MimoMtpBenchmarkMode.D3
    assert result.generated_tokens == 3
    assert result.decode_seconds == 2.0
    assert result.attempted_depth_counts == {3: 2, 1: 1}
    assert result.accepted_depth_counts == {2: 2, 0: 1}
    assert observed_histories == [(1, 2), (1, 2, 10, 11)]


def test_requested_depth_for_benchmark_mode() -> None:
    from exo.worker.engines.mlx.mimo_mtp_fast.benchmark import requested_depth_for_mode

    assert requested_depth_for_mode(MimoMtpBenchmarkMode.D1) == 1
    assert requested_depth_for_mode(MimoMtpBenchmarkMode.D2) == 2
    assert requested_depth_for_mode(MimoMtpBenchmarkMode.D3) == 3
    assert requested_depth_for_mode(MimoMtpBenchmarkMode.AUTO) == 3
    assert requested_depth_for_mode(MimoMtpBenchmarkMode.AR) == 0


def test_build_benchmark_runner_dispatches_ar_and_mtp_modes() -> None:
    from exo.worker.engines.mlx.mimo_mtp_fast.benchmark import (
        ArBenchmarkRequest,
        BenchmarkRunResult,
        MtpBenchmarkRequest,
        build_benchmark_runner,
    )
    from exo.worker.engines.mlx.mimo_mtp_fast.one_cycle import MimoMtpOneCycleResult

    ar_requests: list[ArBenchmarkRequest] = []
    mtp_requests: list[MtpBenchmarkRequest] = []
    one_cycle_calls: list[tuple[tuple[int, ...], int]] = []

    def fake_ar(request: ArBenchmarkRequest) -> BenchmarkRunResult:
        ar_requests.append(request)
        return BenchmarkRunResult(
            mode=MimoMtpBenchmarkMode.AR,
            generated_tokens=request.max_tokens,
            decode_seconds=1.0,
            attempted_depth_counts={},
            accepted_depth_counts={},
        )

    def fake_one_cycle(history: tuple[int, ...], depth: int) -> MimoMtpOneCycleResult:
        one_cycle_calls.append((history, depth))
        return MimoMtpOneCycleResult(
            proposed_token_ids=(10, 11, 12),
            accepted_token_ids=(10,),
            fallback_token_id=99,
            attempted_depth=depth,
            accepted_depth=1,
            elapsed_seconds=0.001,
        )

    def fake_mtp(request: MtpBenchmarkRequest, *, one_cycle: object) -> BenchmarkRunResult:
        mtp_requests.append(request)
        assert one_cycle is fake_one_cycle
        return BenchmarkRunResult(
            mode=request.mode,
            generated_tokens=request.max_tokens,
            decode_seconds=2.0,
            attempted_depth_counts={request.requested_depth: 1},
            accepted_depth_counts={1: 1},
        )

    runner = build_benchmark_runner(
        model="model",
        tokenizer="tokenizer",
        model_id="kernelpool/MiMo-V2.5-Pro-6bit",
        prompt="hello",
        prompt_token_history=(1, 2, 3),
        max_tokens=4,
        one_cycle=fake_one_cycle,
        ar_benchmark=fake_ar,
        mtp_benchmark=fake_mtp,
    )

    ar_result = runner(MimoMtpBenchmarkMode.AR)
    d2_result = runner(MimoMtpBenchmarkMode.D2)

    assert ar_result.mode == MimoMtpBenchmarkMode.AR
    assert d2_result.mode == MimoMtpBenchmarkMode.D2
    assert ar_requests == [
        ArBenchmarkRequest(
            model="model",
            tokenizer="tokenizer",
            model_id="kernelpool/MiMo-V2.5-Pro-6bit",
            prompt="hello",
            max_tokens=4,
        )
    ]
    assert mtp_requests == [
        MtpBenchmarkRequest(
            mode=MimoMtpBenchmarkMode.D2,
            token_history=(1, 2, 3),
            max_tokens=4,
            requested_depth=2,
        )
    ]
    assert one_cycle_calls == []
