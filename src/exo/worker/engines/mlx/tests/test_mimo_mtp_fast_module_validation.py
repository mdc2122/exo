from __future__ import annotations

from exo.worker.engines.mlx.mimo_mtp_fast.module_validation import (
    REQUIRED_MIMO_MTP_FASTPATH_MODULES,
    validate_mimo_mtp_fastpath_modules,
)


def test_required_fastpath_modules_match_existing_vertical_slice_modules() -> None:
    assert REQUIRED_MIMO_MTP_FASTPATH_MODULES == (
        "exo.worker.engines.mlx.mimo_mtp_fast.benchmark",
        "exo.worker.engines.mlx.mimo_mtp_fast.one_cycle",
        "exo.worker.engines.mlx.mimo_mtp_fast.providers",
        "exo.worker.engines.mlx.mimo_mtp_fast.sidecar_contract",
        "exo.worker.engines.mlx.mimo_mtp_fast.sidecar_loader",
        "exo.worker.engines.mlx.mimo_mtp_fast.sidecar_module",
        "exo.worker.engines.mlx.mimo_mtp_fast.speculative_loop",
    )


def test_validate_fastpath_modules_imports_every_required_module() -> None:
    report = validate_mimo_mtp_fastpath_modules()

    assert report.ready is True
    assert report.required_module_count == len(REQUIRED_MIMO_MTP_FASTPATH_MODULES)
    assert tuple(result.module_name for result in report.results) == (
        REQUIRED_MIMO_MTP_FASTPATH_MODULES
    )
    assert {result.status for result in report.results} == {"imported"}
    assert all(result.error is None for result in report.results)


def test_validate_fastpath_modules_reports_missing_module_without_throwing() -> None:
    missing_module = "exo.worker.engines.mlx.mimo_mtp_fast.missing_required_module"

    report = validate_mimo_mtp_fastpath_modules(required_modules=(missing_module,))

    assert report.ready is False
    assert report.required_module_count == 1
    assert len(report.results) == 1
    result = report.results[0]
    assert result.module_name == missing_module
    assert result.status == "missing"
    assert result.error is not None
    assert missing_module in result.error
