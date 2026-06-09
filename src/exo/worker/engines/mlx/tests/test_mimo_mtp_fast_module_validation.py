from __future__ import annotations

from pathlib import Path

from exo.worker.engines.mlx.mimo_mtp_fast import module_validation
from exo.worker.engines.mlx.mimo_mtp_fast.module_validation import (
    MIMO_MTP_FASTPATH_INVENTORY_SCHEMA_VERSION,
    MIMO_MTP_FASTPATH_INVENTORY_SHA256,
    REQUIRED_MIMO_MTP_FASTPATH_MODULES,
    build_mimo_mtp_fastpath_inventory_evidence,
    validate_mimo_mtp_fastpath_modules,
)


def test_required_fastpath_modules_match_existing_vertical_slice_modules() -> None:
    assert REQUIRED_MIMO_MTP_FASTPATH_MODULES == (
        "exo.worker.engines.mlx.mimo_mtp_fast.benchmark",
        "exo.worker.engines.mlx.mimo_mtp_fast.mtp_generate",
        "exo.worker.engines.mlx.mimo_mtp_fast.one_cycle",
        "exo.worker.engines.mlx.mimo_mtp_fast.providers",
        "exo.worker.engines.mlx.mimo_mtp_fast.sidecar_contract",
        "exo.worker.engines.mlx.mimo_mtp_fast.sidecar_loader",
        "exo.worker.engines.mlx.mimo_mtp_fast.sidecar_module",
        "exo.worker.engines.mlx.mimo_mtp_fast.speculative_loop",
        "exo.worker.engines.mlx.mimo_mtp_fast.worker_fastpath",
    )


def test_fastpath_inventory_contract_hash_pins_unchanged_module_inventory() -> None:
    assert (
        MIMO_MTP_FASTPATH_INVENTORY_SCHEMA_VERSION == "mimo_mtp_fastpath_inventory.v1"
    )
    assert (
        MIMO_MTP_FASTPATH_INVENTORY_SHA256
        == "8f78fba22cd26d99362b9cd1976f1055856fe36c3521e40dfb3ece1e7ba6a93f"
    )


def test_build_fastpath_inventory_evidence_is_runnable_canonical_contract() -> None:
    evidence = build_mimo_mtp_fastpath_inventory_evidence(repo_root=Path.cwd())

    assert evidence["schema_version"] == MIMO_MTP_FASTPATH_INVENTORY_SCHEMA_VERSION
    assert evidence["evidence_kind"] == "mimo_mtp_fastpath_module_inventory"
    assert evidence["row_status"] == "passed"
    assert evidence["inventory_status"] == "unchanged"
    assert evidence["inventory_sha256"] == MIMO_MTP_FASTPATH_INVENTORY_SHA256
    assert evidence["expected_inventory_sha256"] == MIMO_MTP_FASTPATH_INVENTORY_SHA256
    assert evidence["module_count"] == len(REQUIRED_MIMO_MTP_FASTPATH_MODULES)
    assert evidence["existing_module_count"] == len(REQUIRED_MIMO_MTP_FASTPATH_MODULES)
    assert evidence["missing_module_count"] == 0
    assert evidence["performance_claim"] == "none"

    modules = evidence["modules"]
    assert isinstance(modules, list)
    assert modules[0] == {
        "module_name": "exo.worker.engines.mlx.mimo_mtp_fast.benchmark",
        "relative_path": "src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py",
        "exists": True,
    }
    assert modules[-1] == {
        "module_name": "exo.worker.engines.mlx.mimo_mtp_fast.worker_fastpath",
        "relative_path": "src/exo/worker/engines/mlx/mimo_mtp_fast/worker_fastpath.py",
        "exists": True,
    }


def test_validate_fastpath_modules_imports_every_required_module() -> None:
    report = validate_mimo_mtp_fastpath_modules()

    assert report.ready is True
    assert report.required_module_count == len(REQUIRED_MIMO_MTP_FASTPATH_MODULES)
    assert tuple(result.module_name for result in report.results) == (
        REQUIRED_MIMO_MTP_FASTPATH_MODULES
    )
    assert {result.status for result in report.results} == {"imported"}
    assert all(result.error is None for result in report.results)


def test_validate_fastpath_modules_confirms_required_modules_exist_at_expected_paths() -> (
    None
):
    report = validate_mimo_mtp_fastpath_modules()

    assert report.ready is True
    assert tuple(result.relative_path for result in report.results) == (
        "src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py",
        "src/exo/worker/engines/mlx/mimo_mtp_fast/mtp_generate.py",
        "src/exo/worker/engines/mlx/mimo_mtp_fast/one_cycle.py",
        "src/exo/worker/engines/mlx/mimo_mtp_fast/providers.py",
        "src/exo/worker/engines/mlx/mimo_mtp_fast/sidecar_contract.py",
        "src/exo/worker/engines/mlx/mimo_mtp_fast/sidecar_loader.py",
        "src/exo/worker/engines/mlx/mimo_mtp_fast/sidecar_module.py",
        "src/exo/worker/engines/mlx/mimo_mtp_fast/speculative_loop.py",
        "src/exo/worker/engines/mlx/mimo_mtp_fast/worker_fastpath.py",
    )
    assert all(result.exists for result in report.results)


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


def test_discover_fastpath_modules_reports_existing_repository_files(
    tmp_path: Path,
) -> None:
    package_dir = (
        tmp_path / "src" / "exo" / "worker" / "engines" / "mlx" / "mimo_mtp_fast"
    )
    package_dir.mkdir(parents=True)
    (package_dir / "benchmark.py").write_text("# present\n")

    report = module_validation.discover_mimo_mtp_fastpath_modules(
        repo_root=tmp_path,
        expected_modules=(
            "exo.worker.engines.mlx.mimo_mtp_fast.benchmark",
            "exo.worker.engines.mlx.mimo_mtp_fast.missing_module",
        ),
    )

    assert report.expected_module_count == 2
    assert report.existing_module_count == 1
    assert report.missing_module_count == 1
    assert report.ready is False
    assert (
        report.results[0].module_name
        == "exo.worker.engines.mlx.mimo_mtp_fast.benchmark"
    )
    assert report.results[0].exists is True
    assert (
        report.results[0].relative_path
        == "src/exo/worker/engines/mlx/mimo_mtp_fast/benchmark.py"
    )
    assert (
        report.results[1].module_name
        == "exo.worker.engines.mlx.mimo_mtp_fast.missing_module"
    )
    assert report.results[1].exists is False
    assert (
        report.results[1].relative_path
        == "src/exo/worker/engines/mlx/mimo_mtp_fast/missing_module.py"
    )
