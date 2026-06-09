from __future__ import annotations

import hashlib
import importlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Final, Literal, final

MimoMtpFastpathModuleStatus = Literal["imported", "missing", "import_error"]

MIMO_MTP_FASTPATH_INVENTORY_SCHEMA_VERSION: Final[str] = (
    "mimo_mtp_fastpath_inventory.v1"
)
MIMO_MTP_FASTPATH_INVENTORY_SHA256: Final[str] = (
    "8f78fba22cd26d99362b9cd1976f1055856fe36c3521e40dfb3ece1e7ba6a93f"
)

REQUIRED_MIMO_MTP_FASTPATH_MODULES: Final[tuple[str, ...]] = (
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


@final
@dataclass(frozen=True, slots=True)
class MimoMtpFastpathModuleValidationResult:
    module_name: str
    status: MimoMtpFastpathModuleStatus
    relative_path: str
    exists: bool
    error: str | None = None


@final
@dataclass(frozen=True, slots=True)
class MimoMtpFastpathModuleValidationReport:
    results: tuple[MimoMtpFastpathModuleValidationResult, ...]

    @property
    def ready(self) -> bool:
        return all(result.status == "imported" for result in self.results)

    @property
    def required_module_count(self) -> int:
        return len(self.results)


@final
@dataclass(frozen=True, slots=True)
class MimoMtpFastpathModuleDiscoveryResult:
    module_name: str
    relative_path: str
    exists: bool
    blob_sha: str | None = None


@final
@dataclass(frozen=True, slots=True)
class MimoMtpFastpathModuleDiscoveryReport:
    results: tuple[MimoMtpFastpathModuleDiscoveryResult, ...]

    @property
    def ready(self) -> bool:
        return all(result.exists for result in self.results)

    @property
    def expected_module_count(self) -> int:
        return len(self.results)

    @property
    def existing_module_count(self) -> int:
        return sum(1 for result in self.results if result.exists)

    @property
    def missing_module_count(self) -> int:
        return self.expected_module_count - self.existing_module_count


def _module_missing_error(module_name: str, exc: ModuleNotFoundError) -> bool:
    return exc.name == module_name


def _module_relative_path(module_name: str) -> Path:
    return Path("src") / Path(*module_name.split(".")).with_suffix(".py")


def _import_required_module(module_name: str) -> ModuleType:
    return importlib.import_module(module_name)


def _git_index_blob_sha(repo_root: Path, relative_path: str) -> str | None:
    """Return the git index blob SHA for a tracked file, or None if unavailable."""
    try:
        completed = subprocess.run(
            ["git", "ls-files", "-s", "--", relative_path],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    line = completed.stdout.strip()
    if not line:
        return None
    parts = line.split()
    if len(parts) < 2:
        return None
    return parts[1]


def _canonical_inventory_text(
    results: tuple[MimoMtpFastpathModuleDiscoveryResult, ...],
) -> str:
    return "\n".join(
        f"{result.module_name}\t{result.relative_path}" for result in results
    )


def _inventory_sha256(
    results: tuple[MimoMtpFastpathModuleDiscoveryResult, ...],
) -> str:
    return hashlib.sha256(_canonical_inventory_text(results).encode()).hexdigest()


def discover_mimo_mtp_fastpath_modules(
    *,
    repo_root: Path,
    expected_modules: tuple[str, ...] = REQUIRED_MIMO_MTP_FASTPATH_MODULES,
    include_blob_shas: bool = False,
) -> MimoMtpFastpathModuleDiscoveryReport:
    results: list[MimoMtpFastpathModuleDiscoveryResult] = []
    for module_name in expected_modules:
        relative_path = _module_relative_path(module_name)
        blob_sha: str | None = None
        if include_blob_shas:
            blob_sha = _git_index_blob_sha(repo_root, relative_path.as_posix())
        results.append(
            MimoMtpFastpathModuleDiscoveryResult(
                module_name=module_name,
                relative_path=relative_path.as_posix(),
                exists=(repo_root / relative_path).is_file(),
                blob_sha=blob_sha,
            )
        )
    return MimoMtpFastpathModuleDiscoveryReport(results=tuple(results))


def build_mimo_mtp_fastpath_inventory_evidence(*, repo_root: Path) -> dict[str, object]:
    report = discover_mimo_mtp_fastpath_modules(repo_root=repo_root)
    inventory_sha256 = _inventory_sha256(report.results)
    inventory_unchanged = inventory_sha256 == MIMO_MTP_FASTPATH_INVENTORY_SHA256
    return {
        "schema_version": MIMO_MTP_FASTPATH_INVENTORY_SCHEMA_VERSION,
        "evidence_kind": "mimo_mtp_fastpath_module_inventory",
        "row_status": "passed" if report.ready and inventory_unchanged else "failed",
        "inventory_status": "unchanged" if inventory_unchanged else "changed",
        "inventory_sha256": inventory_sha256,
        "expected_inventory_sha256": MIMO_MTP_FASTPATH_INVENTORY_SHA256,
        "module_count": report.expected_module_count,
        "existing_module_count": report.existing_module_count,
        "missing_module_count": report.missing_module_count,
        "performance_claim": "none",
        "modules": [
            {
                "module_name": result.module_name,
                "relative_path": result.relative_path,
                "exists": result.exists,
                **({"blob_sha": result.blob_sha} if result.blob_sha is not None else {}),
            }
            for result in report.results
        ],
    }


def validate_mimo_mtp_fastpath_modules(
    *,
    required_modules: tuple[str, ...] = REQUIRED_MIMO_MTP_FASTPATH_MODULES,
) -> MimoMtpFastpathModuleValidationReport:
    repo_root = Path.cwd()
    results: list[MimoMtpFastpathModuleValidationResult] = []
    for module_name in required_modules:
        relative_path = _module_relative_path(module_name)
        relative_path_text = relative_path.as_posix()
        exists = (repo_root / relative_path).is_file()
        try:
            _import_required_module(module_name)
        except ModuleNotFoundError as exc:
            status: MimoMtpFastpathModuleStatus = (
                "missing" if _module_missing_error(module_name, exc) else "import_error"
            )
            results.append(
                MimoMtpFastpathModuleValidationResult(
                    module_name=module_name,
                    status=status,
                    relative_path=relative_path_text,
                    exists=exists,
                    error=f"Failed to import required MiMo MTP fastpath module {module_name}: {exc}",
                )
            )
        except ImportError as exc:
            results.append(
                MimoMtpFastpathModuleValidationResult(
                    module_name=module_name,
                    status="import_error",
                    relative_path=relative_path_text,
                    exists=exists,
                    error=f"Failed to import required MiMo MTP fastpath module {module_name}: {exc}",
                )
            )
        else:
            results.append(
                MimoMtpFastpathModuleValidationResult(
                    module_name=module_name,
                    status="imported",
                    relative_path=relative_path_text,
                    exists=exists,
                )
            )
    return MimoMtpFastpathModuleValidationReport(results=tuple(results))
