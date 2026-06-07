from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Final, Literal, final

MimoMtpFastpathModuleStatus = Literal["imported", "missing", "import_error"]

REQUIRED_MIMO_MTP_FASTPATH_MODULES: Final[tuple[str, ...]] = (
    "exo.worker.engines.mlx.mimo_mtp_fast.benchmark",
    "exo.worker.engines.mlx.mimo_mtp_fast.one_cycle",
    "exo.worker.engines.mlx.mimo_mtp_fast.providers",
    "exo.worker.engines.mlx.mimo_mtp_fast.sidecar_contract",
    "exo.worker.engines.mlx.mimo_mtp_fast.sidecar_loader",
    "exo.worker.engines.mlx.mimo_mtp_fast.sidecar_module",
    "exo.worker.engines.mlx.mimo_mtp_fast.speculative_loop",
)


@final
@dataclass(frozen=True, slots=True)
class MimoMtpFastpathModuleValidationResult:
    module_name: str
    status: MimoMtpFastpathModuleStatus
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


def _module_missing_error(module_name: str, exc: ModuleNotFoundError) -> bool:
    return exc.name == module_name


def _import_required_module(module_name: str) -> ModuleType:
    return importlib.import_module(module_name)


def validate_mimo_mtp_fastpath_modules(
    *,
    required_modules: tuple[str, ...] = REQUIRED_MIMO_MTP_FASTPATH_MODULES,
) -> MimoMtpFastpathModuleValidationReport:
    results: list[MimoMtpFastpathModuleValidationResult] = []
    for module_name in required_modules:
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
                    error=f"Failed to import required MiMo MTP fastpath module {module_name}: {exc}",
                )
            )
        except ImportError as exc:
            results.append(
                MimoMtpFastpathModuleValidationResult(
                    module_name=module_name,
                    status="import_error",
                    error=f"Failed to import required MiMo MTP fastpath module {module_name}: {exc}",
                )
            )
        else:
            results.append(
                MimoMtpFastpathModuleValidationResult(
                    module_name=module_name,
                    status="imported",
                )
            )
    return MimoMtpFastpathModuleValidationReport(results=tuple(results))
