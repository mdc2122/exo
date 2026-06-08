from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

REQUIRED_REPO_ROOT_MARKERS: Final[tuple[str, ...]] = (
    "pyproject.toml",
    "Cargo.toml",
    "src/exo",
)


class RepoRootValidationError(ValueError):
    """Raised when the expected exo repository root cannot be confirmed."""


@dataclass(frozen=True, slots=True)
class RepoRootValidation:
    root: Path
    expected_root: Path | None
    start_path: Path
    markers: tuple[str, ...]


def _resolve_start_path(start_path: str | Path | None) -> Path:
    if start_path is None:
        return Path.cwd().resolve()
    return Path(start_path).resolve()


def _candidate_directories(start_path: Path) -> tuple[Path, ...]:
    first_candidate = start_path if start_path.is_dir() else start_path.parent
    return (first_candidate, *first_candidate.parents)


def _has_required_markers(candidate: Path) -> bool:
    return all((candidate / marker).exists() for marker in REQUIRED_REPO_ROOT_MARKERS)


def _detect_repo_root(start_path: Path) -> Path | None:
    for candidate in _candidate_directories(start_path):
        if _has_required_markers(candidate):
            return candidate
    return None


def validate_repo_root(
    *,
    start_path: str | Path | None = None,
    expected_root: str | Path | None = None,
) -> RepoRootValidation:
    """Detect and confirm the exo repository root from a root or nested path.

    The optimized MiMo MTP rollout runs from generated worker directories under
    the repository. This helper walks upward from ``start_path`` until it finds
    the expected exo root markers, then optionally confirms the detected root
    against an externally supplied immutable expected root.
    """

    resolved_start_path = _resolve_start_path(start_path)
    detected_root = _detect_repo_root(resolved_start_path)
    if detected_root is None:
        raise RepoRootValidationError(
            "could not find expected exo repository root from "
            f"{resolved_start_path}; required markers: "
            f"{', '.join(REQUIRED_REPO_ROOT_MARKERS)}"
        )

    resolved_expected_root = (
        None if expected_root is None else Path(expected_root).resolve()
    )
    if resolved_expected_root is not None and detected_root != resolved_expected_root:
        raise RepoRootValidationError(
            "detected repository root does not match expected root: "
            f"detected={detected_root}, expected={resolved_expected_root}"
        )

    return RepoRootValidation(
        root=detected_root,
        expected_root=resolved_expected_root,
        start_path=resolved_start_path,
        markers=REQUIRED_REPO_ROOT_MARKERS,
    )


def main(argv: list[str] | None = None) -> int:
    """Report the detected exo repository root for the current working directory."""

    del argv
    try:
        validation = validate_repo_root()
    except RepoRootValidationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(validation.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
