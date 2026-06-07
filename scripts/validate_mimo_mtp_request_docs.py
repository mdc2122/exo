from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

from exo.api.types.api import ChatCompletionRequest

MTP_DOC_MARKERS: Final[tuple[str, ...]] = ("Experimental", "Disabled by default")
DEFAULT_DOCS_PATH: Final[Path] = Path("docs/api.md")


def discover_mimo_mtp_request_fields() -> tuple[str, ...]:
    return tuple(
        field_name
        for field_name in ChatCompletionRequest.model_fields
        if field_name.startswith("mimo_mtp_")
    )


def _line_for_field(documentation_text: str, field_name: str) -> str | None:
    for line in documentation_text.splitlines():
        if f"`{field_name}`" in line:
            return line
    return None


def validate_mimo_mtp_request_docs(docs_path: Path = DEFAULT_DOCS_PATH) -> list[str]:
    documentation_text = docs_path.read_text()
    failures: list[str] = []

    for field_name in discover_mimo_mtp_request_fields():
        line = _line_for_field(documentation_text, field_name)
        if line is None:
            failures.append(f"{field_name}: missing documentation row")
            continue

        lower_line = line.lower()
        for marker in MTP_DOC_MARKERS:
            if marker.lower() not in lower_line:
                failures.append(f"{field_name}: missing required marker {marker!r}")

    return failures


def main() -> int:
    docs_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DOCS_PATH
    failures = validate_mimo_mtp_request_docs(docs_path)
    if failures:
        print("MiMo MTP request documentation validation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"MiMo MTP request documentation validation passed: {docs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
