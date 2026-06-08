from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final, cast

from exo.api.types.api import ChatCompletionRequest

MTP_DOC_DEFAULT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"Disabled by default \(`(?P<default>[^`]+)`\)"
)
MTP_DOC_MARKERS: Final[tuple[str, ...]] = ("Experimental", "Disabled by default")
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_DOCS_PATH: Final[Path] = REPO_ROOT / "docs/api.md"


def discover_mimo_mtp_request_fields() -> tuple[str, ...]:
    return tuple(
        field_name
        for field_name in ChatCompletionRequest.model_fields
        if field_name.startswith("mimo_mtp_")
    )


def discover_mimo_mtp_request_field_defaults() -> dict[str, str]:
    schema = cast(Mapping[object, object], ChatCompletionRequest.model_json_schema())
    properties_value = schema.get("properties")
    if not isinstance(properties_value, Mapping):
        raise TypeError("ChatCompletionRequest schema properties must be a mapping")
    properties = cast(Mapping[object, object], properties_value)

    defaults: dict[str, str] = {}
    for field_name in discover_mimo_mtp_request_fields():
        field_schema_value = properties.get(field_name)
        if not isinstance(field_schema_value, Mapping):
            raise TypeError(
                f"ChatCompletionRequest schema field {field_name!r} must be a mapping"
            )
        field_schema = cast(Mapping[object, object], field_schema_value)
        defaults[field_name] = json.dumps(field_schema.get("default"))
    return defaults


def _line_for_field(documentation_text: str, field_name: str) -> str | None:
    for line in documentation_text.splitlines():
        if f"`{field_name}`" in line:
            return line
    return None


def validate_mimo_mtp_request_docs(docs_path: Path = DEFAULT_DOCS_PATH) -> list[str]:
    documentation_text = docs_path.read_text()
    failures: list[str] = []
    schema_defaults = discover_mimo_mtp_request_field_defaults()

    for field_name in discover_mimo_mtp_request_fields():
        line = _line_for_field(documentation_text, field_name)
        if line is None:
            failures.append(f"{field_name}: missing documentation row")
            continue

        lower_line = line.lower()
        for marker in MTP_DOC_MARKERS:
            if marker.lower() not in lower_line:
                failures.append(f"{field_name}: missing required marker {marker!r}")

        default_match = MTP_DOC_DEFAULT_PATTERN.search(line)
        if default_match is None:
            failures.append(
                f"{field_name}: missing documented disabled-by-default schema default "
                f"`{schema_defaults[field_name]}`"
            )
            continue

        documented_default = default_match.group("default")
        schema_default = schema_defaults[field_name]
        if documented_default != schema_default:
            failures.append(
                f"{field_name}: documented default `{documented_default}` does not "
                f"match request schema default `{schema_default}`"
            )

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
