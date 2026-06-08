from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.validate_mimo_mtp_request_docs import (
    MTP_DOC_MARKERS,
    discover_mimo_mtp_request_fields,
    validate_mimo_mtp_request_docs,
)

DOC_WITH_ALL_FIELDS = """
# API

## Experimental MiMo MTP request fields

* `mimo_mtp_fastpath`: Experimental. Disabled by default (`false`).
* `mimo_mtp_depth`: Experimental. Disabled by default (`null`).
* `mimo_mtp_sidecar_path`: Experimental. Disabled by default (`null`).
* `mimo_mtp_fail_closed`: Experimental. Disabled by default (`true`).
"""


@pytest.mark.parametrize(
    ("field_name", "wrong_documented_default"),
    [
        ("mimo_mtp_fastpath", "true"),
        ("mimo_mtp_depth", "1"),
        ("mimo_mtp_sidecar_path", '"/tmp/model_mtp.safetensors"'),
        ("mimo_mtp_fail_closed", "false"),
    ],
)
def test_validation_fails_when_documented_default_does_not_match_schema_default(
    tmp_path: Path, field_name: str, wrong_documented_default: str
) -> None:
    docs_path = tmp_path / "api.md"
    schema_defaults = {
        "mimo_mtp_fastpath": "false",
        "mimo_mtp_depth": "null",
        "mimo_mtp_sidecar_path": "null",
        "mimo_mtp_fail_closed": "true",
    }
    docs_path.write_text(
        DOC_WITH_ALL_FIELDS.replace(
            f"`{field_name}`: Experimental. Disabled by default "
            f"(`{schema_defaults[field_name]}`)",
            f"`{field_name}`: Experimental. Disabled by default "
            f"(`{wrong_documented_default}`)",
        )
    )

    failures = validate_mimo_mtp_request_docs(docs_path)

    assert any(
        field_name in failure and wrong_documented_default in failure
        for failure in failures
    )


def test_discovers_mtp_request_fields_from_chat_completion_schema() -> None:
    assert discover_mimo_mtp_request_fields() == (
        "mimo_mtp_fastpath",
        "mimo_mtp_depth",
        "mimo_mtp_sidecar_path",
        "mimo_mtp_fail_closed",
    )


@pytest.mark.parametrize(
    "removed_field",
    [
        "mimo_mtp_fastpath",
        "mimo_mtp_depth",
        "mimo_mtp_sidecar_path",
        "mimo_mtp_fail_closed",
    ],
)
def test_validation_fails_when_mtp_request_field_is_missing(
    tmp_path: Path, removed_field: str
) -> None:
    docs_path = tmp_path / "api.md"
    docs_path.write_text(
        DOC_WITH_ALL_FIELDS.replace(f"* `{removed_field}`", "* `other_field`")
    )

    failures = validate_mimo_mtp_request_docs(docs_path)

    assert any(removed_field in failure for failure in failures)


@pytest.mark.parametrize("marker", MTP_DOC_MARKERS)
def test_validation_fails_when_field_is_not_marked_experimental_or_disabled(
    tmp_path: Path, marker: str
) -> None:
    docs_path = tmp_path / "api.md"
    docs_path.write_text(DOC_WITH_ALL_FIELDS.replace(marker, "required marker missing"))

    failures = validate_mimo_mtp_request_docs(docs_path)

    assert failures
    assert any(marker in failure for failure in failures)


def test_validation_passes_when_every_mtp_field_has_required_markers(
    tmp_path: Path,
) -> None:
    docs_path = tmp_path / "api.md"
    docs_path.write_text(DOC_WITH_ALL_FIELDS)

    assert validate_mimo_mtp_request_docs(docs_path) == []


def test_documented_validator_command_runs_from_ultrawork_directory() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts/validate_mimo_mtp_request_docs.py")],
        cwd=repo_root / ".goose-ultrawork",
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "MiMo MTP request documentation validation passed" in result.stdout
