from __future__ import annotations

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

* `mimo_mtp_fastpath`: Experimental. Disabled by default.
* `mimo_mtp_depth`: Experimental. Disabled by default.
* `mimo_mtp_sidecar_path`: Experimental. Disabled by default.
* `mimo_mtp_fail_closed`: Experimental. Disabled by default.
"""


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
