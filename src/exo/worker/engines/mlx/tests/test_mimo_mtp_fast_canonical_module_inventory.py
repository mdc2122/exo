"""Test that the inventory enumeration function output matches the canonical module fixture exactly.

This test loads the known fixture defining the expected canonical module list and git index
blob SHAs, runs the inventory enumeration function with blob SHA resolution enabled,
and asserts every field matches exactly: module_name, relative_path, and blob_sha per row,
in the same order.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, cast

from exo.worker.engines.mlx.mimo_mtp_fast.module_validation import (
    MIMO_MTP_FASTPATH_INVENTORY_SCHEMA_VERSION,
    discover_mimo_mtp_fastpath_modules,
)


class _FixtureModuleRow(TypedDict):
    module_name: str
    relative_path: str
    blob_sha: str


class _FixtureData(TypedDict):
    schema_version: str
    fixture_description: str
    performance_claim: str
    modules: list[_FixtureModuleRow]


FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent.parent.parent.parent
    / "scripts"
    / "fixtures"
    / "mimo_mtp_fastpath_canonical_module_inventory.json"
)


def _load_fixture() -> _FixtureData:
    raw: object = json.loads(FIXTURE_PATH.read_text())
    return cast(_FixtureData, raw)


def test_fixture_exists_and_is_loadable() -> None:
    assert FIXTURE_PATH.is_file(), f"Fixture not found at {FIXTURE_PATH}"
    fixture = _load_fixture()
    assert "schema_version" in fixture
    assert "modules" in fixture


def test_fixture_schema_version_matches_inventory_schema_version() -> None:
    fixture = _load_fixture()
    assert fixture["schema_version"] == MIMO_MTP_FASTPATH_INVENTORY_SCHEMA_VERSION


def test_enumeration_output_matches_fixture_exactly() -> None:
    """Run the inventory enumeration with blob SHAs and assert exact match with fixture."""
    fixture = _load_fixture()
    fixture_modules = fixture["modules"]

    report = discover_mimo_mtp_fastpath_modules(
        repo_root=Path.cwd(),
        include_blob_shas=True,
    )

    # Same number of modules
    assert len(report.results) == len(fixture_modules), (
        f"Enumeration returned {len(report.results)} modules, "
        f"fixture has {len(fixture_modules)} modules"
    )

    # Per-row exact match: module_name, relative_path, blob_sha
    mismatches: list[str] = []
    for index, (result, fixture_row) in enumerate(
        zip(report.results, fixture_modules, strict=True)
    ):
        fixture_module_name = fixture_row["module_name"]
        fixture_relative_path = fixture_row["relative_path"]
        fixture_blob_sha = fixture_row["blob_sha"]

        if result.module_name != fixture_module_name:
            mismatches.append(
                f"Row {index}: module_name mismatch: "
                f"got {result.module_name!r}, expected {fixture_module_name!r}"
            )
        if result.relative_path != fixture_relative_path:
            mismatches.append(
                f"Row {index}: relative_path mismatch: "
                f"got {result.relative_path!r}, expected {fixture_relative_path!r}"
            )
        if result.blob_sha != fixture_blob_sha:
            mismatches.append(
                f"Row {index}: blob_sha mismatch: "
                f"got {result.blob_sha!r}, expected {fixture_blob_sha!r}"
            )

    assert not mismatches, (
        "Enumeration output does not match fixture exactly:\n"
        + "\n".join(mismatches)
    )


def test_enumeration_without_blob_shas_omits_blob_sha_field() -> None:
    """Verify backward compatibility: without include_blob_shas, blob_sha is None."""
    report = discover_mimo_mtp_fastpath_modules(
        repo_root=Path.cwd(),
        include_blob_shas=False,
    )
    for result in report.results:
        assert result.blob_sha is None, (
            f"Expected blob_sha=None for {result.module_name} when "
            f"include_blob_shas=False, got {result.blob_sha!r}"
        )


def test_fixture_blob_shas_are_40_char_hex() -> None:
    """Sanity check that fixture blob SHAs look like valid git object IDs."""
    fixture = _load_fixture()
    fixture_modules = fixture["modules"]
    for index, fixture_row in enumerate(fixture_modules):
        blob_sha = fixture_row["blob_sha"]
        assert isinstance(blob_sha, str), (
            f"Fixture row {index} blob_sha must be str, got {type(blob_sha).__name__}"
        )
        assert len(blob_sha) == 40, (
            f"Fixture row {index} blob_sha must be 40 chars, got {len(blob_sha)}"
        )
        assert all(c in "0123456789abcdef" for c in blob_sha), (
            f"Fixture row {index} blob_sha must be lowercase hex, got {blob_sha!r}"
        )
