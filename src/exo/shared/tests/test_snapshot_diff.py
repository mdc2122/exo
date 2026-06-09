"""Unit tests for compare_snapshots, SnapshotDiff, and ValueChange (AC Sub-AC 3).

Verifies:
- Identical snapshots produce an empty diff.
- Added keys are detected and sorted alphabetically.
- Removed keys are detected and sorted alphabetically.
- Changed values are detected with old/new, sorted by key.
- Combinations of added, removed, and changed keys all appear together.
- __str__ rendering is human-readable and suitable for assertion error messages.
- is_empty property reflects diff content.
- ValueChange is frozen, strict, and extra-forbid.
- SnapshotDiff is frozen, strict, and extra-forbid.
- Pure function: inputs are not mutated.
- Shallow comparison: nested dicts are compared by value, not recursively diffed.
- Semantic equality: 1 and 1.0 are considered equal (Python equality semantics).
- Deterministic ordering: keys are sorted regardless of insertion order.
- Round-trip serialization of SnapshotDiff and ValueChange.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from exo.shared.types.snapshot_diff import SnapshotDiff, ValueChange, compare_snapshots

# ---------------------------------------------------------------------------
# Identical snapshots
# ---------------------------------------------------------------------------


class TestIdenticalSnapshots:
    def test_empty_dicts_produce_empty_diff(self) -> None:
        diff = compare_snapshots({}, {})
        assert diff.is_empty

    def test_same_dict_produces_empty_diff(self) -> None:
        snap = {"model_id": "test", "mode": "ar", "mtp_enabled": False}
        diff = compare_snapshots(snap, snap)
        assert diff.is_empty
        assert diff.added_keys == ()
        assert diff.removed_keys == ()
        assert diff.changed_values == ()

    def test_identical_but_distinct_dicts_produce_empty_diff(self) -> None:
        before = {"a": 1, "b": "hello"}
        after = {"a": 1, "b": "hello"}
        diff = compare_snapshots(before, after)
        assert diff.is_empty

    def test_str_for_identical_snapshots(self) -> None:
        diff = compare_snapshots({"x": 1}, {"x": 1})
        assert str(diff) == "Snapshots are identical (no differences)."


# ---------------------------------------------------------------------------
# Added keys
# ---------------------------------------------------------------------------


class TestAddedKeys:
    def test_single_added_key(self) -> None:
        diff = compare_snapshots({"a": 1}, {"a": 1, "b": 2})
        assert diff.added_keys == ("b",)
        assert diff.is_empty is False

    def test_multiple_added_keys_sorted(self) -> None:
        diff = compare_snapshots({}, {"z": 1, "a": 2, "m": 3})
        assert diff.added_keys == ("a", "m", "z")

    def test_added_key_with_none_value(self) -> None:
        diff = compare_snapshots({"a": 1}, {"a": 1, "b": None})
        assert diff.added_keys == ("b",)

    def test_str_shows_added_keys(self) -> None:
        diff = compare_snapshots({}, {"mtp_depth": 3})
        rendered = str(diff)
        assert "Added keys:" in rendered
        assert "+ mtp_depth" in rendered


# ---------------------------------------------------------------------------
# Removed keys
# ---------------------------------------------------------------------------


class TestRemovedKeys:
    def test_single_removed_key(self) -> None:
        diff = compare_snapshots({"a": 1, "b": 2}, {"a": 1})
        assert diff.removed_keys == ("b",)
        assert diff.is_empty is False

    def test_multiple_removed_keys_sorted(self) -> None:
        diff = compare_snapshots({"z": 1, "a": 2, "m": 3}, {})
        assert diff.removed_keys == ("a", "m", "z")

    def test_removed_key_with_none_value(self) -> None:
        diff = compare_snapshots({"a": 1, "b": None}, {"a": 1})
        assert diff.removed_keys == ("b",)

    def test_str_shows_removed_keys(self) -> None:
        diff = compare_snapshots({"mtp_enabled": False}, {})
        rendered = str(diff)
        assert "Removed keys:" in rendered
        assert "- mtp_enabled" in rendered


# ---------------------------------------------------------------------------
# Changed values
# ---------------------------------------------------------------------------


class TestChangedValues:
    def test_single_changed_value(self) -> None:
        diff = compare_snapshots({"mode": "ar"}, {"mode": "mtp"})
        assert len(diff.changed_values) == 1
        assert diff.changed_values[0].key == "mode"
        assert diff.changed_values[0].old_value == "ar"
        assert diff.changed_values[0].new_value == "mtp"

    def test_multiple_changed_values_sorted_by_key(self) -> None:
        diff = compare_snapshots(
            {"z": 0, "a": 0, "m": 0},
            {"z": 1, "a": 1, "m": 1},
        )
        keys = [c.key for c in diff.changed_values]
        assert keys == ["a", "m", "z"]

    def test_changed_value_with_type_change(self) -> None:
        diff = compare_snapshots({"count": 1}, {"count": "two"})
        assert diff.changed_values[0].old_value == 1
        assert diff.changed_values[0].new_value == "two"

    def test_changed_value_from_none(self) -> None:
        diff = compare_snapshots({"x": None}, {"x": 42})
        assert diff.changed_values[0].old_value is None
        assert diff.changed_values[0].new_value == 42

    def test_changed_value_to_none(self) -> None:
        diff = compare_snapshots({"x": 42}, {"x": None})
        assert diff.changed_values[0].old_value == 42
        assert diff.changed_values[0].new_value is None

    def test_str_shows_changed_values(self) -> None:
        diff = compare_snapshots({"mode": "ar"}, {"mode": "mtp"})
        rendered = str(diff)
        assert "Changed values:" in rendered
        assert "mode:" in rendered
        assert "'ar'" in rendered
        assert "'mtp'" in rendered

    def test_value_change_str_format(self) -> None:
        change = ValueChange(key="generation_tps", old_value=18.5, new_value=35.0)
        rendered = str(change)
        assert "generation_tps" in rendered
        assert "18.5" in rendered
        assert "35.0" in rendered
        assert "→" in rendered


# ---------------------------------------------------------------------------
# Semantic equality
# ---------------------------------------------------------------------------


class TestSemanticEquality:
    def test_int_and_float_equal_values_not_changed(self) -> None:
        """Python equality: 1 == 1.0, so this is NOT a change."""
        diff = compare_snapshots({"count": 1}, {"count": 1.0})
        assert diff.is_empty

    def test_bool_and_int_not_equal(self) -> None:
        """In strict Python, True != 1 for dict value comparison when types differ.
        Actually Python: True == 1 is True. So this should be no change."""
        diff = compare_snapshots({"flag": True}, {"flag": 1})
        assert diff.is_empty

    def test_string_and_int_not_equal(self) -> None:
        diff = compare_snapshots({"val": "1"}, {"val": 1})
        assert len(diff.changed_values) == 1
        assert diff.changed_values[0].old_value == "1"
        assert diff.changed_values[0].new_value == 1


# ---------------------------------------------------------------------------
# Shallow comparison (nested dicts)
# ---------------------------------------------------------------------------


class TestShallowComparison:
    def test_nested_dict_same_reference_no_change(self) -> None:
        nested = {"inner": 1}
        diff = compare_snapshots({"data": nested}, {"data": nested})
        assert diff.is_empty

    def test_nested_dict_equal_values_no_change(self) -> None:
        """Two equal nested dicts are not a change (Python == on dicts is deep)."""
        diff = compare_snapshots(
            {"data": {"inner": 1}},
            {"data": {"inner": 1}},
        )
        assert diff.is_empty

    def test_nested_dict_different_values_is_change(self) -> None:
        diff = compare_snapshots(
            {"data": {"inner": 1}},
            {"data": {"inner": 2}},
        )
        assert len(diff.changed_values) == 1
        assert diff.changed_values[0].key == "data"
        # Old and new are the nested dicts themselves (shallow diff)
        assert diff.changed_values[0].old_value == {"inner": 1}
        assert diff.changed_values[0].new_value == {"inner": 2}

    def test_nested_dict_shown_as_whole_in_change(self) -> None:
        """The diff does not recurse; the whole nested dict is the old/new value."""
        diff = compare_snapshots(
            {"topology": {"nodes": ["a", "b"]}},
            {"topology": {"nodes": ["a", "b", "c"]}},
        )
        assert len(diff.changed_values) == 1
        assert diff.changed_values[0].key == "topology"


# ---------------------------------------------------------------------------
# Combinations
# ---------------------------------------------------------------------------


class TestCombinations:
    def test_added_and_removed_and_changed(self) -> None:
        diff = compare_snapshots(
            {"model_id": "old", "mode": "ar", "mtp_enabled": False},
            {"model_id": "new", "mode": "ar", "mtp_depth": 3},
        )
        assert diff.added_keys == ("mtp_depth",)
        assert diff.removed_keys == ("mtp_enabled",)
        assert len(diff.changed_values) == 1
        assert diff.changed_values[0].key == "model_id"

    def test_str_renders_all_categories(self) -> None:
        diff = compare_snapshots(
            {"a": 1, "b": 2, "c": 3},
            {"a": 10, "b": 2, "d": 4},
        )
        rendered = str(diff)
        assert "Added keys:" in rendered
        assert "Removed keys:" in rendered
        assert "Changed values:" in rendered

    def test_only_added_no_removed_or_changed(self) -> None:
        diff = compare_snapshots({"a": 1}, {"a": 1, "b": 2})
        assert diff.added_keys == ("b",)
        assert diff.removed_keys == ()
        assert diff.changed_values == ()

    def test_only_removed_no_added_or_changed(self) -> None:
        diff = compare_snapshots({"a": 1, "b": 2}, {"a": 1})
        assert diff.added_keys == ()
        assert diff.removed_keys == ("b",)
        assert diff.changed_values == ()


# ---------------------------------------------------------------------------
# Deterministic ordering
# ---------------------------------------------------------------------------


class TestDeterministicOrdering:
    def test_keys_sorted_regardless_of_dict_insertion_order(self) -> None:
        before: dict[str, object] = {}
        after: dict[str, object] = {"z_key": 1, "a_key": 2, "m_key": 3}
        diff = compare_snapshots(before, after)
        assert diff.added_keys == ("a_key", "m_key", "z_key")

    def test_changed_values_sorted_by_key(self) -> None:
        diff = compare_snapshots(
            {"z": 0, "a": 0, "m": 0},
            {"z": 1, "a": 1, "m": 1},
        )
        keys = [c.key for c in diff.changed_values]
        assert keys == ["a", "m", "z"]


# ---------------------------------------------------------------------------
# Pure function: inputs not mutated
# ---------------------------------------------------------------------------


class TestInputImmutability:
    def test_before_dict_not_mutated(self) -> None:
        before: dict[str, object] = {"a": 1}
        after: dict[str, object] = {"a": 1, "b": 2}
        compare_snapshots(before, after)
        assert before == {"a": 1}
        assert "b" not in before

    def test_after_dict_not_mutated(self) -> None:
        before: dict[str, object] = {"a": 1, "b": 2}
        after: dict[str, object] = {"a": 1}
        compare_snapshots(before, after)
        assert after == {"a": 1}
        assert "b" not in after


# ---------------------------------------------------------------------------
# Frozen model enforcement
# ---------------------------------------------------------------------------


class TestFrozenModel:
    def test_value_change_is_frozen(self) -> None:
        change = ValueChange(key="x", old_value=1, new_value=2)
        with pytest.raises(ValidationError):
            change.key = "y"

    def test_snapshot_diff_is_frozen(self) -> None:
        diff = SnapshotDiff(
            added_keys=(),
            removed_keys=(),
            changed_values=(),
        )
        with pytest.raises(ValidationError):
            diff.added_keys = ("new",)


# ---------------------------------------------------------------------------
# Strict mode: type coercion rejected
# ---------------------------------------------------------------------------


class TestStrictMode:
    def test_value_change_rejects_coerced_key(self) -> None:
        with pytest.raises(ValidationError):
            ValueChange(key=123, old_value=1, new_value=2)  # type: ignore[arg-type]

    def test_snapshot_diff_rejects_coerced_added_keys_list(self) -> None:
        """Strict mode rejects a list where a tuple is expected."""
        with pytest.raises(ValidationError):
            SnapshotDiff(
                added_keys=["a"],  # type: ignore[arg-type]
                removed_keys=(),
                changed_values=(),
            )


# ---------------------------------------------------------------------------
# Extra fields forbidden
# ---------------------------------------------------------------------------


class TestExtraForbidden:
    def test_value_change_rejects_extra_field(self) -> None:
        with pytest.raises(ValidationError):
            ValueChange(key="x", old_value=1, new_value=2, extra=True)  # type: ignore[call-arg]

    def test_snapshot_diff_rejects_extra_field(self) -> None:
        with pytest.raises(ValidationError):
            SnapshotDiff(
                added_keys=(),
                removed_keys=(),
                changed_values=(),
                extra_field=True,  # type: ignore[call-arg]
            )


# ---------------------------------------------------------------------------
# Round-trip serialization
# ---------------------------------------------------------------------------


class TestRoundTripSerialization:
    def test_value_change_round_trip(self) -> None:
        change = ValueChange(key="generation_tps", old_value=18.5, new_value=35.0)
        dumped = change.model_dump()
        restored = ValueChange.model_validate(dumped)
        assert restored == change

    def test_value_change_json_round_trip(self) -> None:
        change = ValueChange(key="mode", old_value="ar", new_value="mtp")
        json_str = change.model_dump_json()
        restored = ValueChange.model_validate_json(json_str)
        assert restored == change

    def test_snapshot_diff_round_trip(self) -> None:
        diff = compare_snapshots(
            {"a": 1, "b": 2},
            {"a": 10, "c": 3},
        )
        dumped = diff.model_dump()
        restored = SnapshotDiff.model_validate(dumped)
        assert restored == diff

    def test_snapshot_diff_json_round_trip(self) -> None:
        diff = compare_snapshots(
            {"mode": "ar", "mtp_enabled": False},
            {"mode": "mtp", "mtp_depth": 3},
        )
        json_str = diff.model_dump_json()
        restored = SnapshotDiff.model_validate_json(json_str)
        assert restored == diff


# ---------------------------------------------------------------------------
# is_empty property
# ---------------------------------------------------------------------------


class TestIsEmptyProperty:
    def test_empty_diff_is_empty(self) -> None:
        diff = compare_snapshots({"a": 1}, {"a": 1})
        assert diff.is_empty is True

    def test_added_keys_makes_not_empty(self) -> None:
        diff = compare_snapshots({}, {"a": 1})
        assert diff.is_empty is False

    def test_removed_keys_makes_not_empty(self) -> None:
        diff = compare_snapshots({"a": 1}, {})
        assert diff.is_empty is False

    def test_changed_values_makes_not_empty(self) -> None:
        diff = compare_snapshots({"a": 1}, {"a": 2})
        assert diff.is_empty is False


# ---------------------------------------------------------------------------
# Typical benchmark snapshot use case
# ---------------------------------------------------------------------------


class TestBenchmarkSnapshotUseCase:
    """Exercise compare_snapshots with realistic benchmark row snapshots."""

    def test_comparing_ar_vs_mtp_row_snapshots(self) -> None:
        ar_snapshot: dict[str, object] = {
            "schema_version": "1.0.0",
            "mode": "ar",
            "mtp_enabled": False,
            "mtp_depth": 0,
            "generation_tps": 18.5,
        }
        mtp_snapshot: dict[str, object] = {
            "schema_version": "1.0.0",
            "mode": "mtp",
            "mtp_enabled": True,
            "mtp_depth": 3,
            "generation_tps": 35.0,
        }
        diff = compare_snapshots(ar_snapshot, mtp_snapshot)
        assert diff.is_empty is False
        assert diff.added_keys == ()
        assert diff.removed_keys == ()
        changed_keys = {c.key for c in diff.changed_values}
        assert "mode" in changed_keys
        assert "mtp_enabled" in changed_keys
        assert "mtp_depth" in changed_keys
        assert "generation_tps" in changed_keys

    def test_assertion_error_message_pattern(self) -> None:
        """The str() output should be suitable for use in an assert message."""
        before: dict[str, object] = {
            "model_id": "test-model",
            "mode": "ar",
            "mtp_enabled": False,
        }
        after: dict[str, object] = {
            "model_id": "test-model",
            "mode": "mtp",
            "mtp_depth": 3,
        }
        diff = compare_snapshots(before, after)
        message = f"Snapshot regression detected:\n{diff}"
        assert "Snapshot regression detected:" in message
        assert "+ mtp_depth" in message
        assert "- mtp_enabled" in message
        assert "mode:" in message
