"""Pure function that compares two snapshot dicts field-by-field and produces a structured diff.

This module provides :func:`compare_snapshots`, a pure function that takes two
``dict[str, object]`` snapshots and returns a :class:`SnapshotDiff` with
categorised changes: added keys, removed keys, and changed values (with old/new).
The result is designed to produce clear, actionable assertion error messages
without requiring callers to format diffs manually.

Design decisions:
- **Pure function** — no side effects, no I/O, no mutation of inputs.
- **Shallow comparison** — nested dicts are compared by value identity
  (``!=``), not recursively diffed. This matches the typical use case of
  flat snapshot dicts serialised from Pydantic models.
- **Deterministic ordering** — keys are sorted so that error messages are
  stable across runs and Python versions.
- **Typed result** — :class:`SnapshotDiff` is a frozen Pydantic model with
  ``__str__`` rendering a human-readable diff suitable for ``assert`` messages.
- **Semantic equality** — values are compared with ``!=`` (not ``is not``),
  so ``1`` and ``1.0`` are considered equal if Python equality says so.
  This is intentional for snapshot dicts that may mix int/float from JSON.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import final

from pydantic import BaseModel, ConfigDict


@final
class ValueChange(BaseModel):
    """A single field-level change: the key, its old value, and its new value.

    Frozen and strict per project conventions.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    key: str
    old_value: object
    new_value: object

    def __str__(self) -> str:
        return f"  {self.key}: {self.old_value!r} → {self.new_value!r}"


@final
class SnapshotDiff(BaseModel):
    """Structured diff result from comparing two snapshot dicts.

    Attributes
    ----------
    added_keys:
        Keys present in ``after`` but absent from ``before``, sorted
        alphabetically.
    removed_keys:
        Keys present in ``before`` but absent from ``after``, sorted
        alphabetically.
    changed_values:
        Keys present in both dicts but with different values, sorted by key
        alphabetically.  Each entry carries the old and new value.
    is_empty:
        ``True`` when no differences were found (all three collections are
        empty).

    The model is frozen and strict.  Its ``__str__`` method renders a
    human-readable multi-line diff suitable for inclusion in assertion error
    messages.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    added_keys: tuple[str, ...]
    removed_keys: tuple[str, ...]
    changed_values: tuple[ValueChange, ...]

    @property
    def is_empty(self) -> bool:
        """True when the two snapshots are identical."""
        return (
            len(self.added_keys) == 0
            and len(self.removed_keys) == 0
            and len(self.changed_values) == 0
        )

    def __str__(self) -> str:
        if self.is_empty:
            return "Snapshots are identical (no differences)."

        lines: list[str] = []
        if self.added_keys:
            lines.append("Added keys:")
            for key in self.added_keys:
                lines.append(f"  + {key}")
        if self.removed_keys:
            lines.append("Removed keys:")
            for key in self.removed_keys:
                lines.append(f"  - {key}")
        if self.changed_values:
            lines.append("Changed values:")
            for change in self.changed_values:
                lines.append(str(change))
        return "\n".join(lines)


def compare_snapshots(
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> SnapshotDiff:
    """Compare two snapshot dicts field-by-field and return a structured diff.

    This is a **pure function**: it does not mutate its inputs and has no
    side effects.  The returned :class:`SnapshotDiff` carries categorised
    changes suitable for assertion error messages.

    Parameters
    ----------
    before:
        The baseline / expected snapshot mapping.
    after:
        The current / actual snapshot mapping.

    Returns
    -------
    SnapshotDiff
        Structured diff with added keys, removed keys, and changed values.

    Examples
    --------
    >>> diff = compare_snapshots(
    ...     {"model_id": "test", "mode": "ar", "mtp_enabled": False},
    ...     {"model_id": "test", "mode": "mtp", "mtp_depth": 3},
    ... )
    >>> diff.added_keys
    ('mtp_depth',)
    >>> diff.removed_keys
    ('mtp_enabled',)
    >>> diff.changed_values[0].key
    'mode'
    >>> diff.changed_values[0].old_value
    'ar'
    >>> diff.changed_values[0].new_value
    'mtp'
    >>> print(diff)  # doctest: +NORMALIZE_WHITESPACE
    Added keys:
      + mtp_depth
    Removed keys:
      - mtp_enabled
    Changed values:
      mode: 'ar' → 'mtp'
    """
    before_keys = set(before.keys())
    after_keys = set(after.keys())

    added = sorted(after_keys - before_keys)
    removed = sorted(before_keys - after_keys)

    common_keys = sorted(before_keys & after_keys)
    changed: list[ValueChange] = []
    for key in common_keys:
        old_val = before[key]
        new_val = after[key]
        if old_val != new_val:
            changed.append(
                ValueChange(key=key, old_value=old_val, new_value=new_val)
            )

    return SnapshotDiff(
        added_keys=tuple(added),
        removed_keys=tuple(removed),
        changed_values=tuple(changed),
    )
