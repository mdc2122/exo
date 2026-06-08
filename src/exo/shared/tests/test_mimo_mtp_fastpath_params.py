"""Focused unit tests for MimoMtpFastpathParams: construction, field defaults, immutability, and strict-mode constraints.

These tests verify the AC requirements:
- Immutable Pydantic model (frozen/strict)
- Optional fields defaulting to None/disabled
- Unit test construction, field defaults, and immutability constraints
"""

# pyright: reportAny=false

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from exo.shared.types.text_generation import (
    SUPPORTED_MIMO_MTP_FASTPATH_DEPTHS,
    MimoMtpFastpathParams,
)


def _coerced_dict(**kwargs: Any) -> dict[str, Any]:  # noqa: ANN003
    """Build a dict with intentionally wrong types for strict-mode rejection tests."""
    return dict(kwargs)


# ---------------------------------------------------------------------------
# Construction & field defaults
# ---------------------------------------------------------------------------


class TestMimoMtpFastpathParamsConstruction:
    """Verify that MimoMtpFastpathParams can be constructed with defaults and explicit values."""

    def test_default_construction_sets_correct_defaults(self) -> None:
        params = MimoMtpFastpathParams()
        assert params.enabled is True
        assert params.depth is None
        assert params.sidecar_path is None
        assert params.fail_closed is True

    def test_default_construction_produces_expected_dump(self) -> None:
        params = MimoMtpFastpathParams()
        dumped = params.model_dump()
        assert dumped == {
            "enabled": True,
            "depth": None,
            "sidecar_path": None,
            "fail_closed": True,
        }

    def test_optional_fields_are_none_by_default(self) -> None:
        """The AC requires Optional fields defaulting to None/disabled."""
        params = MimoMtpFastpathParams()
        # depth and sidecar_path are the Optional fields; they must be None
        assert params.depth is None, "depth must default to None"
        assert params.sidecar_path is None, "sidecar_path must default to None"

    def test_enabled_defaults_to_true(self) -> None:
        """When a caller explicitly constructs MimoMtpFastpathParams, enabled=True signals MTP intent.

        Default AR generation is preserved at the TextGenerationTaskParams level
        where mimo_mtp_fastpath_params itself defaults to None (absent = AR).
        """
        params = MimoMtpFastpathParams()
        assert params.enabled is True

    def test_fail_closed_defaults_to_true(self) -> None:
        """fail_closed=True is the safe default: if MTP cannot proceed, raise rather than silently fall back to AR."""
        params = MimoMtpFastpathParams()
        assert params.fail_closed is True

    def test_explicit_construction_with_all_fields(self) -> None:
        params = MimoMtpFastpathParams(
            enabled=True,
            depth=3,
            sidecar_path="/models/mimo-v2.5-pro-mtp.safetensors",
            fail_closed=True,
        )
        assert params.enabled is True
        assert params.depth == 3
        assert params.sidecar_path == "/models/mimo-v2.5-pro-mtp.safetensors"
        assert params.fail_closed is True

    def test_explicit_construction_with_disabled_enabled(self) -> None:
        """A caller can explicitly set enabled=False for explicit disabled intent."""
        params = MimoMtpFastpathParams(
            enabled=False,
            depth=None,
            sidecar_path=None,
            fail_closed=True,
        )
        assert params.enabled is False
        assert params.depth is None
        assert params.sidecar_path is None
        assert params.fail_closed is True

    def test_explicit_construction_with_fail_closed_false(self) -> None:
        params = MimoMtpFastpathParams(
            enabled=True,
            depth=2,
            sidecar_path="/models/model_mtp.safetensors",
            fail_closed=False,
        )
        assert params.fail_closed is False

    def test_each_supported_depth_is_accepted(self) -> None:
        for depth in SUPPORTED_MIMO_MTP_FASTPATH_DEPTHS:
            params = MimoMtpFastpathParams(depth=depth)
            assert params.depth == depth

    def test_depth_none_is_accepted(self) -> None:
        params = MimoMtpFastpathParams(depth=None)
        assert params.depth is None

    def test_model_config_is_frozen(self) -> None:
        """The model_config must declare frozen=True."""
        assert MimoMtpFastpathParams.model_config.get("frozen") is True

    def test_model_config_is_strict(self) -> None:
        """The model_config must declare strict=True."""
        assert MimoMtpFastpathParams.model_config.get("strict") is True

    def test_model_config_forbids_extra(self) -> None:
        """The model_config must forbid extra fields."""
        assert MimoMtpFastpathParams.model_config.get("extra") == "forbid"


# ---------------------------------------------------------------------------
# Immutability constraints
# ---------------------------------------------------------------------------


class TestMimoMtpFastpathParamsImmutability:
    """Verify that MimoMtpFastpathParams instances are truly immutable.

    These tests intentionally attempt to assign to frozen model attributes,
    so we suppress the type checker's read-only attribute warnings.
    """

    def test_cannot_set_enabled(self) -> None:
        params = MimoMtpFastpathParams()
        with pytest.raises(ValidationError):
            params.enabled = False  # type: ignore[reportAttributeAccessIssue]

    def test_cannot_set_depth(self) -> None:
        params = MimoMtpFastpathParams()
        with pytest.raises(ValidationError):
            params.depth = 2  # type: ignore[reportAttributeAccessIssue]

    def test_cannot_set_sidecar_path(self) -> None:
        params = MimoMtpFastpathParams()
        with pytest.raises(ValidationError):
            params.sidecar_path = "/other/path"  # type: ignore[reportAttributeAccessIssue]

    def test_cannot_set_fail_closed(self) -> None:
        params = MimoMtpFastpathParams()
        with pytest.raises(ValidationError):
            params.fail_closed = False  # type: ignore[reportAttributeAccessIssue]

    def test_cannot_delete_enabled(self) -> None:
        params = MimoMtpFastpathParams()
        with pytest.raises(ValidationError):
            del params.enabled  # type: ignore[reportAttributeAccessIssue]

    def test_cannot_delete_depth(self) -> None:
        params = MimoMtpFastpathParams()
        with pytest.raises(ValidationError):
            del params.depth  # type: ignore[reportAttributeAccessIssue]

    def test_cannot_mutate_after_explicit_construction(self) -> None:
        params = MimoMtpFastpathParams(
            enabled=True,
            depth=3,
            sidecar_path="/models/mtp.safetensors",
            fail_closed=True,
        )
        with pytest.raises(ValidationError):
            params.depth = 1  # type: ignore[reportAttributeAccessIssue]
        # Original value must be preserved
        assert params.depth == 3

    def test_model_copy_with_update_produces_new_instance(self) -> None:
        """model_copy is the only way to get a modified version; the original stays immutable."""
        original = MimoMtpFastpathParams(enabled=True, depth=2, fail_closed=True)
        modified = original.model_copy(update={"depth": 3})
        assert original.depth == 2, "original must remain unchanged"
        assert modified.depth == 3
        assert modified is not original


# ---------------------------------------------------------------------------
# Strict-mode & type enforcement
# ---------------------------------------------------------------------------


class TestMimoMtpFastpathParamsStrictMode:
    """Verify that strict=True prevents type coercion."""

    def test_rejects_string_enabled(self) -> None:
        with pytest.raises(ValidationError):
            MimoMtpFastpathParams.model_validate(_coerced_dict(enabled="true"))

    def test_rejects_string_depth(self) -> None:
        with pytest.raises(ValidationError):
            MimoMtpFastpathParams.model_validate(_coerced_dict(depth="3"))

    def test_rejects_int_sidecar_path(self) -> None:
        with pytest.raises(ValidationError):
            MimoMtpFastpathParams.model_validate(_coerced_dict(sidecar_path=123))

    def test_rejects_string_fail_closed(self) -> None:
        with pytest.raises(ValidationError):
            MimoMtpFastpathParams.model_validate(_coerced_dict(fail_closed="false"))

    def test_rejects_float_depth(self) -> None:
        with pytest.raises(ValidationError):
            MimoMtpFastpathParams.model_validate(_coerced_dict(depth=2.0))

    def test_rejects_extra_field(self) -> None:
        with pytest.raises(ValidationError):
            MimoMtpFastpathParams.model_validate(
                _coerced_dict(enabled=True, unexpected_field=42)
            )

    def test_rejects_zero_depth(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            MimoMtpFastpathParams(depth=0)
        error_message = str(exc_info.value)
        assert "unsupported MTP depth 0" in error_message

    def test_rejects_negative_depth(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            MimoMtpFastpathParams(depth=-1)
        error_message = str(exc_info.value)
        assert "unsupported MTP depth -1" in error_message

    def test_rejects_depth_4(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            MimoMtpFastpathParams(depth=4)
        error_message = str(exc_info.value)
        assert "unsupported MTP depth 4" in error_message
        assert "disable_reason=unsupported_depth" in error_message

    def test_rejects_depth_100(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            MimoMtpFastpathParams(depth=100)
        error_message = str(exc_info.value)
        assert "unsupported MTP depth 100" in error_message

    def test_model_validate_round_trip_preserves_types(self) -> None:
        original = MimoMtpFastpathParams(
            enabled=True, depth=2, sidecar_path="/m.safetensors", fail_closed=True
        )
        data = original.model_dump()
        restored = MimoMtpFastpathParams.model_validate(data)
        assert restored == original
        assert isinstance(restored.enabled, bool)
        assert isinstance(restored.depth, int)
        assert isinstance(restored.sidecar_path, str)
        assert isinstance(restored.fail_closed, bool)

    def test_model_validate_json_round_trip(self) -> None:
        original = MimoMtpFastpathParams(
            enabled=True, depth=1, sidecar_path="/m.safetensors", fail_closed=False
        )
        json_str = original.model_dump_json()
        restored = MimoMtpFastpathParams.model_validate_json(json_str)
        assert restored == original


# ---------------------------------------------------------------------------
# Backward compatibility: default AR preservation
# ---------------------------------------------------------------------------


class TestMimoMtpFastpathParamsDefaultArPreservation:
    """Verify that the model's defaults align with the constraint that default generation must remain AR.

    The key invariant is: when mimo_mtp_fastpath_params is None on TextGenerationTaskParams,
    generation is AR. When it is present, the caller has explicitly opted in.
    """

    def test_params_object_is_not_none_when_constructed(self) -> None:
        """A constructed MimoMtpFastpathParams is a real object, not None."""
        params = MimoMtpFastpathParams()
        assert params is not None

    def test_enabled_true_signals_explicit_mtp_intent(self) -> None:
        """When the params object exists with enabled=True, this is explicit MTP intent."""
        params = MimoMtpFastpathParams()
        assert params.enabled is True, "enabled=True on constructed params signals MTP intent"

    def test_fail_closed_true_prevents_silent_ar_fallback(self) -> None:
        """fail_closed=True ensures that if MTP cannot proceed, an error is raised rather than silently falling back to AR."""
        params = MimoMtpFastpathParams()
        assert params.fail_closed is True, "fail_closed=True prevents silent AR fallback"
