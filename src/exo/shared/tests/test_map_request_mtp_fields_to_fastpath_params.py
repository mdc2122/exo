"""Unit tests for map_request_mtp_fields_to_fastpath_params.

AC Sub-AC 5.3.1: Verify that each guarded request field (enabled flag,
depth, sidecar_path, fail_closed) maps to the correct MimoMtpFastpathParams
attribute with correct types and defaults.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from exo.shared.types.text_generation import (
    MimoMtpFastpathParams,
    MimoMtpRequestFields,
    map_request_mtp_fields_to_fastpath_params,
)

# ---------------------------------------------------------------------------
# Disabled-default: mimo_mtp_fastpath=False returns None
# ---------------------------------------------------------------------------


class TestDisabledDefaultReturnsNone:
    """When mimo_mtp_fastpath is False (the default), the function must
    return None regardless of what the other fields carry."""

    def test_all_defaults_returns_none(self) -> None:
        fields = MimoMtpRequestFields(
            mimo_mtp_fastpath=False,
            mimo_mtp_depth=None,
            mimo_mtp_sidecar_path=None,
            mimo_mtp_fail_closed=True,
        )
        assert map_request_mtp_fields_to_fastpath_params(fields) is None

    def test_disabled_with_explicit_depth_returns_none(self) -> None:
        fields = MimoMtpRequestFields(
            mimo_mtp_fastpath=False,
            mimo_mtp_depth=3,
            mimo_mtp_sidecar_path=None,
            mimo_mtp_fail_closed=True,
        )
        assert map_request_mtp_fields_to_fastpath_params(fields) is None

    def test_disabled_with_sidecar_path_returns_none(self) -> None:
        fields = MimoMtpRequestFields(
            mimo_mtp_fastpath=False,
            mimo_mtp_depth=None,
            mimo_mtp_sidecar_path="/models/model_mtp.safetensors",
            mimo_mtp_fail_closed=True,
        )
        assert map_request_mtp_fields_to_fastpath_params(fields) is None

    def test_disabled_with_fail_closed_false_returns_none(self) -> None:
        fields = MimoMtpRequestFields(
            mimo_mtp_fastpath=False,
            mimo_mtp_depth=None,
            mimo_mtp_sidecar_path=None,
            mimo_mtp_fail_closed=False,
        )
        assert map_request_mtp_fields_to_fastpath_params(fields) is None

    def test_disabled_with_all_fields_set_returns_none(self) -> None:
        """Even when all MTP fields are populated, enabled=False gates None."""
        fields = MimoMtpRequestFields(
            mimo_mtp_fastpath=False,
            mimo_mtp_depth=2,
            mimo_mtp_sidecar_path="/models/model_mtp.safetensors",
            mimo_mtp_fail_closed=False,
        )
        assert map_request_mtp_fields_to_fastpath_params(fields) is None


# ---------------------------------------------------------------------------
# Enabled intent: mimo_mtp_fastpath=True returns MimoMtpFastpathParams
# ---------------------------------------------------------------------------


class TestEnabledIntentReturnsFastpathParams:
    """When mimo_mtp_fastpath is True, the function must return a
    MimoMtpFastpathParams with enabled=True and fields mapped 1:1."""

    def test_enabled_with_all_defaults_except_flag(self) -> None:
        fields = MimoMtpRequestFields(
            mimo_mtp_fastpath=True,
            mimo_mtp_depth=None,
            mimo_mtp_sidecar_path=None,
            mimo_mtp_fail_closed=True,
        )
        result = map_request_mtp_fields_to_fastpath_params(fields)
        assert isinstance(result, MimoMtpFastpathParams)
        assert result.enabled is True
        assert result.depth is None
        assert result.sidecar_path is None
        assert result.fail_closed is True

    def test_enabled_with_explicit_depth_maps_depth(self) -> None:
        fields = MimoMtpRequestFields(
            mimo_mtp_fastpath=True,
            mimo_mtp_depth=3,
            mimo_mtp_sidecar_path=None,
            mimo_mtp_fail_closed=True,
        )
        result = map_request_mtp_fields_to_fastpath_params(fields)
        assert isinstance(result, MimoMtpFastpathParams)
        assert result.depth == 3

    def test_enabled_with_depth_1_maps_depth(self) -> None:
        fields = MimoMtpRequestFields(
            mimo_mtp_fastpath=True,
            mimo_mtp_depth=1,
            mimo_mtp_sidecar_path=None,
            mimo_mtp_fail_closed=True,
        )
        result = map_request_mtp_fields_to_fastpath_params(fields)
        assert isinstance(result, MimoMtpFastpathParams)
        assert result.depth == 1

    def test_enabled_with_depth_2_maps_depth(self) -> None:
        fields = MimoMtpRequestFields(
            mimo_mtp_fastpath=True,
            mimo_mtp_depth=2,
            mimo_mtp_sidecar_path=None,
            mimo_mtp_fail_closed=True,
        )
        result = map_request_mtp_fields_to_fastpath_params(fields)
        assert isinstance(result, MimoMtpFastpathParams)
        assert result.depth == 2

    def test_enabled_with_sidecar_path_maps_sidecar_path(self) -> None:
        fields = MimoMtpRequestFields(
            mimo_mtp_fastpath=True,
            mimo_mtp_depth=None,
            mimo_mtp_sidecar_path="/models/model_mtp.safetensors",
            mimo_mtp_fail_closed=True,
        )
        result = map_request_mtp_fields_to_fastpath_params(fields)
        assert isinstance(result, MimoMtpFastpathParams)
        assert result.sidecar_path == "/models/model_mtp.safetensors"

    def test_enabled_with_fail_closed_false_maps_fail_closed(self) -> None:
        fields = MimoMtpRequestFields(
            mimo_mtp_fastpath=True,
            mimo_mtp_depth=None,
            mimo_mtp_sidecar_path=None,
            mimo_mtp_fail_closed=False,
        )
        result = map_request_mtp_fields_to_fastpath_params(fields)
        assert isinstance(result, MimoMtpFastpathParams)
        assert result.fail_closed is False

    def test_enabled_with_fail_closed_true_maps_fail_closed(self) -> None:
        fields = MimoMtpRequestFields(
            mimo_mtp_fastpath=True,
            mimo_mtp_depth=None,
            mimo_mtp_sidecar_path=None,
            mimo_mtp_fail_closed=True,
        )
        result = map_request_mtp_fields_to_fastpath_params(fields)
        assert isinstance(result, MimoMtpFastpathParams)
        assert result.fail_closed is True

    def test_enabled_with_all_fields_set_maps_all_fields(self) -> None:
        fields = MimoMtpRequestFields(
            mimo_mtp_fastpath=True,
            mimo_mtp_depth=3,
            mimo_mtp_sidecar_path="/models/model_mtp.safetensors",
            mimo_mtp_fail_closed=False,
        )
        result = map_request_mtp_fields_to_fastpath_params(fields)
        assert isinstance(result, MimoMtpFastpathParams)
        assert result.enabled is True
        assert result.depth == 3
        assert result.sidecar_path == "/models/model_mtp.safetensors"
        assert result.fail_closed is False


# ---------------------------------------------------------------------------
# Type correctness: returned attributes have the expected Python types
# ---------------------------------------------------------------------------


class TestReturnTypeCorrectness:
    """Verify that the returned MimoMtpFastpathParams attributes have
    the exact Python types dictated by the model definition."""

    def test_enabled_is_bool(self) -> None:
        fields = MimoMtpRequestFields(
            mimo_mtp_fastpath=True,
            mimo_mtp_depth=None,
            mimo_mtp_sidecar_path=None,
            mimo_mtp_fail_closed=True,
        )
        result = map_request_mtp_fields_to_fastpath_params(fields)
        assert result is not None
        assert isinstance(result.enabled, bool)

    def test_depth_none_is_none_type(self) -> None:
        fields = MimoMtpRequestFields(
            mimo_mtp_fastpath=True,
            mimo_mtp_depth=None,
            mimo_mtp_sidecar_path=None,
            mimo_mtp_fail_closed=True,
        )
        result = map_request_mtp_fields_to_fastpath_params(fields)
        assert result is not None
        assert result.depth is None

    def test_depth_int_is_int_type(self) -> None:
        fields = MimoMtpRequestFields(
            mimo_mtp_fastpath=True,
            mimo_mtp_depth=2,
            mimo_mtp_sidecar_path=None,
            mimo_mtp_fail_closed=True,
        )
        result = map_request_mtp_fields_to_fastpath_params(fields)
        assert result is not None
        assert isinstance(result.depth, int)

    def test_sidecar_path_none_is_none_type(self) -> None:
        fields = MimoMtpRequestFields(
            mimo_mtp_fastpath=True,
            mimo_mtp_depth=None,
            mimo_mtp_sidecar_path=None,
            mimo_mtp_fail_closed=True,
        )
        result = map_request_mtp_fields_to_fastpath_params(fields)
        assert result is not None
        assert result.sidecar_path is None

    def test_sidecar_path_str_is_str_type(self) -> None:
        fields = MimoMtpRequestFields(
            mimo_mtp_fastpath=True,
            mimo_mtp_depth=None,
            mimo_mtp_sidecar_path="/models/model_mtp.safetensors",
            mimo_mtp_fail_closed=True,
        )
        result = map_request_mtp_fields_to_fastpath_params(fields)
        assert result is not None
        assert isinstance(result.sidecar_path, str)

    def test_fail_closed_is_bool(self) -> None:
        fields = MimoMtpRequestFields(
            mimo_mtp_fastpath=True,
            mimo_mtp_depth=None,
            mimo_mtp_sidecar_path=None,
            mimo_mtp_fail_closed=False,
        )
        result = map_request_mtp_fields_to_fastpath_params(fields)
        assert result is not None
        assert isinstance(result.fail_closed, bool)


# ---------------------------------------------------------------------------
# 1:1 field mapping contract verification
# ---------------------------------------------------------------------------


class TestFieldMappingContract:
    """Verify the exact 1:1 mapping contract between request fields
    and MimoMtpFastpathParams attributes."""

    @pytest.mark.parametrize(
        ("depth", "sidecar_path", "fail_closed"),
        [
            (None, None, True),
            (1, None, True),
            (2, "/models/sidecar.safetensors", False),
            (3, "/opt/mimo/mtp.safetensors", True),
            (None, "/data/model_mtp.safetensors", False),
        ],
    )
    def test_each_request_field_maps_to_correct_attribute(
        self,
        depth: int | None,
        sidecar_path: str | None,
        fail_closed: bool,
    ) -> None:
        fields = MimoMtpRequestFields(
            mimo_mtp_fastpath=True,
            mimo_mtp_depth=depth,
            mimo_mtp_sidecar_path=sidecar_path,
            mimo_mtp_fail_closed=fail_closed,
        )
        result = map_request_mtp_fields_to_fastpath_params(fields)
        assert result is not None
        # enabled is always True when mimo_mtp_fastpath is True
        assert result.enabled is True
        # depth, sidecar_path, and fail_closed map 1:1
        assert result.depth == depth
        assert result.sidecar_path == sidecar_path
        assert result.fail_closed == fail_closed

    def test_enabled_always_true_when_fastpath_flag_is_true(self) -> None:
        """The mapping always sets enabled=True, it does NOT carry the
        request's mimo_mtp_fastpath flag as the enabled value."""
        fields = MimoMtpRequestFields(
            mimo_mtp_fastpath=True,
            mimo_mtp_depth=None,
            mimo_mtp_sidecar_path=None,
            mimo_mtp_fail_closed=True,
        )
        result = map_request_mtp_fields_to_fastpath_params(fields)
        assert result is not None
        assert result.enabled is True


# ---------------------------------------------------------------------------
# Depth validation: unsupported depths rejected by MimoMtpFastpathParams
# ---------------------------------------------------------------------------


class TestDepthValidation:
    """The mapping function delegates depth validation to the
    MimoMtpFastpathParams validator. Unsupported depths must raise."""

    @pytest.mark.parametrize("unsupported_depth", [0, 4, 5, -1, 10])
    def test_unsupported_depth_raises_validation_error(
        self,
        unsupported_depth: int,
    ) -> None:
        fields = MimoMtpRequestFields(
            mimo_mtp_fastpath=True,
            mimo_mtp_depth=unsupported_depth,
            mimo_mtp_sidecar_path=None,
            mimo_mtp_fail_closed=True,
        )
        with pytest.raises(ValidationError) as exc_info:
            map_request_mtp_fields_to_fastpath_params(fields)
        error_message = str(exc_info.value)
        assert "unsupported MTP depth" in error_message
        assert "disable_reason=unsupported_depth" in error_message

    @pytest.mark.parametrize("supported_depth", [1, 2, 3])
    def test_supported_depths_do_not_raise(
        self,
        supported_depth: int,
    ) -> None:
        fields = MimoMtpRequestFields(
            mimo_mtp_fastpath=True,
            mimo_mtp_depth=supported_depth,
            mimo_mtp_sidecar_path=None,
            mimo_mtp_fail_closed=True,
        )
        result = map_request_mtp_fields_to_fastpath_params(fields)
        assert result is not None
        assert result.depth == supported_depth


# ---------------------------------------------------------------------------
# Immutability: returned params are frozen
# ---------------------------------------------------------------------------


class TestReturnedParamsAreFrozen:
    """MimoMtpFastpathParams is frozen=True; verify immutability."""

    def test_cannot_set_enabled(self) -> None:
        fields = MimoMtpRequestFields(
            mimo_mtp_fastpath=True,
            mimo_mtp_depth=None,
            mimo_mtp_sidecar_path=None,
            mimo_mtp_fail_closed=True,
        )
        result = map_request_mtp_fields_to_fastpath_params(fields)
        assert result is not None
        with pytest.raises(ValidationError):
            result.enabled = False  # type: ignore[misc]

    def test_cannot_set_depth(self) -> None:
        fields = MimoMtpRequestFields(
            mimo_mtp_fastpath=True,
            mimo_mtp_depth=2,
            mimo_mtp_sidecar_path=None,
            mimo_mtp_fail_closed=True,
        )
        result = map_request_mtp_fields_to_fastpath_params(fields)
        assert result is not None
        with pytest.raises(ValidationError):
            result.depth = 3  # type: ignore[misc]

    def test_cannot_set_fail_closed(self) -> None:
        fields = MimoMtpRequestFields(
            mimo_mtp_fastpath=True,
            mimo_mtp_depth=None,
            mimo_mtp_sidecar_path=None,
            mimo_mtp_fail_closed=False,
        )
        result = map_request_mtp_fields_to_fastpath_params(fields)
        assert result is not None
        with pytest.raises(ValidationError):
            result.fail_closed = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MimoMtpRequestFields input validation
# ---------------------------------------------------------------------------


class TestMimoMtpRequestFieldsValidation:
    """MimoMtpRequestFields uses strict=True and extra='forbid';
    verify that invalid inputs are rejected."""

    def test_rejects_coerced_fastpath_flag(self) -> None:
        with pytest.raises(ValidationError):
            MimoMtpRequestFields(
                mimo_mtp_fastpath="true",  # type: ignore[arg-type]
                mimo_mtp_depth=None,
                mimo_mtp_sidecar_path=None,
                mimo_mtp_fail_closed=True,
            )

    def test_rejects_coerced_depth(self) -> None:
        with pytest.raises(ValidationError):
            MimoMtpRequestFields(
                mimo_mtp_fastpath=True,
                mimo_mtp_depth="2",  # type: ignore[arg-type]
                mimo_mtp_sidecar_path=None,
                mimo_mtp_fail_closed=True,
            )

    def test_rejects_coerced_sidecar_path(self) -> None:
        with pytest.raises(ValidationError):
            MimoMtpRequestFields(
                mimo_mtp_fastpath=True,
                mimo_mtp_depth=None,
                mimo_mtp_sidecar_path=123,  # type: ignore[arg-type]
                mimo_mtp_fail_closed=True,
            )

    def test_rejects_coerced_fail_closed(self) -> None:
        with pytest.raises(ValidationError):
            MimoMtpRequestFields(
                mimo_mtp_fastpath=True,
                mimo_mtp_depth=None,
                mimo_mtp_sidecar_path=None,
                mimo_mtp_fail_closed="false",  # type: ignore[arg-type]
            )

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            MimoMtpRequestFields(
                mimo_mtp_fastpath=True,
                mimo_mtp_depth=None,
                mimo_mtp_sidecar_path=None,
                mimo_mtp_fail_closed=True,
                unexpected_field=True,  # type: ignore[call-arg]
            )


# ---------------------------------------------------------------------------
# Pure function: no side effects, deterministic output
# ---------------------------------------------------------------------------


class TestPureFunctionBehavior:
    """The mapping function is pure: same input always yields same output."""

    def test_deterministic_for_enabled_input(self) -> None:
        fields = MimoMtpRequestFields(
            mimo_mtp_fastpath=True,
            mimo_mtp_depth=2,
            mimo_mtp_sidecar_path="/models/sidecar.safetensors",
            mimo_mtp_fail_closed=False,
        )
        result_a = map_request_mtp_fields_to_fastpath_params(fields)
        result_b = map_request_mtp_fields_to_fastpath_params(fields)
        assert result_a is not None
        assert result_b is not None
        assert result_a.enabled == result_b.enabled
        assert result_a.depth == result_b.depth
        assert result_a.sidecar_path == result_b.sidecar_path
        assert result_a.fail_closed == result_b.fail_closed

    def test_deterministic_for_disabled_input(self) -> None:
        fields = MimoMtpRequestFields(
            mimo_mtp_fastpath=False,
            mimo_mtp_depth=None,
            mimo_mtp_sidecar_path=None,
            mimo_mtp_fail_closed=True,
        )
        assert map_request_mtp_fields_to_fastpath_params(fields) is None
        assert map_request_mtp_fields_to_fastpath_params(fields) is None
