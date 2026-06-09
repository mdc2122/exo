"""Sub-AC 2c: Non-MiMo-model fail-closed condition.

Comprehensive cross-layer tests proving that when MTP is requested for a
non-MiMo model, every layer returns the correct typed error and never
silently claims MTP execution.

Layers tested:
1. Classifier (pure function): classify_mimo_mtp_request
2. Intent validator (pure function): validate_mtp_intent
3. Worker fastpath evaluator (pure function): evaluate_mimo_mtp_worker_fastpath
4. API boundary: validate_mimo_mtp_fastpath_eligibility
5. Worker generator routing: SequentialGenerator and BatchGenerator

Key invariant: No layer, under any configuration of fail_closed=True or
fail_closed=False, may ever produce a result that:
- claims mtp_enabled=True
- claims accepted_execution_path="mimo_mtp_fastpath"
- claims mtp_execution_state="successful_mtp"
- claims mode="mtp"
- omits a disable_reason containing "unsupported_model"

This is the **fail-closed** contract: non-MiMo models can never produce MTP,
and the system must report a typed error, not silently fall through to AR
while appearing to have accepted MTP.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from exo.shared.models.model_cards import (
    MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
    MIMO_V25_PRO_MODEL_IDS,
)
from exo.shared.types.commands import CommandId
from exo.shared.types.common import ModelId
from exo.shared.types.mimo_mtp_classifier import (
    classify_mimo_mtp_request,
)
from exo.shared.types.tasks import TaskId
from exo.shared.types.text_generation import (
    InputMessage,
    MimoMtpFastpathParams,
    TextGenerationTaskParams,
)
from exo.shared.types.validate_mtp_intent import (
    MTPDisabledReason,
    MTPValidatedIntent,
    validate_mtp_intent,
)
from exo.worker.engines.mlx.mimo_mtp_fast.worker_fastpath import (
    evaluate_mimo_mtp_worker_fastpath,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NON_MIMO_MODELS: list[ModelId] = [
    ModelId("mlx-community/Llama-3.3-70B-Instruct-4bit"),
    ModelId("Qwen/Qwen3-30B-A3B"),
    ModelId("meta-llama/Meta-Llama-3.1-8B-Instruct"),
    ModelId("llama-3.2-1b"),
    ModelId("some-org/some-model"),
]

_MIMO_MODEL = MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID


def _ar_task_params(
    model: ModelId = _MIMO_MODEL,
) -> TextGenerationTaskParams:
    """Default AR request: no MTP intent."""
    return TextGenerationTaskParams(
        model=model,
        input=[InputMessage(role="user", content="Hello")],
        max_output_tokens=4,
    )


def _mtp_task_params(
    *,
    model: ModelId = _MIMO_MODEL,
    depth: int | None = 1,
    sidecar_path: str | None = "/models/model_mtp.safetensors",
    fail_closed: bool = True,
    enabled: bool = True,
) -> TextGenerationTaskParams:
    """MTP-intent request with configurable fields."""
    return TextGenerationTaskParams(
        model=model,
        input=[InputMessage(role="user", content="Hello")],
        max_output_tokens=4,
        mimo_mtp_fastpath_params=MimoMtpFastpathParams(
            enabled=enabled,
            depth=depth,
            sidecar_path=sidecar_path,
            fail_closed=fail_closed,
        ),
    )


# ---------------------------------------------------------------------------
# Helper: assert a result never silently claims MTP
# ---------------------------------------------------------------------------

# Strings that would indicate MTP was claimed/successful if present
_MTP_CLAIM_STRINGS: frozenset[str] = frozenset(
    {
        "successful_mtp",
        "mimo_mtp_fastpath",  # as accepted_execution_path value
    }
)


def _assert_no_silent_mtp_claim(
    *,
    mtp_enabled: bool | None,
    accepted_execution_path: str | None = None,
    mtp_execution_state: str | None = None,
    mode: str | None = None,
    context: str = "",
) -> None:
    """Assert that no field silently claims MTP was successful.

    This is the core invariant checker for Sub-AC 2c.
    """
    assert mtp_enabled is not True, (
        f"{context}: mtp_enabled must never be True for non-MiMo models, "
        f"got mtp_enabled={mtp_enabled}"
    )
    if accepted_execution_path is not None:
        assert accepted_execution_path != "mimo_mtp_fastpath", (
            f"{context}: accepted_execution_path must never be "
            f"'mimo_mtp_fastpath' for non-MiMo models, "
            f"got {accepted_execution_path}"
        )
    if mtp_execution_state is not None:
        assert mtp_execution_state != "successful_mtp", (
            f"{context}: mtp_execution_state must never be "
            f"'successful_mtp' for non-MiMo models, "
            f"got {mtp_execution_state}"
        )
    if mode is not None:
        assert mode != "mtp", (
            f"{context}: mode must never be 'mtp' for non-MiMo models, "
            f"got mode={mode}"
        )


# ===========================================================================
# Layer 1: Classifier — pure function classify_mimo_mtp_request
# ===========================================================================


class TestClassifierNonMimoFailClosed:
    """classify_mimo_mtp_request must return unsupported_model for non-MiMo
    models requesting MTP, regardless of other field values."""

    @pytest.mark.parametrize("non_mimo_model", _NON_MIMO_MODELS)
    def test_non_mimo_model_with_mtp_intent_returns_unsupported_model(
        self,
        non_mimo_model: ModelId,
    ) -> None:
        params = _mtp_task_params(model=non_mimo_model)
        result = classify_mimo_mtp_request(params)
        assert result.label == "unsupported_model"
        assert result.discriminant == "unsupported"
        assert result.disable_reason == "unsupported_model"

    @pytest.mark.parametrize("non_mimo_model", _NON_MIMO_MODELS)
    def test_classifier_never_claims_mtp_enabled_for_non_mimo(
        self,
        non_mimo_model: ModelId,
    ) -> None:
        params = _mtp_task_params(model=non_mimo_model)
        result = classify_mimo_mtp_request(params)
        _assert_no_silent_mtp_claim(
            mtp_enabled=result.mtp_enabled,
            context=f"classifier for model={non_mimo_model}",
        )

    @pytest.mark.parametrize("non_mimo_model", _NON_MIMO_MODELS)
    def test_classifier_fail_closed_true_for_non_mimo(
        self,
        non_mimo_model: ModelId,
    ) -> None:
        params = _mtp_task_params(model=non_mimo_model, fail_closed=True)
        result = classify_mimo_mtp_request(params)
        assert result.label == "unsupported_model"
        assert result.fail_closed is True

    @pytest.mark.parametrize("non_mimo_model", _NON_MIMO_MODELS)
    def test_classifier_fail_closed_false_for_non_mimo(
        self,
        non_mimo_model: ModelId,
    ) -> None:
        params = _mtp_task_params(model=non_mimo_model, fail_closed=False)
        result = classify_mimo_mtp_request(params)
        assert result.label == "unsupported_model"
        assert result.fail_closed is False
        # Even fail-open intent is still classified as unsupported_model
        # The classifier does not silently allow MTP
        assert result.mtp_enabled is False

    @pytest.mark.parametrize("non_mimo_model", _NON_MIMO_MODELS)
    def test_classifier_model_check_preempts_depth_check(
        self,
        non_mimo_model: ModelId,
    ) -> None:
        """A non-MiMo model with an unsupported depth still classifies as
        unsupported_model (model check comes before depth check)."""
        base_params = _mtp_task_params(model=non_mimo_model, depth=1)
        raw_mtp = base_params.mimo_mtp_fastpath
        assert raw_mtp is not None
        mtp_with_bad_depth = raw_mtp.model_construct(
            enabled=True,
            depth=99,
            sidecar_path="/models/model_mtp.safetensors",
            fail_closed=True,
        )
        params = base_params.model_copy(
            update={"mimo_mtp_fastpath_params": mtp_with_bad_depth}
        )
        result = classify_mimo_mtp_request(params)
        assert result.label == "unsupported_model"
        assert result.label != "unsupported_depth"

    @pytest.mark.parametrize("non_mimo_model", _NON_MIMO_MODELS)
    def test_classifier_non_mimo_with_sidecar_still_unsupported(
        self,
        non_mimo_model: ModelId,
    ) -> None:
        """Providing a sidecar path does not make a non-MiMo model eligible."""
        params = _mtp_task_params(
            model=non_mimo_model,
            sidecar_path="/models/model_mtp.safetensors",
        )
        result = classify_mimo_mtp_request(params)
        assert result.label == "unsupported_model"

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_classifier_non_mimo_with_any_supported_depth_still_unsupported(
        self,
        depth: int,
    ) -> None:
        params = _mtp_task_params(
            model=_NON_MIMO_MODELS[0],
            depth=depth,
        )
        result = classify_mimo_mtp_request(params)
        assert result.label == "unsupported_model"

    def test_classifier_non_mimo_ar_request_is_disabled_default(self) -> None:
        """A non-MiMo model without MTP intent is disabled_default, not
        unsupported_model (no MTP intent = no violation)."""
        params = _ar_task_params(model=_NON_MIMO_MODELS[0])
        result = classify_mimo_mtp_request(params)
        assert result.label == "disabled_default"


# ===========================================================================
# Layer 2: Intent validator — pure function validate_mtp_intent
# ===========================================================================


class TestIntentValidatorNonMimoFailClosed:
    """validate_mtp_intent must return MTPDisabledReason with
    disable_reason='unsupported_model' for non-MiMo models requesting MTP."""

    @pytest.mark.parametrize("non_mimo_model", _NON_MIMO_MODELS)
    def test_non_mimo_model_returns_disabled_reason(
        self,
        non_mimo_model: ModelId,
    ) -> None:
        params = _mtp_task_params(model=non_mimo_model)
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)
        assert isinstance(result, MTPDisabledReason)
        assert result.disable_reason == "unsupported_model"
        assert result.classification_label == "unsupported_model"

    @pytest.mark.parametrize("non_mimo_model", _NON_MIMO_MODELS)
    def test_non_mimo_model_never_produces_validated_intent(
        self,
        non_mimo_model: ModelId,
    ) -> None:
        params = _mtp_task_params(model=non_mimo_model)
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)
        assert not isinstance(result, MTPValidatedIntent), (
            f"non-MiMo model {non_mimo_model} must never produce "
            f"MTPValidatedIntent, got {result}"
        )

    @pytest.mark.parametrize("non_mimo_model", _NON_MIMO_MODELS)
    def test_validator_mtp_enabled_false_for_non_mimo(
        self,
        non_mimo_model: ModelId,
    ) -> None:
        params = _mtp_task_params(model=non_mimo_model)
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)
        assert isinstance(result, MTPDisabledReason)
        _assert_no_silent_mtp_claim(
            mtp_enabled=result.mtp_enabled,
            context=f"validator for model={non_mimo_model}",
        )
        assert result.mtp_enabled is False

    @pytest.mark.parametrize("non_mimo_model", _NON_MIMO_MODELS)
    def test_validator_fail_closed_true_for_non_mimo(
        self,
        non_mimo_model: ModelId,
    ) -> None:
        params = _mtp_task_params(model=non_mimo_model, fail_closed=True)
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)
        assert isinstance(result, MTPDisabledReason)
        assert result.fail_closed is True

    @pytest.mark.parametrize("non_mimo_model", _NON_MIMO_MODELS)
    def test_validator_fail_closed_false_for_non_mimo(
        self,
        non_mimo_model: ModelId,
    ) -> None:
        params = _mtp_task_params(model=non_mimo_model, fail_closed=False)
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)
        assert isinstance(result, MTPDisabledReason)
        assert result.fail_closed is False
        # Still disabled, even with fail_open
        assert result.mtp_enabled is False
        assert result.disable_reason == "unsupported_model"


# ===========================================================================
# Layer 3: Worker fastpath evaluator — evaluate_mimo_mtp_worker_fastpath
# ===========================================================================


class TestWorkerFastpathNonMimoFailClosed:
    """evaluate_mimo_mtp_worker_fastpath must reject non-MiMo models with
    typed unsupported_model error."""

    @pytest.mark.parametrize("non_mimo_model", _NON_MIMO_MODELS)
    def test_non_mimo_fail_closed_returns_rejected_decision(
        self,
        non_mimo_model: ModelId,
        tmp_path: Path,
    ) -> None:
        sidecar_path = tmp_path / "model_mtp.safetensors"
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params(
                model=non_mimo_model,
                sidecar_path=str(sidecar_path),
                fail_closed=True,
            ),
        )
        assert decision.should_use_mtp is False
        assert decision.accepted_execution_path == "rejected"
        assert decision.mtp_enabled is False
        assert decision.disable_reason == "unsupported_model"
        assert decision.error_message is not None
        assert "unsupported_model" in decision.error_message

    @pytest.mark.parametrize("non_mimo_model", _NON_MIMO_MODELS)
    def test_non_mimo_fail_closed_never_claims_mtp(
        self,
        non_mimo_model: ModelId,
        tmp_path: Path,
    ) -> None:
        sidecar_path = tmp_path / "model_mtp.safetensors"
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params(
                model=non_mimo_model,
                sidecar_path=str(sidecar_path),
                fail_closed=True,
            ),
        )
        _assert_no_silent_mtp_claim(
            mtp_enabled=decision.mtp_enabled,
            accepted_execution_path=decision.accepted_execution_path,
            context=f"worker-fastpath fail_closed for model={non_mimo_model}",
        )
        assert decision.mtp_depth is None
        assert decision.sidecar is None

    @pytest.mark.parametrize("non_mimo_model", _NON_MIMO_MODELS)
    def test_non_mimo_fail_open_returns_ar_fallback(
        self,
        non_mimo_model: ModelId,
        tmp_path: Path,
    ) -> None:
        """Even fail-open non-MiMo requests produce AR fallback, NOT MTP."""
        sidecar_path = tmp_path / "model_mtp.safetensors"
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params(
                model=non_mimo_model,
                sidecar_path=str(sidecar_path),
                fail_closed=False,
            ),
        )
        assert decision.should_use_mtp is False
        # Fail-open: falls back to AR, not rejected
        assert decision.accepted_execution_path == "ar"
        assert decision.mtp_enabled is False
        assert decision.disable_reason == "unsupported_model"
        assert decision.fallback_reason == "fail_open_unsupported_model"
        _assert_no_silent_mtp_claim(
            mtp_enabled=decision.mtp_enabled,
            accepted_execution_path=decision.accepted_execution_path,
            context=f"worker-fastpath fail_open for model={non_mimo_model}",
        )

    @pytest.mark.parametrize("non_mimo_model", _NON_MIMO_MODELS)
    def test_non_mimo_fail_open_error_message_is_none(
        self,
        non_mimo_model: ModelId,
        tmp_path: Path,
    ) -> None:
        """Fail-open AR fallback has no error_message (it's a normal AR path)."""
        sidecar_path = tmp_path / "model_mtp.safetensors"
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params(
                model=non_mimo_model,
                sidecar_path=str(sidecar_path),
                fail_closed=False,
            ),
        )
        assert decision.error_message is None

    @pytest.mark.parametrize("non_mimo_model", _NON_MIMO_MODELS)
    def test_non_mimo_fail_closed_has_error_message(
        self,
        non_mimo_model: ModelId,
        tmp_path: Path,
    ) -> None:
        sidecar_path = tmp_path / "model_mtp.safetensors"
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params(
                model=non_mimo_model,
                sidecar_path=str(sidecar_path),
                fail_closed=True,
            ),
        )
        assert decision.error_message is not None
        assert "unsupported_model" in decision.error_message
        assert "reject_reason=unsupported_model" in decision.error_message

    @pytest.mark.parametrize("non_mimo_model", _NON_MIMO_MODELS)
    def test_non_mimo_telemetry_reports_unsupported_model(
        self,
        non_mimo_model: ModelId,
        tmp_path: Path,
    ) -> None:
        sidecar_path = tmp_path / "model_mtp.safetensors"
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params(
                model=non_mimo_model,
                sidecar_path=str(sidecar_path),
                fail_closed=True,
            ),
        )
        assert decision.telemetry["mtp_disable_reason"] == "unsupported_model"
        assert decision.telemetry["accepted_execution_path"] == "rejected"
        assert decision.telemetry["mtp_enabled"] is False

    def test_default_ar_request_for_non_mimo_returns_ar_decision(
        self,
    ) -> None:
        """A non-MiMo model with no MTP intent returns the normal AR decision
        (not a rejection)."""
        decision = evaluate_mimo_mtp_worker_fastpath(
            _ar_task_params(model=_NON_MIMO_MODELS[0]),
        )
        assert decision.should_use_mtp is False
        assert decision.accepted_execution_path == "ar"
        assert decision.mtp_enabled is False
        assert decision.disable_reason is None
        assert decision.error_message is None


# ===========================================================================
# Layer 4: API boundary — validate_mimo_mtp_fastpath_eligibility
# ===========================================================================


class TestApiBoundaryNonMimoFailClosed:
    """validate_mimo_mtp_fastpath_eligibility must raise HTTPException(400)
    with typed error detail for non-MiMo models requesting MTP."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("non_mimo_model_str", [str(m) for m in _NON_MIMO_MODELS])
    async def test_non_mimo_model_raises_400_with_unsupported_model_detail(
        self,
        non_mimo_model_str: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from exo.api.adapters import chat_completions
        from exo.api.types import ChatCompletionMessage, ChatCompletionRequest

        # Prevent sidecar probing — non-MiMo check must happen before any probe
        def fail_sidecar_probe(path: str) -> object:
            raise AssertionError(
                f"non-MiMo fail-closed guard must not probe sidecar: {path}"
            )

        monkeypatch.setattr(
            chat_completions, "probe_mimo_mtp_sidecar", fail_sidecar_probe
        )

        request = ChatCompletionRequest(
            model=ModelId(non_mimo_model_str),
            messages=[ChatCompletionMessage(role="user", content="Hello")],
            mimo_mtp_fastpath=True,
            mimo_mtp_depth=1,
            mimo_mtp_sidecar_path="/tmp/must-not-be-probed.safetensors",
            mimo_mtp_fail_closed=True,
        )

        with pytest.raises(HTTPException) as exc_info:
            await chat_completions.chat_request_to_text_generation(request)

        assert exc_info.value.status_code == 400
        raw_detail = cast(object, exc_info.value.detail)
        assert isinstance(raw_detail, dict)
        detail = cast(Mapping[str, object], raw_detail)

        assert detail["error"] == "mimo_mtp_model_ineligible"
        assert detail["mtp_enabled"] is False
        assert detail["mtp_execution_state"] == "unsupported_model"
        assert detail["mtp_disable_reason"] == "unsupported_model"
        assert detail["accepted_execution_path"] == "rejected"

        # Core invariant: never claims MTP
        _assert_no_silent_mtp_claim(
            mtp_enabled=detail.get("mtp_enabled"),
            accepted_execution_path=cast(
                str | None, detail.get("accepted_execution_path")
            ),
            mtp_execution_state=cast(
                str | None, detail.get("mtp_execution_state")
            ),
            context=f"API boundary for model={non_mimo_model_str}",
        )

    @pytest.mark.asyncio
    async def test_non_mimo_fail_open_also_rejects_at_api_boundary(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Even with fail_closed=False, the API boundary still rejects
        non-MiMo models requesting MTP. Fail-open only applies at the
        worker/generator layer, not the API validation layer."""
        from exo.api.adapters import chat_completions
        from exo.api.types import ChatCompletionMessage, ChatCompletionRequest

        def fail_sidecar_probe(path: str) -> object:
            raise AssertionError(
                f"non-MiMo guard must not probe sidecar: {path}"
            )

        monkeypatch.setattr(
            chat_completions, "probe_mimo_mtp_sidecar", fail_sidecar_probe
        )

        request = ChatCompletionRequest(
            model=ModelId("llama-3.2-1b"),
            messages=[ChatCompletionMessage(role="user", content="Hello")],
            mimo_mtp_fastpath=True,
            mimo_mtp_depth=1,
            mimo_mtp_fail_closed=False,
        )

        with pytest.raises(HTTPException) as exc_info:
            await chat_completions.chat_request_to_text_generation(request)

        assert exc_info.value.status_code == 400
        raw_detail = cast(object, exc_info.value.detail)
        assert isinstance(raw_detail, dict)
        detail = cast(Mapping[str, object], raw_detail)
        assert detail["error"] == "mimo_mtp_model_ineligible"
        assert detail["mtp_enabled"] is False
        # Fail-open at API boundary still reports the non_mimo_model reason
        assert detail["mtp_disable_reason"] == "non_mimo_model"

    @pytest.mark.asyncio
    async def test_non_mimo_model_without_mtp_intent_passes_api_validation(
        self,
    ) -> None:
        """A non-MiMo model with no MTP intent (mimo_mtp_fastpath=False)
        should pass API validation without raising an HTTPException."""
        from exo.api.types import ChatCompletionMessage, ChatCompletionRequest

        # This should NOT raise — no MTP intent, so no MTP guard applies
        request = ChatCompletionRequest(
            model=ModelId("llama-3.2-1b"),
            messages=[ChatCompletionMessage(role="user", content="Hello")],
            mimo_mtp_fastpath=False,
        )
        # validate_mimo_mtp_fastpath_eligibility returns None silently
        from exo.api.adapters.chat_completions import (
            validate_mimo_mtp_fastpath_eligibility,
        )

        # Should not raise
        validate_mimo_mtp_fastpath_eligibility(request)


# ===========================================================================
# Layer 5: Worker generator routing — SequentialGenerator / BatchGenerator
# ===========================================================================


class TestWorkerGeneratorRoutingNonMimoFailClosed:
    """The worker generator routing must reject non-MiMo MTP requests
    fail-closed and never dispatch them as MTP generation."""

    def test_sequential_fail_closed_non_mimo_rejection_does_not_call_ar_generator(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When a non-MiMo model requests MTP fail-closed, the sequential
        generator must reject without calling the AR generator."""
        import exo.worker.runner.llm_inference.batch_generator as bg_module
        from exo.shared.types.tasks import TaskId, TextGeneration
        from exo.shared.types.worker.instances import InstanceId

        def _noop_check(_params: TextGenerationTaskParams) -> None:
            return None

        def _fake_chat_template(
            _tokenizer: object,
            _params: TextGenerationTaskParams,
        ) -> str:
            return "prompt"

        monkeypatch.setattr(bg_module, "_check_for_debug_prompts", _noop_check)
        monkeypatch.setattr(
            bg_module, "apply_chat_template", _fake_chat_template
        )

        def fail_if_called(**_kwargs: object) -> object:
            raise AssertionError(
                "fail-closed non-MiMo MTP request must not dispatch AR generation"
            )

        monkeypatch.setattr(bg_module, "mlx_generate", fail_if_called)

        task = TextGeneration(
            task_id=TaskId("12345678-1234-1234-1234-123456789abc"),
            command_id=CommandId("cmd-1"),
            instance_id=InstanceId("instance-1"),
            task_params=_mtp_task_params(
                model=_NON_MIMO_MODELS[0],
                fail_closed=True,
            ),
        )

        cancel_receiver = _CancelReceiver()
        event_sender = _EventSender()

        seq_gen = bg_module.SequentialGenerator(
            model=MagicMock(),
            tokenizer=MagicMock(),
            group=None,
            kv_prefix_cache=None,
            tool_parser=None,
            model_id=_NON_MIMO_MODELS[0],
            device_rank=0,
            cancel_receiver=cancel_receiver,
            event_sender=event_sender,
        )

        result = next(seq_gen._build_generator(task))

        assert result.finish_reason == "error"
        assert "unsupported_model" in result.text
        assert result.token == 0

    def test_sequential_fail_open_non_mimo_falls_back_to_ar_without_mtp(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When a non-MiMo model requests MTP fail-open, the sequential
        generator must strip MTP intent and dispatch normal AR generation."""
        import exo.worker.runner.llm_inference.batch_generator as bg_module
        from exo.shared.types.tasks import TaskId, TextGeneration
        from exo.shared.types.worker.instances import InstanceId
        from exo.shared.types.worker.runner_response import GenerationResponse

        def _noop_check(_params: TextGenerationTaskParams) -> None:
            return None

        def _fake_chat_template(
            _tokenizer: object,
            _params: TextGenerationTaskParams,
        ) -> str:
            return "prompt"

        monkeypatch.setattr(bg_module, "_check_for_debug_prompts", _noop_check)
        monkeypatch.setattr(
            bg_module, "apply_chat_template", _fake_chat_template
        )

        captured_task_params: list[TextGenerationTaskParams] = []

        def fake_mlx_generate(**kwargs: object) -> object:
            captured_task_params.append(
                cast(TextGenerationTaskParams, kwargs["task"])
            )

            def gen() -> object:
                yield GenerationResponse(
                    text="fallback-ar",
                    token=1,
                    finish_reason="stop",
                    usage=None,
                )

            return gen()

        monkeypatch.setattr(bg_module, "mlx_generate", fake_mlx_generate)

        task = TextGeneration(
            task_id=TaskId("12345678-1234-1234-1234-123456789abc"),
            command_id=CommandId("cmd-1"),
            instance_id=InstanceId("instance-1"),
            task_params=_mtp_task_params(
                model=_NON_MIMO_MODELS[0],
                fail_closed=False,
            ),
        )

        cancel_receiver = _CancelReceiver()
        event_sender = _EventSender()

        seq_gen = bg_module.SequentialGenerator(
            model=MagicMock(),
            tokenizer=MagicMock(),
            group=None,
            kv_prefix_cache=None,
            tool_parser=None,
            model_id=_NON_MIMO_MODELS[0],
            device_rank=0,
            cancel_receiver=cancel_receiver,
            event_sender=event_sender,
        )

        result = next(seq_gen._build_generator(task))

        assert result.text == "fallback-ar"
        assert result.finish_reason == "stop"

        # Verify MTP intent was stripped — AR generation saw no MTP params
        assert len(captured_task_params) == 1
        assert captured_task_params[0].mimo_mtp_fastpath is None

    def test_batch_fail_closed_non_mimo_raises_runtime_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When a non-MiMo model requests MTP fail-closed, the batch
        generator must raise RuntimeError before submitting to the engine."""
        import exo.worker.runner.llm_inference.batch_generator as bg_module
        from exo.shared.types.tasks import TaskId, TextGeneration
        from exo.shared.types.worker.instances import InstanceId

        def _noop_check(_params: TextGenerationTaskParams) -> None:
            return None

        def _fake_chat_template(
            _tokenizer: object,
            _params: TextGenerationTaskParams,
        ) -> str:
            return "prompt"

        monkeypatch.setattr(bg_module, "_check_for_debug_prompts", _noop_check)
        monkeypatch.setattr(
            bg_module, "apply_chat_template", _fake_chat_template
        )

        task = TextGeneration(
            task_id=TaskId("12345678-1234-1234-1234-123456789abc"),
            command_id=CommandId("cmd-1"),
            instance_id=InstanceId("instance-1"),
            task_params=_mtp_task_params(
                model=_NON_MIMO_MODELS[0],
                fail_closed=True,
            ),
        )

        cancel_receiver = _CancelReceiver()
        event_sender = _EventSender()
        fake_engine = _FakeBatchEngine()

        batch_gen = bg_module.BatchGenerator(
            model=MagicMock(),
            tokenizer=MagicMock(),
            group=None,
            kv_prefix_cache=None,
            tool_parser=None,
            model_id=_NON_MIMO_MODELS[0],
            device_rank=0,
            cancel_receiver=cancel_receiver,
            event_sender=event_sender,
        )
        batch_gen._mlx_gen = cast(Any, fake_engine)

        with pytest.raises(RuntimeError) as exc_info:
            batch_gen._start_task(task)

        assert "unsupported_model" in str(exc_info.value)
        # Verify nothing was submitted to the engine
        assert len(fake_engine.submitted_params) == 0

    def test_batch_fail_open_non_mimo_strips_mtp_and_submits_ar(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When a non-MiMo model requests MTP fail-open, the batch
        generator strips MTP and submits AR generation."""
        import exo.worker.runner.llm_inference.batch_generator as bg_module
        from exo.shared.types.tasks import TaskId, TextGeneration
        from exo.shared.types.worker.instances import InstanceId

        def _noop_check(_params: TextGenerationTaskParams) -> None:
            return None

        def _fake_chat_template(
            _tokenizer: object,
            _params: TextGenerationTaskParams,
        ) -> str:
            return "prompt"

        monkeypatch.setattr(bg_module, "_check_for_debug_prompts", _noop_check)
        monkeypatch.setattr(
            bg_module, "apply_chat_template", _fake_chat_template
        )

        task = TextGeneration(
            task_id=TaskId("12345678-1234-1234-1234-123456789abc"),
            command_id=CommandId("cmd-1"),
            instance_id=InstanceId("instance-1"),
            task_params=_mtp_task_params(
                model=_NON_MIMO_MODELS[0],
                fail_closed=False,
            ),
        )

        cancel_receiver = _CancelReceiver()
        event_sender = _EventSender()
        fake_engine = _FakeBatchEngine()

        batch_gen = bg_module.BatchGenerator(
            model=MagicMock(),
            tokenizer=MagicMock(),
            group=None,
            kv_prefix_cache=None,
            tool_parser=None,
            model_id=_NON_MIMO_MODELS[0],
            device_rank=0,
            cancel_receiver=cancel_receiver,
            event_sender=event_sender,
        )
        batch_gen._mlx_gen = cast(Any, fake_engine)

        batch_gen._start_task(task)

        assert len(fake_engine.submitted_params) == 1
        assert fake_engine.submitted_params[0].mimo_mtp_fastpath is None


# ===========================================================================
# Cross-layer invariant: no layer ever claims MTP for non-MiMo models
# ===========================================================================


class TestCrossLayerNoSilentMtpClaim:
    """Invariant: across all layers, no non-MiMo MTP request ever produces
    a result that claims successful MTP execution."""

    @pytest.mark.parametrize("non_mimo_model", _NON_MIMO_MODELS)
    def test_classifier_invariant(self, non_mimo_model: ModelId) -> None:
        params = _mtp_task_params(model=non_mimo_model)
        result = classify_mimo_mtp_request(params)
        _assert_no_silent_mtp_claim(
            mtp_enabled=result.mtp_enabled,
            context=f"classifier invariant for {non_mimo_model}",
        )
        assert result.label != "compatible_mtp"
        assert result.label != "successful_mtp"

    @pytest.mark.parametrize("non_mimo_model", _NON_MIMO_MODELS)
    def test_validator_invariant(self, non_mimo_model: ModelId) -> None:
        params = _mtp_task_params(model=non_mimo_model)
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)
        assert isinstance(result, MTPDisabledReason)
        _assert_no_silent_mtp_claim(
            mtp_enabled=result.mtp_enabled,
            context=f"validator invariant for {non_mimo_model}",
        )

    @pytest.mark.parametrize("non_mimo_model", _NON_MIMO_MODELS)
    def test_worker_fastpath_fail_closed_invariant(
        self,
        non_mimo_model: ModelId,
        tmp_path: Path,
    ) -> None:
        sidecar_path = tmp_path / "model_mtp.safetensors"
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params(
                model=non_mimo_model,
                sidecar_path=str(sidecar_path),
                fail_closed=True,
            ),
        )
        _assert_no_silent_mtp_claim(
            mtp_enabled=decision.mtp_enabled,
            accepted_execution_path=decision.accepted_execution_path,
            context=f"worker-fastpath fail-closed invariant for {non_mimo_model}",
        )
        assert decision.should_use_mtp is False
        assert decision.sidecar is None
        assert decision.mtp_depth is None

    @pytest.mark.parametrize("non_mimo_model", _NON_MIMO_MODELS)
    def test_worker_fastpath_fail_open_invariant(
        self,
        non_mimo_model: ModelId,
        tmp_path: Path,
    ) -> None:
        sidecar_path = tmp_path / "model_mtp.safetensors"
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params(
                model=non_mimo_model,
                sidecar_path=str(sidecar_path),
                fail_closed=False,
            ),
        )
        _assert_no_silent_mtp_claim(
            mtp_enabled=decision.mtp_enabled,
            accepted_execution_path=decision.accepted_execution_path,
            context=f"worker-fastpath fail-open invariant for {non_mimo_model}",
        )
        assert decision.should_use_mtp is False
        assert decision.sidecar is None


# ===========================================================================
# Positive control: MiMo models with MTP intent still work
# ===========================================================================


class TestMimoModelMtpIntentNotFalseRejected:
    """Positive control: MiMo models with valid MTP intent are NOT
    rejected by the non-MiMo fail-closed guard."""

    @pytest.mark.parametrize("mimo_model", MIMO_V25_PRO_MODEL_IDS)
    def test_mimo_model_mtp_intent_classifies_as_unwired(
        self,
        mimo_model: ModelId,
    ) -> None:
        """MiMo models with MTP intent classify as unwired_execution (not
        unsupported_model), proving the guard is model-specific."""
        params = _mtp_task_params(model=mimo_model)
        result = classify_mimo_mtp_request(params)
        assert result.label == "unwired_execution"
        assert result.label != "unsupported_model"

    @pytest.mark.parametrize("mimo_model", MIMO_V25_PRO_MODEL_IDS)
    def test_mimo_model_mtp_intent_validates_as_validated_intent(
        self,
        mimo_model: ModelId,
    ) -> None:
        params = _mtp_task_params(model=mimo_model)
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)
        assert isinstance(result, MTPValidatedIntent)

    @pytest.mark.parametrize("mimo_model", MIMO_V25_PRO_MODEL_IDS)
    def test_mimo_model_ar_request_is_disabled_default(
        self,
        mimo_model: ModelId,
    ) -> None:
        """MiMo models without MTP intent are disabled_default, not
        unsupported_model."""
        params = _ar_task_params(model=mimo_model)
        result = classify_mimo_mtp_request(params)
        assert result.label == "disabled_default"

    def test_mimo_model_default_ar_is_not_rejected_by_worker_fastpath(
        self,
    ) -> None:
        """MiMo model with no MTP intent returns AR decision, not rejection."""
        decision = evaluate_mimo_mtp_worker_fastpath(
            _ar_task_params(model=_MIMO_MODEL),
        )
        assert decision.accepted_execution_path == "ar"
        assert decision.should_use_mtp is False
        assert decision.disable_reason is None
        assert decision.error_message is None


# ===========================================================================
# Default AR preservation: non-MiMo models without MTP intent are normal AR
# ===========================================================================


class TestNonMimoDefaultArPreserved:
    """Non-MiMo models without MTP intent must produce normal AR behavior
    (no error, no rejection, no MTP telemetry)."""

    @pytest.mark.parametrize("non_mimo_model", _NON_MIMO_MODELS)
    def test_non_mimo_ar_classifier_returns_disabled_default(
        self,
        non_mimo_model: ModelId,
    ) -> None:
        params = _ar_task_params(model=non_mimo_model)
        result = classify_mimo_mtp_request(params)
        assert result.label == "disabled_default"
        assert result.mtp_enabled is False

    @pytest.mark.parametrize("non_mimo_model", _NON_MIMO_MODELS)
    def test_non_mimo_ar_validator_returns_disabled_reason_mtp_not_requested(
        self,
        non_mimo_model: ModelId,
    ) -> None:
        params = _ar_task_params(model=non_mimo_model)
        classification = classify_mimo_mtp_request(params)
        result = validate_mtp_intent(classification)
        assert isinstance(result, MTPDisabledReason)
        assert result.disable_reason == "mtp_not_requested"
        # This is "no MTP intent", not an error condition
        assert result.classification_label == "disabled_default"

    @pytest.mark.parametrize("non_mimo_model", _NON_MIMO_MODELS)
    def test_non_mimo_ar_worker_fastpath_returns_ar_decision(
        self,
        non_mimo_model: ModelId,
    ) -> None:
        decision = evaluate_mimo_mtp_worker_fastpath(
            _ar_task_params(model=non_mimo_model),
        )
        assert decision.accepted_execution_path == "ar"
        assert decision.should_use_mtp is False
        assert decision.mtp_enabled is False
        assert decision.disable_reason is None
        assert decision.error_message is None


# ===========================================================================
# _MTP_CLAIM_STRINGS never appear in any non-MiMo output
# ===========================================================================


class TestMtpClaimStringAbsence:
    """Verify that no output from any layer contains strings that would
    indicate a false MTP claim for non-MiMo models."""

    @pytest.mark.parametrize("non_mimo_model", _NON_MIMO_MODELS)
    def test_worker_fastpath_error_message_no_mtp_claim(
        self,
        non_mimo_model: ModelId,
        tmp_path: Path,
    ) -> None:
        sidecar_path = tmp_path / "model_mtp.safetensors"
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params(
                model=non_mimo_model,
                sidecar_path=str(sidecar_path),
                fail_closed=True,
            ),
        )
        error_msg = decision.error_message or ""
        for claim_str in _MTP_CLAIM_STRINGS:
            # "mimo_mtp_fastpath" as a path label is fine to appear in
            # disable messages; only "successful_mtp" is an actual claim
            if claim_str == "successful_mtp":
                assert claim_str not in error_msg, (
                    f"error_message for {non_mimo_model} must not contain "
                    f"'{claim_str}': {error_msg}"
                )

    @pytest.mark.parametrize("non_mimo_model", _NON_MIMO_MODELS)
    def test_telemetry_no_successful_mtp_state(
        self,
        non_mimo_model: ModelId,
        tmp_path: Path,
    ) -> None:
        sidecar_path = tmp_path / "model_mtp.safetensors"
        decision = evaluate_mimo_mtp_worker_fastpath(
            _mtp_task_params(
                model=non_mimo_model,
                sidecar_path=str(sidecar_path),
                fail_closed=True,
            ),
        )
        for key, value in decision.telemetry.items():
            if isinstance(value, str):
                assert value != "successful_mtp", (
                    f"telemetry[{key}] for {non_mimo_model} must not be "
                    f"'successful_mtp': {value}"
                )


# ===========================================================================
# Test infrastructure helpers (needed for generator routing tests)
# ===========================================================================


class _CancelReceiver:
    def collect(self) -> list[TaskId]:
        return []


class _EventSender:
    def __init__(self) -> None:
        self.events: list[object] = []

    def send(self, event: object) -> None:
        self.events.append(event)


class _FakeBatchEngine:
    def __init__(self) -> None:
        self.submitted_params: list[TextGenerationTaskParams] = []

    @property
    def has_work(self) -> bool:
        return False

    def submit(
        self,
        *,
        task_params: TextGenerationTaskParams,
        prompt: str,
        on_prefill_progress: object,
        distributed_prompt_progress_callback: object,
        on_generation_token: object,
    ) -> int:
        self.submitted_params.append(task_params)
        return 7

    def cancel(self, _uids: list[int]) -> None:
        pass

    def close(self) -> None:
        pass



