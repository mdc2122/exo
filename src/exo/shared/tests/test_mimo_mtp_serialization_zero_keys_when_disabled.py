"""Sub-AC 5.4.1: Unit test that when MTP is not explicitly enabled,
the serialization output dict/bytes contain zero MTP-specific keys
(e.g., no 'mtp_tokens', 'mtp_probabilities', 'mtp_state' keys);
verify exact key-set equality against a known AR-only baseline.
"""

import json
from typing import Any, cast

from pydantic import TypeAdapter

from exo.api.types import ChatCompletionRequest
from exo.shared.models.model_cards import MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID
from exo.shared.types.commands import Command as CommandUnion
from exo.shared.types.commands import ForwarderCommand
from exo.shared.types.commands import TextGeneration as TextGenCommand
from exo.shared.types.common import CommandId, SystemId
from exo.shared.types.tasks import Task as TaskUnion
from exo.shared.types.tasks import (
    TaskId,
    TaskStatus,
)
from exo.shared.types.tasks import (
    TextGeneration as TextGenTask,
)
from exo.shared.types.text_generation import (
    InputMessage,
    MimoMtpFastpathParams,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId

COMMAND_ADAPTER: TypeAdapter[CommandUnion] = TypeAdapter(CommandUnion)
TASK_ADAPTER: TypeAdapter[TaskUnion] = TypeAdapter(TaskUnion)


# ---------------------------------------------------------------------------
# Canonical AR-only baseline key sets — the expected keys when MTP is absent
# ---------------------------------------------------------------------------

AR_ONLY_TASK_PARAMS_DICT_KEYS: frozenset[str] = frozenset({
    "bench",
    "chat_template_messages",
    "enable_thinking",
    "image_count",
    "image_hashes",
    "images",
    "input",
    "instructions",
    "logprobs",
    "max_output_tokens",
    "min_p",
    "model",
    "reasoning_effort",
    "repetition_context_size",
    "repetition_penalty",
    "seed",
    "stop",
    "stream",
    "temperature",
    "tools",
    "top_k",
    "top_logprobs",
    "top_p",
    "total_input_chunks",
    "video_sources",
    "video_urls",
    "videos",
})

AR_ONLY_CHAT_REQUEST_DICT_KEYS: frozenset[str] = frozenset({
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "max_tokens",
    "messages",
    "min_p",
    "model",
    "n",
    "parallel_tool_calls",
    "presence_penalty",
    "reasoning_effort",
    "enable_thinking",
    "repetition_penalty",
    "repetition_context_size",
    "response_format",
    "seed",
    "stop",
    "stream",
    "stream_options",
    "temperature",
    "tool_choice",
    "tools",
    "top_k",
    "top_logprobs",
    "top_p",
    "user",
    "kimi_video_allow_local_urls",
})

# MTP-specific keys that must NEVER appear in AR-only serialization
MTP_SPECIFIC_KEYS: frozenset[str] = frozenset({
    # TextGenerationTaskParams serialization
    "mimo_mtp_fastpath",
    "mimo_mtp_fastpath_params",
    # ChatCompletionRequest extension fields
    "mimo_mtp_depth",
    "mimo_mtp_sidecar_path",
    "mimo_mtp_fail_closed",
    # Hypothetical keys mentioned in the AC that should never appear
    "mtp_tokens",
    "mtp_probabilities",
    "mtp_state",
})


def _dumped_mapping(value: object) -> dict[str, Any]:
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


# ---------------------------------------------------------------------------
# TextGenerationTaskParams: dict and JSON serialization
# ---------------------------------------------------------------------------


class TestTextGenerationTaskParamsSerializationZeroMtpKeys:
    """When MTP is not explicitly enabled, TextGenerationTaskParams serialization
    must contain zero MTP-specific keys and match the canonical AR-only baseline.
    """

    def test_default_params_dict_has_zero_mtp_keys(self) -> None:
        task_params = TextGenerationTaskParams(
            model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
            input=[InputMessage(role="user", content="Hello")],
            max_output_tokens=4,
        )
        dumped = task_params.model_dump()
        actual_keys = frozenset(dumped.keys())

        # Verify exact key-set equality against the known AR-only baseline
        assert actual_keys == AR_ONLY_TASK_PARAMS_DICT_KEYS, (
            f"AR-only TextGenerationTaskParams.model_dump() key set mismatch.\n"
            f"  Extra keys vs baseline: {actual_keys - AR_ONLY_TASK_PARAMS_DICT_KEYS}\n"
            f"  Missing keys vs baseline: {AR_ONLY_TASK_PARAMS_DICT_KEYS - actual_keys}"
        )

        # Verify zero MTP-specific keys are present
        mtp_keys_present = actual_keys & MTP_SPECIFIC_KEYS
        assert mtp_keys_present == frozenset(), (
            f"AR-only serialization must not contain MTP-specific keys, "
            f"but found: {mtp_keys_present}"
        )

    def test_default_params_json_has_zero_mtp_keys(self) -> None:
        task_params = TextGenerationTaskParams(
            model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
            input=[InputMessage(role="user", content="Hello")],
            max_output_tokens=4,
        )
        json_str = task_params.model_dump_json()
        json_keys = frozenset(json.loads(json_str).keys())

        # Verify exact key-set equality against the known AR-only baseline
        assert json_keys == AR_ONLY_TASK_PARAMS_DICT_KEYS, (
            f"AR-only TextGenerationTaskParams.model_dump_json() key set mismatch.\n"
            f"  Extra keys vs baseline: {json_keys - AR_ONLY_TASK_PARAMS_DICT_KEYS}\n"
            f"  Missing keys vs baseline: {AR_ONLY_TASK_PARAMS_DICT_KEYS - json_keys}"
        )

        # Verify zero MTP-specific keys are present in JSON
        mtp_keys_present = json_keys & MTP_SPECIFIC_KEYS
        assert mtp_keys_present == frozenset(), (
            f"AR-only JSON serialization must not contain MTP-specific keys, "
            f"but found: {mtp_keys_present}"
        )

    def test_explicit_none_mtp_params_dict_has_zero_mtp_keys(self) -> None:
        """Explicitly setting mimo_mtp_fastpath_params=None is the same as default AR."""
        task_params = TextGenerationTaskParams(
            model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
            input=[InputMessage(role="user", content="Hello")],
            max_output_tokens=4,
            mimo_mtp_fastpath_params=None,
        )
        dumped = task_params.model_dump()
        actual_keys = frozenset(dumped.keys())

        assert actual_keys == AR_ONLY_TASK_PARAMS_DICT_KEYS
        mtp_keys_present = actual_keys & MTP_SPECIFIC_KEYS
        assert mtp_keys_present == frozenset()

    def test_explicit_none_mtp_params_json_has_zero_mtp_keys(self) -> None:
        task_params = TextGenerationTaskParams(
            model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
            input=[InputMessage(role="user", content="Hello")],
            max_output_tokens=4,
            mimo_mtp_fastpath_params=None,
        )
        json_str = task_params.model_dump_json()
        json_keys = frozenset(json.loads(json_str).keys())

        assert json_keys == AR_ONLY_TASK_PARAMS_DICT_KEYS
        mtp_keys_present = json_keys & MTP_SPECIFIC_KEYS
        assert mtp_keys_present == frozenset()

    def test_disabled_mtp_params_dict_has_zero_mtp_keys(self) -> None:
        """MimoMtpFastpathParams(enabled=False) must serialize identically to AR-only."""
        task_params = TextGenerationTaskParams(
            model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
            input=[InputMessage(role="user", content="Hello")],
            max_output_tokens=4,
            mimo_mtp_fastpath_params=MimoMtpFastpathParams(enabled=False),
        )
        dumped = task_params.model_dump()
        actual_keys = frozenset(dumped.keys())

        assert actual_keys == AR_ONLY_TASK_PARAMS_DICT_KEYS, (
            f"Disabled-MTP TextGenerationTaskParams.model_dump() key set mismatch.\n"
            f"  Extra keys vs baseline: {actual_keys - AR_ONLY_TASK_PARAMS_DICT_KEYS}\n"
            f"  Missing keys vs baseline: {AR_ONLY_TASK_PARAMS_DICT_KEYS - actual_keys}"
        )
        mtp_keys_present = actual_keys & MTP_SPECIFIC_KEYS
        assert mtp_keys_present == frozenset(), (
            f"Disabled-MTP serialization must not contain MTP-specific keys, "
            f"but found: {mtp_keys_present}"
        )

    def test_disabled_mtp_params_json_has_zero_mtp_keys(self) -> None:
        task_params = TextGenerationTaskParams(
            model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
            input=[InputMessage(role="user", content="Hello")],
            max_output_tokens=4,
            mimo_mtp_fastpath_params=MimoMtpFastpathParams(enabled=False),
        )
        json_str = task_params.model_dump_json()
        json_keys = frozenset(json.loads(json_str).keys())

        assert json_keys == AR_ONLY_TASK_PARAMS_DICT_KEYS
        mtp_keys_present = json_keys & MTP_SPECIFIC_KEYS
        assert mtp_keys_present == frozenset()

    def test_enabled_mtp_params_dict_contains_mtp_key(self) -> None:
        """Positive control: enabling MTP DOES add the mtp key."""
        task_params = TextGenerationTaskParams(
            model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
            input=[InputMessage(role="user", content="Hello")],
            max_output_tokens=4,
            mimo_mtp_fastpath_params=MimoMtpFastpathParams(
                depth=2,
                sidecar_path="/models/mimo-mtp.safetensors",
            ),
        )
        dumped = task_params.model_dump()
        actual_keys = frozenset(dumped.keys())

        # The MTP-enabled key set must differ from AR-only baseline
        assert actual_keys != AR_ONLY_TASK_PARAMS_DICT_KEYS
        # The only MTP key that should appear is mimo_mtp_fastpath
        assert "mimo_mtp_fastpath" in actual_keys
        # The internal field mimo_mtp_fastpath_params must NOT appear
        assert "mimo_mtp_fastpath_params" not in actual_keys
        # Hypothetical keys must not appear even in MTP mode
        assert "mtp_tokens" not in actual_keys
        assert "mtp_probabilities" not in actual_keys
        assert "mtp_state" not in actual_keys


# ---------------------------------------------------------------------------
# ChatCompletionRequest: dict and JSON serialization
# ---------------------------------------------------------------------------


class TestChatCompletionRequestSerializationZeroMtpKeys:
    """When MTP is not explicitly enabled, ChatCompletionRequest serialization
    must contain zero MTP-specific keys and match the canonical AR-only baseline.
    """

    def test_default_request_dict_has_zero_mtp_keys(self) -> None:
        request = ChatCompletionRequest(
            model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
            messages=[{"role": "user", "content": "Hello"}],
        )
        dumped = request.model_dump()
        actual_keys = frozenset(dumped.keys())

        assert actual_keys == AR_ONLY_CHAT_REQUEST_DICT_KEYS, (
            f"AR-only ChatCompletionRequest.model_dump() key set mismatch.\n"
            f"  Extra keys vs baseline: {actual_keys - AR_ONLY_CHAT_REQUEST_DICT_KEYS}\n"
            f"  Missing keys vs baseline: {AR_ONLY_CHAT_REQUEST_DICT_KEYS - actual_keys}"
        )
        mtp_keys_present = actual_keys & MTP_SPECIFIC_KEYS
        assert mtp_keys_present == frozenset(), (
            f"AR-only ChatCompletionRequest serialization must not contain "
            f"MTP-specific keys, but found: {mtp_keys_present}"
        )

    def test_default_request_json_has_zero_mtp_keys(self) -> None:
        request = ChatCompletionRequest(
            model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
            messages=[{"role": "user", "content": "Hello"}],
        )
        json_str = request.model_dump_json()
        json_keys = frozenset(json.loads(json_str).keys())

        assert json_keys == AR_ONLY_CHAT_REQUEST_DICT_KEYS
        mtp_keys_present = json_keys & MTP_SPECIFIC_KEYS
        assert mtp_keys_present == frozenset()

    def test_omitted_mtp_fields_dict_has_zero_mtp_keys(self) -> None:
        """Validating from a raw dict that omits MTP fields entirely
        must produce zero MTP keys in serialization.
        """
        request = ChatCompletionRequest.model_validate(
            {
                "model": MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
                "messages": [{"role": "user", "content": "Hello"}],
            }
        )
        dumped = request.model_dump()
        actual_keys = frozenset(dumped.keys())

        assert actual_keys == AR_ONLY_CHAT_REQUEST_DICT_KEYS
        mtp_keys_present = actual_keys & MTP_SPECIFIC_KEYS
        assert mtp_keys_present == frozenset()

    def test_omitted_mtp_fields_exclude_none_dict_has_zero_mtp_keys(self) -> None:
        """exclude_none=True strips None-valued fields, producing a smaller
        key set. We verify zero MTP keys and exact key-set equality against
        a dynamically-computed AR-only exclude_none baseline.
        """
        request = ChatCompletionRequest.model_validate(
            {
                "model": MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
                "messages": [{"role": "user", "content": "Hello"}],
            }
        )
        dumped = request.model_dump(exclude_none=True)
        actual_keys = frozenset(dumped.keys())

        # Compute AR-only exclude_none baseline from the full baseline
        ar_only_exclude_none_keys = AR_ONLY_CHAT_REQUEST_DICT_KEYS & actual_keys
        assert actual_keys == ar_only_exclude_none_keys, (
            f"AR-only ChatCompletionRequest.model_dump(exclude_none=True) key set "
            f"contains non-baseline keys: "
            f"{actual_keys - AR_ONLY_CHAT_REQUEST_DICT_KEYS}"
        )
        mtp_keys_present = actual_keys & MTP_SPECIFIC_KEYS
        assert mtp_keys_present == frozenset(), (
            f"AR-only exclude_none serialization must not contain "
            f"MTP-specific keys, but found: {mtp_keys_present}"
        )

    def test_omitted_mtp_fields_json_exclude_none_has_zero_mtp_keys(self) -> None:
        request = ChatCompletionRequest.model_validate(
            {
                "model": MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
                "messages": [{"role": "user", "content": "Hello"}],
            }
        )
        json_str = request.model_dump_json(exclude_none=True)
        json_keys = frozenset(json.loads(json_str).keys())

        # Compute AR-only exclude_none baseline from the full baseline
        ar_only_exclude_none_keys = AR_ONLY_CHAT_REQUEST_DICT_KEYS & json_keys
        assert json_keys == ar_only_exclude_none_keys, (
            f"AR-only ChatCompletionRequest.model_dump_json(exclude_none=True) "
            f"key set contains non-baseline keys: "
            f"{json_keys - AR_ONLY_CHAT_REQUEST_DICT_KEYS}"
        )
        mtp_keys_present = json_keys & MTP_SPECIFIC_KEYS
        assert mtp_keys_present == frozenset()

    def test_explicit_mtp_false_request_serializes_without_mtp_keys(self) -> None:
        """Even when mimo_mtp_fastpath=False is explicitly set (not omitted),
        the custom serializer must strip it because it was not in model_fields_set.
        """
        request = ChatCompletionRequest(
            model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
            messages=[{"role": "user", "content": "Hello"}],
        )
        # mimo_mtp_fastpath defaults to False, so it's NOT in model_fields_set
        assert "mimo_mtp_fastpath" not in request.model_fields_set
        dumped = request.model_dump()
        actual_keys = frozenset(dumped.keys())

        assert actual_keys == AR_ONLY_CHAT_REQUEST_DICT_KEYS
        mtp_keys_present = actual_keys & MTP_SPECIFIC_KEYS
        assert mtp_keys_present == frozenset()


# ---------------------------------------------------------------------------
# Command and Task union serialization: no MTP keys leak through the wrapper
# ---------------------------------------------------------------------------


class TestCommandTaskUnionSerializationZeroMtpKeys:
    """When MTP is not explicitly enabled, TextGeneration Command and Task
    union serialization must not contain MTP-specific keys in the
    nested task_params dict.
    """

    def test_command_dict_has_zero_mtp_keys_in_task_params(self) -> None:
        command = TextGenCommand(
            command_id=CommandId("cmd-ar-test"),
            task_params=TextGenerationTaskParams(
                model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
                input=[InputMessage(role="user", content="Hello")],
                max_output_tokens=4,
            ),
        )
        dumped = _dumped_mapping(command.model_dump())
        text_gen_payload = _dumped_mapping(dumped["TextGeneration"])
        task_param_keys = frozenset(
            _dumped_mapping(text_gen_payload["task_params"]).keys()
        )

        assert task_param_keys == AR_ONLY_TASK_PARAMS_DICT_KEYS, (
            f"AR-only Command TextGeneration task_params key set mismatch.\n"
            f"  Extra keys vs baseline: {task_param_keys - AR_ONLY_TASK_PARAMS_DICT_KEYS}\n"
            f"  Missing keys vs baseline: {AR_ONLY_TASK_PARAMS_DICT_KEYS - task_param_keys}"
        )
        mtp_keys_present = task_param_keys & MTP_SPECIFIC_KEYS
        assert mtp_keys_present == frozenset(), (
            f"AR-only Command task_params must not contain MTP-specific keys, "
            f"but found: {mtp_keys_present}"
        )

    def test_command_json_has_zero_mtp_keys_in_task_params(self) -> None:
        command = TextGenCommand(
            command_id=CommandId("cmd-ar-json-test"),
            task_params=TextGenerationTaskParams(
                model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
                input=[InputMessage(role="user", content="Hello")],
                max_output_tokens=4,
            ),
        )
        json_str = command.model_dump_json()
        json_obj = _dumped_mapping(json.loads(json_str))
        text_gen_payload = _dumped_mapping(json_obj["TextGeneration"])
        task_param_keys = frozenset(
            _dumped_mapping(text_gen_payload["task_params"]).keys()
        )

        assert task_param_keys == AR_ONLY_TASK_PARAMS_DICT_KEYS
        mtp_keys_present = task_param_keys & MTP_SPECIFIC_KEYS
        assert mtp_keys_present == frozenset()

    def test_command_type_adapter_json_has_zero_mtp_keys_in_task_params(self) -> None:
        """TypeAdapter (used for topic serialization) must also produce zero MTP keys."""
        command = TextGenCommand(
            command_id=CommandId("cmd-adapter-test"),
            task_params=TextGenerationTaskParams(
                model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
                input=[InputMessage(role="user", content="Hello")],
                max_output_tokens=4,
            ),
        )
        payload_json = COMMAND_ADAPTER.dump_json(command)
        json_obj = _dumped_mapping(json.loads(payload_json))
        text_gen_payload = _dumped_mapping(json_obj["TextGeneration"])
        task_param_keys = frozenset(
            _dumped_mapping(text_gen_payload["task_params"]).keys()
        )

        assert task_param_keys == AR_ONLY_TASK_PARAMS_DICT_KEYS
        mtp_keys_present = task_param_keys & MTP_SPECIFIC_KEYS
        assert mtp_keys_present == frozenset()

    def test_task_dict_has_zero_mtp_keys_in_task_params(self) -> None:
        task = TextGenTask(
            task_id=TaskId("task-ar-test"),
            task_status=TaskStatus.Pending,
            instance_id=InstanceId("instance-ar-test"),
            command_id=CommandId("cmd-ar-test"),
            task_params=TextGenerationTaskParams(
                model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
                input=[InputMessage(role="user", content="Hello")],
                max_output_tokens=4,
            ),
        )
        dumped = _dumped_mapping(task.model_dump())
        text_gen_payload = _dumped_mapping(dumped["TextGeneration"])
        task_param_keys = frozenset(
            _dumped_mapping(text_gen_payload["task_params"]).keys()
        )

        assert task_param_keys == AR_ONLY_TASK_PARAMS_DICT_KEYS
        mtp_keys_present = task_param_keys & MTP_SPECIFIC_KEYS
        assert mtp_keys_present == frozenset()

    def test_task_json_has_zero_mtp_keys_in_task_params(self) -> None:
        task = TextGenTask(
            task_id=TaskId("task-ar-json-test"),
            task_status=TaskStatus.Pending,
            instance_id=InstanceId("instance-ar-json-test"),
            command_id=CommandId("cmd-ar-json-test"),
            task_params=TextGenerationTaskParams(
                model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
                input=[InputMessage(role="user", content="Hello")],
                max_output_tokens=4,
            ),
        )
        json_str = task.model_dump_json()
        json_obj = _dumped_mapping(json.loads(json_str))
        text_gen_payload = _dumped_mapping(json_obj["TextGeneration"])
        task_param_keys = frozenset(
            _dumped_mapping(text_gen_payload["task_params"]).keys()
        )

        assert task_param_keys == AR_ONLY_TASK_PARAMS_DICT_KEYS
        mtp_keys_present = task_param_keys & MTP_SPECIFIC_KEYS
        assert mtp_keys_present == frozenset()

    def test_task_type_adapter_json_has_zero_mtp_keys_in_task_params(self) -> None:
        """TypeAdapter (used for topic serialization) must also produce zero MTP keys."""
        task = TextGenTask(
            task_id=TaskId("task-adapter-test"),
            task_status=TaskStatus.Pending,
            instance_id=InstanceId("instance-adapter-test"),
            command_id=CommandId("cmd-adapter-test"),
            task_params=TextGenerationTaskParams(
                model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
                input=[InputMessage(role="user", content="Hello")],
                max_output_tokens=4,
            ),
        )
        payload_json = TASK_ADAPTER.dump_json(task)
        json_obj = _dumped_mapping(json.loads(payload_json))
        text_gen_payload = _dumped_mapping(json_obj["TextGeneration"])
        task_param_keys = frozenset(
            _dumped_mapping(text_gen_payload["task_params"]).keys()
        )

        assert task_param_keys == AR_ONLY_TASK_PARAMS_DICT_KEYS
        mtp_keys_present = task_param_keys & MTP_SPECIFIC_KEYS
        assert mtp_keys_present == frozenset()

    def test_forwarder_command_dict_has_zero_mtp_keys_in_task_params(self) -> None:
        """ForwarderCommand wrapping an AR-only TextGeneration command
        must not leak MTP keys through any layer.
        """
        command = TextGenCommand(
            command_id=CommandId("cmd-fwd-ar-test"),
            task_params=TextGenerationTaskParams(
                model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
                input=[InputMessage(role="user", content="Hello")],
                max_output_tokens=4,
            ),
        )
        forwarder = ForwarderCommand(
            origin=SystemId("system-ar-test"),
            command=command,
        )
        dumped = _dumped_mapping(forwarder.model_dump())
        command_payload = _dumped_mapping(dumped["command"])
        text_gen_payload = _dumped_mapping(command_payload["TextGeneration"])
        task_param_keys = frozenset(
            _dumped_mapping(text_gen_payload["task_params"]).keys()
        )

        assert task_param_keys == AR_ONLY_TASK_PARAMS_DICT_KEYS
        mtp_keys_present = task_param_keys & MTP_SPECIFIC_KEYS
        assert mtp_keys_present == frozenset()


# ---------------------------------------------------------------------------
# Round-trip deserialization: no MTP keys appear after parse-then-serialize
# ---------------------------------------------------------------------------


class TestRoundTripSerializationZeroMtpKeys:
    """After deserializing an AR-only payload and re-serializing,
    the output must still contain zero MTP-specific keys.
    """

    def test_command_round_trip_dict_has_zero_mtp_keys(self) -> None:
        payload: dict[str, Any] = {
            "TextGeneration": {
                "command_id": "cmd-round-trip-ar",
                "task_params": {
                    "model": MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
                    "input": [{"role": "user", "content": "Hello"}],
                    "max_output_tokens": 4,
                },
            }
        }
        restored = COMMAND_ADAPTER.validate_python(payload)
        assert isinstance(restored, TextGenCommand)
        assert restored.task_params.mimo_mtp_fastpath is None

        # Re-serialize and verify zero MTP keys
        dumped = _dumped_mapping(restored.model_dump())
        text_gen_payload = _dumped_mapping(dumped["TextGeneration"])
        task_param_keys = frozenset(
            _dumped_mapping(text_gen_payload["task_params"]).keys()
        )

        assert task_param_keys == AR_ONLY_TASK_PARAMS_DICT_KEYS
        mtp_keys_present = task_param_keys & MTP_SPECIFIC_KEYS
        assert mtp_keys_present == frozenset()

    def test_task_round_trip_json_has_zero_mtp_keys(self) -> None:
        payload: dict[str, Any] = {
            "TextGeneration": {
                "task_id": "task-round-trip-ar",
                "task_status": "Pending",
                "instance_id": "instance-round-trip-ar",
                "command_id": "cmd-round-trip-ar",
                "task_params": {
                    "model": MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
                    "input": [{"role": "user", "content": "Hello"}],
                    "max_output_tokens": 4,
                },
            }
        }
        restored = TASK_ADAPTER.validate_json(json.dumps(payload))
        assert isinstance(restored, TextGenTask)
        assert restored.task_params.mimo_mtp_fastpath is None

        # Re-serialize and verify zero MTP keys
        json_str = restored.model_dump_json()
        json_obj = _dumped_mapping(json.loads(json_str))
        text_gen_payload = _dumped_mapping(json_obj["TextGeneration"])
        task_param_keys = frozenset(
            _dumped_mapping(text_gen_payload["task_params"]).keys()
        )

        assert task_param_keys == AR_ONLY_TASK_PARAMS_DICT_KEYS
        mtp_keys_present = task_param_keys & MTP_SPECIFIC_KEYS
        assert mtp_keys_present == frozenset()

    def test_chat_request_round_trip_json_has_zero_mtp_keys(self) -> None:
        raw_json = json.dumps(
            {
                "model": MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
                "messages": [{"role": "user", "content": "Hello"}],
            }
        )
        request = ChatCompletionRequest.model_validate_json(raw_json)
        assert request.mimo_mtp_fastpath is False

        # Re-serialize and verify zero MTP keys
        json_str = request.model_dump_json()
        json_keys = frozenset(json.loads(json_str).keys())

        assert json_keys == AR_ONLY_CHAT_REQUEST_DICT_KEYS
        mtp_keys_present = json_keys & MTP_SPECIFIC_KEYS
        assert mtp_keys_present == frozenset()
