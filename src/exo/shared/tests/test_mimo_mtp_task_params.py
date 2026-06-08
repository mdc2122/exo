import json
from collections.abc import Sequence
from typing import Any, cast

import pytest
from pydantic import TypeAdapter, ValidationError

from exo.routing.topics import COMMANDS
from exo.shared.models.model_cards import MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID
from exo.shared.types.commands import Command as CommandUnion
from exo.shared.types.commands import ForwarderCommand, TextGeneration
from exo.shared.types.common import CommandId, SystemId
from exo.shared.types.tasks import Task as TaskUnion
from exo.shared.types.tasks import TaskId, TaskStatus
from exo.shared.types.tasks import TextGeneration as TextGenerationTask
from exo.shared.types.text_generation import (
    InputMessage,
    MimoMtpFastpathParams,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId


def _mtp_disabled_params() -> TextGenerationTaskParams:
    """TextGenerationTaskParams with MTP explicitly disabled (enabled=False)."""
    return TextGenerationTaskParams(
        model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
        input=[InputMessage(role="user", content="Hello")],
        max_output_tokens=4,
        mimo_mtp_fastpath_params=MimoMtpFastpathParams(enabled=False),
    )


def _mtp_disabled_full_params() -> TextGenerationTaskParams:
    """TextGenerationTaskParams with MTP explicitly disabled but all sub-fields populated."""
    return TextGenerationTaskParams(
        model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
        input=[InputMessage(role="user", content="Hello")],
        max_output_tokens=4,
        mimo_mtp_fastpath_params=MimoMtpFastpathParams(
            enabled=False,
            depth=3,
            sidecar_path="/models/mimo-v2.5-pro-mtp.safetensors",
            fail_closed=False,
        ),
    )


COMMAND_ADAPTER: TypeAdapter[CommandUnion] = TypeAdapter(CommandUnion)
TASK_ADAPTER: TypeAdapter[TaskUnion] = TypeAdapter(TaskUnion)


def _base_text_generation_task_params() -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
        input=[InputMessage(role="user", content="Hello")],
        max_output_tokens=4,
    )


def _dumped_mapping(value: object) -> dict[str, Any]:
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _dumped_mapping_value(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    return _dumped_mapping(cast(object, mapping[key]))


def _validation_error_has_location_suffix(
    validation_error: ValidationError,
    *,
    error_type: str,
    location_suffix: Sequence[object],
) -> bool:
    for raw_error in validation_error.errors():
        error = cast(dict[str, object], cast(object, raw_error))
        raw_error_type = error["type"]
        raw_location = error["loc"]
        assert isinstance(raw_error_type, str)
        assert isinstance(raw_location, tuple)
        if raw_error_type == error_type and raw_location[
            -len(location_suffix) :
        ] == tuple(location_suffix):
            return True
    return False


def _assert_validation_error_has_location_suffix(
    validation_error: ValidationError,
    *,
    error_type: str,
    location_suffix: Sequence[object],
) -> None:
    assert _validation_error_has_location_suffix(
        validation_error,
        error_type=error_type,
        location_suffix=location_suffix,
    ), validation_error.errors()


def _mutate_mtp_field(
    mtp_fastpath_params: MimoMtpFastpathParams, field_name: str, value: object
) -> None:
    setattr(mtp_fastpath_params, field_name, value)


def _valid_enabled_mtp_task_params() -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
        input=[InputMessage(role="user", content="Hello")],
        max_output_tokens=4,
        temperature=0.0,
        seed=1234,
        mimo_mtp_fastpath_params=MimoMtpFastpathParams(
            enabled=True,
            depth=3,
            sidecar_path="/models/mimo-v2.5-pro-mtp.safetensors",
            fail_closed=True,
        ),
    )


def _assert_valid_enabled_mtp_params(
    task_params: TextGenerationTaskParams,
) -> None:
    assert task_params.model == MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID
    assert task_params.max_output_tokens == 4
    assert task_params.temperature == 0.0
    assert task_params.seed == 1234
    assert task_params.input == [InputMessage(role="user", content="Hello")]

    mtp_params = task_params.mimo_mtp_fastpath
    assert isinstance(mtp_params, MimoMtpFastpathParams)
    assert mtp_params.enabled is True
    assert mtp_params.depth == 3
    assert mtp_params.sidecar_path == "/models/mimo-v2.5-pro-mtp.safetensors"
    assert mtp_params.fail_closed is True


def test_default_text_generation_task_params_keep_mtp_intent_absent() -> None:
    task_params = _base_text_generation_task_params()

    assert task_params.mimo_mtp_fastpath is None
    assert "mimo_mtp_fastpath" not in task_params.model_dump()
    assert "mimo_mtp_fastpath" not in task_params.model_dump_json()


def test_explicit_none_text_generation_task_params_keep_mtp_intent_absent() -> None:
    task_params = TextGenerationTaskParams(
        model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
        input=[InputMessage(role="user", content="Hello")],
        max_output_tokens=4,
        mimo_mtp_fastpath_params=None,
    )

    assert task_params.mimo_mtp_fastpath is None
    assert "mimo_mtp_fastpath" not in task_params.model_dump()
    assert "mimo_mtp_fastpath" not in task_params.model_dump_json()


def test_explicit_mtp_fastpath_params_are_immutable_and_serialized_nested() -> None:
    task_params = _valid_enabled_mtp_task_params()

    _assert_valid_enabled_mtp_params(task_params)

    assert task_params.mimo_mtp_fastpath is not None
    with pytest.raises(ValidationError):
        _mutate_mtp_field(task_params.mimo_mtp_fastpath, "depth", 2)

    dumped = task_params.model_dump()
    assert dumped["mimo_mtp_fastpath"] == {
        "enabled": True,
        "depth": 3,
        "sidecar_path": "/models/mimo-v2.5-pro-mtp.safetensors",
        "fail_closed": True,
    }


def test_mtp_fastpath_params_reject_coerced_and_extra_values() -> None:
    with pytest.raises(ValidationError):
        MimoMtpFastpathParams.model_validate(
            {
                "enabled": "true",
                "depth": "2",
                "sidecar_path": "/models/model_mtp.safetensors",
                "fail_closed": "false",
            }
        )

    with pytest.raises(ValidationError):
        MimoMtpFastpathParams.model_validate(
            {
                "enabled": True,
                "depth": 2,
                "sidecar_path": "/models/model_mtp.safetensors",
                "fail_closed": True,
                "unexpected_controller_flag": True,
            }
        )


def test_mtp_fastpath_params_reject_unsupported_depth_with_disable_reason() -> None:
    with pytest.raises(ValidationError) as exc_info:
        MimoMtpFastpathParams(
            depth=4,
            sidecar_path="/models/model_mtp.safetensors",
            fail_closed=False,
        )

    error_message = str(exc_info.value)
    assert "unsupported MTP depth 4" in error_message
    assert "supported depths: 1,2,3" in error_message
    assert "disable_reason=unsupported_depth" in error_message


def test_text_generation_command_round_trips_legacy_payload_without_mtp_key() -> None:
    command = TextGeneration(
        command_id=CommandId("cmd-legacy"),
        task_params=_base_text_generation_task_params(),
    )
    payload = _dumped_mapping(command.model_dump())
    text_generation_payload = _dumped_mapping_value(payload, "TextGeneration")
    task_param_payload = _dumped_mapping_value(text_generation_payload, "task_params")

    assert "mimo_mtp_fastpath" not in task_param_payload

    restored = TextGeneration.model_validate(payload)

    assert restored.task_params.mimo_mtp_fastpath is None
    assert restored.task_params.stream is False
    assert restored.task_params.bench is False
    assert restored.task_params.images == []
    assert restored.task_params.videos == []
    assert "mimo_mtp_fastpath" not in restored.task_params.model_dump()


def test_legacy_text_generation_command_union_payload_without_mtp_key_parses_as_default_ar() -> None:
    payload: dict[str, Any] = {
        "TextGeneration": {
            "command_id": "cmd-legacy-raw",
            "task_params": {
                "model": MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
                "input": [{"role": "user", "content": "Hello"}],
                "max_output_tokens": 4,
            },
        }
    }

    restored = cast(TextGeneration, COMMAND_ADAPTER.validate_python(payload))

    assert isinstance(restored, TextGeneration)
    assert restored.command_id == CommandId("cmd-legacy-raw")
    assert restored.task_params.mimo_mtp_fastpath is None
    assert restored.task_params.stream is False
    assert restored.task_params.bench is False
    assert restored.task_params.images == []
    assert restored.task_params.videos == []

    dumped = _dumped_mapping(restored.model_dump())
    text_generation_payload = _dumped_mapping_value(dumped, "TextGeneration")
    task_param_payload = _dumped_mapping_value(text_generation_payload, "task_params")
    assert "mimo_mtp_fastpath" not in task_param_payload


def test_legacy_command_topic_payload_without_mtp_key_deserializes_as_default_ar() -> None:
    payload = json.dumps(
        {
            "origin": "system-legacy-topic",
            "command": {
                "TextGeneration": {
                    "command_id": "cmd-legacy-topic",
                    "task_params": {
                        "model": MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
                        "input": [{"role": "user", "content": "Hello"}],
                        "max_output_tokens": 4,
                    },
                }
            },
        }
    ).encode("utf-8")

    restored = COMMANDS.deserialize(payload)

    assert restored.origin == SystemId("system-legacy-topic")
    assert isinstance(restored.command, TextGeneration)
    assert restored.command.command_id == CommandId("cmd-legacy-topic")
    task_params = restored.command.task_params
    assert task_params.mimo_mtp_fastpath is None
    assert "mimo_mtp_fastpath_params" not in task_params.model_fields_set
    assert "mimo_mtp_fastpath" not in task_params.model_fields_set
    assert task_params.stream is False
    assert task_params.bench is False
    assert task_params.images == []
    assert task_params.videos == []

    dumped = _dumped_mapping(restored.model_dump())
    command_payload = _dumped_mapping_value(dumped, "command")
    text_generation_payload = _dumped_mapping_value(command_payload, "TextGeneration")
    task_param_payload = _dumped_mapping_value(text_generation_payload, "task_params")
    assert "mimo_mtp_fastpath" not in task_param_payload
    assert "mimo_mtp_fastpath_params" not in task_param_payload


def test_legacy_text_generation_task_union_payload_without_mtp_key_parses_as_default_ar() -> None:
    payload: dict[str, Any] = {
        "TextGeneration": {
            "task_id": "task-legacy-raw",
            "task_status": "Pending",
            "instance_id": "instance-legacy",
            "command_id": "cmd-legacy-raw",
            "task_params": {
                "model": MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
                "input": [{"role": "user", "content": "Hello"}],
                "max_output_tokens": 4,
            },
        }
    }

    restored = cast(TextGenerationTask, TASK_ADAPTER.validate_json(json.dumps(payload)))

    assert isinstance(restored, TextGenerationTask)
    assert restored.task_id == TaskId("task-legacy-raw")
    assert restored.instance_id == InstanceId("instance-legacy")
    assert restored.command_id == CommandId("cmd-legacy-raw")
    assert restored.error_type is None
    assert restored.error_message is None
    assert restored.task_params.mimo_mtp_fastpath is None
    assert restored.task_params.stream is False
    assert restored.task_params.bench is False
    assert restored.task_params.images == []
    assert restored.task_params.videos == []

    dumped = _dumped_mapping(restored.model_dump())
    text_generation_payload = _dumped_mapping_value(dumped, "TextGeneration")
    task_param_payload = _dumped_mapping_value(text_generation_payload, "task_params")
    assert "mimo_mtp_fastpath" not in task_param_payload


def test_forwarder_command_round_trips_explicit_mtp_intent_type_safely() -> None:
    command = TextGeneration(
        command_id=CommandId("cmd-mtp"),
        task_params=TextGenerationTaskParams(
            model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
            input=[InputMessage(role="user", content="Hello")],
            max_output_tokens=4,
            mimo_mtp_fastpath_params=MimoMtpFastpathParams(
                depth=1,
                sidecar_path="/models/model_mtp.safetensors",
                fail_closed=True,
            ),
        ),
    )
    forwarder_command = ForwarderCommand(
        origin=SystemId("system-a"),
        command=command,
    )

    payload = _dumped_mapping(forwarder_command.model_dump())
    restored = ForwarderCommand.model_validate(payload)

    assert isinstance(restored.command, TextGeneration)
    mtp_params = restored.command.task_params.mimo_mtp_fastpath
    assert isinstance(mtp_params, MimoMtpFastpathParams)
    assert mtp_params.enabled is True
    assert mtp_params.depth == 1
    assert mtp_params.sidecar_path == "/models/model_mtp.safetensors"
    assert mtp_params.fail_closed is True


def test_command_topic_serialization_preserves_explicit_enabled_mtp_param_types() -> None:
    command = TextGeneration(
        command_id=CommandId("cmd-topic-mtp"),
        task_params=TextGenerationTaskParams(
            model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
            input=[InputMessage(role="user", content="Hello")],
            max_output_tokens=4,
            temperature=0.0,
            seed=1234,
            mimo_mtp_fastpath_params=MimoMtpFastpathParams(
                enabled=True,
                depth=2,
                sidecar_path="/models/model_mtp.safetensors",
                fail_closed=False,
            ),
        ),
    )
    forwarder_command = ForwarderCommand(
        origin=SystemId("system-topic"),
        command=command,
    )

    payload = COMMANDS.serialize(forwarder_command)
    restored = COMMANDS.deserialize(payload)

    assert isinstance(restored.command, TextGeneration)
    assert restored.origin == SystemId("system-topic")
    assert restored.command.command_id == CommandId("cmd-topic-mtp")
    mtp_params = restored.command.task_params.mimo_mtp_fastpath
    assert isinstance(mtp_params, MimoMtpFastpathParams)
    assert mtp_params.enabled is True
    assert mtp_params.depth == 2
    assert mtp_params.sidecar_path == "/models/model_mtp.safetensors"
    assert mtp_params.fail_closed is False
    assert COMMANDS.serialize(restored) == payload


def test_command_payload_with_valid_enabled_mtp_round_trips_without_type_or_value_loss() -> None:
    command = TextGeneration(
        command_id=CommandId("cmd-valid-mtp"),
        task_params=_valid_enabled_mtp_task_params(),
    )
    payload_json = COMMAND_ADAPTER.dump_json(command)
    restored = cast(TextGeneration, COMMAND_ADAPTER.validate_json(payload_json))

    assert isinstance(restored, TextGeneration)
    assert restored.command_id == CommandId("cmd-valid-mtp")
    _assert_valid_enabled_mtp_params(restored.task_params)
    assert COMMAND_ADAPTER.dump_json(restored) == payload_json


def test_task_payload_with_valid_enabled_mtp_round_trips_without_type_or_value_loss() -> None:
    task = TextGenerationTask(
        task_id=TaskId("task-valid-mtp"),
        task_status=TaskStatus.Running,
        instance_id=InstanceId("instance-valid-mtp"),
        command_id=CommandId("cmd-valid-mtp"),
        task_params=_valid_enabled_mtp_task_params(),
    )
    payload_json = TASK_ADAPTER.dump_json(task)
    restored = cast(TextGenerationTask, TASK_ADAPTER.validate_json(payload_json))

    assert isinstance(restored, TextGenerationTask)
    assert restored.task_id == TaskId("task-valid-mtp")
    assert restored.task_status == "Running"
    assert restored.instance_id == InstanceId("instance-valid-mtp")
    assert restored.command_id == CommandId("cmd-valid-mtp")
    _assert_valid_enabled_mtp_params(restored.task_params)
    assert TASK_ADAPTER.dump_json(restored) == payload_json


def test_text_generation_task_model_json_round_trips_explicit_enabled_mtp_params() -> None:
    task = TextGenerationTask(
        task_id=TaskId("task-model-json-mtp"),
        task_status=TaskStatus.Running,
        instance_id=InstanceId("instance-model-json-mtp"),
        command_id=CommandId("cmd-model-json-mtp"),
        task_params=_valid_enabled_mtp_task_params(),
    )

    payload_json = task.model_dump_json()
    payload_json_object = cast(object, json.loads(payload_json))
    payload = _dumped_mapping(payload_json_object)
    text_generation_payload = _dumped_mapping_value(payload, "TextGeneration")
    task_param_payload = _dumped_mapping_value(text_generation_payload, "task_params")
    mtp_payload = _dumped_mapping_value(task_param_payload, "mimo_mtp_fastpath")

    assert "mimo_mtp_fastpath_params" not in task_param_payload
    assert mtp_payload == {
        "enabled": True,
        "depth": 3,
        "sidecar_path": "/models/mimo-v2.5-pro-mtp.safetensors",
        "fail_closed": True,
    }
    assert isinstance(mtp_payload["enabled"], bool)
    assert isinstance(mtp_payload["depth"], int)
    assert isinstance(mtp_payload["sidecar_path"], str)
    assert isinstance(mtp_payload["fail_closed"], bool)

    restored = TextGenerationTask.model_validate_json(payload_json)

    assert restored.task_id == TaskId("task-model-json-mtp")
    assert restored.task_status is TaskStatus.Running
    assert restored.instance_id == InstanceId("instance-model-json-mtp")
    assert restored.command_id == CommandId("cmd-model-json-mtp")
    _assert_valid_enabled_mtp_params(restored.task_params)
    assert restored.model_dump_json() == payload_json


def test_text_generation_task_params_mtp_none_round_trips_through_command_serialization_with_structural_equality() -> None:
    """TextGenerationTaskParams with MTP=None round-trips through all command
    serialization paths (COMMANDS topic, TypeAdapter JSON, model_dump_json)
    with structural equality on every field and idempotent serialization."""
    original_params = _base_text_generation_task_params()
    assert original_params.mimo_mtp_fastpath is None

    # --- Path 1: COMMANDS topic serialize/deserialize ---
    command_1 = TextGeneration(
        command_id=CommandId("cmd-rt-mtp-none-topic"),
        task_params=original_params,
    )
    forwarder_1 = ForwarderCommand(
        origin=SystemId("system-rt-mtp-none-topic"),
        command=command_1,
    )
    payload_bytes = COMMANDS.serialize(forwarder_1)
    restored_fc = COMMANDS.deserialize(payload_bytes)
    assert isinstance(restored_fc.command, TextGeneration)
    restored_params_topic = restored_fc.command.task_params
    assert restored_params_topic.mimo_mtp_fastpath is None
    assert restored_params_topic.model_dump() == original_params.model_dump()
    assert COMMANDS.serialize(restored_fc) == payload_bytes

    # --- Path 2: TypeAdapter[Command] JSON round-trip ---
    command_2 = TextGeneration(
        command_id=CommandId("cmd-rt-mtp-none-adapter"),
        task_params=original_params,
    )
    payload_json = COMMAND_ADAPTER.dump_json(command_2)
    restored_cmd = cast(TextGeneration, COMMAND_ADAPTER.validate_json(payload_json))
    assert isinstance(restored_cmd, TextGeneration)
    restored_params_adapter = restored_cmd.task_params
    assert restored_params_adapter.mimo_mtp_fastpath is None
    assert restored_params_adapter.model_dump() == original_params.model_dump()
    assert COMMAND_ADAPTER.dump_json(restored_cmd) == payload_json

    # --- Path 3: model_dump_json on TextGenerationTaskParams directly ---
    params_json = original_params.model_dump_json()
    restored_params_direct = TextGenerationTaskParams.model_validate_json(params_json)
    assert restored_params_direct.mimo_mtp_fastpath is None
    assert restored_params_direct.model_dump() == original_params.model_dump()
    assert restored_params_direct.model_dump_json() == params_json

    # --- Cross-path structural equality: all three restored params are equal ---
    assert restored_params_topic.model_dump() == restored_params_adapter.model_dump()
    assert restored_params_adapter.model_dump() == restored_params_direct.model_dump()

    # --- MTP keys must not leak into serialized output ---
    assert "mimo_mtp_fastpath" not in original_params.model_dump()
    assert "mimo_mtp_fastpath_params" not in original_params.model_dump()
    assert "mimo_mtp_fastpath" not in restored_params_topic.model_dump()
    assert "mimo_mtp_fastpath_params" not in restored_params_topic.model_dump()
    assert "mimo_mtp_fastpath" not in restored_params_adapter.model_dump()
    assert "mimo_mtp_fastpath_params" not in restored_params_adapter.model_dump()
    assert "mimo_mtp_fastpath" not in restored_params_direct.model_dump()
    assert "mimo_mtp_fastpath_params" not in restored_params_direct.model_dump()


def test_text_generation_task_params_mtp_disabled_round_trips_through_command_serialization_with_structural_equality() -> None:
    """Sub-AC 5.5c: TextGenerationTaskParams with MTP=disabled round-trips through all command serialization paths with Pydantic == structural equality and idempotent serialization."""
    original_params = _base_text_generation_task_params()
    assert original_params.mimo_mtp_fastpath is None

    # --- Path 1: COMMANDS topic serialize/deserialize ---
    command_topic = TextGeneration(
        command_id=CommandId("cmd-5-5c-mtp-disabled-topic"),
        task_params=original_params,
    )
    forwarder_topic = ForwarderCommand(
        origin=SystemId("system-5-5c-mtp-disabled-topic"),
        command=command_topic,
    )
    payload_bytes = COMMANDS.serialize(forwarder_topic)
    restored_fc = COMMANDS.deserialize(payload_bytes)
    assert isinstance(restored_fc.command, TextGeneration)
    restored_params_topic = restored_fc.command.task_params
    assert restored_params_topic == original_params
    assert restored_params_topic.mimo_mtp_fastpath is None
    assert COMMANDS.serialize(restored_fc) == payload_bytes

    # --- Path 2: TypeAdapter[Command] JSON round-trip ---
    command_adapter = TextGeneration(
        command_id=CommandId("cmd-5-5c-mtp-disabled-adapter"),
        task_params=original_params,
    )
    payload_json = COMMAND_ADAPTER.dump_json(command_adapter)
    restored_cmd = cast(TextGeneration, COMMAND_ADAPTER.validate_json(payload_json))
    assert isinstance(restored_cmd, TextGeneration)
    restored_params_adapter = restored_cmd.task_params
    assert restored_params_adapter == original_params
    assert restored_params_adapter.mimo_mtp_fastpath is None
    assert COMMAND_ADAPTER.dump_json(restored_cmd) == payload_json

    # --- Path 3: model_dump_json / model_validate_json on TextGenerationTaskParams ---
    params_json = original_params.model_dump_json()
    restored_params_direct = TextGenerationTaskParams.model_validate_json(params_json)
    assert restored_params_direct == original_params
    assert restored_params_direct.mimo_mtp_fastpath is None
    assert restored_params_direct.model_dump_json() == params_json

    # --- Cross-path structural equality: all three restored params are == equal ---
    assert restored_params_topic == restored_params_adapter
    assert restored_params_adapter == restored_params_direct

    # --- MTP keys must not leak into any serialized output ---
    assert "mimo_mtp_fastpath" not in original_params.model_dump()
    assert "mimo_mtp_fastpath_params" not in original_params.model_dump()
    assert "mimo_mtp_fastpath" not in restored_params_topic.model_dump()
    assert "mimo_mtp_fastpath_params" not in restored_params_topic.model_dump()
    assert "mimo_mtp_fastpath" not in restored_params_adapter.model_dump()
    assert "mimo_mtp_fastpath_params" not in restored_params_adapter.model_dump()
    assert "mimo_mtp_fastpath" not in restored_params_direct.model_dump()
    assert "mimo_mtp_fastpath_params" not in restored_params_direct.model_dump()


def test_text_generation_task_params_mtp_enabled_round_trips_through_command_type_adapter_with_structural_equality() -> None:
    """Sub-AC 5.5b: TextGenerationTaskParams with MTP=enabled round-trips
    through command TypeAdapter serialization with structural equality."""
    original_params = _valid_enabled_mtp_task_params()
    command = TextGeneration(
        command_id=CommandId("cmd-structural-eq-mtp"),
        task_params=original_params,
    )
    payload_json = COMMAND_ADAPTER.dump_json(command)
    restored = cast(TextGeneration, COMMAND_ADAPTER.validate_json(payload_json))
    assert isinstance(restored, TextGeneration)
    assert restored.task_params == original_params


def test_text_generation_task_params_mtp_enabled_round_trips_through_command_topic_with_structural_equality() -> None:
    """Sub-AC 5.5b: TextGenerationTaskParams with MTP=enabled round-trips
    through command topic serialization with structural equality."""
    original_params = _valid_enabled_mtp_task_params()
    command = TextGeneration(
        command_id=CommandId("cmd-topic-structural-eq-mtp"),
        task_params=original_params,
    )
    forwarder_command = ForwarderCommand(
        origin=SystemId("system-structural-eq"),
        command=command,
    )
    payload = COMMANDS.serialize(forwarder_command)
    restored = COMMANDS.deserialize(payload)
    assert isinstance(restored.command, TextGeneration)
    assert restored.command.task_params == original_params


def test_command_deserialization_rejects_invalid_mtp_param_shape() -> None:
    payload: dict[str, object] = {
        "TextGeneration": {
            "command_id": "cmd-invalid-mtp-shape",
            "task_params": {
                "model": MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
                "input": [{"role": "user", "content": "Hello"}],
                "max_output_tokens": 4,
                "mimo_mtp_fastpath": ["enabled", "depth"],
            },
        }
    }

    with pytest.raises(ValidationError) as exc_info:
        TypeAdapter[CommandUnion](CommandUnion).validate_python(payload)

    _assert_validation_error_has_location_suffix(
        exc_info.value,
        error_type="model_type",
        location_suffix=("task_params", "mimo_mtp_fastpath"),
    )


def test_command_deserialization_rejects_incorrect_mtp_field_types() -> None:
    payload: dict[str, object] = {
        "TextGeneration": {
            "command_id": "cmd-invalid-mtp-field-types",
            "task_params": {
                "model": MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
                "input": [{"role": "user", "content": "Hello"}],
                "max_output_tokens": 4,
                "mimo_mtp_fastpath": {
                    "enabled": "true",
                    "depth": "3",
                    "sidecar_path": 123,
                    "fail_closed": "false",
                },
            },
        }
    }

    with pytest.raises(ValidationError) as exc_info:
        TypeAdapter[CommandUnion](CommandUnion).validate_python(payload)

    validation_error = exc_info.value
    _assert_validation_error_has_location_suffix(
        validation_error,
        error_type="bool_type",
        location_suffix=("task_params", "mimo_mtp_fastpath", "enabled"),
    )
    _assert_validation_error_has_location_suffix(
        validation_error,
        error_type="int_type",
        location_suffix=("task_params", "mimo_mtp_fastpath", "depth"),
    )
    _assert_validation_error_has_location_suffix(
        validation_error,
        error_type="string_type",
        location_suffix=("task_params", "mimo_mtp_fastpath", "sidecar_path"),
    )
    _assert_validation_error_has_location_suffix(
        validation_error,
        error_type="bool_type",
        location_suffix=("task_params", "mimo_mtp_fastpath", "fail_closed"),
    )


def test_task_deserialization_rejects_invalid_mtp_param_shape() -> None:
    payload: dict[str, object] = {
        "TextGeneration": {
            "task_id": "task-invalid-mtp-shape",
            "task_status": "Pending",
            "instance_id": "instance-invalid-mtp-shape",
            "command_id": "cmd-invalid-mtp-shape",
            "task_params": {
                "model": MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
                "input": [{"role": "user", "content": "Hello"}],
                "max_output_tokens": 4,
                "mimo_mtp_fastpath": "enabled",
            },
        }
    }

    with pytest.raises(ValidationError) as exc_info:
        TypeAdapter[TaskUnion](TaskUnion).validate_json(json.dumps(payload))

    _assert_validation_error_has_location_suffix(
        exc_info.value,
        error_type="model_type",
        location_suffix=("task_params", "mimo_mtp_fastpath"),
    )


def test_text_generation_task_params_model_validate_round_trips_without_mtp_key() -> None:
    """Backward compat: model_validate from dict without MTP key yields None MTP."""
    params = TextGenerationTaskParams.model_validate(
        {
            "model": MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
            "input": [{"role": "user", "content": "Hello"}],
            "max_output_tokens": 8,
        }
    )
    assert params.mimo_mtp_fastpath is None
    assert "mimo_mtp_fastpath" not in params.model_dump()


def test_text_generation_task_params_json_round_trip_preserves_absent_mtp() -> None:
    """Backward compat: JSON serialize then deserialize without MTP yields None MTP."""
    original = _base_text_generation_task_params()
    json_str = original.model_dump_json()
    restored = TextGenerationTaskParams.model_validate_json(json_str)
    assert restored.mimo_mtp_fastpath is None
    assert restored.model == original.model
    assert restored.max_output_tokens == original.max_output_tokens
    assert "mimo_mtp_fastpath" not in restored.model_dump()


def test_text_generation_task_params_model_copy_preserves_absent_mtp() -> None:
    """Backward compat: model_copy with unrelated update keeps MTP absent."""
    original = _base_text_generation_task_params()
    copied = original.model_copy(update={"temperature": 0.5})
    assert copied.mimo_mtp_fastpath is None
    assert copied.temperature == 0.5
    assert copied.model == original.model
    assert "mimo_mtp_fastpath" not in copied.model_dump()


def test_text_generation_task_params_default_and_explicit_none_produce_same_serialization() -> None:
    """Backward compat: default and explicit-None MTP produce identical output."""
    default_params = _base_text_generation_task_params()
    explicit_none_params = TextGenerationTaskParams(
        model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
        input=[InputMessage(role="user", content="Hello")],
        max_output_tokens=4,
        mimo_mtp_fastpath_params=None,
    )
    assert default_params.model_dump() == explicit_none_params.model_dump()
    assert default_params.model_dump_json() == explicit_none_params.model_dump_json()


def test_text_generation_task_params_model_fields_set_excludes_mtp_when_absent() -> None:
    """Backward compat: model_fields_set does not contain MTP keys when not provided."""
    params = _base_text_generation_task_params()
    assert "mimo_mtp_fastpath_params" not in params.model_fields_set
    assert "mimo_mtp_fastpath" not in params.model_fields_set


def test_task_deserialization_rejects_incorrect_mtp_field_types() -> None:
    payload: dict[str, object] = {
        "TextGeneration": {
            "task_id": "task-invalid-mtp-field-types",
            "task_status": "Pending",
            "instance_id": "instance-invalid-mtp-field-types",
            "command_id": "cmd-invalid-mtp-field-types",
            "task_params": {
                "model": MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
                "input": [{"role": "user", "content": "Hello"}],
                "max_output_tokens": 4,
                "mimo_mtp_fastpath": {
                    "enabled": "true",
                    "depth": "3",
                    "sidecar_path": 123,
                    "fail_closed": "false",
                },
            },
        }
    }

    with pytest.raises(ValidationError) as exc_info:
        TypeAdapter[TaskUnion](TaskUnion).validate_json(json.dumps(payload))

    validation_error = exc_info.value
    _assert_validation_error_has_location_suffix(
        validation_error,
        error_type="bool_type",
        location_suffix=("task_params", "mimo_mtp_fastpath", "enabled"),
    )
    _assert_validation_error_has_location_suffix(
        validation_error,
        error_type="int_type",
        location_suffix=("task_params", "mimo_mtp_fastpath", "depth"),
    )
    _assert_validation_error_has_location_suffix(
        validation_error,
        error_type="string_type",
        location_suffix=("task_params", "mimo_mtp_fastpath", "sidecar_path"),
    )
    _assert_validation_error_has_location_suffix(
        validation_error,
        error_type="bool_type",
        location_suffix=("task_params", "mimo_mtp_fastpath", "fail_closed"),
    )


class TestExplicitlyDisabledMtpSerializationIdenticalToDefaultNone:
    """AC 5.4.2: When MTP is explicitly disabled (enabled=False), the serialized
    output must be byte-for-byte/structurally identical to the default-None case.
    No divergent fields, no extra keys, no conditional branches."""

    def test_model_dump_keys_identical_disabled_vs_none(self) -> None:
        none_dump = _base_text_generation_task_params().model_dump()
        disabled_dump = _mtp_disabled_params().model_dump()
        assert set(none_dump.keys()) == set(disabled_dump.keys()), (
            f"Key sets differ: none={sorted(none_dump.keys())}, "
            f"disabled={sorted(disabled_dump.keys())}"
        )

    def test_model_dump_values_identical_disabled_vs_none(self) -> None:
        none_dump = _base_text_generation_task_params().model_dump()
        disabled_dump = _mtp_disabled_params().model_dump()
        assert none_dump == disabled_dump, (
            f"model_dump() not equal: none={none_dump}, disabled={disabled_dump}"
        )

    def test_model_dump_json_byte_identical_disabled_vs_none(self) -> None:
        none_json = _base_text_generation_task_params().model_dump_json()
        disabled_json = _mtp_disabled_params().model_dump_json()
        assert none_json == disabled_json, (
            f"model_dump_json() not byte-identical:\n"
            f"  none={none_json}\n  disabled={disabled_json}"
        )

    def test_no_mtp_key_in_disabled_model_dump(self) -> None:
        disabled_dump = _mtp_disabled_params().model_dump()
        assert "mimo_mtp_fastpath" not in disabled_dump, (
            f"mimo_mtp_fastpath key present in disabled dump: "
            f"{disabled_dump['mimo_mtp_fastpath']}"
        )
        assert "mimo_mtp_fastpath_params" not in disabled_dump, (
            "mimo_mtp_fastpath_params key present in disabled dump"
        )

    def test_no_mtp_key_in_disabled_model_dump_json(self) -> None:
        disabled_json = _mtp_disabled_params().model_dump_json()
        disabled_obj = cast(dict[str, object], json.loads(disabled_json))
        assert "mimo_mtp_fastpath" not in disabled_obj, (
            f"mimo_mtp_fastpath key present in disabled JSON: "
            f"{disabled_obj['mimo_mtp_fastpath']}"
        )
        assert "mimo_mtp_fastpath_params" not in disabled_obj, (
            "mimo_mtp_fastpath_params key present in disabled JSON"
        )

    def test_disabled_with_full_subfields_still_identical_to_none(self) -> None:
        """Even when disabled MTP params have depth/sidecar_path/fail_closed set,
        serialization must be identical to default-None (all sub-fields are elided)."""
        none_dump = _base_text_generation_task_params().model_dump()
        disabled_full_dump = _mtp_disabled_full_params().model_dump()
        assert none_dump == disabled_full_dump, (
            f"Full-disabled not equal to none: none={none_dump}, "
            f"disabled_full={disabled_full_dump}"
        )

    def test_disabled_with_full_subfields_json_byte_identical_to_none(self) -> None:
        none_json = _base_text_generation_task_params().model_dump_json()
        disabled_full_json = _mtp_disabled_full_params().model_dump_json()
        assert none_json == disabled_full_json, (
            f"Full-disabled JSON not byte-identical:\n"
            f"  none={none_json}\n  disabled={disabled_full_json}"
        )

    def test_command_level_serialization_disabled_identical_to_none(self) -> None:
        """At the Command level, disabled MTP and default-None produce identical
        task_params serialization."""
        none_cmd = TextGeneration(
            command_id=CommandId("cmd-none"),
            task_params=_base_text_generation_task_params(),
        )
        disabled_cmd = TextGeneration(
            command_id=CommandId("cmd-none"),
            task_params=_mtp_disabled_params(),
        )
        none_payload = _dumped_mapping(none_cmd.model_dump())
        disabled_payload = _dumped_mapping(disabled_cmd.model_dump())
        assert none_payload == disabled_payload, (
            "Command-level serialization differs between disabled and none"
        )

    def test_command_topic_payload_disabled_identical_to_none(self) -> None:
        """Through the COMMANDS topic serialize/deserialize round-trip, disabled MTP
        produces identical wire bytes as default-None."""
        none_cmd = TextGeneration(
            command_id=CommandId("cmd-topic-none"),
            task_params=_base_text_generation_task_params(),
        )
        disabled_cmd = TextGeneration(
            command_id=CommandId("cmd-topic-none"),
            task_params=_mtp_disabled_params(),
        )
        none_wire = COMMANDS.serialize(
            ForwarderCommand(
                origin=SystemId("system-test"),
                command=none_cmd,
            )
        )
        disabled_wire = COMMANDS.serialize(
            ForwarderCommand(
                origin=SystemId("system-test"),
                command=disabled_cmd,
            )
        )
        assert none_wire == disabled_wire, (
            f"Wire bytes differ:\n  none={none_wire}\n  disabled={disabled_wire}"
        )

    def test_task_level_serialization_disabled_identical_to_none(self) -> None:
        """At the Task level, disabled MTP and default-None produce identical
        serialization."""
        none_task = TextGenerationTask(
            task_id=TaskId("task-none"),
            task_status=TaskStatus.Running,
            instance_id=InstanceId("instance-none"),
            command_id=CommandId("cmd-none"),
            task_params=_base_text_generation_task_params(),
        )
        disabled_task = TextGenerationTask(
            task_id=TaskId("task-none"),
            task_status=TaskStatus.Running,
            instance_id=InstanceId("instance-none"),
            command_id=CommandId("cmd-none"),
            task_params=_mtp_disabled_params(),
        )
        assert none_task.model_dump() == disabled_task.model_dump(), (
            "Task-level model_dump differs between disabled and none"
        )
        assert none_task.model_dump_json() == disabled_task.model_dump_json(), (
            "Task-level model_dump_json differs between disabled and none"
        )

    def test_no_divergent_fields_or_conditional_branches(self) -> None:
        """Exhaustive check: every field in the disabled dump exists in the none
        dump with the same value and type; no extra keys leak through."""
        none_dump = _base_text_generation_task_params().model_dump()
        disabled_dump = _mtp_disabled_params().model_dump()
        for key in disabled_dump:
            assert key in none_dump, f"Extra key in disabled dump: {key}"
            assert type(disabled_dump[key]) is type(none_dump[key]), (
                f"Type mismatch for key {key}: "
                f"none={type(none_dump[key]).__name__}, "
                f"disabled={type(disabled_dump[key]).__name__}"
            )
            assert disabled_dump[key] == none_dump[key], (
                f"Value mismatch for key {key}: "
                f"none={none_dump[key]}, disabled={disabled_dump[key]}"
            )
        for key in none_dump:
            assert key in disabled_dump, f"Missing key in disabled dump: {key}"

    def test_enabled_mtp_still_serializes(self) -> None:
        """Guard: enabled MTP must still serialize the mimo_mtp_fastpath key.
        This ensures the disabled-omission is not a blanket removal."""
        enabled_params = TextGenerationTaskParams(
            model=MIMO_V25_PRO_KERNELPOOL_6BIT_MODEL_ID,
            input=[InputMessage(role="user", content="Hello")],
            max_output_tokens=4,
            mimo_mtp_fastpath_params=MimoMtpFastpathParams(enabled=True, depth=2),
        )
        enabled_dump = enabled_params.model_dump()
        assert "mimo_mtp_fastpath" in enabled_dump, (
            "mimo_mtp_fastpath key missing from enabled MTP serialization"
        )
        assert enabled_dump["mimo_mtp_fastpath"]["enabled"] is True
