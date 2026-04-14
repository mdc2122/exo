import exo.worker.engines.mlx.utils_mlx as utils_mlx
from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.tasks import TaskId, TextGeneration
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.shared.types.worker.instances import InstanceId
from exo.worker.runner.llm_inference.batch_generator import advance_coordination_counter


def _task(task_id: str) -> TextGeneration:
    return TextGeneration(
        task_id=TaskId(task_id),
        command_id=CommandId("cmd-1"),
        instance_id=InstanceId("instance-1"),
        task_params=TextGenerationTaskParams(
            model=ModelId("mlx-community/GLM-5.1-8b-crit-6b-exp"),
            input=[InputMessage(role="user", content="hello")],
            max_output_tokens=8,
            temperature=0.0,
        ),
    )


def test_advance_coordination_counter_waits_until_threshold() -> None:
    counter = 0

    counter, should_check = advance_coordination_counter(counter, 3)
    assert counter == 1
    assert should_check is False

    counter, should_check = advance_coordination_counter(counter, 3)
    assert counter == 2
    assert should_check is False

    counter, should_check = advance_coordination_counter(counter, 3)
    assert counter == 0
    assert should_check is True


def test_advance_coordination_counter_clamps_non_positive_threshold() -> None:
    counter, should_check = advance_coordination_counter(0, 0)
    assert counter == 0
    assert should_check is True


def test_mx_all_gather_tasks_returns_local_tasks_without_group() -> None:
    task = _task("12345678-1234-1234-1234-123456789abc")

    agreed, different = utils_mlx.mx_all_gather_tasks([task], group=None)

    assert agreed == [task]
    assert different == []
