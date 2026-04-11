from exo.shared.types.tasks import TaskId
from exo.worker.engines.mlx.utils_mlx import (
    decode_task_id_bytes,
    encode_task_id_bytes,
)


def test_task_id_round_trips_through_byte_encoding() -> None:
    task_id = TaskId("12345678-1234-1234-1234-123456789abc")

    assert decode_task_id_bytes(encode_task_id_bytes(task_id)) == task_id


def test_task_id_decode_tolerates_widened_byte_values() -> None:
    task_id = TaskId("12345678-1234-1234-1234-123456789abc")
    widened = [byte + 256 for byte in encode_task_id_bytes(task_id)]

    assert decode_task_id_bytes(widened) == task_id
