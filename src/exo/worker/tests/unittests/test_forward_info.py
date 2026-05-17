import pytest

from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.events import Event, IndexedEvent, NodeGatheredInfo
from exo.shared.types.profiling import NetworkInterfaceInfo
from exo.utils.channels import channel
from exo.utils.info_gatherer.info_gatherer import GatheredInfo, NodeNetworkInterfaces
from exo.worker.main import Worker
from exo.worker.tests.constants import NODE_A


@pytest.mark.asyncio
async def test_forward_info_skips_duplicate_payloads() -> None:
    event_send, event_recv = channel[Event]()
    command_send, _command_recv = channel[ForwarderCommand]()
    download_send, _download_recv = channel[ForwarderDownloadCommand]()
    _event_send_unused, event_recv_unused = channel[IndexedEvent]()
    info_send, info_recv = channel[GatheredInfo]()

    worker = Worker(
        NODE_A,
        api_port=52415,
        event_receiver=event_recv_unused,
        event_sender=event_send,
        command_sender=command_send,
        download_command_sender=download_send,
    )

    info = NodeNetworkInterfaces(
        ifaces=[
            NetworkInterfaceInfo(name="en0", ip_address="192.168.1.2"),
        ]
    )

    await info_send.send(info)
    await info_send.send(info)
    await info_send.send(
        NodeNetworkInterfaces(
            ifaces=[
                NetworkInterfaceInfo(name="en0", ip_address="192.168.1.3"),
            ]
        )
    )
    info_send.close()

    await worker.forward_info(info_recv)

    first = await event_recv.receive()
    second = await event_recv.receive()

    assert isinstance(first, NodeGatheredInfo)
    assert isinstance(second, NodeGatheredInfo)
    assert first.info == info
    assert second.info != info
    assert event_recv.collect() == []
