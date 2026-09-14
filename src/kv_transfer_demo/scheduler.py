"""A single-threaded event scheduler with immutable records of executed events."""

from dataclasses import dataclass
from collections.abc import Iterable

from .protocol import SlotRef, TransferManager


@dataclass(frozen=True)
class Event:
    action: str
    transfer_id: str
    piece: int | None = None


@dataclass(frozen=True)
class TraceEntry:
    event: Event
    request_id: str
    block: int
    staging: SlotRef | None
    destination: SlotRef


class Scheduler:
    """Run caller-selected events; failed preconditions stop execution immediately."""

    def __init__(self, manager: TransferManager):
        self.manager = manager
        self.trace: list[TraceEntry] = []

    def run(self, events: Iterable[Event]) -> None:
        for event in events:
            self.manager.step(event.action, event.transfer_id, event.piece)
            transfer = self.manager.transfers[event.transfer_id]
            self.trace.append(TraceEntry(event, transfer.request_id, transfer.block,
                                         transfer.staging, transfer.destination))
