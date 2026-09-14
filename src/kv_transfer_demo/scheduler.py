"""A single-threaded event scheduler with immutable records of executed events."""

from dataclasses import dataclass
from collections.abc import Iterable

from .protocol import ProtocolError, SlotRef, TransferManager


@dataclass(frozen=True)
class Event:
    action: str
    transfer_id: str
    piece: int | None = None


@dataclass(frozen=True)
class RequestEvent:
    """A request-wide cancel or retire action, independent of any one transfer."""

    action: str
    request_id: str


@dataclass(frozen=True)
class TraceEntry:
    event: Event | RequestEvent
    request_id: str
    block: int | None
    staging: SlotRef | None
    destination: SlotRef | None


class Scheduler:
    """Run caller-selected events; failed preconditions stop execution immediately."""

    def __init__(self, manager: TransferManager):
        self.manager = manager
        self.trace: list[TraceEntry] = []

    def run(self, events: Iterable[Event | RequestEvent]) -> None:
        for event in events:
            if isinstance(event, RequestEvent):
                if event.action == "cancel":
                    self.manager.cancel_request(event.request_id)
                elif event.action == "retire":
                    self.manager.release_request(event.request_id)
                else:
                    raise ProtocolError(f"Unknown request action: {event.action}")
                self.trace.append(TraceEntry(event, event.request_id, None, None, None))
                continue
            self.manager.step(event.action, event.transfer_id, event.piece)
            transfer = self.manager.transfers[event.transfer_id]
            self.trace.append(TraceEntry(event, transfer.request_id, transfer.block,
                                         transfer.staging, transfer.destination))
