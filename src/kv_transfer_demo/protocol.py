"""Deterministic staged copies and paged KV storage, with checked access rules.

All accesses go through this interface. Reservations are exclusive and more
conservative than production reader/writer pins. Cancellation drains submitted
stages before storage can be retired; it does not abort physical copies.
"""

from dataclasses import dataclass, field

import numpy as np

from .attention import KVCache


class ProtocolError(RuntimeError):
    """An event or memory access is not enabled in the current state."""


@dataclass(frozen=True)
class SlotRef:
    """Identify one allocation of a physical block within a manager's pools.

    pool names the storage region ("staging" or "destination").
    slot is the zero-based physical block index in that pool's K/V arrays,
    not the request's logical block index. generation is the allocation
    counter for that slot, incremented by BlockPool.allocate() on every use.

    For example, SlotRef("staging", 0, 1) refers to the first allocation of
    staging slot 0. After release and reallocation, its reference becomes
    SlotRef("staging", 0, 2). BlockPool.check() rejects the old reference,
    preventing stale accesses or releases from affecting the new allocation.

    This immutable reference contains no tensor data and does not itself
    reserve storage or guarantee readiness. BlockPool tracks reservations;
    TransferManager tracks publication. Generation checks reject stale
    bookkeeping but cannot stop an already-issued physical copy.
    """

    pool: str
    slot: int
    generation: int


class BlockPool:
    """Finite physical blocks; allocation generations reject stale references."""

    def __init__(self, name: str, capacity: int, block_size: int,
                 qk_dim: int, value_dim: int):
        if min(capacity, block_size, qk_dim, value_dim) < 1:
            raise ValueError("Pool dimensions must be positive")
        self.name = name
        self.block_size = block_size
        self.keys = np.full((capacity, block_size, qk_dim), np.nan)
        self.values = np.full((capacity, block_size, value_dim), np.nan)
        self._generation = [0] * capacity
        self._allocated = [False] * capacity
        self._reservation: list[str | None] = [None] * capacity

    @property
    def free_slots(self) -> list[int]:
        return [i for i, allocated in enumerate(self._allocated) if not allocated]

    def allocate(self, slot: int | None = None) -> SlotRef:
        if slot is None:
            if not self.free_slots:
                raise ProtocolError(f"{self.name} pool is full")
            slot = self.free_slots[0]
        if slot not in self.free_slots:
            raise ProtocolError(f"{self.name} slot {slot} is unavailable")
        self._generation[slot] += 1
        self._allocated[slot] = True
        self.keys[slot].fill(np.nan)
        self.values[slot].fill(np.nan)
        return SlotRef(self.name, slot, self._generation[slot])

    def check(self, ref: SlotRef, owner: str | None = None) -> None:
        if (ref.pool != self.name or not 0 <= ref.slot < len(self._allocated)
                or not self._allocated[ref.slot]
                or self._generation[ref.slot] != ref.generation):
            raise ProtocolError("Stale or invalid slot reference")
        if self._reservation[ref.slot] != owner:
            raise ProtocolError("Slot is reserved for a different access")

    def reserve(self, ref: SlotRef, owner: str) -> None:
        self.check(ref)
        self._reservation[ref.slot] = owner

    def unreserve(self, ref: SlotRef, owner: str) -> None:
        self.check(ref, owner)
        self._reservation[ref.slot] = None

    def release(self, ref: SlotRef) -> None:
        self.check(ref)
        self._allocated[ref.slot] = False

    def write_piece(self, ref: SlotRef, piece: int, data: KVCache,
                    owner: str | None = None) -> None:
        self.check(ref, owner)
        if (not 0 <= piece < self.block_size
                or data.keys.shape != self.keys.shape[2:]
                or data.values.shape != self.values.shape[2:]):
            raise ProtocolError("Invalid piece index or K/V dimensions")
        self.keys[ref.slot, piece] = data.keys
        self.values[ref.slot, piece] = data.values

    def read(self, ref: SlotRef, owner: str | None = None) -> KVCache:
        self.check(ref, owner)
        return KVCache(self.keys[ref.slot].copy(), self.values[ref.slot].copy())


@dataclass
class Request:
    """Track one request's producer snapshot and consumer KV storage.

    source is the prompt K/V snapshot copied and made read-only by
    TransferManager.add_request(). Transfers read it; decoding does not extend it.
    length counts valid tokens in the consumer context, initially the prompt
    length and incremented as generated tokens' K/V are appended. It lets
    gather() exclude unused rows in a partially filled final block.

    blocks is the consumer's logical-to-physical block table: blocks[i] is
    the destination SlotRef for logical block i. For example, request A's
    slots [5, 1, 7] hold token rows 0-1, 2-3, and 4-5 respectively. Decoding
    appends K/V directly to destination storage (representing TPU HBM),
    filling the last block or adding a new block; it does not use staging.

    ready contains logical block indices, not physical slot numbers. For A,
    {2} means the block in slot 7 is published. gather() requires every
    context block to be ready. The default factory gives each request its
    own initially empty set.

    cancelled records withdrawn interest. Cancellation clears ready and blocks
    later gather/append operations; source and blocks stay owned until submitted
    transfers drain and release_request() can retire the request safely.
    """

    source: KVCache
    length: int
    blocks: list[SlotRef]
    ready: set[int] = field(default_factory=set)
    cancelled: bool = False


@dataclass
class Transfer:
    """Track one logical KV block moving through staging to its destination.

    request_id identifies the Request in TransferManager.requests; multiple
    transfers can belong to that request. block is its logical block index.
    For example, transfer A2 has request_id="A", block=2, and a destination
    SlotRef for physical slot 7. staging is assigned by the reserve event.
    pieces counts valid token rows, each containing both K and V; a partial
    final block can have fewer pieces than the pool's block size.

    TransferManager.step() advances the copy phase through explicit events:
        planned -> reserve -> producer
        producer -> complete_staging -> staged
        staged -> start_destination -> destination
        destination -> complete_destination -> complete
    In the producer phase, producer_pieces records row indices copied into
    staging. In the destination phase, destination_pieces records row indices
    copied into destination storage. Each completion event requires all its
    pieces; copying the last row does not itself advance the phase.

    After complete_destination removes both reservations and sets the phase
    to complete, three separate flags track the remaining lifecycle:
        staging_released: staging was returned, or never allocated before discard.
        notified: notify recorded an Outcome for client polling.
        published: publish added this logical block to the request's ready set.
    Publication requires notification, and notification requires completion.
    Staging release requires completion but is independent of notification
    and publication. The demo releases staging first, reuses it, then notifies
    and publishes the old transfer. Its retained staging reference therefore
    describes the old allocation and may no longer be valid for memory access.

    Cancellation marks this record permanently, even after request retirement.
    A planned or staged transfer becomes discarded; a producer-stage copy
    drains before becoming discarded. A submitted destination copy drains
    to complete. Discarded means no copy remains that can access storage,
    not that destination data is valid. Cancelled transfers cannot publish.
    If publication preceded cancellation, published remains historical metadata;
    cancellation clears Request.ready and withdraws current consumer access.

    This record stores progress; TransferManager.step() enforces the rules.
    """

    request_id: str
    block: int
    destination: SlotRef
    pieces: int
    staging: SlotRef | None = None
    phase: str = "planned"
    producer_pieces: set[int] = field(default_factory=set)
    destination_pieces: set[int] = field(default_factory=set)
    notified: bool = False
    published: bool = False
    staging_released: bool = False
    cancelled: bool = False


@dataclass(frozen=True)
class Outcome:
    """Historical notification, not a storage reservation or readiness grant.

    status is completed or cancelled at delivery time. A later cancellation
    does not rewrite previously delivered outcomes. A cancelled outcome may
    refer to storage already retired and reallocated; do not dereference it.
    """

    transfer_id: str
    request_id: str
    block: int
    destination: SlotRef
    status: str = "completed"


class TransferManager:
    """Owns request block tables and advances only explicitly selected events."""

    def __init__(self, *, block_size: int = 2, staging_slots: int = 2,
                 destination_slots: int = 8, qk_dim: int = 8, value_dim: int = 8):
        self.block_size = block_size
        self.staging = BlockPool("staging", staging_slots, block_size, qk_dim, value_dim)
        self.destination = BlockPool("destination", destination_slots, block_size,
                                     qk_dim, value_dim)
        self.requests: dict[str, Request] = {}
        self.transfers: dict[str, Transfer] = {}
        self._outcomes: list[Outcome] = []
        self._request_ids: set[str] = set()

    def add_request(self, request_id: str, cache: KVCache,
                    slots: list[int]) -> None:
        """Snapshot producer data and allocate an explicit logical block table.

        Request and transfer IDs are not reused during a manager's lifetime.
        """
        if (cache.keys.ndim != 2 or cache.values.ndim != 2
                or len(cache.keys) == 0 or len(cache.keys) != len(cache.values)
                or cache.keys.shape[1] != self.destination.keys.shape[2]
                or cache.values.shape[1] != self.destination.values.shape[2]):
            raise ProtocolError("Invalid producer K/V shape")
        count = (len(cache.keys) + self.block_size - 1) // self.block_size
        if (request_id in self._request_ids
                or len(slots) != count
                or len(set(slots)) != count
                or any(slot not in self.destination.free_slots for slot in slots)):
            raise ProtocolError("Request needs a unique, available slot per logical block")
        source = KVCache(cache.keys.copy(), cache.values.copy())
        source.keys.flags.writeable = False
        source.values.flags.writeable = False
        blocks = [self.destination.allocate(slot) for slot in slots]
        self.requests[request_id] = Request(source, len(source.keys), blocks)
        self._request_ids.add(request_id)

    def plan(self, transfer_id: str, request_id: str, block: int) -> None:
        request = self.requests.get(request_id)
        if (transfer_id in self.transfers or request is None or request.cancelled
                or not 0 <= block < len(request.blocks) or block in request.ready
                or any(t.request_id == request_id and t.block == block
                       for t in self.transfers.values())):
            raise ProtocolError("Invalid or duplicate transfer plan")
        pieces = min(self.block_size, len(request.source.keys) - block * self.block_size)
        if pieces <= 0:
            raise ProtocolError("No producer data for this block")
        self.transfers[transfer_id] = Transfer(request_id, block,
                                              request.blocks[block], pieces)

    def step(self, action: str, transfer_id: str, piece: int | None = None) -> None:
        """Execute one enabled event; invalid events fail instead of being skipped."""
        if transfer_id not in self.transfers:
            raise ProtocolError("Unknown transfer")
        transfer = self.transfers[transfer_id]
        if action not in ("copy_to_staging", "copy_to_destination") and piece is not None:
            raise ProtocolError("Only copy events accept a piece index")
        # Cancelled tombstones can report a drained result after request retirement.
        # This path only records metadata: it must not touch either physical slot.
        if action == "notify" and transfer.cancelled:
            if transfer.phase not in ("complete", "discarded") or transfer.notified:
                raise ProtocolError("Cancelled completion notification is not enabled")
            self._outcomes.append(Outcome(transfer_id, transfer.request_id,
                                          transfer.block, transfer.destination, "cancelled"))
            transfer.notified = True
            return
        request = self.requests.get(transfer.request_id)
        if (request is None or request.blocks[transfer.block] != transfer.destination):
            raise ProtocolError("Transfer no longer owns its destination mapping")

        if action == "reserve":
            if request.cancelled or transfer.phase != "planned":
                raise ProtocolError("Transfer already reserved")
            self.destination.check(transfer.destination)
            staging = self.staging.allocate()
            self.staging.reserve(staging, transfer_id)
            self.destination.reserve(transfer.destination, transfer_id)
            transfer.staging = staging
            transfer.phase = "producer"
        elif action in ("copy_to_staging", "copy_to_destination"):
            to_staging = action == "copy_to_staging"
            phase = "producer" if to_staging else "destination"
            copied = transfer.producer_pieces if to_staging else transfer.destination_pieces
            if (transfer.phase != phase or piece is None
                    or not 0 <= piece < transfer.pieces or piece in copied):
                raise ProtocolError("Copy piece is not enabled")
            if to_staging:
                row = transfer.block * self.block_size + piece
                data = KVCache(request.source.keys[row], request.source.values[row])
                self.staging.write_piece(transfer.staging, piece, data, transfer_id)
            else:
                staged = self.staging.read(transfer.staging, transfer_id)
                data = KVCache(staged.keys[piece], staged.values[piece])
                self.destination.write_piece(transfer.destination, piece, data, transfer_id)
            copied.add(piece)
        elif action == "complete_staging":
            if transfer.phase != "producer" or len(transfer.producer_pieces) != transfer.pieces:
                raise ProtocolError("Producer-to-staging copy is incomplete")
            if transfer.cancelled:
                self._discard(transfer_id, transfer)
            else:
                transfer.phase = "staged"
        elif action == "start_destination":
            if request.cancelled or transfer.phase != "staged":
                raise ProtocolError("Destination stage cannot start")
            transfer.phase = "destination"
        elif action == "complete_destination":
            if (transfer.phase != "destination"
                    or len(transfer.destination_pieces) != transfer.pieces):
                raise ProtocolError("Staging-to-destination copy is incomplete")
            self.staging.unreserve(transfer.staging, transfer_id)
            self.destination.unreserve(transfer.destination, transfer_id)
            transfer.phase = "complete"
        elif action == "release_staging":
            if transfer.phase not in ("complete", "discarded") or transfer.staging_released:
                raise ProtocolError("Staging cannot be released yet or was already released")
            self.staging.release(transfer.staging)
            transfer.staging_released = True
        elif action == "notify":
            if transfer.phase != "complete" or transfer.notified:
                raise ProtocolError("Completion notification is not enabled")
            self.destination.check(transfer.destination)
            self._outcomes.append(Outcome(transfer_id, transfer.request_id,
                                          transfer.block, transfer.destination))
            transfer.notified = True
        elif action == "publish":
            if request.cancelled or not transfer.notified or transfer.published:
                raise ProtocolError("Publication requires a delivered completion")
            self.destination.check(transfer.destination)
            request.ready.add(transfer.block)
            transfer.published = True
        else:
            raise ProtocolError(f"Unknown action: {action}")

    def poll(self) -> tuple[Outcome, ...]:
        """Consume recorded outcomes without copying, publishing, or releasing data."""
        outcomes = tuple(self._outcomes)
        self._outcomes.clear()
        return outcomes

    def gather(self, request_id: str) -> KVCache:
        """Read ready blocks in logical order into an independent attention snapshot."""
        request = self.requests[request_id]
        if request.cancelled or request.ready != set(range(len(request.blocks))):
            raise ProtocolError("Context is not ready")
        blocks = [self.destination.read(ref) for ref in request.blocks]
        return KVCache(np.concatenate([block.keys for block in blocks])[:request.length],
                       np.concatenate([block.values for block in blocks])[:request.length])

    def append(self, request_id: str, data: KVCache) -> None:
        """Append one locally computed token's K/V, allocating a block on demand."""
        request = self.requests[request_id]
        if (data.keys.shape != (1, self.destination.keys.shape[2])
                or data.values.shape != (1, self.destination.values.shape[2])):
            raise ProtocolError("Append requires exactly one token's K/V")
        if request.cancelled or request.ready != set(range(len(request.blocks))):
            raise ProtocolError("Cannot append to an unavailable context")
        piece = request.length % self.block_size
        if piece == 0:
            request.blocks.append(self.destination.allocate())
        ref = request.blocks[-1]
        self.destination.write_piece(ref, piece, KVCache(data.keys[0], data.values[0]))
        request.ready.add(len(request.blocks) - 1)
        request.length += 1

    def _discard(self, transfer_id: str, transfer: Transfer) -> None:
        """Settle a cancelled transfer only when no submitted copy remains."""
        if transfer.staging is not None:
            self.staging.unreserve(transfer.staging, transfer_id)
            self.destination.unreserve(transfer.destination, transfer_id)
        else:
            transfer.staging_released = True  # No staging was ever allocated.
        transfer.phase = "discarded"

    def cancel_request(self, request_id: str) -> None:
        """Withdraw interest, block new stages, and drain already-submitted work.

        Cancellation stops new copy stages and prevents further reads for the
        request. Any stage already started must finish before its memory can
        be released.
        """
        request = self.requests.get(request_id)
        if request is None or request.cancelled:
            raise ProtocolError("Unknown or already cancelled request")
        request.cancelled = True
        request.ready.clear()
        for transfer_id, transfer in self.transfers.items():
            if transfer.request_id != request_id:
                continue
            transfer.cancelled = True
            if transfer.phase in ("planned", "staged"):
                self._discard(transfer_id, transfer)

    def release_request(self, request_id: str) -> None:
        """Retire a published request, or a cancelled request whose work drained.

        Cancellation alone never grants retirement. Every stage must be settled
        and staging returned first. Cancelled notification delivery may lag
        retirement; retained transfer records handle it without accessing memory.
        """
        request = self.requests.get(request_id)
        if request is None:
            raise ProtocolError("Unknown or already retired request")
        related = [t for t in self.transfers.values() if t.request_id == request_id]
        if request.cancelled:
            settled = all(t.phase in ("complete", "discarded") and t.staging_released
                          for t in related)
        else:
            settled = (request.ready == set(range(len(request.blocks)))
                       and all(t.published and t.staging_released for t in related))
        if not settled:
            raise ProtocolError("Request still has outstanding work")
        for ref in request.blocks:
            self.destination.check(ref)
        for ref in request.blocks:
            self.destination.release(ref)
        # Keep IDs and transfer metadata, not producer arrays, after retirement.
        del self.requests[request_id]
