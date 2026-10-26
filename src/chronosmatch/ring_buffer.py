"""Fixed-capacity Single-Producer Single-Consumer (SPSC) Ring Buffer for ChronosMatch.

Provides ultra-low latency, deterministic FIFO queuing for 64-byte binary order records
backed by memory-mapped shared memory (`SharedMemoryRegion`).

Architecture & Design Principles:
---------------------------------
1. Fixed-Capacity & Fixed-Size Binary Records:
   - Allocates contiguous memory for N records of fixed size (default: ORDER_STRUCT_SIZE = 64 bytes).
   - Backed by `SharedMemoryRegion` (file-backed or anonymous paging-file memory).

2. Monotonic 64-bit Sequence Counters (No Modulo Ambiguity):
   - `head` is the monotonically increasing write sequence counter (total records put).
   - `tail` is the monotonically increasing read sequence counter (total records got).
   - `occupied = head - tail`.
   - Empty state: `occupied == 0` (i.e. `head == tail`).
   - Full state: `occupied == capacity` (i.e. `head - tail == capacity`).
   - This eliminates the classic ring-buffer ambiguity where head == tail can mean either full or empty.
   - Slot mapping uses bitwise masking `head & (capacity - 1)` for power-of-two capacities,
     or `head % capacity` for arbitrary positive integer capacities.

3. Cache-Line Separation & Memory Ordering:
   - Header metadata, producer `head`, consumer `tail`, and record slots are aligned to 64-byte
     CPU cache-line boundaries (Header = 192 bytes = 3 cache lines).
   - Cache line 0 (offsets 0..63): Magic, version, record_size, capacity, flags, reserved.
   - Cache line 1 (offsets 64..127): `head` (producer write pointer) + 56 bytes padding.
   - Cache line 2 (offsets 128..191): `tail` (consumer read pointer) + 56 bytes padding.
   - Cache line 3+ (offsets 192..): Contiguous record slots.
   - Producer writes slot payload *before* advancing `head` in shared memory.
   - Consumer reads slot payload *before* advancing `tail` in shared memory.
   - No unread orders are ever silently overwritten.

4. Thread & Multi-Process Coordination:
   - In-process concurrent threads coordinate via `threading.Condition` (`not_full`, `not_empty`)
     for immediate wakeups without wasteful CPU spin loops.
   - Cross-process instances coordinate directly via the shared-memory header counters (`head` and `tail`)
     with adaptive polling wait support.
   - Explicit `put()` / `get()` operations support both blocking (with configurable timeouts) and
     non-blocking (`block=False` or `put_nowait()` / `get_nowait()`) modes.
"""

from __future__ import annotations

from pathlib import Path
import struct
import threading
import time
from typing import Any, Iterator, Self

from chronosmatch.binary_protocol import (
    ORDER_STRUCT,
    ORDER_STRUCT_SIZE,
    _validate_order_fields,
    decode_order,
    decode_order_from,
    encode_order,
    encode_order_into,
)
from chronosmatch.models import Order
from chronosmatch.shared_memory import (
    MalformedRecordError,
    SharedMemoryClosedError,
    SharedMemoryError,
    SharedMemoryReadOnlyError,
    SharedMemoryRecordSizeError,
    SharedMemoryRegion,
)

# ==============================================================================
# Header Layout Specification & Offsets
# ==============================================================================

RING_MAGIC: bytes = b"CMRING01"  # 8 bytes: ChronosMatch Ring Buffer v1
RING_VERSION: int = 1

HEADER_META_FORMAT: str = "<8sIIQQ32s"
HEADER_META_STRUCT: struct.Struct = struct.Struct(HEADER_META_FORMAT)
HEADER_META_SIZE: int = 64  # Cache Line 0 (64 bytes)

COUNTER_FORMAT: str = "<Q"
COUNTER_STRUCT: struct.Struct = struct.Struct(COUNTER_FORMAT)
COUNTER_SIZE: int = 8

# Byte offsets in the shared-memory mapping
MAGIC_OFFSET: int = 0
VERSION_OFFSET: int = 8
RECORD_SIZE_OFFSET: int = 12
CAPACITY_OFFSET: int = 16
FLAGS_OFFSET: int = 24

# Cache Line 1 (offsets 64..127): Producer Write Pointer
HEAD_OFFSET: int = 64

# Cache Line 2 (offsets 128..191): Consumer Read Pointer
TAIL_OFFSET: int = 128

# Cache Line 3+ (offsets 192..): Record Slots Start
HEADER_SIZE: int = 192
SLOTS_OFFSET: int = HEADER_SIZE

UINT64_MAX: int = (1 << 64) - 1
UINT64_MASK: int = UINT64_MAX


# ==============================================================================
# Ring Buffer Exceptions
# ==============================================================================


class RingBufferError(SharedMemoryError):
    """Base exception for all ChronosMatch ring buffer operations."""


class RingBufferFullError(RingBufferError):
    """Raised when an operation cannot be completed because the ring buffer is full."""


class RingBufferEmptyError(RingBufferError):
    """Raised when an operation cannot be completed because the ring buffer is empty."""


class RingBufferClosedError(RingBufferError, SharedMemoryClosedError):
    """Raised when an operation is attempted on a closed ring buffer."""


class RingBufferConfigError(RingBufferError, ValueError):
    """Raised when ring buffer configuration parameters are invalid."""


# ==============================================================================
# RingBuffer Implementation
# ==============================================================================


class RingBuffer:
    """Fixed-capacity, FIFO circular ring buffer for binary order records.

    Backed by a `SharedMemoryRegion`, coordinating producers and consumers via
    cache-line separated monotonic counters with full wraparound support.
    """

    def __init__(
        self,
        capacity: int | None = None,
        record_size: int = ORDER_STRUCT_SIZE,
        *,
        filepath: str | Path | None = None,
        region: SharedMemoryRegion | None = None,
        create: bool = True,
        read_only: bool = False,
        lock: Any = None,
    ) -> None:
        """Initialize a fixed-capacity ring buffer.

        Args:
            capacity: Maximum number of records the buffer can hold. Required if creating.
            record_size: Size in bytes of each record (default: ORDER_STRUCT_SIZE = 64).
            filepath: Path to backing file (None for anonymous in-memory mapping).
            region: Existing SharedMemoryRegion instance to wrap.
            create: If True, creates and initializes a new ring buffer header and storage.
                    If False, opens and connects to an existing ring buffer.
            read_only: If True, maps region in read-only mode (read/peek only).
            lock: Optional threading.Lock or multiprocessing.Lock for in-process sync.

        Raises:
            RingBufferConfigError: If capacity or record_size is invalid.
            TypeError: If arguments have invalid types.
            FileNotFoundError: If opening an existing file that does not exist.
        """
        self._closed: bool = False
        self._read_only: bool = read_only
        self._owns_region: bool = region is None

        # Thread synchronization primitives
        self._lock = lock if lock is not None else threading.Lock()
        self._not_full = threading.Condition(self._lock)
        self._not_empty = threading.Condition(self._lock)

        if region is not None:
            if not isinstance(region, SharedMemoryRegion):
                raise TypeError(f"region must be a SharedMemoryRegion, got {type(region).__name__}")
            self._region = region
            self._read_only = region.is_read_only

            if create:
                self._validate_config(capacity, record_size)
                assert capacity is not None
                self._capacity: int = capacity
                self._record_size: int = record_size
                required_size = HEADER_SIZE + self._capacity * self._record_size
                if self._region.size < required_size:
                    raise RingBufferConfigError(
                        f"Region size {self._region.size} is smaller than required size {required_size}"
                    )
                self._init_header()
            else:
                self._load_header(expected_capacity=capacity, expected_record_size=record_size)
        elif create:
            self._validate_config(capacity, record_size)
            assert capacity is not None
            self._capacity = capacity
            self._record_size = record_size
            total_size = HEADER_SIZE + self._capacity * self._record_size

            if filepath is not None:
                self._region = SharedMemoryRegion.create(filepath=filepath, size=total_size)
            else:
                self._region = SharedMemoryRegion.create_anonymous(size=total_size)

            self._init_header()
        else:
            if filepath is None:
                raise RingBufferConfigError("filepath is required when opening an existing ring buffer")
            self._region = SharedMemoryRegion.open(filepath=filepath, read_only=read_only)
            self._load_header(expected_capacity=capacity, expected_record_size=record_size)

        # Optimization: bitmasking if capacity is power-of-two
        self._is_power_of_two: bool = (self._capacity & (self._capacity - 1)) == 0 and self._capacity > 0
        self._mask: int | None = (self._capacity - 1) if self._is_power_of_two else None

    # --------------------------------------------------------------------------
    # Validation & Header Management
    # --------------------------------------------------------------------------

    @staticmethod
    def _validate_config(capacity: Any, record_size: Any) -> None:
        """Validate capacity and record_size configuration."""
        if capacity is None:
            raise RingBufferConfigError("capacity is required when creating a ring buffer")
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError(f"capacity must be an integer, got {type(capacity).__name__}")
        if capacity <= 0:
            raise RingBufferConfigError(f"capacity must be strictly positive (> 0), got {capacity}")

        if isinstance(record_size, bool) or not isinstance(record_size, int):
            raise TypeError(f"record_size must be an integer, got {type(record_size).__name__}")
        if record_size <= 0:
            raise RingBufferConfigError(f"record_size must be strictly positive (> 0), got {record_size}")

    def _init_header(self) -> None:
        """Write the initial ring buffer metadata and zero counters to shared memory."""
        meta_bytes = HEADER_META_STRUCT.pack(
            RING_MAGIC,
            RING_VERSION,
            self._record_size,
            self._capacity,
            0,  # flags
            b"\x00" * 32,  # reserved
        )
        self._region.write_bytes(MAGIC_OFFSET, meta_bytes)
        # Initialize head and tail to 0
        COUNTER_STRUCT.pack_into(self._region.mmap, HEAD_OFFSET, 0)
        COUNTER_STRUCT.pack_into(self._region.mmap, TAIL_OFFSET, 0)
        self._region.flush()

    def _load_header(self, expected_capacity: int | None, expected_record_size: int | None) -> None:
        """Read and validate existing header metadata from shared memory."""
        if self._region.size < HEADER_SIZE:
            raise RingBufferConfigError(
                f"Region size {self._region.size} is too small to contain a valid ring buffer header ({HEADER_SIZE} bytes)"
            )
        meta_bytes = self._region.read_bytes(MAGIC_OFFSET, HEADER_META_SIZE)
        magic, version, rec_size, cap, flags, _ = HEADER_META_STRUCT.unpack(meta_bytes)

        if magic != RING_MAGIC:
            raise RingBufferConfigError(
                f"Invalid ring buffer magic: expected {RING_MAGIC!r}, got {magic!r}"
            )
        if version != RING_VERSION:
            raise RingBufferConfigError(
                f"Unsupported ring buffer version: {version} (expected {RING_VERSION})"
            )

        if expected_capacity is not None and expected_capacity != cap:
            raise RingBufferConfigError(
                f"Capacity mismatch: expected {expected_capacity}, but backing buffer has {cap}"
            )
        if expected_record_size is not None and expected_record_size != rec_size:
            raise RingBufferConfigError(
                f"Record size mismatch: expected {expected_record_size}, but backing buffer has {rec_size}"
            )

        self._capacity = cap
        self._record_size = rec_size

    def _check_closed(self) -> None:
        """Verify the ring buffer and underlying region are open."""
        if self._closed or self._region.is_closed:
            raise RingBufferClosedError("Ring buffer is closed")

    def _read_counters(self) -> tuple[int, int]:
        """Read current (head, tail) sequence numbers from shared memory."""
        self._check_closed()
        head = struct.unpack_from(COUNTER_FORMAT, self._region.mmap, HEAD_OFFSET)[0]
        tail = struct.unpack_from(COUNTER_FORMAT, self._region.mmap, TAIL_OFFSET)[0]
        return head, tail

    def _write_head(self, new_head: int) -> None:
        """Update producer write sequence number in shared memory."""
        self._check_closed()
        struct.pack_into(COUNTER_FORMAT, self._region.mmap, HEAD_OFFSET, new_head)

    def _write_tail(self, new_tail: int) -> None:
        """Update consumer read sequence number in shared memory."""
        self._check_closed()
        struct.pack_into(COUNTER_FORMAT, self._region.mmap, TAIL_OFFSET, new_tail)

    def _slot_index(self, seq: int) -> int:
        """Calculate physical slot index for a given sequence number."""
        if self._mask is not None:
            return seq & self._mask
        return seq % self._capacity

    # --------------------------------------------------------------------------
    # Constructors & Factory Methods
    # --------------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        filepath: str | Path,
        capacity: int,
        record_size: int = ORDER_STRUCT_SIZE,
    ) -> RingBuffer:
        """Create a new file-backed ring buffer.

        Args:
            filepath: Path to the backing file on disk.
            capacity: Maximum number of records.
            record_size: Size in bytes of each record (default: 64).

        Returns:
            New RingBuffer instance open for reading and writing.
        """
        return cls(capacity=capacity, record_size=record_size, filepath=filepath, create=True)

    @classmethod
    def open(cls, filepath: str | Path, read_only: bool = False) -> RingBuffer:
        """Open an existing file-backed ring buffer.

        Args:
            filepath: Path to the backing file on disk.
            read_only: If True, opens mapping in read-only mode.

        Returns:
            Connected RingBuffer instance.
        """
        return cls(filepath=filepath, create=False, read_only=read_only)

    @classmethod
    def create_anonymous(
        cls,
        capacity: int,
        record_size: int = ORDER_STRUCT_SIZE,
    ) -> RingBuffer:
        """Create an anonymous in-memory ring buffer.

        Args:
            capacity: Maximum number of records.
            record_size: Size in bytes of each record (default: 64).

        Returns:
            New in-memory RingBuffer instance.
        """
        return cls(capacity=capacity, record_size=record_size, filepath=None, create=True)

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        """Maximum number of records the buffer can hold."""
        return self._capacity

    @property
    def record_size(self) -> int:
        """Byte size of each record in the buffer."""
        return self._record_size

    @property
    def region(self) -> SharedMemoryRegion:
        """Underlying SharedMemoryRegion instance."""
        return self._region

    @property
    def filepath(self) -> Path | None:
        """Backing file path, or None if anonymous."""
        return self._region.filepath

    @property
    def is_closed(self) -> bool:
        """True if the ring buffer is closed."""
        return self._closed or self._region.is_closed

    @property
    def is_read_only(self) -> bool:
        """True if the ring buffer was opened in read-only mode."""
        return self._read_only

    @property
    def head(self) -> int:
        """Current monotonic producer sequence counter."""
        head, _ = self._read_counters()
        return head

    @property
    def tail(self) -> int:
        """Current monotonic consumer sequence counter."""
        _, tail = self._read_counters()
        return tail

    @property
    def occupied(self) -> int:
        """Number of unread records currently residing in the buffer."""
        head, tail = self._read_counters()
        return (head - tail) & UINT64_MASK

    @property
    def is_empty(self) -> bool:
        """True if the buffer contains zero unread records."""
        return self.occupied == 0

    @property
    def is_full(self) -> bool:
        """True if the buffer is at maximum capacity."""
        return self.occupied >= self._capacity

    @property
    def available_write(self) -> int:
        """Number of free record slots available for writing."""
        occ = self.occupied
        return max(0, self._capacity - occ)

    @property
    def available_read(self) -> int:
        """Number of unread records available for reading."""
        return self.occupied

    # --------------------------------------------------------------------------
    # Producer Operations
    # --------------------------------------------------------------------------

    def put(
        self,
        item: Order | bytes | bytearray | memoryview,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        """Put an Order or raw binary record into the ring buffer.

        Args:
            item: Order instance or bytes-like record of length record_size.
            block: If True, blocks until space is available or timeout expires.
                   If False, raises RingBufferFullError immediately if full.
            timeout: Maximum seconds to wait when block=True (None for indefinite).

        Raises:
            RingBufferClosedError: If the buffer is closed.
            SharedMemoryReadOnlyError: If the buffer is read-only.
            TypeError: If item is not an Order or bytes-like object.
            SharedMemoryRecordSizeError: If bytes-like item length != record_size.
            RingBufferFullError: If buffer is full and block=False or timeout expires.
            ValueError: If timeout is negative.
        """
        self._check_closed()
        if self._read_only:
            raise SharedMemoryReadOnlyError("Cannot write to a read-only ring buffer")

        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout < 0:
                raise ValueError(f"timeout must be a non-negative number or None, got {timeout}")

        # Validate and prepare payload
        is_order = isinstance(item, Order)
        if is_order:
            if self._record_size != ORDER_STRUCT_SIZE:
                raise SharedMemoryRecordSizeError(
                    f"Record size mismatch: expected {self._record_size} bytes, got {ORDER_STRUCT_SIZE} bytes"
                )
            order_fields = _validate_order_fields(item)
        elif isinstance(item, (bytes, bytearray, memoryview)):
            if len(item) != self._record_size:
                raise SharedMemoryRecordSizeError(
                    f"Record size mismatch: expected {self._record_size} bytes, got {len(item)} bytes"
                )
        else:
            raise TypeError(f"item must be an Order or bytes-like object, got {type(item).__name__}")

        with self._lock:
            self._check_closed()
            end_time = (time.monotonic() + timeout) if (block and timeout is not None) else None

            while True:
                head, tail = self._read_counters()
                occupied = (head - tail) & UINT64_MASK
                if occupied < self._capacity:
                    break  # Free slot available

                if not block:
                    raise RingBufferFullError(f"Ring buffer is full (capacity={self._capacity})")

                if end_time is not None:
                    remaining = end_time - time.monotonic()
                    if remaining <= 0:
                        raise RingBufferFullError(
                            f"Timeout waiting to put into ring buffer (capacity={self._capacity})"
                        )
                    self._not_full.wait(timeout=min(remaining, 0.005))
                else:
                    self._not_full.wait(timeout=0.005)

                self._check_closed()

            # Space available: calculate slot offset
            slot = self._slot_index(head)
            slot_offset = SLOTS_OFFSET + slot * self._record_size

            # Step 1: Write record payload into slot (data written BEFORE head update)
            if is_order:
                ORDER_STRUCT.pack_into(self._region.mmap, slot_offset, *order_fields)
            else:
                self._region.write_record(slot_offset, item, record_size=self._record_size)

            # Step 2: Advance head sequence counter in shared memory
            new_head = (head + 1) & UINT64_MASK
            self._write_head(new_head)

            # Step 3: Notify waiting consumers
            self._not_empty.notify()

    def put_nowait(self, item: Order | bytes | bytearray | memoryview) -> None:
        """Put an item into the ring buffer without blocking.

        Raises:
            RingBufferFullError: If the buffer is full.
        """
        self.put(item, block=False)

    def put_order(
        self,
        order: Order,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        """Put an Order instance into the ring buffer."""
        self.put(order, block=block, timeout=timeout)

    def put_record(
        self,
        data: bytes | bytearray | memoryview,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        """Put a raw binary record into the ring buffer."""
        self.put(data, block=block, timeout=timeout)

    # --------------------------------------------------------------------------
    # Consumer Operations
    # --------------------------------------------------------------------------

    def get_record(
        self,
        block: bool = True,
        timeout: float | None = None,
    ) -> bytes:
        """Read and remove the next raw binary record from the ring buffer.

        Args:
            block: If True, blocks until a record is available or timeout expires.
                   If False, raises RingBufferEmptyError immediately if empty.
            timeout: Maximum seconds to wait when block=True (None for indefinite).

        Returns:
            Exact bytes of the record (length == record_size).

        Raises:
            RingBufferClosedError: If the buffer is closed.
            SharedMemoryReadOnlyError: If the buffer is read-only.
            RingBufferEmptyError: If buffer is empty and block=False or timeout expires.
            ValueError: If timeout is negative.
        """
        self._check_closed()
        if self._read_only:
            raise SharedMemoryReadOnlyError(
                "Cannot consume from a read-only ring buffer because tail cannot be updated"
            )

        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout < 0:
                raise ValueError(f"timeout must be a non-negative number or None, got {timeout}")

        with self._lock:
            self._check_closed()
            end_time = (time.monotonic() + timeout) if (block and timeout is not None) else None

            while True:
                head, tail = self._read_counters()
                occupied = (head - tail) & UINT64_MASK
                if occupied > 0:
                    break  # Record available

                if not block:
                    raise RingBufferEmptyError("Ring buffer is empty")

                if end_time is not None:
                    remaining = end_time - time.monotonic()
                    if remaining <= 0:
                        raise RingBufferEmptyError("Timeout waiting to get from ring buffer")
                    self._not_empty.wait(timeout=min(remaining, 0.005))
                else:
                    self._not_empty.wait(timeout=0.005)

                self._check_closed()

            # Record available: calculate slot offset
            slot = self._slot_index(tail)
            slot_offset = SLOTS_OFFSET + slot * self._record_size

            # Step 1: Read record payload from slot (read BEFORE tail update)
            data = self._region.read_record(slot_offset, record_size=self._record_size)

            # Step 2: Advance tail sequence counter in shared memory
            new_tail = (tail + 1) & UINT64_MASK
            self._write_tail(new_tail)

            # Step 3: Notify waiting producers
            self._not_full.notify()

            return data

    def get_record_nowait(self) -> bytes:
        """Read and remove the next binary record without blocking.

        Raises:
            RingBufferEmptyError: If the buffer is empty.
        """
        return self.get_record(block=False)

    def get(
        self,
        block: bool = True,
        timeout: float | None = None,
    ) -> Order:
        """Read and remove the next order from the ring buffer, decoding it into an Order.

        Args:
            block: If True, blocks until an order is available or timeout expires.
                   If False, raises RingBufferEmptyError immediately if empty.
            timeout: Maximum seconds to wait when block=True (None for indefinite).

        Returns:
            Decoded Order instance.

        Raises:
            RingBufferClosedError: If the buffer is closed.
            SharedMemoryReadOnlyError: If the buffer is read-only.
            RingBufferEmptyError: If buffer is empty and block=False or timeout expires.
            MalformedRecordError: If binary data in the record fails decoding.
        """
        self._check_closed()
        if self._read_only:
            raise SharedMemoryReadOnlyError(
                "Cannot consume from a read-only ring buffer because tail cannot be updated"
            )

        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout < 0:
                raise ValueError(f"timeout must be a non-negative number or None, got {timeout}")

        with self._lock:
            self._check_closed()
            end_time = (time.monotonic() + timeout) if (block and timeout is not None) else None

            while True:
                head, tail = self._read_counters()
                occupied = (head - tail) & UINT64_MASK
                if occupied > 0:
                    break  # Record available

                if not block:
                    raise RingBufferEmptyError("Ring buffer is empty")

                if end_time is not None:
                    remaining = end_time - time.monotonic()
                    if remaining <= 0:
                        raise RingBufferEmptyError("Timeout waiting to get from ring buffer")
                    self._not_empty.wait(timeout=min(remaining, 0.005))
                else:
                    self._not_empty.wait(timeout=0.005)

                self._check_closed()

            # Record available: calculate slot offset
            slot = self._slot_index(tail)
            slot_offset = SLOTS_OFFSET + slot * self._record_size

            # In-place decode directly from shared memory
            decode_error: Exception | None = None
            try:
                order = decode_order_from(self._region.mmap, slot_offset)
            except (ValueError, TypeError) as e:
                decode_error = e

            # Advance tail sequence counter in shared memory
            new_tail = (tail + 1) & UINT64_MASK
            self._write_tail(new_tail)

            # Notify waiting producers
            self._not_full.notify()

            if decode_error is not None:
                raise MalformedRecordError(
                    f"Malformed order record in ring buffer: {decode_error}"
                ) from decode_error

            return order

    def get_nowait(self) -> Order:
        """Read and remove the next order without blocking.

        Raises:
            RingBufferEmptyError: If the buffer is empty.
        """
        return self.get(block=False)

    def get_order(
        self,
        block: bool = True,
        timeout: float | None = None,
    ) -> Order:
        """Convenience alias for get()."""
        return self.get(block=block, timeout=timeout)

    # --------------------------------------------------------------------------
    # Peek & Inspection Operations
    # --------------------------------------------------------------------------

    def peek_record(self) -> bytes:
        """Inspect the next raw binary record without removing it from the buffer.

        Returns:
            Exact bytes of the next unread record.

        Raises:
            RingBufferClosedError: If closed.
            RingBufferEmptyError: If empty.
        """
        self._check_closed()
        with self._lock:
            head, tail = self._read_counters()
            occupied = (head - tail) & UINT64_MASK
            if occupied == 0:
                raise RingBufferEmptyError("Ring buffer is empty")
            slot = self._slot_index(tail)
            slot_offset = SLOTS_OFFSET + slot * self._record_size
            return self._region.read_record(slot_offset, record_size=self._record_size)

    def peek(self) -> Order:
        """Inspect the next Order without removing it from the buffer.

        Returns:
            Decoded Order instance.

        Raises:
            RingBufferClosedError: If closed.
            RingBufferEmptyError: If empty.
            MalformedRecordError: If binary data fails decoding.
        """
        self._check_closed()
        with self._lock:
            head, tail = self._read_counters()
            occupied = (head - tail) & UINT64_MASK
            if occupied == 0:
                raise RingBufferEmptyError("Ring buffer is empty")
            slot = self._slot_index(tail)
            slot_offset = SLOTS_OFFSET + slot * self._record_size
            try:
                return decode_order_from(self._region.mmap, slot_offset)
            except (ValueError, TypeError) as e:
                raise MalformedRecordError(f"Malformed order record in ring buffer: {e}") from e

    def clear(self) -> None:
        """Clear all unread records from the buffer by setting tail = head.

        Raises:
            RingBufferClosedError: If closed.
            SharedMemoryReadOnlyError: If read-only.
        """
        self._check_closed()
        if self._read_only:
            raise SharedMemoryReadOnlyError("Cannot clear a read-only ring buffer")
        with self._lock:
            head, _ = self._read_counters()
            self._write_tail(head)
            self._not_full.notify_all()

    # --------------------------------------------------------------------------
    # Lifecycle, Cleanup & Protocols
    # --------------------------------------------------------------------------

    def flush(self) -> None:
        """Flush shared memory modifications to backing storage."""
        self._check_closed()
        self._region.flush()

    def close(self) -> None:
        """Close the ring buffer and release underlying shared-memory resources.

        This method is idempotent and safe to call multiple times.
        """
        if self._closed:
            return
        self._closed = True

        with self._lock:
            self._not_full.notify_all()
            self._not_empty.notify_all()

        self._region.close()

    def unlink(self) -> None:
        """Close the ring buffer and delete the backing file from disk."""
        self.close()
        self._region.unlink()

    def __len__(self) -> int:
        """Return the number of unread records in the buffer."""
        return self.occupied

    def __iter__(self) -> Iterator[Order]:
        """Iterate over available orders until the buffer is empty."""
        while not self.is_empty:
            try:
                yield self.get_nowait()
            except RingBufferEmptyError:
                break

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        status = "closed" if self.is_closed else f"occupied={self.occupied}/{self._capacity}"
        return f"<RingBuffer capacity={self._capacity} record_size={self._record_size} {status}>"


# ==============================================================================
# RingBufferProducer & RingBufferConsumer Components
# ==============================================================================


class RingBufferProducer:
    """Dedicated producer component for writing orders to a ring buffer."""

    def __init__(self, ring_buffer: RingBuffer) -> None:
        """Initialize producer wrapping an open RingBuffer.

        Args:
            ring_buffer: Open writable RingBuffer instance.

        Raises:
            TypeError: If ring_buffer is not a RingBuffer.
            SharedMemoryReadOnlyError: If ring_buffer is read-only.
        """
        if not isinstance(ring_buffer, RingBuffer):
            raise TypeError(f"Expected RingBuffer, got {type(ring_buffer).__name__}")
        if ring_buffer.is_read_only:
            raise SharedMemoryReadOnlyError("Cannot create a producer on a read-only ring buffer")
        self._ring_buffer = ring_buffer

    @classmethod
    def create(
        cls,
        filepath: str | Path,
        capacity: int,
        record_size: int = ORDER_STRUCT_SIZE,
    ) -> RingBufferProducer:
        """Create a new file-backed ring buffer and initialize a producer.

        Args:
            filepath: Path to the backing file on disk.
            capacity: Maximum number of records.
            record_size: Record size in bytes (default: 64).

        Returns:
            Configured RingBufferProducer instance.
        """
        rb = RingBuffer.create(filepath=filepath, capacity=capacity, record_size=record_size)
        return cls(rb)

    @classmethod
    def create_anonymous(
        cls,
        capacity: int,
        record_size: int = ORDER_STRUCT_SIZE,
    ) -> RingBufferProducer:
        """Create an anonymous in-memory ring buffer and initialize a producer."""
        rb = RingBuffer.create_anonymous(capacity=capacity, record_size=record_size)
        return cls(rb)

    @property
    def ring_buffer(self) -> RingBuffer:
        """Underlying RingBuffer instance."""
        return self._ring_buffer

    @property
    def capacity(self) -> int:
        """Maximum number of records."""
        return self._ring_buffer.capacity

    @property
    def record_size(self) -> int:
        """Size of each record in bytes."""
        return self._ring_buffer.record_size

    @property
    def is_full(self) -> bool:
        """True if buffer is full."""
        return self._ring_buffer.is_full

    @property
    def available_write(self) -> int:
        """Free record slots available."""
        return self._ring_buffer.available_write

    @property
    def is_closed(self) -> bool:
        """True if buffer is closed."""
        return self._ring_buffer.is_closed

    def put(
        self,
        item: Order | bytes | bytearray | memoryview,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        """Put an Order or raw binary record into the ring buffer."""
        self._ring_buffer.put(item, block=block, timeout=timeout)

    def put_nowait(self, item: Order | bytes | bytearray | memoryview) -> None:
        """Put an item into the ring buffer without blocking."""
        self._ring_buffer.put_nowait(item)

    def put_order(
        self,
        order: Order,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        """Put an Order into the ring buffer."""
        self._ring_buffer.put_order(order, block=block, timeout=timeout)

    def put_record(
        self,
        data: bytes | bytearray | memoryview,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        """Put a raw binary record into the ring buffer."""
        self._ring_buffer.put_record(data, block=block, timeout=timeout)

    def flush(self) -> None:
        """Flush changes to backing storage."""
        self._ring_buffer.flush()

    def close(self) -> None:
        """Close buffer and mapping."""
        self._ring_buffer.close()

    def unlink(self) -> None:
        """Close buffer and remove backing file."""
        self._ring_buffer.unlink()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class RingBufferConsumer:
    """Dedicated consumer component for reading orders from a ring buffer."""

    def __init__(self, ring_buffer: RingBuffer) -> None:
        """Initialize consumer wrapping an open RingBuffer.

        Args:
            ring_buffer: Open RingBuffer instance.

        Raises:
            TypeError: If ring_buffer is not a RingBuffer.
        """
        if not isinstance(ring_buffer, RingBuffer):
            raise TypeError(f"Expected RingBuffer, got {type(ring_buffer).__name__}")
        self._ring_buffer = ring_buffer

    @classmethod
    def open(cls, filepath: str | Path, read_only: bool = False) -> RingBufferConsumer:
        """Open an existing file-backed ring buffer and initialize a consumer.

        Note: read_only=False is required for get() operations so tail can be updated.
        """
        rb = RingBuffer.open(filepath=filepath, read_only=read_only)
        return cls(rb)

    @property
    def ring_buffer(self) -> RingBuffer:
        """Underlying RingBuffer instance."""
        return self._ring_buffer

    @property
    def capacity(self) -> int:
        """Maximum number of records."""
        return self._ring_buffer.capacity

    @property
    def record_size(self) -> int:
        """Size of each record in bytes."""
        return self._ring_buffer.record_size

    @property
    def is_empty(self) -> bool:
        """True if buffer contains zero unread records."""
        return self._ring_buffer.is_empty

    @property
    def available_read(self) -> int:
        """Unread records available to consume."""
        return self._ring_buffer.available_read

    @property
    def is_closed(self) -> bool:
        """True if buffer is closed."""
        return self._ring_buffer.is_closed

    def get(self, block: bool = True, timeout: float | None = None) -> Order:
        """Read and remove the next order from the ring buffer."""
        return self._ring_buffer.get(block=block, timeout=timeout)

    def get_nowait(self) -> Order:
        """Read and remove the next order without blocking."""
        return self._ring_buffer.get_nowait()

    def get_record(self, block: bool = True, timeout: float | None = None) -> bytes:
        """Read and remove the next raw binary record from the ring buffer."""
        return self._ring_buffer.get_record(block=block, timeout=timeout)

    def get_record_nowait(self) -> bytes:
        """Read and remove the next raw binary record without blocking."""
        return self._ring_buffer.get_record_nowait()

    def peek(self) -> Order:
        """Inspect the next order without removing it."""
        return self._ring_buffer.peek()

    def peek_record(self) -> bytes:
        """Inspect the next raw binary record without removing it."""
        return self._ring_buffer.peek_record()

    def close(self) -> None:
        """Close buffer and mapping."""
        self._ring_buffer.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
