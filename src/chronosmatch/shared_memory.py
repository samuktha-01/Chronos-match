"""Shared-memory inter-process communication (IPC) for ChronosMatch.

Provides low-latency memory-mapped regions using Python's built-in `mmap` module,
enabling producers and consumers to exchange fixed-size 64-byte binary order records
without socket overhead, network protocols, or serialization copying between processes.

Zero-Copy Clarification:
------------------------
While the memory mapping itself is backed by shared memory pages shared across processes
without kernel socket transitions, this pure Python implementation should NOT be claimed
as fully zero-copy. Instantiating Python `Order` instances from binary structs and encoding
them into byte buffers involves Python object allocation and serialization copying (`struct.pack`
and `struct.unpack`). True end-to-end zero-copy execution is reserved for compiled Cython/C
kernels operating directly on raw memory pointers.

Platform-Specific Behavior (Windows vs POSIX):
----------------------------------------------
- On Windows, memory mappings are created over files or pagefile-backed objects.
- When creating a file-backed mapping, the file must be explicitly sized (e.g., via
  `file.truncate(size)`) before calling `mmap.mmap()`.
- Windows enforces strict file locking semantics: as long as an `mmap` object or underlying
  file descriptor remains open, Windows prevents deletion of the backing file
  (raising `PermissionError: [WinError 32] The process cannot access the file because it is
  being used by another process`).
- To guarantee reliable cleanup and avoid resource leaks or locked files on Windows, the
  mapping lifecycle ensures `mmap.close()` is called first, followed by closing the backing
  file descriptor, before invoking `os.unlink()` / `Path.unlink()`.
"""

from __future__ import annotations

import mmap
import os
from pathlib import Path
from typing import Any, BinaryIO, Self

from chronosmatch.binary_protocol import (
    ORDER_STRUCT_SIZE,
    decode_order,
    decode_order_from,
    encode_order,
    encode_order_into,
)
from chronosmatch.models import Order

# ==============================================================================
# Shared-Memory IPC Exceptions
# ==============================================================================


class SharedMemoryError(Exception):
    """Base exception for all ChronosMatch shared-memory IPC operations."""


class SharedMemoryClosedError(SharedMemoryError, ValueError):
    """Raised when an operation is attempted on a closed shared-memory mapping."""


class SharedMemoryBoundsError(SharedMemoryError, ValueError, IndexError):
    """Raised when an offset or buffer boundary exceeds the memory region size."""


class SharedMemoryRecordSizeError(SharedMemoryError, ValueError):
    """Raised when a record's byte length does not match the expected fixed record size."""


class SharedMemoryReadOnlyError(SharedMemoryError, PermissionError):
    """Raised when attempting a write operation on a read-only shared-memory mapping."""


class MalformedRecordError(SharedMemoryError, ValueError):
    """Raised when binary record data in shared memory fails decoding."""


# ==============================================================================
# SharedMemoryRegion Implementation
# ==============================================================================


class SharedMemoryRegion:
    """Manages a memory-mapped shared memory region backed by a file or anonymous RAM.

    Provides a clean, deterministic lifecycle for creating, opening, reading,
    writing, flushing, closing, and unlinking shared memory backing files.
    """

    def __init__(
        self,
        filepath: str | Path | None = None,
        size: int | None = None,
        *,
        create: bool = False,
        read_only: bool = False,
    ) -> None:
        """Initialize a shared-memory region.

        Args:
            filepath: Path to the backing file (None for anonymous in-memory mapping).
            size: Desired byte size of the mapping. Required if create=True.
            create: If True, creates and initializes a new backing file/mapping.
            read_only: If True, maps the region in read-only mode.

        Raises:
            TypeError: If arguments have invalid types.
            ValueError: If size or filepath values are invalid.
            FileNotFoundError: If opening an existing file that does not exist.
        """
        self._filepath: Path | None = None
        self._file_obj: BinaryIO | None = None
        self._mmap: mmap.mmap | None = None
        self._closed: bool = False
        self._read_only: bool = read_only

        if filepath is not None:
            if not isinstance(filepath, (str, Path)):
                raise TypeError(f"filepath must be a str or Path, got {type(filepath).__name__}")
            path_str = str(filepath).strip()
            if not path_str:
                raise ValueError("filepath cannot be empty or whitespace")
            self._filepath = Path(filepath).resolve()

        if create:
            if size is None:
                raise ValueError("size is required when creating a new shared-memory region")
            if isinstance(size, bool) or not isinstance(size, int):
                raise TypeError(f"size must be an integer, got {type(size).__name__}")
            if size <= 0:
                raise ValueError(f"size must be strictly positive (> 0), got {size}")
            self._size = size
            self._create_region()
        else:
            if self._filepath is None:
                raise ValueError("filepath is required when opening an existing shared-memory region")
            if not self._filepath.exists():
                raise FileNotFoundError(f"Backing file not found: {self._filepath}")
            file_size = self._filepath.stat().st_size
            if file_size == 0:
                raise ValueError(f"Backing file '{self._filepath}' is empty (0 bytes). Cannot map empty file.")

            if size is not None:
                if isinstance(size, bool) or not isinstance(size, int):
                    raise TypeError(f"size must be an integer, got {type(size).__name__}")
                if size <= 0:
                    raise ValueError(f"size must be strictly positive (> 0), got {size}")
                if size > file_size:
                    raise ValueError(
                        f"Requested size {size} exceeds backing file size {file_size}"
                    )
                self._size = size
            else:
                self._size = file_size

            self._open_region()

    def _create_region(self) -> None:
        """Create the backing file and initialize the memory mapping."""
        if self._filepath is not None:
            self._filepath.parent.mkdir(parents=True, exist_ok=True)
            self._file_obj = open(self._filepath, "w+b")
            self._file_obj.truncate(self._size)
            self._file_obj.flush()
            self._mmap = mmap.mmap(self._file_obj.fileno(), self._size, access=mmap.ACCESS_WRITE)
        else:
            # Anonymous memory mapping (Windows and POSIX compatible)
            self._mmap = mmap.mmap(-1, self._size, access=mmap.ACCESS_WRITE)

    def _open_region(self) -> None:
        """Open an existing backing file and map it."""
        assert self._filepath is not None
        if self._read_only:
            self._file_obj = open(self._filepath, "rb")
            self._mmap = mmap.mmap(
                self._file_obj.fileno(),
                self._size,
                access=mmap.ACCESS_READ,
            )
        else:
            self._file_obj = open(self._filepath, "r+b")
            self._mmap = mmap.mmap(
                self._file_obj.fileno(),
                self._size,
                access=mmap.ACCESS_WRITE,
            )

    @classmethod
    def create(cls, filepath: str | Path, size: int) -> SharedMemoryRegion:
        """Create a new file-backed shared-memory region of the given size.

        Args:
            filepath: Path to the backing file on disk.
            size: Total capacity of the shared memory region in bytes.

        Returns:
            New SharedMemoryRegion instance open for read/write.
        """
        return cls(filepath=filepath, size=size, create=True, read_only=False)

    @classmethod
    def open(cls, filepath: str | Path, read_only: bool = False) -> SharedMemoryRegion:
        """Open an existing file-backed shared-memory region.

        Args:
            filepath: Path to the existing backing file.
            read_only: If True, opens mapping in read-only mode.

        Returns:
            SharedMemoryRegion instance connected to the backing file.
        """
        return cls(filepath=filepath, create=False, read_only=read_only)

    @classmethod
    def create_anonymous(cls, size: int) -> SharedMemoryRegion:
        """Create an anonymous in-memory shared region without a file on disk.

        Args:
            size: Total capacity in bytes.

        Returns:
            SharedMemoryRegion instance backed by RAM/paging file.
        """
        return cls(filepath=None, size=size, create=True, read_only=False)

    # --------------------------------------------------------------------------
    # Lifecycle & Properties
    # --------------------------------------------------------------------------

    @property
    def filepath(self) -> Path | None:
        """Return the backing file path, or None if anonymous."""
        return self._filepath

    @property
    def size(self) -> int:
        """Return the size of the shared-memory region in bytes."""
        return self._size

    @property
    def is_closed(self) -> bool:
        """Return True if the mapping has been closed."""
        return self._closed

    @property
    def is_read_only(self) -> bool:
        """Return True if the mapping was opened in read-only mode."""
        return self._read_only

    @property
    def capacity_records(self) -> int:
        """Return the maximum number of 64-byte order records that fit in this region."""
        return self._size // ORDER_STRUCT_SIZE

    @property
    def mmap(self) -> mmap.mmap:
        """Direct access to the underlying mmap object.

        Raises:
            SharedMemoryClosedError: If the region is closed.
        """
        self._check_closed()
        assert self._mmap is not None
        return self._mmap

    def _check_closed(self) -> None:
        """Verify the region is open.

        Raises:
            SharedMemoryClosedError: If closed.
        """
        if self._closed or self._mmap is None or self._mmap.closed:
            raise SharedMemoryClosedError("Shared memory region is closed")

    def _validate_offset_and_size(self, offset: Any, length: Any) -> tuple[int, int]:
        """Validate offset and byte length against memory bounds.

        Args:
            offset: Byte offset into the memory region.
            length: Number of bytes to read or write.

        Returns:
            Tuple of validated (offset, length).

        Raises:
            SharedMemoryClosedError: If closed.
            TypeError: If offset or length is not an integer.
            SharedMemoryBoundsError: If offset or length is negative, or if offset + length exceeds size.
        """
        self._check_closed()
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise TypeError(f"offset must be an integer, got {type(offset).__name__}")
        if isinstance(length, bool) or not isinstance(length, int):
            raise TypeError(f"length must be an integer, got {type(length).__name__}")
        if offset < 0:
            raise SharedMemoryBoundsError(f"Offset cannot be negative: {offset}")
        if length < 0:
            raise SharedMemoryBoundsError(f"Length cannot be negative: {length}")
        if offset + length > self._size:
            raise SharedMemoryBoundsError(
                f"Buffer boundary exceeded: offset {offset} + length {length} "
                f"({offset + length}) exceeds region size {self._size}"
            )
        return offset, length

    # --------------------------------------------------------------------------
    # Raw Byte I/O
    # --------------------------------------------------------------------------

    def write_bytes(self, offset: int, data: bytes | bytearray | memoryview) -> None:
        """Write raw bytes into the shared-memory region at the given offset.

        Args:
            offset: Byte offset into the memory region.
            data: Bytes-like object to write.

        Raises:
            SharedMemoryClosedError: If the region is closed.
            SharedMemoryReadOnlyError: If the region is read-only.
            TypeError: If offset or data has an invalid type.
            SharedMemoryBoundsError: If offset + len(data) exceeds region size.
        """
        self._check_closed()
        if self._read_only:
            raise SharedMemoryReadOnlyError("Cannot write to a read-only shared-memory mapping")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError(f"Expected bytes-like data, got {type(data).__name__}")

        offset, length = self._validate_offset_and_size(offset, len(data))
        assert self._mmap is not None
        self._mmap[offset : offset + length] = data

    def read_bytes(self, offset: int, length: int) -> bytes:
        """Read raw bytes from the shared-memory region at the given offset.

        Args:
            offset: Byte offset into the memory region.
            length: Number of bytes to read.

        Returns:
            Bytes slice of the specified length.

        Raises:
            SharedMemoryClosedError: If the region is closed.
            TypeError: If offset or length has an invalid type.
            SharedMemoryBoundsError: If offset + length exceeds region size.
        """
        self._check_closed()
        offset, length = self._validate_offset_and_size(offset, length)
        assert self._mmap is not None
        return bytes(self._mmap[offset : offset + length])

    # --------------------------------------------------------------------------
    # Fixed-Size Record I/O
    # --------------------------------------------------------------------------

    def write_record(
        self,
        offset: int,
        data: bytes | bytearray | memoryview,
        record_size: int = ORDER_STRUCT_SIZE,
    ) -> None:
        """Write a fixed-size binary record into the memory region at offset.

        Args:
            offset: Byte offset where record begins.
            data: Binary record to write. Length must equal record_size.
            record_size: Expected record size in bytes (default: ORDER_STRUCT_SIZE = 64).

        Raises:
            SharedMemoryRecordSizeError: If len(data) != record_size.
            SharedMemoryBoundsError: If offset + record_size exceeds region size.
            SharedMemoryClosedError: If the region is closed.
            SharedMemoryReadOnlyError: If the region is read-only.
        """
        if isinstance(record_size, bool) or not isinstance(record_size, int) or record_size <= 0:
            raise ValueError(f"record_size must be a strictly positive integer, got {record_size}")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError(f"Expected bytes-like data, got {type(data).__name__}")
        if len(data) != record_size:
            raise SharedMemoryRecordSizeError(
                f"Record size mismatch: expected {record_size} bytes, got {len(data)} bytes"
            )
        self.write_bytes(offset, data)

    def read_record(self, offset: int, record_size: int = ORDER_STRUCT_SIZE) -> bytes:
        """Read a fixed-size binary record from the memory region at offset.

        Args:
            offset: Byte offset where record begins.
            record_size: Number of bytes to read (default: ORDER_STRUCT_SIZE = 64).

        Returns:
            Exactly record_size bytes.

        Raises:
            ValueError: If record_size <= 0.
            SharedMemoryBoundsError: If offset + record_size exceeds region size.
            SharedMemoryClosedError: If the region is closed.
        """
        if isinstance(record_size, bool) or not isinstance(record_size, int) or record_size <= 0:
            raise ValueError(f"record_size must be a strictly positive integer, got {record_size}")
        return self.read_bytes(offset, record_size)

    # --------------------------------------------------------------------------
    # Order Binary Protocol Integration (Phase 6)
    # --------------------------------------------------------------------------

    def write_order(self, order: Order, offset: int = 0) -> None:
        """Serialize an Order instance and write its 64-byte record at offset.

        Uses the exact binary encoding layout (<Q32sqIBBBBQ).

        Args:
            order: Order instance to encode and write.
            offset: Byte offset in the region (default: 0).

        Raises:
            TypeError: If order is not an Order instance.
            SharedMemoryBoundsError: If offset + 64 exceeds region capacity.
            SharedMemoryClosedError: If region is closed.
            SharedMemoryReadOnlyError: If region is read-only.
        """
        if not isinstance(order, Order):
            raise TypeError(f"Expected Order instance, got {type(order).__name__}")
        self._check_closed()
        if self._read_only:
            raise SharedMemoryReadOnlyError("Cannot write to a read-only shared-memory mapping")
        offset, length = self._validate_offset_and_size(offset, ORDER_STRUCT_SIZE)
        assert self._mmap is not None
        encode_order_into(order, self._mmap, offset=offset)

    def read_order(self, offset: int = 0) -> Order:
        """Read 64 bytes at offset and decode into an Order instance.

        Uses the exact binary decoding routine.

        Args:
            offset: Byte offset in the region where the record begins (default: 0).

        Returns:
            Reconstructed Order instance with all original field types preserved.

        Raises:
            SharedMemoryBoundsError: If offset + 64 exceeds region capacity.
            SharedMemoryClosedError: If region is closed.
            MalformedRecordError: If binary data is corrupt or cannot be decoded.
        """
        self._check_closed()
        offset, length = self._validate_offset_and_size(offset, ORDER_STRUCT_SIZE)
        assert self._mmap is not None
        try:
            return decode_order_from(self._mmap, offset=offset)
        except (ValueError, TypeError) as e:
            raise MalformedRecordError(
                f"Malformed order record at offset {offset}: {e}"
            ) from e

    # --------------------------------------------------------------------------
    # Slot-Based Helpers (64-byte aligned)
    # --------------------------------------------------------------------------

    def write_order_at_slot(self, order: Order, slot: int = 0) -> None:
        """Write an order into the specified record slot index (slot * 64 bytes).

        Args:
            order: Order instance to write.
            slot: Zero-based record slot index.
        """
        if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
            raise SharedMemoryBoundsError(f"Slot must be a non-negative integer, got {slot}")
        self.write_order(order, offset=slot * ORDER_STRUCT_SIZE)

    def read_order_at_slot(self, slot: int = 0) -> Order:
        """Read an order from the specified record slot index (slot * 64 bytes).

        Args:
            slot: Zero-based record slot index.

        Returns:
            Decoded Order instance.
        """
        if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
            raise SharedMemoryBoundsError(f"Slot must be a non-negative integer, got {slot}")
        return self.read_order(offset=slot * ORDER_STRUCT_SIZE)

    # --------------------------------------------------------------------------
    # Resource Cleanup & Context Manager
    # --------------------------------------------------------------------------

    def flush(self) -> None:
        """Flush modifications in memory mapping to backing storage.

        Raises:
            SharedMemoryClosedError: If region is closed.
        """
        self._check_closed()
        assert self._mmap is not None
        self._mmap.flush()

    def close(self) -> None:
        """Close the memory mapping and release backing file handles.

        This method is idempotent and safe to call multiple times.
        Closing the mmap first, then the file object, ensures clean release
        of Windows file locks.
        """
        if self._closed:
            return
        self._closed = True

        if self._mmap is not None:
            try:
                self._mmap.close()
            finally:
                self._mmap = None

        if self._file_obj is not None:
            try:
                self._file_obj.close()
            finally:
                self._file_obj = None

    def unlink(self) -> None:
        """Close the mapping and permanently delete the backing file from disk.

        On Windows, memory mappings and file descriptors hold a mandatory lock.
        Calling close() prior to file removal ensures deletion succeeds without
        PermissionError / WinError 32.
        """
        self.close()
        if self._filepath is not None and self._filepath.exists():
            try:
                self._filepath.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ==============================================================================
# SharedMemoryProducer & SharedMemoryConsumer
# ==============================================================================


class SharedMemoryProducer:
    """Dedicated producer component for writing orders into a shared-memory region."""

    def __init__(self, region: SharedMemoryRegion) -> None:
        """Initialize producer wrapping a shared memory region.

        Args:
            region: Open read-write SharedMemoryRegion.

        Raises:
            TypeError: If region is not a SharedMemoryRegion.
            SharedMemoryReadOnlyError: If region is read-only.
        """
        if not isinstance(region, SharedMemoryRegion):
            raise TypeError(f"Expected SharedMemoryRegion, got {type(region).__name__}")
        if region.is_read_only:
            raise SharedMemoryReadOnlyError("Cannot create a producer on a read-only shared-memory region")
        self._region = region

    @classmethod
    def create(cls, filepath: str | Path, size: int) -> SharedMemoryProducer:
        """Create a new backing shared-memory file and initialize a producer.

        Args:
            filepath: Path to the backing file.
            size: Capacity in bytes.

        Returns:
            Configured SharedMemoryProducer instance.
        """
        region = SharedMemoryRegion.create(filepath=filepath, size=size)
        return cls(region)

    @property
    def region(self) -> SharedMemoryRegion:
        """Underlying SharedMemoryRegion instance."""
        return self._region

    @property
    def size(self) -> int:
        """Capacity in bytes."""
        return self._region.size

    @property
    def capacity_records(self) -> int:
        """Maximum number of 64-byte records."""
        return self._region.capacity_records

    @property
    def filepath(self) -> Path | None:
        """Backing file path."""
        return self._region.filepath

    @property
    def is_closed(self) -> bool:
        """True if region is closed."""
        return self._region.is_closed

    def write_order(self, order: Order, offset: int = 0) -> None:
        """Write an Order at byte offset."""
        self._region.write_order(order, offset=offset)

    def write_order_at_slot(self, order: Order, slot: int = 0) -> None:
        """Write an Order at slot index (slot * 64 bytes)."""
        self._region.write_order_at_slot(order, slot=slot)

    def write_record(
        self,
        offset: int,
        data: bytes | bytearray | memoryview,
        record_size: int = ORDER_STRUCT_SIZE,
    ) -> None:
        """Write a fixed-size binary record at byte offset."""
        self._region.write_record(offset, data, record_size=record_size)

    def write_bytes(self, offset: int, data: bytes | bytearray | memoryview) -> None:
        """Write raw bytes at byte offset."""
        self._region.write_bytes(offset, data)

    def flush(self) -> None:
        """Flush changes to storage."""
        self._region.flush()

    def close(self) -> None:
        """Close mapping and backing file."""
        self._region.close()

    def unlink(self) -> None:
        """Close mapping and remove backing file."""
        self._region.unlink()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class SharedMemoryConsumer:
    """Dedicated consumer component for reading orders from a shared-memory region."""

    def __init__(self, region: SharedMemoryRegion) -> None:
        """Initialize consumer wrapping a shared memory region.

        Args:
            region: Open SharedMemoryRegion (read-only or read-write).

        Raises:
            TypeError: If region is not a SharedMemoryRegion.
        """
        if not isinstance(region, SharedMemoryRegion):
            raise TypeError(f"Expected SharedMemoryRegion, got {type(region).__name__}")
        self._region = region

    @classmethod
    def open(cls, filepath: str | Path, read_only: bool = True) -> SharedMemoryConsumer:
        """Open an existing backing shared-memory file and initialize a consumer.

        Args:
            filepath: Path to the existing backing file.
            read_only: If True, opens mapping in read-only mode (default: True).

        Returns:
            Configured SharedMemoryConsumer instance.
        """
        region = SharedMemoryRegion.open(filepath=filepath, read_only=read_only)
        return cls(region)

    @property
    def region(self) -> SharedMemoryRegion:
        """Underlying SharedMemoryRegion instance."""
        return self._region

    @property
    def size(self) -> int:
        """Capacity in bytes."""
        return self._region.size

    @property
    def capacity_records(self) -> int:
        """Maximum number of 64-byte records."""
        return self._region.capacity_records

    @property
    def filepath(self) -> Path | None:
        """Backing file path."""
        return self._region.filepath

    @property
    def is_closed(self) -> bool:
        """True if region is closed."""
        return self._region.is_closed

    def read_order(self, offset: int = 0) -> Order:
        """Read and decode an Order from byte offset."""
        return self._region.read_order(offset=offset)

    def read_order_at_slot(self, slot: int = 0) -> Order:
        """Read and decode an Order from slot index (slot * 64 bytes)."""
        return self._region.read_order_at_slot(slot=slot)

    def read_record(self, offset: int, record_size: int = ORDER_STRUCT_SIZE) -> bytes:
        """Read fixed-size binary record from byte offset."""
        return self._region.read_record(offset=offset, record_size=record_size)

    def read_bytes(self, offset: int, length: int) -> bytes:
        """Read raw bytes from byte offset."""
        return self._region.read_bytes(offset=offset, length=length)

    def close(self) -> None:
        """Close mapping and backing file."""
        self._region.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
