"""Trade-Event Persistence, Write-Ahead Logging (WAL), and Crash Recovery for ChronosMatch.

Provides deterministic, append-only binary write-ahead logging (WAL), durable trade
persistence, and crash recovery replay mechanisms for the ChronosMatch matching engine
and end-to-end trading pipeline.

Key Components:
- `DurabilityPolicy`: Explicit durability modes (IMMEDIATE_SYNC, FLUSH_ON_WRITE, ASYNC_BUFFERED).
- `WriteAheadLog`: Append-only, versioned, CRC32-checksummed binary log for order, cancel,
  trade, and checkpoint events with corruption detection and truncated record recovery.
- `TradeStore`: Dedicated persistent storage and querying for executed trade records with
  stable identifiers and full execution details.
- `RecoveryManager`: Deterministic crash recovery replaying valid WAL records, reconstructing
  exact Limit Order Book state and trade history without duplicate trade emission.
- `PersistedMatchingEngine`: Transparent wrapper combining MatchingEngine with WAL and TradeStore.
- `AsyncPersistenceWorker`: Background worker decoupling disk I/O from the latency-sensitive matching path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, IntEnum
import os
from pathlib import Path
import queue
import struct
import threading
import time
from typing import Any, BinaryIO, Self
import zlib

from chronosmatch.binary_protocol import (
    MAX_ORDER_ID_LENGTH,
    PRICE_SCALE,
    _decode_order_id,
    _reconstruct_price,
    _validate_and_encode_order_id,
    _validate_and_scale_price,
    decode_order,
    encode_order,
)
from chronosmatch.matching_engine import MatchingEngine, Trade
from chronosmatch.models import Order, OrderSide
from chronosmatch.order_book import DuplicateOrderError, LimitOrderBook, OrderNotFoundError

# ==============================================================================
# Binary Framing & Struct Layout Specifications
# ==============================================================================

# File-level Header (32 bytes at offset 0):
#   4s : magic b"CMWL" (ChronosMatch Write Log)
#   H  : format version (uint16 = 1)
#   H  : header size (uint16 = 32)
#   Q  : created timestamp nanoseconds (uint64)
#   B  : durability policy code (uint8: 1=SYNC, 2=FLUSH, 3=ASYNC)
#   15s: reserved padding
WAL_FILE_MAGIC: bytes = b"CMWL"
WAL_FORMAT_VERSION: int = 1
WAL_FILE_HEADER_SIZE: int = 32
WAL_FILE_HEADER_FORMAT: str = "<4sHHQB15s"
_FILE_HEADER_STRUCT: struct.Struct = struct.Struct(WAL_FILE_HEADER_FORMAT)

# Record Frame Header (24 bytes per record):
#   4s : record magic b"CWAL"
#   Q  : sequence number (uint64 LSN, strictly monotonic >= 1)
#   B  : record type (uint8: 1=ORDER, 2=CANCEL, 3=TRADE, 4=CHECKPOINT)
#   B  : flags (uint8)
#   2s : reserved padding (uint16)
#   I  : payload length (uint32)
#   I  : checksum (uint32 CRC32 of payload)
WAL_RECORD_MAGIC: bytes = b"CWAL"
WAL_RECORD_HEADER_SIZE: int = 24
WAL_RECORD_HEADER_FORMAT: str = "<4sQBB2sII"
_RECORD_HEADER_STRUCT: struct.Struct = struct.Struct(WAL_RECORD_HEADER_FORMAT)

# Payloads:
# 1. ORDER: 64 bytes (exact 64-byte ORDER_STRUCT from binary_protocol.py)
# 2. CANCEL: 40 bytes:
#    32s : order_id (null-padded UTF-8)
#    Q   : timestamp_ns (uint64)
CANCEL_PAYLOAD_SIZE: int = 40
CANCEL_PAYLOAD_FORMAT: str = "<32sQ"
_CANCEL_STRUCT: struct.Struct = struct.Struct(CANCEL_PAYLOAD_FORMAT)

# 3. TRADE: 128 bytes (2 cache lines):
#    32s : trade_id_str (null-padded UTF-8 string representation of trade_id)
#    32s : resting_order_id (maker order_id)
#    32s : incoming_order_id (taker order_id)
#    q   : price_ticks (int64, scaled by PRICE_SCALE 10^8)
#    I   : quantity (uint32)
#    B   : price_type (uint8: 1=float, 2=int, 3=Decimal)
#    B   : id_is_int (uint8: 1 if trade_id was int, 0 if str)
#    Q   : timestamp_ns (uint64 nanoseconds)
#    10s : reserved padding
TRADE_PAYLOAD_SIZE: int = 128
TRADE_PAYLOAD_FORMAT: str = "<32s32s32sqIBBQ10s"
_TRADE_STRUCT: struct.Struct = struct.Struct(TRADE_PAYLOAD_FORMAT)

# 4. CHECKPOINT: 24 bytes:
#    Q : timestamp_ns (uint64)
#    Q : last_order_seq (uint64)
#    Q : last_trade_id (uint64)
CHECKPOINT_PAYLOAD_SIZE: int = 24
CHECKPOINT_PAYLOAD_FORMAT: str = "<QQQ"
_CHECKPOINT_STRUCT: struct.Struct = struct.Struct(CHECKPOINT_PAYLOAD_FORMAT)


# ==============================================================================
# Enumerations and Data Models
# ==============================================================================


class DurabilityPolicy(str, Enum):
    """Durability guarantees for Write-Ahead Log (WAL) and Trade persistence.

    Attributes:
        IMMEDIATE_SYNC: Synchronously flushes OS page cache and issues physical disk
            sync (os.fsync) on every write before returning. Highest durability
            (zero data loss on OS/power failure), higher write latency.
        FLUSH_ON_WRITE: Flushes every record to the operating system page cache
            (file.flush()) without blocking for hardware disk fsync. Survives
            normal process termination or crash; vulnerable to kernel panics or power outages.
        ASYNC_BUFFERED: Writes are buffered in memory and flushed/synced in batches
            by a background persistence worker thread. Keeps disk I/O completely off
            the latency-sensitive matching path; trailing uncommitted buffered records
            may be lost upon abrupt process kill.
    """

    IMMEDIATE_SYNC = "IMMEDIATE_SYNC"
    FLUSH_ON_WRITE = "FLUSH_ON_WRITE"
    ASYNC_BUFFERED = "ASYNC_BUFFERED"


class WALRecordType(IntEnum):
    """Enumeration of valid record types stored in the Write-Ahead Log."""

    ORDER = 1
    CANCEL = 2
    TRADE = 3
    CHECKPOINT = 4


@dataclass(slots=True)
class CancelRecord:
    """Represents an order cancellation event in the WAL."""

    order_id: str
    timestamp_ns: int


@dataclass(slots=True)
class CheckpointRecord:
    """Represents a periodic engine state checkpoint event in the WAL."""

    timestamp_ns: int
    last_order_seq: int
    last_trade_id: int


@dataclass(slots=True)
class WALRecord:
    """Framed record decoded from the Write-Ahead Log.

    Attributes:
        sequence_number: Strictly monotonic Log Sequence Number (LSN >= 1).
        record_type: WALRecordType enum.
        timestamp_ns: Nanosecond timestamp associated with the event.
        data: Deserialized domain object (Order, CancelRecord, Trade, CheckpointRecord).
        offset: Byte offset in the WAL file where this record begins.
        length: Total record byte length (header + payload).
        checksum: CRC32 checksum of the payload.
        flags: Reserved bitflags.
    """

    sequence_number: int
    record_type: WALRecordType
    timestamp_ns: int
    data: Order | CancelRecord | Trade | CheckpointRecord
    offset: int = 0
    length: int = 0
    checksum: int = 0
    flags: int = 0

    @property
    def is_order(self) -> bool:
        """True if this record encapsulates an incoming order."""
        return self.record_type == WALRecordType.ORDER

    @property
    def is_cancel(self) -> bool:
        """True if this record encapsulates an order cancellation."""
        return self.record_type == WALRecordType.CANCEL

    @property
    def is_trade(self) -> bool:
        """True if this record encapsulates an executed trade."""
        return self.record_type == WALRecordType.TRADE

    @property
    def is_checkpoint(self) -> bool:
        """True if this record encapsulates an engine checkpoint."""
        return self.record_type == WALRecordType.CHECKPOINT


@dataclass
class RecoveryResult:
    """Diagnostic outcome of an engine crash recovery and replay procedure."""

    engine: MatchingEngine
    recovered_orders_count: int = 0
    recovered_cancels_count: int = 0
    recovered_trades_count: int = 0
    uncommitted_trades_count: int = 0
    last_sequence_number: int = 0
    truncated_bytes_skipped: int = 0
    recovered_trades: list[Trade] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """True if recovery finished without warnings or truncated trailing records."""
        return len(self.warnings) == 0 and self.truncated_bytes_skipped == 0


# ==============================================================================
# Exceptions
# ==============================================================================


class PersistenceError(Exception):
    """Base exception for all persistence, WAL, and crash recovery failures."""


class WALError(PersistenceError):
    """Base exception for errors originating in the Write-Ahead Log."""


class WALCorruptedRecordError(WALError):
    """Raised when data corruption (magic mismatch, CRC32 failure) is detected."""


class WALTruncatedRecordError(WALError):
    """Raised when an incomplete trailing record is encountered in strict recovery mode."""


class WALClosedError(WALError):
    """Raised when attempting I/O operations on a closed Write-Ahead Log."""


class WALWriteError(WALError):
    """Raised when an append, flush, or sync to disk encounters an I/O error."""


class RecoveryError(PersistenceError):
    """Raised when engine state reconstruction or replay encounters a fatal error."""


class DuplicateTradeError(PersistenceError):
    """Raised when duplicate trade execution or conflicting trade ID is detected."""


# ==============================================================================
# Binary Payload Encoding & Decoding Helpers
# ==============================================================================


def encode_trade_payload(trade: Trade) -> bytes:
    """Encode Trade instance into fixed-size 128-byte binary struct."""
    raw_maker = _validate_and_encode_order_id(trade.resting_order_id)
    raw_taker = _validate_and_encode_order_id(trade.incoming_order_id)

    # Encode trade_id (preserves int vs str representation)
    is_int = 1 if isinstance(trade.trade_id, int) and not isinstance(trade.trade_id, bool) else 0
    trade_id_str = str(trade.trade_id)
    raw_trade_id = _validate_and_encode_order_id(trade_id_str)

    price_ticks, price_type = _validate_and_scale_price(trade.price)

    return _TRADE_STRUCT.pack(
        raw_trade_id,
        raw_maker,
        raw_taker,
        price_ticks,
        trade.quantity,
        price_type,
        is_int,
        trade.timestamp,
        b"\x00" * 10,
    )


def decode_trade_payload(payload: bytes) -> Trade:
    """Decode fixed-size 128-byte binary struct into Trade instance."""
    if len(payload) != TRADE_PAYLOAD_SIZE:
        raise ValueError(f"Expected {TRADE_PAYLOAD_SIZE} bytes for Trade payload, got {len(payload)}")

    (
        raw_trade_id,
        raw_maker,
        raw_taker,
        price_ticks,
        quantity,
        price_type,
        is_int,
        timestamp_ns,
        _,
    ) = _TRADE_STRUCT.unpack(payload)

    trade_id_str = _decode_order_id(raw_trade_id)
    trade_id: int | str = int(trade_id_str) if is_int == 1 else trade_id_str
    maker_id = _decode_order_id(raw_maker)
    taker_id = _decode_order_id(raw_taker)
    price = _reconstruct_price(price_ticks, price_type)

    return Trade(
        trade_id=trade_id,
        resting_order_id=maker_id,
        incoming_order_id=taker_id,
        price=price,
        quantity=quantity,
        timestamp=timestamp_ns,
    )


def encode_cancel_payload(order_id: str, timestamp_ns: int) -> bytes:
    """Encode order cancellation event into fixed-size 40-byte binary struct."""
    raw_order_id = _validate_and_encode_order_id(order_id)
    return _CANCEL_STRUCT.pack(raw_order_id, timestamp_ns)


def decode_cancel_payload(payload: bytes) -> CancelRecord:
    """Decode fixed-size 40-byte binary struct into CancelRecord."""
    if len(payload) != CANCEL_PAYLOAD_SIZE:
        raise ValueError(f"Expected {CANCEL_PAYLOAD_SIZE} bytes for Cancel payload, got {len(payload)}")
    raw_order_id, timestamp_ns = _CANCEL_STRUCT.unpack(payload)
    order_id = _decode_order_id(raw_order_id)
    return CancelRecord(order_id=order_id, timestamp_ns=timestamp_ns)


def encode_checkpoint_payload(timestamp_ns: int, last_order_seq: int, last_trade_id: int) -> bytes:
    """Encode checkpoint into fixed-size 24-byte binary struct."""
    return _CHECKPOINT_STRUCT.pack(timestamp_ns, last_order_seq, last_trade_id)


def decode_checkpoint_payload(payload: bytes) -> CheckpointRecord:
    """Decode fixed-size 24-byte binary struct into CheckpointRecord."""
    if len(payload) != CHECKPOINT_PAYLOAD_SIZE:
        raise ValueError(f"Expected {CHECKPOINT_PAYLOAD_SIZE} bytes for Checkpoint, got {len(payload)}")
    timestamp_ns, last_order_seq, last_trade_id = _CHECKPOINT_STRUCT.unpack(payload)
    return CheckpointRecord(
        timestamp_ns=timestamp_ns,
        last_order_seq=last_order_seq,
        last_trade_id=last_trade_id,
    )


# ==============================================================================
# Write-Ahead Log (WAL)
# ==============================================================================


class WriteAheadLog:
    """Deterministic, append-only, binary framed Write-Ahead Log (WAL).

    Ensures that all mutating state events (order arrivals, cancellations, executions)
    are serialized to persistent storage before or alongside engine state mutation.
    Features:
    - 32-byte file header with magic bytes and versioning.
    - 24-byte record frame headers with monotonic sequence numbers (LSN) and CRC32 checksums.
    - Append-only sequential disk writes conforming to configurable `DurabilityPolicy`.
    - Resilient readback capable of identifying truncated trailing records and corrupted frames.
    - Safe truncation of incomplete trailing writes back to the last valid frame boundary.
    """

    def __init__(
        self,
        filepath: str | Path,
        durability: DurabilityPolicy = DurabilityPolicy.FLUSH_ON_WRITE,
        *,
        auto_flush: bool = True,
        create: bool = True,
    ) -> None:
        """Initialize or reopen an append-only Write-Ahead Log file.

        Args:
            filepath: Path to the WAL log file.
            durability: DurabilityPolicy controlling flush and fsync frequency.
            auto_flush: If True, flushes/syncs after each write according to durability policy.
            create: If True, creates file and parent directories if non-existent.
        """
        self._filepath = Path(filepath)
        self._durability = DurabilityPolicy(durability)
        self._auto_flush = auto_flush
        self._lock = threading.RLock()
        self._is_closed = False
        self._next_lsn: int = 1
        self._record_count: int = 0

        if create:
            self._filepath.parent.mkdir(parents=True, exist_ok=True)

        mode = "r+b" if self._filepath.exists() and self._filepath.stat().st_size > 0 else "w+b"
        try:
            self._file: BinaryIO = open(self._filepath, mode)
        except OSError as e:
            raise WALWriteError(f"Failed to open WAL file '{self._filepath}': {e}") from e

        # Initialize or validate file header
        self._file.seek(0, os.SEEK_END)
        file_size = self._file.tell()
        if file_size == 0:
            self._write_file_header()
        else:
            self._validate_existing_file()

    def _write_file_header(self) -> None:
        """Write the initial 32-byte WAL file header at offset 0."""
        durability_code = {
            DurabilityPolicy.IMMEDIATE_SYNC: 1,
            DurabilityPolicy.FLUSH_ON_WRITE: 2,
            DurabilityPolicy.ASYNC_BUFFERED: 3,
        }.get(self._durability, 2)

        header_bytes = _FILE_HEADER_STRUCT.pack(
            WAL_FILE_MAGIC,
            WAL_FORMAT_VERSION,
            WAL_FILE_HEADER_SIZE,
            time.time_ns(),
            durability_code,
            b"\x00" * 15,
        )
        self._file.seek(0)
        self._file.write(header_bytes)
        self._flush_file()

    def _validate_existing_file(self) -> None:
        """Validate existing WAL header and position write pointer to end of last valid record."""
        self._file.seek(0)
        header_bytes = self._file.read(WAL_FILE_HEADER_SIZE)
        if len(header_bytes) < WAL_FILE_HEADER_SIZE:
            # File is smaller than header; reset file header
            self._write_file_header()
            return

        magic, version, header_size, _, _, _ = _FILE_HEADER_STRUCT.unpack(header_bytes)
        if magic != WAL_FILE_MAGIC:
            raise WALCorruptedRecordError(
                f"Invalid WAL file magic: expected {WAL_FILE_MAGIC!r}, got {magic!r}"
            )
        if version != WAL_FORMAT_VERSION:
            raise WALCorruptedRecordError(
                f"Unsupported WAL version: expected {WAL_FORMAT_VERSION}, got {version}"
            )

        # Fast scan to find the last sequence number and count
        records, warnings, valid_offset = self.read_records(strict=False)
        self._record_count = len(records)
        if records:
            self._next_lsn = records[-1].sequence_number + 1
        else:
            self._next_lsn = 1

        # Position write pointer at the end of last valid record
        self._file.seek(valid_offset)

    def _check_open(self) -> None:
        if self._is_closed:
            raise WALClosedError(f"WriteAheadLog '{self._filepath}' is closed.")

    def _flush_file(self) -> None:
        """Execute flush and/or fsync according to DurabilityPolicy."""
        try:
            self._file.flush()
            if self._durability == DurabilityPolicy.IMMEDIATE_SYNC:
                os.fsync(self._file.fileno())
        except OSError as e:
            raise WALWriteError(f"I/O failure during WAL flush/sync: {e}") from e

    def _write_record_frame(self, record_type: WALRecordType, payload: bytes) -> int:
        """Append a single framed binary record to the WAL.

        Returns:
            The assigned Log Sequence Number (LSN).
        """
        self._check_open()
        with self._lock:
            lsn = self._next_lsn
            checksum = zlib.crc32(payload) & 0xFFFFFFFF
            payload_len = len(payload)

            frame_header = _RECORD_HEADER_STRUCT.pack(
                WAL_RECORD_MAGIC,
                lsn,
                int(record_type),
                0,  # flags
                b"\x00\x00",
                payload_len,
                checksum,
            )

            try:
                self._file.write(frame_header)
                self._file.write(payload)
                if self._auto_flush and self._durability != DurabilityPolicy.ASYNC_BUFFERED:
                    self._flush_file()
            except OSError as e:
                raise WALWriteError(f"Failed to write record {lsn} to WAL: {e}") from e

            self._next_lsn += 1
            self._record_count += 1
            return lsn

    def append_order(self, order: Order) -> int:
        """Serialize and append an incoming Order event to the WAL.

        Args:
            order: The Order to record.

        Returns:
            Assigned Log Sequence Number (LSN).
        """
        payload = encode_order(order)
        return self._write_record_frame(WALRecordType.ORDER, payload)

    def append_cancel(self, order_id: str, timestamp_ns: int | None = None) -> int:
        """Serialize and append an order cancellation event to the WAL.

        Args:
            order_id: Unique identifier of the cancelled order.
            timestamp_ns: Optional cancellation timestamp (defaults to time.time_ns()).

        Returns:
            Assigned Log Sequence Number (LSN).
        """
        ts = timestamp_ns if timestamp_ns is not None else time.time_ns()
        payload = encode_cancel_payload(order_id, ts)
        return self._write_record_frame(WALRecordType.CANCEL, payload)

    def append_trade(self, trade: Trade) -> int:
        """Serialize and append a Trade execution record to the WAL.

        Args:
            trade: Executed Trade to record.

        Returns:
            Assigned Log Sequence Number (LSN).
        """
        payload = encode_trade_payload(trade)
        return self._write_record_frame(WALRecordType.TRADE, payload)

    def append_trades(self, trades: list[Trade]) -> list[int]:
        """Serialize and append a batch of Trade executions to the WAL.

        Args:
            trades: List of executed trades to record.

        Returns:
            List of assigned Log Sequence Numbers.
        """
        if not trades:
            return []
        lsns: list[int] = []
        with self._lock:
            for trade in trades:
                lsns.append(self.append_trade(trade))
        return lsns

    def append_checkpoint(self, last_order_seq: int = 0, last_trade_id: int = 0) -> int:
        """Append an engine state checkpoint record.

        Args:
            last_order_seq: Highest sequence number processed.
            last_trade_id: Highest trade ID assigned.

        Returns:
            Assigned Log Sequence Number (LSN).
        """
        payload = encode_checkpoint_payload(time.time_ns(), last_order_seq, last_trade_id)
        return self._write_record_frame(WALRecordType.CHECKPOINT, payload)

    def flush(self) -> None:
        """Flush internal buffers to the operating system."""
        self._check_open()
        with self._lock:
            self._flush_file()

    def sync(self) -> None:
        """Force synchronous flush to physical non-volatile disk storage (os.fsync)."""
        self._check_open()
        with self._lock:
            try:
                self._file.flush()
                os.fsync(self._file.fileno())
            except OSError as e:
                raise WALWriteError(f"I/O error during WAL fsync: {e}") from e

    def truncate_to_valid(self, valid_length: int) -> None:
        """Truncate the log file back to a known valid byte boundary, discarding trailing corruption."""
        self._check_open()
        with self._lock:
            try:
                self._file.seek(valid_length)
                self._file.truncate(valid_length)
                self._flush_file()
            except OSError as e:
                raise WALWriteError(f"Failed to truncate WAL file to {valid_length}: {e}") from e

    def read_records(
        self, strict: bool = False
    ) -> tuple[list[WALRecord], list[str], int]:
        """Read, validate, and decode all records from the beginning of the log.

        Args:
            strict: If True, raises WALTruncatedRecordError or WALCorruptedRecordError
                immediately upon encountering partial or corrupted records.
                If False (recovery mode), safely ignores trailing partial records and
                returns all intact preceding records with diagnostic warnings.

        Returns:
            Tuple of (valid_records, warnings_list, valid_bytes_offset).
        """
        with self._lock:
            self._file.seek(0, os.SEEK_END)
            total_file_size = self._file.tell()
            if total_file_size < WAL_FILE_HEADER_SIZE:
                return [], ["WAL file is empty or smaller than file header"], 0

            self._file.seek(WAL_FILE_HEADER_SIZE)
            pos = WAL_FILE_HEADER_SIZE
            records: list[WALRecord] = []
            warnings: list[str] = []

            while pos < total_file_size:
                remaining_bytes = total_file_size - pos
                if remaining_bytes < WAL_RECORD_HEADER_SIZE:
                    msg = (
                        f"Incomplete trailing record header at offset {pos} "
                        f"({remaining_bytes} bytes remaining, expected {WAL_RECORD_HEADER_SIZE})"
                    )
                    if strict:
                        raise WALTruncatedRecordError(msg)
                    warnings.append(msg)
                    break

                header_bytes = self._file.read(WAL_RECORD_HEADER_SIZE)
                (
                    magic,
                    seq_num,
                    rec_type_code,
                    flags,
                    _,
                    payload_len,
                    checksum,
                ) = _RECORD_HEADER_STRUCT.unpack(header_bytes)

                if magic != WAL_RECORD_MAGIC:
                    msg = f"Corrupted record magic at offset {pos}: got {magic!r}"
                    if strict:
                        raise WALCorruptedRecordError(msg)
                    warnings.append(msg)
                    break

                if rec_type_code not in (1, 2, 3, 4):
                    msg = f"Unknown record type {rec_type_code} at sequence {seq_num}, offset {pos}"
                    if strict:
                        raise WALCorruptedRecordError(msg)
                    warnings.append(msg)
                    break

                rec_type = WALRecordType(rec_type_code)

                # Check payload completeness
                bytes_after_header = total_file_size - (pos + WAL_RECORD_HEADER_SIZE)
                if bytes_after_header < payload_len:
                    msg = (
                        f"Truncated payload for record seq {seq_num} at offset {pos} "
                        f"({bytes_after_header} bytes available, expected {payload_len})"
                    )
                    if strict:
                        raise WALTruncatedRecordError(msg)
                    warnings.append(msg)
                    break

                payload = self._file.read(payload_len)
                actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
                if actual_crc != checksum:
                    msg = (
                        f"CRC32 checksum mismatch for record seq {seq_num} at offset {pos}: "
                        f"expected {checksum}, got {actual_crc}"
                    )
                    if strict:
                        raise WALCorruptedRecordError(msg)
                    warnings.append(msg)
                    break

                # Decode payload according to record type
                try:
                    if rec_type == WALRecordType.ORDER:
                        data: Any = decode_order(payload)
                        ts = data.timestamp
                    elif rec_type == WALRecordType.CANCEL:
                        data = decode_cancel_payload(payload)
                        ts = data.timestamp_ns
                    elif rec_type == WALRecordType.TRADE:
                        data = decode_trade_payload(payload)
                        ts = data.timestamp
                    elif rec_type == WALRecordType.CHECKPOINT:
                        data = decode_checkpoint_payload(payload)
                        ts = data.timestamp_ns
                    else:
                        raise ValueError(f"Unhandled record type {rec_type}")
                except Exception as e:
                    msg = f"Failed to deserialize payload for record {seq_num} at offset {pos}: {e}"
                    if strict:
                        raise WALCorruptedRecordError(msg) from e
                    warnings.append(msg)
                    break

                record_length = WAL_RECORD_HEADER_SIZE + payload_len
                records.append(
                    WALRecord(
                        sequence_number=seq_num,
                        record_type=rec_type,
                        timestamp_ns=ts,
                        data=data,
                        offset=pos,
                        length=record_length,
                        checksum=checksum,
                        flags=flags,
                    )
                )
                pos += record_length

            return records, warnings, pos

    def close(self) -> None:
        """Flush buffered writes and close the file handle."""
        with self._lock:
            if self._is_closed:
                return
            self._is_closed = True
            try:
                self._flush_file()
            except Exception:
                pass
            finally:
                try:
                    self._file.close()
                except Exception:
                    pass

    @property
    def filepath(self) -> Path:
        """Filesystem path to this WAL file."""
        return self._filepath

    @property
    def is_closed(self) -> bool:
        """True if the WAL file handle is closed."""
        return self._is_closed

    @property
    def last_sequence_number(self) -> int:
        """Last assigned Log Sequence Number (0 if empty)."""
        return self._next_lsn - 1

    @property
    def record_count(self) -> int:
        """Total records appended to this log."""
        return self._record_count

    @property
    def durability_policy(self) -> DurabilityPolicy:
        """Active durability guarantee policy."""
        return self._durability

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


# ==============================================================================
# Trade Store (Dedicated Trade Persistence)
# ==============================================================================


class TradeStore:
    """Dedicated persistent repository for executed Trade records.

    Provides stable identifier indexing, order ID association, total volume aggregation,
    and optional append-only binary disk persistence.
    """

    def __init__(
        self,
        filepath: str | Path | None = None,
        durability: DurabilityPolicy = DurabilityPolicy.FLUSH_ON_WRITE,
    ) -> None:
        """Initialize the TradeStore.

        Args:
            filepath: Optional backing file path for append-only binary persistence.
                If None, operates as an in-memory trade repository.
            durability: DurabilityPolicy for disk flushes.
        """
        self._filepath = Path(filepath) if filepath is not None else None
        self._durability = DurabilityPolicy(durability)
        self._trades: list[Trade] = []
        self._trades_by_id: dict[int | str, Trade] = {}
        self._trades_by_order: dict[str, list[Trade]] = {}
        self._total_volume: int = 0
        self._lock = threading.RLock()
        self._is_closed = False
        self._file: BinaryIO | None = None

        if self._filepath is not None:
            self._filepath.parent.mkdir(parents=True, exist_ok=True)
            mode = "r+b" if self._filepath.exists() and self._filepath.stat().st_size > 0 else "w+b"
            self._file = open(self._filepath, mode)
            self._load_existing_trades()

    def _load_existing_trades(self) -> None:
        """Read any previously persisted trades from the backing file."""
        if self._file is None:
            return
        self._file.seek(0, os.SEEK_END)
        size = self._file.tell()
        if size == 0:
            return

        self._file.seek(0)
        while True:
            payload = self._file.read(TRADE_PAYLOAD_SIZE)
            if len(payload) < TRADE_PAYLOAD_SIZE:
                break
            try:
                trade = decode_trade_payload(payload)
                self._index_trade(trade)
            except Exception:
                break

    def _index_trade(self, trade: Trade) -> None:
        """Index a trade in memory."""
        self._trades.append(trade)
        self._trades_by_id[trade.trade_id] = trade
        self._trades_by_order.setdefault(trade.resting_order_id, []).append(trade)
        self._trades_by_order.setdefault(trade.incoming_order_id, []).append(trade)
        self._total_volume += trade.quantity

    def append_trade(self, trade: Trade) -> None:
        """Persist and index an executed Trade record.

        Args:
            trade: The executed Trade instance.

        Raises:
            DuplicateTradeError: If a trade with identical trade_id but conflicting fields exists.
        """
        if self._is_closed:
            raise PersistenceError("TradeStore is closed.")
        with self._lock:
            # Check idempotency / duplicates
            if trade.trade_id in self._trades_by_id:
                existing = self._trades_by_id[trade.trade_id]
                if (
                    existing.price != trade.price
                    or existing.quantity != trade.quantity
                    or existing.resting_order_id != trade.resting_order_id
                    or existing.incoming_order_id != trade.incoming_order_id
                ):
                    raise DuplicateTradeError(
                        f"Trade ID '{trade.trade_id}' already exists with conflicting data: "
                        f"existing={existing}, new={trade}"
                    )
                # Idempotent re-submission: ignore without duplicating
                return

            self._index_trade(trade)

            if self._file is not None:
                payload = encode_trade_payload(trade)
                try:
                    self._file.write(payload)
                    self._file.flush()
                    if self._durability == DurabilityPolicy.IMMEDIATE_SYNC:
                        os.fsync(self._file.fileno())
                except OSError as e:
                    raise PersistenceError(f"Failed to append trade to disk: {e}") from e

    def append_trades(self, trades: list[Trade]) -> None:
        """Persist a list of executed trades in order."""
        with self._lock:
            for trade in trades:
                self.append_trade(trade)

    def get_trade(self, trade_id: int | str) -> Trade | None:
        """Look up trade by its unique trade_id."""
        with self._lock:
            return self._trades_by_id.get(trade_id)

    def get_trades_for_order(self, order_id: str) -> list[Trade]:
        """Return all executions involving the specified order_id (maker or taker)."""
        with self._lock:
            return list(self._trades_by_order.get(order_id, []))

    def get_all_trades(self) -> list[Trade]:
        """Return defensive copy of all executed trades in historical execution order."""
        with self._lock:
            return list(self._trades)

    def count(self) -> int:
        """Total number of persisted trades."""
        with self._lock:
            return len(self._trades)

    def total_volume(self) -> int:
        """Total aggregated volume executed across all trades."""
        with self._lock:
            return self._total_volume

    def flush(self) -> None:
        """Flush backing file to OS buffer."""
        with self._lock:
            if self._file is not None and not self._is_closed:
                self._file.flush()

    def sync(self) -> None:
        """Synchronously commit backing file to disk."""
        with self._lock:
            if self._file is not None and not self._is_closed:
                self._file.flush()
                os.fsync(self._file.fileno())

    def close(self) -> None:
        """Close the trade store."""
        with self._lock:
            if self._is_closed:
                return
            self._is_closed = True
            if self._file is not None:
                try:
                    self._file.flush()
                except Exception:
                    pass
                try:
                    self._file.close()
                except Exception:
                    pass

    @property
    def filepath(self) -> Path | None:
        """Path to backing trade file, or None if in-memory."""
        return self._filepath

    @property
    def is_closed(self) -> bool:
        """True if the store is closed."""
        return self._is_closed

    @classmethod
    def from_wal(cls, wal_path: str | Path) -> TradeStore:
        """Construct a populated TradeStore by extracting all Trade records from a WAL file."""
        store = cls()
        with WriteAheadLog(wal_path, create=False) as wal:
            records, _, _ = wal.read_records(strict=False)
            for rec in records:
                if rec.is_trade and isinstance(rec.data, Trade):
                    store.append_trade(rec.data)
        return store


# ==============================================================================
# Background Asynchronous Persistence Worker
# ==============================================================================


class AsyncPersistenceWorker:
    """Decoupled background persistence worker thread for DurabilityPolicy.ASYNC_BUFFERED.

    Consumes persistence tasks (orders, cancellations, trades) from an in-memory queue
    and writes them sequentially to the WriteAheadLog and/or TradeStore off the critical path.
    """

    def __init__(
        self,
        wal: WriteAheadLog,
        trade_store: TradeStore | None = None,
        *,
        batch_size: int = 256,
        flush_interval_s: float = 0.05,
    ) -> None:
        self._wal = wal
        self._trade_store = trade_store
        self._batch_size = batch_size
        self._flush_interval_s = flush_interval_s

        self._queue: queue.Queue[Any] = queue.Queue(maxsize=65536)
        self._stop_event = threading.Event()
        self._is_closed = False
        self._worker_thread = threading.Thread(
            target=self._run_loop,
            name="ChronosMatch-PersistenceWorker",
            daemon=True,
        )
        self._worker_thread.start()

    def submit_order(self, order: Order) -> None:
        """Submit an order event for asynchronous logging."""
        self._queue.put(("ORDER", order))

    def submit_cancel(self, order_id: str, timestamp_ns: int | None = None) -> None:
        """Submit an order cancellation event for asynchronous logging."""
        self._queue.put(("CANCEL", order_id, timestamp_ns))

    def submit_trade(self, trade: Trade) -> None:
        """Submit an executed trade for asynchronous logging."""
        self._queue.put(("TRADE", trade))

    def submit_trades(self, trades: list[Trade]) -> None:
        """Submit a batch of executed trades for asynchronous logging."""
        for trade in trades:
            self._queue.put(("TRADE", trade))

    def _run_loop(self) -> None:
        """Background thread loop batching disk writes."""
        while not self._stop_event.is_set() or not self._queue.empty():
            batch: list[Any] = []
            try:
                # Block briefly waiting for at least one item
                item = self._queue.get(timeout=self._flush_interval_s)
                batch.append(item)
                # Drain up to batch_size items without blocking
                while len(batch) < self._batch_size:
                    try:
                        batch.append(self._queue.get_nowait())
                    except queue.Empty:
                        break
            except queue.Empty:
                pass

            if batch:
                for item in batch:
                    action = item[0]
                    if action == "ORDER":
                        self._wal.append_order(item[1])
                    elif action == "CANCEL":
                        self._wal.append_cancel(item[1], item[2])
                    elif action == "TRADE":
                        trade = item[1]
                        self._wal.append_trade(trade)
                        if self._trade_store is not None:
                            self._trade_store.append_trade(trade)
                    self._queue.task_done()

                self._wal.flush()
                if self._trade_store is not None:
                    self._trade_store.flush()

    def flush(self) -> None:
        """Wait until all pending items in the queue have been written and flushed to disk."""
        self._queue.join()
        self._wal.flush()
        if self._trade_store is not None:
            self._trade_store.flush()

    def stop(self) -> None:
        """Signal worker thread to stop and wait for it to join."""
        if self._is_closed:
            return
        self._is_closed = True
        self._stop_event.set()
        self._worker_thread.join(timeout=5.0)

    def close(self) -> None:
        """Stop worker and flush remaining records."""
        self.stop()
        self.flush()


# ==============================================================================
# Crash Recovery & State Reconstruction Manager
# ==============================================================================


class RecoveryManager:
    """Orchestrates deterministic crash recovery and engine state reconstruction.

    Replays valid Write-Ahead Log (WAL) records sequentially through a clean
    MatchingEngine, faithfully reconstructing:
    1. The exact Limit Order Book state (resting bids and asks, price levels, priority queues).
    2. The full historical trade execution tape without duplicate trade emission.
    3. Monotonic trade counter and sequence tracking.
    """

    def __init__(
        self,
        wal_path: str | Path,
        trade_store_path: str | Path | None = None,
        *,
        strict: bool = False,
    ) -> None:
        """Initialize RecoveryManager.

        Args:
            wal_path: Path to the WAL log file.
            trade_store_path: Optional path to existing trade store file.
            strict: If True, halts with error on truncated records; if False, recovers
                all valid records up to the crash point.
        """
        self.wal_path = Path(wal_path)
        self.trade_store_path = Path(trade_store_path) if trade_store_path is not None else None
        self.strict = strict

    def repair_wal(self) -> int:
        """Scan WAL and safely truncate trailing incomplete records.

        Returns:
            The number of valid records preserved.
        """
        if not self.wal_path.exists():
            return 0
        with WriteAheadLog(self.wal_path, create=False) as wal:
            records, warnings, valid_offset = wal.read_records(strict=False)
            wal.truncate_to_valid(valid_offset)
            return len(records)

    def recover(
        self,
        target_engine: MatchingEngine | None = None,
    ) -> RecoveryResult:
        """Execute deterministic crash recovery and state reconstruction.

        Args:
            target_engine: Optional MatchingEngine instance to recover state into
                (instantiates a clean MatchingEngine if None).

        Returns:
            Populated RecoveryResult containing the reconstructed engine and diagnostic metrics.

        Raises:
            RecoveryError: If recovery encounters an unrecoverable failure.
            WALCorruptedRecordError: If log corruption is detected in strict mode.
        """
        if not self.wal_path.exists():
            raise FileNotFoundError(f"WAL file '{self.wal_path}' does not exist.")

        engine = target_engine if target_engine is not None else MatchingEngine()
        engine.clear()

        # Step 1: Read all valid records and detect any crash truncation
        with WriteAheadLog(self.wal_path, create=False) as wal:
            records, warnings, valid_offset = wal.read_records(strict=self.strict)
            file_size = wal.filepath.stat().st_size
            truncated_bytes = max(0, file_size - valid_offset)

        if not records:
            return RecoveryResult(
                engine=engine,
                truncated_bytes_skipped=truncated_bytes,
                warnings=warnings,
            )

        # Step 2: Load known persisted trades from WAL and optional trade store
        persisted_trades_in_wal: dict[int | str, Trade] = {}
        for rec in records:
            if rec.is_trade and isinstance(rec.data, Trade):
                persisted_trades_in_wal[rec.data.trade_id] = rec.data

        # Step 3: Replay order and cancel events deterministically
        recovered_orders = 0
        recovered_cancels = 0
        recovered_trades: list[Trade] = []
        uncommitted_trades_count = 0
        highest_trade_id = 0

        for rec in records:
            if rec.is_order and isinstance(rec.data, Order):
                recovered_orders += 1
                # Process order through matching engine
                # Note: We pass a copy of the order to avoid mutating the record
                order_copy = Order(
                    order_id=rec.data.order_id,
                    side=rec.data.side,
                    price=rec.data.price,
                    quantity=rec.data.quantity,
                    timestamp=rec.data.timestamp,
                    sequence_number=rec.data.sequence_number,
                )

                try:
                    generated_trades = engine.process_order(order_copy)
                except DuplicateOrderError as e:
                    warnings.append(f"Duplicate order ID '{order_copy.order_id}' skipped during replay: {e}")
                    continue

                for gen_trade in generated_trades:
                    # Update highest trade ID
                    if isinstance(gen_trade.trade_id, int):
                        highest_trade_id = max(highest_trade_id, gen_trade.trade_id)

                    # Check if this trade was already persisted before the crash
                    if gen_trade.trade_id in persisted_trades_in_wal:
                        # Trade was already persisted: verify deterministic consistency
                        persisted = persisted_trades_in_wal[gen_trade.trade_id]
                        if (
                            persisted.price != gen_trade.price
                            or persisted.quantity != gen_trade.quantity
                            or persisted.resting_order_id != gen_trade.resting_order_id
                            or persisted.incoming_order_id != gen_trade.incoming_order_id
                        ):
                            raise RecoveryError(
                                f"Replayed trade {gen_trade.trade_id} diverged from persisted record: "
                                f"replayed={gen_trade}, persisted={persisted}"
                            )
                        recovered_trades.append(persisted)
                    else:
                        # Order was durable in WAL, but crash occurred before trade was written
                        # Deterministic execution reconstructed the intended trade!
                        uncommitted_trades_count += 1
                        recovered_trades.append(gen_trade)

            elif rec.is_cancel and isinstance(rec.data, CancelRecord):
                recovered_cancels += 1
                try:
                    engine.cancel_order(rec.data.order_id)
                except OrderNotFoundError:
                    # Order may have been fully filled prior to cancellation; safe to ignore
                    warnings.append(
                        f"Replayed cancel for order '{rec.data.order_id}' not found in book (may be filled)."
                    )

            elif rec.is_trade:
                # Trade records are already indexed; ensure highest_trade_id accounts for them
                if isinstance(rec.data, Trade) and isinstance(rec.data.trade_id, int):
                    highest_trade_id = max(highest_trade_id, rec.data.trade_id)

        # Step 4: Synchronize engine internal trade list and counter
        engine._trades.clear()
        engine._trades.extend(recovered_trades)
        engine._trade_counter = highest_trade_id

        last_lsn = records[-1].sequence_number if records else 0

        return RecoveryResult(
            engine=engine,
            recovered_orders_count=recovered_orders,
            recovered_cancels_count=recovered_cancels,
            recovered_trades_count=len(recovered_trades),
            uncommitted_trades_count=uncommitted_trades_count,
            last_sequence_number=last_lsn,
            truncated_bytes_skipped=truncated_bytes,
            recovered_trades=recovered_trades,
            warnings=warnings,
        )


# ==============================================================================
# Persisted Matching Engine
# ==============================================================================


class PersistedMatchingEngine:
    """Wrapper combining MatchingEngine with Write-Ahead Logging and Trade persistence.

    Seamlessly persists all incoming orders, cancellations, and generated trades
    according to the specified DurabilityPolicy.
    """

    def __init__(
        self,
        engine: MatchingEngine | None = None,
        wal_path: str | Path | None = None,
        *,
        wal: WriteAheadLog | None = None,
        trade_store: TradeStore | None = None,
        durability: DurabilityPolicy = DurabilityPolicy.FLUSH_ON_WRITE,
    ) -> None:
        """Initialize PersistedMatchingEngine.

        Args:
            engine: Underlying MatchingEngine (defaults to clean MatchingEngine).
            wal_path: Path to WAL log file (used if wal is not provided).
            wal: Existing WriteAheadLog instance.
            trade_store: Optional dedicated TradeStore instance.
            durability: DurabilityPolicy for persistence operations.
        """
        self.engine: MatchingEngine = engine if engine is not None else MatchingEngine()
        self._durability = DurabilityPolicy(durability)

        if wal is not None:
            self._wal = wal
            self._owns_wal = False
        elif wal_path is not None:
            self._wal = WriteAheadLog(wal_path, durability=self._durability)
            self._owns_wal = True
        else:
            raise ValueError("Either 'wal' or 'wal_path' must be provided to PersistedMatchingEngine.")

        self._trade_store = trade_store
        self._async_worker: AsyncPersistenceWorker | None = None
        if self._durability == DurabilityPolicy.ASYNC_BUFFERED:
            self._async_worker = AsyncPersistenceWorker(
                wal=self._wal,
                trade_store=self._trade_store,
            )

    def process_order(self, order: Order) -> list[Trade]:
        """Log incoming order to WAL, execute against order book, and log trades.

        Args:
            order: Incoming Order instance.

        Returns:
            List of executed Trade records.
        """
        # 1. Write-Ahead Log: Log the order before or as it enters the engine
        if self._async_worker is not None:
            self._async_worker.submit_order(order)
        else:
            self._wal.append_order(order)

        # 2. Execute order in matching core
        trades = self.engine.process_order(order)

        # 3. Log trade execution events
        if trades:
            if self._async_worker is not None:
                self._async_worker.submit_trades(trades)
            else:
                self._wal.append_trades(trades)
                if self._trade_store is not None:
                    self._trade_store.append_trades(trades)

        return trades

    def cancel_order(self, order_id: str) -> Order:
        """Log cancellation to WAL and cancel resting order in the book.

        Args:
            order_id: Order identifier to cancel.

        Returns:
            The cancelled Order instance.
        """
        if self._async_worker is not None:
            self._async_worker.submit_cancel(order_id)
        else:
            self._wal.append_cancel(order_id)

        return self.engine.cancel_order(order_id)

    @property
    def trades(self) -> list[Trade]:
        """All Trade executions produced by the matching engine."""
        return self.engine.trades

    @property
    def order_book(self) -> LimitOrderBook:
        """Underlying LimitOrderBook instance."""
        return self.engine.order_book

    @property
    def best_bid(self) -> float | int | Decimal | None:
        """Best resting bid price."""
        return self.engine.best_bid

    @property
    def best_ask(self) -> float | int | Decimal | None:
        """Best resting ask price."""
        return self.engine.best_ask

    @property
    def spread(self) -> float | int | Decimal | None:
        """Current bid-ask spread."""
        return self.engine.spread

    @property
    def total_trades(self) -> int:
        """Total trade count."""
        return self.engine.total_trades

    @property
    def total_volume_traded(self) -> int:
        """Total executed volume."""
        return self.engine.total_volume_traded

    @property
    def wal(self) -> WriteAheadLog:
        """Active WriteAheadLog instance."""
        return self._wal

    @property
    def trade_store(self) -> TradeStore | None:
        """Active TradeStore instance if configured."""
        return self._trade_store

    def flush(self) -> None:
        """Flush internal buffers and WAL to disk."""
        if self._async_worker is not None:
            self._async_worker.flush()
        else:
            self._wal.flush()
            if self._trade_store is not None:
                self._trade_store.flush()

    def sync(self) -> None:
        """Force physical disk synchronization."""
        if self._async_worker is not None:
            self._async_worker.flush()
        self._wal.sync()
        if self._trade_store is not None:
            self._trade_store.sync()

    def close(self) -> None:
        """Flush and safely close persistence resources."""
        if self._async_worker is not None:
            self._async_worker.close()
        if self._owns_wal:
            self._wal.close()
        if self._trade_store is not None:
            self._trade_store.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
