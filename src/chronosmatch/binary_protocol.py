"""Binary Order Protocol implementation for ChronosMatch.

Provides deterministic, fixed-size 64-byte binary serialization and deserialization
for Order instances using Python's struct module, aligned to single CPU cache lines (64 bytes)
for ultra-low latency shared-memory and ring buffer transport.
"""

from __future__ import annotations

from decimal import Decimal
import math
import mmap
import struct
import time
from typing import Any

from chronosmatch.models import Order, OrderSide

# ==============================================================================
# Binary Layout Specification
# ==============================================================================
# Byte order: Little-Endian (<), standard for modern x86_64 CPU memory and shared-memory IPC.
# Format string fields:
#   Q   : uint64 (8 bytes)  - timestamp_ns (nanoseconds since Unix epoch)
#   32s : bytes  (32 bytes) - order_id (null-padded UTF-8 string)
#   q   : int64  (8 bytes)  - price_ticks (fixed-point scaled price: price * 10^8)
#   I   : uint32 (4 bytes)  - quantity (contract / share count)
#   B   : uint8  (1 byte)   - side code (1 = BUY, 2 = SELL)
#   B   : uint8  (1 byte)   - price_type (1 = float, 2 = int, 3 = Decimal)
#   B   : uint8  (1 byte)   - has_sequence_number flag (0 = None, 1 = present)
#   B   : uint8  (1 byte)   - flags / reserved (0 by default)
#   Q   : uint64 (8 bytes)  - sequence_number (0 if None)
#
# Total Size: 8 + 32 + 8 + 4 + 1 + 1 + 1 + 1 + 8 = 64 bytes (Exact CPU Cache Line).
# ==============================================================================

ORDER_STRUCT_FORMAT: str = "<Q32sqIBBBBQ"
ORDER_STRUCT: struct.Struct = struct.Struct(ORDER_STRUCT_FORMAT)
ORDER_STRUCT_SIZE: int = ORDER_STRUCT.size  # Exactly 64 bytes

MAX_ORDER_ID_LENGTH: int = 32

# Price Scaling: 8 decimal places (10^8 = 100,000,000 ticks per 1.0 currency unit).
# Accommodates sub-cent equity ticks (0.0001), basis points, and 8-decimal cryptocurrency units.
PRICE_SCALE: int = 100_000_000

# Numeric Side Codes (stable constants)
SIDE_BUY_CODE: int = 1
SIDE_SELL_CODE: int = 2

# Price Type Indicators (ensures exact round-trip type preservation)
PRICE_TYPE_FLOAT: int = 1
PRICE_TYPE_INT: int = 2
PRICE_TYPE_DECIMAL: int = 3

# Numeric Boundaries
UINT32_MAX: int = (1 << 32) - 1
UINT64_MAX: int = (1 << 64) - 1
INT64_MAX: int = (1 << 63) - 1
INT64_MIN: int = -(1 << 63)


def _validate_and_encode_order_id(order_id: Any) -> bytes:
    """Validate and encode order_id into fixed 32-byte UTF-8 bytes.

    Args:
        order_id: Order identifier to validate and encode.

    Returns:
        Encoded UTF-8 bytes (length <= 32).

    Raises:
        TypeError: If order_id is not a string.
        ValueError: If order_id is empty, whitespace, contains null bytes, or exceeds 32 bytes.
    """
    if not isinstance(order_id, str):
        raise TypeError(f"order_id must be a string, got {type(order_id).__name__}")
    if not order_id.strip():
        raise ValueError("order_id cannot be empty or whitespace")
    try:
        raw_bytes = order_id.encode("utf-8")
    except UnicodeEncodeError as e:
        raise ValueError(f"order_id cannot be encoded to UTF-8: {e}") from e

    if len(raw_bytes) > MAX_ORDER_ID_LENGTH:
        raise ValueError(
            f"order_id '{order_id}' length ({len(raw_bytes)} bytes) exceeds maximum "
            f"supported length of {MAX_ORDER_ID_LENGTH} bytes"
        )
    if b"\x00" in raw_bytes:
        raise ValueError("order_id cannot contain null bytes")

    return raw_bytes


def _decode_order_id(raw_id: bytes) -> str:
    """Decode fixed 32-byte field into string.

    Args:
        raw_id: 32 bytes from unpacked binary struct.

    Returns:
        Clean UTF-8 decoded order ID string.

    Raises:
        ValueError: If decoded string is empty, whitespace, or invalid UTF-8.
    """
    id_bytes = raw_id.rstrip(b"\x00")
    if not id_bytes:
        raise ValueError("Decoded order_id cannot be empty")
    try:
        order_id = id_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"Decoded order_id contains invalid UTF-8: {e}") from e
    if not order_id.strip():
        raise ValueError("Decoded order_id cannot be whitespace only")
    return order_id


def _validate_and_encode_side(side: Any) -> int:
    """Validate and encode order side into numeric code (1 = BUY, 2 = SELL).

    Args:
        side: OrderSide enum or valid side string.

    Returns:
        1 for BUY, 2 for SELL.

    Raises:
        ValueError: If side is unknown or invalid.
    """
    if side == OrderSide.BUY or side == "BUY":
        return SIDE_BUY_CODE
    if side == OrderSide.SELL or side == "SELL":
        return SIDE_SELL_CODE
    raise ValueError(f"Invalid side '{side}'. Must be BUY or SELL.")


def _decode_side(side_code: int) -> OrderSide:
    """Decode numeric side code into OrderSide enum.

    Args:
        side_code: 1 for BUY, 2 for SELL.

    Returns:
        OrderSide.BUY or OrderSide.SELL.

    Raises:
        ValueError: If side_code is not 1 or 2.
    """
    if side_code == SIDE_BUY_CODE:
        return OrderSide.BUY
    if side_code == SIDE_SELL_CODE:
        return OrderSide.SELL
    raise ValueError(
        f"Unknown side code: {side_code}. Expected {SIDE_BUY_CODE} (BUY) or {SIDE_SELL_CODE} (SELL)."
    )


def _validate_and_scale_price(price: Any) -> tuple[int, int]:
    """Validate price and convert to scaled integer ticks.

    Args:
        price: Price value (int, float, or Decimal).

    Returns:
        Tuple of (price_ticks: int, price_type: int).

    Raises:
        TypeError: If price is a bool or unsupported type.
        ValueError: If price is non-finite, <= 0, overflows int64, or loses precision.
    """
    if isinstance(price, bool) or not isinstance(price, (int, float, Decimal)):
        raise TypeError(f"price must be a real number (int, float, Decimal), got {type(price).__name__}")
    if isinstance(price, float) and (math.isnan(price) or math.isinf(price)):
        raise ValueError(f"price must be finite, got {price}")
    if isinstance(price, Decimal) and (price.is_nan() or price.is_infinite()):
        raise ValueError(f"price must be finite, got {price}")
    if price <= 0:
        raise ValueError(f"price must be strictly positive (> 0), got {price}")

    if isinstance(price, int):
        price_type = PRICE_TYPE_INT
        ticks = price * PRICE_SCALE
    elif isinstance(price, float):
        price_type = PRICE_TYPE_FLOAT
        s = str(price)
        dot_idx = s.find(".")
        if dot_idx != -1 and (len(s) - dot_idx - 1) > 8:
            d = Decimal(s)
            d_ticks = d * Decimal(PRICE_SCALE)
            if d_ticks != d_ticks.to_integral_value():
                raise ValueError(
                    f"price {price} cannot be represented without losing precision at scale {PRICE_SCALE} "
                    f"(maximum 8 decimal places supported)"
                )
            ticks = int(d_ticks)
        else:
            ticks = round(price * PRICE_SCALE)
    elif isinstance(price, Decimal):
        price_type = PRICE_TYPE_DECIMAL
        d_ticks = price * Decimal(PRICE_SCALE)
        if d_ticks != d_ticks.to_integral_value():
            raise ValueError(
                f"price {price} cannot be represented without losing precision at scale {PRICE_SCALE} "
                f"(maximum 8 decimal places supported)"
            )
        ticks = int(d_ticks)
    else:
        raise TypeError(f"Unsupported price type: {type(price).__name__}")

    if ticks <= 0:
        raise ValueError(f"price ticks must be strictly positive (> 0), got {ticks}")
    if ticks > INT64_MAX:
        raise ValueError(f"price {price} exceeds maximum allowable binary limit")

    return ticks, price_type


def _reconstruct_price(price_ticks: int, price_type: int) -> float | int | Decimal:
    """Reconstruct original price value from integer ticks and price type code.

    Args:
        price_ticks: Integer tick count from binary struct.
        price_type: Type indicator code.

    Returns:
        Reconstructed price as float, int, or Decimal.

    Raises:
        ValueError: If price_ticks <= 0 or price_type is unknown.
    """
    if price_ticks <= 0:
        raise ValueError(f"price ticks must be strictly positive (> 0), got {price_ticks}")

    if price_type == PRICE_TYPE_FLOAT:
        return float(price_ticks) / PRICE_SCALE
    if price_type == PRICE_TYPE_INT:
        if price_ticks % PRICE_SCALE == 0:
            return price_ticks // PRICE_SCALE
        return float(price_ticks) / PRICE_SCALE
    if price_type == PRICE_TYPE_DECIMAL:
        return Decimal(price_ticks) / Decimal(PRICE_SCALE)
    raise ValueError(f"Unknown price type code: {price_type}")


def _validate_quantity(quantity: Any) -> int:
    """Validate quantity fits in uint32.

    Args:
        quantity: Order quantity to validate.

    Returns:
        Validated integer quantity.

    Raises:
        TypeError: If quantity is a boolean or non-integer.
        ValueError: If quantity <= 0 or exceeds uint32 max.
    """
    if isinstance(quantity, bool) or not isinstance(quantity, int):
        raise TypeError(f"quantity must be an integer, got {type(quantity).__name__}")
    if quantity <= 0:
        raise ValueError(f"quantity must be strictly positive (> 0), got {quantity}")
    if quantity > UINT32_MAX:
        raise ValueError(f"quantity {quantity} exceeds 32-bit unsigned integer maximum ({UINT32_MAX})")
    return quantity


def _validate_timestamp(timestamp: Any) -> int:
    """Validate timestamp fits in uint64.

    Args:
        timestamp: Nanosecond timestamp to validate.

    Returns:
        Validated integer timestamp.

    Raises:
        TypeError: If timestamp is a boolean or non-integer.
        ValueError: If timestamp <= 0 or exceeds uint64 max.
    """
    if isinstance(timestamp, bool) or not isinstance(timestamp, int):
        raise TypeError(f"timestamp must be an integer (nanoseconds), got {type(timestamp).__name__}")
    if timestamp <= 0:
        raise ValueError(f"timestamp must be strictly positive (> 0), got {timestamp}")
    if timestamp > UINT64_MAX:
        raise ValueError(f"timestamp {timestamp} exceeds 64-bit unsigned integer maximum ({UINT64_MAX})")
    return timestamp


def _validate_sequence_number(sequence_number: Any) -> tuple[int, int]:
    """Validate sequence_number and return (has_sequence_number, seq_num).

    Args:
        sequence_number: Optional integer sequence number.

    Returns:
        Tuple of (has_seq: int, seq_num: int).

    Raises:
        TypeError: If sequence_number is a boolean or not int/None.
        ValueError: If sequence_number < 0 or exceeds uint64 max.
    """
    if sequence_number is None:
        return 0, 0
    if isinstance(sequence_number, bool) or not isinstance(sequence_number, int):
        raise TypeError(f"sequence_number must be an integer or None, got {type(sequence_number).__name__}")
    if sequence_number < 0:
        raise ValueError(f"sequence_number must be non-negative (>= 0), got {sequence_number}")
    if sequence_number > UINT64_MAX:
        raise ValueError(f"sequence_number {sequence_number} exceeds 64-bit unsigned integer maximum ({UINT64_MAX})")
    return 1, sequence_number


def _validate_order_fields(
    order: Order,
) -> tuple[int, bytes, int, int, int, int, int, int, int]:
    """Validate all Order fields and return the packed tuple elements.

    Args:
        order: Valid Order instance to serialize.

    Returns:
        Tuple of (ts, raw_id, price_ticks, qty, side_code, price_type, has_seq, flags, seq_num).

    Raises:
        TypeError: If order is not an Order instance or has invalid field types.
        ValueError: If any field values are out of allowable range or cannot be represented.
    """
    if not isinstance(order, Order):
        raise TypeError(f"Expected Order instance, got {type(order).__name__}")

    raw_id = _validate_and_encode_order_id(order.order_id)
    side_code = _validate_and_encode_side(order.side)
    price_ticks, price_type = _validate_and_scale_price(order.price)
    qty = _validate_quantity(order.quantity)
    ts = _validate_timestamp(order.timestamp)
    has_seq, seq_num = _validate_sequence_number(order.sequence_number)
    flags = 0

    return (
        ts,
        raw_id,
        price_ticks,
        qty,
        side_code,
        price_type,
        has_seq,
        flags,
        seq_num,
    )


def _reconstruct_order(
    ts: int,
    raw_id: bytes,
    price_ticks: int,
    qty: int,
    side_code: int,
    price_type: int,
    has_seq: int,
    flags: int,
    seq_num: int,
) -> Order:
    """Validate unpacked fields and reconstruct an Order instance.

    Args:
        ts: Nanosecond timestamp.
        raw_id: Fixed 32-byte order ID field.
        price_ticks: Integer scaled price.
        qty: Integer quantity.
        side_code: Numeric side indicator (1 = BUY, 2 = SELL).
        price_type: Type indicator code (float, int, or Decimal).
        has_seq: Sequence number presence flag (0 = None, 1 = present).
        flags: Reserved flags byte.
        seq_num: Unsigned 64-bit sequence number.

    Returns:
        Reconstructed Order instance.

    Raises:
        ValueError: If any unpacked field value is out of allowable range or malformed.
    """
    if ts <= 0:
        raise ValueError(f"Decoded timestamp must be strictly positive (> 0), got {ts}")

    order_id = _decode_order_id(raw_id)
    side = _decode_side(side_code)
    price = _reconstruct_price(price_ticks, price_type)

    if qty <= 0:
        raise ValueError(f"Decoded quantity must be strictly positive (> 0), got {qty}")

    if has_seq == 0:
        seq = None
    elif has_seq == 1:
        seq = seq_num
    else:
        raise ValueError(f"Invalid has_sequence_number flag: {has_seq}")

    return Order(
        order_id=order_id,
        side=side,
        price=price,
        quantity=qty,
        timestamp=ts,
        sequence_number=seq,
    )


def encode_order(order: Order) -> bytes:
    """Encode an Order instance into a fixed 64-byte binary sequence.

    Uses Little-Endian format '<Q32sqIBBBBQ' conforming to single CPU cache-line alignment.

    Args:
        order: Valid Order instance to serialize.

    Returns:
        Exactly 64 bytes of packed binary data.

    Raises:
        TypeError: If order is not an Order instance or has invalid field types.
        ValueError: If any field values are out of allowable range or cannot be represented.
    """
    fields = _validate_order_fields(order)
    return ORDER_STRUCT.pack(*fields)


def encode_order_into(
    order: Order,
    buffer: bytearray | memoryview | mmap.mmap | Any,
    offset: int = 0,
) -> None:
    """Encode an Order directly into an existing writable buffer at the given offset.

    Facilitates zero-copy shared-memory ring buffer writes using struct.Struct.pack_into.

    Args:
        order: Order instance to serialize.
        buffer: Target mutable buffer (bytearray, memoryview, or mmap).
        offset: Byte offset into the buffer (default: 0).

    Raises:
        TypeError: If buffer is not mutable, or offset is not an integer.
        ValueError: If buffer cannot fit 64 bytes at the specified offset.
    """
    # 1. Validate buffer mutability and determine capacity
    if isinstance(buffer, memoryview):
        if buffer.readonly:
            raise TypeError("Expected mutable buffer, got read-only memoryview")
        buf_len = len(buffer)
    elif isinstance(buffer, (bytearray, mmap.mmap)):
        buf_len = len(buffer)
    else:
        try:
            mv = memoryview(buffer)
            if mv.readonly:
                raise TypeError(f"Expected mutable buffer, got read-only {type(buffer).__name__}")
            buf_len = len(mv)
        except TypeError:
            raise TypeError(f"Expected mutable buffer (bytearray, memoryview, or mmap), got {type(buffer).__name__}")

    # 2. Validate offset and capacity bounds
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise TypeError(f"offset must be an integer, got {type(offset).__name__}")
    if offset < 0 or offset + ORDER_STRUCT_SIZE > buf_len:
        raise ValueError(
            f"Buffer too small: offset {offset} + {ORDER_STRUCT_SIZE} exceeds buffer size {buf_len}"
        )

    # 3. Validate order fields
    fields = _validate_order_fields(order)

    # 4. Pack directly into buffer at offset (in-place zero-copy write)
    ORDER_STRUCT.pack_into(buffer, offset, *fields)


def decode_order_from(
    buffer: bytes | bytearray | memoryview | mmap.mmap | Any,
    offset: int = 0,
) -> Order:
    """Decode an Order from a buffer at the given offset.

    Facilitates zero-copy shared-memory ring buffer reads using struct.Struct.unpack_from.

    Args:
        buffer: Source bytes, bytearray, memoryview, or mmap buffer.
        offset: Byte offset into the buffer (default: 0).

    Returns:
        Reconstructed Order instance.

    Raises:
        TypeError: If buffer is not bytes-like or offset is not an integer.
        ValueError: If buffer does not contain at least 64 bytes at the specified offset,
                    or if any field value is invalid/malformed.
    """
    # 1. Validate buffer
    if isinstance(buffer, (bytes, bytearray, memoryview, mmap.mmap)):
        buf_len = len(buffer)
    else:
        try:
            buf_len = len(memoryview(buffer))
        except TypeError:
            raise TypeError(f"Expected bytes-like object, got {type(buffer).__name__}")

    # 2. Validate offset and bounds
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise TypeError(f"offset must be an integer, got {type(offset).__name__}")
    if offset < 0 or offset + ORDER_STRUCT_SIZE > buf_len:
        raise ValueError(
            f"Buffer too small: offset {offset} + {ORDER_STRUCT_SIZE} exceeds buffer size {buf_len}"
        )

    # 3. Unpack directly from buffer at offset (in-place zero-copy read)
    try:
        raw_fields = ORDER_STRUCT.unpack_from(buffer, offset)
    except struct.error as e:
        raise ValueError(f"Malformed binary order data: {e}") from e

    # 4. Reconstruct and validate Order
    return _reconstruct_order(*raw_fields)


def decode_order(data: bytes | bytearray | memoryview | mmap.mmap | Any) -> Order:
    """Decode a 64-byte binary order sequence back into an Order instance.

    Args:
        data: Exactly 64 bytes of binary order data.

    Returns:
        Reconstructed Order instance with original fields and types preserved.

    Raises:
        TypeError: If data is not bytes-like.
        ValueError: If data length is not 64 bytes or any field value is invalid/malformed.
    """
    if isinstance(data, (bytes, bytearray, memoryview, mmap.mmap)):
        data_len = len(data)
    else:
        try:
            data_len = len(memoryview(data))
        except TypeError:
            raise TypeError(f"Expected bytes-like object, got {type(data).__name__}")

    if data_len != ORDER_STRUCT_SIZE:
        raise ValueError(
            f"Expected binary order data of length {ORDER_STRUCT_SIZE} bytes, got {data_len} bytes"
        )

    return decode_order_from(data, 0)

