"""Automated test suite for the ChronosMatch binary order serialization protocol."""

from decimal import Decimal
import math
import mmap
import struct
import time

import pytest

from chronosmatch.binary_protocol import (
    MAX_ORDER_ID_LENGTH,
    ORDER_STRUCT,
    ORDER_STRUCT_FORMAT,
    ORDER_STRUCT_SIZE,
    PRICE_SCALE,
    PRICE_TYPE_DECIMAL,
    PRICE_TYPE_FLOAT,
    PRICE_TYPE_INT,
    SIDE_BUY_CODE,
    SIDE_SELL_CODE,
    UINT32_MAX,
    UINT64_MAX,
    decode_order,
    decode_order_from,
    encode_order,
    encode_order_into,
)
from chronosmatch.models import Order, OrderSide, Side


class TestBinaryProtocolEncoding:
    """Test suite for encoding valid Order objects into fixed 64-byte binary sequences."""

    def test_encode_valid_buy_order(self) -> None:
        """Verify successful encoding of a valid BUY order to exactly 64 bytes."""
        ts = 1700000000123456789
        order = Order(
            order_id="ord-buy-001",
            side=OrderSide.BUY,
            price=150.25,
            quantity=100,
            timestamp=ts,
            sequence_number=1,
        )

        data = encode_order(order)

        assert isinstance(data, bytes)
        assert len(data) == ORDER_STRUCT_SIZE
        assert len(data) == 64

        # Manually unpack to verify byte layout and field mapping
        (
            unpacked_ts,
            unpacked_id,
            unpacked_ticks,
            unpacked_qty,
            unpacked_side,
            unpacked_price_type,
            unpacked_has_seq,
            unpacked_flags,
            unpacked_seq,
        ) = ORDER_STRUCT.unpack(data)

        assert unpacked_ts == ts
        assert unpacked_id.rstrip(b"\x00").decode("utf-8") == "ord-buy-001"
        assert unpacked_ticks == int(150.25 * PRICE_SCALE)
        assert unpacked_qty == 100
        assert unpacked_side == SIDE_BUY_CODE
        assert unpacked_price_type == PRICE_TYPE_FLOAT
        assert unpacked_has_seq == 1
        assert unpacked_flags == 0
        assert unpacked_seq == 1

    def test_encode_valid_sell_order(self) -> None:
        """Verify successful encoding of a valid SELL order to exactly 64 bytes."""
        ts = 1700000000987654321
        order = Order(
            order_id="ord-sell-002",
            side=OrderSide.SELL,
            price=200.50,
            quantity=500,
            timestamp=ts,
            sequence_number=None,
        )

        data = encode_order(order)

        assert isinstance(data, bytes)
        assert len(data) == 64

        (
            unpacked_ts,
            unpacked_id,
            unpacked_ticks,
            unpacked_qty,
            unpacked_side,
            unpacked_price_type,
            unpacked_has_seq,
            unpacked_flags,
            unpacked_seq,
        ) = ORDER_STRUCT.unpack(data)

        assert unpacked_ts == ts
        assert unpacked_id.rstrip(b"\x00").decode("utf-8") == "ord-sell-002"
        assert unpacked_ticks == int(200.50 * PRICE_SCALE)
        assert unpacked_qty == 500
        assert unpacked_side == SIDE_SELL_CODE
        assert unpacked_price_type == PRICE_TYPE_FLOAT
        assert unpacked_has_seq == 0
        assert unpacked_seq == 0

    def test_encode_order_with_max_length_id(self) -> None:
        """Verify order ID of exactly 32 bytes (MAX_ORDER_ID_LENGTH) encodes without null padding."""
        max_id = "A" * MAX_ORDER_ID_LENGTH  # 32 characters
        order = Order(max_id, OrderSide.BUY, 100.0, 10)

        data = encode_order(order)
        assert len(data) == 64

        decoded = decode_order(data)
        assert decoded.order_id == max_id

    def test_encode_rejects_non_order_instance(self) -> None:
        """Verify encode_order rejects objects that are not Order instances."""
        with pytest.raises(TypeError, match="Expected Order instance"):
            encode_order("not_an_order")  # type: ignore[arg-type]


class TestBinaryProtocolDecodingAndRoundTrip:
    """Test suite for decoding binary data and verifying round-trip preservation."""

    def test_round_trip_buy_order_with_float_price(self) -> None:
        """Verify bit-accurate round trip for a BUY order with float price."""
        ts = 1680000000000000000
        original = Order(
            order_id="ord-round-01",
            side=OrderSide.BUY,
            price=123.4567,
            quantity=75,
            timestamp=ts,
            sequence_number=42,
        )

        encoded = encode_order(original)
        decoded = decode_order(encoded)

        assert decoded == original
        assert decoded.order_id == original.order_id
        assert decoded.side == original.side
        assert decoded.price == original.price
        assert type(decoded.price) is float
        assert decoded.quantity == original.quantity
        assert decoded.timestamp == original.timestamp
        assert decoded.sequence_number == original.sequence_number

    def test_round_trip_sell_order_with_int_price(self) -> None:
        """Verify bit-accurate round trip for a SELL order with int price."""
        ts = 1690000000000000000
        original = Order(
            order_id="ord-int-price",
            side=OrderSide.SELL,
            price=250,
            quantity=1000,
            timestamp=ts,
            sequence_number=0,
        )

        encoded = encode_order(original)
        decoded = decode_order(encoded)

        assert decoded == original
        assert decoded.price == 250
        assert type(decoded.price) is int
        assert decoded.sequence_number == 0

    def test_round_trip_with_decimal_price(self) -> None:
        """Verify bit-accurate round trip for an order with Decimal price."""
        ts = 1710000000000000000
        price = Decimal("150.25000001")  # 8 decimal places
        original = Order(
            order_id="ord-dec-01",
            side=OrderSide.BUY,
            price=price,
            quantity=10,
            timestamp=ts,
            sequence_number=999,
        )

        encoded = encode_order(original)
        decoded = decode_order(encoded)

        assert decoded == original
        assert decoded.price == price
        assert type(decoded.price) is Decimal

    def test_round_trip_with_none_sequence_number(self) -> None:
        """Verify sequence_number=None is preserved as None after round-trip."""
        original = Order(
            order_id="ord-none-seq",
            side=OrderSide.SELL,
            price=99.95,
            quantity=20,
            sequence_number=None,
        )

        encoded = encode_order(original)
        decoded = decode_order(encoded)

        assert decoded.sequence_number is None
        assert decoded == original


class TestValidationBeforePacking:
    """Test suite ensuring field constraints and types are validated before packing."""

    def test_reject_order_id_exceeding_max_length(self) -> None:
        """Verify order ID longer than 32 bytes is rejected with ValueError."""
        long_id = "A" * 33
        order = Order("ord-valid", OrderSide.BUY, 100.0, 10)
        object.__setattr__(order, "order_id", long_id)

        with pytest.raises(ValueError, match="exceeds maximum supported length of 32 bytes"):
            encode_order(order)

    def test_reject_order_id_with_null_bytes(self) -> None:
        """Verify order ID containing null byte is rejected."""
        order = Order("ord-valid", OrderSide.BUY, 100.0, 10)
        object.__setattr__(order, "order_id", "bad\x00id")

        with pytest.raises(ValueError, match="cannot contain null bytes"):
            encode_order(order)

    def test_reject_tampered_empty_order_id(self) -> None:
        """Verify empty or whitespace order_id is rejected."""
        order = Order("ord-valid", OrderSide.BUY, 100.0, 10)
        object.__setattr__(order, "order_id", "")

        with pytest.raises(ValueError, match="order_id cannot be empty or whitespace"):
            encode_order(order)

    def test_reject_tampered_invalid_side(self) -> None:
        """Verify unrecognized order side is rejected."""
        order = Order("ord-1", OrderSide.BUY, 100.0, 10)
        object.__setattr__(order, "side", "HOLD")

        with pytest.raises(ValueError, match="Invalid side 'HOLD'"):
            encode_order(order)

    @pytest.mark.parametrize("invalid_price", [0, -10.0, -1, float("nan"), float("inf")])
    def test_reject_non_positive_or_non_finite_price(self, invalid_price: object) -> None:
        """Verify non-positive or non-finite price is rejected."""
        order = Order("ord-1", OrderSide.BUY, 100.0, 10)
        object.__setattr__(order, "price", invalid_price)

        with pytest.raises(ValueError):
            encode_order(order)

    def test_reject_boolean_price_type(self) -> None:
        """Verify boolean price is strictly rejected with TypeError."""
        order = Order("ord-1", OrderSide.BUY, 100.0, 10)
        object.__setattr__(order, "price", True)

        with pytest.raises(TypeError, match="price must be a real number"):
            encode_order(order)

    def test_reject_price_precision_loss(self) -> None:
        """Verify prices exceeding 8 decimal places cannot be represented and raise ValueError."""
        order = Order("ord-1", OrderSide.BUY, 100.0, 10)
        # 9 decimal places
        object.__setattr__(order, "price", Decimal("100.123456789"))

        with pytest.raises(ValueError, match="cannot be represented without losing precision"):
            encode_order(order)

    def test_reject_price_overflowing_int64(self) -> None:
        """Verify prices exceeding int64 ticks limit raise ValueError."""
        huge_price = Decimal("1000000000000000000")  # ticks would exceed 2^63 - 1
        order = Order("ord-1", OrderSide.BUY, 100.0, 10)
        object.__setattr__(order, "price", huge_price)

        with pytest.raises(ValueError, match="exceeds maximum allowable binary limit"):
            encode_order(order)

    @pytest.mark.parametrize("bad_qty", [0, -5, True, False, 1.5, "10"])
    def test_reject_invalid_quantity(self, bad_qty: object) -> None:
        """Verify non-positive or non-integer quantities are rejected."""
        order = Order("ord-1", OrderSide.BUY, 100.0, 10)
        object.__setattr__(order, "quantity", bad_qty)

        with pytest.raises((ValueError, TypeError)):
            encode_order(order)

    def test_reject_quantity_overflowing_uint32(self) -> None:
        """Verify quantity exceeding uint32 max is rejected."""
        order = Order("ord-1", OrderSide.BUY, 100.0, 10)
        object.__setattr__(order, "quantity", UINT32_MAX + 1)

        with pytest.raises(ValueError, match="exceeds 32-bit unsigned integer maximum"):
            encode_order(order)

    @pytest.mark.parametrize("bad_ts", [0, -1, True, False, "12345", 1.23])
    def test_reject_invalid_timestamp(self, bad_ts: object) -> None:
        """Verify non-positive or non-integer timestamps are rejected."""
        order = Order("ord-1", OrderSide.BUY, 100.0, 10)
        object.__setattr__(order, "timestamp", bad_ts)

        with pytest.raises((ValueError, TypeError)):
            encode_order(order)

    def test_reject_timestamp_overflowing_uint64(self) -> None:
        """Verify timestamp exceeding uint64 max is rejected."""
        order = Order("ord-1", OrderSide.BUY, 100.0, 10)
        object.__setattr__(order, "timestamp", UINT64_MAX + 1)

        with pytest.raises(ValueError, match="exceeds 64-bit unsigned integer maximum"):
            encode_order(order)

    @pytest.mark.parametrize("bad_seq", [-1, True, False, "0", 3.14])
    def test_reject_invalid_sequence_number(self, bad_seq: object) -> None:
        """Verify negative or non-integer sequence numbers are rejected."""
        order = Order("ord-1", OrderSide.BUY, 100.0, 10)
        object.__setattr__(order, "sequence_number", bad_seq)

        with pytest.raises((ValueError, TypeError)):
            encode_order(order)

    def test_reject_sequence_number_overflowing_uint64(self) -> None:
        """Verify sequence_number exceeding uint64 max is rejected."""
        order = Order("ord-1", OrderSide.BUY, 100.0, 10)
        object.__setattr__(order, "sequence_number", UINT64_MAX + 1)

        with pytest.raises(ValueError, match="exceeds 64-bit unsigned integer maximum"):
            encode_order(order)


class TestDecodeValidationAndMalformedBinaryData:
    """Test suite ensuring decoding rejects invalid byte length, unknown codes, and corrupted data."""

    def test_reject_input_bytes_too_short(self) -> None:
        """Verify input data with fewer than 64 bytes is rejected with ValueError."""
        short_data = b"\x00" * 63
        with pytest.raises(ValueError, match="Expected binary order data of length 64 bytes, got 63 bytes"):
            decode_order(short_data)

    def test_reject_input_bytes_too_long(self) -> None:
        """Verify input data with more than 64 bytes is rejected with ValueError."""
        long_data = b"\x00" * 65
        with pytest.raises(ValueError, match="Expected binary order data of length 64 bytes, got 65 bytes"):
            decode_order(long_data)

    def test_reject_non_bytes_like_input(self) -> None:
        """Verify decode_order rejects non-bytes-like objects with TypeError."""
        with pytest.raises(TypeError, match="Expected bytes-like object"):
            decode_order("string_is_not_bytes")  # type: ignore[arg-type]

    def test_reject_unknown_side_code(self) -> None:
        """Verify binary data with side code other than 1 or 2 is rejected."""
        order = Order("ord-1", OrderSide.BUY, 100.0, 10)
        data = bytearray(encode_order(order))

        # Side code is at byte offset 52
        data[52] = 3  # Invalid side code

        with pytest.raises(ValueError, match="Unknown side code: 3"):
            decode_order(bytes(data))

    def test_reject_zero_side_code(self) -> None:
        """Verify binary data with side code 0 is rejected."""
        order = Order("ord-1", OrderSide.BUY, 100.0, 10)
        data = bytearray(encode_order(order))
        data[52] = 0

        with pytest.raises(ValueError, match="Unknown side code: 0"):
            decode_order(bytes(data))

    def test_reject_unknown_price_type(self) -> None:
        """Verify binary data with unrecognized price_type code is rejected."""
        order = Order("ord-1", OrderSide.BUY, 100.0, 10)
        data = bytearray(encode_order(order))

        # price_type is at byte offset 53
        data[53] = 99

        with pytest.raises(ValueError, match="Unknown price type code: 99"):
            decode_order(bytes(data))

    def test_reject_invalid_has_sequence_number_flag(self) -> None:
        """Verify has_sequence_number flag other than 0 or 1 is rejected."""
        order = Order("ord-1", OrderSide.BUY, 100.0, 10)
        data = bytearray(encode_order(order))

        # has_sequence_number is at byte offset 54
        data[54] = 2

        with pytest.raises(ValueError, match="Invalid has_sequence_number flag: 2"):
            decode_order(bytes(data))

    def test_reject_empty_or_null_decoded_order_id(self) -> None:
        """Verify binary data where order_id field is all null bytes is rejected."""
        order = Order("ord-1", OrderSide.BUY, 100.0, 10)
        data = bytearray(encode_order(order))

        # order_id is at offset 8 to 40
        data[8:40] = b"\x00" * 32

        with pytest.raises(ValueError, match="Decoded order_id cannot be empty"):
            decode_order(bytes(data))

    def test_reject_decoded_order_id_with_invalid_utf8(self) -> None:
        """Verify binary data where order_id contains invalid UTF-8 is rejected."""
        order = Order("ord-1", OrderSide.BUY, 100.0, 10)
        data = bytearray(encode_order(order))

        # Overwrite with invalid UTF-8 continuation byte
        data[8:12] = b"\xff\xfe\xfd\xfc"

        with pytest.raises(ValueError, match="Decoded order_id contains invalid UTF-8"):
            decode_order(bytes(data))

    def test_reject_decoded_zero_timestamp(self) -> None:
        """Verify binary data with timestamp 0 is rejected."""
        order = Order("ord-1", OrderSide.BUY, 100.0, 10)
        data = bytearray(encode_order(order))

        # timestamp is at offset 0 to 8
        data[0:8] = b"\x00" * 8

        with pytest.raises(ValueError, match="Decoded timestamp must be strictly positive"):
            decode_order(bytes(data))

    def test_reject_decoded_zero_price_ticks(self) -> None:
        """Verify binary data with price ticks 0 is rejected."""
        order = Order("ord-1", OrderSide.BUY, 100.0, 10)
        data = bytearray(encode_order(order))

        # price_ticks is at offset 40 to 48
        data[40:48] = b"\x00" * 8

        with pytest.raises(ValueError, match="price ticks must be strictly positive"):
            decode_order(bytes(data))

    def test_reject_decoded_zero_quantity(self) -> None:
        """Verify binary data with quantity 0 is rejected."""
        order = Order("ord-1", OrderSide.BUY, 100.0, 10)
        data = bytearray(encode_order(order))

        # quantity is at offset 48 to 52
        data[48:52] = b"\x00" * 4

        with pytest.raises(ValueError, match="Decoded quantity must be strictly positive"):
            decode_order(bytes(data))


class TestBufferAndMemoryViewOperations:
    """Test suite for zero-copy buffer packing and unpacking."""

    def test_encode_into_and_decode_from_bytearray(self) -> None:
        """Verify packing into and unpacking from a mutable bytearray buffer."""
        buf = bytearray(128)  # Space for two 64-byte orders
        ord1 = Order("ord-buf-1", OrderSide.BUY, 100.0, 10)
        ord2 = Order("ord-buf-2", OrderSide.SELL, 105.0, 20)

        encode_order_into(ord1, buf, offset=0)
        encode_order_into(ord2, buf, offset=64)

        decoded1 = decode_order_from(buf, offset=0)
        decoded2 = decode_order_from(buf, offset=64)

        assert decoded1 == ord1
        assert decoded2 == ord2

    def test_encode_into_and_decode_from_memoryview(self) -> None:
        """Verify operations using Python memoryview directly."""
        raw_buf = bytearray(64)
        view = memoryview(raw_buf)
        order = Order("ord-view-1", OrderSide.BUY, 99.50, 50)

        encode_order_into(order, view, offset=0)
        decoded = decode_order_from(view, offset=0)

        assert decoded == order

    def test_encode_into_rejects_insufficient_buffer(self) -> None:
        """Verify buffer smaller than 64 bytes raises ValueError."""
        small_buf = bytearray(63)
        order = Order("ord-1", OrderSide.BUY, 100.0, 10)

        with pytest.raises(ValueError, match="Buffer too small"):
            encode_order_into(order, small_buf, offset=0)

    def test_decode_from_rejects_insufficient_buffer(self) -> None:
        """Verify reading past buffer boundary raises ValueError."""
        small_buf = bytearray(64)
        with pytest.raises(ValueError, match="Buffer too small"):
            decode_order_from(small_buf, offset=1)

    def test_encoding_decoding_equivalence_across_price_types(self) -> None:
        """Verify encode_order_into produces bit-identical output to encode_order and decodes equivalently."""
        sample_orders = [
            Order("eq-float", OrderSide.BUY, 123.45, 100, timestamp=1700000000000000001, sequence_number=10),
            Order("eq-int", OrderSide.SELL, 250, 50, timestamp=1700000000000000002, sequence_number=None),
            Order("eq-dec", OrderSide.BUY, Decimal("99.98765432"), 1, timestamp=1700000000000000003, sequence_number=0),
        ]
        buf = bytearray(256)
        for i, order in enumerate(sample_orders):
            offset = i * 64 + 16  # Test nonzero offset
            encode_order_into(order, buf, offset=offset)

            # Equivalence with encode_order
            expected_bytes = encode_order(order)
            assert bytes(buf[offset : offset + 64]) == expected_bytes

            # Equivalence of decode_order and decode_order_from
            from_buf = decode_order_from(buf, offset=offset)
            from_bytes = decode_order(expected_bytes)

            assert from_buf == order
            assert from_bytes == order
            assert from_buf == from_bytes
            assert type(from_buf.price) is type(order.price)
            assert from_buf.sequence_number == order.sequence_number

    def test_encode_into_and_decode_from_mmap(self) -> None:
        """Verify in-place zero-copy serialization directly with mmap regions."""
        mm = mmap.mmap(-1, 256)
        try:
            ord1 = Order("mm-01", OrderSide.BUY, 101.50, 200, sequence_number=1)
            ord2 = Order("mm-02", OrderSide.SELL, 102.75, 300, sequence_number=2)

            encode_order_into(ord1, mm, offset=0)
            encode_order_into(ord2, mm, offset=64)

            # Check that bytes in mmap match encode_order
            assert bytes(mm[0:64]) == encode_order(ord1)
            assert bytes(mm[64:128]) == encode_order(ord2)

            # Decode directly from mmap
            res1 = decode_order_from(mm, offset=0)
            res2 = decode_order_from(mm, offset=64)

            assert res1 == ord1
            assert res2 == ord2
        finally:
            mm.close()

    def test_arbitrary_nonzero_offsets(self) -> None:
        """Verify packing and unpacking at arbitrary nonzero offsets within larger buffers."""
        offsets = [7, 13, 31, 64, 127]
        order = Order("ord-arb-offset", OrderSide.BUY, 77.25, 42, sequence_number=99)
        expected_bytes = encode_order(order)

        for off in offsets:
            buf = bytearray(off + 64 + 32)
            encode_order_into(order, buf, offset=off)
            assert bytes(buf[off : off + 64]) == expected_bytes

            decoded = decode_order_from(buf, offset=off)
            assert decoded == order

    def test_exact_capacity_boundaries(self) -> None:
        """Verify exact boundary fits succeed while 1-byte deficits fail."""
        buf = bytearray(64)
        order = Order("ord-boundary", OrderSide.BUY, 10.0, 1)

        # Exact fit at offset 0
        encode_order_into(order, buf, offset=0)
        assert decode_order_from(buf, offset=0) == order

        # 1-byte overflow at offset 1
        with pytest.raises(ValueError, match="Buffer too small"):
            encode_order_into(order, buf, offset=1)
        with pytest.raises(ValueError, match="Buffer too small"):
            decode_order_from(buf, offset=1)

        # Buffer of size 128
        buf128 = bytearray(128)
        encode_order_into(order, buf128, offset=64)
        assert decode_order_from(buf128, offset=64) == order

        with pytest.raises(ValueError, match="Buffer too small"):
            encode_order_into(order, buf128, offset=65)
        with pytest.raises(ValueError, match="Buffer too small"):
            decode_order_from(buf128, offset=65)

    def test_invalid_offset_types_raise_type_error(self) -> None:
        """Verify non-integer offsets raise TypeError for both encode_order_into and decode_order_from."""
        buf = bytearray(128)
        order = Order("ord-type", OrderSide.BUY, 10.0, 1)

        for bad_offset in ["0", 1.5, True, False, None]:
            with pytest.raises(TypeError, match="offset must be an integer"):
                encode_order_into(order, buf, offset=bad_offset)  # type: ignore[arg-type]
            with pytest.raises(TypeError, match="offset must be an integer"):
                decode_order_from(buf, offset=bad_offset)  # type: ignore[arg-type]

    def test_negative_offsets_raise_value_error(self) -> None:
        """Verify negative offsets raise ValueError."""
        buf = bytearray(128)
        order = Order("ord-neg", OrderSide.BUY, 10.0, 1)

        for neg_offset in [-1, -64, -100]:
            with pytest.raises(ValueError, match="Buffer too small"):
                encode_order_into(order, buf, offset=neg_offset)
            with pytest.raises(ValueError, match="Buffer too small"):
                decode_order_from(buf, offset=neg_offset)

    def test_reject_read_only_buffers_for_encoding(self) -> None:
        """Verify immutable/read-only buffers are rejected with TypeError when encoding."""
        order = Order("ord-ro", OrderSide.BUY, 10.0, 1)

        # Immutable bytes
        ro_bytes = bytes(64)
        with pytest.raises(TypeError, match="Expected mutable buffer"):
            encode_order_into(order, ro_bytes, offset=0)

        # Read-only memoryview
        ro_view = memoryview(ro_bytes)
        assert ro_view.readonly
        with pytest.raises(TypeError, match="Expected mutable buffer"):
            encode_order_into(order, ro_view, offset=0)

    def test_decode_from_corrupt_record_at_nonzero_offset(self) -> None:
        """Verify malformed binary record at nonzero offset raises ValueError."""
        buf = bytearray(128)
        order = Order("ord-valid", OrderSide.BUY, 100.0, 10)
        encode_order_into(order, buf, offset=30)

        # Corrupt side code at offset 30 + 52 = 82
        buf[82] = 99
        with pytest.raises(ValueError, match="Unknown side code: 99"):
            decode_order_from(buf, offset=30)

