"""Comprehensive test suite for ChronosMatch shared-memory IPC module."""

from __future__ import annotations

from decimal import Decimal
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any

import pytest

from chronosmatch.binary_protocol import (
    ORDER_STRUCT,
    ORDER_STRUCT_SIZE,
    PRICE_SCALE,
    SIDE_BUY_CODE,
    SIDE_SELL_CODE,
    encode_order,
)
from chronosmatch.models import Order, OrderSide
from chronosmatch.shared_memory import (
    MalformedRecordError,
    SharedMemoryBoundsError,
    SharedMemoryClosedError,
    SharedMemoryConsumer,
    SharedMemoryError,
    SharedMemoryProducer,
    SharedMemoryReadOnlyError,
    SharedMemoryRecordSizeError,
    SharedMemoryRegion,
)

# ==============================================================================
# Helper Multiprocessing Functions (must be top-level for Windows spawn support)
# ==============================================================================


def _producer_worker(filepath_str: str, order_id: str, price: float, qty: int) -> None:
    """Worker process that writes an order into shared memory."""
    order = Order(
        order_id=order_id,
        side=OrderSide.BUY,
        price=price,
        quantity=qty,
        timestamp=1700000000123456789,
        sequence_number=101,
    )
    with SharedMemoryProducer.create(filepath_str, size=4096) as producer:
        producer.write_order(order, offset=0)
        producer.flush()


def _consumer_worker(filepath_str: str, queue: mp.Queue) -> None:
    """Worker process that reads an order from shared memory."""
    with SharedMemoryConsumer.open(filepath_str, read_only=True) as consumer:
        order = consumer.read_order(offset=0)
        queue.put((order.order_id, order.side.value, float(order.price), order.quantity, order.sequence_number))


# ==============================================================================
# 1. Lifecycle & Resource Management Tests
# ==============================================================================


class TestSharedMemoryLifecycle:
    """Test suite verifying creation, opening, closing, and unlinking of shared memory."""

    def test_create_and_open_roundtrip(self, tmp_path: Path) -> None:
        """Verify creating a backing file and subsequently opening it across instances."""
        filepath = tmp_path / "shm_test.bin"
        size = 2048

        # Create and write test bytes
        shm_writer = SharedMemoryRegion.create(filepath, size=size)
        assert shm_writer.filepath == filepath.resolve()
        assert shm_writer.size == size
        assert not shm_writer.is_closed
        assert not shm_writer.is_read_only
        assert filepath.exists()
        assert filepath.stat().st_size == size

        test_payload = b"ChronosMatch-IPC-Header" + b"\x00" * 41
        shm_writer.write_record(0, test_payload)
        shm_writer.close()
        assert shm_writer.is_closed

        # Reopen existing backing file
        shm_reader = SharedMemoryRegion.open(filepath)
        assert shm_reader.size == size
        assert not shm_reader.is_closed
        assert shm_reader.read_record(0) == test_payload
        shm_reader.close()
        assert shm_reader.is_closed

    def test_open_read_only_mode(self, tmp_path: Path) -> None:
        """Verify opening an existing mapping in read-only mode prevents writes."""
        filepath = tmp_path / "shm_ro.bin"
        size = 1024

        with SharedMemoryRegion.create(filepath, size=size) as creator:
            creator.write_bytes(0, b"ReadOnlyTestData")

        with SharedMemoryRegion.open(filepath, read_only=True) as ro_shm:
            assert ro_shm.is_read_only
            assert ro_shm.read_bytes(0, 16) == b"ReadOnlyTestData"

            # Attempted write must raise SharedMemoryReadOnlyError
            with pytest.raises(SharedMemoryReadOnlyError, match="read-only"):
                ro_shm.write_bytes(0, b"NewData")

            with pytest.raises(SharedMemoryReadOnlyError, match="read-only"):
                ro_shm.write_record(0, b"\x01" * ORDER_STRUCT_SIZE)

            order = Order("ord-ro", "BUY", 100.0, 10)
            with pytest.raises(SharedMemoryReadOnlyError, match="read-only"):
                ro_shm.write_order(order, offset=0)

    def test_create_anonymous_mapping(self) -> None:
        """Verify in-memory anonymous mapping functionality."""
        size = 1024
        shm = SharedMemoryRegion.create_anonymous(size=size)
        assert shm.filepath is None
        assert shm.size == size
        assert not shm.is_closed
        assert not shm.is_read_only

        order = Order("anon-001", "SELL", 250.75, 50)
        shm.write_order(order, offset=0)
        read_order = shm.read_order(offset=0)
        assert read_order.order_id == "anon-001"
        assert read_order.price == 250.75

        # Unlink on anonymous mapping is a safe close
        shm.unlink()
        assert shm.is_closed

    def test_context_manager_automatic_cleanup(self, tmp_path: Path) -> None:
        """Verify context manager automatically closes the memory mapping."""
        filepath = tmp_path / "shm_ctx.bin"

        with SharedMemoryRegion.create(filepath, size=512) as shm:
            assert not shm.is_closed
            shm.write_bytes(0, b"ContextPayload")

        assert shm.is_closed
        with pytest.raises(SharedMemoryClosedError):
            shm.read_bytes(0, 14)

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        """Verify calling close() repeatedly does not cause errors."""
        filepath = tmp_path / "shm_idem.bin"
        shm = SharedMemoryRegion.create(filepath, size=256)
        shm.close()
        assert shm.is_closed
        # Subsequent close calls should be no-ops
        shm.close()
        shm.close()
        assert shm.is_closed

    def test_unlink_permanently_removes_file(self, tmp_path: Path) -> None:
        """Verify unlink() closes mappings and deletes the backing file cleanly on Windows."""
        filepath = tmp_path / "shm_unlink.bin"
        shm = SharedMemoryRegion.create(filepath, size=512)
        assert filepath.exists()

        shm.unlink()
        assert shm.is_closed
        assert not filepath.exists()

        # Idempotent: unlinking already unlinked file does not raise
        shm.unlink()

    def test_capacity_records_calculation(self, tmp_path: Path) -> None:
        """Verify capacity_records returns the exact number of 64-byte slots."""
        filepath = tmp_path / "shm_cap.bin"
        # 10 records = 640 bytes. Allocate 700 bytes (should fit 10 records)
        with SharedMemoryRegion.create(filepath, size=700) as shm:
            assert shm.capacity_records == 700 // ORDER_STRUCT_SIZE
            assert shm.capacity_records == 10

    def test_flush_synchronization(self, tmp_path: Path) -> None:
        """Verify flush() writes memory pages through to disk without errors."""
        filepath = tmp_path / "shm_flush.bin"
        with SharedMemoryRegion.create(filepath, size=256) as shm:
            shm.write_bytes(0, b"FlushCheck")
            shm.flush()

    def test_create_validation_errors(self, tmp_path: Path) -> None:
        """Verify argument validation when creating shared-memory regions."""
        valid_path = tmp_path / "valid.bin"

        # Invalid size types and values
        with pytest.raises(TypeError, match="size must be an integer"):
            SharedMemoryRegion.create(valid_path, size="1024")  # type: ignore

        with pytest.raises(TypeError, match="size must be an integer"):
            SharedMemoryRegion.create(valid_path, size=True)  # type: ignore

        with pytest.raises(ValueError, match="strictly positive"):
            SharedMemoryRegion.create(valid_path, size=0)

        with pytest.raises(ValueError, match="strictly positive"):
            SharedMemoryRegion.create(valid_path, size=-512)

        # Missing size when create=True
        with pytest.raises(ValueError, match="size is required"):
            SharedMemoryRegion(valid_path, create=True)

        # Invalid filepath
        with pytest.raises(TypeError, match="filepath must be a str or Path"):
            SharedMemoryRegion.create(12345, size=1024)  # type: ignore

        with pytest.raises(ValueError, match="filepath cannot be empty"):
            SharedMemoryRegion.create("   ", size=1024)

    def test_open_validation_errors(self, tmp_path: Path) -> None:
        """Verify argument validation when opening shared-memory regions."""
        non_existent = tmp_path / "does_not_exist.bin"
        with pytest.raises(FileNotFoundError, match="Backing file not found"):
            SharedMemoryRegion.open(non_existent)

        # Empty backing file
        empty_file = tmp_path / "empty.bin"
        empty_file.touch()
        with pytest.raises(ValueError, match="is empty"):
            SharedMemoryRegion.open(empty_file)

        # Requested size exceeding backing file
        valid_file = tmp_path / "small.bin"
        with SharedMemoryRegion.create(valid_file, size=256) as shm:
            shm.write_bytes(0, b"x" * 256)

        with pytest.raises(ValueError, match="exceeds backing file size"):
            SharedMemoryRegion(valid_file, size=512, create=False)

        # Invalid size values when opening
        with pytest.raises(ValueError, match="strictly positive"):
            SharedMemoryRegion(valid_file, size=0, create=False)

        with pytest.raises(TypeError, match="size must be an integer"):
            SharedMemoryRegion(valid_file, size="bad", create=False)  # type: ignore


# ==============================================================================
# 2. Binary Record I/O Tests
# ==============================================================================


class TestSharedMemoryBinaryRecordIO:
    """Test suite verifying reading and writing fixed-size binary records."""

    def test_write_and_read_valid_binary_record(self, tmp_path: Path) -> None:
        """Verify writing and reading an exact 64-byte binary record."""
        filepath = tmp_path / "shm_records.bin"
        with SharedMemoryRegion.create(filepath, size=1024) as shm:
            record_data = b"R" * ORDER_STRUCT_SIZE
            shm.write_record(0, record_data)

            read_data = shm.read_record(0)
            assert isinstance(read_data, bytes)
            assert len(read_data) == ORDER_STRUCT_SIZE
            assert read_data == record_data

    def test_multiple_records_at_distinct_offsets(self, tmp_path: Path) -> None:
        """Verify writing and reading multiple distinct records at separated offsets."""
        filepath = tmp_path / "shm_multi_records.bin"
        num_records = 8
        total_size = num_records * ORDER_STRUCT_SIZE

        with SharedMemoryRegion.create(filepath, size=total_size) as shm:
            records = [
                f"Record-{i:03d}-".encode("utf-8") + b"#" * (ORDER_STRUCT_SIZE - 11)
                for i in range(num_records)
            ]

            # Write all records at distinct offsets
            for i, record in enumerate(records):
                offset = i * ORDER_STRUCT_SIZE
                shm.write_record(offset, record)

            # Read all records back and verify
            for i, record in enumerate(records):
                offset = i * ORDER_STRUCT_SIZE
                assert shm.read_record(offset) == record

    def test_invalid_record_sizes(self, tmp_path: Path) -> None:
        """Verify handling of invalid record sizes."""
        filepath = tmp_path / "shm_invalid_size.bin"
        with SharedMemoryRegion.create(filepath, size=1024) as shm:
            # Under-sized record (63 bytes instead of 64)
            with pytest.raises(SharedMemoryRecordSizeError, match="Record size mismatch"):
                shm.write_record(0, b"x" * 63)

            # Over-sized record (65 bytes instead of 64)
            with pytest.raises(SharedMemoryRecordSizeError, match="Record size mismatch"):
                shm.write_record(0, b"x" * 65)

            # Empty record
            with pytest.raises(SharedMemoryRecordSizeError, match="Record size mismatch"):
                shm.write_record(0, b"")

            # Non-bytes data
            with pytest.raises(TypeError, match="Expected bytes-like data"):
                shm.write_record(0, "string_is_not_bytes")  # type: ignore

            # Custom record_size parameter
            custom_data = b"Custom32BytesData123456789012345"
            assert len(custom_data) == 32
            shm.write_record(0, custom_data, record_size=32)
            assert shm.read_record(0, record_size=32) == custom_data

            with pytest.raises(SharedMemoryRecordSizeError):
                shm.write_record(0, custom_data, record_size=64)

            # Invalid record_size parameter
            with pytest.raises(ValueError, match="strictly positive"):
                shm.write_record(0, custom_data, record_size=0)

            with pytest.raises(ValueError, match="strictly positive"):
                shm.read_record(0, record_size=0)

            with pytest.raises(ValueError, match="strictly positive"):
                shm.write_record(0, custom_data, record_size=-10)


# ==============================================================================
# 3. Offsets and Buffer Boundaries Tests
# ==============================================================================


class TestSharedMemoryBoundaries:
    """Test suite validating boundary checks, out-of-range offsets, and negative indices."""

    def test_negative_offsets(self, tmp_path: Path) -> None:
        """Verify negative offsets are rejected."""
        filepath = tmp_path / "shm_bounds.bin"
        with SharedMemoryRegion.create(filepath, size=512) as shm:
            dummy_record = b"x" * ORDER_STRUCT_SIZE
            order = Order("ord-neg", "BUY", 10.0, 1)

            with pytest.raises(SharedMemoryBoundsError, match="cannot be negative"):
                shm.write_record(-1, dummy_record)

            with pytest.raises(SharedMemoryBoundsError, match="cannot be negative"):
                shm.read_record(-1)

            with pytest.raises(SharedMemoryBoundsError, match="cannot be negative"):
                shm.write_bytes(-64, b"x")

            with pytest.raises(SharedMemoryBoundsError, match="cannot be negative"):
                shm.read_bytes(-10, 5)

            with pytest.raises(SharedMemoryBoundsError, match="cannot be negative"):
                shm.write_order(order, offset=-64)

            with pytest.raises(SharedMemoryBoundsError, match="cannot be negative"):
                shm.read_order(offset=-64)

    def test_out_of_range_offsets(self, tmp_path: Path) -> None:
        """Verify offsets extending past the region boundary are rejected."""
        size = 256  # Fits exactly 4 records of 64 bytes (offsets 0, 64, 128, 192)
        filepath = tmp_path / "shm_overflow.bin"
        with SharedMemoryRegion.create(filepath, size=size) as shm:
            dummy_record = b"x" * ORDER_STRUCT_SIZE
            order = Order("ord-over", "BUY", 100.0, 5)

            # Exactly at size boundary (offset 256 has 0 bytes left)
            with pytest.raises(SharedMemoryBoundsError, match="exceeds region size"):
                shm.write_record(size, dummy_record)

            with pytest.raises(SharedMemoryBoundsError, match="exceeds region size"):
                shm.read_record(size)

            # Partial boundary overflow (offset 200 + 64 = 264 > 256)
            with pytest.raises(SharedMemoryBoundsError, match="exceeds region size"):
                shm.write_record(200, dummy_record)

            with pytest.raises(SharedMemoryBoundsError, match="exceeds region size"):
                shm.read_record(200)

            with pytest.raises(SharedMemoryBoundsError, match="exceeds region size"):
                shm.write_order(order, offset=200)

            with pytest.raises(SharedMemoryBoundsError, match="exceeds region size"):
                shm.read_order(offset=200)

            # Distant out-of-range offset
            with pytest.raises(SharedMemoryBoundsError, match="exceeds region size"):
                shm.write_bytes(1000, b"x")

            with pytest.raises(SharedMemoryBoundsError, match="exceeds region size"):
                shm.read_bytes(1000, 10)

    def test_non_integer_offsets(self, tmp_path: Path) -> None:
        """Verify non-integer offset types raise TypeError."""
        filepath = tmp_path / "shm_type.bin"
        with SharedMemoryRegion.create(filepath, size=512) as shm:
            dummy_record = b"x" * ORDER_STRUCT_SIZE

            with pytest.raises(TypeError, match="offset must be an integer"):
                shm.write_record(10.5, dummy_record)  # type: ignore

            with pytest.raises(TypeError, match="offset must be an integer"):
                shm.read_record("0")  # type: ignore

            with pytest.raises(TypeError, match="offset must be an integer"):
                shm.write_record(True, dummy_record)  # type: ignore

            with pytest.raises(TypeError, match="length must be an integer"):
                shm.read_bytes(0, 5.5)  # type: ignore

            with pytest.raises(TypeError, match="length must be an integer"):
                shm.read_bytes(0, False)  # type: ignore


# ==============================================================================
# 4. Operations After Closing Mapping Tests
# ==============================================================================


class TestSharedMemoryClosedOperations:
    """Test suite verifying that all operations raise SharedMemoryClosedError once closed."""

    def test_operations_fail_after_close(self, tmp_path: Path) -> None:
        """Verify attempting any operation on a closed region raises SharedMemoryClosedError."""
        filepath = tmp_path / "shm_closed.bin"
        shm = SharedMemoryRegion.create(filepath, size=512)
        dummy_record = b"x" * ORDER_STRUCT_SIZE
        order = Order("ord-closed", "BUY", 10.0, 1)

        shm.close()
        assert shm.is_closed

        with pytest.raises(SharedMemoryClosedError, match="closed"):
            shm.write_record(0, dummy_record)

        with pytest.raises(SharedMemoryClosedError, match="closed"):
            shm.read_record(0)

        with pytest.raises(SharedMemoryClosedError, match="closed"):
            shm.write_bytes(0, b"x")

        with pytest.raises(SharedMemoryClosedError, match="closed"):
            shm.read_bytes(0, 10)

        with pytest.raises(SharedMemoryClosedError, match="closed"):
            shm.write_order(order, offset=0)

        with pytest.raises(SharedMemoryClosedError, match="closed"):
            shm.read_order(offset=0)

        with pytest.raises(SharedMemoryClosedError, match="closed"):
            shm.flush()

        with pytest.raises(SharedMemoryClosedError, match="closed"):
            _ = shm.mmap


# ==============================================================================
# 5. Integration With Phase 6 Binary Order Protocol Tests
# ==============================================================================


class TestSharedMemoryBinaryProtocolIntegration:
    """Test suite verifying seamless integration with the Phase 6 binary protocol."""

    def test_roundtrip_buy_order_float_price(self, tmp_path: Path) -> None:
        """Verify round-trip encoding, writing, reading, and decoding of BUY order."""
        filepath = tmp_path / "shm_buy_order.bin"
        ts = 1710000000999888777
        original = Order(
            order_id="ord-buy-alpha",
            side=OrderSide.BUY,
            price=150.25,
            quantity=500,
            timestamp=ts,
            sequence_number=42,
        )

        with SharedMemoryRegion.create(filepath, size=1024) as shm:
            shm.write_order(original, offset=0)
            decoded = shm.read_order(offset=0)

            assert isinstance(decoded, Order)
            assert decoded.order_id == original.order_id
            assert decoded.side == OrderSide.BUY
            assert decoded.side == original.side
            assert decoded.price == 150.25
            assert isinstance(decoded.price, float)
            assert decoded.quantity == 500
            assert decoded.timestamp == ts
            assert decoded.sequence_number == 42

    def test_roundtrip_sell_order_decimal_price(self, tmp_path: Path) -> None:
        """Verify round-trip preservation of Decimal price type and SELL side."""
        filepath = tmp_path / "shm_sell_order.bin"
        price = Decimal("199.95000000")
        original = Order(
            order_id="ord-sell-beta",
            side=OrderSide.SELL,
            price=price,
            quantity=1000,
            timestamp=1710000001000000000,
            sequence_number=999,
        )

        with SharedMemoryRegion.create(filepath, size=1024) as shm:
            shm.write_order(original, offset=64)
            decoded = shm.read_order(offset=64)

            assert decoded.order_id == "ord-sell-beta"
            assert decoded.side == OrderSide.SELL
            assert decoded.price == price
            assert isinstance(decoded.price, Decimal)
            assert decoded.quantity == 1000
            assert decoded.sequence_number == 999

    def test_roundtrip_int_price_none_sequence_number(self, tmp_path: Path) -> None:
        """Verify round-trip preservation of int price and sequence_number=None."""
        filepath = tmp_path / "shm_int_order.bin"
        original = Order(
            order_id="ord-int-gamma",
            side=OrderSide.BUY,
            price=100,
            quantity=25,
            timestamp=1710000002000000000,
            sequence_number=None,
        )

        with SharedMemoryRegion.create(filepath, size=1024) as shm:
            shm.write_order(original, offset=128)
            decoded = shm.read_order(offset=128)

            assert decoded.order_id == "ord-int-gamma"
            assert decoded.side == OrderSide.BUY
            assert decoded.price == 100
            assert isinstance(decoded.price, int)
            assert decoded.quantity == 25
            assert decoded.sequence_number is None

    def test_multiple_orders_at_distinct_offsets(self, tmp_path: Path) -> None:
        """Verify writing and reading a sequence of distinct orders across slots."""
        filepath = tmp_path / "shm_multi_orders.bin"
        order_count = 10
        total_size = order_count * ORDER_STRUCT_SIZE

        orders = [
            Order(
                order_id=f"order-{i:03d}",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                price=100.0 + i * 0.5,
                quantity=10 * (i + 1),
                timestamp=1700000000000000000 + i * 1000,
                sequence_number=i,
            )
            for i in range(order_count)
        ]

        with SharedMemoryRegion.create(filepath, size=total_size) as shm:
            # Write all orders at sequential 64-byte offsets
            for i, order in enumerate(orders):
                offset = i * ORDER_STRUCT_SIZE
                shm.write_order(order, offset=offset)

            # Read and verify in reverse order to ensure no crosstalk
            for i in reversed(range(order_count)):
                offset = i * ORDER_STRUCT_SIZE
                decoded = shm.read_order(offset=offset)
                assert decoded.order_id == orders[i].order_id
                assert decoded.side == orders[i].side
                assert decoded.price == orders[i].price
                assert decoded.quantity == orders[i].quantity
                assert decoded.sequence_number == orders[i].sequence_number

    def test_slot_based_helpers(self, tmp_path: Path) -> None:
        """Verify write_order_at_slot and read_order_at_slot helpers."""
        filepath = tmp_path / "shm_slots.bin"
        with SharedMemoryRegion.create(filepath, size=1024) as shm:
            order0 = Order("slot-0", "BUY", 10.0, 100)
            order1 = Order("slot-1", "SELL", 20.0, 200)

            shm.write_order_at_slot(order0, slot=0)
            shm.write_order_at_slot(order1, slot=1)

            assert shm.read_order_at_slot(0).order_id == "slot-0"
            assert shm.read_order_at_slot(1).order_id == "slot-1"

            # Direct offset verification
            assert shm.read_order(offset=0).order_id == "slot-0"
            assert shm.read_order(offset=ORDER_STRUCT_SIZE).order_id == "slot-1"

            # Invalid slot values
            with pytest.raises(SharedMemoryBoundsError, match="Slot must be a non-negative integer"):
                shm.write_order_at_slot(order0, slot=-1)

            with pytest.raises(SharedMemoryBoundsError, match="Slot must be a non-negative integer"):
                shm.read_order_at_slot(-1)

            with pytest.raises(SharedMemoryBoundsError, match="Slot must be a non-negative integer"):
                shm.write_order_at_slot(order0, slot="bad")  # type: ignore

    def test_write_order_type_validation(self, tmp_path: Path) -> None:
        """Verify passing non-Order objects to write_order raises TypeError."""
        filepath = tmp_path / "shm_order_type.bin"
        with SharedMemoryRegion.create(filepath, size=512) as shm:
            with pytest.raises(TypeError, match="Expected Order instance"):
                shm.write_order({"order_id": "fake"}, offset=0)  # type: ignore

            with pytest.raises(TypeError, match="Expected Order instance"):
                shm.write_order("just_a_string", offset=0)  # type: ignore

    def test_read_order_malformed_record(self, tmp_path: Path) -> None:
        """Verify reading corrupted data or invalid binary records raises MalformedRecordError."""
        filepath = tmp_path / "shm_malformed.bin"
        with SharedMemoryRegion.create(filepath, size=512) as shm:
            # 1. Null / zero bytes (invalid timestamp 0 and empty order_id)
            shm.write_record(0, b"\x00" * ORDER_STRUCT_SIZE)
            with pytest.raises(MalformedRecordError, match="Malformed order record"):
                shm.read_order(0)

            # 2. Corrupt side code (e.g. side code 99)
            valid_order = Order("ord-corrupt", "BUY", 100.0, 10, timestamp=1700000000)
            valid_bytes = bytearray(encode_order(valid_order))
            # Offset of side code in struct: ts(8) + id(32) + price(8) + qty(4) = 52
            valid_bytes[52] = 99  # Corrupt side code
            shm.write_record(64, bytes(valid_bytes))

            with pytest.raises(MalformedRecordError, match="Unknown side code"):
                shm.read_order(64)


# ==============================================================================
# 6. Producer & Consumer Role Tests
# ==============================================================================


class TestSharedMemoryProducerConsumer:
    """Test suite verifying dedicated SharedMemoryProducer and SharedMemoryConsumer classes."""

    def test_producer_consumer_cooperation(self, tmp_path: Path) -> None:
        """Verify producer and consumer exchanging orders over a shared file."""
        filepath = tmp_path / "shm_prod_cons.bin"
        size = 4096

        producer = SharedMemoryProducer.create(filepath, size=size)
        assert producer.filepath == filepath.resolve()
        assert producer.size == size
        assert producer.capacity_records == size // ORDER_STRUCT_SIZE

        order1 = Order("pc-001", "BUY", 150.0, 100, sequence_number=1)
        order2 = Order("pc-002", "SELL", 151.0, 200, sequence_number=2)

        producer.write_order(order1, offset=0)
        producer.write_order_at_slot(order2, slot=1)
        producer.flush()

        consumer = SharedMemoryConsumer.open(filepath, read_only=True)
        assert consumer.filepath == filepath.resolve()
        assert consumer.size == size

        read_order1 = consumer.read_order(offset=0)
        read_order2 = consumer.read_order_at_slot(1)

        assert read_order1.order_id == "pc-001"
        assert read_order1.price == 150.0
        assert read_order2.order_id == "pc-002"
        assert read_order2.price == 151.0

        # Consumer is read-only and cannot write
        with pytest.raises(SharedMemoryReadOnlyError):
            consumer.region.write_order(order1, offset=128)

        consumer.close()
        producer.unlink()
        assert not filepath.exists()

    def test_producer_consumer_context_managers(self, tmp_path: Path) -> None:
        """Verify context manager usage with SharedMemoryProducer and SharedMemoryConsumer."""
        filepath = tmp_path / "shm_pc_ctx.bin"
        order = Order("ctx-pc", "BUY", 99.99, 10)

        with SharedMemoryProducer.create(filepath, size=1024) as producer:
            producer.write_order(order, offset=0)

            with SharedMemoryConsumer.open(filepath, read_only=True) as consumer:
                read_order = consumer.read_order(offset=0)
                assert read_order.order_id == order.order_id

        assert producer.is_closed
        assert consumer.is_closed

    def test_producer_on_readonly_region_raises(self, tmp_path: Path) -> None:
        """Verify creating a producer on a read-only region raises SharedMemoryReadOnlyError."""
        filepath = tmp_path / "shm_ro_prod.bin"
        with SharedMemoryRegion.create(filepath, size=512) as creator:
            creator.write_bytes(0, b"init")

        ro_region = SharedMemoryRegion.open(filepath, read_only=True)
        try:
            with pytest.raises(SharedMemoryReadOnlyError, match="read-only"):
                SharedMemoryProducer(ro_region)
        finally:
            ro_region.close()

    def test_producer_consumer_invalid_type_raises(self) -> None:
        """Verify passing non-SharedMemoryRegion objects raises TypeError."""
        with pytest.raises(TypeError, match="Expected SharedMemoryRegion"):
            SharedMemoryProducer("not_a_region")  # type: ignore

        with pytest.raises(TypeError, match="Expected SharedMemoryRegion"):
            SharedMemoryConsumer(12345)  # type: ignore


# ==============================================================================
# 7. Multi-Process Shared-Memory IPC Test
# ==============================================================================


class TestMultiProcessIPC:
    """Test suite verifying inter-process communication across distinct OS processes."""

    def test_cross_process_shared_memory_exchange(self, tmp_path: Path) -> None:
        """Verify a producer child process writing and a consumer child process reading via mmap."""
        filepath = tmp_path / "shm_multiprocess.bin"
        filepath_str = str(filepath)
        queue: mp.Queue = mp.Queue()

        # Step 1: Run producer in a child process
        producer_proc = mp.Process(
            target=_producer_worker,
            args=(filepath_str, "mp-order-001", 123.45, 500),
        )
        producer_proc.start()
        producer_proc.join(timeout=5)
        assert producer_proc.exitcode == 0

        # Step 2: Run consumer in a second child process
        consumer_proc = mp.Process(
            target=_consumer_worker,
            args=(filepath_str, queue),
        )
        consumer_proc.start()
        consumer_proc.join(timeout=5)
        assert consumer_proc.exitcode == 0

        # Step 3: Validate received order
        result = queue.get(timeout=2)
        order_id, side_val, price, quantity, seq = result

        assert order_id == "mp-order-001"
        assert side_val == "BUY"
        assert price == 123.45
        assert quantity == 500
        assert seq == 101

        # Clean up
        if filepath.exists():
            filepath.unlink()
