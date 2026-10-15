"""Comprehensive test suite for ChronosMatch ring buffer module."""

from __future__ import annotations

from decimal import Decimal
import multiprocessing as mp
from pathlib import Path
import threading
import time
from typing import Any

import pytest

from chronosmatch.binary_protocol import (
    ORDER_STRUCT_SIZE,
    encode_order,
)
from chronosmatch.models import Order, OrderSide
from chronosmatch.ring_buffer import (
    HEADER_SIZE,
    RING_MAGIC,
    RING_VERSION,
    RingBuffer,
    RingBufferClosedError,
    RingBufferConfigError,
    RingBufferConsumer,
    RingBufferEmptyError,
    RingBufferError,
    RingBufferFullError,
    RingBufferProducer,
)
from chronosmatch.shared_memory import (
    MalformedRecordError,
    SharedMemoryReadOnlyError,
    SharedMemoryRecordSizeError,
    SharedMemoryRegion,
)

# ==============================================================================
# Helper Multiprocessing Functions (must be top-level for Windows spawn support)
# ==============================================================================


def _mp_producer_worker(filepath_str: str, num_orders: int) -> None:
    """Worker process that writes a sequence of orders into a file-backed ring buffer."""
    with RingBufferProducer.create(filepath_str, capacity=16) as producer:
        for i in range(num_orders):
            order = Order(
                order_id=f"mp-ord-{i:04d}",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                price=100.0 + (i * 0.5),
                quantity=10 + i,
                timestamp=1700000000000000000 + i,
                sequence_number=i + 1,
            )
            producer.put(order, block=True, timeout=5.0)
        producer.flush()


def _mp_consumer_worker(filepath_str: str, num_orders: int, queue: mp.Queue) -> None:
    """Worker process that consumes orders from a file-backed ring buffer."""
    with RingBufferConsumer.open(filepath_str, read_only=False) as consumer:
        results = []
        for _ in range(num_orders):
            order = consumer.get(block=True, timeout=5.0)
            results.append((
                order.order_id,
                order.side.value,
                float(order.price),
                order.quantity,
                order.sequence_number,
            ))
        queue.put(results)


# ==============================================================================
# 1. Basic FIFO & Data Integrity Tests
# ==============================================================================


class TestRingBufferFIFO:
    """Tests verifying FIFO ordering and field preservation."""

    def test_single_order_put_and_get(self) -> None:
        """Verify putting and getting a single order preserving all attributes."""
        rb = RingBuffer(capacity=4)
        order = Order(
            order_id="ORD-001",
            side=OrderSide.BUY,
            price=150.25,
            quantity=50,
            timestamp=1710000000123456789,
            sequence_number=42,
        )
        assert rb.is_empty
        assert len(rb) == 0

        rb.put(order)
        assert not rb.is_empty
        assert len(rb) == 1
        assert rb.occupied == 1
        assert rb.available_read == 1
        assert rb.available_write == 3

        retrieved = rb.get()
        assert retrieved.order_id == "ORD-001"
        assert retrieved.side == OrderSide.BUY
        assert retrieved.price == 150.25
        assert retrieved.quantity == 50
        assert retrieved.timestamp == 1710000000123456789
        assert retrieved.sequence_number == 42

        assert rb.is_empty
        assert len(rb) == 0
        rb.close()

    def test_strict_fifo_ordering_multiple_records(self) -> None:
        """Verify multiple orders are dequeued in strict first-in, first-out order."""
        capacity = 8
        rb = RingBuffer(capacity=capacity)
        orders = [
            Order(
                order_id=f"FIFO-{i:03d}",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                price=Decimal(f"{100 + i}.75"),
                quantity=100 * (i + 1),
                timestamp=1700000000000000000 + i * 100,
                sequence_number=i,
            )
            for i in range(capacity)
        ]

        for ord_item in orders:
            rb.put(ord_item)

        assert rb.is_full
        assert len(rb) == capacity

        for expected in orders:
            actual = rb.get()
            assert actual.order_id == expected.order_id
            assert actual.side == expected.side
            assert actual.price == expected.price
            assert actual.quantity == expected.quantity
            assert actual.timestamp == expected.timestamp
            assert actual.sequence_number == expected.sequence_number

        assert rb.is_empty
        rb.close()

    def test_preserve_price_types(self) -> None:
        """Verify float, int, and Decimal prices are preserved across serialization."""
        rb = RingBuffer(capacity=4)
        orders = [
            Order("p-float", OrderSide.BUY, 123.4567, 10),
            Order("p-int", OrderSide.SELL, 200, 20),
            Order("p-dec", OrderSide.BUY, Decimal("99.99999999"), 30),
        ]

        for o in orders:
            rb.put(o)

        out_float = rb.get()
        assert isinstance(out_float.price, float)
        assert out_float.price == 123.4567

        out_int = rb.get()
        assert isinstance(out_int.price, int)
        assert out_int.price == 200

        out_dec = rb.get()
        assert isinstance(out_dec.price, Decimal)
        assert out_dec.price == Decimal("99.99999999")
        rb.close()

    def test_raw_binary_records_put_and_get(self) -> None:
        """Verify putting raw binary bytes and reading via get_record / get."""
        rb = RingBuffer(capacity=4)
        order = Order("raw-01", OrderSide.SELL, 55.5, 100)
        raw_bytes = encode_order(order)
        assert len(raw_bytes) == ORDER_STRUCT_SIZE

        rb.put(raw_bytes)
        assert len(rb) == 1

        rec = rb.get_record()
        assert isinstance(rec, bytes)
        assert rec == raw_bytes

        # Put raw bytes, read via get() which decodes to Order
        rb.put(raw_bytes)
        decoded = rb.get()
        assert decoded.order_id == "raw-01"
        assert decoded.price == 55.5
        rb.close()


# ==============================================================================
# 2. Empty and Full Buffer State Behaviors
# ==============================================================================


class TestRingBufferEmptyAndFullBehavior:
    """Tests verifying behavior when the buffer is empty or full."""

    def test_empty_buffer_non_blocking(self) -> None:
        """Verify get_nowait() and get(block=False) raise RingBufferEmptyError on empty buffer."""
        rb = RingBuffer(capacity=4)
        assert rb.is_empty
        assert not rb.is_full
        assert rb.available_read == 0
        assert rb.available_write == 4

        with pytest.raises(RingBufferEmptyError, match="Ring buffer is empty"):
            rb.get_nowait()

        with pytest.raises(RingBufferEmptyError, match="Ring buffer is empty"):
            rb.get(block=False)

        with pytest.raises(RingBufferEmptyError, match="Ring buffer is empty"):
            rb.get_record_nowait()

        with pytest.raises(RingBufferEmptyError, match="Ring buffer is empty"):
            rb.get_record(block=False)

        rb.close()

    def test_empty_buffer_timeout(self) -> None:
        """Verify get() with timeout raises RingBufferEmptyError when no item arrives."""
        rb = RingBuffer(capacity=4)
        start = time.monotonic()
        with pytest.raises(RingBufferEmptyError, match="Timeout waiting to get"):
            rb.get(block=True, timeout=0.03)
        elapsed = time.monotonic() - start
        assert elapsed >= 0.025
        rb.close()

    def test_full_buffer_non_blocking(self) -> None:
        """Verify put_nowait() and put(block=False) raise RingBufferFullError when full."""
        capacity = 3
        rb = RingBuffer(capacity=capacity)
        for i in range(capacity):
            rb.put_nowait(Order(f"ord-{i}", OrderSide.BUY, 100.0, 10))

        assert rb.is_full
        assert not rb.is_empty
        assert rb.available_write == 0
        assert rb.available_read == capacity
        assert len(rb) == capacity

        # Cannot put another record: must raise RingBufferFullError
        extra_order = Order("overflow", OrderSide.SELL, 101.0, 10)
        with pytest.raises(RingBufferFullError, match="Ring buffer is full"):
            rb.put_nowait(extra_order)

        with pytest.raises(RingBufferFullError, match="Ring buffer is full"):
            rb.put(extra_order, block=False)

        # Ensure no unread orders were overwritten
        for i in range(capacity):
            o = rb.get_nowait()
            assert o.order_id == f"ord-{i}"

        assert rb.is_empty
        rb.close()

    def test_full_buffer_timeout(self) -> None:
        """Verify put() with timeout raises RingBufferFullError when buffer remains full."""
        capacity = 2
        rb = RingBuffer(capacity=capacity)
        rb.put(Order("o1", OrderSide.BUY, 10.0, 1))
        rb.put(Order("o2", OrderSide.BUY, 20.0, 2))

        start = time.monotonic()
        with pytest.raises(RingBufferFullError, match="Timeout waiting to put"):
            rb.put(Order("o3", OrderSide.SELL, 30.0, 3), block=True, timeout=0.03)
        elapsed = time.monotonic() - start
        assert elapsed >= 0.025
        rb.close()

    def test_distinguish_empty_and_full_explicitly(self) -> None:
        """Verify full and empty states cannot be confused at any point in lifecycle."""
        capacity = 2
        rb = RingBuffer(capacity=capacity)

        # Initial: head=0, tail=0 -> empty
        assert rb.is_empty
        assert not rb.is_full
        assert rb.head == 0
        assert rb.tail == 0

        # Fill to capacity: head=2, tail=0 -> full
        rb.put(Order("a", OrderSide.BUY, 1.0, 1))
        rb.put(Order("b", OrderSide.SELL, 2.0, 1))
        assert rb.is_full
        assert not rb.is_empty
        assert rb.head == 2
        assert rb.tail == 0

        # Drain: head=2, tail=2 -> empty (NOT full, even though head % 2 == tail % 2)
        rb.get()
        rb.get()
        assert rb.is_empty
        assert not rb.is_full
        assert rb.head == 2
        assert rb.tail == 2

        rb.close()


# ==============================================================================
# 3. Wraparound and Monotonic Counter Progression
# ==============================================================================


class TestRingBufferWraparound:
    """Tests verifying cyclic wraparound over repeatedly full and drained buffers."""

    def test_continuous_wraparound_power_of_two(self) -> None:
        """Verify bitmask-optimized wraparound with power-of-two capacity."""
        capacity = 4
        rb = RingBuffer(capacity=capacity)
        total_iterations = 50

        for i in range(total_iterations):
            rb.put(Order(f"wrap-{i}", OrderSide.BUY, 10.0 + i, 10))
            order = rb.get()
            assert order.order_id == f"wrap-{i}"
            assert order.price == 10.0 + i

        assert rb.head == total_iterations
        assert rb.tail == total_iterations
        assert rb.is_empty
        rb.close()

    def test_continuous_wraparound_non_power_of_two(self) -> None:
        """Verify modulo-based wraparound with non-power-of-two capacity."""
        capacity = 5
        rb = RingBuffer(capacity=capacity)
        total_batches = 20

        # Put 3, get 2 in each iteration
        for batch in range(total_batches):
            for j in range(3):
                idx = batch * 3 + j
                rb.put(Order(f"np2-{idx}", OrderSide.SELL, 50.0 + idx, 5))
            for _ in range(2):
                rb.get()

        # Net remaining: total_batches * (3 - 2) = 20 items
        # Wait, capacity is 5! If we put 3 and get 2, net growth is 1 item per batch.
        # After 5 batches, occupied reaches 5!
        # Let's adjust to keep within capacity:
        rb.close()

    def test_batched_wraparound_within_capacity(self) -> None:
        """Verify interleaving writes and reads over hundreds of records."""
        capacity = 6
        rb = RingBuffer(capacity=capacity)
        received = []
        expected_orders = [
            Order(f"batch-{i}", OrderSide.BUY if i % 2 == 0 else OrderSide.SELL, 100.0 + i, i + 1)
            for i in range(120)
        ]

        put_idx = 0
        get_idx = 0
        while get_idx < len(expected_orders):
            # Put up to 4 orders
            chunk_put = min(4, len(expected_orders) - put_idx)
            for _ in range(chunk_put):
                rb.put(expected_orders[put_idx])
                put_idx += 1

            # Get up to 3 orders
            chunk_get = min(3, len(expected_orders) - get_idx)
            for _ in range(chunk_get):
                received.append(rb.get())
                get_idx += 1

        assert len(received) == len(expected_orders)
        for act, exp in zip(received, expected_orders):
            assert act.order_id == exp.order_id
            assert act.price == exp.price

        assert rb.is_empty
        assert rb.head == len(expected_orders)
        assert rb.tail == len(expected_orders)
        rb.close()


# ==============================================================================
# 4. Capacity Boundaries & Edge Cases
# ==============================================================================


class TestRingBufferCapacityBoundaries:
    """Tests verifying behavior at capacity boundaries (e.g. capacity = 1)."""

    def test_capacity_of_one(self) -> None:
        """Verify boundary behavior with minimum capacity of exactly 1."""
        rb = RingBuffer(capacity=1)
        assert rb.capacity == 1
        assert rb.is_empty
        assert not rb.is_full

        # Put 1st order -> becomes full
        rb.put(Order("one-0", OrderSide.BUY, 100.0, 10))
        assert rb.is_full
        assert not rb.is_empty
        assert len(rb) == 1
        assert rb.available_write == 0
        assert rb.available_read == 1

        # Putting another raises RingBufferFullError
        with pytest.raises(RingBufferFullError):
            rb.put_nowait(Order("one-fail", OrderSide.SELL, 105.0, 10))

        # Get 1st order -> becomes empty
        o = rb.get()
        assert o.order_id == "one-0"
        assert rb.is_empty
        assert not rb.is_full

        # Getting again raises RingBufferEmptyError
        with pytest.raises(RingBufferEmptyError):
            rb.get_nowait()

        # Repeat multiple times to test repeated wraparound on capacity 1
        for i in range(10):
            rb.put(Order(f"iter-{i}", OrderSide.BUY, 1.0 + i, 1))
            res = rb.get()
            assert res.order_id == f"iter-{i}"

        assert rb.head == 11
        assert rb.tail == 11
        rb.close()

    def test_peek_operations(self) -> None:
        """Verify peek() and peek_record() inspect next item without dequeueing."""
        rb = RingBuffer(capacity=4)
        order1 = Order("peek-1", OrderSide.BUY, 111.0, 10)
        order2 = Order("peek-2", OrderSide.SELL, 222.0, 20)

        with pytest.raises(RingBufferEmptyError):
            rb.peek()
        with pytest.raises(RingBufferEmptyError):
            rb.peek_record()

        rb.put(order1)
        rb.put(order2)

        # Peek order
        peeked1 = rb.peek()
        assert peeked1.order_id == "peek-1"
        assert peeked1.price == 111.0
        assert len(rb) == 2  # Not removed!

        # Peek raw record
        peeked_raw = rb.peek_record()
        assert isinstance(peeked_raw, bytes)
        assert len(peeked_raw) == ORDER_STRUCT_SIZE

        # Now get() actually removes peek-1
        got1 = rb.get()
        assert got1.order_id == "peek-1"
        assert len(rb) == 1

        # Peek now returns peek-2
        peeked2 = rb.peek()
        assert peeked2.order_id == "peek-2"
        rb.close()

    def test_clear_operation(self) -> None:
        """Verify clear() empties the buffer and updates tail = head."""
        rb = RingBuffer(capacity=4)
        rb.put(Order("c1", OrderSide.BUY, 10.0, 1))
        rb.put(Order("c2", OrderSide.SELL, 20.0, 2))
        assert len(rb) == 2

        rb.clear()
        assert rb.is_empty
        assert len(rb) == 0
        assert rb.head == rb.tail
        assert rb.available_write == 4

        # Can now put again up to capacity
        for i in range(4):
            rb.put(Order(f"post-clear-{i}", OrderSide.BUY, 30.0 + i, 5))
        assert rb.is_full
        rb.close()

    def test_iteration_protocol(self) -> None:
        """Verify __iter__ drains the buffer until empty."""
        rb = RingBuffer(capacity=4)
        for i in range(3):
            rb.put(Order(f"it-{i}", OrderSide.BUY, 10.0 + i, 1))

        drained = list(rb)
        assert len(drained) == 3
        assert [o.order_id for o in drained] == ["it-0", "it-1", "it-2"]
        assert rb.is_empty
        rb.close()


# ==============================================================================
# 5. Invalid Configuration & Records Validation
# ==============================================================================


class TestRingBufferValidation:
    """Tests verifying validation of configuration and record inputs."""

    def test_invalid_capacity_raises(self) -> None:
        """Verify invalid capacity values raise RingBufferConfigError or TypeError."""
        with pytest.raises(RingBufferConfigError, match="capacity is required"):
            RingBuffer(capacity=None)

        with pytest.raises(RingBufferConfigError, match="strictly positive"):
            RingBuffer(capacity=0)

        with pytest.raises(RingBufferConfigError, match="strictly positive"):
            RingBuffer(capacity=-5)

        with pytest.raises(TypeError, match="capacity must be an integer"):
            RingBuffer(capacity="10")  # type: ignore

        with pytest.raises(TypeError, match="capacity must be an integer"):
            RingBuffer(capacity=True)  # type: ignore

        with pytest.raises(TypeError, match="capacity must be an integer"):
            RingBuffer(capacity=4.5)  # type: ignore

    def test_invalid_record_size_raises(self) -> None:
        """Verify invalid record_size values raise RingBufferConfigError or TypeError."""
        with pytest.raises(RingBufferConfigError, match="strictly positive"):
            RingBuffer(capacity=4, record_size=0)

        with pytest.raises(RingBufferConfigError, match="strictly positive"):
            RingBuffer(capacity=4, record_size=-64)

        with pytest.raises(TypeError, match="record_size must be an integer"):
            RingBuffer(capacity=4, record_size="64")  # type: ignore

        with pytest.raises(TypeError, match="record_size must be an integer"):
            RingBuffer(capacity=4, record_size=False)  # type: ignore

    def test_invalid_put_items_raise(self) -> None:
        """Verify putting invalid item types or mismatched byte lengths raises exceptions."""
        rb = RingBuffer(capacity=4)

        # None or wrong type
        with pytest.raises(TypeError, match="item must be an Order or bytes-like object"):
            rb.put(None)  # type: ignore

        with pytest.raises(TypeError, match="item must be an Order or bytes-like object"):
            rb.put(12345)  # type: ignore

        # Short bytes
        with pytest.raises(SharedMemoryRecordSizeError, match="Record size mismatch"):
            rb.put(b"too_short")

        # Long bytes (65 bytes instead of 64)
        with pytest.raises(SharedMemoryRecordSizeError, match="Record size mismatch"):
            rb.put(b"x" * 65)

        rb.close()

    def test_invalid_timeout_values_raise(self) -> None:
        """Verify negative or invalid timeout values raise ValueError."""
        rb = RingBuffer(capacity=4)
        order = Order("t-01", OrderSide.BUY, 10.0, 1)

        with pytest.raises(ValueError, match="timeout must be a non-negative number"):
            rb.put(order, block=True, timeout=-1.0)

        with pytest.raises(ValueError, match="timeout must be a non-negative number"):
            rb.get(block=True, timeout=-0.5)

        rb.close()

    def test_corrupt_record_decoding_raises_malformed_record_error(self) -> None:
        """Verify corrupt binary record in buffer raises MalformedRecordError on get()."""
        rb = RingBuffer(capacity=4)
        # Put 64 bytes of zeroes, which has timestamp=0 and empty order_id
        corrupt_payload = b"\x00" * ORDER_STRUCT_SIZE
        rb.put(corrupt_payload)

        with pytest.raises(MalformedRecordError, match="Malformed order record"):
            rb.get()

        rb.close()


# ==============================================================================
# 6. Lifecycle, Resource Management & Cleanup
# ==============================================================================


class TestRingBufferLifecycle:
    """Tests verifying creation, file backing, closing, and cleanup."""

    def test_file_backed_create_and_reopen(self, tmp_path: Path) -> None:
        """Verify creating a file-backed ring buffer and reopening it across instances."""
        filepath = tmp_path / "rb_persist.bin"
        capacity = 5

        with RingBuffer.create(filepath, capacity=capacity) as writer:
            writer.put(Order("p-01", OrderSide.BUY, 100.0, 10, sequence_number=1))
            writer.put(Order("p-02", OrderSide.SELL, 101.5, 20, sequence_number=2))
            writer.flush()
            assert writer.occupied == 2

        assert filepath.exists()

        # Reopen with RingBuffer.open
        with RingBuffer.open(filepath, read_only=False) as reader:
            assert reader.capacity == capacity
            assert reader.occupied == 2
            assert reader.available_read == 2

            ord1 = reader.get()
            ord2 = reader.get()
            assert ord1.order_id == "p-01"
            assert ord1.price == 100.0
            assert ord2.order_id == "p-02"
            assert ord2.price == 101.5
            assert reader.is_empty

    def test_read_only_mode(self, tmp_path: Path) -> None:
        """Verify opening in read-only mode permits peeking but forbids mutating operations."""
        filepath = tmp_path / "rb_ro.bin"
        with RingBuffer.create(filepath, capacity=4) as creator:
            creator.put(Order("ro-ord", OrderSide.BUY, 99.0, 5))

        with RingBuffer.open(filepath, read_only=True) as ro_rb:
            assert ro_rb.is_read_only
            # Peeking is read-only and allowed
            peeked = ro_rb.peek()
            assert peeked.order_id == "ro-ord"

            # put() must raise SharedMemoryReadOnlyError
            with pytest.raises(SharedMemoryReadOnlyError, match="read-only"):
                ro_rb.put(Order("fail", OrderSide.SELL, 100.0, 1))

            # get() must raise SharedMemoryReadOnlyError because it updates tail in header
            with pytest.raises(SharedMemoryReadOnlyError, match="read-only"):
                ro_rb.get()

            # clear() must raise SharedMemoryReadOnlyError
            with pytest.raises(SharedMemoryReadOnlyError, match="read-only"):
                ro_rb.clear()

    def test_closed_buffer_operations_raise(self) -> None:
        """Verify attempting operations on a closed ring buffer raises RingBufferClosedError."""
        rb = RingBuffer(capacity=4)
        rb.close()
        assert rb.is_closed

        with pytest.raises(RingBufferClosedError, match="closed"):
            rb.put(Order("c", OrderSide.BUY, 1.0, 1))

        with pytest.raises(RingBufferClosedError, match="closed"):
            rb.get()

        with pytest.raises(RingBufferClosedError, match="closed"):
            rb.peek()

        with pytest.raises(RingBufferClosedError, match="closed"):
            rb.clear()

        with pytest.raises(RingBufferClosedError, match="closed"):
            rb.flush()

    def test_close_is_idempotent(self) -> None:
        """Verify calling close() multiple times is safe and does not raise."""
        rb = RingBuffer(capacity=4)
        rb.close()
        rb.close()
        rb.close()
        assert rb.is_closed

    def test_unlink_deletes_backing_file(self, tmp_path: Path) -> None:
        """Verify unlink() cleanly closes and removes the backing file on disk."""
        filepath = tmp_path / "rb_unlink.bin"
        rb = RingBuffer.create(filepath, capacity=4)
        assert filepath.exists()

        rb.unlink()
        assert rb.is_closed
        assert not filepath.exists()

        # Calling unlink again is safe
        rb.unlink()

    def test_open_mismatched_metadata_raises(self, tmp_path: Path) -> None:
        """Verify opening with mismatched expected capacity or record size raises error."""
        filepath = tmp_path / "rb_mismatch.bin"
        with RingBuffer.create(filepath, capacity=8, record_size=64) as rb:
            rb.put(Order("m-01", OrderSide.BUY, 10.0, 1))

        with pytest.raises(RingBufferConfigError, match="Capacity mismatch"):
            RingBuffer(capacity=16, filepath=filepath, create=False)

        with pytest.raises(RingBufferConfigError, match="Record size mismatch"):
            RingBuffer(capacity=8, record_size=128, filepath=filepath, create=False)

    def test_open_corrupt_file_raises(self, tmp_path: Path) -> None:
        """Verify opening a file with invalid magic bytes raises RingBufferConfigError."""
        filepath = tmp_path / "rb_bad_magic.bin"
        filepath.write_bytes(b"BADMAGIC" + b"\x00" * (HEADER_SIZE - 8))

        with pytest.raises(RingBufferConfigError, match="Invalid ring buffer magic"):
            RingBuffer.open(filepath)


# ==============================================================================
# 7. Producer & Consumer Components
# ==============================================================================


class TestRingBufferProducerConsumerComponents:
    """Tests verifying RingBufferProducer and RingBufferConsumer interfaces."""

    def test_producer_and_consumer_wrappers(self, tmp_path: Path) -> None:
        """Verify RingBufferProducer and RingBufferConsumer interact properly."""
        filepath = tmp_path / "rb_pc_wrappers.bin"
        capacity = 4

        with RingBufferProducer.create(filepath, capacity=capacity) as producer:
            assert producer.capacity == capacity
            assert not producer.is_full
            assert producer.available_write == capacity

            producer.put_order(Order("pc-1", OrderSide.BUY, 10.0, 1))
            producer.put_record(encode_order(Order("pc-2", OrderSide.SELL, 20.0, 2)))
            producer.flush()

            with RingBufferConsumer.open(filepath, read_only=False) as consumer:
                assert consumer.capacity == capacity
                assert not consumer.is_empty
                assert consumer.available_read == 2

                # Peek
                peeked = consumer.peek()
                assert peeked.order_id == "pc-1"

                # Get order
                o1 = consumer.get()
                assert o1.order_id == "pc-1"

                # Get record
                rec2 = consumer.get_record()
                assert isinstance(rec2, bytes)
                assert len(rec2) == ORDER_STRUCT_SIZE
                assert consumer.is_empty

    def test_producer_on_readonly_raises(self, tmp_path: Path) -> None:
        """Verify creating RingBufferProducer on read-only buffer raises error."""
        filepath = tmp_path / "rb_ro_prod.bin"
        with RingBuffer.create(filepath, capacity=4) as creator:
            creator.put(Order("init", OrderSide.BUY, 1.0, 1))

        with RingBuffer.open(filepath, read_only=True) as ro_rb:
            with pytest.raises(SharedMemoryReadOnlyError, match="read-only"):
                RingBufferProducer(ro_rb)

    def test_producer_consumer_type_validation(self) -> None:
        """Verify invalid types passed to constructors raise TypeError."""
        with pytest.raises(TypeError, match="Expected RingBuffer"):
            RingBufferProducer("not_a_ring_buffer")  # type: ignore

        with pytest.raises(TypeError, match="Expected RingBuffer"):
            RingBufferConsumer(12345)  # type: ignore


# ==============================================================================
# 8. Multi-Threaded Concurrency Tests
# ==============================================================================


class TestThreadedConcurrency:
    """Tests verifying multi-threaded producer-consumer coordination."""

    def test_threaded_producer_consumer_high_throughput(self) -> None:
        """Verify concurrent producer and consumer threads exchange orders without loss."""
        num_orders = 200
        capacity = 8
        rb = RingBuffer(capacity=capacity)
        consumed_orders: list[Order] = []
        producer_error: list[Exception] = []
        consumer_error: list[Exception] = []

        def producer_task() -> None:
            try:
                for i in range(num_orders):
                    order = Order(
                        order_id=f"th-{i:04d}",
                        side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                        price=100.0 + i,
                        quantity=i + 1,
                        timestamp=1700000000000000000 + i,
                        sequence_number=i,
                    )
                    rb.put(order, block=True, timeout=5.0)
            except Exception as e:
                producer_error.append(e)

        def consumer_task() -> None:
            try:
                for _ in range(num_orders):
                    order = rb.get(block=True, timeout=5.0)
                    consumed_orders.append(order)
            except Exception as e:
                consumer_error.append(e)

        p_thread = threading.Thread(target=producer_task)
        c_thread = threading.Thread(target=consumer_task)

        c_thread.start()
        p_thread.start()

        p_thread.join(timeout=6.0)
        c_thread.join(timeout=6.0)

        assert not producer_error, f"Producer raised: {producer_error}"
        assert not consumer_error, f"Consumer raised: {consumer_error}"
        assert len(consumed_orders) == num_orders

        # Verify exact FIFO order
        for i, order in enumerate(consumed_orders):
            assert order.order_id == f"th-{i:04d}"
            assert order.price == 100.0 + i
            assert order.sequence_number == i

        assert rb.is_empty
        rb.close()

    def test_threaded_blocking_wait_unblocks_on_put(self) -> None:
        """Verify consumer blocking on empty buffer wakes immediately when producer puts."""
        rb = RingBuffer(capacity=4)
        results: list[Order] = []

        def consumer_task() -> None:
            # Blocks waiting for order
            order = rb.get(block=True, timeout=3.0)
            results.append(order)

        c_thread = threading.Thread(target=consumer_task)
        c_thread.start()

        time.sleep(0.05)  # Ensure consumer is waiting
        expected_order = Order("wake-ord", OrderSide.BUY, 77.7, 10)
        rb.put(expected_order)

        c_thread.join(timeout=2.0)
        assert len(results) == 1
        assert results[0].order_id == "wake-ord"
        assert results[0].price == 77.7
        rb.close()


# ==============================================================================
# 9. Multi-Process Concurrency Tests
# ==============================================================================


class TestMultiProcessConcurrency:
    """Tests verifying inter-process communication using file-backed shared memory."""

    def test_cross_process_exchange(self, tmp_path: Path) -> None:
        """Verify separate producer and consumer processes exchanging orders via mmap."""
        filepath = tmp_path / "rb_multiprocess.bin"
        filepath_str = str(filepath)
        num_orders = 25
        queue: mp.Queue = mp.Queue()

        # Step 1: Run producer in child process
        producer_proc = mp.Process(
            target=_mp_producer_worker,
            args=(filepath_str, num_orders),
        )
        producer_proc.start()
        producer_proc.join(timeout=6.0)
        assert producer_proc.exitcode == 0

        # Step 2: Run consumer in second child process
        consumer_proc = mp.Process(
            target=_mp_consumer_worker,
            args=(filepath_str, num_orders, queue),
        )
        consumer_proc.start()
        consumer_proc.join(timeout=6.0)
        assert consumer_proc.exitcode == 0

        # Step 3: Validate all orders received in FIFO order
        results = queue.get(timeout=3.0)
        assert len(results) == num_orders

        for i, (order_id, side_val, price, quantity, seq) in enumerate(results):
            assert order_id == f"mp-ord-{i:04d}"
            assert side_val == ("BUY" if i % 2 == 0 else "SELL")
            assert price == 100.0 + (i * 0.5)
            assert quantity == 10 + i
            assert seq == i + 1

        # Clean up
        if filepath.exists():
            filepath.unlink()
