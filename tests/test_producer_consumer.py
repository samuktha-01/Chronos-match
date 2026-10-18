"""Comprehensive integration test suite for cross-process producer-consumer order streaming.

Validates:
- Transfer of orders across independent operating system processes using shared memory.
- Strict FIFO ordering and exact order-data field preservation.
- Concurrent draining when order count far exceeds ring-buffer capacity.
- Non-blocking and blocking wait semantics on empty and full buffers.
- Process failure detection, error propagation, bounded timeouts, and clean resource unlinking on Windows.
"""

from __future__ import annotations

from decimal import Decimal
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any

import pytest

from chronosmatch.models import Order, OrderSide
from chronosmatch.producer_consumer import (
    ConsumerProcess,
    ConsumerProcessError,
    OrderConsumer,
    OrderPipeline,
    OrderProducer,
    ProcessIntegrationError,
    ProcessTimeoutError,
    ProducerProcess,
    ProducerProcessError,
)
from chronosmatch.ring_buffer import (
    RingBufferEmptyError,
    RingBufferFullError,
)

# ==============================================================================
# Helper Multiprocessing Workers (Top-level for Windows `spawn` support)
# ==============================================================================


def _crashing_producer_worker(
    filepath_str: str,
    ready_event: Any = None,
    stop_event: Any = None,
    error_queue: Any = None,
) -> None:
    """Worker that simulates an unexpected failure inside the producer process."""
    try:
        producer = OrderProducer(filepath_str, create=False)
        if ready_event is not None:
            ready_event.set()
        # Simulate fatal error during publishing
        raise RuntimeError("Simulated catastrophic producer hardware fault")
    except Exception as e:
        if error_queue is not None:
            import traceback

            error_queue.put((type(e).__name__, str(e), traceback.format_exc()))
        raise


def _crashing_consumer_worker(
    filepath_str: str,
    ready_event: Any = None,
    error_queue: Any = None,
) -> None:
    """Worker that simulates an unexpected failure inside the consumer process."""
    try:
        if ready_event is not None:
            ready_event.wait(timeout=5.0)
        consumer = OrderConsumer(filepath_str)
        # Simulate fatal parsing error
        raise ValueError("Simulated invalid binary memory record encountered")
    except Exception as e:
        if error_queue is not None:
            import traceback

            error_queue.put((type(e).__name__, str(e), traceback.format_exc()))
        raise


def _infinite_producer_worker(
    filepath_str: str,
    ready_event: Any = None,
    stop_event: Any = None,
    error_queue: Any = None,
) -> None:
    """Worker that produces orders indefinitely until stop_event is signaled."""
    producer = None
    try:
        producer = OrderProducer(filepath_str, create=False)
        if ready_event is not None:
            ready_event.set()
        seq = 0
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            order = Order(
                order_id=f"inf-{seq}",
                side=OrderSide.BUY,
                price=100.0,
                quantity=10,
                timestamp=1700000000000000000 + seq,
                sequence_number=seq,
            )
            try:
                producer.publish(order, block=True, timeout=0.05)
                seq += 1
            except RingBufferFullError:
                continue
    except Exception as e:
        if error_queue is not None:
            import traceback

            error_queue.put((type(e).__name__, str(e), traceback.format_exc()))
        raise
    finally:
        if producer is not None:
            producer.close()


# ==============================================================================
# 1. Direct Component Tests (OrderProducer & OrderConsumer)
# ==============================================================================


class TestOrderProducerConsumerDirect:
    """Verify OrderProducer and OrderConsumer interface contracts and semantics."""

    def test_order_producer_and_consumer_lifecycle(self, tmp_path: Path) -> None:
        """Verify creating, writing, reading, and closing with OrderProducer/Consumer."""
        filepath = tmp_path / "orders_direct.bin"
        capacity = 8

        with OrderProducer.create(filepath, capacity=capacity) as producer:
            assert producer.capacity == capacity
            assert producer.occupied == 0
            assert producer.available_write == capacity
            assert not producer.is_full
            assert not producer.is_closed
            assert producer.filepath == filepath.resolve()

            order = Order(
                order_id="ORD-001",
                side=OrderSide.BUY,
                price=150.50,
                quantity=100,
                timestamp=1700000000123456789,
                sequence_number=1,
            )
            producer.publish(order)
            assert producer.occupied == 1
            assert producer.available_write == capacity - 1

            with OrderConsumer.open(filepath) as consumer:
                assert consumer.capacity == capacity
                assert consumer.occupied == 1
                assert consumer.available_read == 1
                assert not consumer.is_empty
                assert not consumer.is_closed

                received = consumer.receive()
                assert received.order_id == "ORD-001"
                assert received.side == OrderSide.BUY
                assert received.price == 150.50
                assert received.quantity == 100
                assert received.timestamp == 1700000000123456789
                assert received.sequence_number == 1

                assert consumer.is_empty
                assert consumer.occupied == 0

    def test_publish_batch_and_receive_batch(self, tmp_path: Path) -> None:
        """Verify batch publishing and batch receiving operations."""
        filepath = tmp_path / "orders_batch.bin"
        capacity = 16

        orders = [
            Order(
                order_id=f"B-{i:02d}",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                price=Decimal(f"{100 + i}.25"),
                quantity=10 * (i + 1),
                timestamp=1700000000000000000 + i,
                sequence_number=i,
            )
            for i in range(10)
        ]

        with OrderProducer.create(filepath, capacity=capacity) as producer:
            published_count = producer.publish_batch(orders)
            assert published_count == 10
            assert producer.occupied == 10

            with OrderConsumer.open(filepath) as consumer:
                # Receive first 4 orders in batch
                batch_1 = consumer.receive_batch(max_orders=4)
                assert len(batch_1) == 4
                assert [o.order_id for o in batch_1] == [f"B-{i:02d}" for i in range(4)]

                # Drain remaining 6 orders
                remaining = consumer.drain()
                assert len(remaining) == 6
                assert [o.order_id for o in remaining] == [f"B-{i:02d}" for i in range(4, 10)]
                assert consumer.is_empty

    def test_poll_and_peek_semantics(self, tmp_path: Path) -> None:
        """Verify non-blocking/timed poll and peek operations."""
        filepath = tmp_path / "orders_poll.bin"
        capacity = 4

        with OrderProducer.create(filepath, capacity=capacity) as producer:
            with OrderConsumer.open(filepath) as consumer:
                # Poll empty buffer returns None without raising
                assert consumer.poll(timeout=0.01) is None
                assert consumer.poll() is None

                # Publish an order
                order = Order("P-1", OrderSide.SELL, 99.0, 5)
                producer.publish(order)

                # Peek returns order without removing it
                peeked = consumer.peek()
                assert peeked.order_id == "P-1"
                assert consumer.occupied == 1

                # Poll retrieves the order
                polled = consumer.poll(timeout=0.05)
                assert polled is not None
                assert polled.order_id == "P-1"
                assert consumer.is_empty

    def test_invalid_type_raises_type_error(self, tmp_path: Path) -> None:
        """Verify publishing an invalid non-Order object raises TypeError."""
        filepath = tmp_path / "orders_invalid.bin"
        with OrderProducer.create(filepath, capacity=4) as producer:
            with pytest.raises(TypeError, match="order must be an Order instance"):
                producer.publish("not-an-order")  # type: ignore[arg-type]


# ==============================================================================
# 2. Cross-Process Order Transfer & FIFO Ordering
# ==============================================================================


class TestCrossProcessTransferAndFIFO:
    """Verify transferring orders across child OS processes maintains strict FIFO ordering."""

    def test_cross_process_transfer_multiple_orders(self, tmp_path: Path) -> None:
        """Verify transferring 50 orders between separate producer and consumer processes."""
        filepath = tmp_path / "mp_fifo_50.bin"
        num_orders = 50
        capacity = 64

        orders = [
            Order(
                order_id=f"MP-{i:04d}",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                price=50.0 + (i * 0.25),
                quantity=100 + i,
                timestamp=1700000000000000000 + i * 1000,
                sequence_number=i + 1,
            )
            for i in range(num_orders)
        ]

        with OrderPipeline(filepath=filepath, capacity=capacity) as pipeline:
            pipeline.start(orders=orders, timeout=5.0)
            results = pipeline.drain_results(timeout=8.0)

            assert len(results) == num_orders
            for act, exp in zip(results, orders):
                assert act.order_id == exp.order_id
                assert act.side == exp.side
                assert act.price == exp.price
                assert act.quantity == exp.quantity
                assert act.timestamp == exp.timestamp
                assert act.sequence_number == exp.sequence_number

    def test_fifo_ordering_and_data_integrity_heterogeneous(self, tmp_path: Path) -> None:
        """Verify full precision roundtrip of float, int, and Decimal prices in FIFO order."""
        filepath = tmp_path / "mp_fifo_data_types.bin"
        capacity = 32

        orders = [
            Order(
                order_id=f"ORD-DEC-{i}",
                side=OrderSide.BUY,
                price=Decimal("12345.67890123"),
                quantity=5000,
                timestamp=1710000000000000001 + i,
                sequence_number=100 + i,
            )
            if i % 3 == 0
            else Order(
                order_id=f"ORD-FLT-{i}",
                side=OrderSide.SELL,
                price=199.95,
                quantity=10,
                timestamp=1710000000000000002 + i,
                sequence_number=200 + i,
            )
            if i % 3 == 1
            else Order(
                order_id=f"ORD-INT-{i}",
                side=OrderSide.BUY,
                price=25000,
                quantity=1,
                timestamp=1710000000000000003 + i,
                sequence_number=300 + i,
            )
            for i in range(30)
        ]

        with OrderPipeline(filepath=filepath, capacity=capacity) as pipeline:
            pipeline.start(orders=orders, timeout=5.0)
            results = pipeline.drain_results(timeout=8.0)

            assert len(results) == len(orders)
            for act, exp in zip(results, orders):
                assert act.order_id == exp.order_id
                assert act.side == exp.side
                assert act.price == exp.price
                assert type(act.price) is type(exp.price)
                assert act.quantity == exp.quantity
                assert act.timestamp == exp.timestamp
                assert act.sequence_number == exp.sequence_number


# ==============================================================================
# 3. Buffer Full / Over-Capacity Concurrent Draining
# ==============================================================================


class TestBufferCapacityAndConcurrentDraining:
    """Verify that when orders exceed buffer capacity, producer blocks and consumer drains concurrently."""

    def test_more_orders_than_capacity_concurrent_draining(self, tmp_path: Path) -> None:
        """Verify 128 orders streaming through a small ring buffer of capacity 8 (16x capacity)."""
        filepath = tmp_path / "mp_draining_8_128.bin"
        capacity = 8
        num_orders = 128

        orders = [
            Order(
                order_id=f"STREAM-{i:03d}",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                price=round(10.0 + (i * 0.1), 2),
                quantity=i + 1,
                timestamp=1700000000000000000 + i,
                sequence_number=i,
            )
            for i in range(num_orders)
        ]

        with OrderPipeline(filepath=filepath, capacity=capacity) as pipeline:
            # Start producer and consumer processes concurrently
            pipeline.start(orders=orders, timeout=5.0)
            results = pipeline.drain_results(timeout=10.0)

            assert len(results) == num_orders
            for i, (act, exp) in enumerate(zip(results, orders)):
                assert act.order_id == exp.order_id
                assert act.sequence_number == i
                assert act.price == exp.price
                assert act.quantity == exp.quantity

    def test_high_volume_concurrency(self, tmp_path: Path) -> None:
        """Verify 300 orders streaming through capacity 16 with zero drops or deadlocks."""
        filepath = tmp_path / "mp_high_volume_300.bin"
        capacity = 16
        num_orders = 300

        orders = [
            Order(
                order_id=f"HV-{i:04d}",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                price=100.0 + i,
                quantity=10,
                timestamp=1700000000000000000 + i,
                sequence_number=i,
            )
            for i in range(num_orders)
        ]

        with OrderPipeline(filepath=filepath, capacity=capacity) as pipeline:
            pipeline.start(orders=orders, timeout=5.0)
            results = pipeline.drain_results(timeout=12.0)

            assert len(results) == num_orders
            assert results[0].order_id == "HV-0000"
            assert results[-1].order_id == f"HV-{num_orders - 1:04d}"


# ==============================================================================
# 4. Empty & Full Buffer Waiting, Timeouts, and Interleaved Coordination
# ==============================================================================


class TestBufferWaitingAndTimeoutSemantics:
    """Verify non-blocking and timed blocking behaviors on empty and full buffers."""

    def test_consumer_empty_buffer_non_blocking_raises(self, tmp_path: Path) -> None:
        """Verify receive_nowait on empty buffer raises RingBufferEmptyError immediately."""
        filepath = tmp_path / "empty_nonblocking.bin"
        with OrderProducer.create(filepath, capacity=4):
            with OrderConsumer.open(filepath) as consumer:
                assert consumer.is_empty
                with pytest.raises(RingBufferEmptyError):
                    consumer.receive_nowait()

    def test_consumer_empty_buffer_blocking_timeout_raises(self, tmp_path: Path) -> None:
        """Verify receive with timeout on empty buffer raises RingBufferEmptyError after timeout."""
        filepath = tmp_path / "empty_timeout.bin"
        with OrderProducer.create(filepath, capacity=4):
            with OrderConsumer.open(filepath) as consumer:
                start = time.monotonic()
                with pytest.raises(RingBufferEmptyError):
                    consumer.receive(block=True, timeout=0.1)
                elapsed = time.monotonic() - start
                assert elapsed >= 0.08, f"Expected wait >= 0.08s, got {elapsed:.3f}s"

    def test_producer_full_buffer_non_blocking_raises(self, tmp_path: Path) -> None:
        """Verify publish_nowait on full buffer raises RingBufferFullError immediately."""
        filepath = tmp_path / "full_nonblocking.bin"
        capacity = 2

        with OrderProducer.create(filepath, capacity=capacity) as producer:
            producer.publish(Order("O-1", OrderSide.BUY, 100.0, 1))
            producer.publish(Order("O-2", OrderSide.SELL, 101.0, 2))
            assert producer.is_full

            with pytest.raises(RingBufferFullError):
                producer.publish_nowait(Order("O-3", OrderSide.BUY, 102.0, 3))

    def test_producer_full_buffer_blocking_timeout_raises(self, tmp_path: Path) -> None:
        """Verify publish with timeout on full buffer raises RingBufferFullError after timeout."""
        filepath = tmp_path / "full_timeout.bin"
        capacity = 2

        with OrderProducer.create(filepath, capacity=capacity) as producer:
            producer.publish(Order("O-1", OrderSide.BUY, 100.0, 1))
            producer.publish(Order("O-2", OrderSide.SELL, 101.0, 2))
            assert producer.is_full

            start = time.monotonic()
            with pytest.raises(RingBufferFullError):
                producer.publish(Order("O-3", OrderSide.BUY, 102.0, 3), block=True, timeout=0.1)
            elapsed = time.monotonic() - start
            assert elapsed >= 0.08, f"Expected wait >= 0.08s, got {elapsed:.3f}s"

    def test_cross_process_delayed_producer_unblocks_consumer(self, tmp_path: Path) -> None:
        """Verify consumer starts first and blocks, unblocking when producer subsequently writes."""
        filepath = tmp_path / "delayed_prod.bin"
        capacity = 4

        orders = [
            Order(f"DELAY-{i}", OrderSide.BUY, 100.0 + i, 10, sequence_number=i)
            for i in range(5)
        ]

        with OrderPipeline(filepath=filepath, capacity=capacity) as pipeline:
            # Start consumer first
            pipeline.start_consumer(expected_count=len(orders), timeout=5.0)

            # Pause briefly to ensure consumer is actively blocked waiting
            time.sleep(0.15)

            # Start producer second
            pipeline.start_producer(orders=orders, timeout=5.0)

            results = pipeline.drain_results(timeout=8.0)
            assert len(results) == len(orders)
            for act, exp in zip(results, orders):
                assert act.order_id == exp.order_id

    def test_cross_process_delayed_consumer_unblocks_full_producer(self, tmp_path: Path) -> None:
        """Verify producer fills buffer and blocks, unblocking when consumer subsequently drains."""
        filepath = tmp_path / "delayed_cons.bin"
        capacity = 4
        num_orders = 10  # Exceeds capacity 4

        orders = [
            Order(f"DELAY-C-{i}", OrderSide.SELL, 50.0 + i, 5, sequence_number=i)
            for i in range(num_orders)
        ]

        with OrderPipeline(filepath=filepath, capacity=capacity) as pipeline:
            # Start producer first (will fill 4 slots and block on 5th)
            pipeline.start_producer(orders=orders, timeout=5.0)

            # Pause briefly so producer fills buffer and blocks
            time.sleep(0.15)

            # Start consumer second to drain and unblock producer
            pipeline.start_consumer(expected_count=num_orders, timeout=5.0)

            results = pipeline.drain_results(timeout=8.0)
            assert len(results) == num_orders
            for act, exp in zip(results, orders):
                assert act.order_id == exp.order_id


# ==============================================================================
# 5. Process Failure Handling, Clean Shutdown & Windows Cleanup
# ==============================================================================


class TestProcessFailureAndCleanup:
    """Verify error propagation, graceful shutdown, and leak-free Windows resource cleanup."""

    def test_producer_failure_propagates_to_parent(self, tmp_path: Path) -> None:
        """Verify that an exception in the producer child process is caught and propagated."""
        filepath = tmp_path / "prod_failure.bin"
        capacity = 8

        with OrderPipeline(filepath=filepath, capacity=capacity) as pipeline:
            # Start producer with custom crashing target
            pipeline.start_producer(target=_crashing_producer_worker, timeout=5.0)

            with pytest.raises((ProducerProcessError, ProcessIntegrationError)) as exc_info:
                pipeline.join(timeout=5.0)

            assert "Simulated catastrophic producer hardware fault" in str(exc_info.value) or (
                "Producer process failed with exit code" in str(exc_info.value)
            )

    def test_consumer_failure_propagates_to_parent(self, tmp_path: Path) -> None:
        """Verify that an exception in the consumer child process is caught and propagated."""
        filepath = tmp_path / "cons_failure.bin"
        capacity = 8

        with OrderPipeline(filepath=filepath, capacity=capacity) as pipeline:
            pipeline.start_consumer(target=_crashing_consumer_worker, timeout=5.0)
            pipeline.ready_event.set()

            with pytest.raises((ConsumerProcessError, ProcessIntegrationError)) as exc_info:
                pipeline.join(timeout=5.0)

            assert "Simulated invalid binary memory record encountered" in str(exc_info.value) or (
                "Consumer process failed with exit code" in str(exc_info.value)
            )

    def test_graceful_pipeline_stop_indefinite_producer(self, tmp_path: Path) -> None:
        """Verify pipeline.stop() cleanly shuts down continuous background processes."""
        filepath = tmp_path / "infinite_stop.bin"
        capacity = 16

        with OrderPipeline(filepath=filepath, capacity=capacity) as pipeline:
            pipeline.start_producer(target=_infinite_producer_worker, timeout=5.0)

            # Let producer run briefly
            time.sleep(0.1)
            assert pipeline.is_alive

            # Stop pipeline gracefully
            pipeline.stop(timeout=4.0)

            assert not pipeline.is_alive
            assert pipeline.producer_process is not None
            assert pipeline.producer_process.exitcode == 0

    def test_windows_cleanup_and_unlink(self, tmp_path: Path) -> None:
        """Verify backing file can be unlinked after pipeline execution without WinError 32."""
        filepath = tmp_path / "cleanup_unlink.bin"
        capacity = 8
        num_orders = 10

        orders = [
            Order(f"CLN-{i}", OrderSide.BUY, 100.0, 1)
            for i in range(num_orders)
        ]

        pipeline = OrderPipeline(filepath=filepath, capacity=capacity)
        pipeline.start(orders=orders, timeout=5.0)
        results = pipeline.drain_results(timeout=8.0)
        assert len(results) == num_orders

        # Verify backing file exists while running
        assert filepath.exists()

        # Cleanup should close all handles and unlink the backing file
        pipeline.cleanup()
        assert not filepath.exists(), "Backing file should be deleted cleanly after cleanup"

    def test_pipeline_context_manager_cleanup_on_exception(self, tmp_path: Path) -> None:
        """Verify context manager cleans up backing file and processes even if test raises."""
        filepath = tmp_path / "ctx_cleanup_on_error.bin"
        capacity = 8

        with pytest.raises(ZeroDivisionError):
            with OrderPipeline(filepath=filepath, capacity=capacity) as pipeline:
                pipeline.start_producer(orders=[Order("O-1", OrderSide.BUY, 100.0, 1)])
                time.sleep(0.05)
                # Intentionally trigger an unhandled error inside context manager
                _ = 1 / 0

        # After exiting context, file should be unlinked and process terminated
        assert not filepath.exists()


# ==============================================================================
# 6. Standalone ProducerProcess and ConsumerProcess Tests
# ==============================================================================


class TestStandaloneProcesses:
    """Verify ProducerProcess and ConsumerProcess work independently with local endpoints."""

    def test_producer_process_with_main_consumer(self, tmp_path: Path) -> None:
        """Run ProducerProcess in child process and consume in main test thread."""
        filepath = tmp_path / "standalone_prod.bin"
        capacity = 8
        num_orders = 15

        orders = [
            Order(f"STAND-P-{i}", OrderSide.BUY, 75.0 + i, 10 + i, sequence_number=i)
            for i in range(num_orders)
        ]

        # Initialize backing file
        producer_init = OrderProducer.create(filepath, capacity=capacity)
        producer_init.close()

        ready_event = mp.Event()
        stop_event = mp.Event()
        error_queue = mp.Queue()

        prod_proc = ProducerProcess(
            filepath=filepath,
            orders=orders,
            ready_event=ready_event,
            stop_event=stop_event,
            error_queue=error_queue,
            timeout=5.0,
        )

        try:
            prod_proc.start()
            assert ready_event.wait(timeout=5.0)

            received: list[Order] = []
            with OrderConsumer.open(filepath) as consumer:
                for _ in range(num_orders):
                    received.append(consumer.receive(block=True, timeout=5.0))

            prod_proc.join(timeout=5.0)
            assert not prod_proc.is_alive()
            assert prod_proc.exitcode == 0
            assert len(received) == num_orders
            for act, exp in zip(received, orders):
                assert act.order_id == exp.order_id
        finally:
            if prod_proc.is_alive():
                prod_proc.terminate()
                prod_proc.join(timeout=1.0)
            error_queue.close()
            error_queue.join_thread()
            if filepath.exists():
                filepath.unlink()

    def test_consumer_process_with_main_producer(self, tmp_path: Path) -> None:
        """Run ConsumerProcess in child process and produce in main test thread."""
        filepath = tmp_path / "standalone_cons.bin"
        capacity = 8
        num_orders = 15

        orders = [
            Order(f"STAND-C-{i}", OrderSide.SELL, 80.0 + i, 20 + i, sequence_number=i)
            for i in range(num_orders)
        ]

        # Initialize backing file
        prod = OrderProducer.create(filepath, capacity=capacity)

        ready_event = mp.Event()
        stop_event = mp.Event()
        start_event = mp.Event()
        error_queue = mp.Queue()
        result_queue = mp.Queue()

        cons_proc = ConsumerProcess(
            filepath=filepath,
            expected_count=num_orders,
            result_queue=result_queue,
            ready_event=ready_event,
            stop_event=stop_event,
            error_queue=error_queue,
            timeout=5.0,
            start_event=start_event,
        )

        try:
            cons_proc.start()
            ready_event.set()
            assert start_event.wait(timeout=5.0)

            # Publish from main process
            for order in orders:
                prod.publish(order, block=True, timeout=5.0)
            prod.flush()

            cons_proc.join(timeout=5.0)
            assert not cons_proc.is_alive()
            assert cons_proc.exitcode == 0

            results: list[Order] = []
            for _ in range(num_orders):
                results.append(result_queue.get(timeout=3.0))

            assert len(results) == num_orders
            for act, exp in zip(results, orders):
                assert act.order_id == exp.order_id
        finally:
            prod.close()
            if cons_proc.is_alive():
                cons_proc.terminate()
                cons_proc.join(timeout=1.0)
            result_queue.close()
            result_queue.join_thread()
            error_queue.close()
            error_queue.join_thread()
            if filepath.exists():
                filepath.unlink()
