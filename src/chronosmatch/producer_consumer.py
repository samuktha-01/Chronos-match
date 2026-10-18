"""Cross-Process Producer-Consumer Integration for ChronosMatch.

Enables safe, low-latency, deterministic transfer of binary orders between
independent operating system processes backed by shared memory and the
fixed-capacity SPSC ring buffer.

Key Components:
- `OrderProducer`: Dedicated producer interface for publishing orders.
- `OrderConsumer`: Dedicated consumer interface for receiving and polling orders.
- `ProducerProcess`: Multiprocessing worker process for streaming orders.
- `ConsumerProcess`: Multiprocessing worker process for consuming orders.
- `OrderPipeline`: End-to-end lifecycle orchestrator for concurrent cross-process execution.
- `ProcessIntegrationError`, `ProducerProcessError`, `ConsumerProcessError`, `ProcessTimeoutError`:
  Structured exceptions for process lifecycle and communication errors.

Windows & Python 3.12 Compatibility:
- Uses module-level worker functions compatible with the Windows multiprocessing `spawn` start method.
- Ensures all file descriptors and memory mappings are properly closed before file unlinking,
  preventing Windows `PermissionError` (WinError 32).
- Employs bounded waits and joins to guarantee test suite and application termination.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback
from typing import Any, Callable, Self

from chronosmatch.binary_protocol import (
    ORDER_STRUCT_SIZE,
    decode_order,
    encode_order,
)
from chronosmatch.models import Order
from chronosmatch.ring_buffer import (
    RingBuffer,
    RingBufferClosedError,
    RingBufferConsumer,
    RingBufferEmptyError,
    RingBufferFullError,
    RingBufferProducer,
)
from chronosmatch.shared_memory import (
    SharedMemoryError,
    SharedMemoryReadOnlyError,
)

# ==============================================================================
# Exceptions
# ==============================================================================


class ProcessIntegrationError(SharedMemoryError):
    """Base exception for ChronosMatch cross-process producer-consumer operations."""


class ProducerProcessError(ProcessIntegrationError):
    """Raised when a producer child process fails or exits with an error."""


class ConsumerProcessError(ProcessIntegrationError):
    """Raised when a consumer child process fails or exits with an error."""


class ProcessTimeoutError(ProcessIntegrationError, TimeoutError):
    """Raised when a cross-process operation, sync event, or join exceeds its timeout."""


# ==============================================================================
# OrderProducer Interface
# ==============================================================================


class OrderProducer:
    """High-level producer interface for publishing orders to a shared-memory ring buffer.

    Can be constructed directly from a backing file path (creating or opening the ring buffer),
    or by wrapping an existing `RingBuffer` or `RingBufferProducer` instance.
    """

    def __init__(
        self,
        target: str | Path | RingBuffer | RingBufferProducer,
        capacity: int | None = None,
        *,
        create: bool = False,
        record_size: int = ORDER_STRUCT_SIZE,
    ) -> None:
        """Initialize an OrderProducer.

        Args:
            target: Backing file path (str or Path), existing RingBuffer, or RingBufferProducer.
            capacity: Buffer capacity in records. Required if create=True.
            create: If True, creates and initializes a new backing file. If False, opens existing.
            record_size: Record size in bytes (default: ORDER_STRUCT_SIZE = 64).

        Raises:
            TypeError: If target has an invalid type.
            ValueError: If configuration arguments are invalid.
        """
        self._owns_ring_buffer: bool = False
        self._producer: RingBufferProducer

        if isinstance(target, RingBufferProducer):
            self._producer = target
        elif isinstance(target, RingBuffer):
            self._producer = RingBufferProducer(target)
        elif isinstance(target, (str, Path)):
            filepath = Path(target).resolve()
            if create:
                if capacity is None:
                    raise ValueError("capacity is required when create=True")
                rb = RingBuffer.create(
                    filepath=filepath,
                    capacity=capacity,
                    record_size=record_size,
                )
            else:
                rb = RingBuffer.open(filepath=filepath, read_only=False)
            self._producer = RingBufferProducer(rb)
            self._owns_ring_buffer = True
        else:
            raise TypeError(
                f"target must be a str, Path, RingBuffer, or RingBufferProducer, got {type(target).__name__}"
            )

    @classmethod
    def create(
        cls,
        filepath: str | Path,
        capacity: int,
        record_size: int = ORDER_STRUCT_SIZE,
    ) -> OrderProducer:
        """Factory method to create a new file-backed OrderProducer."""
        return cls(filepath, capacity=capacity, create=True, record_size=record_size)

    @classmethod
    def open(cls, filepath: str | Path) -> OrderProducer:
        """Factory method to open an existing file-backed OrderProducer."""
        return cls(filepath, create=False)

    # --------------------------------------------------------------------------
    # Publishing API
    # --------------------------------------------------------------------------

    def publish(
        self,
        order: Order,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        """Publish an order into the ring buffer.

        Args:
            order: Valid Order instance to serialize and push.
            block: If True, blocks until space is available or timeout expires.
                   If False, raises RingBufferFullError immediately if full.
            timeout: Maximum seconds to wait when block=True (None for indefinite).

        Raises:
            TypeError: If order is not an Order instance.
            RingBufferFullError: If buffer is full and block=False or timeout expires.
            RingBufferClosedError: If buffer is closed.
        """
        if not isinstance(order, Order):
            raise TypeError(f"order must be an Order instance, got {type(order).__name__}")
        self._producer.put(order, block=block, timeout=timeout)

    def publish_nowait(self, order: Order) -> None:
        """Publish an order without blocking.

        Raises:
            RingBufferFullError: If buffer is full.
        """
        self.publish(order, block=False)

    def publish_batch(
        self,
        orders: Iterable[Order],
        block: bool = True,
        timeout: float | None = None,
    ) -> int:
        """Publish an iterable of orders into the ring buffer.

        Args:
            orders: Iterable of Order instances.
            block: If True, blocks on full buffer.
            timeout: Timeout per order when block=True.

        Returns:
            Number of orders successfully published.
        """
        count = 0
        for order in orders:
            self.publish(order, block=block, timeout=timeout)
            count += 1
        return count

    # Convenience aliases
    send = publish
    send_nowait = publish_nowait
    put = publish
    put_nowait = publish_nowait

    def flush(self) -> None:
        """Flush shared-memory writes to backing storage."""
        self._producer.flush()

    def close(self) -> None:
        """Close the producer and release shared-memory resources."""
        self._producer.close()

    def unlink(self) -> None:
        """Close producer and permanently remove backing file from disk."""
        self._producer.unlink()

    # --------------------------------------------------------------------------
    # Properties & Protocols
    # --------------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        """Total buffer capacity in orders."""
        return self._producer.capacity

    @property
    def occupied(self) -> int:
        """Number of unconsumed orders currently in the buffer."""
        return self._producer.ring_buffer.occupied

    @property
    def available_write(self) -> int:
        """Number of free order slots available for writing."""
        return self._producer.available_write

    @property
    def is_full(self) -> bool:
        """True if the buffer is at maximum capacity."""
        return self._producer.is_full

    @property
    def is_closed(self) -> bool:
        """True if the producer is closed."""
        return self._producer.is_closed

    @property
    def filepath(self) -> Path | None:
        """Path to backing file, or None if anonymous."""
        return self._producer.ring_buffer.filepath

    @property
    def ring_buffer(self) -> RingBuffer:
        """Underlying RingBuffer instance."""
        return self._producer.ring_buffer

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        status = "closed" if self.is_closed else f"occupied={self.occupied}/{self.capacity}"
        return f"<OrderProducer {status}>"


# ==============================================================================
# OrderConsumer Interface
# ==============================================================================


class OrderConsumer:
    """High-level consumer interface for receiving orders from a shared-memory ring buffer.

    Can be constructed from a backing file path or an existing `RingBuffer` / `RingBufferConsumer`.
    """

    def __init__(
        self,
        target: str | Path | RingBuffer | RingBufferConsumer,
    ) -> None:
        """Initialize an OrderConsumer.

        Args:
            target: Backing file path (str or Path), existing RingBuffer, or RingBufferConsumer.

        Raises:
            TypeError: If target has an invalid type.
            FileNotFoundError: If opening a file that does not exist.
        """
        self._owns_ring_buffer: bool = False
        self._consumer: RingBufferConsumer

        if isinstance(target, RingBufferConsumer):
            self._consumer = target
        elif isinstance(target, RingBuffer):
            self._consumer = RingBufferConsumer(target)
        elif isinstance(target, (str, Path)):
            filepath = Path(target).resolve()
            rb = RingBuffer.open(filepath=filepath, read_only=False)
            self._consumer = RingBufferConsumer(rb)
            self._owns_ring_buffer = True
        else:
            raise TypeError(
                f"target must be a str, Path, RingBuffer, or RingBufferConsumer, got {type(target).__name__}"
            )

    @classmethod
    def open(cls, filepath: str | Path) -> OrderConsumer:
        """Factory method to open an existing file-backed OrderConsumer."""
        return cls(filepath)

    # --------------------------------------------------------------------------
    # Receiving API
    # --------------------------------------------------------------------------

    def receive(
        self,
        block: bool = True,
        timeout: float | None = None,
    ) -> Order:
        """Receive and remove the next order from the ring buffer.

        Args:
            block: If True, blocks until an order is available or timeout expires.
                   If False, raises RingBufferEmptyError immediately if empty.
            timeout: Maximum seconds to wait when block=True (None for indefinite).

        Returns:
            Reconstructed Order instance.

        Raises:
            RingBufferEmptyError: If buffer is empty and block=False or timeout expires.
            RingBufferClosedError: If buffer is closed.
        """
        return self._consumer.get(block=block, timeout=timeout)

    def receive_nowait(self) -> Order:
        """Receive the next order without blocking.

        Raises:
            RingBufferEmptyError: If buffer is empty.
        """
        return self.receive(block=False)

    def poll(self, timeout: float | None = None) -> Order | None:
        """Poll for an order with a timeout, returning None instead of raising if empty.

        Args:
            timeout: Maximum seconds to wait (None to check non-blocking).

        Returns:
            Order if available within timeout, else None.
        """
        try:
            if timeout is None:
                return self.receive_nowait()
            return self.receive(block=True, timeout=timeout)
        except RingBufferEmptyError:
            return None

    def receive_batch(
        self,
        max_orders: int | None = None,
        block: bool = False,
        timeout: float | None = None,
    ) -> list[Order]:
        """Receive a batch of up to `max_orders` from the ring buffer.

        Args:
            max_orders: Maximum number of orders to receive (None for all available).
            block: If True, waits for at least one order if buffer is empty.
            timeout: Timeout for initial wait when block=True.

        Returns:
            List of received Order instances.
        """
        results: list[Order] = []
        if block:
            try:
                first = self.receive(block=True, timeout=timeout)
                results.append(first)
            except RingBufferEmptyError:
                return results

        while max_orders is None or len(results) < max_orders:
            try:
                results.append(self.receive_nowait())
            except RingBufferEmptyError:
                break
        return results

    def drain(self) -> list[Order]:
        """Drain and return all currently available orders without blocking."""
        return self.receive_batch(max_orders=None, block=False)

    def peek(self) -> Order:
        """Inspect the next order without removing it from the buffer.

        Returns:
            Next Order instance.

        Raises:
            RingBufferEmptyError: If buffer is empty.
        """
        return self._consumer.peek()

    # Convenience aliases
    get = receive
    get_nowait = receive_nowait

    def close(self) -> None:
        """Close the consumer and release shared-memory resources."""
        self._consumer.close()

    # --------------------------------------------------------------------------
    # Properties & Protocols
    # --------------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        """Total buffer capacity in orders."""
        return self._consumer.capacity

    @property
    def occupied(self) -> int:
        """Number of unconsumed orders currently in the buffer."""
        return self._consumer.ring_buffer.occupied

    @property
    def available_read(self) -> int:
        """Number of unconsumed orders available for reading."""
        return self._consumer.available_read

    @property
    def is_empty(self) -> bool:
        """True if the buffer contains zero unconsumed orders."""
        return self._consumer.is_empty

    @property
    def is_closed(self) -> bool:
        """True if the consumer is closed."""
        return self._consumer.is_closed

    @property
    def filepath(self) -> Path | None:
        """Path to backing file, or None if anonymous."""
        return self._consumer.ring_buffer.filepath

    @property
    def ring_buffer(self) -> RingBuffer:
        """Underlying RingBuffer instance."""
        return self._consumer.ring_buffer

    def __iter__(self) -> Iterator[Order]:
        """Iterate over available orders until the buffer is empty."""
        while not self.is_empty:
            try:
                yield self.receive_nowait()
            except RingBufferEmptyError:
                break

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        status = "closed" if self.is_closed else f"occupied={self.occupied}/{self.capacity}"
        return f"<OrderConsumer {status}>"


# ==============================================================================
# Worker Functions (Top-Level for Windows `spawn` Subprocesses)
# ==============================================================================


def _default_producer_worker(
    filepath_str: str,
    orders: list[Order] | None,
    ready_event: Any = None,
    stop_event: Any = None,
    error_queue: Any = None,
    timeout: float | None = 5.0,
    sent_counter: Any = None,
    input_queue: Any = None,
) -> None:
    """Subprocess entry point for producing orders into a shared ring buffer."""
    producer: OrderProducer | None = None
    try:
        producer = OrderProducer(filepath_str, create=False)
        if ready_event is not None:
            ready_event.set()

        # Send static list of orders if provided
        if orders is not None:
            for order in orders:
                if stop_event is not None and stop_event.is_set():
                    break
                producer.publish(order, block=True, timeout=timeout)
                if sent_counter is not None:
                    with sent_counter.get_lock():
                        sent_counter.value += 1

        # Stream dynamic orders from input queue if provided
        if input_queue is not None:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                try:
                    item = input_queue.get(timeout=0.05)
                except Exception:
                    continue
                if item is None:  # Sentinel value signaling end of stream
                    break
                producer.publish(item, block=True, timeout=timeout)
                if sent_counter is not None:
                    with sent_counter.get_lock():
                        sent_counter.value += 1

        producer.flush()
    except Exception as e:
        if error_queue is not None:
            try:
                error_queue.put((type(e).__name__, str(e), traceback.format_exc()))
            except Exception:
                pass
        raise
    finally:
        if producer is not None:
            try:
                producer.close()
            except Exception:
                pass


def _default_consumer_worker(
    filepath_str: str,
    expected_count: int | None,
    result_queue: Any,
    ready_event: Any = None,
    stop_event: Any = None,
    error_queue: Any = None,
    timeout: float | None = 5.0,
    start_event: Any = None,
) -> None:
    """Subprocess entry point for consuming orders from a shared ring buffer."""
    consumer: OrderConsumer | None = None
    try:
        # Wait until producer / parent indicates backing file is initialized
        if ready_event is not None:
            if not ready_event.wait(timeout=10.0):
                raise TimeoutError("Consumer timed out waiting for producer ready_event")

        consumer = OrderConsumer(filepath_str)
        if start_event is not None:
            start_event.set()

        received_count = 0
        idle_start = time.monotonic()

        while True:
            # Check exit conditions
            if expected_count is not None and received_count >= expected_count:
                break
            if stop_event is not None and stop_event.is_set() and consumer.is_empty:
                break

            try:
                order = consumer.receive(block=True, timeout=0.05)
                received_count += 1
                idle_start = time.monotonic()
                if result_queue is not None:
                    result_queue.put(order)
            except RingBufferEmptyError:
                if stop_event is not None and stop_event.is_set() and consumer.is_empty:
                    break
                if timeout is not None and (time.monotonic() - idle_start) >= timeout:
                    raise RingBufferEmptyError(
                        f"Consumer timed out after {timeout:.2f}s waiting for order #{received_count + 1}"
                    )
    except Exception as e:
        if error_queue is not None:
            try:
                error_queue.put((type(e).__name__, str(e), traceback.format_exc()))
            except Exception:
                pass
        raise
    finally:
        if consumer is not None:
            try:
                consumer.close()
            except Exception:
                pass


# ==============================================================================
# Process Subclasses
# ==============================================================================


class ProducerProcess(mp.Process):
    """Dedicated child process for publishing orders into a shared ring buffer."""

    def __init__(
        self,
        filepath: str | Path,
        orders: list[Order] | None = None,
        *,
        ready_event: Any = None,
        stop_event: Any = None,
        error_queue: Any = None,
        timeout: float | None = 5.0,
        sent_counter: Any = None,
        input_queue: Any = None,
        target: Callable[..., Any] | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize ProducerProcess.

        Args:
            filepath: Backing file path for the ring buffer.
            orders: Sequence of orders to publish.
            ready_event: Multiprocessing Event signaled when producer is initialized.
            stop_event: Multiprocessing Event signaled to trigger shutdown.
            error_queue: Multiprocessing Queue for error propagation.
            timeout: Per-order put timeout in seconds.
            sent_counter: Optional multiprocessing Value tracking orders published.
            input_queue: Optional multiprocessing Queue streaming dynamic orders.
            target: Optional custom worker function. If None, uses default worker.
            args: Arguments for custom target function.
            kwargs: Keyword arguments for custom target function.
        """
        if target is not None:
            super().__init__(target=target, args=args, kwargs=kwargs or {})
        else:
            super().__init__()

        self.filepath_str = str(Path(filepath).resolve())
        self.orders = orders
        self.ready_event = ready_event
        self.stop_event = stop_event
        self.error_queue = error_queue
        self.timeout = timeout
        self.sent_counter = sent_counter
        self.input_queue = input_queue
        self._custom_target = target is not None

    def run(self) -> None:
        """Run default producer worker if no custom target was supplied."""
        if self._custom_target:
            super().run()
        else:
            _default_producer_worker(
                filepath_str=self.filepath_str,
                orders=self.orders,
                ready_event=self.ready_event,
                stop_event=self.stop_event,
                error_queue=self.error_queue,
                timeout=self.timeout,
                sent_counter=self.sent_counter,
                input_queue=self.input_queue,
            )


class ConsumerProcess(mp.Process):
    """Dedicated child process for consuming orders from a shared ring buffer."""

    def __init__(
        self,
        filepath: str | Path,
        expected_count: int | None = None,
        *,
        result_queue: Any = None,
        ready_event: Any = None,
        stop_event: Any = None,
        error_queue: Any = None,
        timeout: float | None = 5.0,
        start_event: Any = None,
        target: Callable[..., Any] | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize ConsumerProcess.

        Args:
            filepath: Backing file path for the ring buffer.
            expected_count: Number of orders expected before shutting down (None for open-ended).
            result_queue: Multiprocessing Queue for received orders.
            ready_event: Multiprocessing Event waited on before opening the ring buffer.
            stop_event: Multiprocessing Event signaled to trigger shutdown.
            error_queue: Multiprocessing Queue for error propagation.
            timeout: Consumer idle timeout in seconds.
            start_event: Multiprocessing Event signaled when consumer begins reading.
            target: Optional custom worker function. If None, uses default worker.
            args: Arguments for custom target function.
            kwargs: Keyword arguments for custom target function.
        """
        if target is not None:
            super().__init__(target=target, args=args, kwargs=kwargs or {})
        else:
            super().__init__()

        self.filepath_str = str(Path(filepath).resolve())
        self.expected_count = expected_count
        self.result_queue = result_queue
        self.ready_event = ready_event
        self.stop_event = stop_event
        self.error_queue = error_queue
        self.timeout = timeout
        self.start_event = start_event
        self._custom_target = target is not None

    def run(self) -> None:
        """Run default consumer worker if no custom target was supplied."""
        if self._custom_target:
            super().run()
        else:
            _default_consumer_worker(
                filepath_str=self.filepath_str,
                expected_count=self.expected_count,
                result_queue=self.result_queue,
                ready_event=self.ready_event,
                stop_event=self.stop_event,
                error_queue=self.error_queue,
                timeout=self.timeout,
                start_event=self.start_event,
            )


# ==============================================================================
# OrderPipeline Orchestrator
# ==============================================================================


class OrderPipeline:
    """Manages the full lifecycle of cross-process producer and consumer execution.

    Coordinates shared memory initialization, concurrent process spawning, bounded joins,
    error propagation, and deterministic resource cleanup.
    """

    def __init__(
        self,
        filepath: str | Path,
        capacity: int,
        record_size: int = ORDER_STRUCT_SIZE,
    ) -> None:
        """Initialize an OrderPipeline.

        Args:
            filepath: Path to the backing shared memory file.
            capacity: Buffer capacity in order records.
            record_size: Record size in bytes (default: ORDER_STRUCT_SIZE = 64).
        """
        self._filepath: Path = Path(filepath).resolve()
        self._capacity: int = capacity
        self._record_size: int = record_size

        self._producer_proc: ProducerProcess | None = None
        self._consumer_proc: ConsumerProcess | None = None

        self._ready_event: Any = mp.Event()
        self._stop_event: Any = mp.Event()
        self._start_event: Any = mp.Event()
        self._error_queue: mp.Queue = mp.Queue()
        self._result_queue: mp.Queue = mp.Queue()

        self._local_producer: OrderProducer | None = None
        self._local_consumer: OrderConsumer | None = None
        self._buffered_results: list[Order] = []
        self._is_closed: bool = False

    def _ensure_buffer_initialized(self) -> None:
        """Create and initialize the backing file header if not already created."""
        if not self._filepath.exists():
            rb = RingBuffer.create(
                filepath=self._filepath,
                capacity=self._capacity,
                record_size=self._record_size,
            )
            rb.close()

    def start_producer(
        self,
        orders: list[Order] | None = None,
        timeout: float = 5.0,
        sent_counter: Any = None,
        input_queue: Any = None,
        target: Callable[..., Any] | None = None,
        args: tuple[Any, ...] | None = None,
    ) -> ProducerProcess:
        """Start producer process."""
        self._ensure_buffer_initialized()
        worker_args: tuple[Any, ...] = ()
        if target is not None:
            if args is not None:
                worker_args = args
            else:
                worker_args = (str(self._filepath), self._ready_event, self._stop_event, self._error_queue)

        self._producer_proc = ProducerProcess(
            filepath=self._filepath,
            orders=orders,
            ready_event=self._ready_event,
            stop_event=self._stop_event,
            error_queue=self._error_queue,
            timeout=timeout,
            sent_counter=sent_counter,
            input_queue=input_queue,
            target=target,
            args=worker_args,
        )
        self._producer_proc.start()
        return self._producer_proc

    def start_consumer(
        self,
        expected_count: int | None = None,
        timeout: float = 5.0,
        target: Callable[..., Any] | None = None,
        args: tuple[Any, ...] | None = None,
    ) -> ConsumerProcess:
        """Start consumer process."""
        self._ensure_buffer_initialized()
        worker_args: tuple[Any, ...] = ()
        if target is not None:
            if args is not None:
                worker_args = args
            else:
                worker_args = (str(self._filepath), self._ready_event, self._error_queue)

        self._consumer_proc = ConsumerProcess(
            filepath=self._filepath,
            expected_count=expected_count,
            result_queue=self._result_queue,
            ready_event=self._ready_event,
            stop_event=self._stop_event,
            error_queue=self._error_queue,
            timeout=timeout,
            start_event=self._start_event,
            target=target,
            args=worker_args,
        )
        self._consumer_proc.start()
        return self._consumer_proc

    def start(
        self,
        orders: list[Order] | None = None,
        expected_count: int | None = None,
        timeout: float = 5.0,
    ) -> tuple[ProducerProcess, ConsumerProcess]:
        """Start both producer and consumer processes concurrently.

        Args:
            orders: Sequence of orders for producer to stream.
            expected_count: Expected order count for consumer (defaults to len(orders) if provided).
            timeout: Put/get timeout for worker operations.

        Returns:
            Tuple of (ProducerProcess, ConsumerProcess).
        """
        if expected_count is None and orders is not None:
            expected_count = len(orders)

        self._ensure_buffer_initialized()
        producer = self.start_producer(orders=orders, timeout=timeout)
        consumer = self.start_consumer(expected_count=expected_count, timeout=timeout)
        return producer, consumer

    def get_producer(self) -> OrderProducer:
        """Get or create an in-process OrderProducer connected to the pipeline buffer."""
        self._ensure_buffer_initialized()
        if self._local_producer is None or self._local_producer.is_closed:
            self._local_producer = OrderProducer.open(self._filepath)
        return self._local_producer

    def get_consumer(self) -> OrderConsumer:
        """Get or create an in-process OrderConsumer connected to the pipeline buffer."""
        self._ensure_buffer_initialized()
        if self._local_consumer is None or self._local_consumer.is_closed:
            self._local_consumer = OrderConsumer.open(self._filepath)
        return self._local_consumer

    def join(self, timeout: float = 10.0) -> None:
        """Wait for started child processes to finish within bounded timeout and propagate errors.

        Continuously drains result_queue while waiting to prevent child processes from blocking
        on full OS pipe buffers.

        Args:
            timeout: Maximum seconds to wait for each process to terminate.

        Raises:
            ProcessTimeoutError: If any process fails to terminate within timeout.
            ProducerProcessError: If the producer process exited with an error.
            ConsumerProcessError: If the consumer process exited with an error.
            ProcessIntegrationError: If a child process raised an unhandled exception.
        """
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            # Drain result queue into buffered results to prevent pipe deadlock
            if self._result_queue is not None:
                while True:
                    try:
                        self._buffered_results.append(self._result_queue.get_nowait())
                    except Exception:
                        break

            if not self.is_alive:
                break

            time.sleep(0.005)

        # Drain one more time
        if self._result_queue is not None:
            while True:
                try:
                    self._buffered_results.append(self._result_queue.get_nowait())
                except Exception:
                    break

        # Check if any process is still alive after deadline
        for name, proc in (("producer", self._producer_proc), ("consumer", self._consumer_proc)):
            if proc is not None and proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)
                raise ProcessTimeoutError(f"{name.capitalize()} process timed out after {timeout}s")

        # Check error queue for propagated exceptions
        if not self._error_queue.empty():
            err_type, err_msg, tb = self._error_queue.get_nowait()
            raise ProcessIntegrationError(
                f"Child process encountered {err_type}: {err_msg}\n{tb}"
            )

        # Check process exit codes
        if self._producer_proc is not None and self._producer_proc.exitcode != 0:
            raise ProducerProcessError(
                f"Producer process failed with exit code {self._producer_proc.exitcode}"
            )
        if self._consumer_proc is not None and self._consumer_proc.exitcode != 0:
            raise ConsumerProcessError(
                f"Consumer process failed with exit code {self._consumer_proc.exitcode}"
            )

    def stop(self, timeout: float = 5.0) -> None:
        """Signal clean shutdown to child processes and wait bounded time for completion."""
        self._stop_event.set()
        try:
            self.join(timeout=timeout)
        except ProcessTimeoutError:
            pass

    def get_results(self, timeout: float = 1.0) -> list[Order]:
        """Collect and return all orders accumulated in the result queue.

        Args:
            timeout: Per-item queue read timeout.

        Returns:
            List of decoded Order instances.
        """
        results: list[Order] = list(self._buffered_results)
        self._buffered_results.clear()
        while True:
            try:
                results.append(self._result_queue.get(timeout=0.01))
            except Exception:
                break
        return results

    def drain_results(self, timeout: float = 10.0) -> list[Order]:
        """Join child processes and retrieve all results.

        Args:
            timeout: Timeout for join.

        Returns:
            List of received Order instances.
        """
        self.join(timeout=timeout)
        return self.get_results(timeout=1.0)


    def cleanup(self) -> None:
        """Clean up all processes, queues, mappings, and backing files."""
        if self._is_closed:
            return
        self._is_closed = True

        # Terminate any remaining child processes
        for proc in (self._producer_proc, self._consumer_proc):
            if proc is not None and proc.is_alive():
                try:
                    proc.terminate()
                    proc.join(timeout=1.0)
                except Exception:
                    pass

        # Close local handles
        if self._local_producer is not None:
            try:
                self._local_producer.close()
            except Exception:
                pass
            self._local_producer = None

        if self._local_consumer is not None:
            try:
                self._local_consumer.close()
            except Exception:
                pass
            self._local_consumer = None

        # Close queues
        for q in (self._error_queue, self._result_queue):
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass

        # Remove backing file
        if self._filepath.exists():
            try:
                self._filepath.unlink()
            except Exception:
                pass

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------

    @property
    def filepath(self) -> Path:
        """Path to backing file."""
        return self._filepath

    @property
    def capacity(self) -> int:
        """Buffer capacity in records."""
        return self._capacity

    @property
    def producer_process(self) -> ProducerProcess | None:
        """Producer child process instance, or None if not started."""
        return self._producer_proc

    @property
    def consumer_process(self) -> ConsumerProcess | None:
        """Consumer child process instance, or None if not started."""
        return self._consumer_proc

    @property
    def ready_event(self) -> Any:
        """Multiprocessing ready event."""
        return self._ready_event

    @property
    def stop_event(self) -> Any:
        """Multiprocessing stop event."""
        return self._stop_event

    @property
    def is_alive(self) -> bool:
        """True if either child process is running."""
        prod_alive = self._producer_proc is not None and self._producer_proc.is_alive()
        cons_alive = self._consumer_proc is not None and self._consumer_proc.is_alive()
        return prod_alive or cons_alive

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cleanup()
