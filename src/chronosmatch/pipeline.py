"""End-to-End Trading Pipeline for ChronosMatch.

Integrates the synthetic market simulator, 64-byte binary protocol, shared-memory IPC,
lock-free SPSC ring buffer, order producer/consumer, and deterministic limit order book
matching engine into a unified, high-performance simulation pipeline.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
import math
from pathlib import Path
import time
from typing import Any, Self

from chronosmatch.binary_protocol import (
    ORDER_STRUCT_SIZE,
    decode_order,
    encode_order,
)
from chronosmatch.cython_engine import (
    CythonMatchingEngine,
    is_cython_compiled,
)
from chronosmatch.market_simulator import (
    BackpressureMode,
    BackpressureTimeoutError,
    MarketFeedConfig,
    MarketSimulator,
    OrderCancel,
    SimulationMode,
)
from chronosmatch.matching_engine import MatchingEngine, Trade
from chronosmatch.models import Order, OrderSide
from chronosmatch.persistence import (
    DurabilityPolicy,
    RecoveryManager,
    TradeStore,
    WriteAheadLog,
)
from chronosmatch.order_book import DuplicateOrderError, LimitOrderBook, OrderNotFoundError
from chronosmatch.producer_consumer import OrderConsumer, OrderProducer
from chronosmatch.ring_buffer import (
    RingBuffer,
    RingBufferClosedError,
    RingBufferEmptyError,
    RingBufferFullError,
)
from chronosmatch.shared_memory import (
    MalformedRecordError,
    SharedMemoryError,
)

# ==============================================================================
# Pipeline Metrics
# ==============================================================================


@dataclass
class TradingPipelineMetrics:
    """Comprehensive observability metrics for the end-to-end trading pipeline."""

    orders_generated: int = 0
    orders_published: int = 0
    orders_consumed: int = 0
    orders_processed: int = 0
    orders_cancelled: int = 0
    orders_dropped: int = 0
    malformed_messages: int = 0
    duplicate_orders: int = 0
    trades_executed: int = 0
    volume_executed: int = 0
    backpressure_events: int = 0
    start_time_ns: int | None = None
    stop_time_ns: int | None = None
    wall_start_time_ns: int | None = None
    wall_stop_time_ns: int | None = None

    @property
    def elapsed_seconds(self) -> float:
        """Total elapsed runtime in seconds measured via monotonic high-resolution clock."""
        if self.start_time_ns is None:
            return 0.0
        end_ns = self.stop_time_ns if self.stop_time_ns is not None else time.perf_counter_ns()
        return max(0.0, (end_ns - self.start_time_ns) / 1_000_000_000.0)

    @property
    def throughput_orders_per_sec(self) -> float:
        """Orders successfully processed through the matching engine per second."""
        elapsed = self.elapsed_seconds
        if elapsed <= 0.0:
            return 0.0
        return self.orders_processed / elapsed

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a diagnostic dictionary."""
        return {
            "orders_generated": self.orders_generated,
            "orders_published": self.orders_published,
            "orders_consumed": self.orders_consumed,
            "orders_processed": self.orders_processed,
            "orders_cancelled": self.orders_cancelled,
            "orders_dropped": self.orders_dropped,
            "malformed_messages": self.malformed_messages,
            "duplicate_orders": self.duplicate_orders,
            "trades_executed": self.trades_executed,
            "volume_executed": self.volume_executed,
            "backpressure_events": self.backpressure_events,
            "elapsed_seconds": self.elapsed_seconds,
            "throughput_orders_per_sec": self.throughput_orders_per_sec,
            "start_time_ns": self.start_time_ns,
            "stop_time_ns": self.stop_time_ns,
        }

    def reset(self) -> None:
        """Reset all metric counters to zero."""
        self.orders_generated = 0
        self.orders_published = 0
        self.orders_consumed = 0
        self.orders_processed = 0
        self.orders_cancelled = 0
        self.orders_dropped = 0
        self.malformed_messages = 0
        self.duplicate_orders = 0
        self.trades_executed = 0
        self.volume_executed = 0
        self.backpressure_events = 0
        self.start_time_ns = None
        self.stop_time_ns = None
        self.wall_start_time_ns = None
        self.wall_stop_time_ns = None


# ==============================================================================
# TradingPipeline Orchestrator
# ==============================================================================


class TradingPipeline:
    """End-to-end simulated trading pipeline orchestrator.

    Bridges:
        Simulated Market Feed -> RingBuffer (Shared Memory IPC) -> Matching Engine (LOB)

    Provides deterministic single-step execution, high-throughput asynchronous batch
    streaming, resting order cancellation, buffer saturation control, and resilience
    against malformed records.
    """

    def __init__(
        self,
        config: MarketFeedConfig | None = None,
        *,
        capacity: int = 1024,
        filepath: str | Path | None = None,
        engine: Any = None,
        ring_buffer: RingBuffer | None = None,
        wal_path: str | Path | None = None,
        wal: WriteAheadLog | None = None,
        trade_store: TradeStore | None = None,
        durability_policy: DurabilityPolicy = DurabilityPolicy.FLUSH_ON_WRITE,
        recover_trades: bool = False,
        strict_recovery: bool = False,
    ) -> None:
        """Initialize the end-to-end trading pipeline.

        Args:
            config: MarketFeedConfig for synthetic order generation.
            capacity: RingBuffer capacity in 64-byte slots (ignored if ring_buffer passed).
            filepath: Optional backing file path for shared memory (None for anonymous).
            engine: Matching engine instance (defaults to Python MatchingEngine).
            ring_buffer: Existing RingBuffer instance to attach to.
            wal_path: Optional file path to append-only Write-Ahead Log (WAL).
            wal: Existing WriteAheadLog instance.
            trade_store: Optional TradeStore instance for trade persistence.
            durability_policy: Durability guarantee for disk writes.
            recover_trades: If True, recovers historical trades into the engine.
            strict_recovery: If True, raises PersistenceError on recovery errors.
        """
        self.config: MarketFeedConfig = config if config is not None else MarketFeedConfig()
        self.metrics: TradingPipelineMetrics = TradingPipelineMetrics()
        self.engine: Any = engine if engine is not None else MatchingEngine()
        self._durability_policy = DurabilityPolicy(durability_policy)

        # Persistence setup
        self._owns_wal: bool = False
        self._wal: WriteAheadLog | None
        if wal is not None:
            self._wal = wal
            self._owns_wal = False
        elif wal_path is not None:
            self._wal = WriteAheadLog(wal_path, durability=self._durability_policy)
            self._owns_wal = True
        else:
            self._wal = None
            self._owns_wal = False

        self._trade_store = trade_store

        self._owns_ring_buffer: bool = ring_buffer is None
        self._ring_buffer: RingBuffer
        if ring_buffer is not None:
            self._ring_buffer = ring_buffer
        else:
            self._ring_buffer = RingBuffer(
                capacity=capacity,
                filepath=filepath,
                create=True,
                record_size=ORDER_STRUCT_SIZE,
            )

        self._producer = OrderProducer(self._ring_buffer)
        self._consumer = OrderConsumer(self._ring_buffer)
        self._simulator = MarketSimulator(self.config)

        self._is_closed: bool = False
        self._stop_event: asyncio.Event = asyncio.Event()

        if recover_trades:
            self.recover_trade_history(strict=strict_recovery)

    # --------------------------------------------------------------------------
    # Deterministic Single-Step & Manual Ingestion
    # --------------------------------------------------------------------------

    def process_order(self, order: Order) -> list[Trade]:
        """Directly transport and process an order through ring buffer to matching engine.

        Pushes order into the ring buffer, immediately consumes it, and matches it
        against the order book.

        Args:
            order: Valid Order instance.

        Returns:
            List of Trade executions generated by the order.

        Raises:
            RingBufferFullError: If ring buffer is full and non-blocking.
            RingBufferClosedError: If ring buffer is closed.
        """
        self._check_closed()
        self._producer.publish_nowait(order)
        self.metrics.orders_published += 1

        consumed_order = self._consumer.get_nowait()
        self.metrics.orders_consumed += 1
        self.metrics.orders_processed += 1

        if self._wal is not None:
            self._wal.append_order(consumed_order)

        trades: list[Trade] = []
        try:
            trades = self.engine.process_order(consumed_order)
            self.metrics.trades_executed += len(trades)
            self.metrics.volume_executed += sum(t.quantity for t in trades)
            if self._wal is not None and trades:
                self._wal.append_trades(trades)
            if self._trade_store is not None and trades:
                self._trade_store.append_trades(trades)
        except DuplicateOrderError:
            self.metrics.duplicate_orders += 1
            raise

        return trades

    def process_next_simulated_order(self) -> tuple[Order, list[Trade]]:
        """Generate the next synthetic order from the simulator and process it end-to-end.

        Returns:
            Tuple of (generated Order, list of Trade executions).
        """
        self._check_closed()
        order = self._simulator.next_order()
        self.metrics.orders_generated += 1
        trades = self.process_order(order)
        return order, trades

    def run_steps(self, count: int) -> list[Trade]:
        """Execute a fixed number of simulated orders sequentially and deterministically.

        Args:
            count: Number of orders to generate and match.

        Returns:
            All Trade executions generated during the run.
        """
        self._check_closed()
        if count <= 0:
            return []

        all_trades: list[Trade] = []
        for _ in range(count):
            _, trades = self.process_next_simulated_order()
            all_trades.extend(trades)

        return all_trades

    # --------------------------------------------------------------------------
    # Order Cancellation Interface
    # --------------------------------------------------------------------------

    def cancel_order(self, order_id: str) -> Order:
        """Cancel and remove a resting order from the matching engine's order book.

        Args:
            order_id: Identifier of the resting order to cancel.

        Returns:
            The removed Order instance.

        Raises:
            OrderNotFoundError: If order_id is not resting in the book.
        """
        self._check_closed()
        if self._wal is not None:
            self._wal.append_cancel(order_id)

        if hasattr(self.engine, "cancel_order"):
            cancelled = self.engine.cancel_order(order_id)
        elif hasattr(self.engine, "order_book") and hasattr(self.engine.order_book, "cancel_order"):
            cancelled = self.engine.order_book.cancel_order(order_id)
        elif hasattr(self.engine, "remove_order"):
            cancelled = self.engine.remove_order(order_id)
        else:
            raise AttributeError("Underlying matching engine does not support cancel_order")

        self.metrics.orders_cancelled += 1
        return cancelled

    # --------------------------------------------------------------------------
    # Ring Buffer Draining & Malformed Message Protection
    # --------------------------------------------------------------------------

    def drain_consumer(self, limit: int | None = None) -> int:
        """Drain available orders from the ring buffer into the matching engine.

        Guards against corrupted/malformed binary records without aborting the loop.

        Args:
            limit: Maximum orders to drain (None for all available).

        Returns:
            Number of orders successfully matched.
        """
        self._check_closed()
        processed = 0

        while not self._consumer.is_empty:
            if limit is not None and processed >= limit:
                break

            try:
                order = self._consumer.get_nowait()
            except (MalformedRecordError, SharedMemoryError, ValueError, TypeError):
                # Malformed binary record detected in buffer slot
                self.metrics.orders_consumed += 1
                self.metrics.malformed_messages += 1
                continue
            except RingBufferEmptyError:
                break

            self.metrics.orders_consumed += 1

            if self._wal is not None:
                self._wal.append_order(order)

            self.metrics.orders_processed += 1
            processed += 1

            try:
                trades = self.engine.process_order(order)
                self.metrics.trades_executed += len(trades)
                self.metrics.volume_executed += sum(t.quantity for t in trades)
                if self._wal is not None and trades:
                    self._wal.append_trades(trades)
                if self._trade_store is not None and trades:
                    self._trade_store.append_trades(trades)
            except DuplicateOrderError:
                self.metrics.duplicate_orders += 1
            except Exception:
                self.metrics.malformed_messages += 1

        return processed

    # --------------------------------------------------------------------------
    # Asynchronous Execution
    # --------------------------------------------------------------------------

    async def run_async(
        self,
        total_orders: int | None = None,
    ) -> TradingPipelineMetrics:
        """Run the end-to-end simulated trading pipeline asynchronously.

        Concurrent producer and consumer coroutines stream orders through the shared
        ring buffer, match orders, handle cancellations, and respect backpressure.

        Args:
            total_orders: Number of orders to generate and process (None uses config).

        Returns:
            Populated TradingPipelineMetrics.
        """
        self._check_closed()
        target_count = total_orders if total_orders is not None else self.config.total_orders
        if target_count is not None and target_count <= 0:
            return self.metrics

        self.metrics.start_time_ns = time.perf_counter_ns()
        self.metrics.wall_start_time_ns = time.time_ns()
        self._stop_event.clear()

        # Set up a dedicated cancellation handler on the simulator
        def on_cancel(order_id: str) -> None:
            try:
                self.cancel_order(order_id)
            except Exception:
                pass

        # Concurrent consumer draining coroutine
        async def _consumer_loop() -> None:
            processed = 0
            while not self._stop_event.is_set() or not self._consumer.is_empty:
                if target_count is not None and processed >= target_count:
                    break

                if not self._consumer.is_empty:
                    try:
                        order = self._consumer.get_nowait()
                    except (MalformedRecordError, SharedMemoryError, ValueError, TypeError):
                        self.metrics.orders_consumed += 1
                        self.metrics.malformed_messages += 1
                        continue
                    except RingBufferEmptyError:
                        await asyncio.sleep(0.0005)
                        continue

                    self.metrics.orders_consumed += 1

                    if self._wal is not None:
                        self._wal.append_order(order)

                    self.metrics.orders_processed += 1
                    processed += 1

                    try:
                        trades = self.engine.process_order(order)
                        self.metrics.trades_executed += len(trades)
                        self.metrics.volume_executed += sum(t.quantity for t in trades)
                        if self._wal is not None and trades:
                            self._wal.append_trades(trades)
                        if self._trade_store is not None and trades:
                            self._trade_store.append_trades(trades)
                    except DuplicateOrderError:
                        self.metrics.duplicate_orders += 1
                else:
                    if self._stop_event.is_set():
                        break
                    await asyncio.sleep(0.0005)

        # Producer coroutine: wraps simulator.run()
        sim_config = MarketFeedConfig(
            symbol=self.config.symbol,
            order_prefix=self.config.order_prefix,
            total_orders=target_count,
            rate=self.config.rate,
            mode=self.config.mode,
            seed=self.config.seed,
            initial_price=self.config.initial_price,
            min_price=self.config.min_price,
            max_price=self.config.max_price,
            price_tick=self.config.price_tick,
            price_volatility=self.config.price_volatility,
            min_quantity=self.config.min_quantity,
            max_quantity=self.config.max_quantity,
            quantity_step=self.config.quantity_step,
            buy_ratio=self.config.buy_ratio,
            spread_ticks=self.config.spread_ticks,
            crossing_probability=self.config.crossing_probability,
            cancel_probability=self.config.cancel_probability,
            burst_probability=self.config.burst_probability,
            burst_size_min=self.config.burst_size_min,
            burst_size_max=self.config.burst_size_max,
            burst_rate_multiplier=self.config.burst_rate_multiplier,
            backpressure_mode=self.config.backpressure_mode,
            backpressure_timeout=self.config.backpressure_timeout,
            backpressure_poll_interval=self.config.backpressure_poll_interval,
        )
        sim = MarketSimulator(sim_config)
        self._simulator = sim

        base_gen = self.metrics.orders_generated
        base_pub = self.metrics.orders_published
        base_drop = self.metrics.orders_dropped
        base_bp = self.metrics.backpressure_events

        def on_order(ord: Order) -> None:
            self.metrics.orders_generated += 1

        def on_publish(ord: Order) -> None:
            self.metrics.orders_published += 1

        async def _producer_loop() -> None:
            try:
                await sim.run(
                    producer=self._producer,
                    cancel_callback=on_cancel,
                    order_callback=on_order,
                    publish_callback=on_publish,
                )
            finally:
                # Synchronize final simulator metrics back into pipeline metrics
                self.metrics.orders_generated = base_gen + sim.metrics.orders_generated
                self.metrics.orders_published = base_pub + sim.metrics.orders_published
                self.metrics.orders_dropped = base_drop + sim.metrics.orders_dropped
                self.metrics.backpressure_events = base_bp + sim.metrics.backpressure_events
                self._stop_event.set()

        try:
            producer_task = asyncio.create_task(_producer_loop())
            consumer_task = asyncio.create_task(_consumer_loop())
            await asyncio.gather(producer_task, consumer_task)
        finally:
            self._stop_event.set()
            self.metrics.wall_stop_time_ns = time.time_ns()
            stop_ns = time.perf_counter_ns()
            if self.metrics.start_time_ns is not None and stop_ns <= self.metrics.start_time_ns:
                stop_ns = self.metrics.start_time_ns + 1
            self.metrics.stop_time_ns = stop_ns

        return self.metrics

    def run_sync(self, total_orders: int | None = None) -> TradingPipelineMetrics:
        """Execute the pipeline synchronously to completion."""
        return asyncio.run(self.run_async(total_orders=total_orders))

    # --------------------------------------------------------------------------
    # Lifecycle & Cleanup
    # --------------------------------------------------------------------------

    def _check_closed(self) -> None:
        if self._is_closed or self._ring_buffer.is_closed:
            raise RingBufferClosedError("TradingPipeline ring buffer is closed")

    def stop(self) -> None:
        """Signal the asynchronous pipeline tasks to stop."""
        self._stop_event.set()

    def close(self) -> None:
        """Close producer, consumer, and ring buffer handles."""
        if self._is_closed:
            return
        self._is_closed = True
        self._stop_event.set()

        try:
            self._producer.close()
        except Exception:
            pass

        try:
            self._consumer.close()
        except Exception:
            pass

        if self._owns_ring_buffer:
            try:
                self._ring_buffer.close()
            except Exception:
                pass

        if self._owns_wal and self._wal is not None:
            try:
                self._wal.close()
            except Exception:
                pass

        if self._trade_store is not None:
            try:
                self._trade_store.close()
            except Exception:
                pass

    def cleanup(self) -> None:
        """Close handles and permanently unlink the backing file if file-backed."""
        self.close()
        if self._owns_ring_buffer and self._ring_buffer.filepath is not None:
            try:
                if self._ring_buffer.filepath.exists():
                    self._ring_buffer.filepath.unlink()
            except Exception:
                pass

    @property
    def ring_buffer(self) -> RingBuffer:
        """Underlying RingBuffer instance."""
        return self._ring_buffer

    @property
    def producer(self) -> OrderProducer:
        """OrderProducer interface."""
        return self._producer

    @property
    def consumer(self) -> OrderConsumer:
        """OrderConsumer interface."""
        return self._consumer

    @property
    def simulator(self) -> MarketSimulator:
        """MarketSimulator instance."""
        return self._simulator

    @property
    def wal(self) -> WriteAheadLog | None:
        """Active WriteAheadLog instance, or None if persistence is disabled."""
        return self._wal

    @property
    def trade_store(self) -> TradeStore | None:
        """Active TradeStore instance, or None if not configured."""
        return self._trade_store

    @property
    def durability_policy(self) -> DurabilityPolicy:
        """Active DurabilityPolicy for persistence."""
        return self._durability_policy

    @property
    def trades(self) -> list[Trade]:
        """All Trade executions produced by the matching engine."""
        return list(self.engine.trades)

    @property
    def total_trades(self) -> int:
        """Total executed trade count."""
        return self.engine.total_trades

    @property
    def total_volume(self) -> int:
        """Total volume executed across all trades."""
        return self.engine.total_volume_traded

    @property
    def best_bid(self) -> float | int | Decimal | None:
        """Current best bid in matching engine's order book."""
        return self.engine.best_bid

    @property
    def best_ask(self) -> float | int | Decimal | None:
        """Current best ask in matching engine's order book."""
        return self.engine.best_ask

    @property
    def spread(self) -> float | int | Decimal | None:
        """Current spread in matching engine's order book."""
        return self.engine.spread

    @property
    def is_cython_accelerated(self) -> bool:
        """True only if compiled native Cython extension binary is actually loaded and active."""
        return is_cython_compiled() and isinstance(self.engine, CythonMatchingEngine)

    @property
    def is_closed(self) -> bool:
        """True if the pipeline has been closed."""
        return self._is_closed or self._ring_buffer.is_closed

    def recover_trade_history(self, *, strict: bool = False) -> list[Trade]:
        """Recover persisted trade history into the pipeline's engine without altering the order book.

        Args:
            strict: If True, raises PersistenceError on truncated or corrupted records.

        Returns:
            List of recovered Trade instances in historical execution order.
        """
        self._check_closed()
        if hasattr(self.engine, "recover_trade_history"):
            return self.engine.recover_trade_history(strict=strict)

        recovered_trades: list[Trade] = []
        if self._trade_store is not None:
            self._trade_store.recover_trade_history(strict=strict)
            recovered_trades = self._trade_store.get_all_trades()
        elif self._wal is not None:
            records, _, _ = self._wal.read_records(strict=strict)
            for rec in records:
                if rec.is_trade and isinstance(rec.data, Trade):
                    recovered_trades.append(rec.data)

        if hasattr(self.engine, "load_trade_history"):
            self.engine.load_trade_history(recovered_trades)

        return list(self.trades)

    @classmethod
    def recover(
        cls,
        wal_path: str | Path,
        config: MarketFeedConfig | None = None,
        *,
        trade_store_path: str | Path | None = None,
        capacity: int = 1024,
        filepath: str | Path | None = None,
        strict: bool = False,
        durability_policy: DurabilityPolicy = DurabilityPolicy.FLUSH_ON_WRITE,
    ) -> TradingPipeline:
        """Recover engine state from a Write-Ahead Log and initialize a ready pipeline.

        Args:
            wal_path: Path to the WAL log file.
            config: MarketFeedConfig for synthetic order generation.
            trade_store_path: Optional path to trade store.
            capacity: RingBuffer capacity.
            filepath: Shared memory backing filepath.
            strict: Strict recovery mode.
            durability_policy: Durability policy for post-recovery operations.

        Returns:
            Initialized TradingPipeline with reconstructed engine state and attached WAL.
        """
        recovery_mgr = RecoveryManager(wal_path=wal_path, trade_store_path=trade_store_path, strict=strict)
        result = recovery_mgr.recover()
        wal = WriteAheadLog(wal_path, durability=durability_policy)
        trade_store = (
            TradeStore(trade_store_path, durability=durability_policy)
            if trade_store_path is not None
            else None
        )
        return cls(
            config=config,
            capacity=capacity,
            filepath=filepath,
            engine=result.engine,
            wal=wal,
            trade_store=trade_store,
            durability_policy=durability_policy,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cleanup()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cleanup()


# ==============================================================================
# Factory Helper
# ==============================================================================


def create_pipeline(
    config: MarketFeedConfig | None = None,
    *,
    capacity: int = 1024,
    filepath: str | Path | None = None,
    engine: Any = None,
    wal_path: str | Path | None = None,
    wal: WriteAheadLog | None = None,
    trade_store: TradeStore | None = None,
    durability_policy: DurabilityPolicy = DurabilityPolicy.FLUSH_ON_WRITE,
) -> TradingPipeline:
    """Convenience factory function to instantiate a TradingPipeline."""
    return TradingPipeline(
        config=config,
        capacity=capacity,
        filepath=filepath,
        engine=engine,
        wal_path=wal_path,
        wal=wal,
        trade_store=trade_store,
        durability_policy=durability_policy,
    )
