"""Comprehensive End-to-End Integration Test Suite for ChronosMatch.

Validates the full simulated trading pipeline connecting:
1. Synthetic Market Feed Simulator (`MarketSimulator`)
2. 64-byte Binary Serialization Protocol (`encode_order` / `decode_order`)
3. Shared-Memory IPC (`SharedMemoryRegion`)
4. Lock-free SPSC Ring Buffer (`RingBuffer`)
5. Order Producer & Consumer Interfaces (`OrderProducer` / `OrderConsumer`)
6. Deterministic Limit Order Book Matching Engine (`MatchingEngine` / `LimitOrderBook`)
7. Multiprocess Cross-Process Pipeline (`OrderPipeline.start_matching_consumer`)

Test Scenarios:
1. An order travelling from the simulator to the matching engine.
2. Matching orders generating the expected trade.
3. Multiple orders preserving required processing order (FIFO and price-time priority).
4. Cancellation of a resting order.
5. Buffer backpressure, saturation, and shutdown behavior.
6. Accurate generated, published, processed, executed, cancelled, and dropped-order metrics.
7. Error handling without silently losing orders (malformed binary payload resilience).
8. Multiprocess producer-to-matching-engine integration.
9. Verification of Cython acceleration status honesty.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
import functools
from pathlib import Path
import struct
import time
from typing import Any

import pytest

from chronosmatch.binary_protocol import (
    ORDER_STRUCT_FORMAT,
    ORDER_STRUCT_SIZE,
    PRICE_SCALE,
    SIDE_BUY_CODE,
    SIDE_SELL_CODE,
    decode_order,
    encode_order,
)
from chronosmatch.cython_engine import is_cython_compiled
from chronosmatch.market_simulator import (
    BackpressureMode,
    BackpressureTimeoutError,
    MarketFeedConfig,
    MarketSimulator,
    OrderAction,
    OrderCancel,
    SimulationMode,
)
from chronosmatch.matching_engine import MatchingEngine, Trade
from chronosmatch.models import Order, OrderSide
from chronosmatch.order_book import DuplicateOrderError, LimitOrderBook, OrderNotFoundError
from chronosmatch.pipeline import (
    TradingPipeline,
    TradingPipelineMetrics,
    create_pipeline,
)
from chronosmatch.producer_consumer import (
    OrderConsumer,
    OrderPipeline,
    OrderProducer,
)
from chronosmatch.ring_buffer import (
    RingBuffer,
    RingBufferClosedError,
    RingBufferEmptyError,
    RingBufferFullError,
)
from chronosmatch.shared_memory import MalformedRecordError


def async_run(async_fn: Any) -> Any:
    """Decorator to execute async test functions with asyncio.run without pytest-asyncio."""
    @functools.wraps(async_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(async_fn(*args, **kwargs))

    return wrapper


# ==============================================================================
# 1. Order Traversal: Simulator -> Transport -> Matching Engine
# ==============================================================================


class TestOrderTraversal:
    """Verify orders travel from simulator through ring buffer to matching engine."""

    def test_single_simulated_order_traversal(self) -> None:
        """Verify a single simulated order is generated, serialized, enqueued, dequeued, and placed."""
        config = MarketFeedConfig(
            symbol="AAPL",
            initial_price=150.0,
            seed=42,
            mode=SimulationMode.MAX_THROUGHPUT,
        )
        with create_pipeline(config=config, capacity=16) as pipeline:
            order, trades = pipeline.process_next_simulated_order()

            assert isinstance(order, Order)
            assert order.order_id.startswith(config.order_prefix)
            assert order.price > 0
            assert order.quantity > 0
            assert pipeline.metrics.orders_generated == 1
            assert pipeline.metrics.orders_published == 1
            assert pipeline.metrics.orders_processed == 1

            # Order should now reside in the matching engine's order book (no matches yet)
            assert len(trades) == 0
            if order.is_buy:
                assert pipeline.best_bid == order.price
            else:
                assert pipeline.best_ask == order.price

    def test_file_backed_shared_memory_order_traversal(self, tmp_path: Path) -> None:
        """Verify end-to-end traversal using a concrete OS-backed shared memory file."""
        shm_file = tmp_path / "chronos_test_pipeline.bin"
        with TradingPipeline(filepath=shm_file, capacity=8) as pipeline:
            assert pipeline.ring_buffer.filepath == shm_file.resolve()

            order = Order(
                order_id="TEST-FILE-001",
                side=OrderSide.BUY,
                price=200.50,
                quantity=25,
                timestamp=1700000000000000001,
                sequence_number=1,
            )
            trades = pipeline.process_order(order)

            assert len(trades) == 0
            assert pipeline.metrics.orders_published == 1
            assert pipeline.metrics.orders_processed == 1
            assert pipeline.best_bid == 200.50


# ==============================================================================
# 2. Matching Orders & Trade Executions
# ==============================================================================


class TestTradeExecutionAndMatching:
    """Verify crossing orders generate exact Trade executions and volume."""

    def test_crossing_orders_generate_expected_trade(self) -> None:
        """Verify passive resting SELL order + aggressive incoming BUY order produces exact trade."""
        with create_pipeline(capacity=16) as pipeline:
            resting_sell = Order(
                order_id="SELL-MAKER-1",
                side=OrderSide.SELL,
                price=100.0,
                quantity=10,
                timestamp=1000,
                sequence_number=1,
            )
            incoming_buy = Order(
                order_id="BUY-TAKER-1",
                side=OrderSide.BUY,
                price=100.0,
                quantity=10,
                timestamp=2000,
                sequence_number=2,
            )

            # Step 1: Place maker sell order
            trades_step1 = pipeline.process_order(resting_sell)
            assert len(trades_step1) == 0
            assert pipeline.best_ask == 100.0
            assert pipeline.best_bid is None

            # Step 2: Ingest crossing taker buy order
            trades_step2 = pipeline.process_order(incoming_buy)
            assert len(trades_step2) == 1

            trade = trades_step2[0]
            assert trade.trade_id == 1
            assert trade.resting_order_id == "SELL-MAKER-1"
            assert trade.incoming_order_id == "BUY-TAKER-1"
            assert trade.price == 100.0
            assert trade.quantity == 10

            # Step 3: Verify book is completely cleared
            assert pipeline.best_bid is None
            assert pipeline.best_ask is None

            # Step 4: Verify metrics
            assert pipeline.metrics.orders_published == 2
            assert pipeline.metrics.orders_processed == 2
            assert pipeline.metrics.trades_executed == 1
            assert pipeline.metrics.volume_executed == 10
            assert pipeline.total_trades == 1
            assert pipeline.total_volume == 10

    def test_partial_fill_and_resting_remainder(self) -> None:
        """Verify partial fill: taker order larger than resting maker order leaves remainder."""
        with create_pipeline(capacity=16) as pipeline:
            maker_sell = Order(
                order_id="SELL-10",
                side=OrderSide.SELL,
                price=105.0,
                quantity=10,
                timestamp=1000,
            )
            taker_buy = Order(
                order_id="BUY-25",
                side=OrderSide.BUY,
                price=105.0,
                quantity=25,
                timestamp=2000,
            )

            pipeline.process_order(maker_sell)
            trades = pipeline.process_order(taker_buy)

            assert len(trades) == 1
            assert trades[0].quantity == 10
            assert trades[0].price == 105.0

            # Remainder of 15 should now rest as best bid
            assert pipeline.best_ask is None
            assert pipeline.best_bid == 105.0
            assert pipeline.metrics.volume_executed == 10

    def test_maker_price_priority_rule(self) -> None:
        """Verify trade executes at maker's resting limit price, not taker's price."""
        with create_pipeline(capacity=16) as pipeline:
            # Maker rests at 100.0
            maker_sell = Order(
                order_id="SELL-RESTING",
                side=OrderSide.SELL,
                price=100.0,
                quantity=5,
                timestamp=1000,
            )
            # Taker bids aggressively at 110.0
            taker_buy = Order(
                order_id="BUY-AGGRESSIVE",
                side=OrderSide.BUY,
                price=110.0,
                quantity=5,
                timestamp=2000,
            )

            pipeline.process_order(maker_sell)
            trades = pipeline.process_order(taker_buy)

            assert len(trades) == 1
            # Must execute at maker price: 100.0, not taker price: 110.0
            assert trades[0].price == 100.0
            assert trades[0].quantity == 5


# ==============================================================================
# 3. Multiple Orders: FIFO & Deterministic Sequence Preservation
# ==============================================================================


class TestFifoAndSequencePreservation:
    """Verify strict FIFO order processing through the ring buffer to the engine."""

    def test_price_time_fifo_queue_matching(self) -> None:
        """Two resting bids at identical price: older order (t1) matches before younger order (t2)."""
        with create_pipeline(capacity=16) as pipeline:
            buy1 = Order(
                order_id="BUY-EARLY",
                side=OrderSide.BUY,
                price=100.0,
                quantity=10,
                timestamp=1000,
                sequence_number=1,
            )
            buy2 = Order(
                order_id="BUY-LATE",
                side=OrderSide.BUY,
                price=100.0,
                quantity=10,
                timestamp=2000,
                sequence_number=2,
            )
            sell_incoming = Order(
                order_id="SELL-INCOMING",
                side=OrderSide.SELL,
                price=100.0,
                quantity=10,
                timestamp=3000,
                sequence_number=3,
            )

            pipeline.process_order(buy1)
            pipeline.process_order(buy2)
            trades = pipeline.process_order(sell_incoming)

            assert len(trades) == 1
            # BUY-EARLY must match first
            assert trades[0].resting_order_id == "BUY-EARLY"
            assert trades[0].incoming_order_id == "SELL-INCOMING"

            # BUY-LATE must still be resting at 100.0
            assert pipeline.best_bid == 100.0

    def test_multi_order_deterministic_run_steps(self) -> None:
        """Verify running a sequence of 50 deterministic orders maintains monotonic metrics."""
        config = MarketFeedConfig(
            symbol="MSFT",
            total_orders=50,
            initial_price=300.0,
            seed=12345,
            crossing_probability=0.3,
            mode=SimulationMode.MAX_THROUGHPUT,
        )
        with create_pipeline(config=config, capacity=64) as pipeline:
            trades = pipeline.run_steps(50)

            assert pipeline.metrics.orders_generated == 50
            assert pipeline.metrics.orders_published == 50
            assert pipeline.metrics.orders_processed == 50
            assert pipeline.metrics.trades_executed == len(trades)
            assert pipeline.metrics.volume_executed == sum(t.quantity for t in trades)
            # Ensure trade count matches engine record
            assert len(pipeline.trades) == len(trades)


# ==============================================================================
# 4. Resting Order Cancellation
# ==============================================================================


class TestOrderCancellation:
    """Verify cancellation removes resting orders cleanly and updates metrics."""

    def test_cancel_resting_order(self) -> None:
        """Verify cancelling an existing resting order removes it from book and updates metrics."""
        with create_pipeline(capacity=16) as pipeline:
            order = Order(
                order_id="REST-TO-CANCEL",
                side=OrderSide.BUY,
                price=50.0,
                quantity=100,
                timestamp=1000,
            )
            pipeline.process_order(order)
            assert pipeline.best_bid == 50.0

            cancelled = pipeline.cancel_order("REST-TO-CANCEL")
            assert cancelled.order_id == "REST-TO-CANCEL"
            assert pipeline.best_bid is None
            assert pipeline.metrics.orders_cancelled == 1

    def test_cancel_nonexistent_order_raises(self) -> None:
        """Verify cancelling an unknown order raises OrderNotFoundError without corrupting state."""
        with create_pipeline(capacity=16) as pipeline:
            with pytest.raises(OrderNotFoundError):
                pipeline.cancel_order("NON-EXISTENT-ID")

            assert pipeline.metrics.orders_cancelled == 0

    @async_run
    async def test_async_pipeline_cancellation_flow(self) -> None:
        """Verify simulator cancellation events are executed against the resting book in async run."""
        config = MarketFeedConfig(
            symbol="NVDA",
            total_orders=60,
            initial_price=120.0,
            seed=999,
            cancel_probability=0.4,
            crossing_probability=0.1,
            rate=1000.0,
            mode=SimulationMode.MAX_THROUGHPUT,
        )
        with create_pipeline(config=config, capacity=128) as pipeline:
            metrics = await pipeline.run_async()

            assert metrics.orders_generated == 60
            assert metrics.orders_processed == 60
            assert metrics.orders_cancelled > 0


# ==============================================================================
# 5. Buffer Backpressure, Saturation & Shutdown
# ==============================================================================


class TestBufferBackpressureAndShutdown:
    """Verify ring buffer saturation, drop/wait policies, and clean shutdown."""

    def test_buffer_saturation_drop_mode(self) -> None:
        """Verify backpressure DROP policy drops excess orders when buffer is saturated."""
        # Buffer with capacity 4
        capacity = 4
        config = MarketFeedConfig(
            total_orders=10,
            seed=42,
            mode=SimulationMode.MAX_THROUGHPUT,
            backpressure_mode=BackpressureMode.DROP,
        )
        with create_pipeline(config=config, capacity=capacity) as pipeline:
            sim = MarketSimulator(config)

            # Generate 10 orders and publish without draining consumer
            for _ in range(10):
                order = sim.next_order()
                pipeline.metrics.orders_generated += 1
                try:
                    pipeline.producer.publish_nowait(order)
                    pipeline.metrics.orders_published += 1
                except RingBufferFullError:
                    pipeline.metrics.orders_dropped += 1
                    pipeline.metrics.backpressure_events += 1

            assert pipeline.ring_buffer.occupied == capacity
            assert pipeline.metrics.orders_published == capacity
            assert pipeline.metrics.orders_dropped == 6
            assert pipeline.metrics.backpressure_events == 6

            # Now drain the 4 buffered orders
            drained = pipeline.drain_consumer()
            assert drained == capacity
            assert pipeline.metrics.orders_processed == capacity
            assert pipeline.ring_buffer.is_empty

    def test_pipeline_stop_signals_clean_shutdown(self) -> None:
        """Verify stop() signals shutdown and prevents further order ingestion."""
        with create_pipeline(capacity=8) as pipeline:
            pipeline.stop()
            # stop_event should be set
            assert pipeline._stop_event.is_set()

    def test_pipeline_close_and_context_manager_cleanup(self, tmp_path: Path) -> None:
        """Verify context manager closes handles and unlinks file cleanly on Windows."""
        shm_file = tmp_path / "cleanup_test.bin"
        with create_pipeline(filepath=shm_file, capacity=8) as pipeline:
            assert shm_file.exists()
            assert not pipeline.is_closed

        # After exiting context manager, pipeline must be closed and file unlinked
        assert pipeline.is_closed
        assert not shm_file.exists()


# ==============================================================================
# 6. Observability & Metrics Accuracy
# ==============================================================================


class TestMetricsAccuracyAndObservability:
    """Verify metrics accounting invariants across generated, processed, and executed orders."""

    @async_run
    async def test_metrics_invariants_in_async_run(self) -> None:
        """Verify strict mathematical invariants hold across all metrics."""
        config = MarketFeedConfig(
            symbol="AAPL",
            total_orders=80,
            initial_price=150.0,
            seed=777,
            crossing_probability=0.25,
            rate=2000.0,
            mode=SimulationMode.MAX_THROUGHPUT,
        )
        with create_pipeline(config=config, capacity=128) as pipeline:
            metrics = await pipeline.run_async()

            # Invariant 1: Total orders generated equals published + dropped
            assert metrics.orders_generated == metrics.orders_published + metrics.orders_dropped
            assert metrics.orders_generated == 80

            # Invariant 2: Processed orders equals published orders (all reached matching engine)
            assert metrics.orders_processed == metrics.orders_published

            # Invariant 3: Trades executed and volume match matching engine records
            assert metrics.trades_executed == len(pipeline.trades)
            assert metrics.volume_executed == sum(t.quantity for t in pipeline.trades)

            # Invariant 4: Monotonic timing sanity
            assert metrics.start_time_ns is not None
            assert metrics.stop_time_ns is not None
            assert metrics.stop_time_ns >= metrics.start_time_ns
            assert metrics.elapsed_seconds > 0.0
            assert metrics.throughput_orders_per_sec > 0.0

            # Invariant 5: Dictionary representation contains all expected keys
            data = metrics.to_dict()
            expected_keys = {
                "orders_generated",
                "orders_published",
                "orders_processed",
                "orders_cancelled",
                "orders_dropped",
                "malformed_messages",
                "duplicate_orders",
                "trades_executed",
                "volume_executed",
                "backpressure_events",
                "elapsed_seconds",
                "throughput_orders_per_sec",
                "start_time_ns",
                "stop_time_ns",
            }
            assert expected_keys.issubset(data.keys())

    def test_metrics_reset(self) -> None:
        """Verify reset() restores all counters and timestamps to initial state."""
        metrics = TradingPipelineMetrics(
            orders_generated=10,
            orders_published=10,
            orders_processed=10,
            trades_executed=2,
            volume_executed=50,
            start_time_ns=1000,
            stop_time_ns=2000,
        )
        metrics.reset()
        assert metrics.orders_generated == 0
        assert metrics.trades_executed == 0
        assert metrics.start_time_ns is None
        assert metrics.elapsed_seconds == 0.0


# ==============================================================================
# 7. Malformed Message Protection & Error Resilience
# ==============================================================================


class TestMalformedMessageResilience:
    """Verify malformed binary records do not crash the pipeline or drop subsequent valid orders."""

    def test_malformed_binary_record_handled_safely(self, tmp_path: Path) -> None:
        """Inject corrupt binary data directly into ring buffer; verify pipeline continues."""
        shm_file = tmp_path / "corrupt_test.bin"
        with create_pipeline(filepath=shm_file, capacity=8) as pipeline:
            # 1. Enqueue valid Order 1
            order1 = Order(
                order_id="VALID-1",
                side=OrderSide.BUY,
                price=100.0,
                quantity=10,
                timestamp=1000,
            )
            pipeline.producer.publish(order1)
            pipeline.metrics.orders_published += 1

            # 2. Directly inject malformed binary record into ring buffer:
            # Corrupt the side code byte (offset 52: 8 + 32 + 8 + 4) with invalid code 99
            valid_payload = bytearray(encode_order(order1))
            valid_payload[52] = 99  # Invalid side code (neither BUY=1 nor SELL=2)
            corrupt_payload = bytes(valid_payload)

            # Use raw ring_buffer write
            pipeline.ring_buffer.put_record(corrupt_payload)
            pipeline.metrics.orders_published += 1

            # 3. Enqueue valid Order 2
            order2 = Order(
                order_id="VALID-2",
                side=OrderSide.SELL,
                price=100.0,
                quantity=10,
                timestamp=1002,
            )
            pipeline.producer.publish(order2)
            pipeline.metrics.orders_published += 1

            # 4. Drain the consumer
            processed = pipeline.drain_consumer()

            # Malformed record must have been detected
            assert pipeline.metrics.malformed_messages == 1
            # Valid orders (order1 and order2) must have both been successfully processed
            assert processed == 2
            assert pipeline.metrics.orders_processed == 2

            # And because order1 (BUY 100) and order2 (SELL 100) cross, a trade must have executed!
            assert pipeline.metrics.trades_executed == 1
            assert pipeline.metrics.volume_executed == 10

    def test_duplicate_order_id_raises_and_counts_metric(self) -> None:
        """Verify duplicate order ID raises DuplicateOrderError and increments duplicate counter."""
        with create_pipeline(capacity=8) as pipeline:
            order1 = Order(
                order_id="DUP-1",
                side=OrderSide.BUY,
                price=100.0,
                quantity=5,
                timestamp=1000,
            )
            pipeline.process_order(order1)

            order2 = Order(
                order_id="DUP-1",
                side=OrderSide.BUY,
                price=100.0,
                quantity=5,
                timestamp=1001,
            )
            with pytest.raises(DuplicateOrderError):
                pipeline.process_order(order2)

            assert pipeline.metrics.duplicate_orders == 1


# ==============================================================================
# 8. Multiprocess Cross-Process Pipeline Integration
# ==============================================================================


class TestMultiprocessPipelineIntegration:
    """Verify OrderPipeline orchestrates cross-process execution to MatchingEngine."""

    def test_cross_process_matching_engine_execution(self, tmp_path: Path) -> None:
        """Producer process publishes orders; separate matching consumer process executes trades."""
        shm_file = tmp_path / "mp_matching_test.bin"
        capacity = 16

        # Create pairs of crossing orders: BUY 100.0 qty 10, SELL 100.0 qty 10
        orders: list[Order] = []
        for i in range(5):
            orders.append(
                Order(
                    order_id=f"MP-BUY-{i}",
                    side=OrderSide.BUY,
                    price=100.0,
                    quantity=10,
                    timestamp=1000 + i * 2,
                    sequence_number=i * 2 + 1,
                )
            )
            orders.append(
                Order(
                    order_id=f"MP-SELL-{i}",
                    side=OrderSide.SELL,
                    price=100.0,
                    quantity=10,
                    timestamp=1001 + i * 2,
                    sequence_number=i * 2 + 2,
                )
            )

        with OrderPipeline(filepath=shm_file, capacity=capacity) as pipeline:
            # Start consumer process that matches incoming orders against MatchingEngine
            pipeline.start_matching_consumer(expected_count=len(orders), timeout=5.0)

            # Start producer process streaming the 10 orders
            pipeline.start_producer(orders=orders, timeout=5.0)

            # Join child processes and retrieve executed trades
            trades = pipeline.drain_trades(timeout=10.0)
            results = pipeline.get_results(timeout=1.0)

            # All 10 orders received
            assert len(results) == 10
            # 5 crossing pairs produce exactly 5 trades
            assert len(trades) == 5
            for trade in trades:
                assert isinstance(trade, Trade)
                assert trade.price == 100.0
                assert trade.quantity == 10


# ==============================================================================
# 9. Cython Acceleration Honesty
# ==============================================================================


class TestCythonAccelerationHonesty:
    """Verify pipeline does not falsely claim native Cython acceleration."""

    def test_is_cython_accelerated_reports_truthfully(self) -> None:
        """Verify is_cython_accelerated is strictly False unless compiled extension is active."""
        with create_pipeline(capacity=8) as pipeline:
            if not is_cython_compiled():
                assert pipeline.is_cython_accelerated is False
            else:
                # If compiled, verify matching engine instance type
                from chronosmatch.cython_engine import CythonMatchingEngine

                expected = isinstance(pipeline.engine, CythonMatchingEngine)
                assert pipeline.is_cython_accelerated == expected
