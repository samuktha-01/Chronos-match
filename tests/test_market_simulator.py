"""Comprehensive test suite for ChronosMatch Asynchronous Synthetic Market Feed Simulator.

Tests:
1. Configuration & Validation (default parameters, custom configs, strict boundary checks).
2. Order Validity & Schema Compliance (all generated orders pass Order model validation).
3. Reproducibility & Determinism (identical seeds produce bit-identical sequences).
4. Timing & Arrival Distributions (Uniform, Poisson exponential, Burst, Max Throughput).
5. Market Dynamics (Bid/ask spread, passive vs aggressive crossing orders, buy ratio).
6. Order Cancellation (FIFO resting pool tracking, cancel event generation, engine cancellation).
7. Async Execution, Cancellation & Graceful Shutdown (async generator, task cancel, stop()).
8. Producer Integration & Backpressure (WAIT, DROP, RAISE modes, bounded buffer draining).
9. End-to-End Pipeline & Synchronous Bridge (simulator streaming through ring buffer to matching engine).
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
import math
import time
from typing import Any

import pytest

from chronosmatch.binary_protocol import decode_order, encode_order
from chronosmatch.market_simulator import (
    BackpressureMode,
    BackpressureTimeoutError,
    MarketFeedConfig,
    MarketSimulator,
    MarketSimulatorError,
    OrderAction,
    OrderCancel,
    SimulationMode,
    SimulatorConfigError,
    SimulatorMetrics,
    create_simulator,
    run_simulator_sync,
)
from chronosmatch.matching_engine import MatchingEngine
from chronosmatch.models import Order, OrderSide
from chronosmatch.order_book import LimitOrderBook
from chronosmatch.producer_consumer import OrderConsumer, OrderProducer
from chronosmatch.ring_buffer import (
    RingBuffer,
    RingBufferClosedError,
    RingBufferFullError,
)


def async_run(async_fn: Any) -> Any:
    """Decorator to execute async test functions with standard asyncio.run without pytest-asyncio."""
    import functools

    @functools.wraps(async_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(async_fn(*args, **kwargs))

    return wrapper


# ==============================================================================
# 1. Configuration & Validation Tests
# ==============================================================================


class TestMarketFeedConfig:
    """Test suite for MarketFeedConfig validation and boundary conditions."""

    def test_default_config_is_valid(self) -> None:
        """Verify default configuration initializes with standard parameters."""
        config = MarketFeedConfig()

        assert config.symbol == "CHRONOS"
        assert config.order_prefix == "SIM"
        assert config.total_orders is None
        assert config.rate == 1000.0
        assert config.mode == SimulationMode.POISSON
        assert config.seed is None
        assert config.initial_price == 100.0
        assert config.min_price == 1.0
        assert config.max_price == 10000.0
        assert config.price_tick == 0.01
        assert config.min_quantity == 1
        assert config.max_quantity == 100
        assert config.buy_ratio == 0.5
        assert config.spread_ticks == 2
        assert config.crossing_probability == 0.1
        assert config.cancel_probability == 0.05
        assert config.backpressure_mode == BackpressureMode.WAIT

    def test_string_enums_coerced(self) -> None:
        """Verify string inputs for mode and backpressure_mode are properly coerced."""
        config = MarketFeedConfig(mode="uniform", backpressure_mode="drop")
        assert config.mode == SimulationMode.UNIFORM
        assert config.backpressure_mode == BackpressureMode.DROP

        config2 = MarketFeedConfig(mode="BURST", backpressure_mode="RAISE")
        assert config2.mode == SimulationMode.BURST
        assert config2.backpressure_mode == BackpressureMode.RAISE

    @pytest.mark.parametrize("invalid_symbol", ["", "   ", None, 123])
    def test_invalid_symbol_raises(self, invalid_symbol: Any) -> None:
        """Verify invalid symbols raise SimulatorConfigError."""
        with pytest.raises(SimulatorConfigError):
            MarketFeedConfig(symbol=invalid_symbol)

    @pytest.mark.parametrize(
        "invalid_prefix",
        ["", "   ", "TOOLONGPREFIXEXCEEDING20CHARS", "PREFIX WITH SPACE", "BAD\x00PRE"],
    )
    def test_invalid_order_prefix_raises(self, invalid_prefix: str) -> None:
        """Verify invalid order prefixes raise SimulatorConfigError."""
        with pytest.raises(SimulatorConfigError):
            MarketFeedConfig(order_prefix=invalid_prefix)

    @pytest.mark.parametrize("invalid_rate", [0, -10.0, float("nan"), float("inf"), "100", True])
    def test_invalid_rate_raises(self, invalid_rate: Any) -> None:
        """Verify non-positive or non-finite rates raise SimulatorConfigError."""
        with pytest.raises(SimulatorConfigError):
            MarketFeedConfig(rate=invalid_rate)

    @pytest.mark.parametrize("invalid_total", [0, -5, "100", 3.14, True])
    def test_invalid_total_orders_raises(self, invalid_total: Any) -> None:
        """Verify invalid total_orders values raise SimulatorConfigError."""
        with pytest.raises(SimulatorConfigError):
            MarketFeedConfig(total_orders=invalid_total)

    @pytest.mark.parametrize("invalid_seed", ["seed123", 4.5, True])
    def test_invalid_seed_raises(self, invalid_seed: Any) -> None:
        """Verify non-integer seeds raise SimulatorConfigError."""
        with pytest.raises(SimulatorConfigError):
            MarketFeedConfig(seed=invalid_seed)

    def test_invalid_price_relationships_raise(self) -> None:
        """Verify invalid price bounds raise SimulatorConfigError."""
        with pytest.raises(SimulatorConfigError):
            # min > max
            MarketFeedConfig(min_price=200.0, max_price=100.0)

        with pytest.raises(SimulatorConfigError):
            # initial < min
            MarketFeedConfig(initial_price=50.0, min_price=100.0, max_price=200.0)

        with pytest.raises(SimulatorConfigError):
            # initial > max
            MarketFeedConfig(initial_price=250.0, min_price=100.0, max_price=200.0)

        with pytest.raises(SimulatorConfigError):
            # price_tick > range
            MarketFeedConfig(min_price=10.0, max_price=12.0, price_tick=5.0)

        with pytest.raises(SimulatorConfigError):
            # price <= 0
            MarketFeedConfig(min_price=-1.0)

    def test_invalid_quantity_relationships_raise(self) -> None:
        """Verify invalid quantity bounds raise SimulatorConfigError."""
        with pytest.raises(SimulatorConfigError):
            # min > max
            MarketFeedConfig(min_quantity=50, max_quantity=20)

        with pytest.raises(SimulatorConfigError):
            # quantity <= 0
            MarketFeedConfig(min_quantity=0)

        with pytest.raises(SimulatorConfigError):
            # quantity_step > range
            MarketFeedConfig(min_quantity=10, max_quantity=20, quantity_step=25)

    @pytest.mark.parametrize(
        ("param", "val"),
        [
            ("buy_ratio", -0.1),
            ("buy_ratio", 1.1),
            ("crossing_probability", -0.05),
            ("crossing_probability", 1.5),
            ("cancel_probability", -0.1),
            ("cancel_probability", 2.0),
            ("burst_probability", -0.01),
            ("burst_probability", 1.01),
        ],
    )
    def test_invalid_probabilities_raise(self, param: str, val: float) -> None:
        """Verify probabilities outside [0.0, 1.0] raise SimulatorConfigError."""
        kwargs = {param: val}
        with pytest.raises(SimulatorConfigError):
            MarketFeedConfig(**kwargs)

    def test_invalid_burst_config_raises(self) -> None:
        """Verify invalid burst parameters raise SimulatorConfigError."""
        with pytest.raises(SimulatorConfigError):
            # burst_min > burst_max
            MarketFeedConfig(burst_size_min=50, burst_size_max=10)

        with pytest.raises(SimulatorConfigError):
            # burst_multiplier <= 0
            MarketFeedConfig(burst_rate_multiplier=0)

    def test_invalid_backpressure_config_raises(self) -> None:
        """Verify invalid backpressure parameters raise SimulatorConfigError."""
        with pytest.raises(SimulatorConfigError):
            MarketFeedConfig(backpressure_mode="INVALID_MODE")

        with pytest.raises(SimulatorConfigError):
            MarketFeedConfig(backpressure_timeout=-1.0)

        with pytest.raises(SimulatorConfigError):
            MarketFeedConfig(backpressure_poll_interval=0.0)


# ==============================================================================
# 2. Order Validity & Schema Compliance Tests
# ==============================================================================


class TestOrderValidityAndSchema:
    """Test suite verifying all generated orders strictly adhere to Order domain rules."""

    def test_generated_orders_satisfy_all_invariants(self) -> None:
        """Verify 500 consecutive generated orders pass strict Order validation."""
        sim = MarketSimulator(
            MarketFeedConfig(
                seed=42,
                min_price=10.0,
                max_price=200.0,
                price_tick=0.05,
                min_quantity=5,
                max_quantity=500,
                quantity_step=5,
            )
        )

        for _ in range(500):
            order = sim.next_order()

            # Dataclass validation already executed in __post_init__
            assert isinstance(order, Order)
            assert isinstance(order.order_id, str)
            assert order.order_id.startswith("SIM_")
            assert len(order.order_id.encode("utf-8")) <= 32
            assert order.side in (OrderSide.BUY, OrderSide.SELL)
            assert 10.0 <= order.price <= 200.0
            assert 5 <= order.quantity <= 500
            assert order.quantity % 5 == 0  # Lot size step verified
            assert order.timestamp > 0
            assert order.sequence_number is not None
            assert order.sequence_number >= 1

    def test_binary_protocol_serialization_roundtrip(self) -> None:
        """Verify all generated orders can be serialized and deserialized to 64-byte binary."""
        sim = MarketSimulator(MarketFeedConfig(seed=123, total_orders=100))

        for _ in range(100):
            order = sim.next_order()

            # Encode to 64 bytes
            data = encode_order(order)
            assert len(data) == 64

            # Decode back
            decoded = decode_order(data)
            assert decoded.order_id == order.order_id
            assert decoded.side == order.side
            assert abs(decoded.price - order.price) < 1e-6
            assert decoded.quantity == order.quantity
            assert decoded.sequence_number == order.sequence_number

    def test_strictly_monotonic_sequence_numbers(self) -> None:
        """Verify sequence numbers strictly increment 1, 2, 3..."""
        sim = MarketSimulator(MarketFeedConfig(seed=999))
        seqs = [sim.next_order().sequence_number for _ in range(50)]
        assert seqs == list(range(1, 51))


# ==============================================================================
# 3. Reproducibility & Determinism Tests
# ==============================================================================


class TestReproducibility:
    """Test suite verifying bit-identical reproducibility when seeded."""

    def test_identical_seeds_produce_identical_order_sequences(self) -> None:
        """Verify two independent simulator instances with identical seed generate identical orders."""
        cfg1 = MarketFeedConfig(seed=2026, total_orders=100)
        cfg2 = MarketFeedConfig(seed=2026, total_orders=100)

        sim1 = MarketSimulator(cfg1)
        sim2 = MarketSimulator(cfg2)

        for _ in range(100):
            o1 = sim1.next_order()
            o2 = sim2.next_order()

            assert o1.order_id == o2.order_id
            assert o1.side == o2.side
            assert o1.price == o2.price
            assert o1.quantity == o2.quantity
            assert o1.sequence_number == o2.sequence_number

    def test_reset_reproduces_identical_sequence(self) -> None:
        """Verify resetting a simulator reproduces the exact same sequence."""
        sim = MarketSimulator(MarketFeedConfig(seed=777))
        batch1 = [sim.next_order() for _ in range(50)]

        sim.reset()
        batch2 = [sim.next_order() for _ in range(50)]

        for o1, o2 in zip(batch1, batch2, strict=True):
            assert o1.order_id == o2.order_id
            assert o1.side == o2.side
            assert o1.price == o2.price
            assert o1.quantity == o2.quantity

    def test_different_seeds_produce_different_sequences(self) -> None:
        """Verify distinct seeds produce divergent order streams."""
        sim1 = MarketSimulator(MarketFeedConfig(seed=100))
        sim2 = MarketSimulator(MarketFeedConfig(seed=200))

        orders1 = [sim1.next_order() for _ in range(30)]
        orders2 = [sim2.next_order() for _ in range(30)]

        # Verify that prices or quantities diverge
        differences = sum(
            1 for o1, o2 in zip(orders1, orders2, strict=True) if o1.price != o2.price or o1.quantity != o2.quantity
        )
        assert differences > 10


# ==============================================================================
# 4. Timing & Arrival Distributions Tests
# ==============================================================================


class TestTimingDistributions:
    """Test suite verifying inter-arrival timing for all simulation modes."""

    def test_max_throughput_mode_has_zero_delay(self) -> None:
        """Verify MAX_THROUGHPUT returns 0.0 delay."""
        sim = MarketSimulator(MarketFeedConfig(mode=SimulationMode.MAX_THROUGHPUT))
        for _ in range(20):
            assert sim.next_inter_arrival_time() == 0.0

    def test_uniform_mode_has_constant_delay(self) -> None:
        """Verify UNIFORM returns exact 1.0 / rate delay."""
        rate = 250.0
        sim = MarketSimulator(MarketFeedConfig(mode=SimulationMode.UNIFORM, rate=rate))
        expected_delay = 1.0 / rate

        for _ in range(30):
            assert abs(sim.next_inter_arrival_time() - expected_delay) < 1e-9

    def test_poisson_mode_empirical_mean_matches_lambda(self) -> None:
        """Verify Poisson mode exponential inter-arrival times converge to 1 / lambda."""
        rate = 500.0  # lambda = 500 orders/sec
        expected_mean = 1.0 / rate  # 0.002s
        sim = MarketSimulator(MarketFeedConfig(mode=SimulationMode.POISSON, rate=rate, seed=12345))

        sample_size = 5000
        delays = [sim.next_inter_arrival_time() for _ in range(sample_size)]

        # All delays must be non-negative
        assert all(d >= 0.0 for d in delays)

        sample_mean = sum(delays) / sample_size
        # Mean of Exp(lambda) should be within 5% of expected 1/lambda for N=5000
        relative_error = abs(sample_mean - expected_mean) / expected_mean
        assert relative_error < 0.05

    def test_burst_mode_alternates_burst_and_steady_state(self) -> None:
        """Verify BURST mode generates fast clusters followed by inter-burst delays."""
        sim = MarketSimulator(
            MarketFeedConfig(
                mode=SimulationMode.BURST,
                rate=100.0,
                burst_probability=0.2,
                burst_size_min=10,
                burst_size_max=10,
                burst_rate_multiplier=50.0,
                seed=42,
            )
        )

        delays = [sim.next_inter_arrival_time() for _ in range(200)]
        assert any(d < 0.0005 for d in delays)  # Fast burst delays observed
        assert any(d > 0.002 for d in delays)  # Normal steady-state delays observed


# ==============================================================================
# 5. Market Dynamics & Spread Simulation Tests
# ==============================================================================


class TestMarketDynamics:
    """Test suite verifying spread formation, passive vs crossing orders, and matching."""

    def test_passive_bid_and_ask_spread_relationship(self) -> None:
        """Verify passive orders respect reference spread boundaries."""
        config = MarketFeedConfig(
            seed=42,
            initial_price=100.0,
            spread_ticks=4,
            price_tick=0.1,
            crossing_probability=0.0,  # 100% passive orders
            price_volatility=0.0001,
        )
        sim = MarketSimulator(config)

        bids: list[float] = []
        asks: list[float] = []

        for _ in range(100):
            order = sim.next_order()
            if order.is_buy:
                bids.append(order.price)
            else:
                asks.append(order.price)

        # Passive bids should generally be below 100.0; passive asks should be above 100.0
        assert len(bids) > 0 and len(asks) > 0
        assert max(bids) <= 100.0
        assert min(asks) >= 100.0

    def test_crossing_probability_extremes(self) -> None:
        """Verify 0.0 and 1.0 crossing probabilities behave deterministically."""
        # 100% crossing
        sim_cross = MarketSimulator(MarketFeedConfig(seed=1, crossing_probability=1.0))
        for _ in range(30):
            sim_cross.next_order()
        assert sim_cross.metrics.crossing_orders == 30
        assert sim_cross.metrics.passive_orders == 0

        # 0% crossing
        sim_pass = MarketSimulator(MarketFeedConfig(seed=1, crossing_probability=0.0))
        for _ in range(30):
            sim_pass.next_order()
        assert sim_pass.metrics.crossing_orders == 0
        assert sim_pass.metrics.passive_orders == 30

    def test_buy_ratio_extremes(self) -> None:
        """Verify buy_ratio=1.0 generates only BUYs, and 0.0 generates only SELLs."""
        sim_buys = MarketSimulator(MarketFeedConfig(seed=1, buy_ratio=1.0))
        assert all(sim_buys.next_order().is_buy for _ in range(25))

        sim_sells = MarketSimulator(MarketFeedConfig(seed=1, buy_ratio=0.0))
        assert all(sim_sells.next_order().is_sell for _ in range(25))

    def test_crossing_orders_execute_trades_in_matching_engine(self) -> None:
        """Verify feeding simulated orders with crossing probability executes trades in MatchingEngine."""
        config = MarketFeedConfig(
            seed=42,
            initial_price=100.0,
            price_tick=0.5,
            spread_ticks=2,
            crossing_probability=0.3,
            total_orders=100,
        )
        sim = MarketSimulator(config)
        engine = MatchingEngine()

        for _ in range(100):
            order = sim.next_order()
            engine.process_order(order)

        # Trades must have executed due to aggressive crossing orders
        assert engine.total_trades > 0
        assert engine.total_volume_traded > 0


# ==============================================================================
# 6. Cancellation Simulation Tests
# ==============================================================================


class TestCancellationSimulation:
    """Test suite verifying order cancellation mechanics."""

    def test_next_cancel_returns_none_when_pool_empty(self) -> None:
        """Verify next_cancel returns None when no orders have been generated."""
        sim = MarketSimulator(MarketFeedConfig())
        assert sim.next_cancel() is None

    def test_next_cancel_returns_valid_resting_order_id(self) -> None:
        """Verify next_cancel extracts resting order IDs in FIFO sequence."""
        sim = MarketSimulator(MarketFeedConfig(seed=42, crossing_probability=0.0))

        # Generate 5 passive orders
        orders = [sim.next_order() for _ in range(5)]
        assert sim.resting_order_count == 5

        # Cancel orders one by one
        c1 = sim.next_cancel()
        assert c1 is not None
        assert c1.order_id == orders[0].order_id
        assert sim.resting_order_count == 4

        c2 = sim.next_cancel()
        assert c2 is not None
        assert c2.order_id == orders[1].order_id
        assert sim.resting_order_count == 3

    def test_cancellation_execution_on_matching_engine(self) -> None:
        """Verify cancellation successfully removes an order from the engine's book."""
        engine = MatchingEngine()
        sim = MarketSimulator(
            MarketFeedConfig(
                seed=42,
                initial_price=100.0,
                crossing_probability=0.0,
                cancel_probability=0.2,
                total_orders=30,
            )
        )

        cancelled_ids: list[str] = []

        def on_cancel(order_id: str) -> None:
            cancelled_ids.append(order_id)

        # Run synchronously
        metrics = asyncio.run(sim.run(target_engine=engine, cancel_callback=on_cancel))

        assert metrics.orders_cancelled > 0
        assert len(cancelled_ids) == metrics.orders_cancelled

        # Cancelled orders should no longer reside in the engine's order book
        for cid in cancelled_ids:
            assert cid not in engine.order_book

    @async_run
    async def test_generate_events_yields_both_orders_and_cancels(self) -> None:
        """Verify generate_events async generator yields both Order and OrderCancel."""
        config = MarketFeedConfig(
            seed=42,
            total_orders=30,
            cancel_probability=0.3,
            mode=SimulationMode.MAX_THROUGHPUT,
        )
        sim = MarketSimulator(config)

        events: list[Order | OrderCancel] = []
        async for event in sim.generate_events():
            events.append(event)

        orders = [e for e in events if isinstance(e, Order)]
        cancels = [e for e in events if isinstance(e, OrderCancel)]

        assert len(orders) == 30
        assert len(cancels) > 0


# ==============================================================================
# 7. Async Execution & Graceful Shutdown Tests
# ==============================================================================


class TestAsyncExecutionAndShutdown:
    """Test suite verifying asyncio execution, stop signaling, and cancellation."""

    @async_run
    async def test_generate_orders_terminates_at_total_orders(self) -> None:
        """Verify generate_orders stops once total_orders is reached."""
        sim = MarketSimulator(
            MarketFeedConfig(total_orders=25, mode=SimulationMode.MAX_THROUGHPUT)
        )

        orders = [order async for order in sim.generate_orders()]
        assert len(orders) == 25
        assert sim.metrics.orders_generated == 25

    @async_run
    async def test_stop_halts_generation_early(self) -> None:
        """Verify calling stop() halts continuous order generation gracefully."""
        sim = MarketSimulator(MarketFeedConfig(total_orders=10000, rate=500.0))

        async def _consumer() -> list[Order]:
            collected: list[Order] = []
            async for order in sim.generate_orders():
                collected.append(order)
                if len(collected) >= 15:
                    sim.stop()
            return collected

        collected = await _consumer()
        assert len(collected) == 15
        assert sim.is_stopped is True

    @async_run
    async def test_asyncio_task_cancellation_handled_cleanly(self) -> None:
        """Verify cancelling an asyncio task running sim.run() raises CancelledError and preserves state."""
        sim = MarketSimulator(MarketFeedConfig(total_orders=10000, rate=100.0))

        task = asyncio.create_task(sim.run())
        await asyncio.sleep(0.02)  # Allow a few orders to generate
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert sim.is_running is False
        assert sim.is_stopped is True
        assert sim.metrics.orders_generated > 0

    @async_run
    async def test_async_context_manager(self) -> None:
        """Verify async context manager auto-stops simulator on block exit."""
        async with MarketSimulator(MarketFeedConfig(rate=1000.0)) as sim:
            assert sim.is_stopped is False
        assert sim.is_stopped is True


# ==============================================================================
# 8. Producer Integration & Backpressure Tests
# ==============================================================================


class TestProducerIntegrationAndBackpressure:
    """Test suite verifying integration with OrderProducer and backpressure handling."""

    @async_run
    async def test_simulator_publishes_to_order_producer(self) -> None:
        """Verify MarketSimulator publishes orders directly into an in-memory RingBuffer via OrderProducer."""
        rb = RingBuffer(capacity=64, create=True)
        producer = OrderProducer(rb)
        consumer = OrderConsumer(rb)

        config = MarketFeedConfig(
            seed=42,
            total_orders=40,
            mode=SimulationMode.MAX_THROUGHPUT,
            crossing_probability=0.0,
            cancel_probability=0.0,
        )
        sim = MarketSimulator(config)

        metrics = await sim.run(producer=producer)

        assert metrics.orders_published == 40
        assert rb.occupied == 40

        # Drain and verify FIFO integrity
        drained: list[Order] = []
        for _ in range(40):
            drained.append(consumer.get())

        assert len(drained) == 40
        assert [o.sequence_number for o in drained] == list(range(1, 41))

        producer.close()
        consumer.close()

    @async_run
    async def test_backpressure_wait_mode_with_concurrent_consumer(self) -> None:
        """Verify WAIT mode handles a small buffer (capacity=8) with 40 orders when drained concurrently."""
        rb = RingBuffer(capacity=8, create=True)
        producer = OrderProducer(rb)
        consumer = OrderConsumer(rb)

        config = MarketFeedConfig(
            seed=42,
            total_orders=40,
            mode=SimulationMode.MAX_THROUGHPUT,
            backpressure_mode=BackpressureMode.WAIT,
            backpressure_poll_interval=0.001,
            cancel_probability=0.0,
        )
        sim = MarketSimulator(config)

        consumed_orders: list[Order] = []

        async def _drain_worker() -> None:
            # Let producer fill buffer to ensure backpressure wait occurs
            await asyncio.sleep(0.05)
            while len(consumed_orders) < 40:
                if not consumer.is_empty:
                    order = consumer.get_nowait()
                    consumed_orders.append(order)
                else:
                    await asyncio.sleep(0.001)

        producer_task = asyncio.create_task(sim.run(producer=producer))
        drain_task = asyncio.create_task(_drain_worker())

        metrics, _ = await asyncio.gather(producer_task, drain_task)

        assert metrics.orders_published == 40
        assert len(consumed_orders) == 40
        assert metrics.backpressure_events > 0  # Buffer filled and waited
        assert [o.sequence_number for o in consumed_orders] == list(range(1, 41))

        producer.close()
        consumer.close()

    @async_run
    async def test_backpressure_wait_timeout_raises(self) -> None:
        """Verify backpressure wait raises BackpressureTimeoutError if buffer remains full."""
        rb = RingBuffer(capacity=4, create=True)
        producer = OrderProducer(rb)

        config = MarketFeedConfig(
            seed=42,
            total_orders=10,
            mode=SimulationMode.MAX_THROUGHPUT,
            backpressure_mode=BackpressureMode.WAIT,
            backpressure_timeout=0.03,  # Fast timeout
            backpressure_poll_interval=0.002,
        )
        sim = MarketSimulator(config)

        with pytest.raises(BackpressureTimeoutError):
            await sim.run(producer=producer)

        assert sim.metrics.orders_published == 4  # Exactly filled capacity
        assert sim.metrics.backpressure_events > 0

        producer.close()

    @async_run
    async def test_backpressure_drop_mode(self) -> None:
        """Verify DROP mode drops excess orders without blocking or raising."""
        rb = RingBuffer(capacity=5, create=True)
        producer = OrderProducer(rb)

        config = MarketFeedConfig(
            seed=42,
            total_orders=15,
            mode=SimulationMode.MAX_THROUGHPUT,
            backpressure_mode=BackpressureMode.DROP,
            cancel_probability=0.0,
        )
        sim = MarketSimulator(config)

        metrics = await sim.run(producer=producer)

        assert metrics.orders_published == 5
        assert metrics.orders_dropped == 10
        assert rb.occupied == 5

        producer.close()

    @async_run
    async def test_backpressure_raise_mode(self) -> None:
        """Verify RAISE mode immediately raises RingBufferFullError when full."""
        rb = RingBuffer(capacity=3, create=True)
        producer = OrderProducer(rb)

        config = MarketFeedConfig(
            seed=42,
            total_orders=10,
            mode=SimulationMode.MAX_THROUGHPUT,
            backpressure_mode=BackpressureMode.RAISE,
            cancel_probability=0.0,
        )
        sim = MarketSimulator(config)

        with pytest.raises(RingBufferFullError):
            await sim.run(producer=producer)

        assert sim.metrics.orders_published == 3
        producer.close()

    @async_run
    async def test_publish_to_closed_producer_raises(self) -> None:
        """Verify publishing to a closed producer raises RingBufferClosedError."""
        rb = RingBuffer(capacity=8, create=True)
        producer = OrderProducer(rb)
        producer.close()

        sim = MarketSimulator()
        order = sim.next_order()

        with pytest.raises(RingBufferClosedError):
            await sim.publish_order(order, producer)


# ==============================================================================
# 9. Pipeline & Synchronous Bridge Tests
# ==============================================================================


class TestPipelineAndSyncBridge:
    """Test suite verifying end-to-end integration and synchronous execution bridge."""

    def test_run_simulator_sync_executes_to_completion(self) -> None:
        """Verify run_simulator_sync executes synchronously and returns accurate metrics."""
        config = MarketFeedConfig(
            seed=42,
            total_orders=50,
            mode=SimulationMode.MAX_THROUGHPUT,
        )

        metrics = run_simulator_sync(config)

        assert metrics.orders_generated >= 50
        assert metrics.elapsed_seconds > 0.0
        assert metrics.effective_rate >= 0.0
        metrics_dict = metrics.to_dict()
        assert "orders_generated" in metrics_dict
        assert "effective_rate" in metrics_dict

    def test_create_simulator_factory_helper(self) -> None:
        """Verify create_simulator factory function initializes configured instance."""
        sim = create_simulator(rate=750.0, total_orders=100, mode="UNIFORM", seed=99)
        assert isinstance(sim, MarketSimulator)
        assert sim.config.rate == 750.0
        assert sim.config.total_orders == 100
        assert sim.config.mode == SimulationMode.UNIFORM
        assert sim.config.seed == 99

    @async_run
    async def test_end_to_end_streaming_to_matching_engine(self) -> None:
        """Verify end-to-end flow: Simulator -> Producer -> RingBuffer -> Consumer -> MatchingEngine."""
        rb = RingBuffer(capacity=128, create=True)
        producer = OrderProducer(rb)
        consumer = OrderConsumer(rb)
        engine = MatchingEngine()

        config = MarketFeedConfig(
            seed=42,
            total_orders=80,
            crossing_probability=0.25,
            cancel_probability=0.0,
            mode=SimulationMode.MAX_THROUGHPUT,
        )
        sim = MarketSimulator(config)

        # 1. Run simulator to populate ring buffer
        metrics = await sim.run(producer=producer)
        assert metrics.orders_published == 80
        assert rb.occupied == 80

        # 2. Drain consumer into matching engine
        processed_count = 0
        while not consumer.is_empty:
            order = consumer.get_nowait()
            engine.process_order(order)
            processed_count += 1

        assert processed_count == 80
        assert rb.occupied == 0
        assert engine.total_trades > 0  # Crossing orders triggered trades
        assert engine.best_bid is not None or engine.best_ask is not None

        producer.close()
        consumer.close()
