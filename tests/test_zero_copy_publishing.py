"""Comprehensive tests for zero-copy direct-to-buffer order publishing.

Validates:
1. Binary protocol layout (64-byte aligned) and round-trip compatibility.
2. RingBuffer.put_order_fields / put_order_fields_nowait direct serialization.
3. OrderProducer.publish_order_fields / publish_order_fields_nowait.
4. Input validation and boundary enforcement (null bytes, oversized ID, types, ranges).
5. Buffer full, empty, and backpressure conditions (WAIT, DROP, RAISE).
6. Concurrent producer-consumer synchronization and FIFO sequencing.
7. End-to-end integration with MarketSimulator and TradingPipeline.
"""

import asyncio
import time
from decimal import Decimal
from pathlib import Path

import pytest

from chronosmatch.binary_protocol import (
    ORDER_STRUCT_SIZE,
    PRICE_SCALE,
    PRICE_TYPE_DECIMAL,
    PRICE_TYPE_FLOAT,
    PRICE_TYPE_INT,
    SIDE_BUY_CODE,
    SIDE_SELL_CODE,
    decode_order_from,
    encode_order,
)
from chronosmatch.market_simulator import (
    BackpressureMode,
    BackpressureTimeoutError,
    MarketFeedConfig,
    MarketSimulator,
    SimulationMode,
)
from chronosmatch.models import Order, OrderSide
from chronosmatch.pipeline import TradingPipeline
from chronosmatch.producer_consumer import OrderConsumer, OrderProducer
from chronosmatch.ring_buffer import (
    RingBuffer,
    RingBufferClosedError,
    RingBufferEmptyError,
    RingBufferFullError,
)


class TestZeroCopyRingBufferDirectPublishing:
    """Test direct-to-buffer serialization in RingBuffer."""

    def test_direct_put_order_fields_roundtrip(self) -> None:
        """Verify fields written via put_order_fields decode identically to standard Order."""
        rb = RingBuffer(capacity=16, create=True)
        ts = 1_700_000_000_000_000_000
        order_id = "ORD_ZC_001"
        price_val = 100.50
        price_ticks = int(round(price_val * PRICE_SCALE))
        quantity = 50
        side_code = SIDE_BUY_CODE
        seq_num = 42

        rb.put_order_fields(
            timestamp=ts,
            raw_id=order_id,
            price_ticks=price_ticks,
            quantity=quantity,
            side_code=side_code,
            price_type=PRICE_TYPE_FLOAT,
            has_seq=1,
            flags=0,
            seq_num=seq_num,
        )

        assert rb.occupied == 1
        order = rb.get()
        assert isinstance(order, Order)
        assert order.order_id == order_id
        assert order.side == OrderSide.BUY
        assert order.price == 100.50
        assert order.quantity == 50
        assert order.timestamp == ts
        assert order.sequence_number == 42
        rb.close()

    def test_direct_put_with_bytes_id_and_enum_side(self) -> None:
        """Verify put_order_fields accepts bytes raw_id and OrderSide enum."""
        rb = RingBuffer(capacity=8, create=True)
        price_ticks = int(round(250 * PRICE_SCALE))
        rb.put_order_fields(
            timestamp=123456789,
            raw_id=b"RAW_BYTES_01",
            price_ticks=price_ticks,
            quantity=10,
            side_code=OrderSide.SELL,
            price_type=PRICE_TYPE_INT,
            has_seq=1,
            seq_num=999,
        )

        order = rb.get()
        assert order.order_id == "RAW_BYTES_01"
        assert order.side == OrderSide.SELL
        assert order.price == 250
        assert order.quantity == 10
        assert order.sequence_number == 999
        rb.close()

    def test_direct_put_binary_layout_matches_encode_order(self) -> None:
        """Verify the exact 64 bytes in shared memory match encode_order()."""
        rb = RingBuffer(capacity=4, create=True)
        ts = 987654321
        order_id = "EXACT_LAYOUT"
        price_val = 123.45
        price_ticks = int(round(price_val * PRICE_SCALE))
        qty = 75
        seq = 101

        expected_order = Order(
            order_id=order_id,
            side=OrderSide.BUY,
            price=price_val,
            quantity=qty,
            timestamp=ts,
            sequence_number=seq,
        )
        expected_bytes = encode_order(expected_order)
        assert len(expected_bytes) == ORDER_STRUCT_SIZE == 64

        rb.put_order_fields_nowait(
            timestamp=ts,
            raw_id=order_id,
            price_ticks=price_ticks,
            quantity=qty,
            side_code=SIDE_BUY_CODE,
            price_type=PRICE_TYPE_FLOAT,
            has_seq=1,
            seq_num=seq,
        )

        # Inspect raw mmap slot (SLOTS_OFFSET = 192)
        from chronosmatch.ring_buffer import SLOTS_OFFSET
        slot_bytes = bytes(rb._region.mmap[SLOTS_OFFSET : SLOTS_OFFSET + 64])
        assert slot_bytes == expected_bytes
        rb.close()

    def test_direct_put_full_buffer_non_blocking_raises(self) -> None:
        """Verify put_order_fields_nowait raises RingBufferFullError when full."""
        rb = RingBuffer(capacity=2, create=True)
        rb.put_order_fields_nowait(1, "O1", 1000, 1, 1)
        rb.put_order_fields_nowait(2, "O2", 2000, 1, 1)
        assert rb.is_full

        with pytest.raises(RingBufferFullError):
            rb.put_order_fields_nowait(3, "O3", 3000, 1, 1)

        with pytest.raises(RingBufferFullError):
            rb.put_order_fields(3, "O3", 3000, 1, 1, block=False)
        rb.close()

    def test_direct_put_validation_errors(self) -> None:
        """Verify strict validation rejection on invalid arguments."""
        rb = RingBuffer(capacity=4, create=True)

        # Invalid timestamp
        with pytest.raises(ValueError):
            rb.put_order_fields(-1, "O1", 1000, 1, 1)
        with pytest.raises(TypeError):
            rb.put_order_fields("not_int", "O1", 1000, 1, 1)  # type: ignore

        # Invalid order ID
        with pytest.raises(ValueError):
            rb.put_order_fields(1, "", 1000, 1, 1)
        with pytest.raises(ValueError):
            rb.put_order_fields(1, "A" * 33, 1000, 1, 1)  # Exceeds 32 bytes
        with pytest.raises(ValueError):
            rb.put_order_fields(1, b"NULL\x00BYTE", 1000, 1, 1)

        # Invalid price ticks
        with pytest.raises(ValueError):
            rb.put_order_fields(1, "O1", 0, 1, 1)
        with pytest.raises(ValueError):
            rb.put_order_fields(1, "O1", -500, 1, 1)

        # Invalid quantity
        with pytest.raises(ValueError):
            rb.put_order_fields(1, "O1", 1000, 0, 1)
        with pytest.raises(ValueError):
            rb.put_order_fields(1, "O1", 1000, -10, 1)

        # Invalid side
        with pytest.raises(ValueError):
            rb.put_order_fields(1, "O1", 1000, 1, 99)

        rb.close()

    def test_direct_put_on_closed_buffer_raises(self) -> None:
        """Verify writing to closed buffer raises RingBufferClosedError."""
        rb = RingBuffer(capacity=4, create=True)
        rb.close()
        with pytest.raises(RingBufferClosedError):
            rb.put_order_fields(1, "O1", 1000, 1, 1)
        with pytest.raises(RingBufferClosedError):
            rb.put_order_fields_nowait(1, "O1", 1000, 1, 1)


class TestZeroCopyProducerConsumer:
    """Test OrderProducer and OrderConsumer with direct field publishing."""

    def test_producer_publish_fields_api(self) -> None:
        """Verify OrderProducer.publish_order_fields and publish_fields alias."""
        rb = RingBuffer(capacity=8, create=True)
        producer = OrderProducer(rb)
        consumer = OrderConsumer(rb)

        producer.publish_order_fields(
            timestamp=1001,
            raw_id="P_ORD_1",
            price_ticks=int(round(500.0 * PRICE_SCALE)),
            quantity=20,
            side_code=SIDE_BUY_CODE,
        )
        producer.publish_fields_nowait(
            timestamp=1002,
            raw_id="P_ORD_2",
            price_ticks=int(round(510.0 * PRICE_SCALE)),
            quantity=30,
            side_code=SIDE_SELL_CODE,
        )

        o1 = consumer.get()
        o2 = consumer.get()
        assert o1.order_id == "P_ORD_1" and o1.price == 500.0 and o1.quantity == 20
        assert o2.order_id == "P_ORD_2" and o2.price == 510.0 and o2.quantity == 30

        producer.close()
        consumer.close()


class TestZeroCopyMarketSimulatorPublishing:
    """Test MarketSimulator zero-copy publishing and backpressure."""

    @pytest.mark.asyncio
    async def test_simulator_publish_order_fields_wait_backpressure(self) -> None:
        """Verify simulator publish_order_fields handles WAIT mode correctly."""
        rb = RingBuffer(capacity=4, create=True)
        producer = OrderProducer(rb)
        consumer = OrderConsumer(rb)

        config = MarketFeedConfig(
            seed=42,
            total_orders=10,
            mode=SimulationMode.MAX_THROUGHPUT,
            backpressure_mode=BackpressureMode.WAIT,
            backpressure_poll_interval=0.001,
        )
        sim = MarketSimulator(config)

        consumed: list[Order] = []

        async def drain() -> None:
            await asyncio.sleep(0.01)
            while len(consumed) < 10:
                if not consumer.is_empty:
                    consumed.append(consumer.get_nowait())
                else:
                    await asyncio.sleep(0.001)

        prod_task = asyncio.create_task(sim.run(producer=producer))
        drain_task = asyncio.create_task(drain())

        metrics, _ = await asyncio.gather(prod_task, drain_task)
        assert metrics.orders_published == 10
        assert len(consumed) == 10
        assert [o.sequence_number for o in consumed] == list(range(1, 11))
        assert metrics.backpressure_events > 0

        producer.close()
        consumer.close()

    @pytest.mark.asyncio
    async def test_simulator_publish_order_fields_drop_backpressure(self) -> None:
        """Verify simulator publish_order_fields drops orders when buffer is full."""
        rb = RingBuffer(capacity=4, create=True)
        producer = OrderProducer(rb)

        config = MarketFeedConfig(
            seed=42,
            total_orders=10,
            mode=SimulationMode.MAX_THROUGHPUT,
            backpressure_mode=BackpressureMode.DROP,
            cancel_probability=0.0,
        )
        sim = MarketSimulator(config)

        metrics = await sim.run(producer=producer)
        assert metrics.orders_published == 4
        assert metrics.orders_dropped == 6
        assert rb.occupied == 4
        producer.close()

    @pytest.mark.asyncio
    async def test_simulator_publish_order_fields_raise_backpressure(self) -> None:
        """Verify simulator publish_order_fields raises when buffer is full."""
        rb = RingBuffer(capacity=3, create=True)
        producer = OrderProducer(rb)

        config = MarketFeedConfig(
            seed=42,
            total_orders=8,
            mode=SimulationMode.MAX_THROUGHPUT,
            backpressure_mode=BackpressureMode.RAISE,
            cancel_probability=0.0,
        )
        sim = MarketSimulator(config)

        with pytest.raises(RingBufferFullError):
            await sim.run(producer=producer)

        assert sim.metrics.orders_published == 3
        producer.close()


class TestZeroCopyPipelineEndToEnd:
    """Test full asynchronous trading pipeline with zero-copy publishing."""

    @pytest.mark.asyncio
    async def test_pipeline_async_run_zero_copy(self) -> None:
        """Verify pipeline.run_async executes to completion with correct trade reconciliation."""
        config = MarketFeedConfig(
            symbol="BTC-USD",
            mode=SimulationMode.MAX_THROUGHPUT,
            total_orders=100,
            crossing_probability=0.3,
            cancel_probability=0.05,
            seed=42,
        )
        pipeline = TradingPipeline(config=config, capacity=64)

        metrics = await pipeline.run_async(total_orders=100)

        assert metrics.orders_generated >= 100
        assert metrics.orders_published >= 100
        assert metrics.orders_processed >= 100
        assert metrics.trades_executed > 0
        assert metrics.volume_executed > 0
        assert metrics.duplicate_orders == 0
        assert metrics.malformed_messages == 0

        pipeline.close()
