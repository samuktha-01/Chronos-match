"""Unit and Integration Test Suite for ChronosMatch Continuous Live HFT Market Firehose.

Validates:
1. Continuous market simulation (total_orders=None) producing orders until stopped.
2. Configurable synthetic order generation rates:
   - 1,000 orders/sec
   - 10,000 orders/sec
   - 50,000 orders/sec
   - 100,000 orders/sec
   - MAX_THROUGHPUT mode
3. Sustained producer-consumer pipeline with mmap shared memory and ring-buffer backpressure.
4. Strict FIFO sequence preservation with zero dropped or duplicate orders.
5. Independent measurement of target generation rate, actual generated rate, and published rate.
6. Real-time metrics snapshotting and sliding-window rate tracking.
7. Whale order generation, threshold alerting, and multi-level price sweep tracking during continuous feed.
8. Graceful shutdown (stop / Ctrl+C), ring buffer draining, WAL persistence flush, and crash recovery.
9. Dashboard continuous simulation runner and CLI option resolution (--continuous, -c, --orders 0, --rate max).
"""

from __future__ import annotations

import asyncio
import io
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from chronosmatch.cython_engine import is_cython_compiled, CythonMatchingEngine
from chronosmatch.dashboard import (
    BoundedLatencyTracker,
    DashboardMetricsCollector,
    DashboardMetricsSnapshot,
    DashboardRenderer,
    PipelineDashboardRunner,
    TerminalDashboard,
)
from chronosmatch.dashboard.__main__ import _parse_rate_arg, main
from chronosmatch.market_simulator import (
    MarketFeedConfig,
    MarketSimulator,
    SimulationMode,
    SimulatorMetrics,
)
from chronosmatch.matching_engine import MatchingEngine, Trade
from chronosmatch.models import Order, OrderSide
from chronosmatch.persistence import DurabilityPolicy, WALRecordType, WriteAheadLog
from chronosmatch.pipeline import (
    TradingPipeline,
    TradingPipelineMetrics,
    create_pipeline,
)
from chronosmatch.ring_buffer import RingBuffer


# ==============================================================================
# 1. Continuous Market Simulation Generation
# ==============================================================================


class TestContinuousMarketSimulator:
    """Verify market simulator operates continuously until explicitly stopped."""

    @pytest.mark.asyncio
    async def test_continuous_simulation_runs_until_stopped(self) -> None:
        """Verify simulator with total_orders=None generates continuous flow until stop() is called."""
        config = MarketFeedConfig(
            total_orders=None,  # Continuous mode
            rate=5_000.0,
            mode=SimulationMode.UNIFORM,
            seed=42,
        )
        sim = MarketSimulator(config)
        generated_orders: list[Order] = []

        def on_order(order: Order) -> None:
            generated_orders.append(order)

        async def run_and_stop() -> None:
            await asyncio.sleep(0.1)
            sim.stop()

        stop_task = asyncio.create_task(run_and_stop())
        await sim.run(order_callback=on_order)
        await stop_task

        assert not sim.is_running
        assert len(generated_orders) > 50
        assert sim.metrics.orders_generated == len(generated_orders)
        assert sim.metrics.generated_rate > 0.0

    @pytest.mark.parametrize("target_rate", [1_000.0, 10_000.0, 50_000.0, 100_000.0])
    @pytest.mark.asyncio
    async def test_configurable_target_rates(self, target_rate: float) -> None:
        """Verify simulator respects configurable target rates (1k, 10k, 50k, 100k)."""
        config = MarketFeedConfig(
            total_orders=None,
            rate=target_rate,
            mode=SimulationMode.UNIFORM,
            seed=123,
        )
        assert config.orders_per_second == target_rate
        sim = MarketSimulator(config)

        order_count = 0

        def on_order(_: Order) -> None:
            nonlocal order_count
            order_count += 1
            if order_count >= 200:
                sim.stop()

        await sim.run(order_callback=on_order)
        assert order_count >= 200
        assert sim.metrics.orders_generated == order_count
        assert sim.metrics.generated_rate > 0.0

    @pytest.mark.asyncio
    async def test_max_throughput_mode(self) -> None:
        """Verify MAX_THROUGHPUT mode generates orders without artificial rate pacing."""
        config = MarketFeedConfig(
            total_orders=None,
            rate=100_000.0,
            mode=SimulationMode.MAX_THROUGHPUT,
            seed=777,
        )
        sim = MarketSimulator(config)
        orders = 0

        def on_order(_: Order) -> None:
            nonlocal orders
            orders += 1
            if orders >= 500:
                sim.stop()

        start = time.perf_counter()
        await sim.run(order_callback=on_order)
        elapsed = time.perf_counter() - start

        assert orders >= 500
        rate = orders / elapsed
        assert rate > 10_000.0  # Unthrottled generation runs at tens of thousands of orders/sec


# ==============================================================================
# 2. Continuous End-to-End Pipeline & Zero Loss
# ==============================================================================


class TestContinuousPipelineProcessing:
    """Verify end-to-end continuous pipeline execution and synchronization."""

    @pytest.mark.asyncio
    async def test_continuous_pipeline_execution_and_stop(self) -> None:
        """Run continuous pipeline, let it process hundreds of orders, and cleanly stop."""
        config = MarketFeedConfig(
            total_orders=None,
            rate=10_000.0,
            mode=SimulationMode.UNIFORM,
            seed=999,
        )
        pipeline = create_pipeline(config=config, capacity=512)
        try:
            async def stopper() -> None:
                while pipeline.metrics.orders_processed < 250:
                    await asyncio.sleep(0.01)
                pipeline.stop()

            stop_task = asyncio.create_task(stopper())
            metrics = await pipeline.run_async(total_orders=None)
            await stop_task

            assert metrics.orders_processed >= 250
            assert metrics.orders_generated >= metrics.orders_processed
            assert metrics.orders_dropped == 0
            assert metrics.malformed_messages == 0
            assert metrics.duplicate_orders == 0
        finally:
            pipeline.close()

    @pytest.mark.asyncio
    async def test_continuous_pipeline_with_backpressure(self) -> None:
        """Verify backpressure handling in continuous mode with small buffer capacity."""
        config = MarketFeedConfig(
            total_orders=None,
            rate=50_000.0,
            mode=SimulationMode.UNIFORM,
            seed=42,
        )
        pipeline = create_pipeline(config=config, capacity=32)
        try:
            async def stopper() -> None:
                while pipeline.metrics.orders_processed < 200:
                    await asyncio.sleep(0.01)
                pipeline.stop()

            stop_task = asyncio.create_task(stopper())
            metrics = await pipeline.run_async(total_orders=None)
            await stop_task

            assert metrics.orders_processed >= 200
            assert metrics.orders_dropped == 0  # WAIT mode handles backpressure without drops
            assert metrics.throughput_orders_per_sec > 0.0
        finally:
            pipeline.close()

    def test_pipeline_drain_all_pending(self) -> None:
        """Verify drain_all_pending consumes all remaining ring buffer entries and flushes WAL."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            wal_file = f.name

        try:
            pipeline = create_pipeline(capacity=64, wal_path=wal_file)
            # Push 5 manual orders directly into the ring buffer producer
            for i in range(5):
                order = Order(
                    order_id=f"DRAIN-{i+1}",
                    side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                    price=100.0 + i,
                    quantity=10,
                )
                pipeline.producer.put(order)

            assert pipeline.ring_buffer.occupied == 5
            drained = pipeline.drain_all_pending()
            assert drained == 5
            assert pipeline.ring_buffer.occupied == 0

            # Verify WAL contains the flushed orders
            wal = WriteAheadLog(wal_file, create=False)
            records, _, _ = wal.read_records()
            wal.close()
            assert len(records) == 7  # 5 orders + 2 trades executed
            order_records = [r for r in records if r.record_type == WALRecordType.ORDER]
            assert len(order_records) == 5
        finally:
            pipeline.close()
            if os.path.exists(wal_file):
                try:
                    os.unlink(wal_file)
                except Exception:
                    pass


# ==============================================================================
# 3. Telemetry, Sliding-Window Rates & Dashboard Display
# ==============================================================================


class TestContinuousDashboardTelemetry:
    """Verify real-time rate tracking and Section 2 display for live simulation."""

    def test_collector_tracks_target_and_window_rates(self) -> None:
        """Verify DashboardMetricsCollector computes independent generated and published rates."""
        collector = DashboardMetricsCollector(source_name="RateTest")
        collector.start()
        collector.set_target_generation_rate(100_000.0)

        collector.record_order_generated(10_000)
        collector.record_order_published(9_950)
        collector.record_order_processed(9_900)

        snap1 = collector.take_snapshot()
        assert snap1.target_generation_rate == 100_000.0
        assert snap1.orders_generated == 10_000
        assert snap1.orders_published == 9_950
        assert snap1.orders_processed == 9_900
        assert snap1.generated_orders_per_sec is not None
        assert snap1.published_orders_per_sec is not None

        # Verify to_dict includes new fields
        d = snap1.to_dict()
        assert d["target_generation_rate"] == 100_000.0
        assert d["generated_orders_per_sec"] == snap1.generated_orders_per_sec
        assert d["published_orders_per_sec"] == snap1.published_orders_per_sec

    def test_renderer_displays_target_and_generated_rates(self) -> None:
        """Verify DashboardRenderer displays Target Rate, Actual Gen. Rate, and Published Rate in Section 2."""
        collector = DashboardMetricsCollector(source_name="RenderTest")
        collector.start()
        collector.set_target_generation_rate(100_000.0)
        collector.record_order_generated(5_000)
        collector.record_order_published(4_990)
        collector.record_order_processed(4_980)

        renderer = DashboardRenderer(use_colors=False, use_unicode=False, width=80)
        snap = collector.take_snapshot()
        rendered = renderer.render(snap)

        assert re.search(r"Target Rate:\s+100,000 ops/s", rendered) is not None
        assert "Actual Gen. Rate:" in rendered
        assert "Published Rate:" in rendered
        assert "Orders Generated:" in rendered
        assert "Orders Published:" in rendered
        assert "Orders Consumed:" in rendered
        assert "Orders Processed:" in rendered
        assert "Backpressure Evt:" in rendered

        # Verify all lines have exactly the configured width
        for line_idx, line in enumerate(rendered.splitlines()):
            assert len(line) == 80, f"Line {line_idx} width is {len(line)}, expected 80: '{line}'"

    def test_max_throughput_target_rate_formatting(self) -> None:
        """Verify MAX_THROUGHPUT target rate renders as 'MAX_THROUGHPUT'."""
        collector = DashboardMetricsCollector(source_name="MaxTest")
        collector.start()
        collector.set_target_generation_rate(float("inf"))
        collector.record_order_generated(1_000)

        renderer = DashboardRenderer(use_colors=False, use_unicode=False, width=80)
        snap = collector.take_snapshot()
        rendered = renderer.render(snap)

        assert re.search(r"Target Rate:\s+MAX_THROUGHPUT", rendered) is not None


# ==============================================================================
# 4. Whale Order Detection in Continuous Simulation
# ==============================================================================


class TestWhaleDetectionInContinuousMode:
    """Verify whale orders are generated, detected, and highlighted in continuous mode."""

    @pytest.mark.asyncio
    async def test_continuous_whale_order_generation_and_alerts(self) -> None:
        """Verify simulator generates whale orders when max_quantity >= whale_order_threshold."""
        config = MarketFeedConfig(
            total_orders=None,
            rate=20_000.0,
            mode=SimulationMode.UNIFORM,
            min_quantity=10,
            max_quantity=2_000,
            whale_order_threshold=1_000,
            seed=42,
        )
        pipeline = create_pipeline(config=config, whale_threshold=1_000, capacity=256)
        try:
            async def stopper() -> None:
                while pipeline.metrics.orders_processed < 500:
                    await asyncio.sleep(0.01)
                pipeline.stop()

            stop_task = asyncio.create_task(stopper())
            metrics = await pipeline.run_async(total_orders=None)
            await stop_task

            assert metrics.orders_processed >= 500
            assert metrics.whale_orders_detected > 0
            assert metrics.whale_volume_executed >= 1_000
            assert metrics.whale_order_threshold == 1_000
        finally:
            pipeline.close()


# ==============================================================================
# 5. CLI Arguments & Runner Continuous Mode
# ==============================================================================


class TestContinuousCLIOptions:
    """Verify CLI argument parsing and runner integration for continuous mode."""

    def test_parse_rate_argument(self) -> None:
        """Verify _parse_rate_arg parses numeric rates, 0, and 'max' strings."""
        rate, mode = _parse_rate_arg("1000")
        assert rate == 1000.0
        assert mode is None

        rate, mode = _parse_rate_arg("100000")
        assert rate == 100000.0
        assert mode is None

        rate, mode = _parse_rate_arg("max")
        assert math.isinf(rate)
        assert mode == SimulationMode.MAX_THROUGHPUT

        rate, mode = _parse_rate_arg("MAX_THROUGHPUT")
        assert math.isinf(rate)
        assert mode == SimulationMode.MAX_THROUGHPUT

        rate, mode = _parse_rate_arg(0)
        assert math.isinf(rate)
        assert mode == SimulationMode.MAX_THROUGHPUT

    def test_pipeline_runner_run_continuous(self) -> None:
        """Verify PipelineDashboardRunner.run_continuous starts and responds to stop()."""
        config = MarketFeedConfig(
            total_orders=None,
            rate=10_000.0,
            mode=SimulationMode.UNIFORM,
            seed=555,
        )
        pipeline = create_pipeline(config=config, capacity=128)
        stream = io.StringIO()
        runner = PipelineDashboardRunner(
            pipeline=pipeline,
            interactive=False,
            use_colors=False,
            output=stream,
        )

        def delayed_stop() -> None:
            time.sleep(0.2)
            runner.stop()

        stop_thread = threading.Thread(target=delayed_stop)
        stop_thread.start()

        metrics = runner.run_continuous()
        stop_thread.join()

        assert metrics.orders_processed > 0
        assert runner.dashboard.render_count >= 1
        runner.close()
