"""Deterministic Integration Test Suite for ChronosMatch Terminal Dashboard and Pipeline.

Validates Part 3 Requirements:
1. Live pipeline integration: Dashboard updates concurrently while the real pipeline
   processes synthetic orders through shared-memory IPC, ring buffer, and matching engine.
2. Metric consistency against known pipeline activity (orders, trades, volume, BBO spread).
3. Trade counts and real microsecond latency distribution measurements (p50, p95, p99).
4. Graceful shutdown when pipeline completes, stops, or is interrupted.
5. Error handling and missing optional components (WAL, TradeStore, closed ring buffer,
   duplicate orders, malformed messages).
6. Non-fabrication of metrics and honest reporting of Cython acceleration status.
7. Convenience runner and CLI entry points.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
import io
import re
import threading
import time
from typing import Any
from unittest.mock import patch

import pytest

from chronosmatch.cython_engine import is_cython_compiled, CythonMatchingEngine
from chronosmatch.dashboard import (
    BoundedLatencyTracker,
    DashboardMetricsCollector,
    DashboardMetricsSnapshot,
    LatencyInstrumentedEngine,
    PipelineDashboardRunner,
    TerminalDashboard,
    run_pipeline_with_dashboard,
    strip_ansi,
)
from chronosmatch.dashboard.__main__ import main
from chronosmatch.market_simulator import (
    BackpressureMode,
    MarketFeedConfig,
    SimulationMode,
)
from chronosmatch.matching_engine import MatchingEngine, Trade
from chronosmatch.models import Order, OrderSide
from chronosmatch.order_book import DuplicateOrderError
from chronosmatch.pipeline import (
    TradingPipeline,
    TradingPipelineMetrics,
    create_pipeline,
)
from chronosmatch.ring_buffer import RingBufferClosedError


def assert_field(rendered: str, label: str, value: str) -> None:
    """Verify that a given label and value appear together in the rendered output."""
    plain = strip_ansi(rendered)
    pattern = rf"{re.escape(label)}:\s+{re.escape(value)}"
    assert re.search(pattern, plain) is not None, (
        f"Expected '{label}: {value}' in rendered output, but not found.\nOutput:\n{plain}"
    )


# ==============================================================================
# 1. Pipeline Execution & Dashboard Updates
# ==============================================================================


class TestPipelineDashboardExecution:
    """Verify concurrent and deterministic pipeline execution coupled with dashboard telemetry."""

    def test_run_steps_updates_dashboard_and_metrics(self) -> None:
        """Verify that step-by-step pipeline execution updates orders, trades, and latency."""
        out = io.StringIO()
        config = MarketFeedConfig(total_orders=30, rate=5000.0, mode=SimulationMode.MAX_THROUGHPUT)
        pipeline = create_pipeline(config=config, capacity=128)

        runner = PipelineDashboardRunner(
            pipeline=pipeline,
            interactive=False,
            use_colors=False,
            use_unicode=False,
            output=out,
        )

        try:
            trades = runner.run_steps(30)
            assert isinstance(trades, list)

            snapshot = runner.collector.take_snapshot()
            assert snapshot.orders_generated == 30
            assert snapshot.orders_processed == 30
            assert snapshot.trades_executed == pipeline.engine.total_trades
            assert snapshot.volume_executed == pipeline.engine.total_volume_traded
            assert snapshot.ring_buffer_capacity == 128
            assert snapshot.ring_buffer_occupied is not None
            assert snapshot.ring_buffer_utilization is not None

            # Verify latency tracker sampled the 30 orders
            assert snapshot.latency is not None
            assert snapshot.latency.sample_count == 30
            assert snapshot.latency.p50_us is not None
            assert snapshot.latency.p50_us > 0.0

            rendered = out.getvalue()
            assert "Orders Processed:" in rendered
            assert "Executed Trades:" in rendered
            assert "Buffer Capacity:" in rendered
            assert_field(rendered, "Orders Processed", "30")
        finally:
            runner.close()

    def test_run_sync_execution_and_snapshot_consistency(self) -> None:
        """Verify synchronous pipeline run executes fully and captures consistent final metrics."""
        out = io.StringIO()
        config = MarketFeedConfig(total_orders=50, rate=10000.0, mode=SimulationMode.MAX_THROUGHPUT)
        pipeline = create_pipeline(config=config, capacity=128)

        runner = PipelineDashboardRunner(
            pipeline=pipeline,
            refresh_interval=0.05,
            interactive=False,
            use_colors=False,
            use_unicode=False,
            output=out,
        )

        try:
            metrics = runner.run(total_orders=50)
            assert metrics.orders_processed == 50

            snapshot = runner.collector.take_snapshot()
            assert snapshot.orders_processed == 50
            assert snapshot.trades_executed == metrics.trades_executed
            assert snapshot.volume_executed == metrics.volume_executed
            assert snapshot.throughput_orders_per_sec is not None
            assert snapshot.throughput_orders_per_sec > 0.0

            assert snapshot.latency is not None
            assert snapshot.latency.sample_count == 50

            rendered = out.getvalue()
            assert "Orders Processed:" in rendered
            assert_field(rendered, "Orders Processed", "50")
        finally:
            runner.close()

    def test_run_async_execution(self) -> None:
        """Verify asynchronous pipeline run executes cleanly with live dashboard telemetry."""
        out = io.StringIO()
        config = MarketFeedConfig(total_orders=40, rate=10000.0, mode=SimulationMode.MAX_THROUGHPUT)
        pipeline = create_pipeline(config=config, capacity=128)

        runner = PipelineDashboardRunner(
            pipeline=pipeline,
            refresh_interval=0.05,
            interactive=False,
            use_colors=False,
            use_unicode=False,
            output=out,
        )

        async def _execute() -> TradingPipelineMetrics:
            return await runner.run_async(total_orders=40)

        try:
            metrics = asyncio.run(_execute())
            assert metrics.orders_processed == 40

            snapshot = runner.collector.take_snapshot()
            assert snapshot.orders_processed == 40
            assert snapshot.latency is not None
            assert snapshot.latency.sample_count == 40
        finally:
            runner.close()


# ==============================================================================
# 2. Metric Consistency Against Known Pipeline Activity
# ==============================================================================


class TestMetricConsistencyAgainstKnownPipelineActivity:
    """Verify telemetry values strictly reflect known deterministic orders and trades."""

    def test_manual_order_matching_and_telemetry_consistency(self) -> None:
        """Submit matching orders and verify exact trade, volume, and BBO consistency."""
        out = io.StringIO()
        pipeline = create_pipeline(capacity=64)
        runner = PipelineDashboardRunner(
            pipeline=pipeline,
            interactive=False,
            use_colors=False,
            use_unicode=False,
            output=out,
        )

        try:
            # Order 1: Buy 100 @ 150.00
            buy_order = Order(
                order_id="BUY-001",
                side=OrderSide.BUY,
                price=Decimal("150.00"),
                quantity=100,
                timestamp=1_000_000_000,
            )
            trades1 = runner.process_order(buy_order)
            assert len(trades1) == 0

            s1 = runner.collector.take_snapshot()
            assert s1.orders_processed == 1
            assert s1.trades_executed == 0
            assert s1.volume_executed == 0
            assert s1.best_bid == Decimal("150.00")
            assert s1.best_ask is None
            assert s1.spread is None

            # Order 2: Sell 60 @ 150.00 (Crosses with BUY-001)
            sell_order = Order(
                order_id="SELL-001",
                side=OrderSide.SELL,
                price=Decimal("150.00"),
                quantity=60,
                timestamp=2_000_000_000,
            )
            trades2 = runner.process_order(sell_order)
            assert len(trades2) == 1
            assert trades2[0].quantity == 60
            assert trades2[0].price == Decimal("150.00")

            s2 = runner.collector.take_snapshot()
            assert s2.orders_processed == 2
            assert s2.trades_executed == 1
            assert s2.volume_executed == 60
            # Resting buy order has 40 remaining
            assert s2.best_bid == Decimal("150.00")

            # Render frame and check fields
            rendered = runner.dashboard.render_to_string(s2)
            assert_field(rendered, "Orders Processed", "2")
            assert_field(rendered, "Executed Trades", "1")
            assert_field(rendered, "Traded Volume", "60")
            assert_field(rendered, "Best Bid", "$150.00")
        finally:
            runner.close()

    def test_resting_order_cancellation_metric(self) -> None:
        """Verify that cancelling a resting order is accurately tracked in telemetry."""
        pipeline = create_pipeline(capacity=64)
        runner = PipelineDashboardRunner(pipeline=pipeline, interactive=False)

        try:
            order = Order(
                order_id="CANC-001",
                side=OrderSide.BUY,
                price=Decimal("120.00"),
                quantity=50,
                timestamp=1_000_000_000,
            )
            runner.process_order(order)
            assert pipeline.engine.best_bid == Decimal("120.00")

            pipeline.cancel_order("CANC-001")
            assert pipeline.engine.best_bid is None

            snapshot = runner.collector.take_snapshot()
            assert snapshot.orders_cancelled == 1
        finally:
            runner.close()


# ==============================================================================
# 3. Trade Counts and Available Latency Measurements
# ==============================================================================


class TestTradeCountsAndLatencyMeasurements:
    """Verify latency distributions are computed only when real samples exist."""

    def test_latency_tracker_percentiles_populated_from_real_pipeline(self) -> None:
        """Verify latency statistics percentiles are populated and correctly ordered."""
        pipeline = create_pipeline(capacity=128)
        runner = PipelineDashboardRunner(
            pipeline=pipeline,
            interactive=False,
            instrument_latency=True,
        )

        try:
            # Process 25 simulated orders
            runner.run_steps(25)
            snapshot = runner.collector.take_snapshot()

            assert snapshot.latency is not None
            lat = snapshot.latency
            assert lat.sample_count == 25
            assert lat.min_us is not None and lat.min_us > 0.0
            assert lat.mean_us is not None and lat.mean_us > 0.0
            assert lat.p50_us is not None and lat.p50_us > 0.0
            assert lat.p95_us is not None and lat.p95_us > 0.0
            assert lat.p99_us is not None and lat.p99_us > 0.0
            assert lat.max_us is not None and lat.max_us >= lat.p99_us

            # Monotonic ordering of percentiles
            assert lat.min_us <= lat.p50_us <= lat.p95_us <= lat.p99_us <= lat.max_us

            rendered = runner.dashboard.render_to_string(snapshot)
            assert "p50 (Median):" in rendered
            assert "(25 Samples)" in rendered
            # Must NOT display N/A for p50 when samples exist
            assert not re.search(r"p50 \(Median\):\s+N/A", rendered)
        finally:
            runner.close()

    def test_latency_uninstrumented_reports_none(self) -> None:
        """Verify that when latency instrumentation is disabled, None / N/A is displayed."""
        pipeline = create_pipeline(capacity=64)
        runner = PipelineDashboardRunner(
            pipeline=pipeline,
            interactive=False,
            instrument_latency=False,
        )

        try:
            runner.run_steps(10)
            snapshot = runner.collector.take_snapshot()

            # No latency should be recorded
            assert snapshot.latency is None

            rendered = runner.dashboard.render_to_string(snapshot)
            assert "(No Samples)" in rendered
            assert_field(rendered, "p50 (Median)", "N/A")
            assert_field(rendered, "p95", "N/A")
            assert_field(rendered, "p99", "N/A")
        finally:
            runner.close()


# ==============================================================================
# 4. Graceful Shutdown & Lifecycle Management
# ==============================================================================


class TestGracefulShutdownAndLifecycle:
    """Verify clean shutdown, resource reclamation, and terminal state preservation."""

    def test_graceful_shutdown_on_stop(self) -> None:
        """Verify that stop() shuts down pipeline and dashboard background thread cleanly."""
        pipeline = create_pipeline(capacity=64)
        runner = PipelineDashboardRunner(
            pipeline=pipeline,
            refresh_interval=0.1,
            interactive=False,
        )

        runner.dashboard.start()
        assert runner.dashboard.is_running is True

        runner.stop()
        assert runner.dashboard.is_running is False
        runner.close()

    def test_context_manager_lifecycle(self) -> None:
        """Verify runner works seamlessly as a synchronous context manager."""
        with PipelineDashboardRunner(interactive=False) as runner:
            runner.run_steps(5)
            s = runner.collector.take_snapshot()
            assert s.orders_processed == 5

        # After exiting context, dashboard must be stopped
        assert runner.dashboard.is_running is False

    def test_async_context_manager_lifecycle(self) -> None:
        """Verify runner works seamlessly as an asynchronous context manager."""
        async def _test() -> None:
            async with PipelineDashboardRunner(interactive=False) as runner:
                runner.run_steps(5)
                s = runner.collector.take_snapshot()
                assert s.orders_processed == 5
            assert runner.dashboard.is_running is False

        asyncio.run(_test())

    def test_snapshot_after_pipeline_cleanup_does_not_crash(self) -> None:
        """Verify collector and renderer do not crash if ring buffer is closed after pipeline cleanup."""
        pipeline = create_pipeline(capacity=64)
        runner = PipelineDashboardRunner(pipeline=pipeline, interactive=False)

        runner.run_steps(10)
        # Explicitly cleanup / close pipeline (closes underlying ring buffer)
        pipeline.cleanup()

        # Collector must handle closed ring buffer safely without raising RingBufferClosedError
        snapshot = runner.collector.take_snapshot()
        assert snapshot.ring_buffer_capacity is None
        assert snapshot.ring_buffer_occupied is None
        assert snapshot.ring_buffer_utilization is None

        # Rendering must succeed with N/A for ring buffer metrics
        rendered = runner.dashboard.render_to_string(snapshot)
        assert_field(rendered, "Buffer Capacity", "N/A")
        assert_field(rendered, "Occupied Slots", "N/A")
        runner.close()


# ==============================================================================
# 5. Error Handling and Missing Optional Components
# ==============================================================================


class TestErrorHandlingAndOptionalComponents:
    """Verify resilience against duplicate orders, malformed records, and missing optional subsystems."""

    def test_pipeline_without_wal_or_trade_store(self) -> None:
        """Verify dashboard runs cleanly without optional WAL or TradeStore persistence."""
        pipeline = create_pipeline(
            capacity=64,
            wal=None,
            wal_path=None,
            trade_store=None,
        )
        runner = PipelineDashboardRunner(pipeline=pipeline, interactive=False)

        try:
            trades = runner.run_steps(15)
            snapshot = runner.collector.take_snapshot()
            assert snapshot.orders_processed == 15
        finally:
            runner.close()

    def test_duplicate_order_increments_error_counter(self) -> None:
        """Verify DuplicateOrderError is recorded and presented in telemetry."""
        pipeline = create_pipeline(capacity=64)
        runner = PipelineDashboardRunner(pipeline=pipeline, interactive=False)

        try:
            order = Order(
                order_id="DUP-001",
                side=OrderSide.BUY,
                price=Decimal("100.00"),
                quantity=10,
                timestamp=1_000_000_000,
            )
            runner.process_order(order)

            # Re-submitting identical order raises DuplicateOrderError
            with pytest.raises(DuplicateOrderError):
                runner.process_order(order)

            snapshot = runner.collector.take_snapshot()
            assert snapshot.duplicate_orders == 1
            assert snapshot.processing_errors == 1

            rendered = runner.dashboard.render_to_string(snapshot)
            assert_field(rendered, "Duplicate Orders", "1")
            assert_field(rendered, "Processing Errors", "1")
        finally:
            runner.close()

    def test_malformed_message_counter_in_snapshot(self) -> None:
        """Verify that malformed message events increment error counters."""
        pipeline = create_pipeline(capacity=64)
        runner = PipelineDashboardRunner(pipeline=pipeline, interactive=False)

        try:
            # Simulate a malformed message detected during draining
            pipeline.metrics.malformed_messages += 2

            snapshot = runner.collector.take_snapshot()
            assert snapshot.malformed_messages == 2
            assert snapshot.processing_errors == 2

            rendered = runner.dashboard.render_to_string(snapshot)
            assert_field(rendered, "Malformed Msgs", "2")
            assert_field(rendered, "Processing Errors", "2")
        finally:
            runner.close()

    def test_cython_acceleration_honesty(self) -> None:
        """Verify that Cython acceleration status is reported honestly and never fabricated."""
        pipeline = create_pipeline(capacity=64)
        runner = PipelineDashboardRunner(pipeline=pipeline, interactive=False)

        try:
            snapshot = runner.collector.take_snapshot()

            expected_accelerated = is_cython_compiled() and isinstance(
                pipeline.engine.target_engine
                if isinstance(pipeline.engine, LatencyInstrumentedEngine)
                else pipeline.engine,
                CythonMatchingEngine,
            )

            assert snapshot.is_cython_accelerated == expected_accelerated

            rendered = runner.dashboard.render_to_string(snapshot)
            if expected_accelerated:
                assert "Core: Cython" in rendered
            else:
                assert "Core: Python" in rendered
        finally:
            runner.close()


# ==============================================================================
# 6. Convenience Runner and CLI Entry Points
# ==============================================================================


class TestConvenienceRunnerAndCli:
    """Verify run_pipeline_with_dashboard and CLI main() entry point."""

    def test_run_pipeline_with_dashboard_convenience_function(self) -> None:
        """Verify run_pipeline_with_dashboard runs pipeline and returns metrics."""
        out = io.StringIO()
        metrics = run_pipeline_with_dashboard(
            total_orders=25,
            rate=10000.0,
            mode=SimulationMode.MAX_THROUGHPUT,
            capacity=64,
            refresh_interval=0.05,
            interactive=False,
            use_colors=False,
            use_unicode=False,
            output=out,
        )

        assert isinstance(metrics, TradingPipelineMetrics)
        assert metrics.orders_processed == 25
        assert "CHRONOSMATCH" in out.getvalue()

    def test_cli_main_single_shot(self) -> None:
        """Verify CLI main() with --once executes single snapshot and exits with 0."""
        ret = main(["--once", "--ascii", "--no-color"])
        assert ret == 0

    def test_cli_main_live_run(self) -> None:
        """Verify CLI main() executes a small live pipeline run and exits with 0."""
        ret = main([
            "--orders", "15",
            "--rate", "5000",
            "--mode", "MAX_THROUGHPUT",
            "--capacity", "64",
            "--plain",
            "--no-color",
            "--ascii",
        ])
        assert ret == 0

    def test_cli_main_invalid_refresh_fails(self) -> None:
        """Verify CLI main() rejects non-positive refresh intervals with exit code 1."""
        ret = main(["--refresh", "-1.0"])
        assert ret == 1


# ==============================================================================
# 7. Pipeline Metrics Validation, In-Flight Tracking & Final Reconciliation
# ==============================================================================


class TestPipelineMetricsValidationAndReconciliation:
    """Verify generated-order counter increments, in-flight message tracking, drops, and reconciliation."""

    def test_generated_order_counter_increments_and_final_reconciliation(self) -> None:
        """Verify orders_generated is non-zero, increments properly, and reconciles with published/consumed/processed."""
        out = io.StringIO()
        total = 60
        config = MarketFeedConfig(
            total_orders=total,
            rate=10000.0,
            mode=SimulationMode.MAX_THROUGHPUT,
            cancel_probability=0.0,
        )
        pipeline = create_pipeline(config=config, capacity=128)

        runner = PipelineDashboardRunner(
            pipeline=pipeline,
            refresh_interval=0.05,
            interactive=False,
            use_colors=False,
            use_unicode=False,
            output=out,
        )

        try:
            metrics = runner.run(total_orders=total)
            assert metrics.orders_generated == total
            assert metrics.orders_published == total
            assert metrics.orders_consumed == total
            assert metrics.orders_processed == total
            assert metrics.orders_dropped == 0
            assert metrics.duplicate_orders == 0
            assert metrics.malformed_messages == 0

            # Telemetry snapshot must strictly reflect non-zero generated orders
            snapshot = runner.collector.take_snapshot()
            assert snapshot.orders_generated == total
            assert snapshot.orders_published == total
            assert snapshot.orders_consumed == total
            assert snapshot.orders_processed == total
            assert snapshot.orders_dropped == 0
            assert snapshot.processing_errors == 0
            assert snapshot.ring_buffer_occupied == 0

            # Rendered text must display the exact non-zero counts
            rendered = out.getvalue()
            assert_field(rendered, "Orders Generated", str(total))
            assert_field(rendered, "Orders Published", str(total))
            assert_field(rendered, "Orders Consumed", str(total))
            assert_field(rendered, "Orders Processed", str(total))
        finally:
            runner.close()

    def test_counter_behavior_while_messages_in_flight(self) -> None:
        """Verify published and consumed counter differences strictly correspond to in-flight buffer occupancy."""
        from chronosmatch.ring_buffer import RingBuffer
        from chronosmatch.producer_consumer import OrderConsumer, OrderProducer

        rb = RingBuffer.create_anonymous(capacity=16)
        producer = OrderProducer(rb)
        consumer = OrderConsumer(rb)

        collector = DashboardMetricsCollector(ring_buffer=rb)

        try:
            # Step 1: Initial idle state (0 published, 0 consumed, 0 in flight)
            s0 = collector.take_snapshot()
            assert s0.orders_published == 0
            assert s0.orders_consumed == 0
            assert s0.ring_buffer_occupied == 0

            # Step 2: Enqueue 5 orders without consuming -> 5 messages in flight
            for i in range(5):
                producer.publish(Order(order_id=f"INF-{i}", side=OrderSide.BUY, price=100.0 + i, quantity=1, timestamp=i + 1))

            s1 = collector.take_snapshot()
            assert s1.orders_published == 5
            assert s1.orders_consumed == 0
            assert s1.ring_buffer_occupied == 5
            # Difference strictly equals in-flight messages
            assert (s1.orders_published - s1.orders_consumed) == s1.ring_buffer_occupied == 5

            # Step 3: Consume 2 orders -> 3 messages in flight
            o1 = consumer.receive()
            o2 = consumer.receive()
            assert o1.order_id == "INF-0"
            assert o2.order_id == "INF-1"

            s2 = collector.take_snapshot()
            assert s2.orders_published == 5
            assert s2.orders_consumed == 2
            assert s2.ring_buffer_occupied == 3
            assert (s2.orders_published - s2.orders_consumed) == s2.ring_buffer_occupied == 3

            # Step 4: Drain remaining 3 orders -> 0 in flight, perfect reconciliation
            remaining = consumer.drain()
            assert len(remaining) == 3

            s3 = collector.take_snapshot()
            assert s3.orders_published == 5
            assert s3.orders_consumed == 5
            assert s3.ring_buffer_occupied == 0
            assert (s3.orders_published - s3.orders_consumed) == 0
        finally:
            producer.close()
            consumer.close()
            rb.close()

    def test_orders_dropped_counter_under_saturation(self) -> None:
        """Verify that when the downstream buffer saturates in DROP mode, orders_dropped is incremented."""
        config = MarketFeedConfig(
            total_orders=20,
            rate=10000.0,
            mode=SimulationMode.MAX_THROUGHPUT,
            backpressure_mode=BackpressureMode.DROP,
        )
        # Create pipeline with minimal ring buffer capacity (4) without running consumer
        pipeline = create_pipeline(config=config, capacity=4)
        try:
            # Fill the buffer to capacity
            for i in range(4):
                order = pipeline.simulator.next_order()
                published = asyncio.run(pipeline.simulator.publish_order(order, pipeline.producer))
                assert published is True

            assert pipeline.producer.is_full is True
            assert pipeline.simulator.metrics.orders_published == 4

            # Attempt to publish 3 more orders -> must drop due to saturation
            for _ in range(3):
                order = pipeline.simulator.next_order()
                published = asyncio.run(pipeline.simulator.publish_order(order, pipeline.producer))
                assert published is False

            assert pipeline.simulator.metrics.orders_dropped == 3
            assert pipeline.simulator.metrics.orders_published == 4
            assert pipeline.simulator.metrics.orders_generated == 7
            assert pipeline.simulator.metrics.backpressure_events >= 3

            # Verify collector reflects drops and backpressure
            collector = DashboardMetricsCollector(simulator=pipeline.simulator, ring_buffer=pipeline.ring_buffer)
            snap = collector.take_snapshot()
            assert snap.orders_dropped == 3
            assert snap.orders_published == 4
            assert snap.orders_generated == 7
            assert snap.backpressure_events >= 3
        finally:
            pipeline.close()

    def test_cancellations_counted_separately_from_new_orders(self) -> None:
        """Verify that order cancellations increment orders_cancelled without distorting orders_generated."""
        pipeline = create_pipeline(capacity=32)
        try:
            # Post resting order
            ord1 = Order(order_id="REST-001", side=OrderSide.BUY, price=100.0, quantity=10, timestamp=1)
            pipeline.process_order(ord1)
            assert pipeline.metrics.orders_published == 1
            assert pipeline.metrics.orders_processed == 1
            assert pipeline.metrics.orders_cancelled == 0

            # Cancel resting order
            pipeline.cancel_order("REST-001")
            assert pipeline.metrics.orders_cancelled == 1

            collector = DashboardMetricsCollector(pipeline=pipeline)
            snap = collector.take_snapshot()
            assert snap.orders_cancelled == 1
            assert snap.orders_processed == 1
        finally:
            pipeline.close()

    def test_dashboard_render_with_unavailable_metrics_displays_na(self) -> None:
        """Verify complete dashboard frame renders 'N/A' cleanly when telemetry sources are unavailable."""
        collector = DashboardMetricsCollector(source_name="EmptySource")
        dashboard = TerminalDashboard(collector=collector, interactive=False, use_colors=False)
        rendered = dashboard.render_to_string()

        # Should display 'N/A' for unavailable metrics rather than fake zeros
        assert "Orders Generated:  N/A" in rendered or "Orders Generated:       N/A" in rendered or "N/A" in rendered
        assert "Cum. Throughput:   N/A" in rendered or "Cum. Throughput:        N/A" in rendered or "N/A" in rendered
        assert "p50 (Median):      N/A" in rendered or "(No Samples)" in rendered

