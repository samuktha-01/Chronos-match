"""Unit Test Suite for ChronosMatch Terminal Dashboard Metrics and Data Layer.

Validates:
1. Empty metrics and unavailable values (None vs actual 0).
2. Known metric values and calculations (orders, trades, volume, ring buffer utilization).
3. Throughput and latency calculations (p50, p95, p99, cumulative and window throughput).
4. Bounded latency sample storage (fixed circular reservoir, memory cap).
5. Consistent, immutable point-in-time metric snapshots.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal
import threading
import time
from unittest.mock import patch

import pytest

from chronosmatch.dashboard import (
    BoundedLatencyTracker,
    DashboardMetricsCollector,
    DashboardMetricsSnapshot,
    LatencyStats,
    calculate_percentile,
)
from chronosmatch.matching_engine import MatchingEngine
from chronosmatch.models import Order, OrderSide
from chronosmatch.pipeline import TradingPipeline, create_pipeline
from chronosmatch.ring_buffer import RingBuffer


# ==============================================================================
# 1. Empty Metrics and Unavailable Values
# ==============================================================================


class TestEmptyAndUnavailableMetrics:
    """Verify that unattached or uninstrumented components report None, distinguishing unavailable metrics from actual zero values."""

    def test_unattached_collector_reports_none_for_unavailable_metrics(self) -> None:
        """Verify an unattached collector reports None for all uninstrumented metrics."""
        collector = DashboardMetricsCollector(source_name="TestEmpty")
        snapshot = collector.take_snapshot()

        assert snapshot.source_name == "TestEmpty"
        assert snapshot.orders_generated is None
        assert snapshot.orders_published is None
        assert snapshot.orders_consumed is None
        assert snapshot.orders_processed is None
        assert snapshot.orders_cancelled is None
        assert snapshot.orders_dropped is None
        assert snapshot.trades_executed is None
        assert snapshot.volume_executed is None
        assert snapshot.ring_buffer_capacity is None
        assert snapshot.ring_buffer_occupied is None
        assert snapshot.ring_buffer_utilization is None
        assert snapshot.malformed_messages is None
        assert snapshot.duplicate_orders is None
        assert snapshot.processing_errors is None
        assert snapshot.backpressure_events is None
        assert snapshot.latency is None
        assert snapshot.elapsed_seconds is None
        assert snapshot.throughput_orders_per_sec is None
        assert snapshot.window_throughput_orders_per_sec is None
        assert snapshot.best_bid is None
        assert snapshot.best_ask is None
        assert snapshot.spread is None

    def test_distinguish_unavailable_none_from_actual_zero(self) -> None:
        """Verify that an explicit zero metric is distinguished from an unavailable (None) metric."""
        collector = DashboardMetricsCollector()

        # Explicitly record 0 dropped orders (known zero)
        collector.record_order_dropped(0)
        # Explicitly record 0 processed orders
        collector.record_order_processed(0)

        snapshot = collector.take_snapshot()

        # Should be actual 0 (int), NOT None
        assert snapshot.orders_dropped is not None
        assert snapshot.orders_dropped == 0
        assert snapshot.orders_processed is not None
        assert snapshot.orders_processed == 0

        # Untracked metrics MUST remain None
        assert snapshot.orders_generated is None
        assert snapshot.orders_cancelled is None
        assert snapshot.trades_executed is None

    def test_partial_source_matching_engine_only(self) -> None:
        """Verify attaching only a MatchingEngine exposes trade/book metrics but leaves ring buffer/feed as None."""
        engine = MatchingEngine()
        collector = DashboardMetricsCollector(engine=engine)
        snapshot = collector.take_snapshot()

        # Engine metrics are known and available
        assert snapshot.trades_executed == 0
        assert snapshot.volume_executed == 0
        assert snapshot.best_bid is None  # book is empty, so BBO is None
        assert snapshot.order_book_depth_levels == 0

        # Feed and ring buffer metrics MUST be None (never fabricate)
        assert snapshot.orders_generated is None
        assert snapshot.orders_published is None
        assert snapshot.orders_consumed is None
        assert snapshot.ring_buffer_capacity is None
        assert snapshot.ring_buffer_utilization is None

    def test_partial_source_ring_buffer_only(self) -> None:
        """Verify attaching only a RingBuffer exposes capacity and occupancy but leaves trades as None."""
        rb = RingBuffer.create_anonymous(capacity=512)
        try:
            collector = DashboardMetricsCollector(ring_buffer=rb)
            snapshot = collector.take_snapshot()

            assert snapshot.ring_buffer_capacity == 512
            assert snapshot.ring_buffer_occupied == 0
            assert snapshot.ring_buffer_utilization == 0.0
            assert snapshot.orders_published == 0
            assert snapshot.orders_consumed == 0

            # Matching engine metrics MUST be None
            assert snapshot.trades_executed is None
            assert snapshot.volume_executed is None
            assert snapshot.orders_processed is None
        finally:
            rb.close()


# ==============================================================================
# 2. Known Metric Values and Calculations
# ==============================================================================


class TestKnownMetricValuesAndCalculations:
    """Verify metrics calculation against known inputs and attached TradingPipeline."""

    def test_known_values_from_live_trading_pipeline(self) -> None:
        """Verify collector accurately extracts telemetry from an active TradingPipeline."""
        pipeline = create_pipeline(capacity=64)
        try:
            collector = DashboardMetricsCollector(pipeline=pipeline)

            # 1. Place resting BUY order
            pipeline.process_order(Order(order_id="BUY-1", side=OrderSide.BUY, price=100.0, quantity=50, timestamp=1))
            # 2. Place resting BUY order at lower price
            pipeline.process_order(Order(order_id="BUY-2", side=OrderSide.BUY, price=95.0, quantity=30, timestamp=2))
            # 3. Match aggressive SELL order against BUY-1 (fill 20)
            trades = pipeline.process_order(Order(order_id="SELL-1", side=OrderSide.SELL, price=100.0, quantity=20, timestamp=3))
            assert len(trades) == 1

            snapshot = collector.take_snapshot()

            assert snapshot.orders_published == 3
            assert snapshot.orders_processed == 3
            assert snapshot.trades_executed == 1
            assert snapshot.volume_executed == 20
            assert snapshot.ring_buffer_capacity == 64
            assert snapshot.ring_buffer_occupied == 0
            assert snapshot.ring_buffer_utilization == 0.0
            assert snapshot.best_bid == 100.0
            assert snapshot.order_book_depth_levels == 2  # 2 bid levels (100.0 and 95.0)

            # Cancel remaining BUY-2
            pipeline.cancel_order("BUY-2")
            snapshot2 = collector.take_snapshot()
            assert snapshot2.orders_cancelled == 1
            assert snapshot2.order_book_depth_levels == 1  # 1 bid level remaining
        finally:
            pipeline.close()

    def test_ring_buffer_utilization_ratio(self) -> None:
        """Verify ring buffer utilization ratio is correctly calculated between 0.0 and 1.0."""
        rb = RingBuffer.create_anonymous(capacity=100)
        try:
            collector = DashboardMetricsCollector(ring_buffer=rb)

            # Write 25 orders
            for i in range(25):
                rb.put(Order(order_id=f"O{i}", side=OrderSide.BUY, price=100, quantity=1, timestamp=i + 1))

            snapshot = collector.take_snapshot()
            assert snapshot.ring_buffer_capacity == 100
            assert snapshot.ring_buffer_occupied == 25
            assert snapshot.ring_buffer_utilization == pytest.approx(0.25, abs=1e-5)
        finally:
            rb.close()

    def test_processing_errors_aggregation(self) -> None:
        """Verify malformed and duplicate order errors aggregate into processing_errors."""
        collector = DashboardMetricsCollector()
        collector.record_error(count=3)
        snapshot = collector.take_snapshot()
        assert snapshot.processing_errors == 3

    def test_snapshot_to_dict_keys_and_values(self) -> None:
        """Verify snapshot.to_dict() produces valid serializable dictionary."""
        collector = DashboardMetricsCollector(source_name="DictTest")
        collector.record_order_processed(10)
        collector.record_trade(quantity=50)
        collector.record_latency(1500)

        snapshot = collector.take_snapshot()
        d = snapshot.to_dict()

        assert isinstance(d, dict)
        assert d["source_name"] == "DictTest"
        assert d["orders_processed"] == 10
        assert d["trades_executed"] == 1
        assert d["volume_executed"] == 50
        assert d["latency"]["p50_ns"] == 1500.0
        assert d["latency"]["p50_us"] == 1.5


# ==============================================================================
# 3. Throughput and Latency Calculations
# ==============================================================================


class TestThroughputAndLatencyCalculations:
    """Verify mathematical correctness of percentile and throughput calculations."""

    def test_calculate_percentile_helper(self) -> None:
        """Verify percentile calculation on single, odd, and even sample lists."""
        # Single sample
        assert calculate_percentile([42.0], 50.0) == 42.0
        assert calculate_percentile([42.0], 99.0) == 42.0

        # Exact odd list: [10, 20, 30, 40, 50]
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert calculate_percentile(data, 0.0) == 10.0
        assert calculate_percentile(data, 50.0) == 30.0
        assert calculate_percentile(data, 100.0) == 50.0

        # Linear interpolation
        # k = 4 * 0.25 = 1.0 -> 20.0
        assert calculate_percentile(data, 25.0) == 20.0
        # k = 4 * 0.75 = 3.0 -> 40.0
        assert calculate_percentile(data, 75.0) == 40.0

    def test_calculate_percentile_bounds_errors(self) -> None:
        """Verify invalid percentiles or empty lists raise ValueError."""
        with pytest.raises(ValueError):
            calculate_percentile([], 50.0)
        with pytest.raises(ValueError):
            calculate_percentile([1.0], -0.1)
        with pytest.raises(ValueError):
            calculate_percentile([1.0], 100.1)

    def test_latency_tracker_percentiles_and_units(self) -> None:
        """Verify BoundedLatencyTracker produces exact min, max, mean, p50, p95, p99 in ns and us."""
        tracker = BoundedLatencyTracker(max_samples=1000)

        # 100 samples uniformly spaced from 1,000 ns to 100,000 ns (1 us to 100 us)
        for i in range(1, 101):
            tracker.record_latency(i * 1000)

        stats = tracker.get_stats()
        assert stats.sample_count == 100
        assert stats.min_ns == 1000.0
        assert stats.max_ns == 100000.0
        assert stats.mean_ns == pytest.approx(50500.0, abs=1.0)
        assert stats.p50_ns == pytest.approx(50500.0, abs=1.0)
        assert stats.p95_ns == pytest.approx(95050.0, abs=1.0)
        assert stats.p99_ns == pytest.approx(99010.0, abs=1.0)
        assert stats.latest_ns == 100000.0

        # Verify microsecond convenience properties
        assert stats.min_us == 1.0
        assert stats.max_us == 100.0
        assert stats.p50_us == pytest.approx(50.5, abs=0.1)
        assert stats.p95_us == pytest.approx(95.05, abs=0.1)
        assert stats.p99_us == pytest.approx(99.01, abs=0.1)

    def test_cumulative_throughput_calculation(self) -> None:
        """Verify cumulative throughput = processed_orders / elapsed_seconds."""
        collector = DashboardMetricsCollector()
        collector.start()

        # Simulate 500 processed orders over 2.0 seconds
        collector.record_order_processed(500)

        with patch.object(collector, "_compute_elapsed_seconds", return_value=2.0):
            snapshot = collector.take_snapshot()
            assert snapshot.elapsed_seconds == 2.0
            assert snapshot.throughput_orders_per_sec == 250.0

    def test_window_throughput_calculation(self) -> None:
        """Verify incremental sliding-window throughput between consecutive snapshots."""
        collector = DashboardMetricsCollector()

        # Snapshot 1 at t=1000ms: 100 orders
        collector.record_order_processed(100)
        with patch("time.perf_counter_ns", return_value=1_000_000_000):
            s1 = collector.take_snapshot()
            assert s1.window_throughput_orders_per_sec is None  # no previous baseline

        # Snapshot 2 at t=2000ms: 250 orders (delta: 150 orders in 1.0 sec)
        collector.record_order_processed(150)
        with patch("time.perf_counter_ns", return_value=2_000_000_000):
            s2 = collector.take_snapshot()
            assert s2.window_throughput_orders_per_sec == pytest.approx(150.0, abs=1e-3)


# ==============================================================================
# 4. Bounded Latency Sample Storage
# ==============================================================================


class TestBoundedLatencySampleStorage:
    """Verify that latency sample storage is strictly bounded by max_samples."""

    def test_sample_capacity_is_strictly_bounded(self) -> None:
        """Verify reservoir never retains more than max_samples, evicting oldest samples."""
        max_cap = 10
        tracker = BoundedLatencyTracker(max_samples=max_cap)

        # Record 100 samples (1 to 100)
        for i in range(1, 101):
            tracker.record_latency(i)

        assert tracker.max_samples == max_cap
        assert tracker.current_sample_count == max_cap  # strictly 10
        assert tracker.total_recorded_count == 100

        # The 10 retained samples MUST be the latest: 91..100
        stats = tracker.get_stats()
        assert stats.sample_count == 10
        assert stats.min_ns == 91.0
        assert stats.max_ns == 100.0
        assert stats.latest_ns == 100.0

    def test_tracker_clear_resets_storage(self) -> None:
        """Verify clear() empties the sample reservoir."""
        tracker = BoundedLatencyTracker(max_samples=50)
        for i in range(25):
            tracker.record_latency(i * 10)

        assert tracker.current_sample_count == 25
        tracker.clear()
        assert tracker.current_sample_count == 0
        assert tracker.total_recorded_count == 0

        stats = tracker.get_stats()
        assert stats.sample_count == 0
        assert stats.p50_ns is None

    def test_invalid_parameters_raise(self) -> None:
        """Verify non-positive max_samples or negative latency raises ValueError."""
        with pytest.raises(ValueError):
            BoundedLatencyTracker(max_samples=0)
        with pytest.raises(ValueError):
            BoundedLatencyTracker(max_samples=-5)

        tracker = BoundedLatencyTracker(max_samples=10)
        with pytest.raises(ValueError):
            tracker.record_latency(-1.0)


# ==============================================================================
# 5. Consistent Metric Snapshots
# ==============================================================================


class TestConsistentMetricSnapshots:
    """Verify immutability and thread safety of point-in-time snapshots."""

    def test_snapshot_is_immutable(self) -> None:
        """Verify DashboardMetricsSnapshot instances cannot be modified after creation."""
        collector = DashboardMetricsCollector()
        collector.record_order_processed(10)
        snapshot = collector.take_snapshot()

        with pytest.raises(FrozenInstanceError):
            snapshot.orders_processed = 999  # type: ignore[misc]

    def test_collector_mutations_do_not_affect_existing_snapshots(self) -> None:
        """Verify mutating collector state after a snapshot does not alter the previous snapshot."""
        collector = DashboardMetricsCollector()
        collector.record_order_processed(50)
        s1 = collector.take_snapshot()

        # Mutate collector state
        collector.record_order_processed(100)
        collector.record_trade(quantity=20)
        s2 = collector.take_snapshot()

        assert s1.orders_processed == 50
        assert s1.trades_executed is None

        assert s2.orders_processed == 150
        assert s2.trades_executed == 1
        assert s2.volume_executed == 20

    def test_concurrent_sampling_and_snapshots_thread_safety(self) -> None:
        """Verify thread-safe concurrent recording and snapshotting without race conditions."""
        collector = DashboardMetricsCollector()
        errors: list[Exception] = []

        def worker_producer() -> None:
            try:
                for i in range(500):
                    collector.record_latency(100 + i)
                    collector.record_order_processed(1)
            except Exception as e:
                errors.append(e)

        def worker_reader() -> None:
            try:
                for _ in range(50):
                    snap = collector.take_snapshot()
                    assert snap.timestamp_ns > 0
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=worker_producer)
        t2 = threading.Thread(target=worker_reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0
        final_snap = collector.take_snapshot()
        assert final_snap.orders_processed == 500
        assert final_snap.latency is not None
        assert final_snap.latency.sample_count == 500


# ==============================================================================
# 6. Controlled Throughput, Trade Accuracy & Unavailable Metric Behavior
# ==============================================================================


class TestMetricsCollectorValidationAndEdgeCases:
    """Verify controlled throughput calculations, trade accuracy, and unavailable metric handling."""

    def test_controlled_cumulative_throughput(self) -> None:
        """Verify cumulative throughput with exact controlled inputs, sub-microsecond and zero intervals."""
        collector = DashboardMetricsCollector()

        # Normal interval
        assert collector._compute_cumulative_throughput(100, 2.0) == 50.0
        assert collector._compute_cumulative_throughput(1000, 0.25) == 4000.0

        # Zero or negative interval returns 0.0 without division by zero
        assert collector._compute_cumulative_throughput(100, 0.0) == 0.0
        assert collector._compute_cumulative_throughput(100, -1.0) == 0.0

        # Sub-microsecond interval (<= 1e-6) returns 0.0 safely
        assert collector._compute_cumulative_throughput(100, 1e-7) == 0.0
        assert collector._compute_cumulative_throughput(100, 1e-6) == 0.0

        # None inputs
        assert collector._compute_cumulative_throughput(None, 1.0) is None
        assert collector._compute_cumulative_throughput(100, None) is None
        assert collector._compute_cumulative_throughput(None, None) is None

    def test_controlled_window_throughput(self) -> None:
        """Verify sliding-window throughput using controlled order deltas and nanosecond intervals."""
        collector = DashboardMetricsCollector()

        # Uninitialized previous state returns None
        assert collector._compute_window_throughput(100, 1_000_000_000) is None

        # Seed previous state
        collector._prev_processed_orders = 100
        collector._prev_snapshot_time_ns = 1_000_000_000

        # Advance by 0.5s (500_000_000 ns) with 40 new orders -> 80 ops/s
        rate = collector._compute_window_throughput(140, 1_500_000_000)
        assert rate == pytest.approx(80.0)

        # Zero elapsed delta returns 0.0 safely
        rate_zero_time = collector._compute_window_throughput(150, 1_000_000_000)
        assert rate_zero_time == 0.0

        # Sub-microsecond elapsed delta (<= 1e-6 seconds = 1000 ns) returns 0.0 safely
        rate_micro = collector._compute_window_throughput(150, 1_000_000_500)
        assert rate_micro == 0.0

        # Negative order delta returns 0.0
        rate_neg = collector._compute_window_throughput(90, 2_000_000_000)
        assert rate_neg == 0.0

    def test_trade_count_and_volume_accuracy_on_partial_fills(self) -> None:
        """Verify executed trade count and traded volume accuracy when an incoming order partially fills multiple resting orders."""
        engine = MatchingEngine()
        collector = DashboardMetricsCollector(engine=engine)

        # Post 2 resting SELL orders at different prices
        # SELL 1: 10 @ 100.00
        engine.process_order(Order(order_id="SELL-A", side=OrderSide.SELL, price=100.0, quantity=10, timestamp=1))
        # SELL 2: 20 @ 102.00
        engine.process_order(Order(order_id="SELL-B", side=OrderSide.SELL, price=102.0, quantity=20, timestamp=2))

        # Check initial collector state: 0 trades, 0 volume
        s0 = collector.take_snapshot()
        assert s0.trades_executed == 0
        assert s0.volume_executed == 0

        # Incoming BUY: 25 @ 102.00 (Crosses SELL-A completely for 10 @ 100.00, and SELL-B partially for 15 @ 102.00)
        trades = engine.process_order(Order(order_id="BUY-SWEEP", side=OrderSide.BUY, price=102.0, quantity=25, timestamp=3))

        assert len(trades) == 2
        assert trades[0].quantity == 10
        assert trades[0].price == 100.0
        assert trades[1].quantity == 15
        assert trades[1].price == 102.0

        # Engine level checks
        assert engine.total_trades == 2
        assert engine.total_volume_traded == 25

        # Snapshot checks
        s1 = collector.take_snapshot()
        assert s1.trades_executed == 2
        assert s1.volume_executed == 25
        # Remaining resting ask: 5 @ 102.00
        assert s1.best_ask == 102.0

    def test_dashboard_formatters_when_metric_unavailable(self) -> None:
        """Verify dashboard formatters return 'N/A' and never fabricate zeros when metrics are None."""
        from chronosmatch.dashboard.terminal import (
            AnsiStyler,
            ASCII_BOX,
            format_counter_with_alert,
            format_int,
            format_latency,
            format_price,
            format_rate,
            format_time,
            format_utilization_bar,
            strip_ansi,
        )

        styler = AnsiStyler(enabled=False)

        assert strip_ansi(format_int(None, styler)) == "N/A"
        assert strip_ansi(format_rate(None, styler)) == "N/A"
        assert strip_ansi(format_time(None, styler)) == "N/A"
        assert strip_ansi(format_price(None, styler)) == "N/A"
        assert strip_ansi(format_counter_with_alert(None, styler)) == "N/A"
        assert strip_ansi(format_latency(None, styler)) == "N/A"
        assert strip_ansi(format_utilization_bar(None, 10, ASCII_BOX, styler)) == "N/A"

        # Explicit zero should NOT be 'N/A'
        assert strip_ansi(format_int(0, styler)) == "0"
        assert strip_ansi(format_rate(0.0, styler)) == "0.00 ops/s"
        assert strip_ansi(format_counter_with_alert(0, styler)) == "0"
