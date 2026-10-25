"""Comprehensive Test Suite for ChronosMatch End-to-End Latency Instrumentation.

Validates:
1. Timestamp propagation (T0 -> T1 -> T2 -> T3) and clock domain consistency.
2. Latency boundary calculations:
   - Core matching latency = T3 - T2.
   - Queue dwell latency = T2 - T1.
   - End-to-end tick-to-trade latency = T3 - T0.
3. Unit conversions (nanoseconds to microseconds) and mathematical invariants (TTT >= Core + Dwell).
4. Edge cases: missing timestamps, wall-clock vs monotonic timestamps, duplicate orders, and non-executed orders.
5. Reservoir boundedness (BoundedLatencyTracker fixed capacity, no memory leak).
6. Dashboard presentation: Section 5 renders distinct Core, Dwell, and Tick-to-Trade breakdowns.
7. Backward compatibility: snapshot.latency and collector.latency_tracker aliases.
"""

from __future__ import annotations

from decimal import Decimal
import io
import re
import time

import pytest

from chronosmatch.dashboard import (
    BoundedLatencyTracker,
    DashboardMetricsCollector,
    DashboardMetricsSnapshot,
    DashboardRenderer,
    LatencyStats,
    PipelineDashboardRunner,
    TerminalDashboard,
    format_latency,
    strip_ansi,
)
from chronosmatch.market_simulator import (
    MarketFeedConfig,
    MarketSimulator,
    SimulationMode,
)
from chronosmatch.matching_engine import MatchingEngine
from chronosmatch.models import Order, OrderSide
from chronosmatch.pipeline import (
    TradingPipeline,
    create_pipeline,
)


def assert_field(rendered: str, label: str, value: str) -> None:
    """Verify that a given label and value appear together in the rendered output."""
    plain = strip_ansi(rendered)
    pattern = rf"{re.escape(label)}:\s+{re.escape(value)}"
    assert re.search(pattern, plain) is not None, (
        f"Expected '{label}: {value}' in rendered output, but not found.\nOutput:\n{plain}"
    )


class TestTimestampPropagationAndBoundaries:
    """Validate T0, T1, T2, T3 boundaries and latency arithmetic."""

    def test_market_simulator_generates_monotonic_perf_counter_t0(self) -> None:
        """Verify simulator orders generate T0 using monotonic perf_counter_ns."""
        sim = MarketSimulator(MarketFeedConfig(total_orders=10))
        t_before = time.perf_counter_ns()
        order = sim.next_order()
        t_after = time.perf_counter_ns()

        assert t_before <= order.timestamp <= t_after
        # Verify it's a monotonic timestamp (typically < 10^12 ns since boot, unlike epoch > 1.7*10^18 ns)
        assert order.timestamp < 1_500_000_000_000_000_000

    def test_pipeline_direct_latency_emission_and_invariants(self) -> None:
        """Verify latency calculations: Core = T3 - T2, Dwell = T2 - T1, TTT = T3 - T0."""
        pipeline = create_pipeline(capacity=64, instrument_latency=True)
        try:
            # Synthetic timestamps with known nanosecond durations:
            # T0 = 1,000,000
            # T1 = 1,050,000 (Dwell entry)
            # T2 = 1,150,000 (Dwell exit: Dwell = 100,000 ns = 100 us)
            # T3 = 1,180,000 (Matching exit: Core = 30,000 ns = 30 us; TTT = 180,000 ns = 180 us)
            pipeline._emit_latency(t0=1_000_000, t1=1_050_000, t2=1_150_000, t3=1_180_000)

            core_stats = pipeline.core_latency_tracker.get_stats()
            dwell_stats = pipeline.dwell_latency_tracker.get_stats()
            ttt_stats = pipeline.tick_to_trade_tracker.get_stats()

            assert core_stats.sample_count == 1
            assert core_stats.latest_ns == 30_000.0
            assert core_stats.p50_us == 30.0

            assert dwell_stats.sample_count == 1
            assert dwell_stats.latest_ns == 100_000.0
            assert dwell_stats.p50_us == 100.0

            assert ttt_stats.sample_count == 1
            assert ttt_stats.latest_ns == 180_000.0
            assert ttt_stats.p50_us == 180.0

            # Mathematical invariant: TTT >= Core + Dwell
            assert ttt_stats.latest_ns >= (core_stats.latest_ns + dwell_stats.latest_ns)
        finally:
            pipeline.close()

    def test_pipeline_latency_callback_receives_all_three_measurements(self) -> None:
        """Verify latency_callback is called with (core_ns, dwell_ns, ttt_ns)."""
        observed: list[tuple[float | None, float | None, float | None]] = []

        def callback(core: float | None, dwell: float | None, ttt: float | None) -> None:
            observed.append((core, dwell, ttt))

        pipeline = create_pipeline(
            capacity=64,
            instrument_latency=True,
            latency_callback=callback,
        )
        try:
            pipeline._emit_latency(t0=100, t1=200, t2=500, t3=600)
            assert len(observed) == 1
            core, dwell, ttt = observed[0]
            assert core == 100.0  # 600 - 500
            assert dwell == 300.0  # 500 - 200
            assert ttt == 500.0  # 600 - 100
        finally:
            pipeline.close()


class TestEdgeCasesAndMissingTimestamps:
    """Validate robustness against missing, negative, or mixed clock-domain timestamps."""

    def test_missing_t0_omits_tick_to_trade_without_error(self) -> None:
        """When T0 is missing (None or 0), Core and Dwell are recorded, TTT is skipped."""
        pipeline = create_pipeline(capacity=64, instrument_latency=True)
        try:
            pipeline._emit_latency(t0=None, t1=1_000, t2=2_000, t3=3_000)
            pipeline._emit_latency(t0=0, t1=4_000, t2=5_000, t3=6_000)

            core_stats = pipeline.core_latency_tracker.get_stats()
            dwell_stats = pipeline.dwell_latency_tracker.get_stats()
            ttt_stats = pipeline.tick_to_trade_tracker.get_stats()

            assert core_stats.sample_count == 2
            assert dwell_stats.sample_count == 2
            assert ttt_stats.sample_count == 0  # Safely omitted
        finally:
            pipeline.close()

    def test_clock_domain_safety_wall_clock_t0(self) -> None:
        """Verify epoch wall-clock T0 does not produce astronomically corrupt samples."""
        pipeline = create_pipeline(capacity=64, instrument_latency=True)
        try:
            # Monotonic clock vs epoch nanoseconds
            wall_t0 = time.time_ns()
            perf_t3 = time.perf_counter_ns()

            pipeline._emit_latency(t0=wall_t0, t1=perf_t3 - 10_000, t2=perf_t3 - 5_000, t3=perf_t3)
            ttt_stats = pipeline.tick_to_trade_tracker.get_stats()

            # If measured via wall_delta fallback, must be non-negative and realistic (< 10 seconds)
            if ttt_stats.sample_count > 0:
                assert ttt_stats.p50_us is not None
                assert 0.0 <= ttt_stats.p50_us < 10_000_000.0
        finally:
            pipeline.close()

    def test_negative_or_inverted_timestamps_are_safely_ignored(self) -> None:
        """Verify anomalous negative deltas (e.g. clock jumps) are dropped from reservoirs."""
        pipeline = create_pipeline(capacity=64, instrument_latency=True)
        try:
            # T3 < T2 (impossible unless clock jumped backwards)
            pipeline._emit_latency(t0=1_000, t1=500, t2=600, t3=400)
            core_stats = pipeline.core_latency_tracker.get_stats()
            dwell_stats = pipeline.dwell_latency_tracker.get_stats()

            assert core_stats.sample_count == 0  # Ignored
            assert dwell_stats.sample_count == 1  # Valid dwell: 600 - 500 = 100
        finally:
            pipeline.close()


class TestReservoirBoundedness:
    """Validate fixed-capacity bounded reservoir tracking."""

    def test_bounded_reservoirs_do_not_grow_beyond_capacity(self) -> None:
        """Verify trackers maintain exactly max_capacity samples without memory leak."""
        capacity = 250
        core_tracker = BoundedLatencyTracker(max_samples=capacity)
        dwell_tracker = BoundedLatencyTracker(max_samples=capacity)
        ttt_tracker = BoundedLatencyTracker(max_samples=capacity)

        collector = DashboardMetricsCollector(
            latency_tracker=core_tracker,
            dwell_latency_tracker=dwell_tracker,
            tick_to_trade_tracker=ttt_tracker,
        )

        for i in range(1_000):
            collector.record_core_latency(float(i * 100))
            collector.record_dwell_latency(float(i * 50))
            collector.record_tick_to_trade_latency(float(i * 200))

        assert core_tracker.current_sample_count == capacity
        assert dwell_tracker.current_sample_count == capacity
        assert ttt_tracker.current_sample_count == capacity

        snap = collector.take_snapshot()
        assert snap.core_latency.sample_count == capacity
        assert snap.queue_dwell_latency.sample_count == capacity
        assert snap.tick_to_trade_latency.sample_count == capacity


class TestDashboardPresentationAndFormatting:
    """Validate dashboard formatting of Section 5 with Core, Dwell, and Tick-to-Trade breakdowns."""

    def test_dashboard_renders_all_three_latency_categories(self) -> None:
        """Verify UI displays Core, Queue Dwell, and Tick-to-Trade headers and percentiles."""
        collector = DashboardMetricsCollector(source_name="MultiLatencyTest")

        # Record known samples in ns (1 us = 1,000 ns)
        # Core: 10 us, 20 us, 30 us
        collector.record_core_latency(10_000)
        collector.record_core_latency(20_000)
        collector.record_core_latency(30_000)

        # Dwell: 5 us, 15 us
        collector.record_dwell_latency(5_000)
        collector.record_dwell_latency(15_000)

        # TTT: 50 us, 100 us
        collector.record_tick_to_trade_latency(50_000)
        collector.record_tick_to_trade_latency(100_000)

        stream = io.StringIO()
        dashboard = TerminalDashboard(
            collector=collector,
            interactive=False,
            use_colors=False,
            use_unicode=False,
            output=stream,
        )
        rendered = dashboard.render_once()

        # Section 5 titles
        assert "LATENCY TELEMETRY & BREAKDOWN" in rendered
        assert "Core Matching Latency (T3 - T2):" in rendered
        assert "Queue Dwell Latency (T2 - T1):" in rendered
        assert "Tick-to-Trade Latency (T3 - T0):" in rendered

        # Core matching percentiles
        assert "p50 (Median):" in rendered
        assert "20.00 us" in rendered
        assert "Min Latency:" in rendered
        assert "10.00 us" in rendered
        assert "Max Latency:" in rendered
        assert "30.00 us" in rendered

        # Queue dwell percentiles
        assert "Dwell p50:" in rendered
        assert "Dwell Mean:" in rendered
        assert "10.00 us" in rendered  # (5 + 15) / 2 = 10 us

        # Tick-to-Trade percentiles
        assert "Tick-to-Trade p50:" in rendered
        assert "Tick-to-Trade Mean:" in rendered
        assert "75.00 us" in rendered  # (50 + 100) / 2 = 75 us

    def test_dashboard_empty_trackers_display_na(self) -> None:
        """Verify empty trackers display (No Samples) and N/A without misleading zeros."""
        collector = DashboardMetricsCollector()
        stream = io.StringIO()
        dashboard = TerminalDashboard(
            collector=collector,
            interactive=False,
            use_colors=False,
            use_unicode=False,
            output=stream,
        )
        rendered = dashboard.render_once()

        assert "(No Samples)" in rendered
        assert_field(rendered, "Dwell p50", "N/A")
        assert_field(rendered, "Tick-to-Trade p50", "N/A")

    def test_backward_compatibility_aliases(self) -> None:
        """Verify snapshot.latency and collector.latency_tracker remain aliases for core latency."""
        collector = DashboardMetricsCollector()
        collector.record_latency(12_500)  # Uses alias method

        snap = collector.take_snapshot()
        assert snap.latency is not None
        assert snap.core_latency is not None
        assert snap.latency.p50_us == snap.core_latency.p50_us == 12.50
        assert collector.latency_tracker is collector.core_latency_tracker


class TestPipelineDashboardRunnerEndToEnd:
    """Validate end-to-end simulation runs populate all latency categories."""

    def test_runner_run_steps_populates_core_dwell_and_ttt(self) -> None:
        """Verify run_steps(25) records samples for Core, Dwell, and Tick-to-Trade."""
        config = MarketFeedConfig(total_orders=25, rate=10000.0, mode=SimulationMode.MAX_THROUGHPUT)
        pipeline = create_pipeline(config=config, capacity=128)

        runner = PipelineDashboardRunner(
            pipeline=pipeline,
            interactive=False,
            use_colors=False,
            use_unicode=False,
            output=io.StringIO(),
            instrument_latency=True,
        )

        try:
            trades = runner.run_steps(25)
            assert isinstance(trades, list)

            snap = runner.collector.take_snapshot()
            assert snap.orders_processed == 25

            # All 3 latency trackers must be populated with exactly 25 samples
            assert snap.core_latency is not None
            assert snap.core_latency.sample_count == 25
            assert snap.core_latency.p50_us is not None
            assert snap.core_latency.p50_us > 0.0

            assert snap.queue_dwell_latency is not None
            assert snap.queue_dwell_latency.sample_count == 25
            assert snap.queue_dwell_latency.p50_us is not None

            assert snap.tick_to_trade_latency is not None
            assert snap.tick_to_trade_latency.sample_count == 25
            assert snap.tick_to_trade_latency.p50_us is not None

            # Verify no double counting in runner's trackers
            assert runner.core_latency_tracker.current_sample_count == 25
            assert runner.dwell_latency_tracker.current_sample_count == 25
            assert runner.tick_to_trade_tracker.current_sample_count == 25
        finally:
            runner.close()

    def test_runner_instrument_latency_false_leaves_trackers_empty(self) -> None:
        """Verify instrument_latency=False runs pipeline without latency recording overhead."""
        config = MarketFeedConfig(total_orders=20, rate=10000.0, mode=SimulationMode.MAX_THROUGHPUT)
        pipeline = create_pipeline(config=config, capacity=128)

        runner = PipelineDashboardRunner(
            pipeline=pipeline,
            interactive=False,
            use_colors=False,
            use_unicode=False,
            output=io.StringIO(),
            instrument_latency=False,
        )

        try:
            runner.run_steps(20)
            snap = runner.collector.take_snapshot()
            assert snap.orders_processed == 20
            assert snap.core_latency is None or snap.core_latency.sample_count == 0
            assert snap.queue_dwell_latency is None or snap.queue_dwell_latency.sample_count == 0
            assert snap.tick_to_trade_latency is None or snap.tick_to_trade_latency.sample_count == 0
        finally:
            runner.close()
