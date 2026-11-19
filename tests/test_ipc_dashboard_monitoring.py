"""Comprehensive tests for ChronosMatch 100K IPC Throughput Monitoring.

Covers:
- Telemetry metrics capture, isolation, and calculation.
- Target achievement calculation strictly based on published IPC throughput.
- Distinction between N/A (unavailable) and 0 (zero values).
- Bounded 5-second rolling average and sustained target status evaluation.
- Warning badges (producer slow, consumer lagging, buffer occupancy >= 80%).
- Real-time progress bar formatting and rate color coding.
- DashboardRenderer exact column-width alignment and plain-text output.
- CursesRenderer drawing and boundary safety.
- PipelineDashboardRunner integration with real pipeline instrumentation.
"""

from __future__ import annotations

import collections
import io
import math
import time
import unittest
from unittest.mock import MagicMock

from chronosmatch.dashboard.metrics import (
    BoundedLatencyTracker,
    DashboardMetricsCollector,
    DashboardMetricsSnapshot,
    LatencyStats,
)
from chronosmatch.dashboard.terminal import (
    ASCII_BOX,
    UNICODE_BOX,
    AnsiStyler,
    DashboardRenderer,
    TerminalDashboard,
    format_ipc_rate,
    format_pct,
    format_target_progress_bar,
    strip_ansi,
    visible_len,
)
from chronosmatch.dashboard.curses_backend import (
    CursesColorTheme,
    CursesRenderer,
)
from chronosmatch.dashboard.pipeline_runner import (
    PipelineDashboardRunner,
)
from chronosmatch.market_simulator import MarketFeedConfig, SimulationMode
from chronosmatch.pipeline import create_pipeline


class TestIpcMetricsSnapshot(unittest.TestCase):
    """Test DashboardMetricsSnapshot IPC fields, defaults, and serialization."""

    def test_default_ipc_fields(self) -> None:
        snap = DashboardMetricsSnapshot(
            timestamp_ns=1_000_000_000,
            wall_timestamp_ns=1_000_000_000,
            elapsed_seconds=1.0,
            throughput_orders_per_sec=10_000.0,
            window_throughput_orders_per_sec=10_000.0,
            orders_generated=100,
            orders_published=100,
            orders_consumed=100,
            orders_processed=100,
            orders_cancelled=0,
            orders_dropped=0,
            trades_executed=50,
            volume_executed=500,
            ring_buffer_capacity=1024,
            ring_buffer_occupied=0,
            ring_buffer_utilization=0.0,
            malformed_messages=0,
            duplicate_orders=0,
            processing_errors=0,
            backpressure_events=0,
            latency=None,
            best_bid=100.0,
            best_ask=101.0,
            spread=1.0,
        )
        self.assertEqual(snap.ipc_target_ops_per_sec, 100_000.0)
        self.assertIsNone(snap.ipc_target_achieved_pct)
        self.assertIsNone(snap.ipc_current_window_rate)
        self.assertIsNone(snap.ipc_rolling_avg_rate)
        self.assertIsNone(snap.ipc_min_rate)
        self.assertIsNone(snap.ipc_max_rate)
        self.assertIsNone(snap.ipc_sustained_status)
        self.assertIsNone(snap.ring_buffer_occupancy_pct)
        self.assertIsNone(snap.failed_publications)
        self.assertEqual(snap.producer_status, "PROD:OK")
        self.assertEqual(snap.consumer_status, "CONS:OK")
        self.assertEqual(snap.matching_engine_status, "MATCH:OK")
        self.assertIsNone(snap.cpu_percent)
        self.assertIsNone(snap.memory_mb)
        self.assertIsNone(snap.ipc_warning)
        self.assertFalse(snap.ipc_monitoring_enabled)

    def test_to_dict_includes_all_ipc_fields(self) -> None:
        snap = DashboardMetricsSnapshot(
            timestamp_ns=1_000_000_000,
            wall_timestamp_ns=1_000_000_000,
            elapsed_seconds=5.0,
            throughput_orders_per_sec=80_000.0,
            window_throughput_orders_per_sec=85_000.0,
            orders_generated=500_000,
            orders_published=450_000,
            orders_consumed=440_000,
            orders_processed=430_000,
            orders_cancelled=10,
            orders_dropped=5,
            trades_executed=200_000,
            volume_executed=2_000_000,
            ring_buffer_capacity=2048,
            ring_buffer_occupied=200,
            ring_buffer_utilization=200 / 2048,
            malformed_messages=0,
            duplicate_orders=0,
            processing_errors=1,
            backpressure_events=3,
            latency=None,
            best_bid=150.0,
            best_ask=150.5,
            spread=0.5,
            ipc_target_ops_per_sec=100_000.0,
            ipc_target_achieved_pct=85.0,
            ipc_current_window_rate=85_000.0,
            ipc_rolling_avg_rate=82_000.0,
            ipc_min_rate=78_000.0,
            ipc_max_rate=88_000.0,
            ipc_sustained_status="TARGET UNMET (85.0% of 100K)",
            ring_buffer_occupancy_pct=9.77,
            failed_publications=2,
            producer_status="PROD:OK",
            consumer_status="CONS:OK",
            matching_engine_status="MATCH:OK",
            cpu_percent=25.4,
            memory_mb=128.5,
            ipc_warning="CONSUMER LAGGING",
            ipc_monitoring_enabled=True,
        )
        d = snap.to_dict()
        self.assertEqual(d["ipc_target_ops_per_sec"], 100_000.0)
        self.assertEqual(d["ipc_target_achieved_pct"], 85.0)
        self.assertEqual(d["ipc_current_window_rate"], 85_000.0)
        self.assertEqual(d["ipc_rolling_avg_rate"], 82_000.0)
        self.assertEqual(d["ipc_min_rate"], 78_000.0)
        self.assertEqual(d["ipc_max_rate"], 88_000.0)
        self.assertEqual(d["ipc_sustained_status"], "TARGET UNMET (85.0% of 100K)")
        self.assertEqual(d["ring_buffer_occupancy_pct"], 9.77)
        self.assertEqual(d["failed_publications"], 2)
        self.assertEqual(d["producer_status"], "PROD:OK")
        self.assertEqual(d["consumer_status"], "CONS:OK")
        self.assertEqual(d["matching_engine_status"], "MATCH:OK")
        self.assertEqual(d["cpu_percent"], 25.4)
        self.assertEqual(d["memory_mb"], 128.5)
        self.assertEqual(d["ipc_warning"], "CONSUMER LAGGING")
        self.assertTrue(d["ipc_monitoring_enabled"])


class TestIpcMetricsCollector(unittest.TestCase):
    """Test telemetry calculations and state transitions in DashboardMetricsCollector."""

    def test_enable_ipc_monitoring_and_set_target(self) -> None:
        collector = DashboardMetricsCollector()
        self.assertFalse(collector.ipc_monitoring_enabled)
        collector.enable_ipc_monitoring(True)
        self.assertTrue(collector.ipc_monitoring_enabled)

        collector.set_ipc_target(150_000.0)
        self.assertEqual(collector._ipc_target_ops_per_sec, 150_000.0)

        # Negative or zero target raises ValueError
        with self.assertRaises(ValueError):
            collector.set_ipc_target(0)
        with self.assertRaises(ValueError):
            collector.set_ipc_target(-100)

    def test_record_failed_publication(self) -> None:
        collector = DashboardMetricsCollector()
        self.assertEqual(collector.failed_publications, 0)
        collector.record_failed_publication(1)
        self.assertEqual(collector.failed_publications, 1)
        collector.record_failed_publication(5)
        self.assertEqual(collector.failed_publications, 6)

        with self.assertRaises(ValueError):
            collector.record_failed_publication(-1)

    def test_target_achievement_percentage_isolation(self) -> None:
        """Target achievement must strictly use successfully published orders / target_rate * 100.

        Must NOT confuse generated orders with published orders.
        """
        collector = DashboardMetricsCollector()
        collector.enable_ipc_monitoring(True)
        collector.set_ipc_target(100_000.0)
        collector.start()

        # Simulate 200,000 generated but only 80,000 published
        collector.record_order_generated(200_000)
        collector.record_order_published(80_000)

        # Artificially set elapsed monotonic time for rate calculation
        t0 = collector._start_time_ns
        t1 = t0 + 1_000_000_000  # exactly 1.0 second

        snap = collector.take_snapshot(timestamp_ns=t1)
        self.assertIsNotNone(snap.published_orders_per_sec)
        self.assertAlmostEqual(snap.published_orders_per_sec, 80_000.0, places=1)

        # Target percentage must be 80.0% (80,000 / 100,000 * 100), NOT 200.0%
        self.assertIsNotNone(snap.ipc_target_achieved_pct)
        self.assertAlmostEqual(snap.ipc_target_achieved_pct, 80.0, places=1)

    def test_buffer_occupancy_percentage(self) -> None:
        collector = DashboardMetricsCollector()
        collector.enable_ipc_monitoring(True)
        collector.start()

        collector.set_ring_buffer_state(capacity=1024, occupied=512)
        snap = collector.take_snapshot()
        self.assertIsNotNone(snap.ring_buffer_occupancy_pct)
        self.assertAlmostEqual(snap.ring_buffer_occupancy_pct, 50.0, places=1)

    def test_sustained_vs_burst_status(self) -> None:
        """Verify explicit sustained-target status over the documented 5-second window."""
        collector = DashboardMetricsCollector()
        collector.enable_ipc_monitoring(True)
        collector.set_ipc_target(100_000.0)
        collector.start()

        t = collector._start_time_ns

        # 1. Under target: 50,000 ops/s
        collector.record_order_published(50_000)
        t += 1_000_000_000
        snap1 = collector.take_snapshot(timestamp_ns=t)
        self.assertIn("TARGET UNMET", snap1.ipc_sustained_status or "")

        # 2. Burst above 100K: published rate spikes to 120,000 ops/s in single second
        collector.record_order_published(120_000)
        t += 1_000_000_000
        snap2 = collector.take_snapshot(timestamp_ns=t)
        # Because rolling average is (50k + 120k)/2 = 85k < 100k, it must report BURST ONLY
        self.assertIn("BURST ONLY", snap2.ipc_sustained_status or "")

        # 3. Sustained achievement over documented window (>= 100K rolling average)
        for _ in range(5):
            collector.record_order_published(105_000)
            t += 1_000_000_000
            snap3 = collector.take_snapshot(timestamp_ns=t)

        # After the initial 50k sample rolls off and rolling avg >= 100k:
        self.assertIn("SUSTAINED: ACHIEVED", snap3.ipc_sustained_status or "")

    def test_warning_badges(self) -> None:
        collector = DashboardMetricsCollector()
        collector.enable_ipc_monitoring(True)
        collector.start()

        # 1. High buffer occupancy warning
        collector.set_ring_buffer_state(capacity=1000, occupied=850)
        snap = collector.take_snapshot()
        self.assertIn("buffer occupancy 85.0%", snap.ipc_warning or "")

        # 2. Explicit override warning
        collector.set_ipc_warning("MANUAL BACKPRESSURE ALERT")
        snap = collector.take_snapshot()
        self.assertEqual(snap.ipc_warning, "MANUAL BACKPRESSURE ALERT")

        collector.set_ipc_warning(None)
        # Should clear override


class TestFormatters(unittest.TestCase):
    """Test formatters for progress bar, IPC rates, and percentages."""

    def test_format_target_progress_bar_plain(self) -> None:
        styler = AnsiStyler(enabled=False)
        box = ASCII_BOX

        # None -> N/A
        bar_na = format_target_progress_bar(None, None, 100_000.0, 10, box, styler)
        self.assertIn("N/A", bar_na)
        self.assertIn("[----------]", bar_na)

        # 50%
        bar_50 = format_target_progress_bar(50.0, 50_000.0, 100_000.0, 10, box, styler)
        self.assertIn("[#####-----]", bar_50)
        self.assertIn("50.00%", bar_50)
        self.assertIn("(50,000/100,000 ops/s)", bar_50)

        # 100%
        bar_100 = format_target_progress_bar(100.0, 100_000.0, 100_000.0, 10, box, styler)
        self.assertIn("[##########]", bar_100)
        self.assertIn("100.00%", bar_100)

        # Clamping above 100%
        bar_120 = format_target_progress_bar(120.0, 120_000.0, 100_000.0, 10, box, styler)
        self.assertIn("[##########]", bar_120)
        self.assertIn("120.00%", bar_120)

    def test_format_target_progress_bar_colored(self) -> None:
        styler = AnsiStyler(enabled=True)
        box = UNICODE_BOX

        # Low rate (< 50%) -> Yellow
        bar_low = format_target_progress_bar(25.0, 25_000.0, 100_000.0, 10, box, styler)
        self.assertIn(styler.YELLOW, bar_low)

        # Medium rate (50-99%) -> Cyan
        bar_mid = format_target_progress_bar(75.0, 75_000.0, 100_000.0, 10, box, styler)
        self.assertIn(styler.CYAN, bar_mid)

        # High rate (>= 100%) -> Green + Bold
        bar_high = format_target_progress_bar(100.0, 100_000.0, 100_000.0, 10, box, styler)
        self.assertIn(styler.GREEN, bar_high)
        self.assertIn(styler.BOLD, bar_high)

    def test_format_ipc_rate(self) -> None:
        styler = AnsiStyler(enabled=True)

        self.assertEqual(strip_ansi(format_ipc_rate(None, styler)), "N/A")
        self.assertEqual(strip_ansi(format_ipc_rate(0.0, styler)), "0.0 ops/s")

        # < 50K -> Red
        low = format_ipc_rate(45_000.0, styler, target=100_000.0)
        self.assertIn(styler.RED, low)

        # 50K - 99.9K -> Yellow
        mid = format_ipc_rate(75_000.0, styler, target=100_000.0)
        self.assertIn(styler.YELLOW, mid)

        # >= 100K -> Bold Green
        high = format_ipc_rate(105_000.0, styler, target=100_000.0)
        self.assertIn(styler.GREEN, high)
        self.assertIn(styler.BOLD, high)

    def test_format_pct(self) -> None:
        styler = AnsiStyler(enabled=True)

        self.assertEqual(strip_ansi(format_pct(None, styler)), "N/A")
        low = format_pct(30.0, styler)
        self.assertIn(styler.RED, low)
        mid = format_pct(80.0, styler)
        self.assertIn(styler.YELLOW, mid)
        high = format_pct(100.0, styler)
        self.assertIn(styler.GREEN, high)


class TestDashboardRendererIpc(unittest.TestCase):
    """Test DashboardRenderer layout, column alignment, and exact borders with IPC section."""

    def test_renderer_omits_ipc_section_when_disabled(self) -> None:
        renderer = DashboardRenderer(width=80, show_ipc_section=False)
        snap = DashboardMetricsSnapshot(
            timestamp_ns=1_000_000_000,
            wall_timestamp_ns=1_000_000_000,
            elapsed_seconds=1.0,
            throughput_orders_per_sec=100.0,
            window_throughput_orders_per_sec=100.0,
            orders_generated=100,
            orders_published=100,
            orders_consumed=100,
            orders_processed=100,
            orders_cancelled=0,
            orders_dropped=0,
            trades_executed=10,
            volume_executed=100,
            ring_buffer_capacity=1024,
            ring_buffer_occupied=0,
            ring_buffer_utilization=0.0,
            malformed_messages=0,
            duplicate_orders=0,
            processing_errors=0,
            backpressure_events=0,
            latency=None,
            best_bid=None,
            best_ask=None,
            spread=None,
            ipc_monitoring_enabled=False,
        )
        rendered = renderer.render(snap)
        self.assertNotIn("100K IPC TARGET", rendered)
        lines = rendered.splitlines()
        self.assertEqual(len(lines), 40)

    def test_renderer_includes_ipc_section_when_enabled(self) -> None:
        renderer = DashboardRenderer(width=80, show_ipc_section=True, use_colors=True)
        snap = DashboardMetricsSnapshot(
            timestamp_ns=1_000_000_000,
            wall_timestamp_ns=1_000_000_000,
            elapsed_seconds=5.0,
            throughput_orders_per_sec=95_000.0,
            window_throughput_orders_per_sec=96_000.0,
            orders_generated=500_000,
            orders_published=490_000,
            orders_consumed=480_000,
            orders_processed=470_000,
            orders_cancelled=5,
            orders_dropped=0,
            trades_executed=200_000,
            volume_executed=2_000_000,
            ring_buffer_capacity=4096,
            ring_buffer_occupied=320,
            ring_buffer_utilization=320 / 4096,
            malformed_messages=0,
            duplicate_orders=0,
            processing_errors=0,
            backpressure_events=0,
            latency=None,
            best_bid=100.0,
            best_ask=100.5,
            spread=0.5,
            published_orders_per_sec=98_000.0,
            generated_orders_per_sec=100_000.0,
            consumed_orders_per_sec=97_000.0,
            processed_orders_per_sec=96_000.0,
            ipc_target_ops_per_sec=100_000.0,
            ipc_target_achieved_pct=98.0,
            ipc_current_window_rate=98_000.0,
            ipc_rolling_avg_rate=96_500.0,
            ipc_min_rate=94_000.0,
            ipc_max_rate=99_000.0,
            ipc_sustained_status="TARGET UNMET (98.0% of 100K)",
            ring_buffer_occupancy_pct=7.8,
            failed_publications=0,
            producer_status="PROD:OK",
            consumer_status="CONS:OK",
            matching_engine_status="MATCH:OK",
            cpu_percent=32.0,
            memory_mb=140.2,
            ipc_warning=None,
            ipc_monitoring_enabled=True,
        )
        rendered = renderer.render(snap)
        self.assertIn("100K IPC TARGET & TELEMETRY MONITORING", rendered)
        self.assertIn("Target Progress:", rendered)
        self.assertIn("Sustained Status:", rendered)
        self.assertIn("Required IPC Target", rendered)
        self.assertIn("Published IPC Rate", rendered)
        self.assertIn("5s Rolling Avg Rate", rendered)
        self.assertIn("Component Health", rendered)

        # Every single line must strictly adhere to exactly 80 columns
        for idx, line in enumerate(rendered.splitlines()):
            self.assertEqual(
                visible_len(line),
                80,
                f"Line {idx} width mismatch: '{strip_ansi(line)}' (len={visible_len(line)}, expected 80)",
            )

    def test_renderer_with_warning_badge(self) -> None:
        renderer = DashboardRenderer(width=80, show_ipc_section=True, use_colors=False)
        snap = DashboardMetricsSnapshot(
            timestamp_ns=1_000_000_000,
            wall_timestamp_ns=1_000_000_000,
            elapsed_seconds=2.0,
            throughput_orders_per_sec=10_000.0,
            window_throughput_orders_per_sec=10_000.0,
            orders_generated=50_000,
            orders_published=40_000,
            orders_consumed=30_000,
            orders_processed=30_000,
            orders_cancelled=0,
            orders_dropped=0,
            trades_executed=10_000,
            volume_executed=100_000,
            ring_buffer_capacity=1024,
            ring_buffer_occupied=850,
            ring_buffer_utilization=0.85,
            malformed_messages=0,
            duplicate_orders=0,
            processing_errors=0,
            backpressure_events=5,
            latency=None,
            best_bid=None,
            best_ask=None,
            spread=None,
            ipc_target_ops_per_sec=100_000.0,
            ipc_target_achieved_pct=40.0,
            ipc_warning="BUFFER OCCUPANCY >= 80%",
            ipc_monitoring_enabled=True,
        )
        rendered = renderer.render(snap)
        self.assertIn("(! BUFFER OCCUPANCY >= 80% !)", rendered)
        for idx, line in enumerate(rendered.splitlines()):
            self.assertEqual(visible_len(line), 80, f"Line {idx} width mismatch")


class TestCursesRendererIpc(unittest.TestCase):
    """Test CursesRenderer drawing methods for the IPC section."""

    def test_curses_draw_ipc_section(self) -> None:
        theme = CursesColorTheme()
        renderer = CursesRenderer(theme=theme, show_ipc_section=True)

        # Mock curses window
        win = MagicMock()
        win.getmaxyx.return_value = (50, 100)

        snap = DashboardMetricsSnapshot(
            timestamp_ns=1_000_000_000,
            wall_timestamp_ns=1_000_000_000,
            elapsed_seconds=10.0,
            throughput_orders_per_sec=100_000.0,
            window_throughput_orders_per_sec=100_000.0,
            orders_generated=1_000_000,
            orders_published=1_000_000,
            orders_consumed=1_000_000,
            orders_processed=1_000_000,
            orders_cancelled=0,
            orders_dropped=0,
            trades_executed=500_000,
            volume_executed=5_000_000,
            ring_buffer_capacity=4096,
            ring_buffer_occupied=100,
            ring_buffer_utilization=100 / 4096,
            malformed_messages=0,
            duplicate_orders=0,
            processing_errors=0,
            backpressure_events=0,
            latency=None,
            best_bid=100.0,
            best_ask=100.1,
            spread=0.1,
            published_orders_per_sec=100_000.0,
            generated_orders_per_sec=100_000.0,
            consumed_orders_per_sec=100_000.0,
            processed_orders_per_sec=100_000.0,
            ipc_target_ops_per_sec=100_000.0,
            ipc_target_achieved_pct=100.0,
            ipc_current_window_rate=100_000.0,
            ipc_rolling_avg_rate=100_000.0,
            ipc_sustained_status="SUSTAINED: ACHIEVED (>=100K)",
            ring_buffer_occupancy_pct=2.4,
            failed_publications=0,
            producer_status="PROD:OK",
            consumer_status="CONS:OK",
            matching_engine_status="MATCH:OK",
            cpu_percent=45.0,
            memory_mb=150.0,
            ipc_warning=None,
            ipc_monitoring_enabled=True,
        )

        # Renders without raising any exception
        renderer.render(win, snap)
        self.assertTrue(win.addstr.called or win.addnstr.called)


class TestPipelineDashboardRunnerIpcIntegration(unittest.TestCase):
    """Test PipelineDashboardRunner integration with real TradingPipeline."""

    def test_runner_initializes_ipc_monitoring(self) -> None:
        config = MarketFeedConfig(
            total_orders=100,
            rate=5000.0,
            mode=SimulationMode.UNIFORM,
        )
        pipeline = create_pipeline(config=config, capacity=512)

        buf = io.StringIO()
        runner = PipelineDashboardRunner(
            pipeline=pipeline,
            enable_ipc_monitoring=True,
            ipc_target_ops_per_sec=100_000.0,
            output=buf,
            interactive=False,
        )

        self.assertTrue(runner.collector.ipc_monitoring_enabled)
        self.assertEqual(runner.collector._ipc_target_ops_per_sec, 100_000.0)

        # Run pipeline
        try:
            metrics = runner.run(total_orders=100)
            self.assertEqual(metrics.orders_processed, 100)
            snap = runner.collector.take_snapshot()
            self.assertTrue(snap.ipc_monitoring_enabled)
            self.assertEqual(snap.ipc_target_ops_per_sec, 100_000.0)
            self.assertIsNotNone(snap.published_orders_per_sec)
        finally:
            runner.close()


if __name__ == "__main__":
    unittest.main()
