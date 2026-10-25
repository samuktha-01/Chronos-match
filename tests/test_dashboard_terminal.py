"""Unit Test Suite for ChronosMatch Terminal Dashboard Presentation Layer.

Validates:
1. Rendering known metric values (pipeline, simulator, engine, ring buffer, latency).
2. Empty and unavailable metrics (strict distinction between None / "N/A" and actual zero values).
3. Configurable refresh behavior (intervals, dynamic updates, on_render callbacks).
4. Graceful shutdown (thread lifecycle, immediate wakeup, idempotence, context manager, cursor restore).
5. Non-interactive and plain-text output (ANSI stripping, ASCII box fallback, StringIO redirection).
"""

from __future__ import annotations

from decimal import Decimal
import io
import re
import threading
import time
from unittest.mock import MagicMock

import pytest

from chronosmatch.dashboard import (
    ASCII_BOX,
    UNICODE_BOX,
    AnsiStyler,
    BoundedLatencyTracker,
    BoxStyle,
    DashboardMetricsCollector,
    DashboardMetricsSnapshot,
    DashboardRenderer,
    LatencyStats,
    TerminalDashboard,
    format_counter_with_alert,
    format_int,
    format_latency,
    format_price,
    format_rate,
    format_time,
    format_utilization_bar,
    format_wall_clock,
    format_wall_time,
    strip_ansi,
    visible_len,
)
from chronosmatch.matching_engine import MatchingEngine
from chronosmatch.models import Order, OrderSide
from chronosmatch.pipeline import create_pipeline
from chronosmatch.ring_buffer import RingBuffer


def assert_field(rendered: str, label: str, value: str) -> None:
    """Verify that a given label and value appear together in the rendered output."""
    pattern = rf"{re.escape(label)}:\s+{re.escape(value)}"
    assert re.search(pattern, rendered) is not None, (
        f"Expected '{label}: {value}' in rendered output, but not found.\nOutput:\n{rendered}"
    )


# ==============================================================================
# 1. Rendering Known Metric Values
# ==============================================================================


class TestRenderingKnownMetrics:
    """Verify that known, populated metrics are accurately rendered into the UI."""

    def test_render_known_values_from_trading_pipeline(self) -> None:
        """Verify dashboard accurately formats telemetry from an active TradingPipeline."""
        pipeline = create_pipeline(capacity=128)
        try:
            # 1. Place resting orders
            pipeline.process_order(
                Order(order_id="BUY-1", side=OrderSide.BUY, price=100.0, quantity=50, timestamp=1)
            )
            pipeline.process_order(
                Order(order_id="SELL-1", side=OrderSide.SELL, price=105.0, quantity=30, timestamp=2)
            )
            # 2. Place crossing aggressive order to trigger a match
            pipeline.process_order(
                Order(order_id="BUY-2", side=OrderSide.BUY, price=105.0, quantity=30, timestamp=3)
            )

            collector = DashboardMetricsCollector(pipeline=pipeline, source_name="PipelineTest")
            collector.record_latency(1_500)  # 1.5 µs

            stream = io.StringIO()
            dashboard = TerminalDashboard(
                collector=collector,
                interactive=False,
                use_colors=False,
                use_unicode=False,
                output=stream,
            )
            out = dashboard.render_once()

            # Pipeline metrics assertions
            assert_field(out, "Orders Processed", "3")
            assert_field(out, "Executed Trades", "1")
            assert_field(out, "Traded Volume", "30")
            assert_field(out, "Buffer Capacity", "128 slots")
            assert_field(out, "Best Bid", "$100.00")
            assert_field(out, "Best Ask", "N/A")  # All resting sells consumed
            assert_field(out, "p50 (Median)", "1.50 us")
            assert "(1 Samples)" in out
        finally:
            pipeline.close()

    def test_render_manual_counters_and_error_states(self) -> None:
        """Verify manually recorded counters, dropped orders, and errors render accurately."""
        collector = DashboardMetricsCollector(source_name="ManualTest")
        collector.record_order_generated(100_000)
        collector.record_order_published(99_500)
        collector.record_order_consumed(99_000)
        collector.record_order_processed(98_500)
        collector.record_order_cancelled(500)
        collector.record_order_dropped(12)
        collector.record_error(3)
        collector.record_trade(150)

        stream = io.StringIO()
        dashboard = TerminalDashboard(
            collector=collector,
            interactive=False,
            use_colors=False,
            use_unicode=False,
            output=stream,
        )
        out = dashboard.render_once()

        assert_field(out, "Orders Generated", "100,000")
        assert_field(out, "Orders Published", "99,500")
        assert_field(out, "Orders Consumed", "99,000")
        assert_field(out, "Orders Processed", "98,500")
        assert_field(out, "Orders Cancelled", "500")
        assert_field(out, "Orders Dropped", "12")
        assert_field(out, "Processing Errors", "3")
        assert_field(out, "Executed Trades", "1")
        assert_field(out, "Traded Volume", "150")

    def test_render_latency_distribution_statistics(self) -> None:
        """Verify latency summary statistics (p50, p95, p99, p99.9, min, mean, max) render correctly."""
        collector = DashboardMetricsCollector(source_name="LatencyTest")
        # Record samples in nanoseconds: 1000ns = 1us, 2000ns = 2us, 5000ns = 5us, 10000ns = 10us
        collector.record_latency(1_000)
        collector.record_latency(2_000)
        collector.record_latency(5_000)
        collector.record_latency(10_000)

        stream = io.StringIO()
        dashboard = TerminalDashboard(
            collector=collector,
            interactive=False,
            use_colors=False,
            use_unicode=False,
            output=stream,
        )
        out = dashboard.render_once()

        assert "(4 Samples)" in out
        assert_field(out, "Min Latency", "1.00 us")
        assert_field(out, "Max Latency", "10.00 us")
        assert_field(out, "Mean Latency", "4.50 us")
        assert_field(out, "Latest Sample", "10.00 us")
        assert "p50 (Median):" in out
        assert "p95:" in out
        assert "p99:" in out
        assert "p99.9:" in out

    def test_render_ring_buffer_utilization_bar(self) -> None:
        """Verify graphical utilization bar formatting at 0%, 50%, and 100% capacity."""
        rb = RingBuffer.create_anonymous(capacity=100)
        try:
            collector = DashboardMetricsCollector(ring_buffer=rb)

            # 0% utilization
            stream = io.StringIO()
            dashboard = TerminalDashboard(
                collector=collector,
                interactive=False,
                use_colors=False,
                use_unicode=False,
                output=stream,
            )
            out = dashboard.render_once()
            assert "[------------------------] 0.00%" in out
            assert_field(out, "Buffer Capacity", "100 slots")
            assert_field(out, "Occupied Slots", "0 slots")

            # With Unicode
            unicode_stream = io.StringIO()
            unicode_dashboard = TerminalDashboard(
                collector=collector,
                interactive=False,
                use_colors=False,
                use_unicode=True,
                output=unicode_stream,
            )
            u_out = unicode_dashboard.render_once()
            assert "[░░░░░░░░░░░░░░░░░░░░░░░░] 0.00%" in u_out
        finally:
            rb.close()


# ==============================================================================
# 2. Empty and Unavailable Metrics
# ==============================================================================


class TestEmptyAndUnavailableMetrics:
    """Verify that uninstrumented or unattached metrics render clearly as N/A without misleading zeros."""

    def test_empty_snapshot_renders_all_unavailable_fields_as_na(self) -> None:
        """Verify an uninstrumented collector renders N/A for all unavailable telemetry."""
        collector = DashboardMetricsCollector(source_name="EmptyCollector")
        stream = io.StringIO()
        dashboard = TerminalDashboard(
            collector=collector,
            interactive=False,
            use_colors=False,
            use_unicode=False,
            output=stream,
        )
        out = dashboard.render_once()

        assert "Source: EmptyCollector" in out
        assert "Status: IDLE" in out
        assert "(No Samples)" in out

        # Must display N/A for all unavailable fields
        assert_field(out, "Elapsed Runtime", "N/A")
        assert_field(out, "Cum. Throughput", "N/A")
        assert_field(out, "Win. Throughput", "N/A")
        assert_field(out, "Orders Generated", "N/A")
        assert_field(out, "Orders Published", "N/A")
        assert_field(out, "Orders Consumed", "N/A")
        assert_field(out, "Orders Processed", "N/A")
        assert_field(out, "Orders Cancelled", "N/A")
        assert_field(out, "Orders Dropped", "N/A")
        assert_field(out, "Executed Trades", "N/A")
        assert_field(out, "Traded Volume", "N/A")
        assert_field(out, "Best Bid", "N/A")
        assert_field(out, "Best Ask", "N/A")
        assert_field(out, "BBO Spread", "N/A")
        assert_field(out, "Buffer Capacity", "N/A")
        assert "Utilization:        N/A" in out
        assert_field(out, "p50 (Median)", "N/A")
        assert_field(out, "Min Latency", "N/A")

        # Strictly verify no misleading zeros appear for unavailable metrics
        assert "0.00 us" not in out
        assert "0.00 µs" not in out
        assert "0.00 ops/s" not in out

    def test_distinguish_explicit_zero_from_unavailable_na(self) -> None:
        """Verify explicit zero values (0 dropped, 0 errors) render as '0', while untracked remain 'N/A'."""
        collector = DashboardMetricsCollector()
        collector.record_order_dropped(0)
        collector.record_order_processed(0)
        collector.record_error(0)

        stream = io.StringIO()
        dashboard = TerminalDashboard(
            collector=collector,
            interactive=False,
            use_colors=False,
            use_unicode=False,
            output=stream,
        )
        out = dashboard.render_once()

        # Explicit zeros MUST be displayed as 0
        assert_field(out, "Orders Dropped", "0")
        assert_field(out, "Orders Processed", "0")
        assert_field(out, "Processing Errors", "0")

        # Untracked metrics MUST remain N/A
        assert_field(out, "Orders Generated", "N/A")
        assert_field(out, "Orders Cancelled", "N/A")
        assert_field(out, "Executed Trades", "N/A")

    def test_partial_engine_attachment_leaves_ring_buffer_na(self) -> None:
        """Verify attaching only a MatchingEngine populates trades/book but leaves ring buffer as N/A."""
        engine = MatchingEngine()
        collector = DashboardMetricsCollector(engine=engine)

        stream = io.StringIO()
        dashboard = TerminalDashboard(
            collector=collector,
            interactive=False,
            use_colors=False,
            use_unicode=False,
            output=stream,
        )
        out = dashboard.render_once()

        # Available from engine
        assert_field(out, "Executed Trades", "0")
        assert_field(out, "Traded Volume", "0")
        assert_field(out, "Book Depth Levels", "0")

        # Unavailable components MUST be N/A
        assert_field(out, "Buffer Capacity", "N/A")
        assert_field(out, "Orders Generated", "N/A")
        assert_field(out, "Orders Dropped", "N/A")


# ==============================================================================
# 3. Refresh Behavior
# ==============================================================================


class TestRefreshBehavior:
    """Verify configurable refresh interval and background refresh loop execution."""

    def test_refresh_interval_configuration_and_validation(self) -> None:
        """Verify default and custom refresh intervals and rejection of non-positive intervals."""
        collector = DashboardMetricsCollector()

        # Default interval is 0.5s
        d1 = TerminalDashboard(collector=collector)
        assert d1.refresh_interval == 0.5

        # Custom interval
        d2 = TerminalDashboard(collector=collector, refresh_interval=0.1)
        assert d2.refresh_interval == 0.1

        # Rejection of invalid intervals
        with pytest.raises(ValueError, match="strictly positive"):
            TerminalDashboard(collector=collector, refresh_interval=0.0)

        with pytest.raises(ValueError, match="strictly positive"):
            TerminalDashboard(collector=collector, refresh_interval=-1.0)

        # Dynamic setter validation
        d1.refresh_interval = 1.0
        assert d1.refresh_interval == 1.0
        with pytest.raises(ValueError, match="strictly positive"):
            d1.refresh_interval = -0.5

    def test_background_refresh_loop_updates_and_invokes_callback(self) -> None:
        """Verify background worker loop triggers periodic renders and on_render callback."""
        collector = DashboardMetricsCollector()
        renders: list[tuple[DashboardMetricsSnapshot, str]] = []

        def callback(snapshot: DashboardMetricsSnapshot, rendered_str: str) -> None:
            renders.append((snapshot, rendered_str))

        stream = io.StringIO()
        dashboard = TerminalDashboard(
            collector=collector,
            refresh_interval=0.05,
            interactive=False,
            use_colors=False,
            output=stream,
            on_render=callback,
        )

        try:
            dashboard.start()
            # Wait for at least 3 refresh intervals
            time.sleep(0.22)
        finally:
            dashboard.stop()

        assert len(renders) >= 2
        assert dashboard.render_count >= 2
        assert dashboard.last_snapshot is not None

    def test_live_metric_updates_reflected_across_refreshes(self) -> None:
        """Verify that metric changes made during background execution are captured in subsequent renders."""
        collector = DashboardMetricsCollector()
        frames: list[str] = []

        dashboard = TerminalDashboard(
            collector=collector,
            refresh_interval=0.05,
            interactive=False,
            use_colors=False,
            output=io.StringIO(),
            on_render=lambda s, r: frames.append(r),
        )

        try:
            dashboard.start()
            collector.record_order_generated(100)
            time.sleep(0.12)
            collector.record_order_generated(200)
            time.sleep(0.12)
        finally:
            dashboard.stop()

        # The later frames must contain the updated order counts
        assert any(re.search(r"Orders Generated:\s+100", f) for f in frames)
        assert any(re.search(r"Orders Generated:\s+300", f) for f in frames)


# ==============================================================================
# 4. Graceful Shutdown
# ==============================================================================


class TestGracefulShutdown:
    """Verify thread lifecycle, fast termination, idempotence, and context management."""

    def test_start_stop_lifecycle_and_thread_termination(self) -> None:
        """Verify background thread begins on start() and ceases promptly on stop()."""
        collector = DashboardMetricsCollector()
        dashboard = TerminalDashboard(
            collector=collector,
            refresh_interval=0.05,
            interactive=False,
            use_colors=False,
            output=io.StringIO(),
        )

        assert not dashboard.is_running
        dashboard.start()
        assert dashboard.is_running

        dashboard.stop(timeout=1.0)
        assert not dashboard.is_running

    def test_immediate_wakeup_on_shutdown(self) -> None:
        """Verify stop() terminates immediately even if configured with a long refresh interval."""
        collector = DashboardMetricsCollector()
        dashboard = TerminalDashboard(
            collector=collector,
            refresh_interval=10.0,  # 10 second refresh interval
            interactive=False,
            use_colors=False,
            output=io.StringIO(),
        )

        dashboard.start()
        t0 = time.perf_counter()
        dashboard.stop(timeout=2.0)
        elapsed = time.perf_counter() - t0

        assert not dashboard.is_running
        # Shutdown must complete within 0.5s, not waiting for the 10s timeout
        assert elapsed < 0.5

    def test_shutdown_idempotence(self) -> None:
        """Verify stop() can be called repeatedly or on an unstarted dashboard without error."""
        collector = DashboardMetricsCollector()
        dashboard = TerminalDashboard(
            collector=collector,
            output=io.StringIO(),
        )

        # Stop unstarted dashboard
        dashboard.stop()
        assert not dashboard.is_running

        # Start, stop, and call stop again
        dashboard.start()
        dashboard.stop()
        dashboard.stop()
        dashboard.stop()
        assert not dashboard.is_running

    def test_double_start_raises_runtime_error(self) -> None:
        """Verify calling start() on an already-running dashboard raises RuntimeError."""
        collector = DashboardMetricsCollector()
        dashboard = TerminalDashboard(
            collector=collector,
            refresh_interval=0.1,
            output=io.StringIO(),
        )

        try:
            dashboard.start()
            with pytest.raises(RuntimeError, match="already running"):
                dashboard.start()
        finally:
            dashboard.stop()

    def test_context_manager_lifecycle(self) -> None:
        """Verify context manager starts and stops dashboard cleanly, even on exceptions."""
        collector = DashboardMetricsCollector()
        dashboard = TerminalDashboard(
            collector=collector,
            refresh_interval=0.05,
            output=io.StringIO(),
        )

        with dashboard as d:
            assert d.is_running

        assert not dashboard.is_running

        # Test context manager cleanup during exception
        with pytest.raises(ZeroDivisionError):
            with dashboard:
                assert dashboard.is_running
                _ = 1 / 0

        assert not dashboard.is_running


# ==============================================================================
# 5. Non-Interactive and Plain-Text Output
# ==============================================================================


class TestNonInteractiveOutput:
    """Verify plain text generation, ANSI stripping, ASCII box mode, and single-shot methods."""

    def test_plain_text_mode_has_no_ansi_codes(self) -> None:
        """Verify use_colors=False produces clean text containing no ANSI escape sequences."""
        collector = DashboardMetricsCollector()
        collector.record_order_generated(10)
        collector.record_order_dropped(2)

        dashboard = TerminalDashboard(
            collector=collector,
            use_colors=False,
            interactive=False,
            output=io.StringIO(),
        )
        rendered = dashboard.render_to_string()

        # Regular expression for ANSI escape sequences
        ansi_regex = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
        assert not ansi_regex.search(rendered)

    def test_pure_ascii_box_mode(self) -> None:
        """Verify use_unicode=False renders standard ASCII characters and 'us' for microseconds."""
        collector = DashboardMetricsCollector()
        collector.record_latency(2_500)

        dashboard = TerminalDashboard(
            collector=collector,
            use_colors=False,
            use_unicode=False,
            output=io.StringIO(),
        )
        rendered = dashboard.render_to_string()

        # Must contain ASCII borders
        assert "+---" in rendered
        assert "|   " in rendered
        assert "2.50 us" in rendered

        # Must encode cleanly as pure 7-bit ASCII
        encoded = rendered.encode("ascii")
        assert len(encoded) > 0

    def test_single_shot_methods_do_not_start_thread(self) -> None:
        """Verify render_once, render_snapshot, and render_to_string execute synchronously without background tasks."""
        collector = DashboardMetricsCollector()
        stream = io.StringIO()
        dashboard = TerminalDashboard(
            collector=collector,
            interactive=False,
            use_colors=False,
            output=stream,
        )

        assert not dashboard.is_running

        s1 = dashboard.render_to_string()
        assert not dashboard.is_running
        assert len(s1) > 0

        s2 = dashboard.render_once()
        assert not dashboard.is_running
        assert len(s2) > 0
        assert dashboard.render_count == 1

        snap = collector.take_snapshot()
        s3 = dashboard.render_snapshot(snap)
        assert not dashboard.is_running
        assert len(s3) > 0
        assert dashboard.render_count == 2

    def test_rendered_lines_have_exact_uniform_column_width(self) -> None:
        """Verify every line produced by DashboardRenderer has exactly the configured width."""
        collector = DashboardMetricsCollector(source_name="WidthTestEngine")
        collector.start()
        collector.record_order_generated(1_000_000)
        collector.record_order_processed(999_000)
        collector.record_trade(500)
        collector.record_latency(1_250)

        for target_width in (76, 80, 90, 100):
            renderer = DashboardRenderer(
                use_colors=False,
                use_unicode=False,
                width=target_width,
            )
            snapshot = collector.take_snapshot()
            rendered = renderer.render(snapshot)
            lines = rendered.strip().split("\n")

            assert len(lines) == 40
            for line_idx, line in enumerate(lines):
                assert len(line) == target_width, (
                    f"Line {line_idx} width is {len(line)}, expected {target_width}: '{line}'"
                )


# ==============================================================================
# 6. Formatters & Component Unit Tests
# ==============================================================================


class TestFormattersAndUtilities:
    """Verify discrete formatting functions handle None, zero, and boundary values accurately."""

    def test_format_int(self) -> None:
        styler = AnsiStyler(enabled=False)
        assert format_int(None, styler) == "N/A"
        assert format_int(0, styler) == "0"
        assert format_int(1_234_567, styler) == "1,234,567"

    def test_format_rate(self) -> None:
        styler = AnsiStyler(enabled=False)
        assert format_rate(None, styler) == "N/A"
        assert format_rate(0.0, styler) == "0.00 ops/s"
        assert format_rate(14_250.5, styler) == "14,250.50 ops/s"

    def test_format_time(self) -> None:
        styler = AnsiStyler(enabled=False)
        assert format_time(None, styler) == "N/A"
        assert format_time(0.0, styler) == "00:00.000"
        assert format_time(65.432, styler) == "01:05.432"
        assert format_time(3665.123, styler) == "01:01:05.123"

    def test_format_latency(self) -> None:
        styler = AnsiStyler(enabled=False)
        assert format_latency(None, styler) == "N/A"
        assert format_latency(1.25, styler, "µs") == "1.25 µs"
        assert format_latency(1.25, styler, "us") == "1.25 us"

    def test_format_price(self) -> None:
        styler = AnsiStyler(enabled=False)
        assert format_price(None, styler) == "N/A"
        assert format_price(100.25, styler) == "$100.25"
        assert format_price(Decimal("50.5"), styler) == "$50.50"

    def test_format_counter_with_alert(self) -> None:
        styler = AnsiStyler(enabled=False)
        assert format_counter_with_alert(None, styler) == "N/A"
        assert format_counter_with_alert(0, styler) == "0"
        assert format_counter_with_alert(5, styler) == "5"

    def test_format_utilization_bar(self) -> None:
        styler = AnsiStyler(enabled=False)
        assert format_utilization_bar(None, 10, ASCII_BOX, styler) == "N/A"
        assert format_utilization_bar(0.0, 10, ASCII_BOX, styler) == "[----------] 0.00%"
        assert format_utilization_bar(0.5, 10, ASCII_BOX, styler) == "[#####-----] 50.00%"
        assert format_utilization_bar(1.0, 10, ASCII_BOX, styler) == "[##########] 100.00%"

    def test_ansi_styler_enabled_vs_disabled(self) -> None:
        enabled = AnsiStyler(enabled=True)
        disabled = AnsiStyler(enabled=False)

        assert "\033[" in enabled.cyan("hello")
        assert disabled.cyan("hello") == "hello"
        assert strip_ansi(enabled.bold("test")) == "test"
        assert visible_len(enabled.green("text")) == 4


# ==============================================================================
# 7. In-Place Rendering & Non-Interactive Regression Tests
# ==============================================================================


class TestInPlaceRenderingAndNonInteractiveOutput:
    """Regression tests verifying single display in-place updating and non-interactive output safety."""

    def test_interactive_repeated_renders_update_in_place_with_clear(self) -> None:
        """Verify repeated renders in clear_screen mode use \033[H\033[J to cleanly overwrite display in place."""
        collector = DashboardMetricsCollector()
        stream = io.StringIO()
        dashboard = TerminalDashboard(
            collector=collector,
            interactive=True,
            clear_screen=True,
            use_colors=False,
            output=stream,
        )

        snap1 = collector.take_snapshot()
        dashboard.render_snapshot(snap1, clear=True)
        out1 = stream.getvalue()
        # First frame clears screen and homes cursor
        assert out1.startswith("\033[2J\033[H")
        assert dashboard.render_count == 1

        # Second frame MUST use \033[H\033[J to home cursor AND clear entire display
        stream.seek(0)
        stream.truncate(0)
        snap2 = collector.take_snapshot()
        dashboard.render_snapshot(snap2, clear=True)
        out2 = stream.getvalue()
        assert out2.startswith("\033[H\033[J")
        assert dashboard.render_count == 2

        # Third frame continues updating in place using \033[H\033[J
        stream.seek(0)
        stream.truncate(0)
        snap3 = collector.take_snapshot()
        dashboard.render_snapshot(snap3, clear=True)
        out3 = stream.getvalue()
        assert out3.startswith("\033[H\033[J")
        assert dashboard.render_count == 3

    def test_interactive_inline_repeated_renders_rewind_cursor(self) -> None:
        """Verify repeated renders in inline mode (clear_screen=False) rewind cursor up by exact line count."""
        collector = DashboardMetricsCollector()
        stream = io.StringIO()
        dashboard = TerminalDashboard(
            collector=collector,
            interactive=True,
            clear_screen=False,
            use_colors=False,
            output=stream,
        )

        snap1 = collector.take_snapshot()
        dashboard.render_snapshot(snap1, clear=False)
        out1 = stream.getvalue()
        # First frame starts without prefix
        assert not out1.startswith("\033[")
        lines_count = len(out1.splitlines())
        assert lines_count == 40

        # Second frame MUST rewind cursor up by 40 lines and clear down (\r\033[40A\033[J)
        stream.seek(0)
        stream.truncate(0)
        snap2 = collector.take_snapshot()
        dashboard.render_snapshot(snap2, clear=False)
        out2 = stream.getvalue()
        expected_rewind = f"\r\033[{lines_count}A\033[J"
        assert out2.startswith(expected_rewind)
        assert dashboard.render_count == 2

    def test_non_interactive_background_loop_does_not_flood_output(self) -> None:
        """Verify non-interactive background refresh thread does not repeatedly dump full screens to output."""
        collector = DashboardMetricsCollector()
        stream = io.StringIO()
        dashboard = TerminalDashboard(
            collector=collector,
            refresh_interval=0.02,
            interactive=False,
            use_colors=False,
            output=stream,
        )

        try:
            dashboard.start()
            # Wait for at least 4 refresh cycles
            time.sleep(0.12)
        finally:
            dashboard.stop()

        # Render count must have incremented in background
        assert dashboard.render_count >= 3
        # Output stream MUST NOT have been flooded with repeated 33-line full boxes during background loop
        assert stream.getvalue() == ""

        # Explicit render_once() must still cleanly write the complete final dashboard snapshot
        dashboard.render_once()
        final_out = stream.getvalue()
        assert len(final_out) > 0
        assert "CHRONOSMATCH" in final_out
        # Exactly one box should be in final_out
        assert final_out.count("CHRONOSMATCH") == 1

    def test_auto_detect_interactive_mode_from_stream(self) -> None:
        """Verify interactive property auto-detects TTY vs non-TTY stream when interactive=None."""
        collector = DashboardMetricsCollector()

        # Mock interactive TTY stream
        mock_tty = MagicMock()
        mock_tty.isatty.return_value = True
        dash_tty = TerminalDashboard(collector=collector, output=mock_tty)
        assert dash_tty.interactive is True

        # Mock non-interactive pipe stream
        mock_pipe = MagicMock()
        mock_pipe.isatty.return_value = False
        dash_pipe = TerminalDashboard(collector=collector, output=mock_pipe)
        assert dash_pipe.interactive is False

        # Explicit override takes precedence
        dash_forced_int = TerminalDashboard(collector=collector, output=mock_pipe, interactive=True)
        assert dash_forced_int.interactive is True

        dash_forced_non_int = TerminalDashboard(collector=collector, output=mock_tty, interactive=False)
        assert dash_forced_non_int.interactive is False

    def test_cursor_hidden_and_restored_cleanly(self) -> None:
        """Verify cursor hide (\033[?25l) and restore (\033[?25h\033[0m) codes in interactive mode."""
        collector = DashboardMetricsCollector()
        stream = io.StringIO()
        dashboard = TerminalDashboard(
            collector=collector,
            interactive=True,
            use_colors=True,
            output=stream,
        )

        dashboard.start()
        started_out = stream.getvalue()
        assert "\033[?25l" in started_out  # Cursor hidden

        stream.seek(0)
        stream.truncate(0)
        dashboard.stop()
        stopped_out = stream.getvalue()
        assert "\033[?25h\033[0m" in stopped_out  # Cursor restored and style reset

