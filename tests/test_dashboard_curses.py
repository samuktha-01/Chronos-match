"""Unit and Integration Test Suite for ChronosMatch Curses Terminal Dashboard Backend.

Validates:
1. Backend availability and selection (ANSI default on Windows, CURSES optional, AUTO detection).
2. Unsupported terminal handling (non-TTY, missing curses, dumb TERM, clean CursesUnsupportedError).
3. Comprehensive metrics rendering on Curses windows:
   - Best bid, best ask, and bid-ask spread.
   - Order count, trade count, and executed volume.
   - Throughput (cumulative and sliding-window).
   - Microsecond latency distributions (p50, p95, p99, min, mean, max).
   - Ring buffer capacity and capacity utilization gauge.
   - System diagnostics (malformed records, duplicates, backpressure).
   - Whale order alerts and multi-level price sweep highlights.
4. Terminal resizing and dynamic dimension adaptation.
5. Insufficient dimension protection (< 68x16 warning box, zero crashes, safe text clipping).
6. Non-blocking keyboard exit ('q', 'Q', 'ESC') and responsive polling.
7. Graceful Ctrl+C and stop() shutdown with terminal state restoration.
8. Backward compatibility for the ANSI backend and existing pipeline runners.
"""

try:
    import curses
except Exception:
    curses = None  # type: ignore[assignment]

import io
import os
import sys
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from chronosmatch.dashboard import (
    BoundedLatencyTracker,
    DashboardMetricsCollector,
    DashboardMetricsSnapshot,
    DashboardRenderer,
    LatencyStats,
    PipelineDashboardRunner,
    TerminalDashboard,
)
from chronosmatch.dashboard.curses_backend import (
    CursesColorTheme,
    CursesDashboard,
    CursesRenderer,
    CursesUnsupportedError,
    DashboardBackend,
    can_initialize_curses,
    create_dashboard,
    is_curses_available,
    resolve_backend,
    safe_addstr,
)
from chronosmatch.models import Order, OrderSide
from chronosmatch.pipeline import create_pipeline


# ==============================================================================
# Mock Curses Window for Deterministic Headless Testing
# ==============================================================================


class MockCursesWindow:
    """Mock curses window simulating a terminal screen buffer for deterministic unit testing."""

    def __init__(self, height: int = 24, width: int = 80) -> None:
        self.height = height
        self.width = width
        self.buffer: dict[tuple[int, int], str] = {}
        self.chars_written: list[tuple[int, int, str, int]] = []
        self.key_queue: list[int] = []
        self.erased: bool = False
        self.cleared: bool = False
        self.refreshed: int = 0
        self.nodelay_val: bool = False
        self.timeout_val: int = -1

    def getmaxyx(self) -> tuple[int, int]:
        return (self.height, self.width)

    def erase(self) -> None:
        self.buffer.clear()
        self.erased = True

    def clear(self) -> None:
        self.buffer.clear()
        self.cleared = True

    def refresh(self) -> None:
        self.refreshed += 1

    def noutrefresh(self) -> None:
        self.refreshed += 1

    def nodelay(self, val: bool) -> None:
        self.nodelay_val = val

    def timeout(self, ms: int) -> None:
        self.timeout_val = ms

    def keypad(self, flag: bool) -> None:
        pass

    def addstr(self, y: int, x: int, text: str, attr: int = 0) -> None:
        self.chars_written.append((y, x, text, attr))
        for idx, ch in enumerate(text):
            if 0 <= y < self.height and 0 <= (x + idx) < self.width:
                self.buffer[(y, x + idx)] = ch

    def addnstr(self, y: int, x: int, text: str, n: int, attr: int = 0) -> None:
        self.addstr(y, x, text[:n], attr)

    def getch(self) -> int:
        if self.key_queue:
            return self.key_queue.pop(0)
        return -1

    def get_line(self, y: int) -> str:
        """Reconstruct visible string for a given row index."""
        chars = [self.buffer.get((y, x), " ") for x in range(self.width)]
        return "".join(chars).rstrip()

    def get_all_text(self) -> str:
        """Reconstruct entire buffer content as a multi-line string."""
        lines = [self.get_line(y) for y in range(self.height)]
        return "\n".join(lines)


# ==============================================================================
# 1. Backend Selection & Environment Detection
# ==============================================================================


class TestBackendSelection:
    """Verify backend resolution rules across platforms and terminal capabilities."""

    def test_default_on_windows_is_ansi(self) -> None:
        """On Windows, resolve_backend() must return ANSI by default."""
        with patch("os.name", "nt"):
            assert resolve_backend(None) == DashboardBackend.ANSI
            assert resolve_backend(DashboardBackend.AUTO) == DashboardBackend.ANSI
            assert resolve_backend("auto") == DashboardBackend.ANSI

    def test_explicit_ansi_always_returns_ansi(self) -> None:
        """Explicit ANSI selection returns ANSI on any platform."""
        assert resolve_backend(DashboardBackend.ANSI) == DashboardBackend.ANSI
        assert resolve_backend("ansi") == DashboardBackend.ANSI

    def test_explicit_curses_when_supported(self) -> None:
        """Explicit CURSES returns CURSES when terminal supports it."""
        with patch("chronosmatch.dashboard.curses_backend.can_initialize_curses", return_value=True):
            assert resolve_backend(DashboardBackend.CURSES) == DashboardBackend.CURSES
            assert resolve_backend("curses") == DashboardBackend.CURSES

    def test_explicit_curses_fallback_to_ansi(self) -> None:
        """Explicit CURSES with fallback_to_ansi=True falls back to ANSI when unsupported."""
        with patch("chronosmatch.dashboard.curses_backend.can_initialize_curses", return_value=False):
            assert resolve_backend("curses", fallback_to_ansi=True) == DashboardBackend.ANSI

    def test_explicit_curses_strict_raises_error(self) -> None:
        """Explicit CURSES with fallback_to_ansi=False raises CursesUnsupportedError when unsupported."""
        with patch("chronosmatch.dashboard.curses_backend.can_initialize_curses", return_value=False):
            with pytest.raises(CursesUnsupportedError, match="Curses backend is not supported"):
                resolve_backend("curses", fallback_to_ansi=False)

    def test_auto_on_posix_with_tty_selects_curses(self) -> None:
        """On POSIX systems with an interactive TTY, AUTO selects CURSES."""
        with patch("os.name", "posix"):
            with patch("chronosmatch.dashboard.curses_backend.can_initialize_curses", return_value=True):
                assert resolve_backend(DashboardBackend.AUTO) == DashboardBackend.CURSES

    def test_auto_on_posix_without_tty_selects_ansi(self) -> None:
        """On POSIX systems without a TTY, AUTO selects ANSI."""
        with patch("os.name", "posix"):
            with patch("chronosmatch.dashboard.curses_backend.can_initialize_curses", return_value=False):
                assert resolve_backend(DashboardBackend.AUTO) == DashboardBackend.ANSI

    def test_create_dashboard_factory_instantiates_correct_backend(self) -> None:
        """create_dashboard factory returns CursesDashboard or TerminalDashboard as appropriate."""
        collector = DashboardMetricsCollector()
        with patch("chronosmatch.dashboard.curses_backend.can_initialize_curses", return_value=True):
            dash_curses = create_dashboard(collector, backend=DashboardBackend.CURSES)
            assert isinstance(dash_curses, CursesDashboard)
            assert dash_curses.backend == DashboardBackend.CURSES

        dash_ansi = create_dashboard(collector, backend=DashboardBackend.ANSI)
        assert isinstance(dash_ansi, TerminalDashboard)
        assert dash_ansi.backend == DashboardBackend.ANSI


# ==============================================================================
# 2. Unsupported Terminal Handling
# ==============================================================================


class TestUnsupportedTerminalHandling:
    """Verify that unsupported terminal environments are handled gracefully."""

    def test_non_tty_stream_cannot_initialize_curses(self) -> None:
        """A non-TTY StringIO stream cannot initialize curses."""
        stream = io.StringIO()
        assert not can_initialize_curses(stream)

    def test_missing_curses_module(self) -> None:
        """When curses module is not importable, is_curses_available() returns False."""
        with patch("chronosmatch.dashboard.curses_backend.curses", None):
            assert not is_curses_available()
            assert not can_initialize_curses()

    def test_posix_dumb_term_cannot_initialize(self) -> None:
        """On POSIX, TERM='dumb' or empty TERM cannot initialize curses."""
        fake_stream = MagicMock()
        fake_stream.isatty.return_value = True
        fake_stdin = MagicMock()
        fake_stdin.isatty.return_value = True

        with patch("os.name", "posix"):
            with patch.object(sys, "stdin", fake_stdin):
                with patch.dict(os.environ, {"TERM": "dumb"}):
                    assert not can_initialize_curses(fake_stream)
                with patch.dict(os.environ, {"TERM": ""}):
                    assert not can_initialize_curses(fake_stream)

    def test_curses_dashboard_start_raises_when_unsupported(self) -> None:
        """CursesDashboard.start() raises CursesUnsupportedError when environment is unsupported."""
        collector = DashboardMetricsCollector()
        dashboard = CursesDashboard(collector, stream=io.StringIO())
        with pytest.raises(CursesUnsupportedError, match="requires interactive terminal TTY"):
            dashboard.start()


# ==============================================================================
# 3. Comprehensive Metrics Rendering on Curses Windows
# ==============================================================================


class TestCursesRendererMetricsDisplay:
    """Verify all required telemetry fields are accurately drawn on a Curses screen."""

    @pytest.fixture
    def populated_snapshot(self) -> DashboardMetricsSnapshot:
        """Create a fully-populated DashboardMetricsSnapshot for rendering tests."""
        lat = LatencyStats(
            sample_count=1000,
            min_ns=1_000.0,
            max_ns=50_000.0,
            mean_ns=4_500.0,
            p50_ns=2_500.0,
            p95_ns=8_000.0,
            p99_ns=15_000.0,
            p99_9_ns=30_000.0,
            latest_ns=2_600.0,
        )
        return DashboardMetricsSnapshot(
            timestamp_ns=1_000_000_000,
            wall_timestamp_ns=1_700_000_000_000_000_000,
            elapsed_seconds=12.5,
            throughput_orders_per_sec=15_250.75,
            window_throughput_orders_per_sec=14_800.50,
            orders_generated=5000,
            orders_published=4950,
            orders_consumed=4900,
            orders_processed=4850,
            orders_cancelled=50,
            orders_dropped=2,
            trades_executed=250,
            volume_executed=12_500,
            ring_buffer_capacity=1024,
            ring_buffer_occupied=64,
            ring_buffer_utilization=0.0625,
            malformed_messages=0,
            duplicate_orders=1,
            processing_errors=1,
            backpressure_events=0,
            latency=lat,
            core_latency=lat,
            queue_dwell_latency=lat,
            tick_to_trade_latency=lat,
            best_bid=100.50,
            best_ask=101.25,
            spread=0.75,
            order_book_depth_levels=10,
            is_cython_accelerated=True,
            source_name="ChronosMatch-Test",
            whale_orders_detected=1,
            whale_volume_executed=2500,
            multi_level_sweeps=1,
            whale_order_threshold=1000,
            last_whale_order={
                "order_id": "WHALE-101",
                "side": "BUY",
                "price": 101.25,
                "quantity": 2500,
                "swept_levels": 2,
            },
        )

    def test_render_all_core_metrics(self, populated_snapshot: DashboardMetricsSnapshot) -> None:
        """Verify presence of best bid, ask, spread, orders, trades, throughput, and latency."""
        win = MockCursesWindow(height=35, width=80)
        renderer = CursesRenderer()
        renderer.render(win, populated_snapshot, refresh_interval=0.5)

        full_text = win.get_all_text()

        # 1. Header & Source
        assert "CHRONOSMATCH" in full_text
        assert "ChronosMatch-T" in full_text
        assert "Cython" in full_text
        assert "ACTIVE" in full_text

        # 2. System Runtime & Throughput
        assert "Elapsed Runtime:" in full_text
        assert "15,250.75 ops/s" in full_text
        assert "14,800.50 ops/s" in full_text

        # 3. Order Flow
        assert "Orders Processed:" in full_text
        assert "4,850" in full_text
        assert "Orders Dropped:" in full_text
        assert "2" in full_text

        # 4. Trades & Market State
        assert "Executed Trades:" in full_text
        assert "250" in full_text
        assert "Traded Volume:" in full_text
        assert "12,500" in full_text
        assert "Best Bid:" in full_text
        assert "$100.50" in full_text
        assert "Best Ask:" in full_text
        assert "$101.25" in full_text
        assert "BBO Spread:" in full_text
        assert "$0.75" in full_text

        # 5. Ring Buffer
        assert "Buffer Capacity:" in full_text
        assert "1,024 slots" in full_text
        assert "Occupied Slots:" in full_text
        assert "64 slots" in full_text

        # 6. Latency Metrics (p50, p95, p99)
        assert "p50 (Median):" in full_text
        assert "2.50 us" in full_text
        assert "p95:" in full_text
        assert "8.00 us" in full_text
        assert "p99:" in full_text
        assert "15.00 us" in full_text
        assert "Min Latency:" in full_text
        assert "1.00 us" in full_text
        assert "Mean Latency:" in full_text
        assert "4.50 us" in full_text
        assert "Max Latency:" in full_text
        assert "50.00 us" in full_text

        # 7. Whale Order & Multi-Level Sweep Telemetry
        assert "[WHALE ORDER & SWEEP TELEMETRY]" in full_text
        assert "(! 1 DETECTED !)" in full_text
        assert "Whale Threshold:" in full_text
        assert ">= 1,000" in full_text
        assert "Whale Volume:" in full_text
        assert "2,500" in full_text
        assert "Last Whale Order:   BUY 2,500 @ $101.25 (WHALE-101)" in full_text
        assert ">>> MULTI-LEVEL SWEEP: Executed across 2 price levels! <<<" in full_text

        # 8. Hotkey Footer
        assert "[q] Quit" in full_text
        assert "[r] Refresh" in full_text

    def test_render_to_string(self, populated_snapshot: DashboardMetricsSnapshot) -> None:
        """render_to_string() produces a formatted representation containing key metrics."""
        renderer = CursesRenderer()
        text = renderer.render_to_string(populated_snapshot)
        assert "CHRONOSMATCH" in text
        assert "Orders Processed" in text
        assert "Executed Trades" in text


# ==============================================================================
# 4. Terminal Resizing & Insufficient Dimensions
# ==============================================================================


class TestCursesResizingAndDimensionProtection:
    """Verify clean dynamic resizing and boundary protection for small screens."""

    def test_insufficient_dimensions_renders_warning_box(self) -> None:
        """When terminal dimensions are smaller than 68x16, renders clean warning box."""
        win = MockCursesWindow(height=12, width=50)
        renderer = CursesRenderer()
        collector = DashboardMetricsCollector()
        renderer.render(win, collector.take_snapshot())

        text = win.get_all_text()
        assert "TERMINAL WINDOW TOO SMALL" in text
        assert "Size: 50x12" in text
        assert "Required: 68x16" in text
        assert "Press 'q' to quit" in text

    def test_insufficient_dimensions_tiny_window(self) -> None:
        """Even on a tiny 5x10 window, render() completes without raising exceptions."""
        win = MockCursesWindow(height=5, width=10)
        renderer = CursesRenderer()
        collector = DashboardMetricsCollector()
        renderer.render(win, collector.take_snapshot())
        text = win.get_all_text()
        assert "TERMINAL" in text

    def test_terminal_resizing_from_small_to_normal(self) -> None:
        """Transitioning from too-small to full-size terminal redraws full dashboard."""
        win = MockCursesWindow(height=10, width=50)
        renderer = CursesRenderer()
        collector = DashboardMetricsCollector()
        renderer.render(win, collector.take_snapshot())
        assert "TERMINAL WINDOW TOO SMALL" in win.get_all_text()

        # Simulate resize
        win.height = 30
        win.width = 85
        renderer.render(win, collector.take_snapshot())
        new_text = win.get_all_text()
        assert "TERMINAL WINDOW TOO SMALL" not in new_text
        assert "CHRONOSMATCH" in new_text

    def test_safe_addstr_boundary_clipping(self) -> None:
        """safe_addstr handles out-of-bounds coordinates and clipping without errors."""
        win = MockCursesWindow(height=10, width=20)
        # Should cleanly ignore negative or out-of-bounds coordinates
        safe_addstr(win, -1, 0, "test")
        safe_addstr(win, 10, 0, "test")
        safe_addstr(win, 0, -1, "test")
        safe_addstr(win, 0, 20, "test")
        safe_addstr(win, 0, 0, "safe string")
        assert "safe string" in win.get_line(0)

        # Bottom-right corner protection: string truncated so it doesn't touch bottom-right cell
        safe_addstr(win, 9, 0, "A" * 30)
        assert len(win.get_line(9)) <= 19


# ==============================================================================
# 5. Keyboard Exit & Graceful Shutdown
# ==============================================================================


class TestCursesKeyboardExitAndShutdown:
    """Verify non-blocking keyboard polling and graceful terminal restoration."""

    def test_keyboard_exit_on_q(self) -> None:
        """Pressing 'q' sets the stop event and triggers on_exit callback."""
        exit_called = threading.Event()
        collector = DashboardMetricsCollector()
        dashboard = CursesDashboard(
            collector,
            refresh_interval=0.1,
            on_exit=exit_called.set,
        )

        mock_win = MockCursesWindow(height=24, width=80)
        mock_win.key_queue.append(ord("q"))

        with patch("chronosmatch.dashboard.curses_backend.curses") as mock_curses:
            mock_curses.initscr.return_value = mock_win
            mock_curses.has_colors.return_value = False
            with patch("chronosmatch.dashboard.curses_backend.can_initialize_curses", return_value=True):
                dashboard.start()
                assert dashboard.is_running
                dashboard.join(timeout=1.0)
                assert not dashboard.is_running
                assert exit_called.is_set()

    def test_keyboard_exit_on_esc(self) -> None:
        """Pressing ESC (27) triggers exit."""
        exit_called = threading.Event()
        collector = DashboardMetricsCollector()
        dashboard = CursesDashboard(
            collector,
            refresh_interval=0.1,
            on_exit=exit_called.set,
        )

        mock_win = MockCursesWindow(height=24, width=80)
        mock_win.key_queue.append(27)  # ESC

        with patch("chronosmatch.dashboard.curses_backend.curses") as mock_curses:
            mock_curses.initscr.return_value = mock_win
            mock_curses.has_colors.return_value = False
            with patch("chronosmatch.dashboard.curses_backend.can_initialize_curses", return_value=True):
                dashboard.start()
                dashboard.join(timeout=1.0)
                assert not dashboard.is_running
                assert exit_called.is_set()

    def test_key_resize_and_refresh_do_not_exit(self) -> None:
        """KEY_RESIZE and 'r' clear the screen without triggering exit."""
        collector = DashboardMetricsCollector()
        dashboard = CursesDashboard(collector, refresh_interval=0.1)

        mock_win = MockCursesWindow(height=24, width=80)
        resize_key = getattr(curses, "KEY_RESIZE", 410) if curses is not None else 410
        mock_win.key_queue.extend([resize_key, ord("r")])

        with patch("chronosmatch.dashboard.curses_backend.curses") as mock_curses:
            mock_curses.initscr.return_value = mock_win
            mock_curses.has_colors.return_value = False
            mock_curses.KEY_RESIZE = 410
            with patch("chronosmatch.dashboard.curses_backend.can_initialize_curses", return_value=True):
                dashboard.start()
                time.sleep(0.15)
                # Should still be running after resize and 'r'
                assert dashboard.is_running
                dashboard.stop()
                assert not dashboard.is_running

    def test_stop_restores_curses_state(self) -> None:
        """stop() releases curses screen and restores normal terminal mode."""
        collector = DashboardMetricsCollector()
        dashboard = CursesDashboard(collector, refresh_interval=0.1)
        mock_win = MockCursesWindow(height=24, width=80)

        with patch("chronosmatch.dashboard.curses_backend.curses") as mock_curses:
            mock_curses.initscr.return_value = mock_win
            mock_curses.has_colors.return_value = False
            with patch("chronosmatch.dashboard.curses_backend.can_initialize_curses", return_value=True):
                dashboard.start()
                dashboard.stop()

                mock_curses.nocbreak.assert_called()
                mock_curses.echo.assert_called()
                mock_curses.endwin.assert_called()

    def test_idempotent_stop(self) -> None:
        """Calling stop() multiple times does not raise exceptions."""
        collector = DashboardMetricsCollector()
        dashboard = CursesDashboard(collector)
        dashboard.stop()
        dashboard.stop()
        assert not dashboard.is_running


# ==============================================================================
# 6. Pipeline Runner & ANSI Backward Compatibility
# ==============================================================================


class TestAnsiBackendBackwardCompatibility:
    """Verify that ANSI backend functionality and existing pipeline workflows remain intact."""

    def test_terminal_dashboard_backend_property(self) -> None:
        """TerminalDashboard exposes backend == DashboardBackend.ANSI."""
        collector = DashboardMetricsCollector()
        ansi_dash = TerminalDashboard(collector, interactive=False)
        assert ansi_dash.backend == DashboardBackend.ANSI

    def test_pipeline_dashboard_runner_defaults_to_ansi_on_windows(self) -> None:
        """PipelineDashboardRunner uses ANSI by default on Windows."""
        with patch("os.name", "nt"):
            runner = PipelineDashboardRunner(interactive=False)
            try:
                assert runner.backend == DashboardBackend.ANSI
                assert isinstance(runner.dashboard, TerminalDashboard)
            finally:
                runner.close()

    def test_pipeline_dashboard_runner_explicit_curses_selection(self) -> None:
        """PipelineDashboardRunner accepts backend='curses' when supported."""
        with patch("chronosmatch.dashboard.curses_backend.can_initialize_curses", return_value=True):
            runner = PipelineDashboardRunner(backend=DashboardBackend.CURSES, interactive=False)
            try:
                assert runner.backend == DashboardBackend.CURSES
                assert isinstance(runner.dashboard, CursesDashboard)
            finally:
                runner.close()

    def test_pipeline_dashboard_runner_run_steps_ansi(self) -> None:
        """run_steps() functions seamlessly with ANSI backend."""
        runner = PipelineDashboardRunner(backend=DashboardBackend.ANSI, interactive=False)
        try:
            trades = runner.run_steps(5)
            assert isinstance(trades, list)
            assert runner.collector.take_snapshot().orders_processed == 5
        finally:
            runner.close()
