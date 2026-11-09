"""Curses-based Terminal Dashboard Backend for ChronosMatch Telemetry.

Provides an optional, high-performance, non-blocking Python Curses rendering backend
while preserving the existing ANSI Virtual Terminal dashboard as the default on Windows.

Key Components:
- `DashboardBackend`: Enum defining supported rendering backends (ANSI, CURSES, AUTO).
- `CursesUnsupportedError`: Raised when Curses is explicitly requested but unavailable.
- `is_curses_available()`: Checks whether the `curses` module is importable.
- `can_initialize_curses()`: Validates that curses can be safely initialized in the current TTY environment.
- `resolve_backend()`: Resolves the active backend with safe Windows defaults and automatic fallback.
- `CursesColorTheme`: Encapsulates terminal color pairs, bold, and dim styling attributes.
- `safe_addstr()`: Out-of-bounds safe text rendering guarding against terminal boundaries.
- `CursesRenderer`: Formats and renders a `DashboardMetricsSnapshot` onto a Curses window with
  dynamic resizing and minimum dimension protection.
- `CursesDashboard`: Thread-safe lifecycle controller managing background curses execution,
  keyboard exit ('q', 'Q', 'ESC'), resize events, and clean terminal restoration.
- `create_dashboard()`: Unified factory returning `TerminalDashboard` or `CursesDashboard`.
"""

from __future__ import annotations

import atexit
import datetime
from decimal import Decimal
from enum import Enum
import math
import os
import sys
import threading
import time
from typing import Any, Callable, TextIO

from chronosmatch.dashboard.metrics import (
    DashboardMetricsCollector,
    DashboardMetricsSnapshot,
    LatencyStats,
)
from chronosmatch.dashboard.terminal import (
    AnsiStyler,
    DashboardRenderer,
    TerminalDashboard,
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

# Optional curses import with safe fallback
try:
    import curses
    _CURSES_IMPORT_ERROR: Exception | None = None
except Exception as _err:
    curses = None  # type: ignore[assignment]
    _CURSES_IMPORT_ERROR = _err


# ==============================================================================
# Backend Enum & Detection Utilities
# ==============================================================================


class DashboardBackend(str, Enum):
    """Supported terminal dashboard rendering backends."""

    ANSI = "ansi"
    CURSES = "curses"
    AUTO = "auto"


class CursesUnsupportedError(RuntimeError):
    """Raised when the Curses dashboard backend is requested but unsupported in the current environment."""


def is_curses_available() -> bool:
    """Return True if the Python curses module is importable in the current environment."""
    return curses is not None and hasattr(curses, "initscr")


def can_initialize_curses(stream: Any = None) -> bool:
    """Return True if curses can safely be initialized in the current terminal environment.

    Requirements:
    1. curses module must be importable.
    2. Standard input and the target output stream must be connected to an interactive TTY.
    3. On POSIX systems, TERM must not be empty or 'dumb'.
    """
    if not is_curses_available():
        return False

    out = stream if stream is not None else sys.stdout
    if not (hasattr(out, "isatty") and out.isatty()):
        return False
    if not (hasattr(sys.stdin, "isatty") and sys.stdin.isatty()):
        return False

    if os.name != "nt":
        term = os.environ.get("TERM", "").strip()
        if not term or term == "dumb":
            return False

    return True


def resolve_backend(
    requested: str | DashboardBackend | None = None,
    stream: Any = None,
    fallback_to_ansi: bool = True,
) -> DashboardBackend:
    """Resolve the active dashboard rendering backend with safe defaults.

    Rules:
    - Default on Windows is ANSI.
    - AUTO selects CURSES if curses can safely initialize, otherwise ANSI.
    - Explicit CURSES checks availability; if unavailable and fallback_to_ansi is True,
      falls back to ANSI; otherwise raises CursesUnsupportedError.
    - Explicit ANSI returns ANSI.

    Args:
        requested: Requested backend ('ansi', 'curses', 'auto', or None).
        stream: Output stream to inspect for interactive terminal capabilities.
        fallback_to_ansi: If True, falls back to ANSI when CURSES is unavailable.

    Returns:
        Resolved DashboardBackend (ANSI or CURSES).

    Raises:
        CursesUnsupportedError: If CURSES is requested, unavailable, and fallback_to_ansi is False.
    """
    # Normalize requested backend
    if isinstance(requested, DashboardBackend):
        req = requested
    elif isinstance(requested, str):
        cleaned = requested.strip().lower()
        if cleaned in ("curses", "dashboardbackend.curses"):
            req = DashboardBackend.CURSES
        elif cleaned in ("auto", "dashboardbackend.auto"):
            req = DashboardBackend.AUTO
        else:
            req = DashboardBackend.ANSI
    else:
        req = DashboardBackend.AUTO

    if req == DashboardBackend.AUTO:
        if os.name == "nt":
            # Windows defaults to ANSI for maximum console compatibility
            return DashboardBackend.ANSI
        if can_initialize_curses(stream):
            return DashboardBackend.CURSES
        return DashboardBackend.ANSI

    if req == DashboardBackend.CURSES:
        if can_initialize_curses(stream):
            return DashboardBackend.CURSES
        if fallback_to_ansi:
            return DashboardBackend.ANSI
        raise CursesUnsupportedError(
            "Curses backend is not supported in the current environment "
            "(requires an interactive terminal TTY and curses module). "
            f"Module available: {is_curses_available()}."
        )

    return DashboardBackend.ANSI


# ==============================================================================
# Curses Color Theme & Safe Drawing Helpers
# ==============================================================================


class CursesColorTheme:
    """Manages Curses color pairs and styling attributes."""

    PAIR_DEFAULT = 0
    PAIR_GREEN = 1
    PAIR_YELLOW = 2
    PAIR_RED = 3
    PAIR_CYAN = 4
    PAIR_MAGENTA = 5
    PAIR_BLUE = 6
    PAIR_GRAY = 7

    def __init__(self) -> None:
        self.has_colors: bool = False
        self._initialized: bool = False

    def init_colors(self) -> None:
        """Initialize color pairs in the active curses session."""
        if curses is None:
            return
        if not hasattr(curses, "has_colors") or not curses.has_colors():
            self.has_colors = False
            return

        try:
            curses.start_color()
            if hasattr(curses, "use_default_colors"):
                try:
                    curses.use_default_colors()
                    bg = -1
                except Exception:
                    bg = curses.COLOR_BLACK
            else:
                bg = curses.COLOR_BLACK

            curses.init_pair(self.PAIR_GREEN, curses.COLOR_GREEN, bg)
            curses.init_pair(self.PAIR_YELLOW, curses.COLOR_YELLOW, bg)
            curses.init_pair(self.PAIR_RED, curses.COLOR_RED, bg)
            curses.init_pair(self.PAIR_CYAN, curses.COLOR_CYAN, bg)
            curses.init_pair(self.PAIR_MAGENTA, curses.COLOR_MAGENTA, bg)
            curses.init_pair(self.PAIR_BLUE, curses.COLOR_BLUE, bg)
            curses.init_pair(self.PAIR_GRAY, curses.COLOR_WHITE, bg)
            self.has_colors = True
            self._initialized = True
        except Exception:
            self.has_colors = False

    def color(self, pair_num: int, bold: bool = False, dim: bool = False) -> int:
        """Return combined curses attribute integer."""
        if curses is None or not self.has_colors:
            attr = curses.A_BOLD if (bold and curses is not None and hasattr(curses, "A_BOLD")) else 0
            return attr
        attr = curses.color_pair(pair_num) if hasattr(curses, "color_pair") else 0
        if bold and hasattr(curses, "A_BOLD"):
            attr |= curses.A_BOLD
        if dim and hasattr(curses, "A_DIM"):
            attr |= curses.A_DIM
        return attr

    def green(self, bold: bool = False) -> int:
        return self.color(self.PAIR_GREEN, bold=bold)

    def yellow(self, bold: bool = False) -> int:
        return self.color(self.PAIR_YELLOW, bold=bold)

    def red(self, bold: bool = False) -> int:
        return self.color(self.PAIR_RED, bold=bold)

    def cyan(self, bold: bool = False) -> int:
        return self.color(self.PAIR_CYAN, bold=bold)

    def magenta(self, bold: bool = False) -> int:
        return self.color(self.PAIR_MAGENTA, bold=bold)

    def blue(self, bold: bool = False) -> int:
        return self.color(self.PAIR_BLUE, bold=bold)

    def gray(self, dim: bool = True) -> int:
        return self.color(self.PAIR_GRAY, dim=dim)

    def normal(self) -> int:
        return 0


def safe_addstr(
    win: Any,
    y: int,
    x: int,
    text: str,
    attr: int = 0,
    max_len: int | None = None,
) -> None:
    """Safely write text to a curses window without throwing out-of-bounds exceptions.

    Guards against terminal boundaries and the curses bottom-right corner exception.
    """
    if win is None:
        return
    try:
        max_y, max_x = win.getmaxyx()
    except Exception:
        return

    if y < 0 or y >= max_y or x < 0 or x >= max_x:
        return

    avail_cols = max_x - x
    if avail_cols <= 0:
        return

    clean_text = strip_ansi(str(text))
    if max_len is not None:
        clean_text = clean_text[:max_len]
    clean_text = clean_text[:avail_cols]

    # Curses raises curses.error when writing to the bottom-right cell
    if y == max_y - 1 and len(clean_text) >= avail_cols:
        clean_text = clean_text[:-1]

    if not clean_text:
        return

    try:
        if hasattr(win, "addnstr"):
            win.addnstr(y, x, clean_text, len(clean_text), attr)
        elif hasattr(win, "addstr"):
            win.addstr(y, x, clean_text, attr)
    except Exception:
        pass


# ==============================================================================
# Curses Dashboard Renderer
# ==============================================================================


class CursesRenderer:
    """Formats and renders ChronosMatch telemetry onto a Curses window.

    Features:
    - Guaranteed error-free drawing on small or resized windows.
    - Clear warning display when terminal dimensions are insufficient (< 68x16).
    - Dynamic horizontal centering and boundary clipping.
    - Full telemetry rendering: best bid/ask, spread, orders, trades, volume,
      throughput, microsecond latency distributions (p50, p95, p99), ring buffer
      utilization, and whale order / price sweep alerts.
    """

    MIN_WIDTH: int = 68
    MIN_HEIGHT: int = 16

    def __init__(self, theme: CursesColorTheme | None = None) -> None:
        self.theme: CursesColorTheme = theme if theme is not None else CursesColorTheme()
        self._ansi_styler: AnsiStyler = AnsiStyler(enabled=False)

    def render(
        self,
        win: Any,
        snapshot: DashboardMetricsSnapshot,
        *,
        refresh_interval: float | None = None,
        status: str | None = None,
    ) -> None:
        """Render snapshot onto curses window with dimension and boundary protection.

        Args:
            win: Curses window instance (e.g. stdscr).
            snapshot: Immutable point-in-time metrics snapshot.
            refresh_interval: Seconds between live UI re-renders.
            status: Optional status string override.
        """
        if win is None:
            return

        try:
            max_y, max_x = win.getmaxyx()
        except Exception:
            return

        # Check for insufficient terminal dimensions
        if max_y < self.MIN_HEIGHT or max_x < self.MIN_WIDTH:
            self._render_small_terminal(win, max_y, max_x)
            return

        theme = self.theme
        width = min(max_x, 80)
        start_x = max(0, (max_x - width) // 2)

        try:
            win.erase()
        except Exception:
            pass

        interval_str = f"{refresh_interval:.2f}s" if refresh_interval is not None else "0.50s"
        status_text, status_attr = self._resolve_status(snapshot, status)
        wall_clock = format_wall_clock(snapshot.wall_timestamp_ns)
        engine_type = "Cython" if snapshot.is_cython_accelerated else "Python"
        source = snapshot.source_name[:14]

        cur_y = 0
        h_border = "+" + "-" * (width - 2) + "+"

        # Top border & Header
        safe_addstr(win, cur_y, start_x, h_border, theme.cyan(bold=True))
        cur_y += 1

        title_str = "  CHRONOSMATCH | Curses High-Frequency Telemetry Dashboard"
        safe_addstr(win, cur_y, start_x, "|", theme.cyan(bold=True))
        safe_addstr(win, cur_y, start_x + 1, title_str.ljust(width - 2), theme.cyan(bold=True))
        safe_addstr(win, cur_y, start_x + width - 1, "|", theme.cyan(bold=True))
        cur_y += 1

        sub_str = f"  Source: {source:<12} Core: {engine_type:<6} Status: {status_text:<6} Int: {interval_str} {wall_clock}"
        safe_addstr(win, cur_y, start_x, "|", theme.cyan(bold=True))
        safe_addstr(win, cur_y, start_x + 1, sub_str.ljust(width - 2), theme.normal())
        # Highlight status within row
        stat_pos = sub_str.find(f"Status: {status_text}")
        if stat_pos != -1:
            safe_addstr(win, cur_y, start_x + 1 + stat_pos + 8, status_text, status_attr)
        safe_addstr(win, cur_y, start_x + width - 1, "|", theme.cyan(bold=True))
        cur_y += 1

        # Section 1: System Runtime & Throughput
        safe_addstr(win, cur_y, start_x, h_border, theme.gray())
        cur_y += 1
        self._draw_section_header(win, cur_y, start_x, width, "SYSTEM RUNTIME & THROUGHPUT")
        cur_y += 1
        self._draw_two_col(
            win, cur_y, start_x, width,
            "Elapsed Runtime", format_time(snapshot.elapsed_seconds, self._ansi_styler),
            "Cum. Throughput", format_rate(snapshot.throughput_orders_per_sec, self._ansi_styler),
            val2_attr=theme.green(bold=True) if (snapshot.throughput_orders_per_sec or 0) > 0 else theme.normal(),
        )
        cur_y += 1
        self._draw_two_col(
            win, cur_y, start_x, width,
            "Wall Clock Time", wall_clock,
            "Win. Throughput", format_rate(snapshot.window_throughput_orders_per_sec, self._ansi_styler),
            val2_attr=theme.green(bold=True) if (snapshot.window_throughput_orders_per_sec or 0) > 0 else theme.normal(),
        )
        cur_y += 1

        # Section 2: Order Flow & Pipeline Status
        if cur_y + 6 < max_y:
            safe_addstr(win, cur_y, start_x, h_border, theme.gray())
            cur_y += 1
            self._draw_section_header(win, cur_y, start_x, width, "ORDER FLOW & PIPELINE STATUS")
            cur_y += 1
            self._draw_two_col(
                win, cur_y, start_x, width,
                "Orders Generated", format_int(snapshot.orders_generated, self._ansi_styler),
                "Orders Cancelled", format_int(snapshot.orders_cancelled, self._ansi_styler),
            )
            cur_y += 1
            self._draw_two_col(
                win, cur_y, start_x, width,
                "Orders Published", format_int(snapshot.orders_published, self._ansi_styler),
                "Orders Dropped", format_int(snapshot.orders_dropped, self._ansi_styler, default="0"),
                val2_attr=theme.red(bold=True) if (snapshot.orders_dropped or 0) > 0 else theme.green(),
            )
            cur_y += 1
            self._draw_two_col(
                win, cur_y, start_x, width,
                "Orders Consumed", format_int(snapshot.orders_consumed, self._ansi_styler),
                "Processing Errors", format_int(snapshot.processing_errors, self._ansi_styler, default="0"),
                val2_attr=theme.red(bold=True) if (snapshot.processing_errors or 0) > 0 else theme.green(),
            )
            cur_y += 1
            self._draw_two_col(
                win, cur_y, start_x, width,
                "Orders Processed", format_int(snapshot.orders_processed, self._ansi_styler),
                "Backpressure Evt", format_int(snapshot.backpressure_events, self._ansi_styler, default="0"),
                val2_attr=theme.red(bold=True) if (snapshot.backpressure_events or 0) > 0 else theme.green(),
            )
            cur_y += 1

        # Section 3: Trade Executions & Market State
        if cur_y + 5 < max_y:
            safe_addstr(win, cur_y, start_x, h_border, theme.gray())
            cur_y += 1
            self._draw_section_header(win, cur_y, start_x, width, "TRADE EXECUTIONS & MARKET STATE")
            cur_y += 1
            self._draw_two_col(
                win, cur_y, start_x, width,
                "Executed Trades", format_int(snapshot.trades_executed, self._ansi_styler),
                "Traded Volume", format_int(snapshot.volume_executed, self._ansi_styler),
            )
            cur_y += 1
            self._draw_two_col(
                win, cur_y, start_x, width,
                "Best Bid", format_price(snapshot.best_bid, self._ansi_styler),
                "Best Ask", format_price(snapshot.best_ask, self._ansi_styler),
                val1_attr=theme.green(bold=True) if snapshot.best_bid is not None else theme.gray(),
                val2_attr=theme.red(bold=True) if snapshot.best_ask is not None else theme.gray(),
            )
            cur_y += 1
            self._draw_two_col(
                win, cur_y, start_x, width,
                "BBO Spread", format_price(snapshot.spread, self._ansi_styler),
                "Book Depth Levels", format_int(snapshot.order_book_depth_levels, self._ansi_styler),
            )
            cur_y += 1

        # Section 4: Ring Buffer Telemetry
        if cur_y + 4 < max_y:
            safe_addstr(win, cur_y, start_x, h_border, theme.gray())
            cur_y += 1
            self._draw_section_header(win, cur_y, start_x, width, "RING BUFFER TELEMETRY")
            cur_y += 1
            cap_str = f"{format_int(snapshot.ring_buffer_capacity, self._ansi_styler)} slots" if snapshot.ring_buffer_capacity else "N/A"
            occ_str = f"{format_int(snapshot.ring_buffer_occupied, self._ansi_styler)} slots" if snapshot.ring_buffer_occupied is not None else "N/A"
            self._draw_two_col(
                win, cur_y, start_x, width,
                "Buffer Capacity", cap_str,
                "Occupied Slots", occ_str,
            )
            cur_y += 1
            util_val = snapshot.ring_buffer_utilization
            util_str = f"{util_val * 100:.1f}%" if util_val is not None else "N/A"
            gauge_w = 20
            filled = int(round((util_val or 0.0) * gauge_w))
            filled = max(0, min(gauge_w, filled))
            bar_text = f"[{'#' * filled}{'.' * (gauge_w - filled)}] {util_str}"
            self._draw_two_col(
                win, cur_y, start_x, width,
                "Utilization", bar_text,
                "Buffer State", "HEALTHY" if (util_val or 0.0) < 0.9 else "NEAR CAPACITY",
                val1_attr=theme.yellow(bold=True) if (util_val or 0.0) >= 0.7 else theme.green(),
                val2_attr=theme.red(bold=True) if (util_val or 0.0) >= 0.9 else theme.green(),
            )
            cur_y += 1

        # Section 5: Latency Telemetry & Breakdown (p50, p95, p99)
        if cur_y + 5 < max_y:
            safe_addstr(win, cur_y, start_x, h_border, theme.gray())
            cur_y += 1
            core_lat = snapshot.core_latency
            sample_count = core_lat.sample_count if core_lat else 0
            tag = f"({sample_count:,} Samples)" if sample_count > 0 else "(No Samples)"
            self._draw_section_header(win, cur_y, start_x, width, "LATENCY TELEMETRY & BREAKDOWN", extra=tag)
            cur_y += 1

            p50_str = format_latency(core_lat.p50_us if core_lat else None, self._ansi_styler, micro_unit="us")
            p95_str = format_latency(core_lat.p95_us if core_lat else None, self._ansi_styler, micro_unit="us")
            p99_str = format_latency(core_lat.p99_us if core_lat else None, self._ansi_styler, micro_unit="us")
            min_str = format_latency(core_lat.min_us if core_lat else None, self._ansi_styler, micro_unit="us")
            mean_str = format_latency(core_lat.mean_us if core_lat else None, self._ansi_styler, micro_unit="us")
            max_str = format_latency(core_lat.max_us if core_lat else None, self._ansi_styler, micro_unit="us")

            self._draw_two_col(
                win, cur_y, start_x, width,
                "p50 (Median)", p50_str,
                "Min Latency", min_str,
                val1_attr=self._latency_color(core_lat.p50_us if core_lat else None),
            )
            cur_y += 1
            self._draw_two_col(
                win, cur_y, start_x, width,
                "p95", p95_str,
                "Mean Latency", mean_str,
                val1_attr=self._latency_color(core_lat.p95_us if core_lat else None),
            )
            cur_y += 1
            self._draw_two_col(
                win, cur_y, start_x, width,
                "p99", p99_str,
                "Max Latency", max_str,
                val1_attr=self._latency_color(core_lat.p99_us if core_lat else None),
            )
            cur_y += 1

        # Section 6: Whale Orders & Sweep Telemetry (if detected)
        has_whales = snapshot.whale_orders_detected is not None and snapshot.whale_orders_detected > 0
        if has_whales and cur_y + 4 < max_y:
            safe_addstr(win, cur_y, start_x, h_border, theme.gray())
            cur_y += 1
            alert_tag = f"(! {snapshot.whale_orders_detected:,} DETECTED !)"
            self._draw_section_header(win, cur_y, start_x, width, "WHALE ORDER & SWEEP TELEMETRY", extra=alert_tag, attr=theme.magenta(bold=True))
            cur_y += 1

            thresh_str = f">= {format_int(snapshot.whale_order_threshold, self._ansi_styler)}" if snapshot.whale_order_threshold else "N/A"
            w_vol_str = format_int(snapshot.whale_volume_executed, self._ansi_styler)
            self._draw_two_col(
                win, cur_y, start_x, width,
                "Whale Threshold", thresh_str,
                "Whale Volume", w_vol_str,
                val2_attr=theme.magenta(bold=True),
            )
            cur_y += 1
            self._draw_two_col(
                win, cur_y, start_x, width,
                "Whale Orders", format_int(snapshot.whale_orders_detected, self._ansi_styler),
                "Multi-Lvl Sweeps", format_int(snapshot.multi_level_sweeps, self._ansi_styler),
                val1_attr=theme.magenta(bold=True),
                val2_attr=theme.yellow(bold=True) if (snapshot.multi_level_sweeps or 0) > 0 else theme.normal(),
            )
            cur_y += 1

            if snapshot.last_whale_order and cur_y + 2 < max_y:
                ord_info = snapshot.last_whale_order
                oid = str(ord_info.get("order_id", "N/A"))
                side = str(ord_info.get("side", "N/A"))
                qty = format_int(ord_info.get("quantity"), self._ansi_styler)
                pval = ord_info.get("price")
                pstr = format_price(pval, self._ansi_styler) if pval is not None else "MKT"
                swept = ord_info.get("swept_levels", 0)

                desc = f"  Last Whale Order:   {side} {qty} @ {pstr} ({oid})"
                safe_addstr(win, cur_y, start_x, "|", theme.cyan(bold=True))
                safe_addstr(win, cur_y, start_x + 1, desc.ljust(width - 2), theme.magenta(bold=True))
                safe_addstr(win, cur_y, start_x + width - 1, "|", theme.cyan(bold=True))
                cur_y += 1

                if swept > 1 and cur_y + 2 < max_y:
                    sw_text = f"  >>> MULTI-LEVEL SWEEP: Executed across {swept} price levels! <<<"
                    safe_addstr(win, cur_y, start_x, "|", theme.cyan(bold=True))
                    safe_addstr(win, cur_y, start_x + 1, sw_text.ljust(width - 2), theme.yellow(bold=True))
                    safe_addstr(win, cur_y, start_x + width - 1, "|", theme.cyan(bold=True))
                    cur_y += 1

        # Bottom Border & Hotkey Footer
        safe_addstr(win, cur_y, start_x, h_border, theme.cyan(bold=True))
        cur_y += 1
        footer = "  [q] Quit   [r] Refresh   [Ctrl+C] Exit   |   ChronosMatch Curses UI"
        safe_addstr(win, cur_y, start_x, footer.ljust(width), theme.gray())

    def _draw_section_header(
        self,
        win: Any,
        y: int,
        start_x: int,
        width: int,
        title: str,
        extra: str = "",
        attr: int | None = None,
    ) -> None:
        theme = self.theme
        h_attr = attr if attr is not None else theme.cyan(bold=True)
        safe_addstr(win, y, start_x, "|", theme.gray())
        title_full = f"  [{title}]"
        if extra:
            title_full = f"{title_full}  {extra}"
        safe_addstr(win, y, start_x + 1, title_full.ljust(width - 2), h_attr)
        safe_addstr(win, y, start_x + width - 1, "|", theme.gray())

    def _draw_two_col(
        self,
        win: Any,
        y: int,
        start_x: int,
        width: int,
        lbl1: str,
        val1: str,
        lbl2: str,
        val2: str,
        val1_attr: int | None = None,
        val2_attr: int | None = None,
    ) -> None:
        theme = self.theme
        inner_w = width - 4
        sep_x = start_x + (width // 2)

        # Left border, middle divider, right border
        safe_addstr(win, y, start_x, "|", theme.gray())
        safe_addstr(win, y, sep_x, "|", theme.gray())
        safe_addstr(win, y, start_x + width - 1, "|", theme.gray())

        # Col 1: Label at start_x + 2, Value right-aligned at sep_x - 2
        col1_w = sep_x - start_x - 3
        safe_addstr(win, y, start_x + 2, f"{lbl1}:", theme.normal())
        safe_addstr(win, y, start_x + 2 + len(f"{lbl1}:") + 1, " " * max(0, col1_w - len(f"{lbl1}:") - 1 - len(val1)))
        safe_addstr(win, y, sep_x - len(val1) - 1, val1, val1_attr if val1_attr is not None else theme.normal())

        # Col 2: Label at sep_x + 2, Value right-aligned at start_x + width - 3
        col2_w = (start_x + width - 1) - sep_x - 3
        safe_addstr(win, y, sep_x + 2, f"{lbl2}:", theme.normal())
        safe_addstr(win, y, sep_x + 2 + len(f"{lbl2}:") + 1, " " * max(0, col2_w - len(f"{lbl2}:") - 1 - len(val2)))
        safe_addstr(win, y, start_x + width - 1 - len(val2) - 1, val2, val2_attr if val2_attr is not None else theme.normal())

    def _latency_color(self, us: float | None) -> int:
        if us is None:
            return self.theme.gray()
        if us < 10.0:
            return self.theme.green()
        if us < 50.0:
            return self.theme.yellow()
        return self.theme.red(bold=True)

    def _resolve_status(self, snapshot: DashboardMetricsSnapshot, override: str | None) -> tuple[str, int]:
        if override is not None:
            return override, self.theme.cyan(bold=True)
        if snapshot.elapsed_seconds is None:
            return "IDLE", self.theme.yellow()
        if (snapshot.throughput_orders_per_sec or 0) > 0 or (snapshot.orders_processed or 0) > 0:
            return "ACTIVE", self.theme.green(bold=True)
        return "READY", self.theme.cyan()

    def _render_small_terminal(self, win: Any, max_y: int, max_x: int) -> None:
        """Render centered warning box when terminal window dimensions are insufficient."""
        theme = self.theme
        try:
            win.erase()
        except Exception:
            pass

        box_w = min(max_x - 2, 64)
        if box_w < 20:
            safe_addstr(win, 0, 0, "TERMINAL TOO SMALL", theme.red(bold=True))
            return

        start_x = max(0, (max_x - box_w) // 2)
        start_y = max(0, (max_y - 6) // 2)

        top_border = "+" + "-" * (box_w - 2) + "+"
        safe_addstr(win, start_y, start_x, top_border, theme.red(bold=True))
        safe_addstr(win, start_y + 1, start_x, "| " + "TERMINAL WINDOW TOO SMALL".center(box_w - 4) + " |", theme.red(bold=True))
        size_msg = f"Size: {max_x}x{max_y} (Required: {self.MIN_WIDTH}x{self.MIN_HEIGHT})"
        safe_addstr(win, start_y + 2, start_x, "| " + size_msg.center(box_w - 4) + " |", theme.yellow())
        safe_addstr(win, start_y + 3, start_x, "| " + "Please enlarge your terminal window.".center(box_w - 4) + " |", theme.normal())
        safe_addstr(win, start_y + 4, start_x, "| " + "Press 'q' to quit.".center(box_w - 4) + " |", theme.cyan())
        safe_addstr(win, start_y + 5, start_x, top_border, theme.red(bold=True))

    def render_to_string(self, snapshot: DashboardMetricsSnapshot) -> str:
        """Generate a formatted plain-text representation of telemetry matching Curses layout."""
        renderer = DashboardRenderer(use_colors=False, use_unicode=False, width=80)
        return renderer.render(snapshot)


# ==============================================================================
# Curses Dashboard Lifecycle Controller
# ==============================================================================


class CursesDashboard:
    """Thread-safe Curses-based telemetry dashboard for ChronosMatch.

    Features:
    - Dedicated rendering thread decoupled from matching engine critical path.
    - Non-blocking keyboard input polling for 'q', 'Q', 'ESC', and Ctrl+C exit.
    - Terminal resize handling (`curses.KEY_RESIZE`) without crashes.
    - Graceful terminal restoration via `curses.endwin()`.
    - Complete equivalence with `TerminalDashboard` interface.
    """

    def __init__(
        self,
        collector: DashboardMetricsCollector,
        *,
        refresh_interval: float = 0.5,
        on_render: Callable[[DashboardMetricsSnapshot, str], None] | None = None,
        on_exit: Callable[[], None] | None = None,
        stream: TextIO | None = None,
    ) -> None:
        """Initialize CursesDashboard.

        Args:
            collector: `DashboardMetricsCollector` instance providing telemetry snapshots.
            refresh_interval: Seconds between live UI re-renders (must be strictly > 0).
            on_render: Optional callback invoked with `(snapshot, rendered_string)` after each frame.
            on_exit: Optional callback invoked when the user requests keyboard exit ('q' / 'ESC').
            stream: Target output stream for TTY checking (defaults to sys.stdout).

        Raises:
            ValueError: If refresh_interval <= 0.
        """
        if refresh_interval <= 0:
            raise ValueError(f"refresh_interval must be strictly positive (> 0), got {refresh_interval}")

        self._collector = collector
        self._refresh_interval = refresh_interval
        self._on_render = on_render
        self._on_exit = on_exit
        self._stream = stream if stream is not None else sys.stdout

        self._theme = CursesColorTheme()
        self._renderer = CursesRenderer(theme=self._theme)

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stdscr: Any = None
        self._last_snapshot: DashboardMetricsSnapshot | None = None
        self._render_count: int = 0
        self._atexit_registered: bool = False

    @property
    def backend(self) -> DashboardBackend:
        """Active dashboard rendering backend."""
        return DashboardBackend.CURSES

    @property
    def is_running(self) -> bool:
        """Return True if background rendering thread is currently active."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def render_count(self) -> int:
        """Total number of frames rendered."""
        with self._lock:
            return self._render_count

    @property
    def last_snapshot(self) -> DashboardMetricsSnapshot | None:
        """Most recently rendered metrics snapshot."""
        with self._lock:
            return self._last_snapshot

    @property
    def refresh_interval(self) -> float:
        """Configured refresh interval in seconds."""
        return self._refresh_interval

    @refresh_interval.setter
    def refresh_interval(self, value: float) -> None:
        if value <= 0:
            raise ValueError(f"refresh_interval must be strictly positive (> 0), got {value}")
        self._refresh_interval = value

    @property
    def renderer(self) -> CursesRenderer:
        """Underlying `CursesRenderer` instance."""
        return self._renderer

    def start(self) -> None:
        """Start the background curses rendering worker thread.

        Raises:
            RuntimeError: If dashboard background thread is already running.
            CursesUnsupportedError: If curses cannot be initialized in the current environment.
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("CursesDashboard is already running.")

            if not can_initialize_curses(self._stream):
                raise CursesUnsupportedError(
                    "Curses cannot be initialized in the current environment "
                    "(requires interactive terminal TTY on stdin and stdout)."
                )

            if not self._atexit_registered:
                try:
                    atexit.register(self._restore_curses)
                    self._atexit_registered = True
                except Exception:
                    pass

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._render_loop,
                name="ChronosMatch-CursesUI",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Signal background thread to stop, wait for termination, and restore terminal.

        Idempotent: Safe to call repeatedly or on an unstarted dashboard.

        Args:
            timeout: Maximum seconds to wait for worker thread termination.
        """
        with self._lock:
            if not self._stop_event.is_set():
                self._stop_event.set()
            thread = self._thread

        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

        if self._atexit_registered:
            try:
                atexit.unregister(self._restore_curses)
                self._atexit_registered = False
            except Exception:
                pass

        self._restore_curses()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the background rendering thread to finish."""
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def render_snapshot(
        self,
        snapshot: DashboardMetricsSnapshot,
        win: Any = None,
    ) -> None:
        """Render a snapshot onto the active or provided curses window."""
        target_win = win if win is not None else self._stdscr
        if target_win is not None:
            self._renderer.render(target_win, snapshot, refresh_interval=self._refresh_interval)
            try:
                target_win.refresh()
            except Exception:
                pass

        with self._lock:
            self._last_snapshot = snapshot
            self._render_count += 1

        if self._on_render is not None:
            try:
                rendered_str = self._renderer.render_to_string(snapshot)
                self._on_render(snapshot, rendered_str)
            except Exception:
                pass

    def render_once(self) -> str:
        """Capture a live snapshot from the collector, render to window if active, and return string."""
        snapshot = self._collector.take_snapshot()
        self.render_snapshot(snapshot)
        return self._renderer.render_to_string(snapshot)

    def render_to_string(self, snapshot: DashboardMetricsSnapshot | None = None) -> str:
        """Render snapshot directly to a string representation without writing to curses."""
        if snapshot is None:
            snapshot = self._collector.take_snapshot()
        return self._renderer.render_to_string(snapshot)

    def __enter__(self) -> CursesDashboard:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.stop()

    # --------------------------------------------------------------------------
    # Internal Worker Loop & Terminal State Management
    # --------------------------------------------------------------------------

    def _render_loop(self) -> None:
        """Background worker thread running curses event polling and periodic redraws."""
        if curses is None:
            return

        stdscr = None
        try:
            stdscr = curses.initscr()
            with self._lock:
                self._stdscr = stdscr

            curses.noecho()
            curses.cbreak()
            stdscr.keypad(True)
            try:
                curses.curs_set(0)
            except Exception:
                pass
            stdscr.nodelay(True)
            self._theme.init_colors()

            while not self._stop_event.is_set():
                # 1. Non-blocking keyboard input polling
                try:
                    ch = stdscr.getch()
                except Exception:
                    ch = -1

                if ch in (ord("q"), ord("Q"), 27):  # 'q', 'Q', or ESC
                    self._stop_event.set()
                    if self._on_exit is not None:
                        try:
                            self._on_exit()
                        except Exception:
                            pass
                    break
                elif hasattr(curses, "KEY_RESIZE") and ch == curses.KEY_RESIZE:
                    try:
                        stdscr.clear()
                    except Exception:
                        pass
                elif ch in (ord("r"), ord("R")):
                    try:
                        stdscr.clear()
                    except Exception:
                        pass

                # 2. Capture and draw snapshot
                snapshot = self._collector.take_snapshot()
                with self._lock:
                    self._last_snapshot = snapshot
                    self._render_count += 1

                self._renderer.render(stdscr, snapshot, refresh_interval=self._refresh_interval)
                try:
                    stdscr.refresh()
                except Exception:
                    pass

                if self._on_render is not None:
                    try:
                        rendered_str = self._renderer.render_to_string(snapshot)
                        self._on_render(snapshot, rendered_str)
                    except Exception:
                        pass

                # Responsive wait in chunks for low keyboard latency
                poll_deadline = time.monotonic() + self._refresh_interval
                while time.monotonic() < poll_deadline and not self._stop_event.is_set():
                    time.sleep(0.02)
                    try:
                        ch = stdscr.getch()
                    except Exception:
                        ch = -1
                    if ch in (ord("q"), ord("Q"), 27):
                        self._stop_event.set()
                        if self._on_exit is not None:
                            try:
                                self._on_exit()
                            except Exception:
                                pass
                        break
                    elif hasattr(curses, "KEY_RESIZE") and ch == curses.KEY_RESIZE:
                        try:
                            stdscr.clear()
                        except Exception:
                            pass
                        break
                    elif ch in (ord("r"), ord("R")):
                        try:
                            stdscr.clear()
                        except Exception:
                            pass
                        break

        except KeyboardInterrupt:
            self._stop_event.set()
            if self._on_exit is not None:
                try:
                    self._on_exit()
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            self._restore_curses()

    def _restore_curses(self) -> None:
        """Restore terminal state and release curses screen."""
        with self._lock:
            stdscr = self._stdscr
            self._stdscr = None

        if curses is None:
            return

        try:
            if stdscr is not None and hasattr(stdscr, "keypad"):
                try:
                    stdscr.keypad(False)
                except Exception:
                    pass
            try:
                curses.nocbreak()
            except Exception:
                pass
            try:
                curses.echo()
            except Exception:
                pass
            try:
                curses.curs_set(1)
            except Exception:
                pass
            try:
                curses.endwin()
            except Exception:
                pass
        except Exception:
            pass


# ==============================================================================
# Unified Dashboard Factory
# ==============================================================================


def create_dashboard(
    collector: DashboardMetricsCollector,
    *,
    backend: str | DashboardBackend | None = None,
    refresh_interval: float = 0.5,
    interactive: bool | None = None,
    use_colors: bool = True,
    use_unicode: bool | None = None,
    width: int = 80,
    output: TextIO | None = None,
    clear_screen: bool = True,
    on_render: Callable[[DashboardMetricsSnapshot, str], None] | None = None,
    on_exit: Callable[[], None] | None = None,
    fallback_to_ansi: bool = True,
) -> TerminalDashboard | CursesDashboard:
    """Instantiate a dashboard controller with the resolved backend.

    On Windows, the default backend is ANSI. If 'curses' is requested and supported,
    instantiates a CursesDashboard; otherwise instantiates a TerminalDashboard.

    Args:
        collector: `DashboardMetricsCollector` instance providing telemetry snapshots.
        backend: Requested backend ('ansi', 'curses', 'auto', or None).
        refresh_interval: Seconds between live UI re-renders.
        interactive: If True, writes cursor control codes; if False, streaming text.
        use_colors: If True, outputs colors.
        use_unicode: If True, uses Unicode characters.
        width: Terminal column width for ANSI backend.
        output: Destination output stream.
        clear_screen: If True in interactive mode, homes/clears terminal before overwriting.
        on_render: Optional callback invoked with `(snapshot, rendered_string)` after each frame.
        on_exit: Optional callback invoked on user exit request.
        fallback_to_ansi: If True, falls back to ANSI when CURSES is unavailable.

    Returns:
        Configured TerminalDashboard or CursesDashboard instance.
    """
    resolved = resolve_backend(backend, stream=output, fallback_to_ansi=fallback_to_ansi)
    if resolved == DashboardBackend.CURSES:
        return CursesDashboard(
            collector=collector,
            refresh_interval=refresh_interval,
            on_render=on_render,
            on_exit=on_exit,
            stream=output,
        )
    return TerminalDashboard(
        collector=collector,
        refresh_interval=refresh_interval,
        interactive=interactive,
        use_colors=use_colors,
        use_unicode=use_unicode,
        width=width,
        output=output,
        clear_screen=clear_screen,
        on_render=on_render,
    )
