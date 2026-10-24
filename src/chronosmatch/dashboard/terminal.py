"""Real-Time Terminal UI Presentation Layer for ChronosMatch Telemetry.

Provides lightweight, decoupled, non-blocking terminal dashboard rendering
using point-in-time snapshots from `DashboardMetricsCollector`.

Key Components:
- `AnsiStyler`: Terminal text styling and ANSI color manager with plain-text fallback.
- `BoxStyle`: Configurable Unicode and ASCII box-drawing border styles.
- `DashboardRenderer`: Pure formatting engine converting metrics snapshots into styled terminal layouts.
- `TerminalDashboard`: Thread-safe lifecycle controller managing background refresh, terminal state,
  and non-interactive execution.
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass
import datetime
from decimal import Decimal
import io
import math
import os
import re
import sys
import threading
import time
from typing import Any, Callable, TextIO

from chronosmatch.dashboard.metrics import (
    DashboardMetricsCollector,
    DashboardMetricsSnapshot,
    LatencyStats,
)

# ==============================================================================
# ANSI Styling & Text Helpers
# ==============================================================================

ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def strip_ansi(text: str) -> str:
    """Remove all ANSI escape sequences from text for accurate visible width calculation.

    Args:
        text: String possibly containing ANSI color or formatting codes.

    Returns:
        Clean plain-text string without escape sequences.
    """
    return ANSI_ESCAPE_PATTERN.sub("", text)


def visible_len(text: str) -> int:
    """Calculate the visible character width of a string, ignoring ANSI codes.

    Args:
        text: Input string.

    Returns:
        Visible column width in standard monospace terminals.
    """
    return len(strip_ansi(text))


class AnsiStyler:
    """Manages terminal styling with an ANSI color enable/disable toggle.

    When `enabled=False`, all style methods return plain, unaltered text,
    and all style code constants are empty strings.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    @property
    def RESET(self) -> str:
        return "\033[0m" if self.enabled else ""

    @property
    def BOLD(self) -> str:
        return "\033[1m" if self.enabled else ""

    @property
    def DIM(self) -> str:
        return "\033[2m" if self.enabled else ""

    @property
    def CYAN(self) -> str:
        return "\033[36m" if self.enabled else ""

    @property
    def GREEN(self) -> str:
        return "\033[32m" if self.enabled else ""

    @property
    def YELLOW(self) -> str:
        return "\033[33m" if self.enabled else ""

    @property
    def RED(self) -> str:
        return "\033[31m" if self.enabled else ""

    @property
    def MAGENTA(self) -> str:
        return "\033[35m" if self.enabled else ""

    @property
    def BLUE(self) -> str:
        return "\033[34m" if self.enabled else ""

    @property
    def WHITE(self) -> str:
        return "\033[37m" if self.enabled else ""

    @property
    def GRAY(self) -> str:
        return "\033[90m" if self.enabled else ""

    def bold(self, text: str) -> str:
        return f"{self.BOLD}{text}{self.RESET}" if self.enabled else text

    def dim(self, text: str) -> str:
        return f"{self.DIM}{text}{self.RESET}" if self.enabled else text

    def cyan(self, text: str) -> str:
        return f"{self.CYAN}{text}{self.RESET}" if self.enabled else text

    def green(self, text: str) -> str:
        return f"{self.GREEN}{text}{self.RESET}" if self.enabled else text

    def yellow(self, text: str) -> str:
        return f"{self.YELLOW}{text}{self.RESET}" if self.enabled else text

    def red(self, text: str) -> str:
        return f"{self.RED}{text}{self.RESET}" if self.enabled else text

    def gray(self, text: str) -> str:
        return f"{self.GRAY}{text}{self.RESET}" if self.enabled else text


# ==============================================================================
# Box Drawing Styles
# ==============================================================================


@dataclass(frozen=True, slots=True)
class BoxStyle:
    """Border and element characters for terminal boxes and gauges."""

    top_left: str
    top_right: str
    bottom_left: str
    bottom_right: str
    horizontal: str
    vertical: str
    divider_left: str
    divider_right: str
    divider_horizontal: str
    gauge_full: str
    gauge_empty: str
    micro: str


UNICODE_BOX = BoxStyle(
    top_left="┌",
    top_right="┐",
    bottom_left="└",
    bottom_right="┘",
    horizontal="─",
    vertical="│",
    divider_left="├",
    divider_right="┤",
    divider_horizontal="─",
    gauge_full="█",
    gauge_empty="░",
    micro="µs",
)

ASCII_BOX = BoxStyle(
    top_left="+",
    top_right="+",
    bottom_left="+",
    bottom_right="+",
    horizontal="-",
    vertical="|",
    divider_left="+",
    divider_right="+",
    divider_horizontal="-",
    gauge_full="#",
    gauge_empty="-",
    micro="us",
)


# ==============================================================================
# Metric Formatters (Strict Distinction: None -> "N/A" vs 0 -> "0")
# ==============================================================================


def format_int(val: int | None, styler: AnsiStyler, default: str = "N/A") -> str:
    """Format integer metric with thousands separators or return default."""
    if val is None:
        return styler.gray(default)
    return f"{val:,}"


def format_rate(rate: float | None, styler: AnsiStyler, default: str = "N/A") -> str:
    """Format throughput rate in operations per second or return default."""
    if rate is None:
        return styler.gray(default)
    val_str = f"{rate:,.2f} ops/s"
    return styler.green(val_str) if styler.enabled and rate > 0 else val_str


def format_time(seconds: float | None, styler: AnsiStyler, default: str = "N/A") -> str:
    """Format elapsed seconds as HH:MM:SS.mmm or MM:SS.mmm or return default."""
    if seconds is None:
        return styler.gray(default)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    return f"{minutes:02d}:{secs:06.3f}"


def format_latency(
    us: float | None,
    styler: AnsiStyler,
    micro_unit: str = "µs",
    default: str = "N/A",
) -> str:
    """Format microsecond latency with unit and performance-based coloring.

    Never displays misleading zeros: returns default if us is None.
    """
    if us is None:
        return styler.gray(default)
    val_str = f"{us:,.2f} {micro_unit}"
    if not styler.enabled:
        return val_str
    if us < 10.0:
        return styler.green(val_str)
    if us < 50.0:
        return styler.yellow(val_str)
    return styler.red(val_str)


def format_price(
    val: float | int | Decimal | None,
    styler: AnsiStyler,
    default: str = "N/A",
) -> str:
    """Format currency price or return default."""
    if val is None:
        return styler.gray(default)
    return f"${float(val):,.2f}"


def format_counter_with_alert(
    val: int | None,
    styler: AnsiStyler,
    default: str = "N/A",
) -> str:
    """Format error/drop counters: green if 0, bold red if > 0, gray if None."""
    if val is None:
        return styler.gray(default)
    if val == 0:
        return styler.green("0") if styler.enabled else "0"
    val_str = f"{val:,}"
    return styler.bold(styler.red(val_str)) if styler.enabled else val_str


def format_utilization_bar(
    util: float | None,
    width: int,
    box: BoxStyle,
    styler: AnsiStyler,
    default: str = "N/A",
) -> str:
    """Format a graphical capacity utilization bar with percentage.

    Returns default if util is None.
    """
    if util is None:
        return styler.gray(default)

    clamped_util = max(0.0, min(1.0, float(util)))
    filled = int(round(clamped_util * width))
    filled = max(0, min(width, filled))
    empty = width - filled
    pct_str = f"{clamped_util * 100:.2f}%"

    if not styler.enabled:
        bar_chars = box.gauge_full * filled + box.gauge_empty * empty
        return f"[{bar_chars}] {pct_str}"

    if clamped_util < 0.70:
        colored_bar = styler.green(box.gauge_full * filled) + styler.gray(box.gauge_empty * empty)
        return f"[{colored_bar}] {styler.green(pct_str)}"
    elif clamped_util < 0.90:
        colored_bar = styler.yellow(box.gauge_full * filled) + styler.gray(box.gauge_empty * empty)
        return f"[{colored_bar}] {styler.yellow(pct_str)}"
    else:
        colored_bar = styler.red(box.gauge_full * filled) + styler.gray(box.gauge_empty * empty)
        return f"[{colored_bar}] {styler.bold(styler.red(pct_str))}"


def format_wall_time(wall_ns: int) -> str:
    """Format high-resolution wall timestamp to UTC date and time string."""
    secs = wall_ns / 1_000_000_000.0
    dt = datetime.datetime.fromtimestamp(secs, tz=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def format_wall_clock(wall_ns: int) -> str:
    """Format wall timestamp to time-of-day HH:MM:SS.mmm UTC for compact two-column display."""
    secs = wall_ns / 1_000_000_000.0
    dt = datetime.datetime.fromtimestamp(secs, tz=datetime.timezone.utc)
    millis = int((secs % 1.0) * 1000)
    return dt.strftime(f"%H:%M:%S.{millis:03d} UTC")


# ==============================================================================
# Dashboard Renderer (Pure Formatting Engine)
# ==============================================================================


class DashboardRenderer:
    """Pure formatting engine transforming `DashboardMetricsSnapshot` into a styled terminal UI.

    Features:
    - Guaranteed uniform row width with mathematically exact padding.
    - Strips ANSI codes during padding calculation to prevent border misalignment.
    - Full color toggling (plain-text / non-interactive support).
    - Unicode and ASCII box drawing support.
    """

    def __init__(
        self,
        *,
        use_colors: bool = True,
        use_unicode: bool = True,
        width: int = 80,
    ) -> None:
        """Initialize DashboardRenderer.

        Args:
            use_colors: If True, apply ANSI color escapes. If False, produce plain text.
            use_unicode: If True, use Unicode box characters. If False, use standard ASCII.
            width: Total terminal column width (minimum 74).
        """
        self.use_colors = use_colors
        self.use_unicode = use_unicode
        self.width = max(74, width)
        self.styler = AnsiStyler(enabled=use_colors)
        self.box = UNICODE_BOX if use_unicode else ASCII_BOX

    def _pad_row(self, content: str) -> str:
        """Enclose content within left and right vertical borders, padding to exact width."""
        inner_width = self.width - 4  # 1 border + 1 space left, 1 space + 1 border right
        v_len = visible_len(content)
        if v_len > inner_width:
            # Strictly truncate overflow to keep box borders perfectly aligned
            content = strip_ansi(content)[:inner_width]
            v_len = len(content)
        pad = max(0, inner_width - v_len)
        return f"{self.box.vertical} {content}{' ' * pad} {self.box.vertical}"

    def _two_col_row(
        self,
        col1_label: str,
        col1_val: str,
        col2_label: str,
        col2_val: str,
    ) -> str:
        """Format a two-column row with aligned labels and values."""
        inner_width = self.width - 4
        sep = f" {self.box.vertical} "
        sep_len = visible_len(sep)

        col1_target = (inner_width - sep_len) // 2
        col2_target = inner_width - sep_len - col1_target

        c1_label_str = f"  {col1_label}:"
        c1_val_str = str(col1_val)
        c1_pad = max(0, col1_target - visible_len(c1_label_str) - 1 - visible_len(c1_val_str))

        c2_label_str = f"{col2_label}:"
        c2_val_str = str(col2_val)
        c2_pad = max(0, col2_target - visible_len(c2_label_str) - 1 - visible_len(c2_val_str))

        # Balance padding across columns if one column has extra length and the other has surplus padding
        c1_excess = max(0, (visible_len(c1_label_str) + 1 + visible_len(c1_val_str)) - col1_target)
        if c1_excess > 0 and c2_pad >= c1_excess:
            c2_pad -= c1_excess
            c1_pad = 0
        c2_excess = max(0, (visible_len(c2_label_str) + 1 + visible_len(c2_val_str)) - col2_target)
        if c2_excess > 0 and c1_pad >= c2_excess:
            c1_pad -= c2_excess
            c2_pad = 0

        c1 = f"{c1_label_str}{' ' * c1_pad} {c1_val_str}"
        c2 = f"{c2_label_str}{' ' * c2_pad} {c2_val_str}"

        return self._pad_row(f"{c1}{sep}{c2}")

    def _section_header(self, title: str, extra: str = "") -> str:
        """Format a section header line."""
        title_str = f"[{title}]"
        if extra:
            title_str = f"{title_str}  {extra}"
        header_text = self.styler.bold(self.styler.cyan(title_str)) if self.styler.enabled else title_str
        return self._pad_row(f"  {header_text}")

    def _divider(self) -> str:
        """Format an internal section divider line."""
        inner_len = self.width - 2
        return f"{self.box.divider_left}{self.box.divider_horizontal * inner_len}{self.box.divider_right}"

    def _resolve_status_string(self, snapshot: DashboardMetricsSnapshot) -> str:
        """Determine human-readable processing status from snapshot telemetry."""
        if snapshot.elapsed_seconds is None:
            status_text = "IDLE"
            return self.styler.yellow(status_text) if self.styler.enabled else status_text

        has_throughput = (
            snapshot.throughput_orders_per_sec is not None
            and snapshot.throughput_orders_per_sec > 0
        )
        has_processed = (
            snapshot.orders_processed is not None and snapshot.orders_processed > 0
        )

        if has_throughput or has_processed:
            status_text = "ACTIVE"
            return self.styler.bold(self.styler.green(status_text)) if self.styler.enabled else status_text

        status_text = "READY"
        return self.styler.cyan(status_text) if self.styler.enabled else status_text

    def render(
        self,
        snapshot: DashboardMetricsSnapshot,
        *,
        status: str | None = None,
        refresh_interval: float | None = None,
    ) -> str:
        """Render a complete, self-contained terminal UI dashboard string.

        Args:
            snapshot: Immutable point-in-time metrics snapshot.
            status: Optional status string override.
            refresh_interval: Optional refresh interval in seconds to display in header.

        Returns:
            Multi-line formatted dashboard string.
        """
        styler = self.styler
        box = self.box
        inner_len = self.width - 2

        status_str = status if status is not None else self._resolve_status_string(snapshot)
        wall_time_str = format_wall_time(snapshot.wall_timestamp_ns)
        interval_str = f"{refresh_interval:.2f}s" if refresh_interval is not None else "0.50s"

        lines: list[str] = []

        # ----------------------------------------------------------------------
        # Top Border & Header Banner
        # ----------------------------------------------------------------------
        lines.append(f"{box.top_left}{box.horizontal * inner_len}{box.top_right}")

        title_text = "CHRONOSMATCH  " + box.vertical + "  Real-Time Trading Telemetry Dashboard"
        title_styled = styler.bold(styler.cyan(title_text)) if styler.enabled else title_text
        lines.append(self._pad_row(f"  {title_styled}"))

        src_name = snapshot.source_name[:14]
        utc_short = datetime.datetime.fromtimestamp(
            snapshot.wall_timestamp_ns / 1_000_000_000.0,
            tz=datetime.timezone.utc,
        ).strftime("%H:%M:%S UTC")

        engine_core = "Cython" if snapshot.is_cython_accelerated else "Python"
        source_info = (
            f"  Source: {styler.bold(src_name):<14} "
            f"Core: {engine_core} "
            f"Status: {status_str} "
            f"Int: {interval_str} "
            f"{utc_short}"
        )
        lines.append(self._pad_row(source_info))

        # ----------------------------------------------------------------------
        # 1. System Runtime & Throughput
        # ----------------------------------------------------------------------
        lines.append(self._divider())
        lines.append(self._section_header("SYSTEM RUNTIME & THROUGHPUT"))
        lines.append(
            self._two_col_row(
                "Elapsed Runtime",
                format_time(snapshot.elapsed_seconds, styler),
                "Cum. Throughput",
                format_rate(snapshot.throughput_orders_per_sec, styler),
            )
        )
        lines.append(
            self._two_col_row(
                "Wall Clock Time",
                format_wall_clock(snapshot.wall_timestamp_ns),
                "Win. Throughput",
                format_rate(snapshot.window_throughput_orders_per_sec, styler),
            )
        )

        # ----------------------------------------------------------------------
        # 2. Order Flow & Pipeline Status
        # ----------------------------------------------------------------------
        lines.append(self._divider())
        lines.append(self._section_header("ORDER FLOW & PIPELINE STATUS"))
        lines.append(
            self._two_col_row(
                "Orders Generated",
                format_int(snapshot.orders_generated, styler),
                "Orders Cancelled",
                format_int(snapshot.orders_cancelled, styler),
            )
        )
        lines.append(
            self._two_col_row(
                "Orders Published",
                format_int(snapshot.orders_published, styler),
                "Orders Dropped",
                format_counter_with_alert(snapshot.orders_dropped, styler),
            )
        )
        lines.append(
            self._two_col_row(
                "Orders Consumed",
                format_int(snapshot.orders_consumed, styler),
                "Processing Errors",
                format_counter_with_alert(snapshot.processing_errors, styler),
            )
        )
        lines.append(
            self._two_col_row(
                "Orders Processed",
                format_int(snapshot.orders_processed, styler),
                "Backpressure Evt",
                format_counter_with_alert(snapshot.backpressure_events, styler),
            )
        )

        # ----------------------------------------------------------------------
        # 3. Trade Executions & Market State
        # ----------------------------------------------------------------------
        lines.append(self._divider())
        lines.append(self._section_header("TRADE EXECUTIONS & MARKET STATE"))
        lines.append(
            self._two_col_row(
                "Executed Trades",
                format_int(snapshot.trades_executed, styler),
                "Traded Volume",
                format_int(snapshot.volume_executed, styler),
            )
        )
        lines.append(
            self._two_col_row(
                "Best Bid",
                format_price(snapshot.best_bid, styler),
                "Best Ask",
                format_price(snapshot.best_ask, styler),
            )
        )
        lines.append(
            self._two_col_row(
                "BBO Spread",
                format_price(snapshot.spread, styler),
                "Book Depth Levels",
                format_int(snapshot.order_book_depth_levels, styler),
            )
        )

        # ----------------------------------------------------------------------
        # 4. Ring Buffer Telemetry & Utilization
        # ----------------------------------------------------------------------
        lines.append(self._divider())
        lines.append(self._section_header("RING BUFFER TELEMETRY"))
        lines.append(
            self._two_col_row(
                "Buffer Capacity",
                f"{format_int(snapshot.ring_buffer_capacity, styler)} slots"
                if snapshot.ring_buffer_capacity is not None
                else styler.gray("N/A"),
                "Occupied Slots",
                f"{format_int(snapshot.ring_buffer_occupied, styler)} slots"
                if snapshot.ring_buffer_occupied is not None
                else styler.gray("N/A"),
            )
        )
        # Utilization progress gauge
        gauge_width = 24
        util_gauge = format_utilization_bar(
            snapshot.ring_buffer_utilization,
            gauge_width,
            box,
            styler,
        )
        lines.append(self._pad_row(f"  Utilization:        {util_gauge}"))

        # ----------------------------------------------------------------------
        # 5. Order-to-Trade Latency Distribution
        # ----------------------------------------------------------------------
        lines.append(self._divider())
        lat = snapshot.latency
        has_latency_samples = lat is not None and lat.sample_count > 0

        if has_latency_samples and lat is not None:
            extra_sample_tag = styler.dim(f"({lat.sample_count:,} Samples)") if styler.enabled else f"({lat.sample_count:,} Samples)"
            lines.append(self._section_header("ORDER-TO-TRADE LATENCY DISTRIBUTION", extra_sample_tag))
            lines.append(
                self._two_col_row(
                    "p50 (Median)",
                    format_latency(lat.p50_us, styler, box.micro),
                    "Min Latency",
                    format_latency(lat.min_us, styler, box.micro),
                )
            )
            lines.append(
                self._two_col_row(
                    "p95",
                    format_latency(lat.p95_us, styler, box.micro),
                    "Mean Latency",
                    format_latency(lat.mean_us, styler, box.micro),
                )
            )
            lines.append(
                self._two_col_row(
                    "p99",
                    format_latency(lat.p99_us, styler, box.micro),
                    "Max Latency",
                    format_latency(lat.max_us, styler, box.micro),
                )
            )
            lines.append(
                self._two_col_row(
                    "p99.9",
                    format_latency(lat.p99_9_us, styler, box.micro),
                    "Latest Sample",
                    format_latency(lat.latest_us, styler, box.micro),
                )
            )
        else:
            no_samples_tag = styler.dim("(No Samples)") if styler.enabled else "(No Samples)"
            lines.append(self._section_header("ORDER-TO-TRADE LATENCY DISTRIBUTION", no_samples_tag))
            lines.append(
                self._two_col_row(
                    "p50 (Median)",
                    styler.gray("N/A"),
                    "Min Latency",
                    styler.gray("N/A"),
                )
            )
            lines.append(
                self._two_col_row(
                    "p95",
                    styler.gray("N/A"),
                    "Mean Latency",
                    styler.gray("N/A"),
                )
            )
            lines.append(
                self._two_col_row(
                    "p99",
                    styler.gray("N/A"),
                    "Max Latency",
                    styler.gray("N/A"),
                )
            )
            lines.append(
                self._two_col_row(
                    "p99.9",
                    styler.gray("N/A"),
                    "Latest Sample",
                    styler.gray("N/A"),
                )
            )

        # ----------------------------------------------------------------------
        # 6. System Diagnostics & Anomaly Counters
        # ----------------------------------------------------------------------
        lines.append(self._divider())
        lines.append(self._section_header("SYSTEM DIAGNOSTICS & ANOMALIES"))
        lines.append(
            self._two_col_row(
                "Malformed Msgs",
                format_counter_with_alert(snapshot.malformed_messages, styler),
                "Duplicate Orders",
                format_counter_with_alert(snapshot.duplicate_orders, styler),
            )
        )
        lines.append(
            self._two_col_row(
                "Processing Errors",
                format_counter_with_alert(snapshot.processing_errors, styler),
                "Backpressure Evt",
                format_counter_with_alert(snapshot.backpressure_events, styler),
            )
        )

        # ----------------------------------------------------------------------
        # Bottom Border
        # ----------------------------------------------------------------------
        lines.append(f"{box.bottom_left}{box.horizontal * inner_len}{box.bottom_right}")

        return "\n".join(lines)


def can_encode_unicode(stream: Any) -> bool:
    """Check if the given stream encoding supports Unicode box-drawing characters."""
    encoding = getattr(stream, "encoding", None)
    if not isinstance(encoding, str):
        encoding = getattr(sys.stdout, "encoding", None)
        if not isinstance(encoding, str):
            encoding = "utf-8"
    try:
        "┌─┐█░µs".encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError, TypeError):
        return False


def is_interactive_stream(stream: Any) -> bool:
    """Determine whether a stream is connected to an interactive terminal."""
    try:
        return bool(hasattr(stream, "isatty") and stream.isatty())
    except Exception:
        return False


def enable_windows_virtual_terminal() -> bool:
    """Enable Windows Console virtual terminal processing for ANSI cursor and color codes."""
    if os.name != "nt":
        return True
    success = False
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        if handle not in (0, -1):
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
                if not (mode.value & 0x0004):
                    kernel32.SetConsoleMode(handle, mode.value | 0x0004)
                success = True
    except Exception:
        pass

    try:
        import colorama  # type: ignore[import-untyped]

        colorama.just_fix_windows_console()
        success = True
    except Exception:
        pass

    try:
        os.system("")
        success = True
    except Exception:
        pass

    return success


# ==============================================================================
# Terminal Dashboard (Lifecycle Controller & Thread Runner)
# ==============================================================================


class TerminalDashboard:
    """Thread-safe terminal telemetry dashboard for ChronosMatch.

    Renders live snapshots from a `DashboardMetricsCollector` at configurable intervals.
    Maintains complete isolation from the engine's critical execution path:
    - Rendering runs on an independent worker thread or via explicit single-shot calls.
    - Zero bounded/unbounded metric storage leaks (only references latest snapshot).
    - Graceful shutdown handles terminal restoration and thread joining.
    - Non-interactive and plain-text modes for testing and logging.
    """

    def __init__(
        self,
        collector: DashboardMetricsCollector,
        *,
        refresh_interval: float = 0.5,
        interactive: bool | None = None,
        use_colors: bool = True,
        use_unicode: bool | None = None,
        width: int = 80,
        output: TextIO | None = None,
        clear_screen: bool = True,
        on_render: Callable[[DashboardMetricsSnapshot, str], None] | None = None,
    ) -> None:
        """Initialize TerminalDashboard.

        Args:
            collector: `DashboardMetricsCollector` instance providing telemetry snapshots.
            refresh_interval: Seconds between live UI re-renders (must be strictly > 0).
            interactive: If True, writes terminal cursor-control codes. If False, operates in plain streaming mode.
                If None, automatically detects from output stream `isatty()`.
            use_colors: If True, outputs ANSI color escape codes. If False, plain text only.
            use_unicode: If True, uses Unicode box drawing and symbols. If False, ASCII only. If None, auto-detects from stream encoding.
            width: Total terminal column width.
            output: Output stream (defaults to sys.stdout).
            clear_screen: If True in interactive mode, homes/clears terminal before overwriting.
            on_render: Optional callback invoked with `(snapshot, rendered_string)` after each frame.

        Raises:
            ValueError: If refresh_interval <= 0.
        """
        if refresh_interval <= 0:
            raise ValueError(f"refresh_interval must be strictly positive (> 0), got {refresh_interval}")

        self._collector = collector
        self._refresh_interval = refresh_interval
        self._output: TextIO = output if output is not None else sys.stdout

        if interactive is None:
            self._interactive = is_interactive_stream(self._output)
        else:
            self._interactive = bool(interactive)

        self._use_colors = use_colors

        if use_unicode is None:
            self._use_unicode = can_encode_unicode(self._output)
        else:
            self._use_unicode = use_unicode

        self._width = max(74, width)
        self._clear_screen = clear_screen
        self._on_render = on_render

        self._renderer = DashboardRenderer(
            use_colors=self._use_colors,
            use_unicode=self._use_unicode,
            width=self._width,
        )

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last_snapshot: DashboardMetricsSnapshot | None = None
        self._render_count: int = 0
        self._last_rendered_lines: int = 0
        self._atexit_registered: bool = False

        if os.name == "nt" and self._interactive:
            enable_windows_virtual_terminal()

    # --------------------------------------------------------------------------
    # Lifecycle Controls
    # --------------------------------------------------------------------------

    def start(self) -> None:
        """Start the background rendering worker thread.

        Raises:
            RuntimeError: If dashboard background thread is already running.
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("TerminalDashboard is already running.")

            if os.name == "nt" and self._interactive:
                enable_windows_virtual_terminal()

            if self._interactive:
                self._hide_cursor()
                if not self._atexit_registered:
                    try:
                        atexit.register(self._restore_terminal)
                        self._atexit_registered = True
                    except Exception:
                        pass

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._render_loop,
                name="ChronosMatch-DashboardUI",
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
                atexit.unregister(self._restore_terminal)
                self._atexit_registered = False
            except Exception:
                pass

        self._restore_terminal()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the background rendering thread to finish."""
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    @property
    def is_running(self) -> bool:
        """Return True if background rendering thread is currently active."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def interactive(self) -> bool:
        """Return True if dashboard operates with interactive terminal cursor controls."""
        return self._interactive

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
    def renderer(self) -> DashboardRenderer:
        """Underlying `DashboardRenderer` instance."""
        return self._renderer

    # --------------------------------------------------------------------------
    # Synchronous Rendering API (Non-Interactive & Testing)
    # --------------------------------------------------------------------------

    def render_to_string(self, snapshot: DashboardMetricsSnapshot | None = None) -> str:
        """Render snapshot directly to a string without writing to the output stream.

        Args:
            snapshot: Optional snapshot to format (takes live snapshot from collector if None).

        Returns:
            Formatted dashboard string.
        """
        if snapshot is None:
            snapshot = self._collector.take_snapshot()
        return self._renderer.render(
            snapshot,
            refresh_interval=self._refresh_interval,
        )

    def render_snapshot(
        self,
        snapshot: DashboardMetricsSnapshot,
        clear: bool = False,
    ) -> str:
        """Format a snapshot, write to the configured output stream, and return string.

        Args:
            snapshot: Snapshot to render.
            clear: If True in interactive mode, homes/clears terminal before overwriting.

        Returns:
            Rendered string.
        """
        rendered = self._renderer.render(
            snapshot,
            refresh_interval=self._refresh_interval,
        )

        with self._lock:
            self._last_snapshot = snapshot
            self._render_count += 1

            prefix = ""
            if self._interactive:
                if clear:
                    # In interactive clear mode, home cursor and clear entire display to prevent leftover lines
                    prefix = "\033[H\033[J" if self._render_count > 1 else "\033[2J\033[H"
                elif self._render_count > 1 and self._last_rendered_lines > 0:
                    # In inline mode without full screen clear, rewind cursor to the beginning of the previous dashboard
                    prefix = f"\r\033[{self._last_rendered_lines}A\033[J"

            try:
                self._output.write(f"{prefix}{rendered}\n")
            except UnicodeEncodeError:
                # If stream encoding cannot represent Unicode characters, automatically downgrade to ASCII
                self._use_unicode = False
                self._renderer = DashboardRenderer(
                    use_colors=self._use_colors,
                    use_unicode=False,
                    width=self._width,
                )
                rendered = self._renderer.render(
                    snapshot,
                    refresh_interval=self._refresh_interval,
                )
                self._output.write(f"{prefix}{rendered}\n")

            self._last_rendered_lines = len(rendered.splitlines())

            try:
                self._output.flush()
            except Exception:
                pass

        if self._on_render is not None:
            try:
                self._on_render(snapshot, rendered)
            except Exception:
                pass

        return rendered

    def render_once(self) -> str:
        """Capture a live snapshot from the collector, write to output, and return string."""
        snapshot = self._collector.take_snapshot()
        return self.render_snapshot(snapshot, clear=self._interactive and self._clear_screen)

    # --------------------------------------------------------------------------
    # Context Manager
    # --------------------------------------------------------------------------

    def __enter__(self) -> TerminalDashboard:
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
        """Background thread worker loop running at configured refresh interval."""
        try:
            while not self._stop_event.is_set():
                snapshot = self._collector.take_snapshot()
                if self._interactive:
                    self.render_snapshot(snapshot, clear=self._clear_screen)
                else:
                    # In non-interactive mode, avoid flooding output with repeated full screens.
                    # Maintain snapshot metrics, render count, and invoke on_render callbacks.
                    with self._lock:
                        self._last_snapshot = snapshot
                        self._render_count += 1
                    if self._on_render is not None:
                        rendered = self._renderer.render(snapshot, refresh_interval=self._refresh_interval)
                        try:
                            self._on_render(snapshot, rendered)
                        except Exception:
                            pass

                # Responsive wait: exits immediately when stop_event is signaled
                if self._stop_event.wait(timeout=self._refresh_interval):
                    break
        except Exception:
            pass
        finally:
            self._restore_terminal()

    def _hide_cursor(self) -> None:
        """Hide cursor in interactive terminal sessions."""
        if self._interactive and hasattr(self._output, "write"):
            try:
                self._output.write("\033[?25l")
                self._output.flush()
            except Exception:
                pass

    def _restore_terminal(self) -> None:
        """Restore cursor visibility and reset ANSI formatting."""
        if self._interactive and hasattr(self._output, "write"):
            try:
                self._output.write("\033[?25h\033[0m")
                self._output.flush()
            except Exception:
                pass
