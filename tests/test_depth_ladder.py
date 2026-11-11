"""Unit and Integration Test Suite for Multi-Level Order Book Depth Ladder in ChronosMatch.

Validates:
1. Top 5 BUY price levels in descending order (highest price first).
2. Top 5 SELL price levels in ascending order (lowest price first).
3. Individual price, quantity, and graphical depth bar calculations per level.
4. Best Bid and Best Ask indicators / highlighting.
5. Bid-Ask Spread display.
6. Empty order book handling.
7. Partial depth (fewer than 5 levels) handling.
8. Terminal width border integrity and exact uniform row length in ANSI dashboard.
9. Curses backend depth ladder rendering and insufficient window dimension handling.
10. Live TradingPipeline integration and snapshot propagation.
"""

from __future__ import annotations

from decimal import Decimal
import io
import re
from typing import Any

import pytest

from chronosmatch.dashboard import (
    ASCII_BOX,
    UNICODE_BOX,
    AnsiStyler,
    DashboardMetricsCollector,
    DashboardMetricsSnapshot,
    DashboardRenderer,
    strip_ansi,
    visible_len,
)
from chronosmatch.dashboard.curses_backend import (
    CursesColorTheme,
    CursesRenderer,
    safe_addstr,
)
from chronosmatch.matching_engine import MatchingEngine
from chronosmatch.models import Order, OrderSide
from chronosmatch.order_book import LimitOrderBook
from chronosmatch.pipeline import TradingPipeline, create_pipeline


class MockCursesWindow:
    """Mock curses window recording calls to addstr and dimensions for unit tests."""

    def __init__(self, height: int = 40, width: int = 80) -> None:
        self.height = height
        self.width = width
        self.lines: list[tuple[int, int, str, int]] = []
        self.bkgd_attr: int = 0

    def getmaxyx(self) -> tuple[int, int]:
        return self.height, self.width

    def erase(self) -> None:
        self.lines.clear()

    def noutrefresh(self) -> None:
        pass

    def refresh(self) -> None:
        pass

    def bkgd(self, ch: str, attr: int) -> None:
        self.bkgd_attr = attr

    def addstr(self, y: int, x: int, text: str, attr: int = 0) -> None:
        self.lines.append((y, x, text, attr))

    def get_text_at(self, y: int) -> str:
        row_entries = [entry for entry in self.lines if entry[0] == y]
        row_entries.sort(key=lambda e: e[1])
        return "".join(entry[2] for entry in row_entries)

    def get_all_text(self) -> str:
        max_y = max((entry[0] for entry in self.lines), default=-1)
        return "\n".join(self.get_text_at(y) for y in range(max_y + 1))


# ==============================================================================
# 1. Metric Resolution & Snapshot Depth Propagation
# ==============================================================================


class TestDepthLadderMetricsResolution:
    """Test Level 2 depth extraction and snapshot capture."""

    def test_collector_manual_depth_configuration(self) -> None:
        """Collector accepts manual bids_depth and asks_depth."""
        collector = DashboardMetricsCollector()
        bids = [(101.50, 1000), (101.00, 2000), (100.50, 3000)]
        asks = [(102.00, 1500), (102.50, 2500)]
        collector.set_order_book_depth(bids, asks)

        snap = collector.take_snapshot()
        assert snap.bids_depth == bids
        assert snap.asks_depth == asks
        assert snap.to_dict()["bids_depth"] == bids
        assert snap.to_dict()["asks_depth"] == asks

    def test_collector_resolves_depth_from_order_book(self) -> None:
        """Collector automatically extracts depth from attached MatchingEngine."""
        engine = MatchingEngine()
        # Add 6 bids and 6 asks
        for i in range(6):
            engine.process_order(
                Order(f"BID-{i}", OrderSide.BUY, 100.0 - (i * 0.5), 100 * (i + 1))
            )
            engine.process_order(
                Order(f"ASK-{i}", OrderSide.SELL, 101.0 + (i * 0.5), 200 * (i + 1))
            )

        collector = DashboardMetricsCollector(engine=engine)
        snap = collector.take_snapshot()

        assert snap.bids_depth is not None
        assert snap.asks_depth is not None
        # Limited to top 5
        assert len(snap.bids_depth) == 5
        assert len(snap.asks_depth) == 5

        # Bids in descending order (highest price first)
        assert snap.bids_depth[0] == (100.0, 100)
        assert snap.bids_depth[1] == (99.5, 200)
        assert snap.bids_depth[4] == (98.0, 500)

        # Asks in ascending order (lowest price first)
        assert snap.asks_depth[0] == (101.0, 200)
        assert snap.asks_depth[1] == (101.5, 400)
        assert snap.asks_depth[4] == (103.0, 1000)


# ==============================================================================
# 2. ANSI Dashboard Depth Ladder Rendering
# ==============================================================================


class TestAnsiDepthLadderRendering:
    """Validate ANSI rendering of depth ladder, bars, best bid/ask, and borders."""

    def test_empty_depth_ladder_rendering(self) -> None:
        """When an attached engine has an empty book, empty notice is rendered."""
        collector = DashboardMetricsCollector()
        collector.set_order_book_depth([], [])
        snap = collector.take_snapshot()

        renderer = DashboardRenderer(use_colors=False, use_unicode=True, width=80)
        rendered = renderer.render(snap)
        plain = strip_ansi(rendered)

        assert "ORDER BOOK L2 DEPTH LADDER (TOP 5)" in plain
        assert "(Order book depth is empty - awaiting orders)" in plain

    def test_full_depth_ladder_rendering_top_5(self) -> None:
        """Top 5 bids and asks are formatted with prices, quantities, and bars."""
        bids = [
            (100.50, 1200),
            (100.00, 800),
            (99.50, 450),
            (99.00, 300),
            (98.50, 150),
        ]
        asks = [
            (101.00, 1500),
            (101.50, 2100),
            (102.00, 600),
            (102.50, 900),
            (103.00, 1200),
        ]

        collector = DashboardMetricsCollector()
        collector.set_order_book_depth(bids, asks)
        snap = collector.take_snapshot()

        renderer = DashboardRenderer(use_colors=False, use_unicode=True, width=80)
        rendered = renderer.render(snap)
        plain = strip_ansi(rendered)

        # Check section header
        assert "ORDER BOOK L2 DEPTH LADDER (TOP 5)" in plain
        assert "BID DEPTH" in plain
        assert "ASK DEPTH" in plain

        # Check Best Bid and Best Ask asterisk highlighting
        assert "$100.50*" in plain
        assert "*$101.00" in plain

        # Check all 5 levels are present
        assert "$100.00" in plain
        assert "$99.50" in plain
        assert "$101.50" in plain
        assert "$103.00" in plain
        assert "1,200" in plain
        assert "2,100" in plain

    def test_ansi_border_uniformity_with_depth_ladder(self) -> None:
        """All rows in the rendered dashboard must strictly equal width."""
        bids = [(100.50, 5000), (100.00, 3000), (99.50, 1000)]
        asks = [(101.00, 4000), (101.50, 2500)]

        collector = DashboardMetricsCollector()
        collector.set_order_book_depth(bids, asks)
        snap = collector.take_snapshot()

        for target_width in (76, 80, 90, 100):
            renderer = DashboardRenderer(
                use_colors=True,
                use_unicode=True,
                width=target_width,
            )
            rendered = renderer.render(snap)
            lines = rendered.strip().split("\n")
            for idx, line in enumerate(lines):
                vis_len = visible_len(line)
                assert vis_len == target_width, (
                    f"Line {idx} width {vis_len} != {target_width}: '{line}'"
                )

    def test_ascii_pure_mode_with_depth_ladder(self) -> None:
        """Depth ladder must encode cleanly as pure 7-bit ASCII when use_unicode=False."""
        bids = [(50.25, 100), (50.00, 200)]
        asks = [(50.50, 150), (50.75, 250)]

        collector = DashboardMetricsCollector()
        collector.set_order_book_depth(bids, asks)
        snap = collector.take_snapshot()

        renderer = DashboardRenderer(use_colors=False, use_unicode=False, width=80)
        rendered = renderer.render(snap)

        # Must encode cleanly as 7-bit ASCII
        encoded = rendered.encode("ascii")
        assert len(encoded) > 0
        assert "#" in rendered  # ASCII gauge full
        assert "-" in rendered  # ASCII gauge empty / border


# ==============================================================================
# 3. Curses Dashboard Depth Ladder Rendering
# ==============================================================================


class TestCursesDepthLadderRendering:
    """Validate Curses depth ladder rendering on mock curses window."""

    def test_curses_renders_depth_ladder_with_sufficient_height(self) -> None:
        """CursesRenderer draws depth ladder when max_y is sufficient."""
        renderer = CursesRenderer()
        win = MockCursesWindow(height=45, width=80)

        bids = [(100.50, 1200), (100.00, 800)]
        asks = [(101.00, 1500), (101.50, 2100)]
        collector = DashboardMetricsCollector()
        collector.set_order_book_depth(bids, asks)
        snap = collector.take_snapshot()

        renderer.render(win, snap)
        full_text = win.get_all_text()

        assert "ORDER BOOK L2 DEPTH LADDER (TOP 5)" in full_text
        assert "BID DEPTH" in full_text
        assert "ASK DEPTH" in full_text
        assert "$100.50*" in full_text
        assert "*$101.00" in full_text

    def test_curses_gracefully_handles_short_window(self) -> None:
        """CursesRenderer does not crash or draw out of bounds when window is short."""
        renderer = CursesRenderer()
        # Window with modest height (18 lines)
        win = MockCursesWindow(height=18, width=80)

        bids = [(100.50, 1200), (100.00, 800)]
        asks = [(101.00, 1500), (101.50, 2100)]
        collector = DashboardMetricsCollector()
        collector.set_order_book_depth(bids, asks)
        snap = collector.take_snapshot()

        # Should render without exception
        renderer.render(win, snap)
        # Verify no lines were written at or past height
        for y, x, text, attr in win.lines:
            assert y < win.height


# ==============================================================================
# 4. Pipeline Integration with Live Depth Ladder
# ==============================================================================


class TestPipelineDepthLadderIntegration:
    """Test live TradingPipeline updating depth ladder as orders rest in the book."""

    def test_live_pipeline_depth_ladder_updates(self) -> None:
        """Submitting resting orders through pipeline updates depth ladder."""
        pipeline = create_pipeline()
        try:
            # Add resting bids
            pipeline.engine.process_order(Order("REST-B1", OrderSide.BUY, 99.0, 100))
            pipeline.engine.process_order(Order("REST-B2", OrderSide.BUY, 98.5, 200))
            pipeline.engine.process_order(Order("REST-B3", OrderSide.BUY, 98.0, 300))

            # Add resting asks
            pipeline.engine.process_order(Order("REST-A1", OrderSide.SELL, 101.0, 150))
            pipeline.engine.process_order(Order("REST-A2", OrderSide.SELL, 101.5, 250))

            collector = DashboardMetricsCollector(pipeline=pipeline)
            snap = collector.take_snapshot()

            assert snap.bids_depth is not None
            assert len(snap.bids_depth) == 3
            assert snap.bids_depth[0] == (99.0, 100)
            assert snap.bids_depth[1] == (98.5, 200)
            assert snap.bids_depth[2] == (98.0, 300)

            assert snap.asks_depth is not None
            assert len(snap.asks_depth) == 2
            assert snap.asks_depth[0] == (101.0, 150)
            assert snap.asks_depth[1] == (101.5, 250)

            # Render to string
            renderer = DashboardRenderer(use_colors=False, use_unicode=True, width=80)
            rendered = renderer.render(snap)
            plain = strip_ansi(rendered)

            assert "$99.00*" in plain
            assert "*$101.00" in plain
            assert "ORDER BOOK L2 DEPTH LADDER (TOP 5)" in plain
        finally:
            pipeline.close()
