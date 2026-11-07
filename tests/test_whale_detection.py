"""Unit and Integration Test Suite for Whale Order Detection in ChronosMatch.

Validates:
1. Configurable whale-order threshold in MarketFeedConfig and TradingPipeline.
2. Threshold validation (strictly positive integer, rejection of bools, zero, negatives, and non-ints).
3. Exact threshold boundary conditions (T-1 normal, T whale, T+1 whale).
4. Normal orders vs whale orders detection and volume tracking.
5. Detection of multi-level price sweeps (orders that consume liquidity across >=2 price tiers).
6. Telemetry propagation through DashboardMetricsCollector into DashboardMetricsSnapshot.
7. Standalone/manual metrics collection support.
8. Terminal dashboard rendering (Section 6 counter, alert banner, sweep highlight, ASCII and ANSI).
9. End-to-end pipeline execution with whale order detection active.
"""

from __future__ import annotations

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
    TerminalDashboard,
    format_counter_with_alert,
    format_int,
    strip_ansi,
    visible_len,
)
from chronosmatch.market_simulator import (
    MarketFeedConfig,
    SimulatorConfigError,
)
from chronosmatch.matching_engine import MatchingEngine
from chronosmatch.models import Order, OrderSide
from chronosmatch.pipeline import (
    TradingPipeline,
    TradingPipelineMetrics,
    create_pipeline,
)


def assert_field(rendered: str, label: str, value: str) -> None:
    """Verify that a given label and value appear together in the rendered output."""
    pattern = rf"{re.escape(label)}:\s+{re.escape(value)}"
    assert re.search(pattern, rendered) is not None, (
        f"Expected '{label}: {value}' in rendered output, but not found.\nOutput:\n{rendered}"
    )


# ==============================================================================
# 1. Configurable Whale Threshold & Strict Validation
# ==============================================================================


class TestWhaleConfiguration:
    """Verify whale threshold configuration and strict input validation."""

    def test_default_threshold(self) -> None:
        """Default threshold should be 1000 in MarketFeedConfig."""
        cfg = MarketFeedConfig()
        assert cfg.whale_order_threshold == 1000

    def test_custom_positive_threshold(self) -> None:
        """Custom strictly positive integer threshold is accepted."""
        cfg = MarketFeedConfig(whale_order_threshold=500)
        assert cfg.whale_order_threshold == 500

    @pytest.mark.parametrize("invalid_threshold", [0, -1, -500])
    def test_reject_zero_and_negative(self, invalid_threshold: int) -> None:
        """Threshold <= 0 must raise SimulatorConfigError."""
        with pytest.raises(SimulatorConfigError, match="strictly positive"):
            MarketFeedConfig(whale_order_threshold=invalid_threshold)

    @pytest.mark.parametrize("invalid_type", [True, False, 100.5, "1000", [1000], {"t": 1000}])
    def test_reject_non_integer_and_booleans(self, invalid_type: Any) -> None:
        """Non-integer types (including bool) must raise SimulatorConfigError."""
        with pytest.raises(SimulatorConfigError, match="strictly positive integer"):
            MarketFeedConfig(whale_order_threshold=invalid_type)

    def test_pipeline_inherits_threshold_from_config(self) -> None:
        """TradingPipeline should inherit threshold from MarketFeedConfig when not overridden."""
        cfg = MarketFeedConfig(whale_order_threshold=750)
        pipeline = create_pipeline(config=cfg)
        try:
            assert pipeline.whale_threshold == 750
            assert pipeline.metrics.whale_order_threshold == 750
        finally:
            pipeline.close()

    def test_pipeline_explicit_threshold_overrides_config(self) -> None:
        """Explicit whale_threshold passed to pipeline overrides config value."""
        cfg = MarketFeedConfig(whale_order_threshold=1000)
        pipeline = create_pipeline(config=cfg, whale_threshold=2500)
        try:
            assert pipeline.whale_threshold == 2500
            assert pipeline.metrics.whale_order_threshold == 2500
        finally:
            pipeline.close()


# ==============================================================================
# 2. Threshold Boundaries & Detection Logic
# ==============================================================================


class TestWhaleThresholdBoundaries:
    """Verify boundary conditions: T-1 (normal), T (whale), T+1 (whale)."""

    def test_boundary_conditions(self) -> None:
        threshold = 500
        pipeline = create_pipeline(whale_threshold=threshold)
        try:
            # T - 1: normal order (499)
            pipeline.process_order(
                Order(order_id="NORM-1", side=OrderSide.BUY, price=100.0, quantity=499, timestamp=1)
            )
            assert pipeline.metrics.whale_orders_detected == 0
            assert pipeline.metrics.whale_volume_executed == 0
            assert pipeline.last_whale_order is None

            # T: exact threshold (500) -> WHALE
            pipeline.process_order(
                Order(order_id="WHALE-BOUNDARY", side=OrderSide.BUY, price=100.0, quantity=500, timestamp=2)
            )
            assert pipeline.metrics.whale_orders_detected == 1
            assert pipeline.metrics.whale_volume_executed == 500
            assert pipeline.last_whale_order is not None
            assert pipeline.last_whale_order["order_id"] == "WHALE-BOUNDARY"
            assert pipeline.last_whale_order["quantity"] == 500

            # T + 1: strictly exceeding threshold (501) -> WHALE
            pipeline.process_order(
                Order(order_id="WHALE-EXCEED", side=OrderSide.SELL, price=105.0, quantity=501, timestamp=3)
            )
            assert pipeline.metrics.whale_orders_detected == 2
            assert pipeline.metrics.whale_volume_executed == 1001
            assert pipeline.last_whale_order["order_id"] == "WHALE-EXCEED"
            assert pipeline.last_whale_order["quantity"] == 501
            assert pipeline.last_whale_order["side"] == "SELL"
        finally:
            pipeline.close()

    def test_multiple_detections_and_volume_accumulation(self) -> None:
        """Verify cumulative counters across a stream of normal and whale orders."""
        pipeline = create_pipeline(whale_threshold=1000)
        try:
            orders = [
                Order(order_id="N1", side=OrderSide.BUY, price=100.0, quantity=100, timestamp=1),
                Order(order_id="W1", side=OrderSide.BUY, price=100.0, quantity=1500, timestamp=2),
                Order(order_id="N2", side=OrderSide.BUY, price=100.0, quantity=250, timestamp=3),
                Order(order_id="W2", side=OrderSide.BUY, price=100.0, quantity=3000, timestamp=4),
                Order(order_id="N3", side=OrderSide.BUY, price=100.0, quantity=999, timestamp=5),
                Order(order_id="W3", side=OrderSide.BUY, price=100.0, quantity=1000, timestamp=6),
            ]
            for ord in orders:
                pipeline.process_order(ord)

            assert pipeline.metrics.orders_processed == 6
            assert pipeline.metrics.whale_orders_detected == 3
            assert pipeline.metrics.whale_volume_executed == 5500
            assert pipeline.last_whale_order["order_id"] == "W3"
        finally:
            pipeline.close()

    def test_whale_callback_invocation(self) -> None:
        """Verify whale_callback is invoked for whale orders and NOT for normal orders."""
        events: list[tuple[str, int, int]] = []

        def callback(order: Order, trades: list[Any], swept: int) -> None:
            events.append((order.order_id, order.quantity, swept))

        pipeline = create_pipeline(whale_threshold=1000, whale_callback=callback)
        try:
            pipeline.process_order(
                Order(order_id="NORM", side=OrderSide.BUY, price=100.0, quantity=50, timestamp=1)
            )
            assert len(events) == 0

            pipeline.process_order(
                Order(order_id="WHALE", side=OrderSide.BUY, price=100.0, quantity=2500, timestamp=2)
            )
            assert len(events) == 1
            assert events[0] == ("WHALE", 2500, 0)
        finally:
            pipeline.close()


# ==============================================================================
# 3. Multi-Level Price Sweep Detection
# ==============================================================================


class TestMultiLevelPriceSweeps:
    """Verify detection of orders that exhaust and sweep multiple price levels."""

    def test_single_level_vs_multi_level_sweep(self) -> None:
        pipeline = create_pipeline(whale_threshold=1000)
        try:
            # 1. Place resting ask liquidity across three distinct price tiers
            pipeline.process_order(
                Order(order_id="ASK-101", side=OrderSide.SELL, price=101.0, quantity=100, timestamp=1)
            )
            pipeline.process_order(
                Order(order_id="ASK-102", side=OrderSide.SELL, price=102.0, quantity=100, timestamp=2)
            )
            pipeline.process_order(
                Order(order_id="ASK-103", side=OrderSide.SELL, price=103.0, quantity=100, timestamp=3)
            )

            # 2. Aggressive BUY that matches ONLY at single level (101.0)
            trades1 = pipeline.process_order(
                Order(order_id="BUY-SINGLE", side=OrderSide.BUY, price=101.0, quantity=50, timestamp=4)
            )
            assert len(trades1) == 1
            assert pipeline.metrics.multi_level_sweeps == 0

            # 3. Aggressive BUY that sweeps two price tiers (remaining 50 @ 101.0, plus 50 @ 102.0)
            trades2 = pipeline.process_order(
                Order(order_id="BUY-SWEEP-2", side=OrderSide.BUY, price=102.0, quantity=100, timestamp=5)
            )
            assert len(trades2) == 2
            assert pipeline.metrics.multi_level_sweeps == 1

            # 4. Whale aggressive BUY that sweeps remaining level 102.0 and level 103.0
            trades3 = pipeline.process_order(
                Order(order_id="WHALE-SWEEP-3", side=OrderSide.BUY, price=105.0, quantity=1500, timestamp=6)
            )
            assert len(trades3) >= 2
            assert pipeline.metrics.multi_level_sweeps == 2
            assert pipeline.metrics.whale_orders_detected == 1
            assert pipeline.last_whale_order is not None
            assert pipeline.last_whale_order["order_id"] == "WHALE-SWEEP-3"
            assert pipeline.last_whale_order["swept_levels"] >= 2
        finally:
            pipeline.close()


# ==============================================================================
# 4. Telemetry Collector & Snapshot Propagation
# ==============================================================================


class TestDashboardMetricsPropagation:
    """Verify whale metrics correctly populate DashboardMetricsCollector and Snapshot."""

    def test_snapshot_from_active_pipeline(self) -> None:
        pipeline = create_pipeline(whale_threshold=500)
        try:
            pipeline.process_order(
                Order(order_id="NORM", side=OrderSide.BUY, price=100.0, quantity=100, timestamp=1)
            )
            pipeline.process_order(
                Order(order_id="WHALE", side=OrderSide.BUY, price=100.0, quantity=1200, timestamp=2)
            )

            collector = DashboardMetricsCollector(pipeline=pipeline)
            snapshot = collector.take_snapshot()

            assert snapshot.whale_orders_detected == 1
            assert snapshot.whale_volume_executed == 1200
            assert snapshot.whale_order_threshold == 500
            assert snapshot.last_whale_order is not None
            assert snapshot.last_whale_order["order_id"] == "WHALE"

            diag = snapshot.to_dict()
            assert diag["whale_orders_detected"] == 1
            assert diag["whale_volume_executed"] == 1200
            assert diag["whale_order_threshold"] == 500
            assert diag["last_whale_order"]["order_id"] == "WHALE"
        finally:
            pipeline.close()

    def test_standalone_manual_collector(self) -> None:
        """Verify DashboardMetricsCollector standalone manual recording methods."""
        collector = DashboardMetricsCollector(whale_threshold=2000)

        # Before any whale events
        snap0 = collector.take_snapshot()
        assert snap0.whale_orders_detected is None
        assert snap0.whale_volume_executed is None
        assert snap0.whale_order_threshold == 2000

        # Record manual whale order
        collector.record_whale_order(quantity=3500, swept_levels=3, order_id="MANUAL-1", side="BUY", price=150.0)
        snap1 = collector.take_snapshot()
        assert snap1.whale_orders_detected == 1
        assert snap1.whale_volume_executed == 3500
        assert snap1.multi_level_sweeps == 1
        assert snap1.last_whale_order["order_id"] == "MANUAL-1"
        assert snap1.last_whale_order["swept_levels"] == 3

        # Record manual sweep
        collector.record_multi_level_sweep()
        snap2 = collector.take_snapshot()
        assert snap2.multi_level_sweeps == 2


# ==============================================================================
# 5. Terminal Dashboard UI Rendering
# ==============================================================================


class TestDashboardRendering:
    """Verify terminal formatting, alert banner display, and border alignment."""

    def test_render_zero_whale_orders(self) -> None:
        """When no whale orders detected, Section 6 displays 0 and alert banner is hidden."""
        pipeline = create_pipeline(whale_threshold=1000)
        try:
            pipeline.process_order(
                Order(order_id="N1", side=OrderSide.BUY, price=100.0, quantity=50, timestamp=1)
            )
            collector = DashboardMetricsCollector(pipeline=pipeline)
            renderer = DashboardRenderer(use_colors=False, use_unicode=False, width=80)
            out = renderer.render(collector.take_snapshot())

            assert "WHALE ORDER & SWEEP TELEMETRY" not in out
            assert len(out.splitlines()) == 40
        finally:
            pipeline.close()

    def test_render_whale_alert_banner_and_sweep_highlight(self) -> None:
        """When whale orders are detected, clear alert banner and sweep highlight are rendered."""
        pipeline = create_pipeline(whale_threshold=1000)
        try:
            # Place resting sells
            pipeline.process_order(Order(order_id="S1", side=OrderSide.SELL, price=101.0, quantity=50, timestamp=1))
            pipeline.process_order(Order(order_id="S2", side=OrderSide.SELL, price=102.0, quantity=50, timestamp=2))

            # Cross with whale order sweeping both levels
            pipeline.process_order(
                Order(order_id="WHALE-999", side=OrderSide.BUY, price=103.0, quantity=2500, timestamp=3)
            )

            collector = DashboardMetricsCollector(pipeline=pipeline)
            renderer = DashboardRenderer(use_colors=False, use_unicode=True, width=80)
            out = renderer.render(collector.take_snapshot())

            # Section 7 Alert Banner check
            assert "[WHALE ORDER & SWEEP TELEMETRY]" in out
            assert "(! 1 DETECTED !)" in out
            assert_field(out, "Whale Threshold", ">= 1,000")
            assert_field(out, "Whale Volume", "2,500")
            assert_field(out, "Whale Orders", "1")
            assert_field(out, "Multi-Lvl Sweeps", "1")
            assert "Last Whale Order:   BUY 2,500 @ $103.00 (WHALE-999)" in out
            assert ">>> MULTI-LEVEL SWEEP: Executed across 2 price levels! <<<" in out

            # Check border alignment: all lines must have identical visible length
            lines = out.split("\n")
            for idx, line in enumerate(lines):
                assert visible_len(line) == 80, f"Line {idx} width mismatch ({visible_len(line)} != 80): {line}"
        finally:
            pipeline.close()

    def test_render_with_ansi_colors(self) -> None:
        """Verify ANSI color styling preserves exact row alignment."""
        collector = DashboardMetricsCollector(whale_threshold=1000)
        collector.record_whale_order(quantity=5000, swept_levels=3, order_id="COLORED-WHALE", side="BUY", price=250.0)

        renderer = DashboardRenderer(use_colors=True, use_unicode=True, width=80)
        out = renderer.render(collector.take_snapshot())

        # ANSI escapes should be present
        assert "\033[" in out
        assert "WHALE ORDER & SWEEP TELEMETRY" in out
        assert "MULTI-LEVEL SWEEP" in out

        # Stripped lines must all be exact width
        for idx, line in enumerate(out.split("\n")):
            assert visible_len(line) == 80, f"Line {idx} width {visible_len(line)} != 80"


# ==============================================================================
# 6. End-to-End Pipeline Integration
# ==============================================================================


class TestEndToEndPipelineWhaleIntegration:
    """Verify TradingPipeline runs synchronously and asynchronously with whale detection."""

    def test_pipeline_run_sync_with_whale_detection(self) -> None:
        """Run simulated pipeline with a low whale threshold and verify metrics are tracked."""
        cfg = MarketFeedConfig(
            min_quantity=10,
            max_quantity=100,
            whale_order_threshold=50,  # Any order >= 50 will trigger whale detection
        )
        pipeline = create_pipeline(config=cfg, whale_threshold=50)
        try:
            metrics = pipeline.run_sync(total_orders=60)
            assert metrics.orders_processed >= 50
            # Since max_quantity=100 and threshold=50, a significant fraction will be whales
            assert metrics.whale_orders_detected > 0
            assert metrics.whale_volume_executed > 0
            assert metrics.whale_order_threshold == 50
        finally:
            pipeline.close()
