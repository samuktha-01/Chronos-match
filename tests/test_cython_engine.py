"""Comprehensive test suite for Cython Matching Engine and Order Book.

Validates:
- 100% functional equivalence between Python reference MatchingEngine and CythonMatchingEngine.
- Full fills, partial fills, multi-level sweeps, and unmatched resting orders.
- Strict FIFO time priority and Maker-Price execution rule.
- Full precision preservation for float, int, and Decimal prices.
- Strict validation and identical error types for corrupt/invalid orders.
- Standalone CythonOrderBook and CythonPriceLevel data structures.
- Compilation diagnostics and environment build limitation reporting.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import math
import pytest

from chronosmatch import (
    CythonMatchingEngine,
    CythonOrderBook,
    CythonPriceLevel,
    DuplicateOrderError,
    MatchingEngine,
    Order,
    OrderNotFoundError,
    OrderSide,
    get_compilation_status,
    is_cython_compiled,
)
from chronosmatch.cython_engine import _PyCythonMatchingEngine
from chronosmatch.dashboard.metrics import (
    BoundedLatencyTracker,
    DashboardMetricsCollector,
    DashboardMetricsSnapshot,
)
from chronosmatch.dashboard.pipeline_runner import LatencyInstrumentedEngine, PipelineDashboardRunner
from chronosmatch.dashboard.terminal import TerminalDashboard
from chronosmatch.pipeline import TradingPipeline, create_pipeline


# ==============================================================================
# Helper Verification Functions
# ==============================================================================


def assert_engine_equality(engine_py: MatchingEngine, engine_cy: CythonMatchingEngine) -> None:
    """Verify that Python and Cython engines have identical state, trades, and book depth."""
    # Compare trade list lengths
    assert len(engine_py.trades) == len(engine_cy.trades)
    assert engine_py.total_trades == engine_cy.total_trades
    assert engine_py.total_volume_traded == engine_cy.total_volume_traded

    # Compare individual trade fields
    for t_py, t_cy in zip(engine_py.trades, engine_cy.trades):
        assert t_py.trade_id == t_cy.trade_id
        assert t_py.resting_order_id == t_cy.resting_order_id
        assert t_py.incoming_order_id == t_cy.incoming_order_id
        assert t_py.price == t_cy.price
        assert type(t_py.price) is type(t_cy.price)
        assert t_py.quantity == t_cy.quantity

    # Compare order book top-of-book metrics
    assert engine_py.best_bid == engine_cy.best_bid
    assert engine_py.best_ask == engine_cy.best_ask
    assert engine_py.spread == engine_cy.spread
    assert len(engine_py.order_book) == len(engine_cy.order_book)


# ==============================================================================
# 1. Functional Equivalence Tests (Python Reference vs Cython Engine)
# ==============================================================================


class TestCythonEngineEquivalence:
    """Verify side-by-side equivalence of MatchingEngine and CythonMatchingEngine."""

    def test_single_full_match_equivalence(self) -> None:
        """Verify identical execution for single crossing BUY and SELL orders."""
        py_engine = MatchingEngine()
        cy_engine = CythonMatchingEngine()

        maker = Order(order_id="M-1", side=OrderSide.SELL, price=100.50, quantity=50)
        taker = Order(order_id="T-1", side=OrderSide.BUY, price=100.50, quantity=50)

        # Process through Python engine
        trades_py = py_engine.process_order(Order("M-1", OrderSide.SELL, 100.50, 50))
        trades_py += py_engine.process_order(Order("T-1", OrderSide.BUY, 100.50, 50))

        # Process through Cython engine
        trades_cy = cy_engine.process_order(Order("M-1", OrderSide.SELL, 100.50, 50))
        trades_cy += cy_engine.process_order(Order("T-1", OrderSide.BUY, 100.50, 50))

        assert len(trades_py) == len(trades_cy) == 1
        assert trades_py[0].price == trades_cy[0].price == 100.50
        assert trades_py[0].quantity == trades_cy[0].quantity == 50
        assert_engine_equality(py_engine, cy_engine)

    def test_maker_price_execution_and_price_improvement_equivalence(self) -> None:
        """Verify maker-price execution rule and price improvement on both engines."""
        py_engine = MatchingEngine()
        cy_engine = CythonMatchingEngine()

        # Resting ask at 100.0, incoming aggressive bid at 105.0 -> executes at 100.0 (maker price)
        py_engine.process_order(Order("M-ASK", OrderSide.SELL, 100.0, 10))
        t_py = py_engine.process_order(Order("T-BID", OrderSide.BUY, 105.0, 10))

        cy_engine.process_order(Order("M-ASK", OrderSide.SELL, 100.0, 10))
        t_cy = cy_engine.process_order(Order("T-BID", OrderSide.BUY, 105.0, 10))

        assert len(t_py) == len(t_cy) == 1
        assert t_py[0].price == t_cy[0].price == 100.0  # Maker price, not taker's 105.0
        assert_engine_equality(py_engine, cy_engine)

    def test_partial_fill_taker_smaller_than_maker_equivalence(self) -> None:
        """Verify partial fill when incoming taker order is smaller than resting maker order."""
        py_engine = MatchingEngine()
        cy_engine = CythonMatchingEngine()

        # Resting BUY for 100 units at 50.00
        # Incoming SELL for 30 units at 50.00 -> 30 units filled, 70 units remain resting
        py_engine.process_order(Order("M-BUY", OrderSide.BUY, 50.0, 100))
        trades_py = py_engine.process_order(Order("T-SELL", OrderSide.SELL, 50.0, 30))

        cy_engine.process_order(Order("M-BUY", OrderSide.BUY, 50.0, 100))
        trades_cy = cy_engine.process_order(Order("T-SELL", OrderSide.SELL, 50.0, 30))

        assert len(trades_py) == len(trades_cy) == 1
        assert trades_py[0].quantity == trades_cy[0].quantity == 30
        assert py_engine.best_bid == cy_engine.best_bid == 50.0
        assert py_engine.order_book.bids.total_volume == cy_engine.order_book.bids.total_volume == 70
        assert_engine_equality(py_engine, cy_engine)

    def test_partial_fill_taker_larger_than_maker_equivalence(self) -> None:
        """Verify partial fill when incoming taker order exceeds maker and rests remaining balance."""
        py_engine = MatchingEngine()
        cy_engine = CythonMatchingEngine()

        # Resting SELL for 40 units at 200.0
        # Incoming BUY for 100 units at 200.0 -> 40 units filled, 60 units rest as BUY at 200.0
        py_engine.process_order(Order("M-SELL", OrderSide.SELL, 200.0, 40))
        trades_py = py_engine.process_order(Order("T-BUY", OrderSide.BUY, 200.0, 100))

        cy_engine.process_order(Order("M-SELL", OrderSide.SELL, 200.0, 40))
        trades_cy = cy_engine.process_order(Order("T-BUY", OrderSide.BUY, 200.0, 100))

        assert len(trades_py) == len(trades_cy) == 1
        assert trades_py[0].quantity == trades_cy[0].quantity == 40
        assert py_engine.best_ask == cy_engine.best_ask is None
        assert py_engine.best_bid == cy_engine.best_bid == 200.0
        assert py_engine.order_book.bids.total_volume == cy_engine.order_book.bids.total_volume == 60
        assert_engine_equality(py_engine, cy_engine)

    def test_fifo_time_priority_multiple_makers_equivalence(self) -> None:
        """Verify strict arrival time ordering when multiple orders rest at identical price."""
        py_engine = MatchingEngine()
        cy_engine = CythonMatchingEngine()

        # Two resting BUY orders at 100.0: M1 arrived first (qty 50), M2 arrived second (qty 50)
        # Taker SELL arrives for 75 units -> fills M1 (50) fully, M2 (25) partially
        for eng in (py_engine, cy_engine):
            eng.process_order(Order("M-1", OrderSide.BUY, 100.0, 50, sequence_number=1))
            eng.process_order(Order("M-2", OrderSide.BUY, 100.0, 50, sequence_number=2))
            trades = eng.process_order(Order("T-1", OrderSide.SELL, 100.0, 75, sequence_number=3))

            assert len(trades) == 2
            assert trades[0].resting_order_id == "M-1"
            assert trades[0].quantity == 50
            assert trades[1].resting_order_id == "M-2"
            assert trades[1].quantity == 25
            assert eng.order_book.bids.total_volume == 25

        assert_engine_equality(py_engine, cy_engine)

    def test_multi_level_sweep_equivalence(self) -> None:
        """Verify aggressive order sweeping across multiple resting price levels."""
        py_engine = MatchingEngine()
        cy_engine = CythonMatchingEngine()

        # Resting asks at 101.0 (qty 10), 102.0 (qty 20), 103.0 (qty 30)
        # Taker BUY arrives at 105.0 for 45 units -> sweeps 101.0 (10), 102.0 (20), 103.0 (15)
        for eng in (py_engine, cy_engine):
            eng.process_order(Order("A-1", OrderSide.SELL, 101.0, 10))
            eng.process_order(Order("A-2", OrderSide.SELL, 102.0, 20))
            eng.process_order(Order("A-3", OrderSide.SELL, 103.0, 30))
            trades = eng.process_order(Order("T-SWEEP", OrderSide.BUY, 105.0, 45))

            assert len(trades) == 3
            assert trades[0].price == 101.0 and trades[0].quantity == 10
            assert trades[1].price == 102.0 and trades[1].quantity == 20
            assert trades[2].price == 103.0 and trades[2].quantity == 15
            assert eng.best_ask == 103.0
            assert eng.order_book.asks.total_volume == 15

        assert_engine_equality(py_engine, cy_engine)

    def test_unmatched_resting_orders_equivalence(self) -> None:
        """Verify non-crossing orders rest in order book depth without generating trades."""
        py_engine = MatchingEngine()
        cy_engine = CythonMatchingEngine()

        orders = [
            Order("B-1", OrderSide.BUY, 99.0, 10),
            Order("B-2", OrderSide.BUY, 98.5, 20),
            Order("A-1", OrderSide.SELL, 101.0, 15),
            Order("A-2", OrderSide.SELL, 102.0, 25),
        ]

        for ord_item in orders:
            t_py = py_engine.process_order(Order(ord_item.order_id, ord_item.side, ord_item.price, ord_item.quantity))
            t_cy = cy_engine.process_order(Order(ord_item.order_id, ord_item.side, ord_item.price, ord_item.quantity))
            assert len(t_py) == len(t_cy) == 0

        assert py_engine.best_bid == cy_engine.best_bid == 99.0
        assert py_engine.best_ask == cy_engine.best_ask == 101.0
        assert py_engine.spread == cy_engine.spread == 2.0
        assert_engine_equality(py_engine, cy_engine)

    def test_heterogeneous_price_types_equivalence(self) -> None:
        """Verify decimal, float, and integer price types behave identically."""
        # Test Decimal prices
        py_dec = MatchingEngine()
        cy_dec = CythonMatchingEngine()
        py_dec.process_order(Order("DEC-1", OrderSide.BUY, Decimal("150.25"), 10))
        py_dec.process_order(Order("DEC-2", OrderSide.SELL, Decimal("155.75"), 20))
        cy_dec.process_order(Order("DEC-1", OrderSide.BUY, Decimal("150.25"), 10))
        cy_dec.process_order(Order("DEC-2", OrderSide.SELL, Decimal("155.75"), 20))
        assert_engine_equality(py_dec, cy_dec)

        # Test Float prices
        py_flt = MatchingEngine()
        cy_flt = CythonMatchingEngine()
        py_flt.process_order(Order("FLT-1", OrderSide.BUY, 150.25, 10))
        py_flt.process_order(Order("FLT-2", OrderSide.SELL, 155.75, 20))
        cy_flt.process_order(Order("FLT-1", OrderSide.BUY, 150.25, 10))
        cy_flt.process_order(Order("FLT-2", OrderSide.SELL, 155.75, 20))
        assert_engine_equality(py_flt, cy_flt)

        # Test Int prices
        py_int = MatchingEngine()
        cy_int = CythonMatchingEngine()
        py_int.process_order(Order("INT-1", OrderSide.BUY, 150, 10))
        py_int.process_order(Order("INT-2", OrderSide.SELL, 155, 20))
        cy_int.process_order(Order("INT-1", OrderSide.BUY, 150, 10))
        cy_int.process_order(Order("INT-2", OrderSide.SELL, 155, 20))
        assert_engine_equality(py_int, cy_int)


    def test_engine_clear_equivalence(self) -> None:
        """Verify clear() resets trades, trade_counter, and book depth identically."""
        py_engine = MatchingEngine()
        cy_engine = CythonMatchingEngine()

        py_engine.process_order(Order("O-1", OrderSide.BUY, 100.0, 10))
        cy_engine.process_order(Order("O-1", OrderSide.BUY, 100.0, 10))

        py_engine.clear()
        cy_engine.clear()

        assert py_engine.total_trades == cy_engine.total_trades == 0
        assert py_engine.best_bid == cy_engine.best_bid is None
        assert len(py_engine.order_book) == len(cy_engine.order_book) == 0


# ==============================================================================
# 2. Validation & Error Handling Tests
# ==============================================================================


class TestCythonEngineValidationAndErrors:
    """Verify strict input validation and identical exceptions between implementations."""

    def test_non_order_instance_raises_type_error(self) -> None:
        engine = CythonMatchingEngine()
        with pytest.raises(TypeError, match="Expected Order instance"):
            engine.process_order("not-an-order")  # type: ignore[arg-type]

    def test_duplicate_order_id_raises_duplicate_error(self) -> None:
        engine = CythonMatchingEngine()
        engine.process_order(Order("DUP-1", OrderSide.BUY, 100.0, 10))
        with pytest.raises(DuplicateOrderError, match="already exists"):
            engine.process_order(Order("DUP-1", OrderSide.BUY, 100.0, 10))

    def test_empty_or_whitespace_order_id_raises_value_error(self) -> None:
        engine = CythonMatchingEngine()
        with pytest.raises(ValueError, match="order_id cannot be empty or whitespace"):
            # Construct order bypassing __post_init__ or direct check
            order = Order("valid", OrderSide.BUY, 100.0, 10)
            order.order_id = "   "
            engine.process_order(order)

    def test_non_positive_quantity_raises_value_error(self) -> None:
        engine = CythonMatchingEngine()
        order = Order("V-1", OrderSide.BUY, 100.0, 10)
        order.quantity = 0
        with pytest.raises(ValueError, match="quantity must be strictly positive"):
            engine.process_order(order)

    def test_non_finite_price_raises_value_error(self) -> None:
        engine = CythonMatchingEngine()
        order = Order("V-2", OrderSide.BUY, 100.0, 10)
        order.price = float("nan")
        with pytest.raises(ValueError, match="price must be finite"):
            engine.process_order(order)


# ==============================================================================
# 3. Cython Order Book and Price Level Component Tests
# ==============================================================================


class TestCythonOrderBookComponents:
    """Verify CythonPriceLevel and CythonOrderBook independent functionality."""

    def test_price_level_operations(self) -> None:
        level = CythonPriceLevel(100.0)
        assert level.price == 100.0
        assert level.total_volume == 0
        assert len(level) == 0

        o1 = Order("P-1", OrderSide.BUY, 100.0, 30)
        o2 = Order("P-2", OrderSide.BUY, 100.0, 20)
        level.add_order(o1)
        level.add_order(o2)

        assert level.total_volume == 50
        assert len(level) == 2
        assert level.head_order.order_id == "P-1"

        level.reduce_order_quantity("P-1", 10)
        assert level.get_order("P-1").quantity == 20
        assert level.total_volume == 40

        removed = level.remove_order("P-1")
        assert removed.order_id == "P-1"
        assert level.total_volume == 20
        assert level.head_order.order_id == "P-2"

    def test_order_book_depth_and_volumes(self) -> None:
        book = CythonOrderBook()
        book.add_order(Order("B1", OrderSide.BUY, 99.0, 10))
        book.add_order(Order("B2", OrderSide.BUY, 98.0, 20))
        book.add_order(Order("A1", OrderSide.SELL, 101.0, 30))

        assert book.bid_depth == 2
        assert book.ask_depth == 1
        assert book.bid_volume == 30
        assert book.ask_volume == 30
        assert book.total_volume == 60
        assert book.spread == 2.0
        assert "B1" in book
        assert "NONEXISTENT" not in book


# ==============================================================================
# 4. Compilation Diagnostics & Environment Reporting
# ==============================================================================


class TestCythonCompilationStatusAndDiagnostics:
    """Verify Cython build configuration and environment diagnostics."""

    def test_compilation_status_structure(self) -> None:
        status = get_compilation_status()
        assert isinstance(status, dict)
        assert "is_compiled" in status
        assert "c_source_exists" in status
        assert "pyx_source_exists" in status
        assert "build_limitation" in status

        assert status["pyx_source_exists"] is True
        assert status["c_source_exists"] is True

    def test_generated_c_source_file_exists_and_valid(self) -> None:
        c_path = Path(__file__).parent.parent / "src" / "chronosmatch" / "_cython_engine.c"
        assert c_path.exists(), "src/chronosmatch/_cython_engine.c must be generated"
        # Cython 3.3.0 generates ~1.2 MB C file
        assert c_path.stat().st_size > 100_000, "Generated C file should be substantial (> 100 KB)"


# ==============================================================================
# 5. Cancellation and Telemetry Engine Detection Tests
# ==============================================================================


class TestCythonCancellationAndEngineDetection:
    """Verify CythonMatchingEngine order cancellation and dashboard engine detection."""

    def test_cancel_resting_buy_order(self) -> None:
        """Verify cancellation of a resting buy order updates book state and metrics."""
        engine = CythonMatchingEngine()
        engine.process_order(Order("BUY-1", OrderSide.BUY, 99.0, 10))
        assert engine.best_bid == 99.0
        assert engine.order_book.bid_volume == 10
        assert "BUY-1" in engine.order_book

        cancelled = engine.cancel_order("BUY-1")
        assert cancelled.order_id == "BUY-1"
        assert cancelled.quantity == 10
        assert engine.best_bid is None
        assert engine.order_book.bid_volume == 0
        assert "BUY-1" not in engine.order_book
        assert len(engine.order_book) == 0

    def test_cancel_resting_sell_order(self) -> None:
        """Verify cancellation of a resting sell order updates book state and metrics."""
        engine = CythonMatchingEngine()
        engine.process_order(Order("SELL-1", OrderSide.SELL, 101.0, 25))
        assert engine.best_ask == 101.0
        assert engine.order_book.ask_volume == 25
        assert "SELL-1" in engine.order_book

        cancelled = engine.cancel_order("SELL-1")
        assert cancelled.order_id == "SELL-1"
        assert cancelled.quantity == 25
        assert engine.best_ask is None
        assert engine.order_book.ask_volume == 0
        assert "SELL-1" not in engine.order_book
        assert len(engine.order_book) == 0

    def test_cancel_unknown_order_raises(self) -> None:
        """Verify cancelling a non-existent or invalid order raises OrderNotFoundError or ValueError."""
        engine = CythonMatchingEngine()
        with pytest.raises(OrderNotFoundError):
            engine.cancel_order("NONEXISTENT")

        with pytest.raises(ValueError):
            engine.cancel_order("")

    def test_cancel_multiple_orders_sharing_price_level(self) -> None:
        """Verify FIFO queue order preservation when cancelling an order sharing a price level."""
        engine = CythonMatchingEngine()
        engine.process_order(Order("B-1", OrderSide.BUY, 100.0, 10))
        engine.process_order(Order("B-2", OrderSide.BUY, 100.0, 20))

        assert engine.order_book.bid_volume == 30
        assert engine.order_book.bids.depth() == [(100.0, 30)]

        # Cancel head order B-1
        c1 = engine.cancel_order("B-1")
        assert c1.order_id == "B-1"
        assert engine.order_book.bid_volume == 20
        assert engine.best_bid == 100.0
        assert engine.order_book.bids.best_order.order_id == "B-2"

        # Cancel remaining order B-2; level should be pruned
        c2 = engine.cancel_order("B-2")
        assert c2.order_id == "B-2"
        assert engine.best_bid is None
        assert engine.order_book.bid_volume == 0
        assert len(engine.order_book) == 0

    def test_matching_cancellation_behavior_between_python_and_cython(self) -> None:
        """Verify identical cancellation behavior between Python reference and Cython engines."""
        py_engine = MatchingEngine()
        cy_engine = CythonMatchingEngine()

        for eng in (py_engine, cy_engine):
            eng.process_order(Order("B1", OrderSide.BUY, 99.0, 10))
            eng.process_order(Order("B2", OrderSide.BUY, 98.0, 15))
            eng.process_order(Order("A1", OrderSide.SELL, 101.0, 20))
            eng.process_order(Order("A2", OrderSide.SELL, 102.0, 25))

        py_cancelled = py_engine.cancel_order("B1")
        cy_cancelled = cy_engine.cancel_order("B1")

        assert py_cancelled.order_id == cy_cancelled.order_id == "B1"
        assert py_cancelled.quantity == cy_cancelled.quantity == 10
        assert_engine_equality(py_engine, cy_engine)

        # Cancel unknown order raises identically
        with pytest.raises(OrderNotFoundError):
            py_engine.cancel_order("NON_EXISTENT")
        with pytest.raises(OrderNotFoundError):
            cy_engine.cancel_order("NON_EXISTENT")

    def test_pipeline_cancellation_with_cython_engine(self) -> None:
        """Verify end-to-end pipeline cancellation functions with CythonMatchingEngine."""
        pipeline = create_pipeline(engine=CythonMatchingEngine())
        try:
            order = Order("PIPE-BUY", OrderSide.BUY, 150.0, 50)
            pipeline.process_order(order)
            assert pipeline.best_bid == 150.0

            cancelled = pipeline.cancel_order("PIPE-BUY")
            assert cancelled.order_id == "PIPE-BUY"
            assert pipeline.best_bid is None
            assert pipeline.metrics.orders_cancelled == 1

            with pytest.raises(OrderNotFoundError):
                pipeline.cancel_order("PIPE-BUY")
        finally:
            pipeline.cleanup()

    def test_engine_detection_through_latency_wrapper(self) -> None:
        """Verify is_cython_accelerated returns True through LatencyInstrumentedEngine proxy."""
        tracker = BoundedLatencyTracker()
        cy_engine = CythonMatchingEngine()
        wrapped_cy = LatencyInstrumentedEngine(cy_engine, tracker)
        assert wrapped_cy.is_cython_accelerated is True

        py_engine = MatchingEngine()
        wrapped_py = LatencyInstrumentedEngine(py_engine, tracker)
        assert wrapped_py.is_cython_accelerated is False

        pipeline = create_pipeline(engine=CythonMatchingEngine())
        runner = PipelineDashboardRunner(pipeline=pipeline, interactive=False)
        try:
            assert pipeline.is_cython_accelerated is True
            assert runner.collector.source_name == "ChronosMatch-Cython"
            snapshot = runner.collector.take_snapshot()
            assert snapshot.is_cython_accelerated is True
        finally:
            runner.close()

    def test_dashboard_render_shows_core_cython(self) -> None:
        """Verify dashboard terminal rendering accurately distinguishes Cython from Python core."""
        from chronosmatch.dashboard.terminal import DashboardRenderer

        renderer = DashboardRenderer(use_colors=False)

        pipeline_cy = create_pipeline(engine=CythonMatchingEngine())
        try:
            collector_cy = DashboardMetricsCollector(pipeline=pipeline_cy)
            snapshot_cy = collector_cy.take_snapshot()
            assert snapshot_cy.is_cython_accelerated is True
            rendered_cy = renderer.render(snapshot_cy)
            assert "Core: Cython" in rendered_cy
            assert "Core: Python" not in rendered_cy
        finally:
            pipeline_cy.cleanup()

        pipeline_py = create_pipeline(engine=MatchingEngine())
        try:
            collector_py = DashboardMetricsCollector(pipeline=pipeline_py)
            snapshot_py = collector_py.take_snapshot()
            assert snapshot_py.is_cython_accelerated is False
            rendered_py = renderer.render(snapshot_py)
            assert "Core: Python" in rendered_py
            assert "Core: Cython" not in rendered_py
        finally:
            pipeline_py.cleanup()

    def test_fallback_detection_remains_correct(self) -> None:
        """Verify fallback _PyCythonMatchingEngine is not mislabeled as native Cython."""
        tracker = BoundedLatencyTracker()
        fallback_engine = _PyCythonMatchingEngine()
        wrapped = LatencyInstrumentedEngine(fallback_engine, tracker)
        assert wrapped.is_cython_accelerated is False

        pipeline = create_pipeline(engine=fallback_engine)
        try:
            assert pipeline.is_cython_accelerated is False
        finally:
            pipeline.cleanup()


# ==============================================================================
# 5. Pipeline Default Engine Selection & Fallback Tests
# ==============================================================================


class TestPipelineEngineSelection:
    """Verify default engine selection and fallback behavior in TradingPipeline."""

    def test_get_default_matching_engine_selects_cython_when_compiled(self) -> None:
        """Verify get_default_matching_engine() returns CythonMatchingEngine when compiled."""
        from chronosmatch import get_default_matching_engine

        engine = get_default_matching_engine()
        if is_cython_compiled():
            from chronosmatch.cython_engine import _CompiledCythonMatchingEngine

            assert _CompiledCythonMatchingEngine is not None
            assert isinstance(engine, _CompiledCythonMatchingEngine)
        else:
            assert isinstance(engine, MatchingEngine)

    def test_default_pipeline_uses_native_cython_when_available(self) -> None:
        """Verify create_pipeline() selects CythonMatchingEngine by default when compiled."""
        pipeline = create_pipeline(capacity=64)
        try:
            if is_cython_compiled():
                from chronosmatch.cython_engine import _CompiledCythonMatchingEngine

                assert _CompiledCythonMatchingEngine is not None
                assert isinstance(pipeline.engine, _CompiledCythonMatchingEngine)
                assert pipeline.is_cython_accelerated is True
            else:
                assert isinstance(pipeline.engine, MatchingEngine)
                assert pipeline.is_cython_accelerated is False
        finally:
            pipeline.cleanup()

    def test_default_pipeline_falls_back_to_python_when_cython_unavailable(self) -> None:
        """Verify pipeline falls back to MatchingEngine when native acceleration is unavailable."""
        from unittest.mock import patch
        from chronosmatch import get_default_matching_engine

        with patch("chronosmatch.pipeline.is_cython_compiled", return_value=False):
            fallback_engine = get_default_matching_engine()
            assert isinstance(fallback_engine, MatchingEngine)

            pipeline = create_pipeline(capacity=64)
            try:
                assert isinstance(pipeline.engine, MatchingEngine)
                assert pipeline.is_cython_accelerated is False
            finally:
                pipeline.cleanup()

    def test_explicit_engine_takes_precedence(self) -> None:
        """Verify explicitly provided engine override is preserved over the default."""
        # 1. Explicit MatchingEngine (pure-Python override)
        py_engine = MatchingEngine()
        pipeline_py = create_pipeline(engine=py_engine, capacity=64)
        try:
            assert pipeline_py.engine is py_engine
            assert pipeline_py.is_cython_accelerated is False
        finally:
            pipeline_py.cleanup()

        # 2. Explicit CythonMatchingEngine
        cy_engine = CythonMatchingEngine()
        pipeline_cy = create_pipeline(engine=cy_engine, capacity=64)
        try:
            assert pipeline_cy.engine is cy_engine
            assert pipeline_cy.is_cython_accelerated is is_cython_compiled()
        finally:
            pipeline_cy.cleanup()

    def test_pipeline_matching_correctness_under_default_engine(self) -> None:
        """Verify matching and book state under default pipeline match reference engine."""
        pipeline = create_pipeline(capacity=64)
        ref_engine = MatchingEngine()
        try:
            maker = Order("M-01", OrderSide.SELL, 100.0, 50, 1_000_000_000, 1)
            taker = Order("T-01", OrderSide.BUY, 100.0, 30, 2_000_000_000, 2)

            trades_pipe = pipeline.process_order(maker)
            trades_ref = ref_engine.process_order(maker)
            assert len(trades_pipe) == len(trades_ref) == 0

            trades_pipe = pipeline.process_order(taker)
            trades_ref = ref_engine.process_order(taker)
            assert len(trades_pipe) == len(trades_ref) == 1
            assert trades_pipe[0].price == trades_ref[0].price == 100.0
            assert trades_pipe[0].quantity == trades_ref[0].quantity == 30
            assert pipeline.best_ask == ref_engine.best_ask == 100.0
            assert pipeline.best_bid == ref_engine.best_bid is None
        finally:
            pipeline.cleanup()

    def test_pipeline_constructor_and_factory_engine_consistency(self) -> None:
        """Verify TradingPipeline constructor and create_pipeline factory select engines consistently."""
        from chronosmatch.pipeline import TradingPipeline

        pipeline_ctor = TradingPipeline(capacity=64)
        pipeline_factory = create_pipeline(capacity=64)
        try:
            assert type(pipeline_ctor.engine) is type(pipeline_factory.engine)
            assert pipeline_ctor.is_cython_accelerated == pipeline_factory.is_cython_accelerated
        finally:
            pipeline_ctor.cleanup()
            pipeline_factory.cleanup()

    def test_default_pipeline_cancellation_and_persistence_recovery(self, tmp_path: Path) -> None:
        """Verify cancellation, WAL persistence, and recovery succeed under default engine selection."""
        from chronosmatch.persistence import WriteAheadLog, DurabilityPolicy

        wal_file = tmp_path / "default_engine_test.wal"
        pipeline = create_pipeline(capacity=64, wal_path=wal_file, durability_policy=DurabilityPolicy.FLUSH_ON_WRITE)
        try:
            # 1. Place resting maker order
            maker = Order("M-REC-1", OrderSide.SELL, 120.0, 40, 1_000_000_000, 1)
            pipeline.process_order(maker)
            assert pipeline.best_ask == 120.0

            # 2. Cancel order through pipeline
            cancelled = pipeline.cancel_order("M-REC-1")
            assert cancelled.order_id == "M-REC-1"
            assert pipeline.best_ask is None
            assert pipeline.metrics.orders_cancelled == 1

            # 3. Match another pair to create trade events in WAL
            o1 = Order("M-REC-2", OrderSide.BUY, 110.0, 25, 2_000_000_000, 2)
            o2 = Order("T-REC-2", OrderSide.SELL, 110.0, 25, 3_000_000_000, 3)
            pipeline.process_order(o1)
            trades = pipeline.process_order(o2)
            assert len(trades) == 1
            assert pipeline.metrics.trades_executed == 1
        finally:
            pipeline.close()

        # 4. Recover pipeline from WAL log
        recovered_pipeline = TradingPipeline.recover(wal_path=wal_file, capacity=64)
        try:
            assert len(recovered_pipeline.trades) == 1
            assert recovered_pipeline.trades[0].quantity == 25
            assert recovered_pipeline.best_ask is None
            assert recovered_pipeline.best_bid is None
        finally:
            recovered_pipeline.close()

    def test_dashboard_reports_correct_acceleration_under_default_pipeline(self) -> None:
        """Verify dashboard runner accurately reflects default pipeline engine acceleration."""
        pipeline = create_pipeline(capacity=64)
        runner = PipelineDashboardRunner(pipeline=pipeline, interactive=False)
        try:
            snapshot = runner.collector.take_snapshot()
            rendered = runner.dashboard.render_to_string(snapshot)
            if is_cython_compiled():
                assert snapshot.is_cython_accelerated is True
                assert "Core: Cython" in rendered
                assert "Core: Python" not in rendered
            else:
                assert snapshot.is_cython_accelerated is False
                assert "Core: Python" in rendered
                assert "Core: Cython" not in rendered
        finally:
            runner.close()
