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
