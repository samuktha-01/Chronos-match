"""Comprehensive test suite for the ChronosMatch Limit Order Book (LOB)."""

from decimal import Decimal
import math

import pytest

from chronosmatch.models import Order, OrderSide, Side
from chronosmatch.order_book import (
    DuplicateOrderError,
    LimitOrderBook,
    OrderBook,
    OrderBookError,
    OrderBookSide,
    OrderNotFoundError,
    PriceLevel,
)


class TestOrderBookInitializationAndEmptyState:
    """Test suite for initial book creation and empty state behavior."""

    def test_new_book_is_empty(self) -> None:
        """Verify that a newly instantiated LimitOrderBook is completely empty."""
        book = LimitOrderBook()

        assert len(book) == 0
        assert book.is_empty is True
        assert book.bids_count == 0
        assert book.asks_count == 0
        assert book.total_volume == 0
        assert book.bids.is_empty is True
        assert book.asks.is_empty is True
        assert book.bids.order_count == 0
        assert book.asks.order_count == 0

    def test_empty_book_best_prices_and_orders(self) -> None:
        """Verify that best bid/ask and top orders return None on an empty book."""
        book = LimitOrderBook()

        assert book.best_bid is None
        assert book.get_best_bid() is None
        assert book.best_ask is None
        assert book.get_best_ask() is None

        assert book.best_bid_order is None
        assert book.get_best_bid_order() is None
        assert book.best_ask_order is None
        assert book.get_best_ask_order() is None

        assert book.spread is None

    def test_empty_book_queries(self) -> None:
        """Verify that querying arbitrary prices on an empty book returns sensible empty values."""
        book = LimitOrderBook()

        assert book.get_orders_at_price(100.0) == []
        assert book.get_orders_at_price(100.0, side=OrderSide.BUY) == []
        assert book.get_orders_at_price(100.0, side=OrderSide.SELL) == []
        assert book.get_volume_at_price(100.0) == 0
        assert book.get_order_count_at_price(100.0) == 0

        assert book.get_bids_depth() == []
        assert book.get_asks_depth() == []
        assert book.get_l2_depth() == ([], [])

    def test_empty_book_order_lookup(self) -> None:
        """Verify that looking up non-existent orders in an empty book behaves properly."""
        book = LimitOrderBook()

        assert "ord-001" not in book
        assert book.get_order("ord-001") is None

    def test_order_book_alias_equivalence(self) -> None:
        """Verify that OrderBook alias creates a LimitOrderBook instance."""
        assert OrderBook is LimitOrderBook
        book = OrderBook()
        assert isinstance(book, LimitOrderBook)


class TestAddOrders:
    """Test suite for adding BUY and SELL orders to the book."""

    def test_add_single_buy_order(self) -> None:
        """Verify adding a single BUY order correctly updates bid side state."""
        book = LimitOrderBook()
        order = Order(
            order_id="b-1",
            side=OrderSide.BUY,
            price=100.0,
            quantity=50,
            timestamp=1000,
        )
        book.add_order(order)

        assert len(book) == 1
        assert book.is_empty is False
        assert book.bids_count == 1
        assert book.asks_count == 0
        assert book.total_volume == 50

        assert book.best_bid == 100.0
        assert book.get_best_bid() == 100.0
        assert book.best_bid_order == order
        assert book.get_best_bid_order() == order

        assert book.best_ask is None
        assert book.best_ask_order is None
        assert book.spread is None

        assert "b-1" in book
        assert book.get_order("b-1") == order

    def test_add_single_sell_order(self) -> None:
        """Verify adding a single SELL order correctly updates ask side state."""
        book = LimitOrderBook()
        order = Order(
            order_id="s-1",
            side=OrderSide.SELL,
            price=105.5,
            quantity=25,
            timestamp=1000,
        )
        book.add_order(order)

        assert len(book) == 1
        assert book.is_empty is False
        assert book.bids_count == 0
        assert book.asks_count == 1
        assert book.total_volume == 25

        assert book.best_ask == 105.5
        assert book.get_best_ask() == 105.5
        assert book.best_ask_order == order
        assert book.get_best_ask_order() == order

        assert book.best_bid is None
        assert book.best_bid_order is None
        assert book.spread is None

        assert "s-1" in book
        assert book.get_order("s-1") == order

    def test_add_both_buy_and_sell_orders(self) -> None:
        """Verify adding both BUY and SELL orders populates both sides and computes spread."""
        book = LimitOrderBook()
        buy_order = Order("b-1", OrderSide.BUY, 99.0, 100)
        sell_order = Order("s-1", OrderSide.SELL, 101.5, 200)

        book.add_order(buy_order)
        book.add_order(sell_order)

        assert len(book) == 2
        assert book.bids_count == 1
        assert book.asks_count == 1
        assert book.total_volume == 300

        assert book.best_bid == 99.0
        assert book.best_ask == 101.5
        assert book.spread == 2.5
        assert book.best_bid_order == buy_order
        assert book.best_ask_order == sell_order

    def test_add_orders_with_decimal_prices(self) -> None:
        """Verify adding orders with Decimal prices computes exact spread."""
        book = LimitOrderBook()
        buy_order = Order("b-dec", OrderSide.BUY, Decimal("150.25"), 10)
        sell_order = Order("s-dec", OrderSide.SELL, Decimal("150.75"), 20)

        book.add_order(buy_order)
        book.add_order(sell_order)

        assert book.best_bid == Decimal("150.25")
        assert book.best_ask == Decimal("150.75")
        assert book.spread == Decimal("0.50")


class TestBestBidAndAskSelection:
    """Test suite for best bid and best ask selection logic."""

    def test_best_bid_selects_highest_price(self) -> None:
        """Verify that the BUY side selects the highest price as best bid."""
        book = LimitOrderBook()
        orders = [
            Order("b-1", OrderSide.BUY, 100.0, 10),
            Order("b-2", OrderSide.BUY, 105.0, 20),
            Order("b-3", OrderSide.BUY, 95.0, 30),
            Order("b-4", OrderSide.BUY, 102.5, 15),
        ]
        for ord_obj in orders:
            book.add_order(ord_obj)

        assert book.best_bid == 105.0
        assert book.best_bid_order is not None
        assert book.best_bid_order.order_id == "b-2"
        assert book.best_bid_order.price == 105.0

    def test_best_ask_selects_lowest_price(self) -> None:
        """Verify that the SELL side selects the lowest price as best ask."""
        book = LimitOrderBook()
        orders = [
            Order("s-1", OrderSide.SELL, 110.0, 10),
            Order("s-2", OrderSide.SELL, 105.0, 20),
            Order("s-3", OrderSide.SELL, 120.0, 30),
            Order("s-4", OrderSide.SELL, 108.5, 15),
        ]
        for ord_obj in orders:
            book.add_order(ord_obj)

        assert book.best_ask == 105.0
        assert book.best_ask_order is not None
        assert book.best_ask_order.order_id == "s-2"
        assert book.best_ask_order.price == 105.0

    def test_best_bid_ask_updated_after_adding_more_aggressive_order(self) -> None:
        """Verify BBO updates immediately when a higher bid or lower ask arrives."""
        book = LimitOrderBook()
        book.add_order(Order("b-1", OrderSide.BUY, 100.0, 10))
        assert book.best_bid == 100.0

        # Higher bid arrives
        book.add_order(Order("b-2", OrderSide.BUY, 101.0, 10))
        assert book.best_bid == 101.0
        assert book.best_bid_order is not None
        assert book.best_bid_order.order_id == "b-2"

        # Lower bid arrives (should not change best bid)
        book.add_order(Order("b-3", OrderSide.BUY, 98.0, 10))
        assert book.best_bid == 101.0

        # Ask side
        book.add_order(Order("s-1", OrderSide.SELL, 105.0, 10))
        assert book.best_ask == 105.0

        # Lower ask arrives
        book.add_order(Order("s-2", OrderSide.SELL, 103.0, 10))
        assert book.best_ask == 103.0
        assert book.best_ask_order is not None
        assert book.best_ask_order.order_id == "s-2"

        # Higher ask arrives (should not change best ask)
        book.add_order(Order("s-3", OrderSide.SELL, 108.0, 10))
        assert book.best_ask == 103.0


class TestMultiplePriceLevelsAndPriority:
    """Test suite for price ladders and price priority sorting."""

    def test_buy_price_levels_sorted_descending(self) -> None:
        """Verify that bid price levels are ordered descending (highest to lowest)."""
        book = LimitOrderBook()
        prices = [95.0, 105.0, 100.0, 102.5, 90.0]
        for i, p in enumerate(prices):
            book.add_order(Order(f"b-{i}", OrderSide.BUY, p, 10))

        assert book.bids.price_levels == [105.0, 102.5, 100.0, 95.0, 90.0]

    def test_sell_price_levels_sorted_ascending(self) -> None:
        """Verify that ask price levels are ordered ascending (lowest to highest)."""
        book = LimitOrderBook()
        prices = [120.0, 105.0, 110.0, 102.5, 115.0]
        for i, p in enumerate(prices):
            book.add_order(Order(f"s-{i}", OrderSide.SELL, p, 10))

        assert book.asks.price_levels == [102.5, 105.0, 110.0, 115.0, 120.0]

    def test_price_priority_updates_as_levels_are_removed(self) -> None:
        """Verify that as the best level is removed, the next best level takes priority."""
        book = LimitOrderBook()
        book.add_order(Order("b-1", OrderSide.BUY, 105.0, 10))
        book.add_order(Order("b-2", OrderSide.BUY, 102.0, 10))
        book.add_order(Order("b-3", OrderSide.BUY, 100.0, 10))

        assert book.best_bid == 105.0

        book.remove_order("b-1")
        assert book.best_bid == 102.0

        book.remove_order("b-2")
        assert book.best_bid == 100.0

        book.remove_order("b-3")
        assert book.best_bid is None
        assert book.is_empty is True


class TestPriceTimePriorityFIFO:
    """Test suite ensuring strict FIFO ordering at identical price levels."""

    def test_fifo_ordering_at_same_price_buy(self) -> None:
        """Verify BUY orders at the same price retain exact arrival order."""
        book = LimitOrderBook()
        ord1 = Order("b-1", OrderSide.BUY, 100.0, 10, timestamp=1000)
        ord2 = Order("b-2", OrderSide.BUY, 100.0, 20, timestamp=2000)
        ord3 = Order("b-3", OrderSide.BUY, 100.0, 30, timestamp=3000)

        book.add_order(ord1)
        book.add_order(ord2)
        book.add_order(ord3)

        orders_at_100 = book.get_orders_at_price(100.0)
        assert orders_at_100 == [ord1, ord2, ord3]
        assert book.best_bid_order == ord1

    def test_fifo_ordering_at_same_price_sell(self) -> None:
        """Verify SELL orders at the same price retain exact arrival order."""
        book = LimitOrderBook()
        ord1 = Order("s-1", OrderSide.SELL, 105.0, 10, timestamp=1000)
        ord2 = Order("s-2", OrderSide.SELL, 105.0, 20, timestamp=2000)
        ord3 = Order("s-3", OrderSide.SELL, 105.0, 30, timestamp=3000)

        book.add_order(ord1)
        book.add_order(ord2)
        book.add_order(ord3)

        orders_at_105 = book.get_orders_at_price(105.0)
        assert orders_at_105 == [ord1, ord2, ord3]
        assert book.best_ask_order == ord1

    def test_fifo_advancement_after_head_removal(self) -> None:
        """Verify that removing the head order promotes the second arrival to head."""
        book = LimitOrderBook()
        ord1 = Order("b-1", OrderSide.BUY, 100.0, 10, timestamp=1000)
        ord2 = Order("b-2", OrderSide.BUY, 100.0, 20, timestamp=2000)
        ord3 = Order("b-3", OrderSide.BUY, 100.0, 30, timestamp=3000)

        book.add_order(ord1)
        book.add_order(ord2)
        book.add_order(ord3)

        book.remove_order("b-1")

        assert book.best_bid_order == ord2
        assert book.get_orders_at_price(100.0) == [ord2, ord3]

    def test_removing_middle_order_preserves_remaining_fifo(self) -> None:
        """Verify that cancelling an order in the middle of a queue preserves relative FIFO order."""
        book = LimitOrderBook()
        ord1 = Order("b-1", OrderSide.BUY, 100.0, 10, timestamp=1000)
        ord2 = Order("b-2", OrderSide.BUY, 100.0, 20, timestamp=2000)
        ord3 = Order("b-3", OrderSide.BUY, 100.0, 30, timestamp=3000)

        book.add_order(ord1)
        book.add_order(ord2)
        book.add_order(ord3)

        # Cancel middle order ord2
        removed = book.remove_order("b-2")
        assert removed == ord2

        assert book.best_bid_order == ord1
        assert book.get_orders_at_price(100.0) == [ord1, ord3]


class TestInspectingOrdersAtPriceLevel:
    """Test suite for inspecting price levels, order counts, and aggregated volumes."""

    def test_inspect_orders_returns_defensive_copy(self) -> None:
        """Verify that get_orders_at_price returns a copy that does not mutate book internals."""
        book = LimitOrderBook()
        ord1 = Order("b-1", OrderSide.BUY, 100.0, 10)
        book.add_order(ord1)

        result = book.get_orders_at_price(100.0)
        assert result == [ord1]

        # Mutate the returned list
        result.clear()

        # Internal book must remain intact
        assert book.get_orders_at_price(100.0) == [ord1]
        assert len(book) == 1

    def test_volume_and_count_at_price_level(self) -> None:
        """Verify accurate aggregate volume and order count tracking at a price level."""
        book = LimitOrderBook()
        book.add_order(Order("b-1", OrderSide.BUY, 100.0, 10))
        book.add_order(Order("b-2", OrderSide.BUY, 100.0, 25))
        book.add_order(Order("b-3", OrderSide.BUY, 100.0, 15))

        assert book.get_volume_at_price(100.0) == 50
        assert book.get_order_count_at_price(100.0) == 3

        book.remove_order("b-2")
        assert book.get_volume_at_price(100.0) == 25
        assert book.get_order_count_at_price(100.0) == 2

    def test_side_filter_for_orders_at_price(self) -> None:
        """Verify side-specific filtering when inspecting price levels."""
        book = LimitOrderBook()
        b_ord = Order("b-1", OrderSide.BUY, 100.0, 10)
        s_ord = Order("s-1", OrderSide.SELL, 100.0, 20)
        book.add_order(b_ord)
        book.add_order(s_ord)

        # Query with side=BUY
        assert book.get_orders_at_price(100.0, side=OrderSide.BUY) == [b_ord]
        assert book.get_orders_at_price(100.0, side="BUY") == [b_ord]
        assert book.get_volume_at_price(100.0, side=OrderSide.BUY) == 10
        assert book.get_order_count_at_price(100.0, side=OrderSide.BUY) == 1

        # Query with side=SELL
        assert book.get_orders_at_price(100.0, side=OrderSide.SELL) == [s_ord]
        assert book.get_orders_at_price(100.0, side="SELL") == [s_ord]
        assert book.get_volume_at_price(100.0, side=OrderSide.SELL) == 20
        assert book.get_order_count_at_price(100.0, side=OrderSide.SELL) == 1

        # Query with side=None (both)
        assert book.get_orders_at_price(100.0) == [b_ord, s_ord]
        assert book.get_volume_at_price(100.0) == 30
        assert book.get_order_count_at_price(100.0) == 2


class TestRemoveOrder:
    """Test suite for removing existing orders from the book."""

    def test_remove_existing_buy_order(self) -> None:
        """Verify removing a BUY order by ID updates the book and returns the order."""
        book = LimitOrderBook()
        order = Order("b-1", OrderSide.BUY, 100.0, 10)
        book.add_order(order)

        removed = book.remove_order("b-1")
        assert removed == order
        assert len(book) == 0
        assert book.is_empty is True
        assert "b-1" not in book
        assert book.get_order("b-1") is None
        assert book.best_bid is None

    def test_remove_existing_sell_order(self) -> None:
        """Verify removing a SELL order by ID updates the book and returns the order."""
        book = LimitOrderBook()
        order = Order("s-1", OrderSide.SELL, 105.0, 10)
        book.add_order(order)

        removed = book.remove_order("s-1")
        assert removed == order
        assert len(book) == 0
        assert book.is_empty is True
        assert "s-1" not in book
        assert book.get_order("s-1") is None
        assert book.best_ask is None

    def test_cancel_order_convenience_alias(self) -> None:
        """Verify cancel_order functions identically to remove_order."""
        book = LimitOrderBook()
        order = Order("b-1", OrderSide.BUY, 100.0, 10)
        book.add_order(order)

        removed = book.cancel_order("b-1")
        assert removed == order
        assert len(book) == 0

    def test_empty_price_level_is_pruned(self) -> None:
        """Verify that when the last order at a price level is removed, the price level is pruned."""
        book = LimitOrderBook()
        book.add_order(Order("b-1", OrderSide.BUY, 100.0, 10))
        book.add_order(Order("b-2", OrderSide.BUY, 95.0, 10))

        assert book.bids.price_levels == [100.0, 95.0]

        book.remove_order("b-1")

        assert book.bids.price_levels == [95.0]
        assert book.get_orders_at_price(100.0) == []
        assert book.get_volume_at_price(100.0) == 0
        assert book.best_bid == 95.0


class TestRemoveUnknownOrder:
    """Test suite for error handling when removing unknown or non-existent orders."""

    def test_remove_nonexistent_order_id_raises_order_not_found_error(self) -> None:
        """Verify attempting to remove an unknown order raises OrderNotFoundError (and KeyError)."""
        book = LimitOrderBook()

        with pytest.raises(OrderNotFoundError, match="Order with ID 'unknown-1' not found"):
            book.remove_order("unknown-1")

        # Must also be catchable as KeyError and OrderBookError
        with pytest.raises(KeyError):
            book.remove_order("unknown-2")

        with pytest.raises(OrderBookError):
            book.remove_order("unknown-3")

    def test_remove_already_removed_order_raises(self) -> None:
        """Verify attempting to remove an order twice raises OrderNotFoundError on the second call."""
        book = LimitOrderBook()
        order = Order("b-1", OrderSide.BUY, 100.0, 10)
        book.add_order(order)

        book.remove_order("b-1")

        with pytest.raises(OrderNotFoundError):
            book.remove_order("b-1")

    def test_book_state_preserved_after_failed_removal(self) -> None:
        """Verify that a failed removal attempt does not mutate existing book state."""
        book = LimitOrderBook()
        order = Order("b-1", OrderSide.BUY, 100.0, 10)
        book.add_order(order)

        with pytest.raises(OrderNotFoundError):
            book.remove_order("ghost-id")

        assert len(book) == 1
        assert "b-1" in book
        assert book.best_bid == 100.0
        assert book.total_volume == 10


class TestDuplicateOrderIdHandling:
    """Test suite for duplicate order ID rejection."""

    def test_duplicate_order_id_same_side_raises_duplicate_order_error(self) -> None:
        """Verify adding an order with an existing ID on the same side is rejected."""
        book = LimitOrderBook()
        order1 = Order("ord-dup", OrderSide.BUY, 100.0, 10)
        order2 = Order("ord-dup", OrderSide.BUY, 105.0, 20)

        book.add_order(order1)

        with pytest.raises(DuplicateOrderError, match="Order with ID 'ord-dup' already exists"):
            book.add_order(order2)

        # Must also be catchable as ValueError and OrderBookError
        with pytest.raises(ValueError):
            book.add_order(order2)

        with pytest.raises(OrderBookError):
            book.add_order(order2)

    def test_duplicate_order_id_across_different_sides_raises(self) -> None:
        """Verify adding an order with an existing ID on the opposite side is also rejected."""
        book = LimitOrderBook()
        buy_order = Order("ord-cross", OrderSide.BUY, 100.0, 10)
        sell_order = Order("ord-cross", OrderSide.SELL, 105.0, 10)

        book.add_order(buy_order)

        with pytest.raises(DuplicateOrderError, match="Order with ID 'ord-cross' already exists"):
            book.add_order(sell_order)

    def test_book_state_preserved_after_duplicate_rejection(self) -> None:
        """Verify that rejecting a duplicate order leaves the original order intact."""
        book = LimitOrderBook()
        original = Order("ord-1", OrderSide.BUY, 100.0, 10)
        duplicate = Order("ord-1", OrderSide.BUY, 105.0, 20)

        book.add_order(original)

        with pytest.raises(DuplicateOrderError):
            book.add_order(duplicate)

        assert len(book) == 1
        assert book.best_bid == 100.0
        assert book.total_volume == 10
        assert book.get_order("ord-1") == original


class TestInvalidInputs:
    """Test suite for strict input validation across all order book methods."""

    @pytest.mark.parametrize("invalid_order", [None, 123, "not_an_order", {"order_id": "1"}, [1, 2, 3]])
    def test_add_order_rejects_non_order_instance(self, invalid_order: object) -> None:
        """Verify add_order rejects objects that are not Order instances."""
        book = LimitOrderBook()
        with pytest.raises(TypeError, match="Expected Order instance"):
            book.add_order(invalid_order)  # type: ignore[arg-type]

    @pytest.mark.parametrize("invalid_id_type", [None, 123, True, False, 1.5, ("id",)])
    def test_remove_order_rejects_non_string_id(self, invalid_id_type: object) -> None:
        """Verify remove_order rejects non-string order IDs."""
        book = LimitOrderBook()
        with pytest.raises(TypeError, match="order_id must be a string"):
            book.remove_order(invalid_id_type)  # type: ignore[arg-type]

    @pytest.mark.parametrize("empty_id", ["", "   ", "\t\n\r  "])
    def test_remove_order_rejects_empty_or_whitespace_id(self, empty_id: str) -> None:
        """Verify remove_order rejects empty or whitespace-only strings."""
        book = LimitOrderBook()
        with pytest.raises(ValueError, match="order_id cannot be empty or whitespace"):
            book.remove_order(empty_id)

    @pytest.mark.parametrize("invalid_id_type", [None, 123, True, False])
    def test_get_order_rejects_non_string_id(self, invalid_id_type: object) -> None:
        """Verify get_order rejects non-string order IDs."""
        book = LimitOrderBook()
        with pytest.raises(TypeError, match="order_id must be a string"):
            book.get_order(invalid_id_type)  # type: ignore[arg-type]

    @pytest.mark.parametrize("empty_id", ["", "   ", "\t\n"])
    def test_get_order_rejects_empty_id(self, empty_id: str) -> None:
        """Verify get_order rejects empty or whitespace strings."""
        book = LimitOrderBook()
        with pytest.raises(ValueError, match="order_id cannot be empty or whitespace"):
            book.get_order(empty_id)

    @pytest.mark.parametrize("invalid_price_type", [None, "100.0", True, False, [100.0]])
    def test_get_orders_at_price_rejects_invalid_price_type(self, invalid_price_type: object) -> None:
        """Verify get_orders_at_price rejects non-numeric price types."""
        book = LimitOrderBook()
        with pytest.raises(TypeError, match="price must be a real number"):
            book.get_orders_at_price(invalid_price_type)  # type: ignore[arg-type]

    @pytest.mark.parametrize("invalid_price_val", [0, 0.0, -1, -50.5, Decimal("0"), Decimal("-10")])
    def test_get_orders_at_price_rejects_zero_or_negative_price(
        self, invalid_price_val: float | int | Decimal
    ) -> None:
        """Verify get_orders_at_price rejects zero or negative prices."""
        book = LimitOrderBook()
        with pytest.raises(ValueError, match="price must be strictly positive"):
            book.get_orders_at_price(invalid_price_val)

    @pytest.mark.parametrize(
        "non_finite_price",
        [float("nan"), float("inf"), float("-inf"), Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")],
    )
    def test_get_orders_at_price_rejects_non_finite_price(
        self, non_finite_price: float | Decimal
    ) -> None:
        """Verify get_orders_at_price rejects NaN and Infinite prices."""
        book = LimitOrderBook()
        with pytest.raises(ValueError, match="price must be finite"):
            book.get_orders_at_price(non_finite_price)

    @pytest.mark.parametrize("invalid_side", ["HOLD", "buy", "sell", "UNKNOWN", 123])
    def test_get_orders_at_price_rejects_invalid_side(self, invalid_side: object) -> None:
        """Verify get_orders_at_price rejects invalid side values."""
        book = LimitOrderBook()
        with pytest.raises((ValueError, TypeError)):
            book.get_orders_at_price(100.0, side=invalid_side)  # type: ignore[arg-type]


class TestSideIsolationAndInteractions:
    """Test suite ensuring BUY and SELL sides operate independently without side-effects."""

    def test_removal_from_buy_side_does_not_affect_sell_side(self) -> None:
        """Verify removing a BUY order leaves all SELL orders and depth completely unaffected."""
        book = LimitOrderBook()
        b1 = Order("b-1", OrderSide.BUY, 100.0, 10)
        b2 = Order("b-2", OrderSide.BUY, 99.0, 20)
        s1 = Order("s-1", OrderSide.SELL, 105.0, 15)
        s2 = Order("s-2", OrderSide.SELL, 106.0, 25)

        book.add_order(b1)
        book.add_order(b2)
        book.add_order(s1)
        book.add_order(s2)

        # Remove BUY order b1
        book.remove_order("b-1")

        # BUY side updated
        assert book.best_bid == 99.0
        assert book.bids_count == 1
        assert "b-1" not in book
        assert "b-2" in book

        # SELL side completely intact
        assert book.best_ask == 105.0
        assert book.asks_count == 2
        assert book.best_ask_order == s1
        assert "s-1" in book
        assert "s-2" in book
        assert book.get_orders_at_price(105.0) == [s1]
        assert book.get_orders_at_price(106.0) == [s2]

    def test_removal_from_sell_side_does_not_affect_buy_side(self) -> None:
        """Verify removing a SELL order leaves all BUY orders and depth completely unaffected."""
        book = LimitOrderBook()
        b1 = Order("b-1", OrderSide.BUY, 100.0, 10)
        b2 = Order("b-2", OrderSide.BUY, 99.0, 20)
        s1 = Order("s-1", OrderSide.SELL, 105.0, 15)
        s2 = Order("s-2", OrderSide.SELL, 106.0, 25)

        book.add_order(b1)
        book.add_order(b2)
        book.add_order(s1)
        book.add_order(s2)

        # Remove SELL order s1
        book.remove_order("s-1")

        # SELL side updated
        assert book.best_ask == 106.0
        assert book.asks_count == 1
        assert "s-1" not in book
        assert "s-2" in book

        # BUY side completely intact
        assert book.best_bid == 100.0
        assert book.bids_count == 2
        assert book.best_bid_order == b1
        assert "b-1" in book
        assert "b-2" in book

    def test_emptying_one_side_leaves_other_side_operational(self) -> None:
        """Verify emptying the BUY side completely does not disrupt the SELL side."""
        book = LimitOrderBook()
        b1 = Order("b-1", OrderSide.BUY, 100.0, 10)
        s1 = Order("s-1", OrderSide.SELL, 105.0, 15)

        book.add_order(b1)
        book.add_order(s1)

        book.remove_order("b-1")

        assert book.bids_count == 0
        assert book.best_bid is None
        assert book.best_bid_order is None
        assert book.bids.is_empty is True

        assert book.asks_count == 1
        assert book.best_ask == 105.0
        assert book.best_ask_order == s1
        assert book.asks.is_empty is False

    def test_clear_method_resets_both_sides(self) -> None:
        """Verify clear() cleanly resets all internal state on both sides."""
        book = LimitOrderBook()
        book.add_order(Order("b-1", OrderSide.BUY, 100.0, 10))
        book.add_order(Order("s-1", OrderSide.SELL, 105.0, 20))

        assert len(book) == 2
        book.clear()

        assert len(book) == 0
        assert book.is_empty is True
        assert book.best_bid is None
        assert book.best_ask is None
        assert book.bids_count == 0
        assert book.asks_count == 0
        assert "b-1" not in book
        assert "s-1" not in book


class TestMarketDepthAndLevel2:
    """Test suite for Level 2 aggregated market depth."""

    def test_depth_aggregation_and_ordering(self) -> None:
        """Verify multiple orders at same price aggregate volume and preserve priority ordering."""
        book = LimitOrderBook()
        # Bids: 100 (vol 30), 99 (vol 20), 98 (vol 50)
        book.add_order(Order("b-1", OrderSide.BUY, 100.0, 10))
        book.add_order(Order("b-2", OrderSide.BUY, 100.0, 20))
        book.add_order(Order("b-3", OrderSide.BUY, 99.0, 20))
        book.add_order(Order("b-4", OrderSide.BUY, 98.0, 50))

        # Asks: 101 (vol 15), 102 (vol 25)
        book.add_order(Order("s-1", OrderSide.SELL, 101.0, 5))
        book.add_order(Order("s-2", OrderSide.SELL, 101.0, 10))
        book.add_order(Order("s-3", OrderSide.SELL, 102.0, 25))

        bids_depth = book.get_bids_depth(depth=2)
        assert bids_depth == [(100.0, 30), (99.0, 20)]

        asks_depth = book.get_asks_depth(depth=2)
        assert asks_depth == [(101.0, 15), (102.0, 25)]

        l2_bids, l2_asks = book.get_l2_depth(depth=10)
        assert l2_bids == [(100.0, 30), (99.0, 20), (98.0, 50)]
        assert l2_asks == [(101.0, 15), (102.0, 25)]


class TestPriceLevelAndSideDirectUnits:
    """Test suite for PriceLevel and OrderBookSide direct methods and reprs."""

    def test_price_level_repr_and_properties(self) -> None:
        """Verify PriceLevel properties, len, contains, and repr."""
        pl = PriceLevel(100.0)
        ord1 = Order("b-1", OrderSide.BUY, 100.0, 10)
        pl.add_order(ord1)

        assert len(pl) == 1
        assert pl.order_count == 1
        assert pl.total_volume == 10
        assert pl.head_order == ord1
        assert "b-1" in pl
        assert "b-2" not in pl
        assert "PriceLevel" in repr(pl)

    def test_order_book_side_validation(self) -> None:
        """Verify OrderBookSide constructor rejects invalid side type."""
        with pytest.raises(TypeError, match="side must be an OrderSide instance"):
            OrderBookSide("BUY")  # type: ignore[arg-type]

    def test_order_book_side_mismatched_order_rejected(self) -> None:
        """Verify OrderBookSide.add_order rejects orders with the wrong side."""
        bids = OrderBookSide(OrderSide.BUY)
        sell_order = Order("s-1", OrderSide.SELL, 105.0, 10)

        with pytest.raises(ValueError, match="Cannot add order with side SELL to BUY side"):
            bids.add_order(sell_order)

    def test_limit_order_book_repr(self) -> None:
        """Verify LimitOrderBook repr is informative."""
        book = LimitOrderBook()
        book.add_order(Order("b-1", OrderSide.BUY, 100.0, 10))
        rep = repr(book)
        assert "LimitOrderBook" in rep
        assert "best_bid=100.0" in rep


class TestStressAndInvariants:
    """Stress tests and invariant checking across large numbers of orders and prices."""

    def test_hundred_orders_randomized_invariants(self) -> None:
        """Verify invariants hold during insertion and cancellation of 100 orders."""
        import random

        rng = random.Random(42)
        book = LimitOrderBook()
        active_bids: list[Order] = []
        active_asks: list[Order] = []

        prices = [90.0, 95.0, 99.0, 100.0, 101.0, 105.0, 110.0]

        for i in range(100):
            side = OrderSide.BUY if rng.random() < 0.5 else OrderSide.SELL
            price = rng.choice(prices)
            qty = rng.randint(1, 50)
            order = Order(order_id=f"stress-{i}", side=side, price=price, quantity=qty)
            book.add_order(order)
            if side == OrderSide.BUY:
                active_bids.append(order)
            else:
                active_asks.append(order)

            # Invariant check
            expected_total_vol = sum(o.quantity for o in active_bids) + sum(o.quantity for o in active_asks)
            assert book.total_volume == expected_total_vol
            assert len(book) == len(active_bids) + len(active_asks)

            if active_bids:
                max_bid_price = max(o.price for o in active_bids)
                assert book.best_bid == max_bid_price
            else:
                assert book.best_bid is None

            if active_asks:
                min_ask_price = min(o.price for o in active_asks)
                assert book.best_ask == min_ask_price
            else:
                assert book.best_ask is None

        # Randomly cancel half of them
        all_orders = active_bids + active_asks
        rng.shuffle(all_orders)
        orders_to_cancel = all_orders[:50]

        for o in orders_to_cancel:
            removed = book.remove_order(o.order_id)
            assert removed == o
            if o in active_bids:
                active_bids.remove(o)
            else:
                active_asks.remove(o)

            if active_bids:
                assert book.best_bid == max(b.price for b in active_bids)
            else:
                assert book.best_bid is None

            if active_asks:
                assert book.best_ask == min(a.price for a in active_asks)
            else:
                assert book.best_ask is None

        # Cancel remaining
        for o in all_orders[50:]:
            book.remove_order(o.order_id)

        assert len(book) == 0
        assert book.is_empty is True
        assert book.best_bid is None
        assert book.best_ask is None


class TestReduceOrderQuantity:
    """Test suite for partial fill quantity reduction in PriceLevel, OrderBookSide, and LimitOrderBook."""

    def test_reduce_buy_order_quantity(self) -> None:
        """Verify reducing a resting BUY order updates order quantity, level volume, and book volume."""
        book = LimitOrderBook()
        order = Order("b-1", OrderSide.BUY, 100.0, 50)
        book.add_order(order)

        assert book.total_volume == 50
        assert book.get_volume_at_price(100.0) == 50

        reduced = book.reduce_order_quantity("b-1", 20)
        assert reduced == order
        assert order.quantity == 30
        assert book.total_volume == 30
        assert book.get_volume_at_price(100.0) == 30
        assert book.get_order("b-1").quantity == 30
        assert len(book) == 1

    def test_reduce_sell_order_quantity(self) -> None:
        """Verify reducing a resting SELL order updates order quantity, level volume, and book volume."""
        book = LimitOrderBook()
        order = Order("s-1", OrderSide.SELL, 105.0, 40)
        book.add_order(order)

        reduced = book.reduce_order_quantity("s-1", 15)
        assert reduced == order
        assert order.quantity == 25
        assert book.total_volume == 25
        assert book.get_volume_at_price(105.0) == 25

    def test_reduce_order_preserves_fifo_position(self) -> None:
        """Verify partially reducing head order preserves its top priority in FIFO queue."""
        book = LimitOrderBook()
        ord1 = Order("b-1", OrderSide.BUY, 100.0, 50, timestamp=1000)
        ord2 = Order("b-2", OrderSide.BUY, 100.0, 30, timestamp=2000)
        book.add_order(ord1)
        book.add_order(ord2)

        book.reduce_order_quantity("b-1", 20)

        orders = book.get_orders_at_price(100.0)
        assert orders == [ord1, ord2]
        assert book.best_bid_order == ord1
        assert ord1.quantity == 30
        assert ord2.quantity == 30

    def test_reduce_unknown_order_raises(self) -> None:
        """Verify reducing non-existent order raises OrderNotFoundError."""
        book = LimitOrderBook()
        with pytest.raises(OrderNotFoundError):
            book.reduce_order_quantity("ghost", 10)

    @pytest.mark.parametrize("invalid_qty", [0, -1, -50])
    def test_reduce_non_positive_quantity_raises(self, invalid_qty: int) -> None:
        """Verify reducing with zero or negative quantity raises ValueError."""
        book = LimitOrderBook()
        book.add_order(Order("b-1", OrderSide.BUY, 100.0, 50))
        with pytest.raises(ValueError, match="strictly positive"):
            book.reduce_order_quantity("b-1", invalid_qty)

    @pytest.mark.parametrize("invalid_type", [None, 10.5, "10", True, False])
    def test_reduce_invalid_quantity_type_raises(self, invalid_type: object) -> None:
        """Verify non-integer or boolean quantity raises TypeError."""
        book = LimitOrderBook()
        book.add_order(Order("b-1", OrderSide.BUY, 100.0, 50))
        with pytest.raises(TypeError, match="filled_qty must be an integer"):
            book.reduce_order_quantity("b-1", invalid_type)  # type: ignore[arg-type]

    def test_reduce_equal_or_exceeding_quantity_raises(self) -> None:
        """Verify reducing by full quantity or more raises ValueError (remove_order should be used)."""
        book = LimitOrderBook()
        book.add_order(Order("b-1", OrderSide.BUY, 100.0, 50))

        with pytest.raises(ValueError, match="strictly less than order quantity"):
            book.reduce_order_quantity("b-1", 50)

        with pytest.raises(ValueError, match="strictly less than order quantity"):
            book.reduce_order_quantity("b-1", 51)


