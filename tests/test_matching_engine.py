"""Comprehensive automated test suite for the ChronosMatch Matching Engine."""

from decimal import Decimal
import time

import pytest

from chronosmatch.matching_engine import MatchingEngine, Trade, TradeEvent
from chronosmatch.models import Order, OrderSide, Side
from chronosmatch.order_book import DuplicateOrderError, LimitOrderBook, OrderNotFoundError


class TestNoMatchAndRestingOrders:
    """Test suite for scenarios where no compatible orders exist and orders rest in the book."""

    def test_buy_order_rests_in_empty_book(self) -> None:
        """Verify an incoming BUY order rests in an empty book without generating trades."""
        engine = MatchingEngine()
        order = Order("b-1", OrderSide.BUY, 100.0, 50)

        trades = engine.process_order(order)

        assert trades == []
        assert engine.total_trades == 0
        assert engine.best_bid == 100.0
        assert engine.best_ask is None
        assert "b-1" in engine.order_book
        assert engine.order_book.get_order("b-1") == order
        assert order.quantity == 50

    def test_sell_order_rests_in_empty_book(self) -> None:
        """Verify an incoming SELL order rests in an empty book without generating trades."""
        engine = MatchingEngine()
        order = Order("s-1", OrderSide.SELL, 105.5, 30)

        trades = engine.process_order(order)

        assert trades == []
        assert engine.total_trades == 0
        assert engine.best_ask == 105.5
        assert engine.best_bid is None
        assert "s-1" in engine.order_book
        assert engine.order_book.get_order("s-1") == order
        assert order.quantity == 30

    def test_no_trade_when_prices_do_not_cross_sell_higher_than_buy(self) -> None:
        """Verify that when buy price < sell price, orders do not match and both rest in book."""
        engine = MatchingEngine()
        # Seed book with resting BUY at 99.0
        engine.process_order(Order("b-1", OrderSide.BUY, 99.0, 10))

        # Incoming SELL at 101.0 (sell price > buy price: does not cross)
        sell_order = Order("s-1", OrderSide.SELL, 101.0, 15)
        trades = engine.process_order(sell_order)

        assert trades == []
        assert engine.total_trades == 0
        assert engine.best_bid == 99.0
        assert engine.best_ask == 101.0
        assert engine.spread == 2.0
        assert len(engine.order_book) == 2

    def test_no_trade_when_buy_price_lower_than_resting_ask(self) -> None:
        """Verify incoming BUY with price lower than resting ask rests without trade."""
        engine = MatchingEngine()
        # Seed book with resting SELL at 105.0
        engine.process_order(Order("s-1", OrderSide.SELL, 105.0, 20))

        # Incoming BUY at 104.0 (buy price < sell price: does not cross)
        buy_order = Order("b-1", OrderSide.BUY, 104.0, 10)
        trades = engine.process_order(buy_order)

        assert trades == []
        assert engine.total_trades == 0
        assert engine.best_bid == 104.0
        assert engine.best_ask == 105.0
        assert engine.spread == 1.0


class TestBasicSingleMatches:
    """Test suite for basic BUY matching SELL and SELL matching BUY at compatible prices."""

    def test_buy_matches_sell_at_exact_price(self) -> None:
        """Verify an incoming BUY order matches a resting SELL order at the exact same price."""
        engine = MatchingEngine()
        maker = Order("s-1", OrderSide.SELL, 100.0, 25)
        engine.process_order(maker)

        taker = Order("b-1", OrderSide.BUY, 100.0, 25)
        trades = engine.process_order(taker)

        assert len(trades) == 1
        trade = trades[0]
        assert trade.resting_order_id == "s-1"
        assert trade.incoming_order_id == "b-1"
        assert trade.price == 100.0
        assert trade.quantity == 25
        assert trade.trade_id == 1

        # Both fully filled; book is empty
        assert len(engine.order_book) == 0
        assert engine.best_bid is None
        assert engine.best_ask is None

    def test_sell_matches_buy_at_exact_price(self) -> None:
        """Verify an incoming SELL order matches a resting BUY order at the exact same price."""
        engine = MatchingEngine()
        maker = Order("b-1", OrderSide.BUY, 100.0, 40)
        engine.process_order(maker)

        taker = Order("s-1", OrderSide.SELL, 100.0, 40)
        trades = engine.process_order(taker)

        assert len(trades) == 1
        trade = trades[0]
        assert trade.resting_order_id == "b-1"
        assert trade.incoming_order_id == "s-1"
        assert trade.price == 100.0
        assert trade.quantity == 40

        # Both fully filled; book is empty
        assert len(engine.order_book) == 0
        assert engine.best_bid is None
        assert engine.best_ask is None


class TestExecutionPriceRule:
    """Test suite verifying the execution price rule: execute at the resting maker order's limit price."""

    def test_execution_price_equals_resting_sell_limit_price(self) -> None:
        """Verify incoming aggressive BUY with price 105 matches resting SELL at 100 executed at 100."""
        engine = MatchingEngine()
        maker_sell = Order("s-1", OrderSide.SELL, 100.0, 10)
        engine.process_order(maker_sell)

        # Incoming BUY offers 105.0 (taker willing to pay up to 105.0)
        taker_buy = Order("b-1", OrderSide.BUY, 105.0, 10)
        trades = engine.process_order(taker_buy)

        assert len(trades) == 1
        trade = trades[0]
        # Price improvement: executed at the resting order's limit price (100.0), NOT 105.0
        assert trade.price == 100.0
        assert trade.resting_order_id == "s-1"
        assert trade.incoming_order_id == "b-1"

    def test_execution_price_equals_resting_buy_limit_price(self) -> None:
        """Verify incoming aggressive SELL with price 95 matches resting BUY at 100 executed at 100."""
        engine = MatchingEngine()
        maker_buy = Order("b-1", OrderSide.BUY, 100.0, 10)
        engine.process_order(maker_buy)

        # Incoming SELL offers 95.0 (taker willing to sell down to 95.0)
        taker_sell = Order("s-1", OrderSide.SELL, 95.0, 10)
        trades = engine.process_order(taker_sell)

        assert len(trades) == 1
        trade = trades[0]
        # Price improvement: executed at the resting order's limit price (100.0), NOT 95.0
        assert trade.price == 100.0
        assert trade.resting_order_id == "b-1"
        assert trade.incoming_order_id == "s-1"

    def test_execution_price_with_decimal_prices(self) -> None:
        """Verify execution price rule holds with exact Decimal prices."""
        engine = MatchingEngine()
        maker_buy = Order("b-dec", OrderSide.BUY, Decimal("150.25"), 10)
        engine.process_order(maker_buy)

        taker_sell = Order("s-dec", OrderSide.SELL, Decimal("149.50"), 10)
        trades = engine.process_order(taker_sell)

        assert len(trades) == 1
        assert trades[0].price == Decimal("150.25")


class TestFullAndPartialFills:
    """Test suite covering full fills, partial fills, and correct remaining quantities."""

    def test_partial_fill_where_resting_order_is_larger(self) -> None:
        """Verify partial fill when resting maker order has larger quantity than incoming taker order."""
        engine = MatchingEngine()
        maker = Order("s-1", OrderSide.SELL, 100.0, 100)
        engine.process_order(maker)

        taker = Order("b-1", OrderSide.BUY, 100.0, 30)
        trades = engine.process_order(taker)

        assert len(trades) == 1
        assert trades[0].quantity == 30
        assert trades[0].price == 100.0

        # Maker remains in book with reduced quantity (100 - 30 = 70)
        assert len(engine.order_book) == 1
        resting_order = engine.order_book.get_order("s-1")
        assert resting_order is not None
        assert resting_order.quantity == 70
        assert engine.order_book.get_volume_at_price(100.0) == 70
        assert engine.order_book.total_volume == 70
        assert engine.best_ask == 100.0

        # Taker was fully filled; not present in book
        assert "b-1" not in engine.order_book
        assert engine.best_bid is None

    def test_partial_fill_where_incoming_order_is_larger_and_rests(self) -> None:
        """Verify partial fill where incoming order is larger; fully fills maker and remainder rests."""
        engine = MatchingEngine()
        maker = Order("s-1", OrderSide.SELL, 100.0, 30)
        engine.process_order(maker)

        taker = Order("b-1", OrderSide.BUY, 100.0, 80)
        trades = engine.process_order(taker)

        assert len(trades) == 1
        assert trades[0].quantity == 30
        assert trades[0].price == 100.0

        # Maker was fully filled and removed
        assert "s-1" not in engine.order_book
        assert engine.best_ask is None

        # Taker unfilled remainder (80 - 30 = 50) rests on BUY side
        assert len(engine.order_book) == 1
        resting_taker = engine.order_book.get_order("b-1")
        assert resting_taker is not None
        assert resting_taker.quantity == 50
        assert engine.order_book.get_volume_at_price(100.0) == 50
        assert engine.order_book.total_volume == 50
        assert engine.best_bid == 100.0


class TestMultiOrderSweep:
    """Test suite where one incoming order matches multiple resting orders."""

    def test_incoming_buy_sweeps_multiple_resting_asks_across_price_levels(self) -> None:
        """Verify an aggressive BUY sweeps multiple ask price levels in ascending order."""
        engine = MatchingEngine()
        # Seed asks:
        # s-1: 100.0, qty 10
        # s-2: 101.0, qty 20
        # s-3: 102.0, qty 30
        # s-4: 105.0, qty 50
        engine.process_order(Order("s-1", OrderSide.SELL, 100.0, 10))
        engine.process_order(Order("s-2", OrderSide.SELL, 101.0, 20))
        engine.process_order(Order("s-3", OrderSide.SELL, 102.0, 30))
        engine.process_order(Order("s-4", OrderSide.SELL, 105.0, 50))

        # Incoming BUY for 45 at limit 102.0
        # Should sweep s-1 (10 @ 100), s-2 (20 @ 101), s-3 (15 @ 102)
        taker = Order("b-sweep", OrderSide.BUY, 102.0, 45)
        trades = engine.process_order(taker)

        assert len(trades) == 3

        # Trade 1
        assert trades[0].resting_order_id == "s-1"
        assert trades[0].price == 100.0
        assert trades[0].quantity == 10

        # Trade 2
        assert trades[1].resting_order_id == "s-2"
        assert trades[1].price == 101.0
        assert trades[1].quantity == 20

        # Trade 3
        assert trades[2].resting_order_id == "s-3"
        assert trades[2].price == 102.0
        assert trades[2].quantity == 15

        # s-1 and s-2 fully filled; s-3 partially filled with 15 remaining; s-4 untouched with 50
        assert "s-1" not in engine.order_book
        assert "s-2" not in engine.order_book
        assert "b-sweep" not in engine.order_book  # fully consumed

        s3 = engine.order_book.get_order("s-3")
        assert s3 is not None
        assert s3.quantity == 15
        assert engine.best_ask == 102.0

        s4 = engine.order_book.get_order("s-4")
        assert s4 is not None
        assert s4.quantity == 50

    def test_incoming_sell_sweeps_multiple_resting_bids_across_price_levels(self) -> None:
        """Verify an aggressive SELL sweeps multiple bid price levels in descending order."""
        engine = MatchingEngine()
        # Seed bids:
        # b-1: 105.0, qty 15
        # b-2: 104.0, qty 25
        # b-3: 103.0, qty 50
        engine.process_order(Order("b-1", OrderSide.BUY, 105.0, 15))
        engine.process_order(Order("b-2", OrderSide.BUY, 104.0, 25))
        engine.process_order(Order("b-3", OrderSide.BUY, 103.0, 50))

        # Incoming SELL for 55 at limit 103.0
        # Should sweep b-1 (15 @ 105), b-2 (25 @ 104), b-3 (15 @ 103)
        taker = Order("s-sweep", OrderSide.SELL, 103.0, 55)
        trades = engine.process_order(taker)

        assert len(trades) == 3

        # Trade 1
        assert trades[0].resting_order_id == "b-1"
        assert trades[0].price == 105.0
        assert trades[0].quantity == 15

        # Trade 2
        assert trades[1].resting_order_id == "b-2"
        assert trades[1].price == 104.0
        assert trades[1].quantity == 25

        # Trade 3
        assert trades[2].resting_order_id == "b-3"
        assert trades[2].price == 103.0
        assert trades[2].quantity == 15

        # b-1 and b-2 removed; b-3 has 35 remaining
        assert "b-1" not in engine.order_book
        assert "b-2" not in engine.order_book
        b3 = engine.order_book.get_order("b-3")
        assert b3 is not None
        assert b3.quantity == 35
        assert engine.best_bid == 103.0

    def test_incoming_order_sweeps_depth_and_unfilled_balance_rests(self) -> None:
        """Verify that when an incoming order exhausts all compatible depth, remainder rests."""
        engine = MatchingEngine()
        engine.process_order(Order("s-1", OrderSide.SELL, 100.0, 20))
        engine.process_order(Order("s-2", OrderSide.SELL, 101.0, 30))

        # Incoming BUY for 80 at 101.0
        # Sweeps s-1 (20) + s-2 (30) = 50 filled; remaining 30 rests on BUY side at 101.0
        taker = Order("b-giant", OrderSide.BUY, 101.0, 80)
        trades = engine.process_order(taker)

        assert len(trades) == 2
        assert trades[0].quantity == 20
        assert trades[1].quantity == 30

        assert "s-1" not in engine.order_book
        assert "s-2" not in engine.order_book
        assert engine.best_ask is None

        # Taker rests with 30
        resting = engine.order_book.get_order("b-giant")
        assert resting is not None
        assert resting.quantity == 30
        assert engine.best_bid == 101.0


class TestPriceTimePriorityMatching:
    """Test suite ensuring strict Price-Time Priority (FIFO) during trade execution."""

    def test_price_priority_precedes_time_priority(self) -> None:
        """Verify that higher price on BUY side is matched before earlier orders at lower prices."""
        engine = MatchingEngine()
        # b-1: price 100, ts 1000
        # b-2: price 102, ts 2000 (arrived later, but higher price)
        engine.process_order(Order("b-1", OrderSide.BUY, 100.0, 20, timestamp=1000))
        engine.process_order(Order("b-2", OrderSide.BUY, 102.0, 20, timestamp=2000))

        # Incoming SELL at 100.0 for 15
        taker = Order("s-1", OrderSide.SELL, 100.0, 15)
        trades = engine.process_order(taker)

        assert len(trades) == 1
        # Must match b-2 first because 102.0 > 100.0
        assert trades[0].resting_order_id == "b-2"
        assert trades[0].price == 102.0
        assert trades[0].quantity == 15

        # b-2 remains with 5; b-1 remains untouched with 20
        assert engine.order_book.get_order("b-2").quantity == 5
        assert engine.order_book.get_order("b-1").quantity == 20

    def test_fifo_time_priority_at_same_price(self) -> None:
        """Verify that at identical price level, orders are filled in exact arrival order (FIFO)."""
        engine = MatchingEngine()
        # Three resting SELL orders at 100.0
        ord1 = Order("s-1", OrderSide.SELL, 100.0, 10, timestamp=1000)
        ord2 = Order("s-2", OrderSide.SELL, 100.0, 10, timestamp=2000)
        ord3 = Order("s-3", OrderSide.SELL, 100.0, 10, timestamp=3000)

        engine.process_order(ord1)
        engine.process_order(ord2)
        engine.process_order(ord3)

        # Incoming BUY for 15
        taker = Order("b-1", OrderSide.BUY, 100.0, 15)
        trades = engine.process_order(taker)

        assert len(trades) == 2

        # First fills s-1 completely (10)
        assert trades[0].resting_order_id == "s-1"
        assert trades[0].quantity == 10

        # Next fills s-2 partially (5)
        assert trades[1].resting_order_id == "s-2"
        assert trades[1].quantity == 5

        # s-1 removed
        assert "s-1" not in engine.order_book

        # s-2 has 5 remaining; s-3 untouched with 10
        assert engine.order_book.get_order("s-2").quantity == 5
        assert engine.order_book.get_order("s-3").quantity == 10

        # FIFO order at price 100.0 is preserved: s-2 is head, s-3 is second
        orders_at_100 = engine.order_book.get_orders_at_price(100.0)
        assert [o.order_id for o in orders_at_100] == ["s-2", "s-3"]


class TestInvalidOrderHandling:
    """Test suite for strict rejection of invalid orders and preserving engine state."""

    @pytest.mark.parametrize("invalid_order", [None, "not_an_order", 123, {"order_id": "1"}, [1, 2]])
    def test_reject_non_order_instance(self, invalid_order: object) -> None:
        """Verify process_order rejects objects that are not Order instances."""
        engine = MatchingEngine()
        with pytest.raises(TypeError, match="Expected Order instance"):
            engine.process_order(invalid_order)  # type: ignore[arg-type]

    def test_reject_duplicate_order_id(self) -> None:
        """Verify process_order rejects orders whose ID is already resting in the book."""
        engine = MatchingEngine()
        order1 = Order("ord-dup", OrderSide.BUY, 100.0, 10)
        engine.process_order(order1)

        order2 = Order("ord-dup", OrderSide.BUY, 100.0, 20)
        with pytest.raises(DuplicateOrderError, match="already exists"):
            engine.process_order(order2)

        # Engine state preserved
        assert len(engine.order_book) == 1
        assert engine.order_book.get_order("ord-dup").quantity == 10

    def test_reject_corrupted_order_zero_quantity(self) -> None:
        """Verify an order instance with manually corrupted non-positive quantity is rejected."""
        engine = MatchingEngine()
        order = Order("b-bad", OrderSide.BUY, 100.0, 10)
        # Simulate tampering
        object.__setattr__(order, "quantity", 0)

        with pytest.raises(ValueError, match="strictly positive"):
            engine.process_order(order)

    def test_engine_state_preserved_on_rejected_order(self) -> None:
        """Verify failed order insertion leaves existing book and trades unchanged."""
        engine = MatchingEngine()
        engine.process_order(Order("b-1", OrderSide.BUY, 100.0, 10))

        with pytest.raises(TypeError):
            engine.process_order("invalid")  # type: ignore[arg-type]

        assert len(engine.order_book) == 1
        assert engine.total_trades == 0
        assert engine.best_bid == 100.0


class TestTradeRepresentation:
    """Test suite for Trade dataclass properties, aliases, validation, and repr."""

    def test_trade_creation_and_aliases(self) -> None:
        """Verify Trade attributes and financial domain aliases."""
        t_now = time.time_ns()
        trade = Trade(
            trade_id=1,
            resting_order_id="m-1",
            incoming_order_id="t-1",
            price=100.25,
            quantity=50,
            timestamp=t_now,
        )

        assert trade.trade_id == 1
        assert trade.resting_order_id == "m-1"
        assert trade.maker_order_id == "m-1"
        assert trade.incoming_order_id == "t-1"
        assert trade.taker_order_id == "t-1"
        assert trade.price == 100.25
        assert trade.execution_price == 100.25
        assert trade.quantity == 50
        assert trade.executed_quantity == 50
        assert trade.execution_qty == 50
        assert trade.timestamp == t_now
        assert trade.timestamp_ns == t_now

    def test_trade_event_alias_equivalence(self) -> None:
        """Verify TradeEvent alias is identical to Trade."""
        assert TradeEvent is Trade
        trade = TradeEvent(1, "m-1", "t-1", 100.0, 10)
        assert isinstance(trade, Trade)

    @pytest.mark.parametrize("invalid_id", [0, -1, "", "   ", None, True, False])
    def test_trade_rejects_invalid_trade_id(self, invalid_id: object) -> None:
        """Verify Trade validation rejects non-positive or empty trade_id."""
        with pytest.raises((ValueError, TypeError)):
            Trade(invalid_id, "m-1", "t-1", 100.0, 10)  # type: ignore[arg-type]

    @pytest.mark.parametrize("invalid_qty", [0, -1, 10.5, "10", True, False, None])
    def test_trade_rejects_invalid_quantity(self, invalid_qty: object) -> None:
        """Verify Trade validation rejects non-integer or non-positive quantity."""
        with pytest.raises((ValueError, TypeError)):
            Trade(1, "m-1", "t-1", 100.0, invalid_qty)  # type: ignore[arg-type]

    @pytest.mark.parametrize("invalid_price", [0, -10.0, float("nan"), float("inf"), None, "100"])
    def test_trade_rejects_invalid_price(self, invalid_price: object) -> None:
        """Verify Trade validation rejects invalid price values and types."""
        with pytest.raises((ValueError, TypeError)):
            Trade(1, "m-1", "t-1", invalid_price, 10)  # type: ignore[arg-type]


class TestMatchingEngineStateAndTelemetry:
    """Test suite for engine lifecycle, history tracking, properties, and reset."""

    def test_engine_tracks_trades_and_volume(self) -> None:
        """Verify engine accumulates trade history and volume correctly."""
        engine = MatchingEngine()
        engine.process_order(Order("s-1", OrderSide.SELL, 100.0, 10))
        engine.process_order(Order("s-2", OrderSide.SELL, 101.0, 20))

        engine.process_order(Order("b-1", OrderSide.BUY, 101.0, 25))

        assert engine.total_trades == 2
        assert engine.total_volume_traded == 25
        assert len(engine.trades) == 2

    def test_engine_clear_resets_all_state(self) -> None:
        """Verify clear() resets book and trade history."""
        engine = MatchingEngine()
        engine.process_order(Order("s-1", OrderSide.SELL, 100.0, 10))
        engine.process_order(Order("b-1", OrderSide.BUY, 100.0, 10))

        assert engine.total_trades == 1
        engine.clear()

        assert engine.total_trades == 0
        assert engine.total_volume_traded == 0
        assert len(engine.trades) == 0
        assert len(engine.order_book) == 0

    def test_engine_repr_and_aliases(self) -> None:
        """Verify engine string representation and method aliases."""
        engine = MatchingEngine()
        assert "MatchingEngine" in repr(engine)
        assert engine.match_order == engine.process_order
        assert engine.match == engine.process_order
