# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""Cython-accelerated Limit Order Book and Matching Engine for ChronosMatch.

Optimized C-level implementation of Price-Time Priority order matching
and book management while strictly maintaining 100% behavioral equivalence
with the Python reference implementation.
"""

from collections import OrderedDict
import bisect
from decimal import Decimal
import math
import time

from chronosmatch.models import Order, OrderSide
from chronosmatch.matching_engine import Trade
from chronosmatch.order_book import (
    DuplicateOrderError,
    OrderNotFoundError,
    validate_order_id,
    validate_price,
)


cdef class CythonPriceLevel:
    """Represents a discrete price level containing resting orders organized by FIFO time priority."""

    cdef public object price
    cdef public object _orders
    cdef public long long _total_volume

    def __init__(self, price):
        validate_price(price)
        self.price = price
        self._orders = OrderedDict()
        self._total_volume = 0

    cpdef void add_order(self, object order):
        """Append an order to the tail of this price level (FIFO)."""
        self._orders[order.order_id] = order
        self._total_volume += order.quantity

    cpdef object remove_order(self, str order_id):
        """Remove and return an order from this price level by order_id."""
        cdef object order = self._orders.pop(order_id)
        self._total_volume -= order.quantity
        return order

    cpdef object reduce_order_quantity(self, str order_id, long long filled_qty):
        """Reduce the resting quantity of an order by filled_qty (partial fill)."""
        if order_id not in self._orders:
            raise KeyError(f"Order '{order_id}' not found at price level {self.price}")
        if filled_qty <= 0:
            raise ValueError(f"filled_qty must be strictly positive (> 0), got {filled_qty}")

        cdef object order = self._orders[order_id]
        if filled_qty >= order.quantity:
            raise ValueError(
                f"filled_qty {filled_qty} must be strictly less than order quantity {order.quantity}. "
                f"Use remove_order for full fills."
            )

        order.quantity -= filled_qty
        self._total_volume -= filled_qty
        return order

    cpdef object get_order(self, str order_id):
        """Retrieve an order by ID at this price level without removing it."""
        return self._orders.get(order_id)

    @property
    def orders(self):
        """Return defensive shallow copy of orders at this level in FIFO order."""
        return list(self._orders.values())

    @property
    def head_order(self):
        """Return the highest time-priority order at this price level (head of FIFO queue)."""
        if not self._orders:
            return None
        return next(iter(self._orders.values()))

    @property
    def total_volume(self):
        """Return the aggregate quantity of all resting orders at this price level."""
        return self._total_volume

    @property
    def is_empty(self):
        return len(self._orders) == 0

    def __len__(self):
        return len(self._orders)

    def __contains__(self, str order_id):
        return order_id in self._orders

    def __repr__(self):
        return f"CythonPriceLevel(price={self.price}, orders={len(self._orders)}, volume={self._total_volume})"


cdef class CythonOrderBookSide:
    """Represents one side (bids or asks) of the CythonOrderBook."""

    cdef public object side
    cdef public object _book

    def __init__(self, side, book):
        self.side = side
        self._book = book

    @property
    def is_buy(self):
        return self.side == OrderSide.BUY

    @property
    def price_levels(self):
        return list(self._book._bids if self.is_buy else self._book._asks)

    @property
    def best_price(self):
        return self._book.best_bid if self.is_buy else self._book.best_ask

    @property
    def best_order(self):
        return self._book.best_bid_order if self.is_buy else self._book.best_ask_order

    @property
    def total_volume(self):
        return self._book.bid_volume if self.is_buy else self._book.ask_volume

    @property
    def is_empty(self):
        cdef list prices = self._book._bids if self.is_buy else self._book._asks
        return len(prices) == 0

    def depth(self, int n=10):
        cdef list prices = (self._book._bids if self.is_buy else self._book._asks)[:n]
        cdef dict mapping = self._book._bids_map if self.is_buy else self._book._asks_map
        return [(p, mapping[p].total_volume) for p in prices]

    def __len__(self):
        cdef dict mapping = self._book._bids_map if self.is_buy else self._book._asks_map
        cdef int total = 0
        cdef CythonPriceLevel lvl
        for lvl in mapping.values():
            total += len(lvl)
        return total


cdef class CythonOrderBook:
    """Cython-accelerated Limit Order Book managing Bid and Ask depth with Price-Time Priority."""

    cdef public list _bids
    cdef public list _asks
    cdef public dict _bids_map
    cdef public dict _asks_map
    cdef public dict _order_id_to_price
    cdef public dict _order_id_to_side
    cdef public CythonOrderBookSide bids
    cdef public CythonOrderBookSide asks

    def __init__(self):
        self._bids = []
        self._asks = []
        self._bids_map = {}
        self._asks_map = {}
        self._order_id_to_price = {}
        self._order_id_to_side = {}
        self.bids = CythonOrderBookSide(OrderSide.BUY, self)
        self.asks = CythonOrderBookSide(OrderSide.SELL, self)

    cdef void _insert_bid_price(self, object price):
        """Insert price into descending _bids list via binary search."""
        cdef int low = 0
        cdef int high = len(self._bids)
        cdef int mid
        while low < high:
            mid = (low + high) // 2
            if self._bids[mid] < price:
                high = mid
            else:
                low = mid + 1
        self._bids.insert(low, price)

    cdef void _insert_ask_price(self, object price):
        """Insert price into ascending _asks list via bisect."""
        bisect.insort(self._asks, price)

    cpdef void add_order(self, object order):
        """Add an order into the order book."""
        if not isinstance(order, Order):
            raise TypeError(f"Expected Order instance, got {type(order).__name__}")
        if order.order_id in self._order_id_to_price:
            raise DuplicateOrderError(
                f"Order with ID '{order.order_id}' already exists in the order book."
            )

        cdef object side = order.side
        cdef object price = order.price
        cdef CythonPriceLevel level

        if side == OrderSide.BUY:
            if price not in self._bids_map:
                level = CythonPriceLevel(price)
                self._bids_map[price] = level
                self._insert_bid_price(price)
            else:
                level = self._bids_map[price]
            level.add_order(order)
            self._order_id_to_price[order.order_id] = price
            self._order_id_to_side[order.order_id] = OrderSide.BUY
        elif side == OrderSide.SELL:
            if price not in self._asks_map:
                level = CythonPriceLevel(price)
                self._asks_map[price] = level
                self._insert_ask_price(price)
            else:
                level = self._asks_map[price]
            level.add_order(order)
            self._order_id_to_price[order.order_id] = price
            self._order_id_to_side[order.order_id] = OrderSide.SELL
        else:
            raise ValueError(f"Invalid side '{side}'. Must be BUY or SELL.")

    cpdef object remove_order(self, str order_id):
        """Remove and return an order from the book by order_id."""
        if order_id not in self._order_id_to_price:
            raise OrderNotFoundError(f"Order '{order_id}' not found in the order book.")

        cdef object price = self._order_id_to_price.pop(order_id)
        cdef object side = self._order_id_to_side.pop(order_id)
        cdef CythonPriceLevel level
        cdef object removed_order

        if side == OrderSide.BUY:
            level = self._bids_map[price]
            removed_order = level.remove_order(order_id)
            if len(level) == 0:
                del self._bids_map[price]
                self._bids.remove(price)
        else:
            level = self._asks_map[price]
            removed_order = level.remove_order(order_id)
            if len(level) == 0:
                del self._asks_map[price]
                self._asks.remove(price)

        return removed_order

    cpdef object reduce_order_quantity(self, str order_id, long long filled_qty):
        """Reduce resting order quantity by filled_qty."""
        if order_id not in self._order_id_to_price:
            raise OrderNotFoundError(f"Order '{order_id}' not found in the order book.")

        cdef object price = self._order_id_to_price[order_id]
        cdef object side = self._order_id_to_side[order_id]
        cdef CythonPriceLevel level

        if side == OrderSide.BUY:
            level = self._bids_map[price]
        else:
            level = self._asks_map[price]

        return level.reduce_order_quantity(order_id, filled_qty)

    cpdef object get_order(self, str order_id):
        """Retrieve an order by ID without removing it."""
        if order_id not in self._order_id_to_price:
            return None
        cdef object price = self._order_id_to_price[order_id]
        cdef object side = self._order_id_to_side[order_id]
        if side == OrderSide.BUY:
            return self._bids_map[price].get_order(order_id)
        return self._asks_map[price].get_order(order_id)

    @property
    def best_bid(self):
        """Highest resting buy price, or None if no bids."""
        return self._bids[0] if self._bids else None

    @property
    def best_ask(self):
        """Lowest resting sell price, or None if no asks."""
        return self._asks[0] if self._asks else None

    @property
    def best_bid_order(self):
        """Earliest arrival order at best bid price, or None."""
        if not self._bids:
            return None
        cdef CythonPriceLevel level = self._bids_map[self._bids[0]]
        return level.head_order

    @property
    def best_ask_order(self):
        """Earliest arrival order at best ask price, or None."""
        if not self._asks:
            return None
        cdef CythonPriceLevel level = self._asks_map[self._asks[0]]
        return level.head_order

    @property
    def spread(self):
        """Difference between best ask and best bid, or None if book is one-sided or empty."""
        if not self._bids or not self._asks:
            return None
        return self._asks[0] - self._bids[0]

    @property
    def bid_depth(self):
        """Number of discrete bid price levels."""
        return len(self._bids)

    @property
    def ask_depth(self):
        """Number of discrete ask price levels."""
        return len(self._asks)

    @property
    def bid_volume(self):
        """Total aggregate quantity across all resting bid orders."""
        cdef long long vol = 0
        cdef CythonPriceLevel lvl
        for lvl in self._bids_map.values():
            vol += lvl.total_volume
        return vol

    @property
    def ask_volume(self):
        """Total aggregate quantity across all resting ask orders."""
        cdef long long vol = 0
        cdef CythonPriceLevel lvl
        for lvl in self._asks_map.values():
            vol += lvl.total_volume
        return vol

    @property
    def total_volume(self):
        """Total aggregate quantity across all resting orders."""
        return self.bid_volume + self.ask_volume

    def clear(self):
        """Reset order book."""
        self._bids.clear()
        self._asks.clear()
        self._bids_map.clear()
        self._asks_map.clear()
        self._order_id_to_price.clear()
        self._order_id_to_side.clear()

    def __len__(self):
        return len(self._order_id_to_price)

    def __contains__(self, str order_id):
        return order_id in self._order_id_to_price

    def __repr__(self):
        return (
            f"CythonOrderBook(bids={len(self._bids)} levels, asks={len(self._asks)} levels, "
            f"best_bid={self.best_bid}, best_ask={self.best_ask})"
        )


cdef class CythonMatchingEngine:
    """Cython-accelerated Matching Engine implementing Price-Time Priority and Maker-Price execution."""

    cdef public CythonOrderBook order_book
    cdef public int _trade_counter
    cdef public list _trades

    def __init__(self, CythonOrderBook order_book=None):
        self.order_book = order_book if order_book is not None else CythonOrderBook()
        self._trade_counter = 0
        self._trades = []

    cdef void _validate_order(self, object order):
        """Strictly validate incoming order before processing."""
        if not isinstance(order, Order):
            raise TypeError(f"Expected Order instance, got {type(order).__name__}")

        if order.order_id in self.order_book:
            raise DuplicateOrderError(
                f"Order with ID '{order.order_id}' already exists in the order book."
            )

        if not isinstance(order.order_id, str) or not order.order_id.strip():
            raise ValueError("order_id cannot be empty or whitespace")
        if order.side not in (OrderSide.BUY, OrderSide.SELL):
            raise ValueError(f"Invalid side '{order.side}'. Must be BUY or SELL.")
        validate_price(order.price)
        if isinstance(order.quantity, bool) or not isinstance(order.quantity, int) or order.quantity <= 0:
            raise ValueError(f"quantity must be strictly positive integer (> 0), got {order.quantity}")

    cpdef list process_order(self, object order):
        """Accept an incoming order and match it against compatible resting orders in the book.

        Args:
            order: Incoming Order instance.

        Returns:
            List of Trade executions generated.
        """
        self._validate_order(order)

        cdef list trades = []
        cdef long long remaining_qty = order.quantity
        cdef bint is_buy = order.is_buy
        cdef object best_ask
        cdef object best_bid
        cdef object maker_order
        cdef long long match_qty
        cdef object exec_price
        cdef object trade

        while remaining_qty > 0:
            if is_buy:
                best_ask = self.order_book.best_ask
                if best_ask is None or order.price < best_ask:
                    break
                maker_order = self.order_book.best_ask_order
            else:
                best_bid = self.order_book.best_bid
                if best_bid is None or order.price > best_bid:
                    break
                maker_order = self.order_book.best_bid_order

            if maker_order is None:
                break

            # Maker-Price execution rule: trade executes at resting order's limit price
            match_qty = remaining_qty if remaining_qty < maker_order.quantity else maker_order.quantity
            exec_price = maker_order.price

            self._trade_counter += 1
            trade = Trade(
                trade_id=self._trade_counter,
                resting_order_id=maker_order.order_id,
                incoming_order_id=order.order_id,
                price=exec_price,
                quantity=match_qty,
                timestamp=time.time_ns(),
            )
            trades.append(trade)
            self._trades.append(trade)

            remaining_qty -= match_qty

            if match_qty == maker_order.quantity:
                self.order_book.remove_order(maker_order.order_id)
            else:
                self.order_book.reduce_order_quantity(maker_order.order_id, match_qty)

        # Unfilled remainder rests in book
        if remaining_qty > 0:
            order.quantity = remaining_qty
            self.order_book.add_order(order)

        return trades

    # Method aliases matching Python reference
    match_order = process_order
    match = process_order

    @property
    def trades(self):
        """Defensive shallow copy of all executed trades."""
        return list(self._trades)

    @property
    def total_trades(self):
        """Total number of trades executed."""
        return len(self._trades)

    @property
    def total_volume_traded(self):
        """Total executed volume across all trades."""
        cdef long long total = 0
        cdef object t
        for t in self._trades:
            total += t.quantity
        return total

    @property
    def best_bid(self):
        return self.order_book.best_bid

    @property
    def best_ask(self):
        return self.order_book.best_ask

    @property
    def spread(self):
        return self.order_book.spread

    def clear(self):
        """Reset the engine and underlying book."""
        self.order_book.clear()
        self._trades.clear()
        self._trade_counter = 0

    def __repr__(self):
        return (
            f"CythonMatchingEngine(trades={len(self._trades)}, "
            f"volume={self.total_volume_traded}, "
            f"book={repr(self.order_book)})"
        )
