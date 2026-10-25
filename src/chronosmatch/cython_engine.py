"""Cython Matching Engine Interface and Fallback for ChronosMatch.

Exposes `CythonMatchingEngine`, `CythonOrderBook`, and `CythonPriceLevel`.
When the compiled C extension (`_cython_engine.pyd` / `.so`) is present, it imports
and delegates to the native extension. If the compiled binary is not present
(for example, when C compilers such as MSVC are not installed in the environment),
it transparently falls back to a high-efficiency pure-Python implementation with
100% behavioral equivalence.
"""

from __future__ import annotations

import bisect
from collections import OrderedDict
from decimal import Decimal
from pathlib import Path
import time
from typing import Any

from chronosmatch.matching_engine import Trade
from chronosmatch.models import Order, OrderSide
from chronosmatch.order_book import (
    DuplicateOrderError,
    OrderNotFoundError,
    validate_order_id,
    validate_price,
)

# ==============================================================================
# Native Extension Import & Fallback Detection
# ==============================================================================

try:
    from chronosmatch._cython_engine import (  # type: ignore[import-not-found]
        CythonMatchingEngine as _CompiledCythonMatchingEngine,
        CythonOrderBook as _CompiledCythonOrderBook,
        CythonPriceLevel as _CompiledCythonPriceLevel,
    )

    IS_CYTHON_COMPILED: bool = True
except ImportError:
    _CompiledCythonMatchingEngine = None  # type: ignore[assignment, misc]
    _CompiledCythonOrderBook = None  # type: ignore[assignment, misc]
    _CompiledCythonPriceLevel = None  # type: ignore[assignment, misc]
    IS_CYTHON_COMPILED = False


def is_cython_compiled() -> bool:
    """Return True if the native Cython C-extension binary is compiled and active."""
    return IS_CYTHON_COMPILED


def get_compilation_status() -> dict[str, Any]:
    """Return diagnostic details regarding Cython compilation status and environment."""
    c_source = Path(__file__).parent / "_cython_engine.c"
    pyx_source = Path(__file__).parent / "_cython_engine.pyx"
    return {
        "is_compiled": IS_CYTHON_COMPILED,
        "c_source_exists": c_source.exists(),
        "pyx_source_exists": pyx_source.exists(),
        "build_limitation": (
            None
            if IS_CYTHON_COMPILED
            else 'Microsoft Visual C++ 14.0 or greater is required to compile .pyd extension on Windows ("Microsoft C++ Build Tools"). Running in high-performance pure-Python mode.'
        ),
    }


# ==============================================================================
# Pure-Python Reference Fallback Implementation
# ==============================================================================


class _PyCythonPriceLevel:
    """Price level maintaining resting orders in strict FIFO time priority."""

    __slots__ = ("price", "_orders", "_total_volume")

    def __init__(self, price: float | int | Decimal) -> None:
        validate_price(price)
        self.price = price
        self._orders: OrderedDict[str, Order] = OrderedDict()
        self._total_volume: int = 0

    def add_order(self, order: Order) -> None:
        """Append an order to the tail of this price level (FIFO)."""
        self._orders[order.order_id] = order
        self._total_volume += order.quantity

    def remove_order(self, order_id: str) -> Order:
        """Remove and return an order from this price level by order_id."""
        order = self._orders.pop(order_id)
        self._total_volume -= order.quantity
        return order

    def reduce_order_quantity(self, order_id: str, filled_qty: int) -> Order:
        """Reduce the resting quantity of an order by filled_qty (partial fill)."""
        if order_id not in self._orders:
            raise KeyError(f"Order '{order_id}' not found at price level {self.price}")
        if filled_qty <= 0:
            raise ValueError(f"filled_qty must be strictly positive (> 0), got {filled_qty}")

        order = self._orders[order_id]
        if filled_qty >= order.quantity:
            raise ValueError(
                f"filled_qty {filled_qty} must be strictly less than order quantity {order.quantity}. "
                f"Use remove_order for full fills."
            )

        order.quantity -= filled_qty
        self._total_volume -= filled_qty
        return order

    def get_order(self, order_id: str) -> Order | None:
        """Retrieve an order by ID at this price level without removing it."""
        return self._orders.get(order_id)

    @property
    def orders(self) -> list[Order]:
        """Return defensive shallow copy of orders at this level in FIFO order."""
        return list(self._orders.values())

    @property
    def head_order(self) -> Order | None:
        """Return the highest time-priority order at this price level (head of FIFO queue)."""
        if not self._orders:
            return None
        return next(iter(self._orders.values()))

    @property
    def total_volume(self) -> int:
        """Return the aggregate quantity of all resting orders at this price level."""
        return self._total_volume

    def __len__(self) -> int:
        return len(self._orders)

    def __contains__(self, order_id: str) -> bool:
        return order_id in self._orders

    def __repr__(self) -> str:
        return f"CythonPriceLevel(price={self.price}, orders={len(self._orders)}, volume={self._total_volume})"


class _PyCythonOrderBookSide:
    """Represents one side (bids or asks) of the CythonOrderBook."""

    __slots__ = ("side", "_book")

    def __init__(self, side: OrderSide, book: _PyCythonOrderBook) -> None:
        self.side = side
        self._book = book

    @property
    def is_buy(self) -> bool:
        return self.side == OrderSide.BUY

    @property
    def price_levels(self) -> list[float | int | Decimal]:
        return list(self._book._bids if self.is_buy else self._book._asks)

    @property
    def best_price(self) -> float | int | Decimal | None:
        return self._book.best_bid if self.is_buy else self._book.best_ask

    @property
    def best_order(self) -> Order | None:
        return self._book.best_bid_order if self.is_buy else self._book.best_ask_order

    @property
    def total_volume(self) -> int:
        return self._book.bid_volume if self.is_buy else self._book.ask_volume

    @property
    def is_empty(self) -> bool:
        prices = self._book._bids if self.is_buy else self._book._asks
        return len(prices) == 0

    def depth(self, n: int = 10) -> list[tuple[float | int | Decimal, int]]:
        prices = (self._book._bids if self.is_buy else self._book._asks)[:n]
        mapping = self._book._bids_map if self.is_buy else self._book._asks_map
        return [(p, mapping[p].total_volume) for p in prices]

    def __len__(self) -> int:
        mapping = self._book._bids_map if self.is_buy else self._book._asks_map
        return sum(len(lvl) for lvl in mapping.values())


class _PyCythonOrderBook:
    """Limit order book managing bids and asks organized by price-time priority."""

    __slots__ = (
        "_bids",
        "_asks",
        "_bids_map",
        "_asks_map",
        "_order_id_to_price",
        "_order_id_to_side",
        "_bids_side",
        "_asks_side",
    )

    def __init__(self) -> None:
        self._bids: list[float | int | Decimal] = []
        self._asks: list[float | int | Decimal] = []
        self._bids_map: dict[float | int | Decimal, _PyCythonPriceLevel] = {}
        self._asks_map: dict[float | int | Decimal, _PyCythonPriceLevel] = {}
        self._order_id_to_price: dict[str, float | int | Decimal] = {}
        self._order_id_to_side: dict[str, OrderSide] = {}
        self._bids_side = _PyCythonOrderBookSide(OrderSide.BUY, self)
        self._asks_side = _PyCythonOrderBookSide(OrderSide.SELL, self)



    def _insert_bid_price(self, price: float | int | Decimal) -> None:
        """Insert price into descending _bids list via binary search."""
        low = 0
        high = len(self._bids)
        while low < high:
            mid = (low + high) // 2
            if self._bids[mid] < price:
                high = mid
            else:
                low = mid + 1
        self._bids.insert(low, price)

    def _insert_ask_price(self, price: float | int | Decimal) -> None:
        """Insert price into ascending _asks list via bisect."""
        bisect.insort(self._asks, price)

    def add_order(self, order: Order) -> None:
        """Add an order into the order book."""
        if not isinstance(order, Order):
            raise TypeError(f"Expected Order instance, got {type(order).__name__}")
        if order.order_id in self._order_id_to_price:
            raise DuplicateOrderError(
                f"Order with ID '{order.order_id}' already exists in the order book."
            )

        side = order.side
        price = order.price

        if side == OrderSide.BUY:
            if price not in self._bids_map:
                level = _PyCythonPriceLevel(price)
                self._bids_map[price] = level
                self._insert_bid_price(price)
            else:
                level = self._bids_map[price]
            level.add_order(order)
            self._order_id_to_price[order.order_id] = price
            self._order_id_to_side[order.order_id] = OrderSide.BUY
        elif side == OrderSide.SELL:
            if price not in self._asks_map:
                level = _PyCythonPriceLevel(price)
                self._asks_map[price] = level
                self._insert_ask_price(price)
            else:
                level = self._asks_map[price]
            level.add_order(order)
            self._order_id_to_price[order.order_id] = price
            self._order_id_to_side[order.order_id] = OrderSide.SELL
        else:
            raise ValueError(f"Invalid side '{side}'. Must be BUY or SELL.")

    def remove_order(self, order_id: str) -> Order:
        """Remove and return an order from the book by order_id."""
        validate_order_id(order_id)
        if order_id not in self._order_id_to_price:
            raise OrderNotFoundError(f"Order '{order_id}' not found in the order book.")

        price = self._order_id_to_price.pop(order_id)
        side = self._order_id_to_side.pop(order_id)

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

    cancel_order = remove_order

    def reduce_order_quantity(self, order_id: str, filled_qty: int) -> Order:
        """Reduce resting order quantity by filled_qty."""
        if order_id not in self._order_id_to_price:
            raise OrderNotFoundError(f"Order '{order_id}' not found in the order book.")

        price = self._order_id_to_price[order_id]
        side = self._order_id_to_side[order_id]

        if side == OrderSide.BUY:
            level = self._bids_map[price]
        else:
            level = self._asks_map[price]

        return level.reduce_order_quantity(order_id, filled_qty)

    def get_order(self, order_id: str) -> Order | None:
        """Retrieve an order by ID without removing it."""
        if order_id not in self._order_id_to_price:
            return None
        price = self._order_id_to_price[order_id]
        side = self._order_id_to_side[order_id]
        if side == OrderSide.BUY:
            return self._bids_map[price].get_order(order_id)
        return self._asks_map[price].get_order(order_id)

    @property
    def best_bid(self) -> float | int | Decimal | None:
        """Highest resting buy price, or None if no bids."""
        return self._bids[0] if self._bids else None

    @property
    def best_ask(self) -> float | int | Decimal | None:
        """Lowest resting sell price, or None if no asks."""
        return self._asks[0] if self._asks else None

    @property
    def best_bid_order(self) -> Order | None:
        """Earliest arrival order at best bid price, or None."""
        if not self._bids:
            return None
        return self._bids_map[self._bids[0]].head_order

    @property
    def best_ask_order(self) -> Order | None:
        """Earliest arrival order at best ask price, or None."""
        if not self._asks:
            return None
        return self._asks_map[self._asks[0]].head_order

    @property
    def spread(self) -> float | int | Decimal | None:
        """Difference between best ask and best bid, or None if book is one-sided or empty."""
        if not self._bids or not self._asks:
            return None
        return self._asks[0] - self._bids[0]

    @property
    def bids(self) -> _PyCythonOrderBookSide:
        """Access buy side of the order book."""
        return self._bids_side

    @property
    def asks(self) -> _PyCythonOrderBookSide:
        """Access sell side of the order book."""
        return self._asks_side

    @property
    def bid_depth(self) -> int:
        """Number of discrete bid price levels."""
        return len(self._bids)

    @property
    def ask_depth(self) -> int:
        """Number of discrete ask price levels."""
        return len(self._asks)

    @property
    def bid_volume(self) -> int:
        """Total aggregate quantity across all resting bid orders."""
        return sum(lvl.total_volume for lvl in self._bids_map.values())

    @property
    def ask_volume(self) -> int:
        """Total aggregate quantity across all resting ask orders."""
        return sum(lvl.total_volume for lvl in self._asks_map.values())

    @property
    def total_volume(self) -> int:
        """Total aggregate quantity across all resting orders."""
        return self.bid_volume + self.ask_volume

    def clear(self) -> None:
        """Reset order book."""
        self._bids.clear()
        self._asks.clear()
        self._bids_map.clear()
        self._asks_map.clear()
        self._order_id_to_price.clear()
        self._order_id_to_side.clear()

    def __len__(self) -> int:
        return len(self._order_id_to_price)

    def __contains__(self, order_id: str) -> bool:
        return order_id in self._order_id_to_price

    def __repr__(self) -> str:
        return (
            f"CythonOrderBook(bids={len(self._bids)} levels, asks={len(self._asks)} levels, "
            f"best_bid={self.best_bid}, best_ask={self.best_ask})"
        )


class _PyCythonMatchingEngine:
    """Matching engine implementing Price-Time Priority and Maker-Price execution."""

    __slots__ = ("order_book", "_trade_counter", "_trades")

    def __init__(self, order_book: Any = None) -> None:
        self.order_book: _PyCythonOrderBook = (
            order_book if order_book is not None else _PyCythonOrderBook()
        )
        self._trade_counter: int = 0
        self._trades: list[Trade] = []

    def _validate_order(self, order: Any) -> None:
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

    def process_order(self, order: Order) -> list[Trade]:
        """Accept an incoming order and match it against compatible resting orders in the book.

        Args:
            order: Incoming Order instance.

        Returns:
            List of Trade executions generated.
        """
        self._validate_order(order)

        trades: list[Trade] = []
        remaining_qty: int = order.quantity
        is_buy: bool = order.is_buy

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
            match_qty = min(remaining_qty, maker_order.quantity)
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

    # Method aliases for flexibility
    match_order = process_order
    match = process_order

    def cancel_order(self, order_id: str) -> Order:
        """Cancel and remove an order from the underlying order book."""
        return self.order_book.remove_order(order_id)

    @property
    def trades(self) -> list[Trade]:
        """Defensive shallow copy of all executed trades."""
        return list(self._trades)

    @property
    def total_trades(self) -> int:
        """Total number of trades executed."""
        return len(self._trades)

    @property
    def total_volume_traded(self) -> int:
        """Total executed volume across all trades."""
        return sum(t.quantity for t in self._trades)

    @property
    def best_bid(self) -> float | int | Decimal | None:
        return self.order_book.best_bid

    @property
    def best_ask(self) -> float | int | Decimal | None:
        return self.order_book.best_ask

    @property
    def spread(self) -> float | int | Decimal | None:
        return self.order_book.spread

    def clear(self) -> None:
        """Reset the engine and underlying book."""
        self.order_book.clear()
        self._trades.clear()
        self._trade_counter = 0

    def __repr__(self) -> str:
        return (
            f"CythonMatchingEngine(trades={len(self._trades)}, "
            f"volume={self.total_volume_traded}, "
            f"book={repr(self.order_book)})"
        )


# ==============================================================================
# Public Module Exports
# ==============================================================================

if IS_CYTHON_COMPILED:
    CythonPriceLevel = _CompiledCythonPriceLevel
    CythonOrderBook = _CompiledCythonOrderBook
    CythonMatchingEngine = _CompiledCythonMatchingEngine
else:
    CythonPriceLevel = _PyCythonPriceLevel  # type: ignore[misc]
    CythonOrderBook = _PyCythonOrderBook  # type: ignore[misc]
    CythonMatchingEngine = _PyCythonMatchingEngine  # type: ignore[misc]

__all__ = [
    "CythonMatchingEngine",
    "CythonOrderBook",
    "CythonPriceLevel",
    "IS_CYTHON_COMPILED",
    "is_cython_compiled",
    "get_compilation_status",
]
