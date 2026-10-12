"""Limit Order Book (LOB) implementation for ChronosMatch.

Provides deterministic in-memory order storage organized by Price-Time Priority (FIFO)
with separate BUY (bid) and SELL (ask) sides.
"""

from __future__ import annotations

import bisect
from collections import OrderedDict
from dataclasses import dataclass, field
from decimal import Decimal
import math
from typing import Any

from chronosmatch.models import Order, OrderSide


class OrderBookError(Exception):
    """Base exception class for all order book related errors."""


class DuplicateOrderError(OrderBookError, ValueError):
    """Raised when an order with an existing order_id is added to the order book."""


class OrderNotFoundError(OrderBookError, KeyError):
    """Raised when an order cannot be found in the order book."""


def validate_price(price: Any) -> None:
    """Validate price following ChronosMatch Order model validation rules.

    Args:
        price: Price value to validate.

    Raises:
        TypeError: If price is not a real number (int, float, Decimal) or is a bool.
        ValueError: If price is non-finite (NaN, inf) or not strictly positive (> 0).
    """
    if isinstance(price, bool) or not isinstance(price, (int, float, Decimal)):
        raise TypeError(f"price must be a real number (int, float, Decimal), got {type(price).__name__}")
    if isinstance(price, float) and (math.isnan(price) or math.isinf(price)):
        raise ValueError(f"price must be finite, got {price}")
    if isinstance(price, Decimal) and (price.is_nan() or price.is_infinite()):
        raise ValueError(f"price must be finite, got {price}")
    if price <= 0:
        raise ValueError(f"price must be strictly positive (> 0), got {price}")


def validate_order_id(order_id: Any) -> None:
    """Validate order_id string.

    Args:
        order_id: Order identifier to validate.

    Raises:
        TypeError: If order_id is not a string.
        ValueError: If order_id is empty or consists solely of whitespace.
    """
    if not isinstance(order_id, str):
        raise TypeError(f"order_id must be a string, got {type(order_id).__name__}")
    if not order_id.strip():
        raise ValueError("order_id cannot be empty or whitespace")


class PriceLevel:
    """Represents a discrete price level containing resting orders.

    Orders at this price level are maintained in strict arrival order (FIFO)
    to enforce price-time priority.
    """

    __slots__ = ("price", "_orders", "_total_volume")

    def __init__(self, price: float | int | Decimal) -> None:
        validate_price(price)
        self.price = price
        self._orders: OrderedDict[str, Order] = OrderedDict()
        self._total_volume: int = 0

    def add_order(self, order: Order) -> None:
        """Append an order to the tail of this price level (FIFO).

        Args:
            order: The Order instance to add.
        """
        self._orders[order.order_id] = order
        self._total_volume += order.quantity

    def remove_order(self, order_id: str) -> Order:
        """Remove and return an order from this price level by order_id.

        Args:
            order_id: Identifier of the order to remove.

        Returns:
            The removed Order instance.

        Raises:
            KeyError: If the order_id is not present at this price level.
        """
        order = self._orders.pop(order_id)
        self._total_volume -= order.quantity
        return order

    def reduce_order_quantity(self, order_id: str, filled_qty: int) -> Order:
        """Reduce the resting quantity of an order by filled_qty (partial fill).

        Args:
            order_id: Identifier of the order to reduce.
            filled_qty: Executed quantity to deduct (must be strictly positive and strictly less than order.quantity).

        Returns:
            The partially filled Order instance.

        Raises:
            KeyError: If order_id is not present at this price level.
            TypeError: If filled_qty is not an integer or is a boolean.
            ValueError: If filled_qty is <= 0 or >= order.quantity.
        """
        if order_id not in self._orders:
            raise KeyError(f"Order '{order_id}' not found at price level {self.price}")
        if isinstance(filled_qty, bool) or not isinstance(filled_qty, int):
            raise TypeError(f"filled_qty must be an integer, got {type(filled_qty).__name__}")
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
        """Return a defensive shallow copy of orders at this level in FIFO order."""
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

    @property
    def order_count(self) -> int:
        """Return the number of resting orders at this price level."""
        return len(self._orders)

    @property
    def is_empty(self) -> bool:
        """Return True if there are no resting orders at this price level."""
        return len(self._orders) == 0

    def __len__(self) -> int:
        return len(self._orders)

    def __contains__(self, order_id: str) -> bool:
        return order_id in self._orders

    def __repr__(self) -> str:
        return f"PriceLevel(price={self.price}, orders={len(self._orders)}, volume={self._total_volume})"


class OrderBookSide:
    """Represents one side of the order book (BUY or SELL).

    Maintains price levels sorted by priority:
    - BUY (bids): Highest price has highest priority.
    - SELL (asks): Lowest price has highest priority.
    """

    __slots__ = ("side", "_price_levels", "_prices")

    def __init__(self, side: OrderSide) -> None:
        if not isinstance(side, OrderSide):
            raise TypeError(f"side must be an OrderSide instance, got {type(side).__name__}")
        self.side = side
        self._price_levels: dict[float | int | Decimal, PriceLevel] = {}
        # _prices is maintained in ascending sorted order
        self._prices: list[float | int | Decimal] = []

    @property
    def is_buy(self) -> bool:
        """Return True if this is the BUY side."""
        return self.side == OrderSide.BUY

    @property
    def is_sell(self) -> bool:
        """Return True if this is the SELL side."""
        return self.side == OrderSide.SELL

    @property
    def best_price(self) -> float | int | Decimal | None:
        """Return the best price on this side (highest for bids, lowest for asks), or None if empty."""
        if not self._prices:
            return None
        return self._prices[-1] if self.is_buy else self._prices[0]

    @property
    def best_order(self) -> Order | None:
        """Return the highest priority order on this side (best price, earliest arrival), or None."""
        bp = self.best_price
        if bp is None:
            return None
        return self._price_levels[bp].head_order

    @property
    def price_levels(self) -> list[float | int | Decimal]:
        """Return all distinct price levels sorted by priority.

        - Bids: Descending order (highest price first).
        - Asks: Ascending order (lowest price first).
        """
        if self.is_buy:
            return list(reversed(self._prices))
        return list(self._prices)

    def add_order(self, order: Order) -> None:
        """Add an order to this side of the book."""
        if order.side != self.side:
            side_str = order.side.value if isinstance(order.side, OrderSide) else order.side
            raise ValueError(f"Cannot add order with side {side_str} to {self.side.value} side of the book.")

        price = order.price
        if price not in self._price_levels:
            level = PriceLevel(price)
            self._price_levels[price] = level
            bisect.insort(self._prices, price)
        else:
            level = self._price_levels[price]

        level.add_order(order)

    def remove_order(self, order_id: str, price: float | int | Decimal) -> Order:
        """Remove an order from a price level. If that level becomes empty, prune it.

        Args:
            order_id: ID of the order to remove.
            price: Price level where the order rests.

        Returns:
            The removed Order instance.

        Raises:
            KeyError: If price level or order_id is not found.
        """
        level = self._price_levels[price]
        order = level.remove_order(order_id)

        if level.is_empty:
            del self._price_levels[price]
            idx = bisect.bisect_left(self._prices, price)
            if idx < len(self._prices) and self._prices[idx] == price:
                del self._prices[idx]
            else:
                self._prices.remove(price)

        return order

    def reduce_order_quantity(
        self, order_id: str, price: float | int | Decimal, filled_qty: int
    ) -> Order:
        """Reduce the resting quantity of an order at a price level by filled_qty.

        Args:
            order_id: ID of the order to reduce.
            price: Price level where the order rests.
            filled_qty: Quantity filled.

        Returns:
            The partially filled Order instance.

        Raises:
            KeyError: If price level or order_id is not found.
            TypeError: If filled_qty has an invalid type.
            ValueError: If filled_qty is invalid.
        """
        level = self._price_levels[price]
        return level.reduce_order_quantity(order_id, filled_qty)

    def get_orders_at_price(self, price: float | int | Decimal) -> list[Order]:
        """Return list of orders resting at the given price level in FIFO order."""
        level = self._price_levels.get(price)
        return level.orders if level is not None else []

    def get_volume_at_price(self, price: float | int | Decimal) -> int:
        """Return total volume resting at the given price level."""
        level = self._price_levels.get(price)
        return level.total_volume if level is not None else 0

    def get_order_count_at_price(self, price: float | int | Decimal) -> int:
        """Return total number of orders resting at the given price level."""
        level = self._price_levels.get(price)
        return level.order_count if level is not None else 0

    def depth(self, n: int = 10) -> list[tuple[float | int | Decimal, int]]:
        """Return Level 2 aggregated market depth [(price, total_volume), ...] up to n levels."""
        result: list[tuple[float | int | Decimal, int]] = []
        for p in self.price_levels[:n]:
            result.append((p, self._price_levels[p].total_volume))
        return result

    @property
    def total_volume(self) -> int:
        """Return total aggregate volume across all price levels on this side."""
        return sum(level.total_volume for level in self._price_levels.values())

    @property
    def order_count(self) -> int:
        """Return total count of orders resting on this side."""
        return sum(level.order_count for level in self._price_levels.values())

    @property
    def is_empty(self) -> bool:
        """Return True if this side has no resting orders."""
        return len(self._prices) == 0

    def clear(self) -> None:
        """Remove all orders and price levels from this side."""
        self._price_levels.clear()
        self._prices.clear()

    def __len__(self) -> int:
        return self.order_count

    def __repr__(self) -> str:
        return (
            f"OrderBookSide(side={self.side.value}, "
            f"levels={len(self._prices)}, "
            f"orders={self.order_count}, "
            f"best={self.best_price})"
        )


class LimitOrderBook:
    """Deterministic Limit Order Book (LOB) implementation for ChronosMatch.

    Maintains separate BUY and SELL sides organized by price-time priority (FIFO).
    Provides O(1) order cancellation by order_id, O(1) BBO (Best Bid & Offer) lookups,
    and Level 2/Level 3 depth inspection.
    """

    __slots__ = ("bids", "asks", "_orders")

    def __init__(self) -> None:
        self.bids = OrderBookSide(OrderSide.BUY)
        self.asks = OrderBookSide(OrderSide.SELL)
        self._orders: dict[str, Order] = {}

    def add_order(self, order: Order) -> None:
        """Add a valid order to the appropriate side of the order book.

        Args:
            order: Order instance to add.

        Raises:
            TypeError: If order is not an Order instance.
            DuplicateOrderError: If order.order_id already exists in the book.
            ValueError: If order has an unrecognized side.
        """
        if not isinstance(order, Order):
            raise TypeError(f"Expected Order instance, got {type(order).__name__}")

        if order.order_id in self._orders:
            raise DuplicateOrderError(
                f"Order with ID '{order.order_id}' already exists in the order book."
            )

        if order.is_buy:
            self.bids.add_order(order)
        elif order.is_sell:
            self.asks.add_order(order)
        else:
            raise ValueError(f"Unsupported order side: {order.side}")

        self._orders[order.order_id] = order

    def remove_order(self, order_id: str) -> Order:
        """Remove an existing order by its ID from the order book.

        Args:
            order_id: Identifier of the order to remove.

        Returns:
            The removed Order instance.

        Raises:
            TypeError: If order_id is not a string.
            ValueError: If order_id is empty or whitespace.
            OrderNotFoundError: If order_id is not found in the book.
        """
        validate_order_id(order_id)

        if order_id not in self._orders:
            raise OrderNotFoundError(
                f"Order with ID '{order_id}' not found in the order book."
            )

        order = self._orders.pop(order_id)
        if order.is_buy:
            self.bids.remove_order(order.order_id, order.price)
        else:
            self.asks.remove_order(order.order_id, order.price)

        return order

    def reduce_order_quantity(self, order_id: str, filled_qty: int) -> Order:
        """Partially fill an existing resting order by reducing its quantity.

        Args:
            order_id: ID of the resting order.
            filled_qty: Quantity executed in the fill.

        Returns:
            The partially filled Order instance.

        Raises:
            TypeError: If order_id or filled_qty has an invalid type.
            ValueError: If order_id is empty or filled_qty is invalid.
            OrderNotFoundError: If order_id is not found in the book.
        """
        validate_order_id(order_id)
        if order_id not in self._orders:
            raise OrderNotFoundError(
                f"Order with ID '{order_id}' not found in the order book."
            )

        order = self._orders[order_id]
        if order.is_buy:
            return self.bids.reduce_order_quantity(order.order_id, order.price, filled_qty)
        else:
            return self.asks.reduce_order_quantity(order.order_id, order.price, filled_qty)

    def cancel_order(self, order_id: str) -> Order:
        """Convenience alias for remove_order."""
        return self.remove_order(order_id)

    def get_order(self, order_id: str) -> Order | None:
        """Retrieve an order by its ID without removing it, or None if not found.

        Args:
            order_id: Identifier of the order.

        Returns:
            The Order instance if present, else None.

        Raises:
            TypeError: If order_id is not a string.
            ValueError: If order_id is empty or whitespace.
        """
        validate_order_id(order_id)
        return self._orders.get(order_id)

    @property
    def best_bid(self) -> float | int | Decimal | None:
        """Highest resting BUY limit price, or None if the bid side is empty."""
        return self.bids.best_price

    @property
    def best_ask(self) -> float | int | Decimal | None:
        """Lowest resting SELL limit price, or None if the ask side is empty."""
        return self.asks.best_price

    def get_best_bid(self) -> float | int | Decimal | None:
        """Return the best bid price, or None if empty."""
        return self.best_bid

    def get_best_ask(self) -> float | int | Decimal | None:
        """Return the best ask price, or None if empty."""
        return self.best_ask

    @property
    def best_bid_order(self) -> Order | None:
        """Highest priority BUY order (best price, earliest arrival), or None if empty."""
        return self.bids.best_order

    @property
    def best_ask_order(self) -> Order | None:
        """Highest priority SELL order (best price, earliest arrival), or None if empty."""
        return self.asks.best_order

    def get_best_bid_order(self) -> Order | None:
        """Return the highest priority BUY order, or None if empty."""
        return self.best_bid_order

    def get_best_ask_order(self) -> Order | None:
        """Return the highest priority SELL order, or None if empty."""
        return self.best_ask_order

    @property
    def spread(self) -> float | int | Decimal | None:
        """Current bid-ask spread (best_ask - best_bid), or None if either side is empty."""
        bb = self.best_bid
        ba = self.best_ask
        if bb is not None and ba is not None:
            return ba - bb
        return None

    def _resolve_side(self, side: OrderSide | str | None) -> OrderSide | None:
        """Normalize side argument if provided."""
        if side is None:
            return None
        if isinstance(side, OrderSide):
            return side
        if isinstance(side, str):
            if side in (OrderSide.BUY.value, OrderSide.SELL.value):
                return OrderSide(side)
            raise ValueError(f"Invalid side '{side}'. Must be BUY or SELL.")
        raise TypeError(f"side must be an OrderSide or str ('BUY' or 'SELL'), got {type(side).__name__}")

    def get_orders_at_price(
        self,
        price: float | int | Decimal,
        side: OrderSide | str | None = None,
    ) -> list[Order]:
        """Return resting orders at the specified price level in FIFO arrival order.

        Args:
            price: Limit price level to inspect.
            side: Optional side filter (BUY or SELL). If None, checks both sides.

        Returns:
            List of Order objects resting at the given price level.

        Raises:
            TypeError: If price or side type is invalid.
            ValueError: If price or side value is invalid.
        """
        validate_price(price)
        resolved_side = self._resolve_side(side)

        if resolved_side == OrderSide.BUY:
            return self.bids.get_orders_at_price(price)
        if resolved_side == OrderSide.SELL:
            return self.asks.get_orders_at_price(price)

        # Neither side specified: combine bids and asks
        return self.bids.get_orders_at_price(price) + self.asks.get_orders_at_price(price)

    def get_volume_at_price(
        self,
        price: float | int | Decimal,
        side: OrderSide | str | None = None,
    ) -> int:
        """Return aggregate volume at the specified price level."""
        validate_price(price)
        resolved_side = self._resolve_side(side)

        if resolved_side == OrderSide.BUY:
            return self.bids.get_volume_at_price(price)
        if resolved_side == OrderSide.SELL:
            return self.asks.get_volume_at_price(price)

        return self.bids.get_volume_at_price(price) + self.asks.get_volume_at_price(price)

    def get_order_count_at_price(
        self,
        price: float | int | Decimal,
        side: OrderSide | str | None = None,
    ) -> int:
        """Return order count at the specified price level."""
        validate_price(price)
        resolved_side = self._resolve_side(side)

        if resolved_side == OrderSide.BUY:
            return self.bids.get_order_count_at_price(price)
        if resolved_side == OrderSide.SELL:
            return self.asks.get_order_count_at_price(price)

        return self.bids.get_order_count_at_price(price) + self.asks.get_order_count_at_price(price)

    def get_bids_depth(self, depth: int = 10) -> list[tuple[float | int | Decimal, int]]:
        """Return Level 2 aggregated bid depth [(price, volume), ...] up to depth levels."""
        return self.bids.depth(depth)

    def get_asks_depth(self, depth: int = 10) -> list[tuple[float | int | Decimal, int]]:
        """Return Level 2 aggregated ask depth [(price, volume), ...] up to depth levels."""
        return self.asks.depth(depth)

    def get_l2_depth(
        self, depth: int = 10
    ) -> tuple[list[tuple[float | int | Decimal, int]], list[tuple[float | int | Decimal, int]]]:
        """Return Level 2 market depth as a (bids_depth, asks_depth) tuple."""
        return self.bids.depth(depth), self.asks.depth(depth)

    @property
    def bids_count(self) -> int:
        """Total number of resting BUY orders."""
        return self.bids.order_count

    @property
    def asks_count(self) -> int:
        """Total number of resting SELL orders."""
        return self.asks.order_count

    @property
    def total_volume(self) -> int:
        """Total aggregate quantity across all resting orders on both sides."""
        return self.bids.total_volume + self.asks.total_volume

    @property
    def is_empty(self) -> bool:
        """Return True if both sides of the book are empty."""
        return len(self._orders) == 0

    def clear(self) -> None:
        """Reset the order book to an empty state."""
        self.bids.clear()
        self.asks.clear()
        self._orders.clear()

    def __contains__(self, order_id: str) -> bool:
        """Return True if order_id is currently resting in the book."""
        return order_id in self._orders

    def __len__(self) -> int:
        """Return total count of orders resting in the book."""
        return len(self._orders)

    def __repr__(self) -> str:
        return (
            f"LimitOrderBook("
            f"bids={self.bids_count} levels={len(self.bids._prices)} best_bid={self.best_bid}, "
            f"asks={self.asks_count} levels={len(self.asks._prices)} best_ask={self.best_ask})"
        )


# Convenience alias
OrderBook = LimitOrderBook
