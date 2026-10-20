"""Deterministic Price-Time Priority Matching Engine for ChronosMatch.

Executes aggressive incoming orders against resting Limit Order Book depth
using strict Price-Time Priority (FIFO) and Maker-Price execution semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
import time
from typing import Any

from chronosmatch.models import Order, OrderSide
from chronosmatch.order_book import (
    DuplicateOrderError,
    LimitOrderBook,
    validate_order_id,
    validate_price,
)


@dataclass(slots=True)
class Trade:
    """Represents an executed trade between a resting (maker) order and an incoming (taker) order.

    Attributes:
        trade_id: Unique sequential identifier for the trade execution.
        resting_order_id: Order ID of the passive resting order in the book (maker).
        incoming_order_id: Order ID of the aggressive incoming order (taker).
        price: Traded price, matching the resting maker order's limit price.
        quantity: Executed contract/share quantity.
        timestamp: Nanosecond timestamp when the match occurred.
    """

    trade_id: int | str
    resting_order_id: str
    incoming_order_id: str
    price: float | int | Decimal
    quantity: int
    timestamp: int = field(default_factory=time.time_ns)

    def __post_init__(self) -> None:
        """Validate trade fields to guarantee execution record integrity."""
        self._validate_trade_id()
        validate_order_id(self.resting_order_id)
        validate_order_id(self.incoming_order_id)
        validate_price(self.price)
        self._validate_quantity()
        self._validate_timestamp()

    def _validate_trade_id(self) -> None:
        if isinstance(self.trade_id, bool):
            raise TypeError("trade_id cannot be a boolean")
        if isinstance(self.trade_id, int):
            if self.trade_id <= 0:
                raise ValueError(f"trade_id must be strictly positive (> 0), got {self.trade_id}")
            return
        if isinstance(self.trade_id, str):
            if not self.trade_id.strip():
                raise ValueError("trade_id cannot be empty or whitespace")
            return
        raise TypeError(f"trade_id must be int or str, got {type(self.trade_id).__name__}")

    def _validate_quantity(self) -> None:
        if isinstance(self.quantity, bool):
            raise TypeError("quantity cannot be a boolean")
        if not isinstance(self.quantity, int):
            raise TypeError(f"quantity must be an integer, got {type(self.quantity).__name__}")
        if self.quantity <= 0:
            raise ValueError(f"quantity must be strictly positive (> 0), got {self.quantity}")

    def _validate_timestamp(self) -> None:
        if isinstance(self.timestamp, bool):
            raise TypeError("timestamp cannot be a boolean")
        if not isinstance(self.timestamp, int):
            raise TypeError(f"timestamp must be an integer (nanoseconds), got {type(self.timestamp).__name__}")
        if self.timestamp <= 0:
            raise ValueError(f"timestamp must be strictly positive (> 0), got {self.timestamp}")

    # Financial / HFT domain convenience properties
    @property
    def maker_order_id(self) -> str:
        """Alias for resting_order_id."""
        return self.resting_order_id

    @property
    def taker_order_id(self) -> str:
        """Alias for incoming_order_id."""
        return self.incoming_order_id

    @property
    def execution_price(self) -> float | int | Decimal:
        """Alias for price."""
        return self.price

    @property
    def executed_quantity(self) -> int:
        """Alias for quantity."""
        return self.quantity

    @property
    def execution_qty(self) -> int:
        """Alias for quantity."""
        return self.quantity

    @property
    def timestamp_ns(self) -> int:
        """Alias for timestamp."""
        return self.timestamp


# Convenience alias matching architecture documentation
TradeEvent = Trade


class MatchingEngine:
    """Core deterministic order-matching engine for ChronosMatch.

    Execution Price Rule:
        Trades execute at the resting (maker) order's limit price.
        The incoming (taker) order receives price improvement whenever its
        limit price is more aggressive than the resting order's limit price.

    Matching Semantics:
        - BUY orders match the lowest-priced resting SELL order when buy price >= sell price.
        - SELL orders match the highest-priced resting BUY order when sell price <= buy price.
        - Resting orders are matched in strict Price-Time Priority (FIFO).
        - Executions are sized to min(incoming quantity, resting order quantity).
        - Partial fills leave remaining resting quantity in the book at the same priority.
        - Fully filled resting orders are pruned from the book.
        - Any unfilled balance of the incoming order rests in the Limit Order Book.
    """

    __slots__ = ("order_book", "_trade_counter", "_trades")

    def __init__(self, order_book: LimitOrderBook | None = None) -> None:
        self.order_book: LimitOrderBook = order_book if order_book is not None else LimitOrderBook()
        self._trade_counter: int = 0
        self._trades: list[Trade] = []

    def _validate_order(self, order: Any) -> None:
        """Strictly validate incoming order before processing."""
        if not isinstance(order, Order):
            raise TypeError(f"Expected Order instance, got {type(order).__name__}")

        # Check for duplicate order ID against existing resting orders
        if order.order_id in self.order_book:
            raise DuplicateOrderError(
                f"Order with ID '{order.order_id}' already exists in the order book."
            )

        # Validate order attributes to ensure no corrupted/silently altered orders enter
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
            order: The incoming Order instance.

        Returns:
            List of Trade executions generated by this order.

        Raises:
            TypeError: If order is not an Order instance or has invalid types.
            ValueError: If order has invalid values.
            DuplicateOrderError: If order.order_id already exists in the book.
        """
        self._validate_order(order)

        trades: list[Trade] = []
        remaining_qty = order.quantity

        while remaining_qty > 0:
            if order.is_buy:
                best_ask = self.order_book.best_ask
                # Compatibility check: BUY matches SELL if buy price >= sell price
                if best_ask is None or order.price < best_ask:
                    break
                maker_order = self.order_book.best_ask_order
            else:
                best_bid = self.order_book.best_bid
                # Compatibility check: SELL matches BUY if sell price <= buy price
                if best_bid is None or order.price > best_bid:
                    break
                maker_order = self.order_book.best_bid_order

            if maker_order is None:
                break

            # Execution details: Maker-Price execution rule
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

            # Update resting maker order in book
            if match_qty == maker_order.quantity:
                # Fully filled: remove from book
                self.order_book.remove_order(maker_order.order_id)
            else:
                # Partially filled: reduce resting quantity in place
                self.order_book.reduce_order_quantity(maker_order.order_id, match_qty)

        # Unfilled incoming balance rests in the book
        if remaining_qty > 0:
            order.quantity = remaining_qty
            self.order_book.add_order(order)

        return trades

    # Method aliases for flexibility
    match_order = process_order
    match = process_order

    def cancel_order(self, order_id: str) -> Order:
        """Cancel and remove an order from the underlying order book.

        Args:
            order_id: Identifier of the resting order to cancel.

        Returns:
            The removed Order instance.

        Raises:
            OrderNotFoundError: If the order_id is not found in the book.
        """
        return self.order_book.cancel_order(order_id)

    @property
    def trades(self) -> list[Trade]:
        """Return defensive shallow copy of all trade executions."""
        return list(self._trades)

    @property
    def total_trades(self) -> int:
        """Total number of trades executed by this engine."""
        return len(self._trades)

    @property
    def total_volume_traded(self) -> int:
        """Total aggregate quantity executed across all trades."""
        return sum(t.quantity for t in self._trades)

    @property
    def best_bid(self) -> float | int | Decimal | None:
        """Current best bid price in the underlying order book."""
        return self.order_book.best_bid

    @property
    def best_ask(self) -> float | int | Decimal | None:
        """Current best ask price in the underlying order book."""
        return self.order_book.best_ask

    @property
    def spread(self) -> float | int | Decimal | None:
        """Current bid-ask spread in the underlying order book."""
        return self.order_book.spread

    def clear(self) -> None:
        """Reset the matching engine and underlying order book."""
        self.order_book.clear()
        self._trades.clear()
        self._trade_counter = 0

    def __repr__(self) -> str:
        return (
            f"MatchingEngine(trades={len(self._trades)}, "
            f"volume={self.total_volume_traded}, "
            f"book={repr(self.order_book)})"
        )
