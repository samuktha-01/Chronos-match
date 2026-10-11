"""Domain models for ChronosMatch order representation and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
import math
import time


class OrderSide(str, Enum):
    """Enumeration of valid order sides."""

    BUY = "BUY"
    SELL = "SELL"


# Convenience alias for OrderSide
Side = OrderSide


@dataclass(slots=True)
class Order:
    """Represents a financial order in the ChronosMatch simulation engine.

    Attributes:
        order_id: Unique identifier for the order (non-empty string).
        side: Order side, must be OrderSide.BUY ('BUY') or OrderSide.SELL ('SELL').
        price: Positive and finite limit price (int, float, or Decimal).
        quantity: Positive integer quantity (not bool or fractional).
        timestamp: Creation timestamp in nanoseconds (positive integer).
        sequence_number: Optional non-negative sequence number for deterministic ordering.
    """

    order_id: str
    side: OrderSide | str
    price: float | int | Decimal
    quantity: int
    timestamp: int = field(default_factory=time.time_ns)
    sequence_number: int | None = None

    def __post_init__(self) -> None:
        """Validate all order fields to ensure integrity."""
        self._validate_order_id()
        self._validate_side()
        self._validate_price()
        self._validate_quantity()
        self._validate_timestamp()
        self._validate_sequence_number()

    def _validate_order_id(self) -> None:
        if not isinstance(self.order_id, str):
            raise TypeError(f"order_id must be a string, got {type(self.order_id).__name__}")
        if not self.order_id.strip():
            raise ValueError("order_id cannot be empty or whitespace")

    def _validate_side(self) -> None:
        if isinstance(self.side, OrderSide):
            return
        if isinstance(self.side, str):
            # Strict validation: accept only exact 'BUY' or 'SELL', do not silently coerce
            if self.side in (OrderSide.BUY.value, OrderSide.SELL.value):
                self.side = OrderSide(self.side)
                return
            raise ValueError(f"Invalid side '{self.side}'. Must be BUY or SELL.")
        raise TypeError(f"side must be an OrderSide or str ('BUY' or 'SELL'), got {type(self.side).__name__}")

    def _validate_price(self) -> None:
        if isinstance(self.price, bool) or not isinstance(self.price, (int, float, Decimal)):
            raise TypeError(f"price must be a real number (int, float, Decimal), got {type(self.price).__name__}")
        if isinstance(self.price, float) and (math.isnan(self.price) or math.isinf(self.price)):
            raise ValueError(f"price must be finite, got {self.price}")
        if isinstance(self.price, Decimal) and (self.price.is_nan() or self.price.is_infinite()):
            raise ValueError(f"price must be finite, got {self.price}")
        if self.price <= 0:
            raise ValueError(f"price must be strictly positive (> 0), got {self.price}")

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

    def _validate_sequence_number(self) -> None:
        if self.sequence_number is not None:
            if isinstance(self.sequence_number, bool):
                raise TypeError("sequence_number cannot be a boolean")
            if not isinstance(self.sequence_number, int):
                raise TypeError(f"sequence_number must be an integer, got {type(self.sequence_number).__name__}")
            if self.sequence_number < 0:
                raise ValueError(f"sequence_number must be non-negative (>= 0), got {self.sequence_number}")

    @property
    def is_buy(self) -> bool:
        """Return True if this is a BUY order."""
        return self.side == OrderSide.BUY

    @property
    def is_sell(self) -> bool:
        """Return True if this is a SELL order."""
        return self.side == OrderSide.SELL
