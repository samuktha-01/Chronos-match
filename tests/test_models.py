"""Automated tests for the ChronosMatch Order model and validation rules."""

from decimal import Decimal
import math
import time

import pytest

from chronosmatch.models import Order, OrderSide, Side


class TestOrderCreation:
    """Test suite for valid order creation scenarios."""

    def test_valid_buy_order_creation_with_enum(self) -> None:
        """Verify successful creation of a valid BUY order using OrderSide enum."""
        timestamp = 1700000000000000000
        order = Order(
            order_id="ord-001",
            side=OrderSide.BUY,
            price=150.25,
            quantity=100,
            timestamp=timestamp,
            sequence_number=1,
        )

        assert order.order_id == "ord-001"
        assert order.side is OrderSide.BUY
        assert order.side == "BUY"
        assert order.price == 150.25
        assert order.quantity == 100
        assert order.timestamp == timestamp
        assert order.sequence_number == 1
        assert order.is_buy is True
        assert order.is_sell is False

    def test_valid_buy_order_creation_with_string(self) -> None:
        """Verify successful creation of a valid BUY order using string side."""
        order = Order(
            order_id="ord-002",
            side="BUY",
            price=200,
            quantity=50,
        )

        assert order.order_id == "ord-002"
        assert order.side is OrderSide.BUY
        assert order.is_buy is True
        assert order.is_sell is False

    def test_valid_sell_order_creation_with_enum(self) -> None:
        """Verify successful creation of a valid SELL order using OrderSide enum."""
        timestamp = 1700000000000000001
        order = Order(
            order_id="ord-003",
            side=OrderSide.SELL,
            price=155.75,
            quantity=200,
            timestamp=timestamp,
            sequence_number=2,
        )

        assert order.order_id == "ord-003"
        assert order.side is OrderSide.SELL
        assert order.side == "SELL"
        assert order.price == 155.75
        assert order.quantity == 200
        assert order.timestamp == timestamp
        assert order.sequence_number == 2
        assert order.is_buy is False
        assert order.is_sell is True

    def test_valid_sell_order_creation_with_string(self) -> None:
        """Verify successful creation of a valid SELL order using string side."""
        order = Order(
            order_id="ord-004",
            side="SELL",
            price=99.99,
            quantity=10,
        )

        assert order.order_id == "ord-004"
        assert order.side is OrderSide.SELL
        assert order.is_sell is True
        assert order.is_buy is False

    def test_side_alias_compatibility(self) -> None:
        """Verify that Side alias is identical to OrderSide."""
        assert Side is OrderSide
        order = Order(order_id="ord-005", side=Side.BUY, price=10.0, quantity=1)
        assert order.side is OrderSide.BUY

    def test_valid_order_with_decimal_price(self) -> None:
        """Verify that Decimal prices are accepted."""
        price = Decimal("150.25")
        order = Order(
            order_id="ord-006",
            side=OrderSide.BUY,
            price=price,
            quantity=5,
        )
        assert order.price == price

    def test_default_timestamp_is_positive_integer(self) -> None:
        """Verify default timestamp is automatically generated as a current positive integer."""
        t_before = time.time_ns()
        order = Order(order_id="ord-007", side=OrderSide.BUY, price=100.0, quantity=1)
        t_after = time.time_ns()

        assert isinstance(order.timestamp, int)
        assert not isinstance(order.timestamp, bool)
        assert t_before <= order.timestamp <= t_after

    def test_default_sequence_number_is_none(self) -> None:
        """Verify sequence_number defaults to None when omitted."""
        order = Order(order_id="ord-008", side=OrderSide.BUY, price=100.0, quantity=1)
        assert order.sequence_number is None

    def test_order_equality_and_repr(self) -> None:
        """Verify equality between identical orders and informative repr."""
        timestamp = 1700000000000000000
        order1 = Order("ord-009", OrderSide.BUY, 100.0, 10, timestamp=timestamp, sequence_number=0)
        order2 = Order("ord-009", OrderSide.BUY, 100.0, 10, timestamp=timestamp, sequence_number=0)

        assert order1 == order2
        assert "Order" in repr(order1)
        assert "ord-009" in repr(order1)


class TestOrderIdValidation:
    """Test suite for order_id validation."""

    def test_reject_empty_string_order_id(self) -> None:
        """Verify rejection of empty string order ID."""
        with pytest.raises(ValueError, match="order_id cannot be empty or whitespace"):
            Order(order_id="", side=OrderSide.BUY, price=100.0, quantity=10)

    def test_reject_whitespace_only_order_id(self) -> None:
        """Verify rejection of whitespace-only string order ID."""
        with pytest.raises(ValueError, match="order_id cannot be empty or whitespace"):
            Order(order_id="   \t\n  ", side=OrderSide.BUY, price=100.0, quantity=10)

    def test_reject_non_string_order_id(self) -> None:
        """Verify rejection of non-string order ID."""
        with pytest.raises(TypeError, match="order_id must be a string"):
            Order(order_id=None, side=OrderSide.BUY, price=100.0, quantity=10)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="order_id must be a string"):
            Order(order_id=12345, side=OrderSide.BUY, price=100.0, quantity=10)  # type: ignore[arg-type]


class TestSideValidation:
    """Test suite for side validation."""

    @pytest.mark.parametrize("invalid_side", ["HOLD", "CANCEL", "buy", "sell", "Buy", "Sell", "", " "])
    def test_reject_invalid_side_string(self, invalid_side: str) -> None:
        """Verify rejection of invalid side string values, including lowercase forms."""
        with pytest.raises(ValueError, match="Invalid side"):
            Order(order_id="ord-010", side=invalid_side, price=100.0, quantity=10)

    @pytest.mark.parametrize("invalid_side_type", [None, 1, 0, True, False, 1.5, ("BUY",)])
    def test_reject_invalid_side_type(self, invalid_side_type: object) -> None:
        """Verify rejection of non-string, non-OrderSide objects."""
        with pytest.raises(TypeError, match="side must be an OrderSide or str"):
            Order(order_id="ord-011", side=invalid_side_type, price=100.0, quantity=10)  # type: ignore[arg-type]


class TestPriceValidation:
    """Test suite for price validation."""

    @pytest.mark.parametrize("zero_or_negative_price", [0, 0.0, -0.01, -1, -100.5, Decimal("0"), Decimal("-1.0")])
    def test_reject_zero_or_negative_price(self, zero_or_negative_price: float | int | Decimal) -> None:
        """Verify rejection of zero and negative prices."""
        with pytest.raises(ValueError, match="price must be strictly positive"):
            Order(order_id="ord-012", side=OrderSide.BUY, price=zero_or_negative_price, quantity=10)

    @pytest.mark.parametrize("non_finite_price", [float("nan"), float("inf"), float("-inf")])
    def test_reject_non_finite_float_price(self, non_finite_price: float) -> None:
        """Verify rejection of NaN and Infinity float prices."""
        with pytest.raises(ValueError, match="price must be finite"):
            Order(order_id="ord-013", side=OrderSide.BUY, price=non_finite_price, quantity=10)

    @pytest.mark.parametrize("non_finite_decimal", [Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")])
    def test_reject_non_finite_decimal_price(self, non_finite_decimal: Decimal) -> None:
        """Verify rejection of NaN and Infinity Decimal prices."""
        with pytest.raises(ValueError, match="price must be finite"):
            Order(order_id="ord-014", side=OrderSide.BUY, price=non_finite_decimal, quantity=10)

    @pytest.mark.parametrize("invalid_price_type", [True, False, "100.0", None, [100.0]])
    def test_reject_invalid_price_type(self, invalid_price_type: object) -> None:
        """Verify rejection of boolean or non-numeric price types."""
        with pytest.raises(TypeError, match="price must be a real number"):
            Order(order_id="ord-015", side=OrderSide.BUY, price=invalid_price_type, quantity=10)  # type: ignore[arg-type]


class TestQuantityValidation:
    """Test suite for quantity validation."""

    @pytest.mark.parametrize("zero_or_negative_qty", [0, -1, -50, -9999])
    def test_reject_zero_or_negative_quantity(self, zero_or_negative_qty: int) -> None:
        """Verify rejection of zero and negative quantities."""
        with pytest.raises(ValueError, match="quantity must be strictly positive"):
            Order(order_id="ord-016", side=OrderSide.BUY, price=100.0, quantity=zero_or_negative_qty)

    @pytest.mark.parametrize("fractional_qty", [10.5, 0.5, -0.5, 10.0, 1.0])
    def test_reject_fractional_float_quantity(self, fractional_qty: float) -> None:
        """Verify rejection of float quantities even if whole numbers (10.0)."""
        with pytest.raises(TypeError, match="quantity must be an integer"):
            Order(order_id="ord-017", side=OrderSide.BUY, price=100.0, quantity=fractional_qty)  # type: ignore[arg-type]

    @pytest.mark.parametrize("boolean_qty", [True, False])
    def test_reject_boolean_quantity(self, boolean_qty: bool) -> None:
        """Verify rejection of boolean values for quantity."""
        with pytest.raises(TypeError, match="quantity cannot be a boolean"):
            Order(order_id="ord-018", side=OrderSide.BUY, price=100.0, quantity=boolean_qty)  # type: ignore[arg-type]

    @pytest.mark.parametrize("invalid_qty_type", ["10", Decimal("10"), None, [10]])
    def test_reject_other_invalid_quantity_types(self, invalid_qty_type: object) -> None:
        """Verify rejection of string or Decimal quantities without silent coercion."""
        with pytest.raises(TypeError, match="quantity must be an integer"):
            Order(order_id="ord-019", side=OrderSide.BUY, price=100.0, quantity=invalid_qty_type)  # type: ignore[arg-type]


class TestTimestampValidation:
    """Test suite for timestamp validation."""

    def test_valid_positive_integer_timestamp(self) -> None:
        """Verify explicit positive integer timestamp is accepted."""
        order = Order(
            order_id="ord-020",
            side=OrderSide.BUY,
            price=100.0,
            quantity=10,
            timestamp=1600000000000000000,
        )
        assert order.timestamp == 1600000000000000000

    @pytest.mark.parametrize("zero_or_negative_ts", [0, -1, -1000000])
    def test_reject_zero_or_negative_timestamp(self, zero_or_negative_ts: int) -> None:
        """Verify rejection of zero and negative timestamps."""
        with pytest.raises(ValueError, match="timestamp must be strictly positive"):
            Order(
                order_id="ord-021",
                side=OrderSide.BUY,
                price=100.0,
                quantity=10,
                timestamp=zero_or_negative_ts,
            )

    @pytest.mark.parametrize("boolean_ts", [True, False])
    def test_reject_boolean_timestamp(self, boolean_ts: bool) -> None:
        """Verify rejection of boolean values for timestamp."""
        with pytest.raises(TypeError, match="timestamp cannot be a boolean"):
            Order(
                order_id="ord-022",
                side=OrderSide.BUY,
                price=100.0,
                quantity=10,
                timestamp=boolean_ts,  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("invalid_ts_type", [1600000000.5, "2026-10-11T12:00:00", None, [123]])
    def test_reject_non_integer_timestamp_types(self, invalid_ts_type: object) -> None:
        """Verify rejection of non-integer timestamp representations without silent conversion."""
        with pytest.raises(TypeError, match="timestamp must be an integer"):
            Order(
                order_id="ord-023",
                side=OrderSide.BUY,
                price=100.0,
                quantity=10,
                timestamp=invalid_ts_type,  # type: ignore[arg-type]
            )


class TestSequenceNumberValidation:
    """Test suite for optional sequence_number validation."""

    @pytest.mark.parametrize("valid_seq", [None, 0, 1, 42, 1000000])
    def test_accept_valid_sequence_numbers(self, valid_seq: int | None) -> None:
        """Verify acceptance of None, zero, and positive integer sequence numbers."""
        order = Order(
            order_id="ord-024",
            side=OrderSide.BUY,
            price=100.0,
            quantity=10,
            sequence_number=valid_seq,
        )
        assert order.sequence_number == valid_seq

    @pytest.mark.parametrize("negative_seq", [-1, -50, -999])
    def test_reject_negative_sequence_number(self, negative_seq: int) -> None:
        """Verify rejection of negative sequence numbers."""
        with pytest.raises(ValueError, match="sequence_number must be non-negative"):
            Order(
                order_id="ord-025",
                side=OrderSide.BUY,
                price=100.0,
                quantity=10,
                sequence_number=negative_seq,
            )

    @pytest.mark.parametrize("boolean_seq", [True, False])
    def test_reject_boolean_sequence_number(self, boolean_seq: bool) -> None:
        """Verify rejection of boolean values for sequence_number."""
        with pytest.raises(TypeError, match="sequence_number cannot be a boolean"):
            Order(
                order_id="ord-026",
                side=OrderSide.BUY,
                price=100.0,
                quantity=10,
                sequence_number=boolean_seq,  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("invalid_seq_type", [1.0, 1.5, "1", Decimal("1"), [1]])
    def test_reject_non_integer_sequence_number_types(self, invalid_seq_type: object) -> None:
        """Verify rejection of non-integer sequence_number types without silent conversion."""
        with pytest.raises(TypeError, match="sequence_number must be an integer"):
            Order(
                order_id="ord-027",
                side=OrderSide.BUY,
                price=100.0,
                quantity=10,
                sequence_number=invalid_seq_type,  # type: ignore[arg-type]
            )
