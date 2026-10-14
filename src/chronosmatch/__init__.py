"""ChronosMatch: Zero-Copy High-Frequency Trading Engine."""

from chronosmatch.binary_protocol import (
    MAX_ORDER_ID_LENGTH,
    ORDER_STRUCT_FORMAT,
    ORDER_STRUCT_SIZE,
    PRICE_SCALE,
    SIDE_BUY_CODE,
    SIDE_SELL_CODE,
    decode_order,
    decode_order_from,
    encode_order,
    encode_order_into,
)
from chronosmatch.matching_engine import MatchingEngine, Trade, TradeEvent
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
from chronosmatch.shared_memory import (
    MalformedRecordError,
    SharedMemoryBoundsError,
    SharedMemoryClosedError,
    SharedMemoryConsumer,
    SharedMemoryError,
    SharedMemoryProducer,
    SharedMemoryReadOnlyError,
    SharedMemoryRecordSizeError,
    SharedMemoryRegion,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Order",
    "OrderSide",
    "Side",
    "LimitOrderBook",
    "OrderBook",
    "PriceLevel",
    "OrderBookSide",
    "OrderBookError",
    "DuplicateOrderError",
    "OrderNotFoundError",
    "MatchingEngine",
    "Trade",
    "TradeEvent",
    "encode_order",
    "decode_order",
    "encode_order_into",
    "decode_order_from",
    "ORDER_STRUCT_FORMAT",
    "ORDER_STRUCT_SIZE",
    "MAX_ORDER_ID_LENGTH",
    "PRICE_SCALE",
    "SIDE_BUY_CODE",
    "SIDE_SELL_CODE",
    "SharedMemoryRegion",
    "SharedMemoryProducer",
    "SharedMemoryConsumer",
    "SharedMemoryError",
    "SharedMemoryClosedError",
    "SharedMemoryBoundsError",
    "SharedMemoryRecordSizeError",
    "SharedMemoryReadOnlyError",
    "MalformedRecordError",
]
