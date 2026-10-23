"""Asynchronous Synthetic Market Feed Simulator for ChronosMatch.

Provides a configurable, deterministic, high-throughput market order generator
built on Python `asyncio`. Emits realistic BUY/SELL limit orders with configurable
price levels, spreads, aggressive crossing orders, order cancellations, burst traffic,
and Poisson/exponential inter-arrival timing.

Integrates directly with the `OrderProducer` ring-buffer interface while respecting
backpressure, bounded buffer capacity, and graceful cancellation.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
import math
from pathlib import Path
import random
import time
from typing import Any, Self

from chronosmatch.models import Order, OrderSide
from chronosmatch.producer_consumer import OrderProducer
from chronosmatch.ring_buffer import (
    RingBufferClosedError,
    RingBufferFullError,
)

# ==============================================================================
# Enumerations
# ==============================================================================


class SimulationMode(str, Enum):
    """Timing and traffic distribution mode for the market simulator."""

    POISSON = "POISSON"  # Exponential inter-arrival times (Poisson point process)
    UNIFORM = "UNIFORM"  # Constant deterministic inter-arrival times
    BURST = "BURST"  # Mixed traffic with intermittent high-frequency burst clusters
    MAX_THROUGHPUT = "MAX_THROUGHPUT"  # Zero artificial sleep, yield to event loop only


class BackpressureMode(str, Enum):
    """Strategy for handling downstream buffer full conditions."""

    WAIT = "WAIT"  # Asynchronously poll/wait until buffer space is freed
    DROP = "DROP"  # Immediately drop order and record in dropped metrics
    RAISE = "RAISE"  # Raise RingBufferFullError immediately


class OrderAction(str, Enum):
    """Simulated market feed action."""

    NEW = "NEW"
    CANCEL = "CANCEL"


# ==============================================================================
# Exceptions
# ==============================================================================


class MarketSimulatorError(Exception):
    """Base exception for all market simulator errors."""


class SimulatorConfigError(MarketSimulatorError, ValueError):
    """Raised when market simulator configuration parameters are invalid."""


class BackpressureTimeoutError(MarketSimulatorError, TimeoutError):
    """Raised when backpressure wait exceeds the configured timeout."""


# ==============================================================================
# Data Structures: Cancellation Event & Metrics
# ==============================================================================


@dataclass(slots=True)
class OrderCancel:
    """Represents a simulated cancellation request for an existing resting order.

    Attributes:
        order_id: Unique identifier of the resting order to cancel.
        timestamp: Nanosecond timestamp when the cancellation was synthesized.
    """

    order_id: str
    timestamp: int = field(default_factory=time.time_ns)

    def __post_init__(self) -> None:
        if not isinstance(self.order_id, str) or not self.order_id.strip():
            raise ValueError("order_id cannot be empty or whitespace")


@dataclass
class SimulatorMetrics:
    """Diagnostic and performance metrics collected during simulation execution."""

    orders_generated: int = 0
    orders_published: int = 0
    orders_cancelled: int = 0
    orders_dropped: int = 0
    crossing_orders: int = 0
    passive_orders: int = 0
    buy_orders: int = 0
    sell_orders: int = 0
    backpressure_events: int = 0
    total_backpressure_wait_seconds: float = 0.0
    start_time_ns: int | None = None
    stop_time_ns: int | None = None
    wall_start_time_ns: int | None = None
    wall_stop_time_ns: int | None = None

    @property
    def elapsed_seconds(self) -> float:
        """Total elapsed duration in seconds measured via high-resolution monotonic timer."""
        if self.start_time_ns is None:
            return 0.0
        end_ns = self.stop_time_ns if self.stop_time_ns is not None else time.perf_counter_ns()
        return max(0.0, (end_ns - self.start_time_ns) / 1_000_000_000.0)

    @property
    def effective_rate(self) -> float:
        """Orders successfully published per second."""
        elapsed = self.elapsed_seconds
        if elapsed <= 0.0:
            return 0.0
        return self.orders_published / elapsed

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a diagnostic dictionary."""
        return {
            "orders_generated": self.orders_generated,
            "orders_published": self.orders_published,
            "orders_cancelled": self.orders_cancelled,
            "orders_dropped": self.orders_dropped,
            "crossing_orders": self.crossing_orders,
            "passive_orders": self.passive_orders,
            "buy_orders": self.buy_orders,
            "sell_orders": self.sell_orders,
            "backpressure_events": self.backpressure_events,
            "total_backpressure_wait_seconds": self.total_backpressure_wait_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "effective_rate": self.effective_rate,
            "start_time_ns": self.start_time_ns,
            "stop_time_ns": self.stop_time_ns,
            "wall_start_time_ns": self.wall_start_time_ns,
            "wall_stop_time_ns": self.wall_stop_time_ns,
        }

    def reset(self) -> None:
        """Reset all metrics to initial states."""
        self.orders_generated = 0
        self.orders_published = 0
        self.orders_cancelled = 0
        self.orders_dropped = 0
        self.crossing_orders = 0
        self.passive_orders = 0
        self.buy_orders = 0
        self.sell_orders = 0
        self.backpressure_events = 0
        self.total_backpressure_wait_seconds = 0.0
        self.start_time_ns = None
        self.stop_time_ns = None
        self.wall_start_time_ns = None
        self.wall_stop_time_ns = None


# ==============================================================================
# Configuration: MarketFeedConfig
# ==============================================================================


@dataclass
class MarketFeedConfig:
    """Configuration specification for the synthetic market feed simulator.

    Attributes:
        symbol: Synthetic instrument symbol.
        order_prefix: Prefix for generated order_ids (max 20 characters).
        total_orders: Total orders to generate (None for continuous execution).
        rate: Target order generation rate in orders per second (lambda for Poisson).
        mode: Traffic timing mode (POISSON, UNIFORM, BURST, MAX_THROUGHPUT).
        seed: Deterministic random seed for reproducible runs (None for non-deterministic).
        initial_price: Reference mid-price to begin simulation.
        min_price: Minimum allowed order limit price (> 0).
        max_price: Maximum allowed order limit price.
        price_tick: Minimum price increment (tick size).
        price_volatility: Standard deviation of mid-price drift per step in ticks.
        min_quantity: Minimum order contract quantity (> 0).
        max_quantity: Maximum order contract quantity.
        quantity_step: Lot size increment for order quantity.
        buy_ratio: Probability of BUY order vs SELL order (in [0.0, 1.0]).
        spread_ticks: Target bid/ask spread expressed in price ticks.
        crossing_probability: Probability of generating an aggressive crossing order.
        cancel_probability: Probability of attempting an order cancellation when possible.
        max_resting_orders_tracked: Maximum size of pool tracking resting orders for cancellation.
        burst_probability: Probability of initiating a burst cluster in BURST mode.
        burst_size_min: Minimum orders in a burst cluster.
        burst_size_max: Maximum orders in a burst cluster.
        burst_rate_multiplier: Speedup factor during burst clusters.
        backpressure_mode: Strategy for buffer full conditions (WAIT, DROP, RAISE).
        backpressure_timeout: Maximum seconds to wait on backpressure in WAIT mode (None = indefinite).
        backpressure_poll_interval: Sleep seconds between buffer status polls during backpressure.
    """

    symbol: str = "CHRONOS"
    order_prefix: str = "SIM"
    total_orders: int | None = None
    rate: float = 1000.0
    mode: SimulationMode | str = SimulationMode.POISSON
    seed: int | None = None

    # Price parameters
    initial_price: float = 100.0
    min_price: float = 1.0
    max_price: float = 10000.0
    price_tick: float = 0.01
    price_volatility: float = 0.05

    # Quantity parameters
    min_quantity: int = 1
    max_quantity: int = 100
    quantity_step: int = 1

    # Market dynamics
    buy_ratio: float = 0.5
    spread_ticks: int = 2
    crossing_probability: float = 0.1
    cancel_probability: float = 0.05
    max_resting_orders_tracked: int = 1000

    # Burst dynamics
    burst_probability: float = 0.05
    burst_size_min: int = 5
    burst_size_max: int = 25
    burst_rate_multiplier: float = 10.0

    # Backpressure & Flow Control
    backpressure_mode: BackpressureMode | str = BackpressureMode.WAIT
    backpressure_timeout: float | None = 5.0
    backpressure_poll_interval: float = 0.001

    def __post_init__(self) -> None:
        """Validate all configuration attributes strictly."""
        self._validate_identifiers()
        self._validate_mode()
        self._validate_rate()
        self._validate_seed()
        self._validate_prices()
        self._validate_quantities()
        self._validate_probabilities()
        self._validate_burst()
        self._validate_backpressure()

    def _validate_identifiers(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise SimulatorConfigError("symbol must be a non-empty string")
        if not isinstance(self.order_prefix, str) or not self.order_prefix.strip():
            raise SimulatorConfigError("order_prefix must be a non-empty string")
        if len(self.order_prefix.encode("utf-8")) > 20:
            raise SimulatorConfigError(
                f"order_prefix length exceeds 20 bytes (got {len(self.order_prefix.encode('utf-8'))})"
            )
        if any(c in self.order_prefix for c in "\x00\r\n\t "):
            raise SimulatorConfigError("order_prefix cannot contain control or whitespace characters")

    def _validate_mode(self) -> None:
        if isinstance(self.mode, str):
            try:
                self.mode = SimulationMode(self.mode.upper())
            except ValueError:
                raise SimulatorConfigError(
                    f"Invalid simulation mode '{self.mode}'. "
                    f"Must be one of {[m.value for m in SimulationMode]}"
                )
        elif not isinstance(self.mode, SimulationMode):
            raise SimulatorConfigError(f"mode must be a SimulationMode or str, got {type(self.mode).__name__}")

    def _validate_rate(self) -> None:
        if isinstance(self.rate, bool) or not isinstance(self.rate, (int, float)):
            raise SimulatorConfigError(f"rate must be a number, got {type(self.rate).__name__}")
        if not math.isfinite(self.rate) or self.rate <= 0:
            raise SimulatorConfigError(f"rate must be finite and strictly positive (> 0), got {self.rate}")

        if self.total_orders is not None:
            if isinstance(self.total_orders, bool) or not isinstance(self.total_orders, int):
                raise SimulatorConfigError(
                    f"total_orders must be an integer or None, got {type(self.total_orders).__name__}"
                )
            if self.total_orders <= 0:
                raise SimulatorConfigError(
                    f"total_orders must be strictly positive (> 0), got {self.total_orders}"
                )

    def _validate_seed(self) -> None:
        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise SimulatorConfigError(f"seed must be an integer or None, got {type(self.seed).__name__}")

    def _validate_prices(self) -> None:
        for name, val in [
            ("initial_price", self.initial_price),
            ("min_price", self.min_price),
            ("max_price", self.max_price),
            ("price_tick", self.price_tick),
            ("price_volatility", self.price_volatility),
        ]:
            if isinstance(val, bool) or not isinstance(val, (int, float, Decimal)):
                raise SimulatorConfigError(f"{name} must be a number, got {type(val).__name__}")
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                raise SimulatorConfigError(f"{name} must be finite, got {val}")
            if val <= 0:
                raise SimulatorConfigError(f"{name} must be strictly positive (> 0), got {val}")

        if self.min_price > self.max_price:
            raise SimulatorConfigError(
                f"min_price ({self.min_price}) cannot exceed max_price ({self.max_price})"
            )
        if not (self.min_price <= self.initial_price <= self.max_price):
            raise SimulatorConfigError(
                f"initial_price ({self.initial_price}) must be within [{self.min_price}, {self.max_price}]"
            )
        if self.price_tick > (self.max_price - self.min_price):
            raise SimulatorConfigError("price_tick cannot exceed the entire price range")

    def _validate_quantities(self) -> None:
        for name, val in [
            ("min_quantity", self.min_quantity),
            ("max_quantity", self.max_quantity),
            ("quantity_step", self.quantity_step),
        ]:
            if isinstance(val, bool) or not isinstance(val, int):
                raise SimulatorConfigError(f"{name} must be an integer, got {type(val).__name__}")
            if val <= 0:
                raise SimulatorConfigError(f"{name} must be strictly positive (> 0), got {val}")

        if self.min_quantity > self.max_quantity:
            raise SimulatorConfigError(
                f"min_quantity ({self.min_quantity}) cannot exceed max_quantity ({self.max_quantity})"
            )
        if self.quantity_step > (self.max_quantity - self.min_quantity + 1):
            raise SimulatorConfigError("quantity_step cannot exceed the quantity range")

    def _validate_probabilities(self) -> None:
        for name, val in [
            ("buy_ratio", self.buy_ratio),
            ("crossing_probability", self.crossing_probability),
            ("cancel_probability", self.cancel_probability),
            ("burst_probability", self.burst_probability),
        ]:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise SimulatorConfigError(f"{name} must be a float in [0.0, 1.0], got {type(val).__name__}")
            if not math.isfinite(val) or not (0.0 <= val <= 1.0):
                raise SimulatorConfigError(f"{name} must be in range [0.0, 1.0], got {val}")

        if isinstance(self.spread_ticks, bool) or not isinstance(self.spread_ticks, int):
            raise SimulatorConfigError(f"spread_ticks must be an integer, got {type(self.spread_ticks).__name__}")
        if self.spread_ticks <= 0:
            raise SimulatorConfigError(f"spread_ticks must be strictly positive (>= 1), got {self.spread_ticks}")

        if (
            isinstance(self.max_resting_orders_tracked, bool)
            or not isinstance(self.max_resting_orders_tracked, int)
            or self.max_resting_orders_tracked <= 0
        ):
            raise SimulatorConfigError("max_resting_orders_tracked must be a strictly positive integer")

    def _validate_burst(self) -> None:
        for name, val in [
            ("burst_size_min", self.burst_size_min),
            ("burst_size_max", self.burst_size_max),
        ]:
            if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
                raise SimulatorConfigError(f"{name} must be a strictly positive integer, got {val}")

        if self.burst_size_min > self.burst_size_max:
            raise SimulatorConfigError(
                f"burst_size_min ({self.burst_size_min}) cannot exceed burst_size_max ({self.burst_size_max})"
            )

        if (
            isinstance(self.burst_rate_multiplier, bool)
            or not isinstance(self.burst_rate_multiplier, (int, float))
            or not math.isfinite(self.burst_rate_multiplier)
            or self.burst_rate_multiplier <= 0
        ):
            raise SimulatorConfigError("burst_rate_multiplier must be a finite positive number")

    def _validate_backpressure(self) -> None:
        if isinstance(self.backpressure_mode, str):
            try:
                self.backpressure_mode = BackpressureMode(self.backpressure_mode.upper())
            except ValueError:
                raise SimulatorConfigError(
                    f"Invalid backpressure mode '{self.backpressure_mode}'. "
                    f"Must be one of {[m.value for m in BackpressureMode]}"
                )
        elif not isinstance(self.backpressure_mode, BackpressureMode):
            raise SimulatorConfigError(
                f"backpressure_mode must be BackpressureMode or str, got {type(self.backpressure_mode).__name__}"
            )

        if self.backpressure_timeout is not None:
            if (
                isinstance(self.backpressure_timeout, bool)
                or not isinstance(self.backpressure_timeout, (int, float))
                or not math.isfinite(self.backpressure_timeout)
                or self.backpressure_timeout < 0
            ):
                raise SimulatorConfigError("backpressure_timeout must be a non-negative number or None")

        if (
            isinstance(self.backpressure_poll_interval, bool)
            or not isinstance(self.backpressure_poll_interval, (int, float))
            or not math.isfinite(self.backpressure_poll_interval)
            or self.backpressure_poll_interval <= 0
        ):
            raise SimulatorConfigError("backpressure_poll_interval must be a strictly positive number")


# ==============================================================================
# Helper Utilities
# ==============================================================================


def _round_to_tick(price: float, tick: float) -> float:
    """Quantize price to the nearest tick and round to avoid float representation jitter."""
    ticks = round(price / tick)
    raw = ticks * tick
    # Precision decimal places based on tick size
    decimals = max(0, -int(math.floor(math.log10(tick)))) if tick < 1.0 else 2
    return round(raw, min(8, decimals + 2))


# ==============================================================================
# MarketSimulator Core Implementation
# ==============================================================================


class MarketSimulator:
    """Asynchronous synthetic market order feed generator.

    Generates realistic, validated limit orders and cancellation requests. Supports
    burst traffic, Poisson inter-arrival intervals, configurable spreads, aggressive
    crossing taker orders, and backpressure handling for bounded ring buffers.
    """

    def __init__(self, config: MarketFeedConfig | None = None) -> None:
        """Initialize MarketSimulator with configuration.

        Args:
            config: MarketFeedConfig specification (or default if None).
        """
        self.config: MarketFeedConfig = config if config is not None else MarketFeedConfig()
        self.metrics: SimulatorMetrics = SimulatorMetrics()
        self._rng: random.Random = random.Random(self.config.seed)
        self._sequence_number: int = 0
        self._current_mid_price: float = float(self.config.initial_price)
        self._resting_order_pool: deque[str] = deque(maxlen=self.config.max_resting_orders_tracked)
        self._stop_event: asyncio.Event = asyncio.Event()
        self._is_running: bool = False
        self._burst_remaining: int = 0

    # --------------------------------------------------------------------------
    # Lifecycle & State Control
    # --------------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the simulator RNG, internal state, and metrics."""
        self._rng = random.Random(self.config.seed)
        self._sequence_number = 0
        self._current_mid_price = float(self.config.initial_price)
        self._resting_order_pool.clear()
        self._stop_event.clear()
        self._is_running = False
        self._burst_remaining = 0
        self.metrics.reset()

    def stop(self) -> None:
        """Signal the simulator to halt execution gracefully."""
        self._stop_event.set()

    @property
    def is_running(self) -> bool:
        """True if the simulation loop is currently actively running."""
        return self._is_running

    @property
    def is_stopped(self) -> bool:
        """True if stop has been signaled."""
        return self._stop_event.is_set()

    @property
    def current_mid_price(self) -> float:
        """Current synthetic reference mid-price."""
        return self._current_mid_price

    @property
    def resting_order_count(self) -> int:
        """Number of resting order IDs currently tracked in the cancellation pool."""
        return len(self._resting_order_pool)

    # --------------------------------------------------------------------------
    # Deterministic Order & Timing Generation
    # --------------------------------------------------------------------------

    def _quantize_price(self, price: float) -> float:
        """Quantize and clamp price within [min_price, max_price]."""
        p = _round_to_tick(price, self.config.price_tick)
        return max(self.config.min_price, min(self.config.max_price, p))

    def _update_mid_price(self) -> None:
        """Drift reference mid-price via Gaussian random walk."""
        delta = self._rng.gauss(0.0, self.config.price_volatility * self.config.price_tick)
        self._current_mid_price = self._quantize_price(self._current_mid_price + delta)

    def next_order(self) -> Order:
        """Synthesize and return the next valid Order according to market rules.

        Returns:
            A strictly validated Order instance.
        """
        self._update_mid_price()

        half_spread = max(
            self.config.price_tick,
            (self.config.spread_ticks * self.config.price_tick) / 2.0,
        )
        ref_bid = self._quantize_price(max(self.config.min_price, self._current_mid_price - half_spread))
        ref_ask = self._quantize_price(min(self.config.max_price, self._current_mid_price + half_spread))
        if ref_bid >= ref_ask:
            ref_ask = self._quantize_price(min(self.config.max_price, ref_bid + self.config.price_tick))

        is_buy: bool = self._rng.random() < self.config.buy_ratio
        is_crossing: bool = self._rng.random() < self.config.crossing_probability

        if is_crossing:
            # Aggressive taker order that crosses the spread
            if is_buy:
                # Crosses ask to match resting asks
                tick_offset = self._rng.randint(0, 2) * self.config.price_tick
                price = self._quantize_price(min(self.config.max_price, ref_ask + tick_offset))
            else:
                # Crosses bid to match resting bids
                tick_offset = self._rng.randint(0, 2) * self.config.price_tick
                price = self._quantize_price(max(self.config.min_price, ref_bid - tick_offset))
            self.metrics.crossing_orders += 1
        else:
            # Passive maker order that posts inside or deep in the book
            depth_steps = self._rng.randint(0, 5) * self.config.price_tick
            if is_buy:
                price = self._quantize_price(max(self.config.min_price, ref_bid - depth_steps))
            else:
                price = self._quantize_price(min(self.config.max_price, ref_ask + depth_steps))
            self.metrics.passive_orders += 1

        # Quantize quantity with step size
        steps = (self.config.max_quantity - self.config.min_quantity) // self.config.quantity_step
        step_idx = self._rng.randint(0, steps) if steps > 0 else 0
        quantity = self.config.min_quantity + step_idx * self.config.quantity_step

        self._sequence_number += 1
        order_id = f"{self.config.order_prefix}_{self._sequence_number:010d}"
        side = OrderSide.BUY if is_buy else OrderSide.SELL

        order = Order(
            order_id=order_id,
            side=side,
            price=price,
            quantity=quantity,
            timestamp=time.time_ns(),
            sequence_number=self._sequence_number,
        )

        if not is_crossing:
            self._resting_order_pool.append(order_id)

        self.metrics.orders_generated += 1
        if is_buy:
            self.metrics.buy_orders += 1
        else:
            self.metrics.sell_orders += 1

        return order

    def next_cancel(self) -> OrderCancel | None:
        """Synthesize a cancellation for an existing resting order, if available.

        Returns:
            OrderCancel instance, or None if no resting orders are available.
        """
        if not self._resting_order_pool:
            return None

        # Cancel oldest resting order (FIFO cancellation)
        order_id = self._resting_order_pool.popleft()
        return OrderCancel(order_id=order_id, timestamp=time.time_ns())

    def next_inter_arrival_time(self) -> float:
        """Compute the inter-arrival delay in seconds for the next event.

        Returns:
            Non-negative delay in seconds.
        """
        mode = self.config.mode

        if mode == SimulationMode.MAX_THROUGHPUT:
            return 0.0

        if mode == SimulationMode.UNIFORM:
            return 1.0 / self.config.rate

        if mode == SimulationMode.POISSON:
            return self._rng.expovariate(self.config.rate)

        if mode == SimulationMode.BURST:
            if self._burst_remaining > 0:
                self._burst_remaining -= 1
                base_delay = 1.0 / self.config.rate
                return base_delay / self.config.burst_rate_multiplier

            # Evaluate whether a new burst begins
            if self._rng.random() < self.config.burst_probability:
                self._burst_remaining = self._rng.randint(
                    self.config.burst_size_min, self.config.burst_size_max
                )
                return 0.0  # Burst initiates immediately

            # Inter-burst steady state (Poisson timing)
            return self._rng.expovariate(self.config.rate)

        # Fallback
        return 1.0 / self.config.rate

    # --------------------------------------------------------------------------
    # Ring Buffer & Backpressure Publishing
    # --------------------------------------------------------------------------

    async def publish_order(self, order: Order, producer: OrderProducer) -> bool:
        """Publish an order to an OrderProducer while respecting backpressure.

        Args:
            order: Order instance to publish.
            producer: Target OrderProducer backed by a ring buffer.

        Returns:
            True if published successfully, False if dropped or stopped.

        Raises:
            RingBufferClosedError: If producer is closed.
            RingBufferFullError: If backpressure mode is RAISE and buffer is full.
            BackpressureTimeoutError: If backpressure wait timeout expires.
        """
        if producer.is_closed:
            raise RingBufferClosedError("Cannot publish to closed OrderProducer")

        mode = self.config.backpressure_mode

        # Check for initial backpressure
        if producer.is_full:
            self.metrics.backpressure_events += 1

            if mode == BackpressureMode.DROP:
                self.metrics.orders_dropped += 1
                return False

            if mode == BackpressureMode.RAISE:
                raise RingBufferFullError(
                    f"Downstream ring buffer is full (capacity={producer.capacity})"
                )

            # WAIT mode: asynchronous backpressure poll
            wait_start = time.monotonic()
            timeout = self.config.backpressure_timeout

            while producer.is_full:
                if self._stop_event.is_set():
                    return False

                if timeout is not None and (time.monotonic() - wait_start) >= timeout:
                    raise BackpressureTimeoutError(
                        f"Backpressure timeout of {timeout}s exceeded waiting for ring buffer space"
                    )

                await asyncio.sleep(self.config.backpressure_poll_interval)

                if producer.is_closed:
                    raise RingBufferClosedError("OrderProducer closed during backpressure wait")

            self.metrics.total_backpressure_wait_seconds += time.monotonic() - wait_start

        # Attempt publication (retry if transient race condition occurs)
        while True:
            if self._stop_event.is_set():
                return False

            try:
                producer.publish_nowait(order)
                self.metrics.orders_published += 1
                return True
            except RingBufferFullError:
                self.metrics.backpressure_events += 1

                if mode == BackpressureMode.DROP:
                    self.metrics.orders_dropped += 1
                    return False

                if mode == BackpressureMode.RAISE:
                    raise

                # Re-enter async wait
                wait_start = time.monotonic()
                timeout = self.config.backpressure_timeout
                while producer.is_full:
                    if self._stop_event.is_set():
                        return False
                    if timeout is not None and (time.monotonic() - wait_start) >= timeout:
                        raise BackpressureTimeoutError(
                            f"Backpressure timeout of {timeout}s exceeded waiting for ring buffer space"
                        )
                    await asyncio.sleep(self.config.backpressure_poll_interval)
                    if producer.is_closed:
                        raise RingBufferClosedError("OrderProducer closed during backpressure wait")
                self.metrics.total_backpressure_wait_seconds += time.monotonic() - wait_start

    # --------------------------------------------------------------------------
    # Async Generators
    # --------------------------------------------------------------------------

    async def generate_orders(self) -> AsyncIterator[Order]:
        """Asynchronously yield simulated Order instances with calibrated timing delays.

        Yields:
            Valid Order instances.
        """
        count = 0
        while not self._stop_event.is_set():
            if self.config.total_orders is not None and count >= self.config.total_orders:
                break

            order = self.next_order()
            yield order
            count += 1

            delay = self.next_inter_arrival_time()
            if delay > 0.0:
                await asyncio.sleep(delay)
            else:
                await asyncio.sleep(0)  # Yield to event loop

    async def generate_events(self) -> AsyncIterator[Order | OrderCancel]:
        """Asynchronously yield both Order and OrderCancel events with calibrated delays.

        Yields:
            Order or OrderCancel instances.
        """
        count = 0
        while not self._stop_event.is_set():
            if self.config.total_orders is not None and count >= self.config.total_orders:
                break

            # Check if this iteration should be a cancellation
            if (
                self.config.cancel_probability > 0.0
                and self._resting_order_pool
                and self._rng.random() < self.config.cancel_probability
            ):
                cancel_event = self.next_cancel()
                if cancel_event is not None:
                    self.metrics.orders_cancelled += 1
                    yield cancel_event
                    delay = self.next_inter_arrival_time()
                    if delay > 0.0:
                        await asyncio.sleep(delay)
                    else:
                        await asyncio.sleep(0)
                    continue

            # Standard order generation
            order = self.next_order()
            yield order
            count += 1

            delay = self.next_inter_arrival_time()
            if delay > 0.0:
                await asyncio.sleep(delay)
            else:
                await asyncio.sleep(0)

    # --------------------------------------------------------------------------
    # Execution Runner
    # --------------------------------------------------------------------------

    async def run(
        self,
        producer: OrderProducer | None = None,
        *,
        target_engine: Any = None,
        cancel_callback: Callable[[str], Any] | None = None,
        order_callback: Callable[[Order], Any] | None = None,
        publish_callback: Callable[[Order], Any] | None = None,
    ) -> SimulatorMetrics:
        """Run the market simulation loop until total_orders reached or stopped.

        Args:
            producer: Optional OrderProducer to publish orders to.
            target_engine: Optional MatchingEngine / LimitOrderBook to match orders
                and apply cancellations against.
            cancel_callback: Optional callable invoked with order_id on cancellation.
            order_callback: Optional callable invoked with Order immediately upon generation.
            publish_callback: Optional callable invoked with Order immediately upon successful publication.

        Returns:
            Populated SimulatorMetrics.
        """
        self._is_running = True
        self.metrics.wall_start_time_ns = time.time_ns()
        self.metrics.start_time_ns = time.perf_counter_ns()
        count = 0

        try:
            while not self._stop_event.is_set():
                if self.config.total_orders is not None and count >= self.config.total_orders:
                    break

                # Check if cancellation should occur
                if (
                    self.config.cancel_probability > 0.0
                    and self._resting_order_pool
                    and self._rng.random() < self.config.cancel_probability
                ):
                    cancel_event = self.next_cancel()
                    if cancel_event is not None:
                        self.metrics.orders_cancelled += 1
                        if target_engine is not None and hasattr(target_engine, "cancel_order"):
                            try:
                                target_engine.cancel_order(cancel_event.order_id)
                            except Exception:
                                pass
                        if cancel_callback is not None:
                            cancel_callback(cancel_event.order_id)

                        delay = self.next_inter_arrival_time()
                        if delay > 0.0:
                            await asyncio.sleep(delay)
                        else:
                            await asyncio.sleep(0)
                        continue

                # Generate new order
                order = self.next_order()
                count += 1
                if order_callback is not None:
                    try:
                        order_callback(order)
                    except Exception:
                        pass

                if producer is not None:
                    published = await self.publish_order(order, producer)
                    if published and publish_callback is not None:
                        try:
                            publish_callback(order)
                        except Exception:
                            pass

                if target_engine is not None:
                    if hasattr(target_engine, "process_order"):
                        target_engine.process_order(order)
                    elif hasattr(target_engine, "add_order"):
                        target_engine.add_order(order)

                delay = self.next_inter_arrival_time()
                if delay > 0.0:
                    await asyncio.sleep(delay)
                else:
                    await asyncio.sleep(0)

        except asyncio.CancelledError:
            self._stop_event.set()
            raise
        finally:
            self._is_running = False
            self.metrics.wall_stop_time_ns = time.time_ns()
            stop_ns = time.perf_counter_ns()
            if self.metrics.start_time_ns is not None and stop_ns <= self.metrics.start_time_ns:
                stop_ns = self.metrics.start_time_ns + 1
            self.metrics.stop_time_ns = stop_ns

        return self.metrics

    # --------------------------------------------------------------------------
    # Context Management
    # --------------------------------------------------------------------------

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


# ==============================================================================
# Helper Factory & Synchronous Bridge
# ==============================================================================


def create_simulator(
    rate: float = 1000.0,
    total_orders: int | None = None,
    mode: SimulationMode | str = SimulationMode.POISSON,
    seed: int | None = None,
    **kwargs: Any,
) -> MarketSimulator:
    """Convenience factory function to create a configured MarketSimulator."""
    config = MarketFeedConfig(
        rate=rate,
        total_orders=total_orders,
        mode=mode,
        seed=seed,
        **kwargs,
    )
    return MarketSimulator(config)


def run_simulator_sync(
    config: MarketFeedConfig,
    producer: OrderProducer | None = None,
    *,
    target_engine: Any = None,
    cancel_callback: Callable[[str], Any] | None = None,
) -> SimulatorMetrics:
    """Synchronous runner bridge to execute the simulator within a new asyncio event loop.

    Useful for executing inside multiprocessing workers (e.g., ProducerProcess).
    """
    sim = MarketSimulator(config)
    return asyncio.run(
        sim.run(producer=producer, target_engine=target_engine, cancel_callback=cancel_callback)
    )
