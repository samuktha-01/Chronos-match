"""Metrics and Data Layer for the ChronosMatch Terminal Dashboard.

Provides decoupled, high-performance metrics collection, latency sampling, and point-in-time
state snapshotting for the ChronosMatch trading simulation engine.

Key Components:
- `LatencyStats`: Statistical summary of order-processing latency (min, max, mean, p50, p95, p99, p99.9).
- `BoundedLatencyTracker`: Thread-safe, bounded-capacity reservoir for latency measurements.
- `DashboardMetricsSnapshot`: Immutable, point-in-time metrics snapshot distinguishing unavailable
  metrics (None) from actual zero values.
- `DashboardMetricsCollector`: Non-blocking adapter collecting live telemetry from pipelines,
  engines, ring buffers, simulators, or manual event streams.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from decimal import Decimal
import math
import threading
import time
from typing import Any

# ==============================================================================
# Percentile Calculation Utilities
# ==============================================================================


def calculate_percentile(sorted_samples: list[float], percentile: float) -> float:
    """Calculate a percentile value from pre-sorted samples using linear interpolation.

    Args:
        sorted_samples: Non-empty list of sorted numerical values.
        percentile: Float between 0.0 and 100.0 inclusive.

    Returns:
        Interpolated percentile value.

    Raises:
        ValueError: If sorted_samples is empty or percentile is outside [0, 100].
    """
    if not sorted_samples:
        raise ValueError("Cannot calculate percentile from an empty sample list.")
    if percentile < 0.0 or percentile > 100.0:
        raise ValueError(f"Percentile must be between 0.0 and 100.0, got {percentile}")

    n = len(sorted_samples)
    if n == 1:
        return float(sorted_samples[0])

    k = (n - 1) * (percentile / 100.0)
    floor_idx = math.floor(k)
    ceil_idx = math.ceil(k)

    if floor_idx == ceil_idx:
        return float(sorted_samples[int(k)])

    weight_ceil = k - floor_idx
    weight_floor = ceil_idx - k
    return float(sorted_samples[floor_idx] * weight_floor + sorted_samples[ceil_idx] * weight_ceil)


# ==============================================================================
# Latency Models & Bounded Sample Tracker
# ==============================================================================


@dataclass(frozen=True, slots=True)
class LatencyStats:
    """Statistical summary of order-processing latency samples.

    All values are expressed in nanoseconds (with convenience microsecond properties).
    None indicates that insufficient samples exist to compute the statistic.
    """

    sample_count: int
    min_ns: float | None = None
    max_ns: float | None = None
    mean_ns: float | None = None
    p50_ns: float | None = None
    p95_ns: float | None = None
    p99_ns: float | None = None
    p99_9_ns: float | None = None
    latest_ns: float | None = None

    @property
    def min_us(self) -> float | None:
        """Minimum latency in microseconds."""
        return self.min_ns / 1000.0 if self.min_ns is not None else None

    @property
    def max_us(self) -> float | None:
        """Maximum latency in microseconds."""
        return self.max_ns / 1000.0 if self.max_ns is not None else None

    @property
    def mean_us(self) -> float | None:
        """Mean latency in microseconds."""
        return self.mean_ns / 1000.0 if self.mean_ns is not None else None

    @property
    def p50_us(self) -> float | None:
        """Median (50th percentile) latency in microseconds."""
        return self.p50_ns / 1000.0 if self.p50_ns is not None else None

    @property
    def p95_us(self) -> float | None:
        """95th percentile latency in microseconds."""
        return self.p95_ns / 1000.0 if self.p95_ns is not None else None

    @property
    def p99_us(self) -> float | None:
        """99th percentile latency in microseconds."""
        return self.p99_ns / 1000.0 if self.p99_ns is not None else None

    @property
    def p99_9_us(self) -> float | None:
        """99.9th percentile latency in microseconds."""
        return self.p99_9_ns / 1000.0 if self.p99_9_ns is not None else None

    @property
    def latest_us(self) -> float | None:
        """Most recently observed latency in microseconds."""
        return self.latest_ns / 1000.0 if self.latest_ns is not None else None

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to a serializable diagnostic dictionary."""
        return {
            "sample_count": self.sample_count,
            "min_ns": self.min_ns,
            "max_ns": self.max_ns,
            "mean_ns": self.mean_ns,
            "p50_ns": self.p50_ns,
            "p95_ns": self.p95_ns,
            "p99_ns": self.p99_ns,
            "p99_9_ns": self.p99_9_ns,
            "min_us": self.min_us,
            "max_us": self.max_us,
            "mean_us": self.mean_us,
            "p50_us": self.p50_us,
            "p95_us": self.p95_us,
            "p99_us": self.p99_us,
            "p99_9_us": self.p99_9_us,
            "latest_ns": self.latest_ns,
            "latest_us": self.latest_us,
        }


class BoundedLatencyTracker:
    """Thread-safe, bounded-capacity reservoir for latency measurements.

    Ensures that memory consumption remains strictly constant even under millions
    of order events by maintaining a fixed-size circular reservoir.
    """

    def __init__(self, max_samples: int = 10_000) -> None:
        """Initialize the BoundedLatencyTracker.

        Args:
            max_samples: Maximum number of latency samples to retain in memory.

        Raises:
            ValueError: If max_samples <= 0.
        """
        if max_samples <= 0:
            raise ValueError(f"max_samples must be strictly positive (> 0), got {max_samples}")
        self._max_samples = max_samples
        self._samples: collections.deque[float] = collections.deque(maxlen=max_samples)
        self._total_recorded: int = 0
        self._lock = threading.Lock()

    def record_latency(self, latency_ns: float | int) -> None:
        """Record an observed latency sample in nanoseconds.

        Args:
            latency_ns: Positive elapsed latency in nanoseconds.

        Raises:
            ValueError: If latency_ns is negative.
        """
        if latency_ns < 0:
            raise ValueError(f"latency_ns cannot be negative, got {latency_ns}")
        with self._lock:
            self._samples.append(float(latency_ns))
            self._total_recorded += 1

    def record_duration(self, start_ns: int, end_ns: int | None = None) -> float:
        """Calculate and record duration from start_ns to end_ns (defaults to current time).

        Args:
            start_ns: Starting monotonic nanosecond timestamp.
            end_ns: Ending monotonic nanosecond timestamp (None for time.perf_counter_ns()).

        Returns:
            The recorded duration in nanoseconds.
        """
        end = end_ns if end_ns is not None else time.perf_counter_ns()
        duration = max(0.0, float(end - start_ns))
        self.record_latency(duration)
        return duration

    def get_stats(self) -> LatencyStats:
        """Compute latency percentiles and summary statistics from current reservoir.

        Returns:
            Populated LatencyStats instance. If zero samples exist, percentile fields are None.
        """
        with self._lock:
            if not self._samples:
                return LatencyStats(sample_count=0)

            # Defensive copy under lock to prevent mutation during sorting
            sorted_samples = sorted(self._samples)
            latest = self._samples[-1]
            count = len(sorted_samples)
            mean_val = sum(sorted_samples) / count

        return LatencyStats(
            sample_count=count,
            min_ns=sorted_samples[0],
            max_ns=sorted_samples[-1],
            mean_ns=mean_val,
            p50_ns=calculate_percentile(sorted_samples, 50.0),
            p95_ns=calculate_percentile(sorted_samples, 95.0),
            p99_ns=calculate_percentile(sorted_samples, 99.0),
            p99_9_ns=calculate_percentile(sorted_samples, 99.9),
            latest_ns=latest,
        )

    def clear(self) -> None:
        """Clear all retained samples and reset counter."""
        with self._lock:
            self._samples.clear()
            self._total_recorded = 0

    @property
    def max_samples(self) -> int:
        """Maximum sample capacity."""
        return self._max_samples

    @property
    def current_sample_count(self) -> int:
        """Number of samples currently residing in the reservoir."""
        with self._lock:
            return len(self._samples)

    @property
    def total_recorded_count(self) -> int:
        """Total aggregate count of samples recorded over the tracker lifetime."""
        with self._lock:
            return self._total_recorded


# ==============================================================================
# Dashboard Metrics Snapshot
# ==============================================================================


@dataclass(frozen=True, slots=True)
class DashboardMetricsSnapshot:
    """Immutable point-in-time telemetry snapshot of ChronosMatch execution state.

    Design Principles:
    - Distinguishes unavailable metrics (None) from actual zero values (0 or 0.0).
    - Never fabricates metrics or estimates values not directly reported by components.
    - All timestamps are captured via high-resolution monotonic clocks.
    """

    timestamp_ns: int
    wall_timestamp_ns: int
    elapsed_seconds: float | None
    throughput_orders_per_sec: float | None
    window_throughput_orders_per_sec: float | None
    orders_generated: int | None
    orders_published: int | None
    orders_consumed: int | None
    orders_processed: int | None
    orders_cancelled: int | None
    orders_dropped: int | None
    trades_executed: int | None
    volume_executed: int | None
    ring_buffer_capacity: int | None
    ring_buffer_occupied: int | None
    ring_buffer_utilization: float | None
    malformed_messages: int | None
    duplicate_orders: int | None
    processing_errors: int | None
    backpressure_events: int | None
    latency: LatencyStats | None
    best_bid: float | int | Decimal | None
    best_ask: float | int | Decimal | None
    spread: float | int | Decimal | None
    core_latency: LatencyStats | None = None
    queue_dwell_latency: LatencyStats | None = None
    tick_to_trade_latency: LatencyStats | None = None
    order_book_depth_levels: int | None = None
    is_cython_accelerated: bool = False
    source_name: str = "ChronosMatch"

    def to_dict(self) -> dict[str, Any]:
        """Convert snapshot to a serializable diagnostic dictionary."""
        return {
            "timestamp_ns": self.timestamp_ns,
            "wall_timestamp_ns": self.wall_timestamp_ns,
            "elapsed_seconds": self.elapsed_seconds,
            "throughput_orders_per_sec": self.throughput_orders_per_sec,
            "window_throughput_orders_per_sec": self.window_throughput_orders_per_sec,
            "orders_generated": self.orders_generated,
            "orders_published": self.orders_published,
            "orders_consumed": self.orders_consumed,
            "orders_processed": self.orders_processed,
            "orders_cancelled": self.orders_cancelled,
            "orders_dropped": self.orders_dropped,
            "trades_executed": self.trades_executed,
            "volume_executed": self.volume_executed,
            "ring_buffer_capacity": self.ring_buffer_capacity,
            "ring_buffer_occupied": self.ring_buffer_occupied,
            "ring_buffer_utilization": self.ring_buffer_utilization,
            "malformed_messages": self.malformed_messages,
            "duplicate_orders": self.duplicate_orders,
            "processing_errors": self.processing_errors,
            "backpressure_events": self.backpressure_events,
            "latency": self.latency.to_dict() if self.latency is not None else None,
            "core_latency": self.core_latency.to_dict() if self.core_latency is not None else None,
            "queue_dwell_latency": (
                self.queue_dwell_latency.to_dict() if self.queue_dwell_latency is not None else None
            ),
            "tick_to_trade_latency": (
                self.tick_to_trade_latency.to_dict() if self.tick_to_trade_latency is not None else None
            ),
            "best_bid": float(self.best_bid) if self.best_bid is not None else None,
            "best_ask": float(self.best_ask) if self.best_ask is not None else None,
            "spread": float(self.spread) if self.spread is not None else None,
            "order_book_depth_levels": self.order_book_depth_levels,
            "is_cython_accelerated": self.is_cython_accelerated,
            "source_name": self.source_name,
        }


# ==============================================================================
# Dashboard Metrics Collector / Adapter
# ==============================================================================


class DashboardMetricsCollector:
    """Non-blocking telemetry adapter and metrics collector for ChronosMatch.

    Gathers live telemetry from attached ChronosMatch components:
    - `TradingPipeline`: Comprehensive end-to-end pipeline metrics.
    - `MatchingEngine` / `CythonMatchingEngine`: Trade executions, volumes, and BBO prices.
    - `RingBuffer`: Capacity, occupied slots, utilization, and read/write counters.
    - `MarketSimulator`: Generated, published, dropped, and backpressure metrics.
    - `BoundedLatencyTracker`: Microsecond latency distribution sampling.

    Maintains high-resolution monotonic timestamps and calculates both cumulative
    and sliding-window order throughput without blocking engine execution.
    """

    def __init__(
        self,
        *,
        pipeline: Any = None,
        engine: Any = None,
        ring_buffer: Any = None,
        simulator: Any = None,
        latency_tracker: BoundedLatencyTracker | None = None,
        core_latency_tracker: BoundedLatencyTracker | None = None,
        dwell_latency_tracker: BoundedLatencyTracker | None = None,
        tick_to_trade_tracker: BoundedLatencyTracker | None = None,
        source_name: str = "ChronosMatch",
    ) -> None:
        """Initialize DashboardMetricsCollector.

        Args:
            pipeline: Optional TradingPipeline instance to adapt.
            engine: Optional MatchingEngine or CythonMatchingEngine instance to adapt.
            ring_buffer: Optional RingBuffer instance to adapt.
            simulator: Optional MarketSimulator instance to adapt.
            latency_tracker: Optional BoundedLatencyTracker instance (legacy alias for core latency).
            core_latency_tracker: Optional BoundedLatencyTracker for core matching latency (T3 - T2).
            dwell_latency_tracker: Optional BoundedLatencyTracker for ring-buffer queue dwell latency (T2 - T1).
            tick_to_trade_tracker: Optional BoundedLatencyTracker for end-to-end tick-to-trade latency (T3 - T0).
            source_name: Descriptive name for diagnostic reporting.
        """
        self.source_name = source_name
        self._pipeline = pipeline
        self._engine = engine
        self._ring_buffer = ring_buffer
        self._simulator = simulator

        # Bounded reservoirs for distinct latency categories
        self._core_latency_tracker: BoundedLatencyTracker = (
            core_latency_tracker
            if core_latency_tracker is not None
            else (latency_tracker if latency_tracker is not None else BoundedLatencyTracker())
        )
        self._dwell_latency_tracker: BoundedLatencyTracker = (
            dwell_latency_tracker if dwell_latency_tracker is not None else BoundedLatencyTracker()
        )
        self._tick_to_trade_tracker: BoundedLatencyTracker = (
            tick_to_trade_tracker if tick_to_trade_tracker is not None else BoundedLatencyTracker()
        )
        # Backward compatibility alias
        self._latency_tracker: BoundedLatencyTracker = self._core_latency_tracker

        # Monotonic execution timer tracking
        self._start_time_ns: int | None = None
        self._stop_time_ns: int | None = None
        self._lock = threading.RLock()

        # Sliding-window throughput tracking
        self._prev_snapshot_time_ns: int | None = None
        self._prev_processed_orders: int | None = None

        # Manual / standalone counter overrides (None indicates metric is unavailable)
        self._manual_orders_generated: int | None = None
        self._manual_orders_published: int | None = None
        self._manual_orders_consumed: int | None = None
        self._manual_orders_processed: int | None = None
        self._manual_orders_cancelled: int | None = None
        self._manual_orders_dropped: int | None = None
        self._manual_trades_executed: int | None = None
        self._manual_volume_executed: int | None = None
        self._manual_malformed_messages: int | None = None
        self._manual_duplicate_orders: int | None = None
        self._manual_processing_errors: int | None = None
        self._manual_backpressure_events: int | None = None

        if pipeline is not None:
            self.attach_pipeline(pipeline)
        elif engine is not None:
            self.attach_engine(engine)

    # --------------------------------------------------------------------------
    # Component Attachment API
    # --------------------------------------------------------------------------

    def attach_pipeline(self, pipeline: Any) -> None:
        """Attach a TradingPipeline instance as the primary metrics source."""
        with self._lock:
            self._pipeline = pipeline
            if hasattr(pipeline, "engine") and pipeline.engine is not None:
                self._engine = pipeline.engine
            if hasattr(pipeline, "ring_buffer") and pipeline.ring_buffer is not None:
                self._ring_buffer = pipeline.ring_buffer
            if hasattr(pipeline, "simulator") and pipeline.simulator is not None:
                self._simulator = pipeline.simulator
            if hasattr(pipeline, "core_latency_tracker") and pipeline.core_latency_tracker is not None:
                self._core_latency_tracker = pipeline.core_latency_tracker
                self._latency_tracker = self._core_latency_tracker
            if hasattr(pipeline, "dwell_latency_tracker") and pipeline.dwell_latency_tracker is not None:
                self._dwell_latency_tracker = pipeline.dwell_latency_tracker
            if hasattr(pipeline, "tick_to_trade_tracker") and pipeline.tick_to_trade_tracker is not None:
                self._tick_to_trade_tracker = pipeline.tick_to_trade_tracker

    def attach_engine(self, engine: Any) -> None:
        """Attach a MatchingEngine instance to monitor trades and order book."""
        with self._lock:
            self._engine = engine

    def attach_ring_buffer(self, ring_buffer: Any) -> None:
        """Attach a RingBuffer instance to monitor capacity and utilization."""
        with self._lock:
            self._ring_buffer = ring_buffer

    def attach_simulator(self, simulator: Any) -> None:
        """Attach a MarketSimulator instance to monitor generated orders."""
        with self._lock:
            self._simulator = simulator

    # --------------------------------------------------------------------------
    # Timer Controls
    # --------------------------------------------------------------------------

    def start(self) -> None:
        """Start the monotonic runtime measurement clock."""
        with self._lock:
            self._start_time_ns = time.perf_counter_ns()
            self._stop_time_ns = None

    def stop(self) -> None:
        """Stop the monotonic runtime measurement clock."""
        with self._lock:
            self._stop_time_ns = time.perf_counter_ns()

    def reset_timer(self) -> None:
        """Reset execution timer to unstarted state."""
        with self._lock:
            self._start_time_ns = None
            self._stop_time_ns = None
            self._prev_snapshot_time_ns = None
            self._prev_processed_orders = None

    # --------------------------------------------------------------------------
    # Manual Event Recording (Standalone Mode)
    # --------------------------------------------------------------------------

    def record_latency(self, latency_ns: float | int) -> None:
        """Record an order processing latency sample in nanoseconds (alias for core latency)."""
        self._core_latency_tracker.record_latency(latency_ns)

    def record_core_latency(self, latency_ns: float | int) -> None:
        """Record a core matching latency sample (T3 - T2) in nanoseconds."""
        self._core_latency_tracker.record_latency(latency_ns)

    def record_dwell_latency(self, latency_ns: float | int) -> None:
        """Record a ring-buffer queue dwell latency sample (T2 - T1) in nanoseconds."""
        self._dwell_latency_tracker.record_latency(latency_ns)

    def record_tick_to_trade_latency(self, latency_ns: float | int) -> None:
        """Record an end-to-end tick-to-trade latency sample (T3 - T0) in nanoseconds."""
        self._tick_to_trade_tracker.record_latency(latency_ns)

    def record_duration(self, start_ns: int, end_ns: int | None = None) -> float:
        """Calculate and record duration from start_ns to end_ns."""
        return self._core_latency_tracker.record_duration(start_ns, end_ns)

    def record_order_generated(self, count: int = 1) -> None:
        """Increment manual count of generated orders."""
        with self._lock:
            current = self._manual_orders_generated or 0
            self._manual_orders_generated = current + count

    def record_order_published(self, count: int = 1) -> None:
        """Increment manual count of published orders."""
        with self._lock:
            current = self._manual_orders_published or 0
            self._manual_orders_published = current + count

    def record_order_consumed(self, count: int = 1) -> None:
        """Increment manual count of consumed orders."""
        with self._lock:
            current = self._manual_orders_consumed or 0
            self._manual_orders_consumed = current + count

    def record_order_processed(self, count: int = 1) -> None:
        """Increment manual count of processed orders."""
        with self._lock:
            current = self._manual_orders_processed or 0
            self._manual_orders_processed = current + count

    def record_order_cancelled(self, count: int = 1) -> None:
        """Increment manual count of cancelled orders."""
        with self._lock:
            current = self._manual_orders_cancelled or 0
            self._manual_orders_cancelled = current + count

    def record_order_dropped(self, count: int = 1) -> None:
        """Increment manual count of dropped orders."""
        with self._lock:
            current = self._manual_orders_dropped or 0
            self._manual_orders_dropped = current + count

    def record_trade(self, quantity: int = 1) -> None:
        """Increment manual executed trade count and traded volume."""
        with self._lock:
            t_curr = self._manual_trades_executed or 0
            v_curr = self._manual_volume_executed or 0
            self._manual_trades_executed = t_curr + 1
            self._manual_volume_executed = v_curr + quantity

    def record_error(self, count: int = 1) -> None:
        """Increment manual processing error count."""
        with self._lock:
            current = self._manual_processing_errors or 0
            self._manual_processing_errors = current + count

    # --------------------------------------------------------------------------
    # Snapshot Generation
    # --------------------------------------------------------------------------

    def take_snapshot(self) -> DashboardMetricsSnapshot:
        """Capture an atomic, thread-safe point-in-time metrics snapshot.

        Returns:
            Immutable DashboardMetricsSnapshot containing all current telemetry.
        """
        now_perf_ns = time.perf_counter_ns()
        now_wall_ns = time.time_ns()

        with self._lock:
            # 1. Elapsed Runtime
            elapsed_sec = self._compute_elapsed_seconds(now_perf_ns)

            # 2. Order Metrics
            gen_orders = self._resolve_orders_generated()
            pub_orders = self._resolve_orders_published()
            con_orders = self._resolve_orders_consumed()
            proc_orders = self._resolve_orders_processed()
            canc_orders = self._resolve_orders_cancelled()
            drop_orders = self._resolve_orders_dropped()

            # 3. Trade Metrics
            trades_count = self._resolve_trades_executed()
            vol_count = self._resolve_volume_executed()

            # 4. Ring Buffer Telemetry
            rb_cap, rb_occ, rb_util = self._resolve_ring_buffer_telemetry()

            # 5. Error Metrics
            malformed, duplicate, proc_errors, backpressure = self._resolve_error_metrics()

            # 6. Market Depth & Prices
            best_bid, best_ask, spread, depth_levels = self._resolve_market_state()

            # 7. Latency Telemetry (Core, Queue Dwell & Tick-to-Trade)
            core_stats = self._core_latency_tracker.get_stats()
            core_result = core_stats if core_stats.sample_count > 0 else None

            dwell_stats = self._dwell_latency_tracker.get_stats()
            dwell_result = dwell_stats if dwell_stats.sample_count > 0 else None

            ttt_stats = self._tick_to_trade_tracker.get_stats()
            ttt_result = ttt_stats if ttt_stats.sample_count > 0 else None

            # Backward-compatible alias pointing to core matching latency
            latency_result = core_result

            # 8. Throughput Calculations
            throughput = self._compute_cumulative_throughput(proc_orders, elapsed_sec)
            win_throughput = self._compute_window_throughput(proc_orders, now_perf_ns)

            # Update window state
            if proc_orders is not None:
                self._prev_processed_orders = proc_orders
                self._prev_snapshot_time_ns = now_perf_ns
            # 9. Acceleration Status
            is_cython = False
            if self._pipeline is not None and hasattr(self._pipeline, "is_cython_accelerated"):
                is_cython = bool(self._pipeline.is_cython_accelerated)
            elif self._engine is not None and hasattr(self._engine, "is_cython_accelerated"):
                is_cython = bool(self._engine.is_cython_accelerated)
            elif self._engine is not None:
                from chronosmatch.cython_engine import is_cython_compiled, CythonMatchingEngine

                is_cython = is_cython_compiled() and isinstance(self._engine, CythonMatchingEngine)

            return DashboardMetricsSnapshot(
                timestamp_ns=now_perf_ns,
                wall_timestamp_ns=now_wall_ns,
                elapsed_seconds=elapsed_sec,
                throughput_orders_per_sec=throughput,
                window_throughput_orders_per_sec=win_throughput,
                orders_generated=gen_orders,
                orders_published=pub_orders,
                orders_consumed=con_orders,
                orders_processed=proc_orders,
                orders_cancelled=canc_orders,
                orders_dropped=drop_orders,
                trades_executed=trades_count,
                volume_executed=vol_count,
                ring_buffer_capacity=rb_cap,
                ring_buffer_occupied=rb_occ,
                ring_buffer_utilization=rb_util,
                malformed_messages=malformed,
                duplicate_orders=duplicate,
                processing_errors=proc_errors,
                backpressure_events=backpressure,
                latency=latency_result,
                core_latency=core_result,
                queue_dwell_latency=dwell_result,
                tick_to_trade_latency=ttt_result,
                best_bid=best_bid,
                best_ask=best_ask,
                spread=spread,
                order_book_depth_levels=depth_levels,
                is_cython_accelerated=is_cython,
                source_name=self.source_name,
            )

    # --------------------------------------------------------------------------
    # Metric Resolution Helpers (Strictly avoids fabricating unavailable metrics)
    # --------------------------------------------------------------------------

    def _compute_elapsed_seconds(self, now_perf_ns: int) -> float | None:
        # Check attached pipeline first
        if self._pipeline is not None and hasattr(self._pipeline, "metrics"):
            m = self._pipeline.metrics
            if m.start_time_ns is not None:
                return m.elapsed_seconds

        # Check attached simulator
        if self._simulator is not None and hasattr(self._simulator, "metrics"):
            m = self._simulator.metrics
            if m.start_time_ns is not None:
                return m.elapsed_seconds

        # Check internal collector timer
        if self._start_time_ns is not None:
            end_ns = self._stop_time_ns if self._stop_time_ns is not None else now_perf_ns
            return max(0.0, (end_ns - self._start_time_ns) / 1_000_000_000.0)

        return None

    def _compute_cumulative_throughput(
        self, processed: int | None, elapsed: float | None
    ) -> float | None:
        if processed is None or elapsed is None:
            return None
        if elapsed <= 1e-6:
            return 0.0
        return float(processed) / elapsed

    def _compute_window_throughput(
        self, current_processed: int | None, now_ns: int
    ) -> float | None:
        if (
            current_processed is None
            or self._prev_processed_orders is None
            or self._prev_snapshot_time_ns is None
        ):
            return None

        delta_orders = current_processed - self._prev_processed_orders
        delta_seconds = (now_ns - self._prev_snapshot_time_ns) / 1_000_000_000.0

        if delta_seconds <= 1e-6 or delta_orders < 0:
            return 0.0
        return float(delta_orders) / delta_seconds

    def _resolve_orders_generated(self) -> int | None:
        if self._pipeline is not None and hasattr(self._pipeline, "metrics"):
            gen = self._pipeline.metrics.orders_generated
            if gen > 0:
                return gen
            sim = getattr(self._pipeline, "simulator", self._simulator)
            if sim is not None and hasattr(sim, "metrics"):
                sim_gen = getattr(sim.metrics, "orders_generated", 0)
                if sim_gen > 0:
                    return sim_gen
            return gen
        if self._simulator is not None and hasattr(self._simulator, "metrics"):
            return self._simulator.metrics.orders_generated
        return self._manual_orders_generated

    def _resolve_orders_published(self) -> int | None:
        if self._pipeline is not None and hasattr(self._pipeline, "metrics"):
            pub = self._pipeline.metrics.orders_published
            if pub > 0:
                return pub
            if not getattr(self._pipeline, "is_closed", False):
                rb = getattr(self._pipeline, "ring_buffer", None)
                if rb is not None and not getattr(rb, "is_closed", False):
                    try:
                        return rb.head
                    except Exception:
                        pass
            sim = getattr(self._pipeline, "simulator", self._simulator)
            if sim is not None and hasattr(sim, "metrics"):
                sim_pub = getattr(sim.metrics, "orders_published", 0)
                if sim_pub > 0:
                    return sim_pub
            return pub
        if self._simulator is not None and hasattr(self._simulator, "metrics"):
            return self._simulator.metrics.orders_published
        if self._ring_buffer is not None and not getattr(self._ring_buffer, "is_closed", False):
            try:
                return self._ring_buffer.head
            except Exception:
                return None
        return self._manual_orders_published

    def _resolve_orders_consumed(self) -> int | None:
        rb = self._ring_buffer
        if rb is None and self._pipeline is not None:
            if not getattr(self._pipeline, "is_closed", False):
                rb = getattr(self._pipeline, "ring_buffer", None)
        if rb is not None and not getattr(rb, "is_closed", False):
            try:
                return rb.tail
            except Exception:
                pass
        if self._pipeline is not None and hasattr(self._pipeline, "metrics"):
            con = getattr(self._pipeline.metrics, "orders_consumed", None)
            if con is not None and con > 0:
                return con
            return self._pipeline.metrics.orders_processed
        return self._manual_orders_consumed

    def _resolve_orders_processed(self) -> int | None:
        if self._pipeline is not None and hasattr(self._pipeline, "metrics"):
            return self._pipeline.metrics.orders_processed
        return self._manual_orders_processed

    def _resolve_orders_cancelled(self) -> int | None:
        if self._pipeline is not None and hasattr(self._pipeline, "metrics"):
            canc = self._pipeline.metrics.orders_cancelled
            if canc > 0:
                return canc
            sim = getattr(self._pipeline, "simulator", self._simulator)
            if sim is not None and hasattr(sim, "metrics"):
                sim_canc = getattr(sim.metrics, "orders_cancelled", 0)
                if sim_canc > 0:
                    return sim_canc
            return canc
        if self._simulator is not None and hasattr(self._simulator, "metrics"):
            return self._simulator.metrics.orders_cancelled
        return self._manual_orders_cancelled

    def _resolve_orders_dropped(self) -> int | None:
        if self._pipeline is not None and hasattr(self._pipeline, "metrics"):
            drop = self._pipeline.metrics.orders_dropped
            if drop > 0:
                return drop
            sim = getattr(self._pipeline, "simulator", self._simulator)
            if sim is not None and hasattr(sim, "metrics"):
                sim_drop = getattr(sim.metrics, "orders_dropped", 0)
                if sim_drop > 0:
                    return sim_drop
            return drop
        if self._simulator is not None and hasattr(self._simulator, "metrics"):
            return self._simulator.metrics.orders_dropped
        return self._manual_orders_dropped

    def _resolve_trades_executed(self) -> int | None:
        if self._engine is not None and hasattr(self._engine, "total_trades"):
            return self.engine_total_trades
        if self._pipeline is not None and hasattr(self._pipeline, "metrics"):
            return self._pipeline.metrics.trades_executed
        if self._pipeline is not None and hasattr(self._pipeline, "total_trades"):
            return self._pipeline.total_trades
        return self._manual_trades_executed

    def _resolve_volume_executed(self) -> int | None:
        if self._engine is not None and hasattr(self._engine, "total_volume_traded"):
            return self._engine.total_volume_traded
        if self._pipeline is not None and hasattr(self._pipeline, "metrics"):
            return self._pipeline.metrics.volume_executed
        if self._pipeline is not None and hasattr(self._pipeline, "total_volume"):
            return self._pipeline.total_volume
        return self._manual_volume_executed

    @property
    def engine_total_trades(self) -> int | None:
        if self._engine is not None:
            return self._engine.total_trades
        return None

    def _resolve_ring_buffer_telemetry(
        self,
    ) -> tuple[int | None, int | None, float | None]:
        if self._pipeline is not None and getattr(self._pipeline, "is_closed", False):
            return None, None, None

        rb = self._ring_buffer
        if rb is None and self._pipeline is not None:
            rb = getattr(self._pipeline, "ring_buffer", None)

        if rb is None or getattr(rb, "is_closed", False):
            return None, None, None

        try:
            cap = getattr(rb, "capacity", None)
            occ = getattr(rb, "occupied", None)
            util = None
            if cap is not None and cap > 0 and occ is not None:
                util = max(0.0, min(1.0, float(occ) / float(cap)))
            return cap, occ, util
        except Exception:
            return None, None, None

    def _resolve_error_metrics(
        self,
    ) -> tuple[int | None, int | None, int | None, int | None]:
        m = getattr(self._pipeline, "metrics", None)
        if m is not None:
            malformed = getattr(m, "malformed_messages", 0)
            duplicate = getattr(m, "duplicate_orders", 0)
            backpressure = getattr(m, "backpressure_events", 0)
            if backpressure == 0:
                sim = getattr(self._pipeline, "simulator", self._simulator)
                if sim is not None and hasattr(sim, "metrics"):
                    sim_bp = getattr(sim.metrics, "backpressure_events", 0)
                    if sim_bp > 0:
                        backpressure = sim_bp
            proc_errors = malformed + duplicate
            return malformed, duplicate, proc_errors, backpressure

        sim_m = getattr(self._simulator, "metrics", None)
        backpressure = getattr(sim_m, "backpressure_events", None) if sim_m else self._manual_backpressure_events

        malformed = self._manual_malformed_messages
        duplicate = self._manual_duplicate_orders
        proc_errors = self._manual_processing_errors

        if proc_errors is None and (malformed is not None or duplicate is not None):
            proc_errors = (malformed or 0) + (duplicate or 0)

        return malformed, duplicate, proc_errors, backpressure

    def _resolve_market_state(
        self,
    ) -> tuple[
        float | int | Decimal | None,
        float | int | Decimal | None,
        float | int | Decimal | None,
        int | None,
    ]:
        eng = self._engine
        if eng is None and self._pipeline is not None:
            eng = getattr(self._pipeline, "engine", None)

        if eng is None:
            return None, None, None, None

        best_bid = getattr(eng, "best_bid", None)
        best_ask = getattr(eng, "best_ask", None)
        spread = getattr(eng, "spread", None)

        depth_levels = None
        book = getattr(eng, "order_book", None)
        if book is not None:
            bids = getattr(book, "bids", None)
            asks = getattr(book, "asks", None)
            b_levels = len(bids.price_levels) if bids and hasattr(bids, "price_levels") else (len(bids.depth()) if bids and hasattr(bids, "depth") else 0)
            a_levels = len(asks.price_levels) if asks and hasattr(asks, "price_levels") else (len(asks.depth()) if asks and hasattr(asks, "depth") else 0)
            depth_levels = b_levels + a_levels

        return best_bid, best_ask, spread, depth_levels

    @property
    def core_latency_tracker(self) -> BoundedLatencyTracker:
        """The underlying BoundedLatencyTracker for core matching latency (T3 - T2)."""
        return self._core_latency_tracker

    @property
    def dwell_latency_tracker(self) -> BoundedLatencyTracker:
        """The underlying BoundedLatencyTracker for ring-buffer queue dwell latency (T2 - T1)."""
        return self._dwell_latency_tracker

    @property
    def tick_to_trade_tracker(self) -> BoundedLatencyTracker:
        """The underlying BoundedLatencyTracker for end-to-end tick-to-trade latency (T3 - T0)."""
        return self._tick_to_trade_tracker

    @property
    def latency_tracker(self) -> BoundedLatencyTracker:
        """The underlying BoundedLatencyTracker instance (alias for core_latency_tracker)."""
        return self._core_latency_tracker
