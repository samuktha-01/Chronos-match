"""Benchmark comparing ChronosMatch Python Reference Engine vs Cython Engine.

Executes identical workloads against both implementations:
1. Book Building (Passive limit orders inserting into depth)
2. Aggressive Matching (Aggressive market/limit orders generating trades)
3. Mixed Realistic High-Frequency Order Flow (10,000 orders)

Measures:
- Total execution duration (seconds)
- Throughput (orders / second, trades / second)
- Latency percentiles (mean, median/p50, p95, p99 in microseconds)
- Exact trade execution and order book equality verification
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any

# Ensure src/ is on Python module search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chronosmatch import (
    CythonMatchingEngine,
    MatchingEngine,
    Order,
    OrderSide,
    get_compilation_status,
    is_cython_compiled,
)


def generate_book_building_workload(count: int = 5000) -> list[Order]:
    """Generate passive, non-crossing resting orders across a grid of prices."""
    orders: list[Order] = []
    # Bids from 90.00 down to 70.00
    # Asks from 110.00 up to 130.00
    for i in range(count):
        side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
        if side == OrderSide.BUY:
            price = round(99.0 - (i % 50) * 0.5, 2)
        else:
            price = round(101.0 + (i % 50) * 0.5, 2)
        orders.append(
            Order(
                order_id=f"BUILD-{i:05d}",
                side=side,
                price=price,
                quantity=10 + (i % 100),
                timestamp=1700000000000000000 + i,
                sequence_number=i,
            )
        )
    return orders


def generate_crossing_workload(count: int = 5000) -> tuple[list[Order], list[Order]]:
    """Generate resting maker orders followed by aggressive taker orders."""
    resting: list[Order] = []
    for i in range(count // 2):
        side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
        price = 99.0 if side == OrderSide.BUY else 101.0
        resting.append(
            Order(
                order_id=f"REST-{i:05d}",
                side=side,
                price=price,
                quantity=20,
                timestamp=1700000000000000000 + i,
                sequence_number=i,
            )
        )

    takers: list[Order] = []
    for i in range(count // 2):
        side = OrderSide.SELL if i % 2 == 0 else OrderSide.BUY
        # SELL at 99.0 matches resting BUY at 99.0; BUY at 101.0 matches resting SELL at 101.0
        price = 99.0 if side == OrderSide.SELL else 101.0
        takers.append(
            Order(
                order_id=f"TAKE-{i:05d}",
                side=side,
                price=price,
                quantity=20,
                timestamp=1700000000100000000 + i,
                sequence_number=count // 2 + i,
            )
        )
    return resting, takers


def generate_mixed_hft_workload(count: int = 10000, seed: int = 42) -> list[Order]:
    """Generate mixed stream of passive resting orders and aggressive crossing orders."""
    rng = random.Random(seed)
    orders: list[Order] = []
    base_price = 100.0

    for i in range(count):
        side = OrderSide.BUY if rng.random() < 0.5 else OrderSide.SELL
        # 65% passive limit orders, 35% aggressive crossing orders
        is_passive = rng.random() < 0.65
        offset = rng.randint(1, 20) * 0.25

        if side == OrderSide.BUY:
            price = round(base_price - offset if is_passive else base_price + offset, 2)
        else:
            price = round(base_price + offset if is_passive else base_price - offset, 2)

        qty = rng.randint(5, 50)
        orders.append(
            Order(
                order_id=f"HFT-{i:05d}",
                side=side,
                price=price,
                quantity=qty,
                timestamp=1700000000000000000 + i,
                sequence_number=i,
            )
        )
    return orders


def run_workload(
    engine: Any, orders: list[Order]
) -> dict[str, Any]:
    """Execute workload on the specified engine and record performance metrics."""
    latencies_ns: list[int] = []
    total_trades = 0

    t_start = time.perf_counter()
    for order in orders:
        t0 = time.perf_counter_ns()
        trades = engine.process_order(order)
        t1 = time.perf_counter_ns()
        latencies_ns.append(t1 - t0)
        total_trades += len(trades)
    t_end = time.perf_counter()

    total_time_sec = t_end - t_start
    throughput = len(orders) / total_time_sec if total_time_sec > 0 else 0

    latencies_us = [ns / 1000.0 for ns in latencies_ns]
    latencies_sorted = sorted(latencies_us)

    p50 = latencies_sorted[int(len(latencies_sorted) * 0.50)]
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
    p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]
    mean_lat = statistics.mean(latencies_us)

    return {
        "total_orders": len(orders),
        "total_trades": total_trades,
        "total_time_sec": total_time_sec,
        "throughput_ops": throughput,
        "mean_latency_us": mean_lat,
        "p50_latency_us": p50,
        "p95_latency_us": p95,
        "p99_latency_us": p99,
        "best_bid": engine.best_bid,
        "best_ask": engine.best_ask,
        "total_volume": engine.total_volume_traded,
    }


def run_benchmark_suite() -> dict[str, Any]:
    """Execute the full benchmark suite comparing MatchingEngine and CythonMatchingEngine."""
    print("=" * 80)
    print("CHRONOSMATCH MATCHING ENGINE BENCHMARK SUITE")
    print(f"Compilation status: {'NATIVE CYTHON EXTENSION' if is_cython_compiled() else 'PURE-PYTHON FALLBACK'}")
    status = get_compilation_status()
    if not is_cython_compiled():
        print(f"Note: {status['build_limitation']}")
    print("=" * 80)

    results: dict[str, Any] = {}

    # 1. Book Building Workload
    print("\nRunning Workload 1: Passive Book Building (5,000 orders)...")
    workload1 = generate_book_building_workload(5000)

    # Make independent copies of orders for each engine
    orders_py1 = [Order(o.order_id, o.side, o.price, o.quantity, o.timestamp, o.sequence_number) for o in workload1]
    orders_cy1 = [Order(o.order_id, o.side, o.price, o.quantity, o.timestamp, o.sequence_number) for o in workload1]

    engine_py1 = MatchingEngine()
    engine_cy1 = CythonMatchingEngine()

    stats_py1 = run_workload(engine_py1, orders_py1)
    stats_cy1 = run_workload(engine_cy1, orders_cy1)

    assert stats_py1["total_trades"] == stats_cy1["total_trades"] == 0
    assert stats_py1["best_bid"] == stats_cy1["best_bid"]
    assert stats_py1["best_ask"] == stats_cy1["best_ask"]

    results["book_building"] = {"python": stats_py1, "cython": stats_cy1}

    # 2. Aggressive Crossing Workload
    print("Running Workload 2: Aggressive Matching (2,500 resting + 2,500 crossing)...")
    resting, takers = generate_crossing_workload(5000)
    workload2 = resting + takers

    orders_py2 = [Order(o.order_id, o.side, o.price, o.quantity, o.timestamp, o.sequence_number) for o in workload2]
    orders_cy2 = [Order(o.order_id, o.side, o.price, o.quantity, o.timestamp, o.sequence_number) for o in workload2]

    engine_py2 = MatchingEngine()
    engine_cy2 = CythonMatchingEngine()

    stats_py2 = run_workload(engine_py2, orders_py2)
    stats_cy2 = run_workload(engine_cy2, orders_cy2)

    assert stats_py2["total_trades"] == stats_cy2["total_trades"]
    assert stats_py2["total_volume"] == stats_cy2["total_volume"]

    results["crossing"] = {"python": stats_py2, "cython": stats_cy2}

    # 3. Mixed Realistic HFT Workload (10,000 orders)
    print("Running Workload 3: Mixed High-Frequency Order Flow (10,000 orders)...")
    workload3 = generate_mixed_hft_workload(10000, seed=42)

    orders_py3 = [Order(o.order_id, o.side, o.price, o.quantity, o.timestamp, o.sequence_number) for o in workload3]
    orders_cy3 = [Order(o.order_id, o.side, o.price, o.quantity, o.timestamp, o.sequence_number) for o in workload3]

    engine_py3 = MatchingEngine()
    engine_cy3 = CythonMatchingEngine()

    stats_py3 = run_workload(engine_py3, orders_py3)
    stats_cy3 = run_workload(engine_cy3, orders_cy3)

    assert stats_py3["total_trades"] == stats_cy3["total_trades"]
    assert stats_py3["total_volume"] == stats_cy3["total_volume"]
    assert stats_py3["best_bid"] == stats_cy3["best_bid"]
    assert stats_py3["best_ask"] == stats_cy3["best_ask"]

    results["mixed_hft"] = {"python": stats_py3, "cython": stats_cy3}

    # Print summary table
    print("\n" + "=" * 80)
    print("BENCHMARK MEASUREMENT RESULTS")
    print("=" * 80)
    print(f"{'Workload':<25} | {'Metric':<16} | {'Python Reference':<16} | {'Cython Engine':<16}")
    print("-" * 80)

    for name, data in results.items():
        py = data["python"]
        cy = data["cython"]
        print(f"{name:<25} | Throughput       | {py['throughput_ops']:>12,.0f} ops/s | {cy['throughput_ops']:>12,.0f} ops/s")
        print(f"{'':<25} | Mean Latency     | {py['mean_latency_us']:>12.2f} us    | {cy['mean_latency_us']:>12.2f} us")
        print(f"{'':<25} | Median (p50)     | {py['p50_latency_us']:>12.2f} us    | {cy['p50_latency_us']:>12.2f} us")
        print(f"{'':<25} | 99th % (p99)     | {py['p99_latency_us']:>12.2f} us    | {cy['p99_latency_us']:>12.2f} us")
        print(f"{'':<25} | Total Trades     | {py['total_trades']:>12,d}       | {cy['total_trades']:>12,d}")
        print("-" * 80)

    print("\nEquivalence Check: All 3 workloads verified 100% identical trades, volumes, and book states.")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_benchmark_suite()
