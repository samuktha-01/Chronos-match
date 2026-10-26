"""Benchmark suite for ChronosMatch Binary Protocol and Ring Buffer Serialization.

Measures:
1. Order Encoding throughput, latency, and memory allocations (encode_order vs encode_order_into).
2. Order Decoding throughput, latency, and memory allocations (decode_order vs decode_order_from).
3. RingBuffer throughput, latency, and memory allocations (put and get).
4. End-to-end TradingPipeline throughput and latency.
"""

from __future__ import annotations

from decimal import Decimal
import mmap
from pathlib import Path
import statistics
import sys
import time
import tracemalloc
from typing import Any

# Ensure src/ is on Python search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chronosmatch.binary_protocol import (
    ORDER_STRUCT_SIZE,
    decode_order,
    decode_order_from,
    encode_order,
    encode_order_into,
)
from chronosmatch.models import Order, OrderSide
from chronosmatch.pipeline import create_pipeline
from chronosmatch.ring_buffer import RingBuffer


def generate_sample_orders(count: int = 10_000) -> list[Order]:
    """Generate representative orders with varied field types."""
    orders: list[Order] = []
    base_ts = 1700000000000000000
    for i in range(count):
        side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
        # Alternate price types: float, int, Decimal
        type_choice = i % 3
        if type_choice == 0:
            price: float | int | Decimal = 100.25 + (i % 50) * 0.1
        elif type_choice == 1:
            price = 100 + (i % 50)
        else:
            price = Decimal(f"{150 + (i % 50)}.12345678")

        seq: int | None = (i + 1) if (i % 5 != 0) else None
        orders.append(
            Order(
                order_id=f"ORD-{i:06d}",
                side=side,
                price=price,
                quantity=10 + (i % 100),
                timestamp=base_ts + i * 1000,
                sequence_number=seq,
            )
        )
    return orders


def measure_latencies(func: Any, iterations: int = 1) -> tuple[float, list[float]]:
    """Measure total wall-clock duration and per-call latencies in microseconds."""
    latencies_us: list[float] = []
    t_start = time.perf_counter()
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        func()
        t1 = time.perf_counter_ns()
        latencies_us.append((t1 - t0) / 1000.0)
    t_end = time.perf_counter()
    return t_end - t_start, latencies_us


def compute_stats(latencies_us: list[float]) -> dict[str, float]:
    """Calculate mean, median (p50), p95, p99 latency in microseconds."""
    sorted_lats = sorted(latencies_us)
    n = len(sorted_lats)
    return {
        "mean": statistics.mean(sorted_lats),
        "p50": sorted_lats[int(n * 0.50)],
        "p95": sorted_lats[int(n * 0.95)],
        "p99": sorted_lats[min(int(n * 0.99), n - 1)],
    }


def benchmark_encoding(orders: list[Order], runs: int = 3) -> dict[str, Any]:
    """Benchmark encode_order vs encode_order_into."""
    count = len(orders)

    # 1. encode_order (creates bytes)
    enc_throughputs = []
    enc_stats_list = []
    enc_alloc_bytes = 0

    for run in range(runs):
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        t_start = time.perf_counter()
        latencies = []
        for order in orders:
            t0 = time.perf_counter_ns()
            _ = encode_order(order)
            t1 = time.perf_counter_ns()
            latencies.append((t1 - t0) / 1000.0)
        t_end = time.perf_counter()

        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        if run == 0:
            stats = snapshot_after.compare_to(snapshot_before, "lineno")
            enc_alloc_bytes = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

        duration = t_end - t_start
        enc_throughputs.append(count / duration)
        enc_stats_list.append(compute_stats(latencies))

    # 2. encode_order_into (into bytearray buffer)
    buf = bytearray(count * ORDER_STRUCT_SIZE)
    into_throughputs = []
    into_stats_list = []
    into_alloc_bytes = 0

    for run in range(runs):
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        t_start = time.perf_counter()
        latencies = []
        offset = 0
        for order in orders:
            t0 = time.perf_counter_ns()
            encode_order_into(order, buf, offset=offset)
            t1 = time.perf_counter_ns()
            latencies.append((t1 - t0) / 1000.0)
            offset += ORDER_STRUCT_SIZE
        t_end = time.perf_counter()

        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        if run == 0:
            stats = snapshot_after.compare_to(snapshot_before, "lineno")
            into_alloc_bytes = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

        duration = t_end - t_start
        into_throughputs.append(count / duration)
        into_stats_list.append(compute_stats(latencies))

    return {
        "encode_order": {
            "throughput": statistics.mean(enc_throughputs),
            "stats": enc_stats_list[-1],
            "alloc_bytes": enc_alloc_bytes,
        },
        "encode_order_into": {
            "throughput": statistics.mean(into_throughputs),
            "stats": into_stats_list[-1],
            "alloc_bytes": into_alloc_bytes,
        },
    }


def benchmark_decoding(orders: list[Order], runs: int = 3) -> dict[str, Any]:
    """Benchmark decode_order vs decode_order_from."""
    count = len(orders)
    encoded_list = [encode_order(o) for o in orders]
    single_buffer = bytearray(b"".join(encoded_list))

    # 1. decode_order from individual bytes
    dec_throughputs = []
    dec_stats_list = []
    dec_alloc_bytes = 0

    for run in range(runs):
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        t_start = time.perf_counter()
        latencies = []
        for raw in encoded_list:
            t0 = time.perf_counter_ns()
            _ = decode_order(raw)
            t1 = time.perf_counter_ns()
            latencies.append((t1 - t0) / 1000.0)
        t_end = time.perf_counter()

        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        if run == 0:
            stats = snapshot_after.compare_to(snapshot_before, "lineno")
            dec_alloc_bytes = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

        duration = t_end - t_start
        dec_throughputs.append(count / duration)
        dec_stats_list.append(compute_stats(latencies))

    # 2. decode_order_from from continuous buffer
    from_throughputs = []
    from_stats_list = []
    from_alloc_bytes = 0

    for run in range(runs):
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        t_start = time.perf_counter()
        latencies = []
        offset = 0
        for _ in range(count):
            t0 = time.perf_counter_ns()
            _ = decode_order_from(single_buffer, offset=offset)
            t1 = time.perf_counter_ns()
            latencies.append((t1 - t0) / 1000.0)
            offset += ORDER_STRUCT_SIZE
        t_end = time.perf_counter()

        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        if run == 0:
            stats = snapshot_after.compare_to(snapshot_before, "lineno")
            from_alloc_bytes = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

        duration = t_end - t_start
        from_throughputs.append(count / duration)
        from_stats_list.append(compute_stats(latencies))

    return {
        "decode_order": {
            "throughput": statistics.mean(dec_throughputs),
            "stats": dec_stats_list[-1],
            "alloc_bytes": dec_alloc_bytes,
        },
        "decode_order_from": {
            "throughput": statistics.mean(from_throughputs),
            "stats": from_stats_list[-1],
            "alloc_bytes": from_alloc_bytes,
        },
    }


def benchmark_ring_buffer(orders: list[Order], runs: int = 3) -> dict[str, Any]:
    """Benchmark RingBuffer put and get operations individually and together."""
    batch_size = 2000
    batch = orders[:batch_size]

    put_latencies = []
    get_latencies = []
    put_throughputs = []
    get_throughputs = []
    alloc_bytes = 0

    for run in range(runs):
        rb = RingBuffer(capacity=batch_size)

        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        # Measure Batch Put
        t0_put = time.perf_counter()
        for o in batch:
            p0 = time.perf_counter_ns()
            rb.put(o)
            p1 = time.perf_counter_ns()
            put_latencies.append((p1 - p0) / 1000.0)
        t1_put = time.perf_counter()
        put_throughputs.append(batch_size / (t1_put - t0_put))

        # Measure Batch Get
        t0_get = time.perf_counter()
        for _ in range(batch_size):
            g0 = time.perf_counter_ns()
            _ = rb.get()
            g1 = time.perf_counter_ns()
            get_latencies.append((g1 - g0) / 1000.0)
        t1_get = time.perf_counter()
        get_throughputs.append(batch_size / (t1_get - t0_get))

        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        if run == 0:
            stats = snapshot_after.compare_to(snapshot_before, "lineno")
            alloc_bytes = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

        rb.close()

    return {
        "put": {
            "throughput": statistics.mean(put_throughputs),
            "stats": compute_stats(put_latencies[-batch_size:]),
        },
        "get": {
            "throughput": statistics.mean(get_throughputs),
            "stats": compute_stats(get_latencies[-batch_size:]),
        },
        "alloc_bytes": alloc_bytes,
    }


def benchmark_pipeline(count: int = 2000, runs: int = 3) -> dict[str, Any]:
    """Benchmark TradingPipeline end-to-end throughput."""
    pipeline_throughputs = []
    latencies = []

    for _ in range(runs):
        pipeline = create_pipeline(capacity=count + 10)
        t_start = time.perf_counter()
        for _ in range(count):
            t0 = time.perf_counter_ns()
            pipeline.process_next_simulated_order()
            t1 = time.perf_counter_ns()
            latencies.append((t1 - t0) / 1000.0)
        t_end = time.perf_counter()
        duration = t_end - t_start
        pipeline_throughputs.append(count / duration)
        pipeline.close()

    return {
        "throughput_orders_per_sec": statistics.mean(pipeline_throughputs),
        "stats": compute_stats(latencies[-count:]),
    }


def run_all_benchmarks() -> None:
    print("=" * 70)
    print("CHRONOSMATCH BINARY PROTOCOL & RING BUFFER BENCHMARK")
    print("=" * 70)

    orders = generate_sample_orders(10_000)
    print(f"Generated {len(orders)} sample orders for benchmark.")

    print("\n--- 1. Order Encoding Benchmark (10,000 orders) ---")
    enc_results = benchmark_encoding(orders)
    print(f"encode_order (allocating bytes):")
    print(f"  Throughput:  {enc_results['encode_order']['throughput']:,.0f} orders/sec")
    print(f"  Mean latency: {enc_results['encode_order']['stats']['mean']:.3f} us")
    print(f"  p50 latency:  {enc_results['encode_order']['stats']['p50']:.3f} us")
    print(f"  p95 latency:  {enc_results['encode_order']['stats']['p95']:.3f} us")
    print(f"  Allocated:    {enc_results['encode_order']['alloc_bytes']:,} bytes")

    print(f"\nencode_order_into (direct buffer write):")
    print(f"  Throughput:  {enc_results['encode_order_into']['throughput']:,.0f} orders/sec")
    print(f"  Mean latency: {enc_results['encode_order_into']['stats']['mean']:.3f} us")
    print(f"  p50 latency:  {enc_results['encode_order_into']['stats']['p50']:.3f} us")
    print(f"  p95 latency:  {enc_results['encode_order_into']['stats']['p95']:.3f} us")
    print(f"  Allocated:    {enc_results['encode_order_into']['alloc_bytes']:,} bytes")

    print("\n--- 2. Order Decoding Benchmark (10,000 orders) ---")
    dec_results = benchmark_decoding(orders)
    print(f"decode_order (from individual bytes):")
    print(f"  Throughput:  {dec_results['decode_order']['throughput']:,.0f} orders/sec")
    print(f"  Mean latency: {dec_results['decode_order']['stats']['mean']:.3f} us")
    print(f"  p50 latency:  {dec_results['decode_order']['stats']['p50']:.3f} us")
    print(f"  p95 latency:  {dec_results['decode_order']['stats']['p95']:.3f} us")
    print(f"  Allocated:    {dec_results['decode_order']['alloc_bytes']:,} bytes")

    print(f"\ndecode_order_from (from continuous buffer at offset):")
    print(f"  Throughput:  {dec_results['decode_order_from']['throughput']:,.0f} orders/sec")
    print(f"  Mean latency: {dec_results['decode_order_from']['stats']['mean']:.3f} us")
    print(f"  p50 latency:  {dec_results['decode_order_from']['stats']['p50']:.3f} us")
    print(f"  p95 latency:  {dec_results['decode_order_from']['stats']['p95']:.3f} us")
    print(f"  Allocated:    {dec_results['decode_order_from']['alloc_bytes']:,} bytes")

    print("\n--- 3. RingBuffer Benchmark (2,000 orders) ---")
    rb_results = benchmark_ring_buffer(orders)
    print(f"RingBuffer.put (in-place encoding):")
    print(f"  Throughput:  {rb_results['put']['throughput']:,.0f} orders/sec")
    print(f"  Mean latency: {rb_results['put']['stats']['mean']:.3f} us")
    print(f"  p50 latency:  {rb_results['put']['stats']['p50']:.3f} us")
    print(f"  p95 latency:  {rb_results['put']['stats']['p95']:.3f} us")

    print(f"\nRingBuffer.get (in-place decoding):")
    print(f"  Throughput:  {rb_results['get']['throughput']:,.0f} orders/sec")
    print(f"  Mean latency: {rb_results['get']['stats']['mean']:.3f} us")
    print(f"  p50 latency:  {rb_results['get']['stats']['p50']:.3f} us")
    print(f"  p95 latency:  {rb_results['get']['stats']['p95']:.3f} us")
    print(f"  Allocated:    {rb_results['alloc_bytes']:,} bytes")

    print("\n--- 4. TradingPipeline End-to-End Simulation (2,000 orders) ---")
    pipe_results = benchmark_pipeline(count=2000)
    print(f"  Throughput:  {pipe_results['throughput_orders_per_sec']:,.0f} orders/sec")
    print(f"  Mean latency: {pipe_results['stats']['mean']:.3f} us")
    print(f"  p50 latency:  {pipe_results['stats']['p50']:.3f} us")
    print(f"  p95 latency:  {pipe_results['stats']['p95']:.3f} us")

    print("=" * 70)


if __name__ == "__main__":
    run_all_benchmarks()
