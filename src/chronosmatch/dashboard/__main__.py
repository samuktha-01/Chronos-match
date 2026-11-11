"""Command-line entry point for ChronosMatch Terminal Dashboard.

Allows launching the real-time terminal UI directly via:
    python -m chronosmatch.dashboard [options]

Features:
- Live ChronosMatch TradingPipeline execution (default).
- Configurable order count (--orders) and throughput rate (--rate).
- Multiple traffic simulation modes (--mode: POISSON, UNIFORM, BURST, MAX_THROUGHPUT).
- Configurable refresh interval (--refresh / -r).
- Demo simulation mode (--demo) generating synthetic manual orders.
- Plain-text mode (--plain) for non-interactive log streaming.
- Pure ASCII box mode (--ascii) for legacy consoles.
- Single snapshot execution (--once).
- Clean Ctrl+C graceful shutdown.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import sys
import threading
import time
from typing import Any

from chronosmatch.cython_engine import is_cython_compiled, CythonMatchingEngine
from chronosmatch.dashboard.metrics import (
    BoundedLatencyTracker,
    DashboardMetricsCollector,
)
from chronosmatch.dashboard.pipeline_runner import (
    PipelineDashboardRunner,
)
from chronosmatch.dashboard.terminal import (
    TerminalDashboard,
)
from chronosmatch.dashboard.curses_backend import (
    DashboardBackend,
    create_dashboard,
)
from chronosmatch.market_simulator import (
    MarketFeedConfig,
    SimulationMode,
)
from chronosmatch.matching_engine import MatchingEngine
from chronosmatch.pipeline import (
    TradingPipeline,
    create_pipeline,
)


def _parse_rate_arg(val: Any) -> tuple[float, SimulationMode | None]:
    """Parse rate argument supporting numeric values or 'max'."""
    if isinstance(val, (int, float)):
        f_val = float(val)
        if f_val <= 0:
            return float("inf"), SimulationMode.MAX_THROUGHPUT
        return f_val, None
    s = str(val).strip().lower()
    if s in ("max", "max_throughput", "infinite", "inf"):
        return float("inf"), SimulationMode.MAX_THROUGHPUT
    try:
        f_val = float(s)
        if f_val <= 0:
            return float("inf"), SimulationMode.MAX_THROUGHPUT
        return f_val, None
    except ValueError:
        return 2000.0, None


def _run_demo_producer(
    collector: DashboardMetricsCollector,
    stop_event: Any,
) -> None:
    """Background simulator thread that generates synthetic orders and trades for demo mode."""
    orders = 0
    trades = 0
    volume = 0
    cancelled = 0
    dropped = 0

    while not stop_event.is_set():
        batch_size = random.randint(50, 200)
        orders += batch_size
        collector.record_order_generated(batch_size)
        collector.record_order_published(batch_size)
        collector.record_order_consumed(batch_size)
        collector.record_order_processed(batch_size)

        if random.random() < 0.6:
            trade_qty = random.randint(10, 100)
            trades += 1
            volume += trade_qty
            collector.record_trade(trade_qty)

            simulated_latency_ns = random.gauss(1_500, 400)
            if random.random() < 0.02:
                simulated_latency_ns = random.uniform(8_000, 35_000)
            collector.record_latency(max(400.0, simulated_latency_ns))

        if random.random() < 0.15:
            canc = random.randint(1, 5)
            cancelled += canc
            collector.record_order_cancelled(canc)

        if random.random() < 0.01:
            dropped += 1
            collector.record_order_dropped(1)

        time.sleep(0.05)


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point for ChronosMatch Terminal Dashboard."""
    parser = argparse.ArgumentParser(
        prog="python -m chronosmatch.dashboard",
        description="ChronosMatch Real-Time Terminal Telemetry Dashboard",
    )
    parser.add_argument(
        "--continuous",
        "-c",
        action="store_true",
        help="Run continuous live market simulation until interrupted (Ctrl+C)",
    )
    parser.add_argument(
        "--orders",
        "-n",
        type=int,
        default=5_000,
        help="Total orders to simulate and match through the pipeline (default: 5,000; set to 0 or use -c for continuous)",
    )
    parser.add_argument(
        "--rate",
        default="2000",
        help="Target synthetic order generation rate in ops/sec (e.g. 1000, 10000, 50000, 100000, or 'max') (default: 2,000)",
    )
    parser.add_argument(
        "--mode",
        choices=["POISSON", "UNIFORM", "BURST", "MAX_THROUGHPUT"],
        default="POISSON",
        help="Market simulator traffic pattern mode (default: POISSON)",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=1024,
        help="Ring buffer slot capacity (default: 1024)",
    )
    parser.add_argument(
        "--refresh",
        "-r",
        type=float,
        default=0.5,
        help="Dashboard refresh interval in seconds (default: 0.5s)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in standalone demo mode without attached TradingPipeline",
    )
    parser.add_argument(
        "--whale-threshold",
        type=int,
        default=1000,
        help="Whale order quantity threshold (default: 1,000)",
    )
    parser.add_argument(
        "--wal",
        type=str,
        default=None,
        help="Optional path to Write-Ahead Log for persistent audit logging",
    )
    parser.add_argument(
        "--cython",
        action="store_true",
        help="Use compiled Cython matching engine if available",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Render a single snapshot and exit immediately",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Disable interactive cursor controls and print as plain streaming text",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Use plain ASCII box borders instead of Unicode",
    )
    parser.add_argument(
        "--backend",
        choices=["ansi", "curses", "auto"],
        default="ansi" if os.name == "nt" else "auto",
        help="Terminal UI rendering backend ('ansi', 'curses', or 'auto'). Default is 'ansi' on Windows.",
    )

    args = parser.parse_args(argv)

    if args.refresh <= 0:
        sys.stderr.write("Error: --refresh interval must be strictly positive (> 0).\n")
        return 1

    # Single-shot execution
    if args.once:
        collector = DashboardMetricsCollector(source_name="ChronosMatch Snapshot")
        dashboard = create_dashboard(
            collector=collector,
            backend=args.backend,
            interactive=False,
            use_colors=not args.no_color,
            use_unicode=False if args.ascii else None,
            fallback_to_ansi=True,
        )
        dashboard.render_once()
        return 0

    # Demo simulation mode (standalone)
    if args.demo:
        collector = DashboardMetricsCollector(source_name="ChronosMatch Demo")
        collector.start()
        stop_event = threading.Event()
        demo_thread = threading.Thread(
            target=_run_demo_producer,
            args=(collector, stop_event),
            daemon=True,
        )
        demo_thread.start()

        dashboard = create_dashboard(
            collector=collector,
            backend=args.backend,
            refresh_interval=args.refresh,
            interactive=False if args.plain else None,
            use_colors=not args.no_color,
            use_unicode=False if args.ascii else None,
            clear_screen=not args.plain,
            fallback_to_ansi=True,
        )

        try:
            dashboard.start()
            while True:
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            stop_event.set()
            dashboard.stop()
            demo_thread.join(timeout=1.0)
            sys.stdout.write("\nChronosMatch Demo Dashboard stopped gracefully.\n")
        return 0

    # Real TradingPipeline execution
    engine = None
    if args.cython:
        if is_cython_compiled():
            engine = CythonMatchingEngine()
        else:
            sys.stderr.write("Note: Native Cython extension not compiled. Falling back to pure Python MatchingEngine.\n")
            engine = MatchingEngine()

    is_continuous = args.continuous or (args.orders <= 0)
    total_orders = None if is_continuous else args.orders

    parsed_rate, forced_mode = _parse_rate_arg(args.rate)
    if forced_mode is not None:
        sim_mode = forced_mode
    else:
        sim_mode = SimulationMode(args.mode)

    config = MarketFeedConfig(
        total_orders=total_orders,
        rate=parsed_rate,
        mode=sim_mode,
        whale_order_threshold=args.whale_threshold,
    )
    pipeline = create_pipeline(
        config=config,
        capacity=args.capacity,
        engine=engine,
        wal_path=args.wal,
        whale_threshold=args.whale_threshold,
    )

    runner = PipelineDashboardRunner(
        pipeline=pipeline,
        refresh_interval=args.refresh,
        interactive=False if args.plain else None,
        use_colors=not args.no_color,
        use_unicode=False if args.ascii else None,
        backend=args.backend,
    )

    try:
        runner.run(total_orders=total_orders)
    except KeyboardInterrupt:
        if is_continuous:
            sys.stdout.write("\nChronosMatch continuous simulation interrupted by user (Ctrl+C).\n")
        else:
            sys.stdout.write("\nChronosMatch Pipeline interrupted by user.\n")
    finally:
        runner.close()
        if is_continuous:
            sys.stdout.write("\nChronosMatch Continuous Live Simulation stopped cleanly.\n")
        else:
            sys.stdout.write("\nChronosMatch Pipeline & Dashboard finished cleanly.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
