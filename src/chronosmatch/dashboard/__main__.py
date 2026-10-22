"""Command-line entry point for ChronosMatch Terminal Dashboard.

Allows launching the real-time terminal UI directly via:
    python -m chronosmatch.dashboard [options]

Features:
- Configurable refresh interval (--refresh / -r).
- Demo simulation mode (--demo) generating live synthetic trades and latencies.
- Plain-text mode (--plain) for non-interactive log streaming.
- Pure ASCII box mode (--ascii) for legacy consoles.
- Single snapshot execution (--once).
- Clean Ctrl+C graceful shutdown.
"""

from __future__ import annotations

import argparse
import random
import sys
import time

from chronosmatch.dashboard.metrics import (
    BoundedLatencyTracker,
    DashboardMetricsCollector,
)
from chronosmatch.dashboard.terminal import (
    TerminalDashboard,
)


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

    best_bid = 100.00
    best_ask = 100.05

    while not stop_event.is_set():
        # Simulate burst of orders
        batch_size = random.randint(50, 200)
        orders += batch_size
        collector.record_order_generated(batch_size)
        collector.record_order_published(batch_size)
        collector.record_order_consumed(batch_size)
        collector.record_order_processed(batch_size)

        # Simulate trades and latencies
        if random.random() < 0.6:
            trade_qty = random.randint(10, 100)
            trades += 1
            volume += trade_qty
            collector.record_trade(trade_qty)

            # Record realistic latency sample in nanoseconds (800ns to 15,000ns)
            simulated_latency_ns = random.gauss(1_500, 400)
            if random.random() < 0.02:
                # Occasional tail latency spike
                simulated_latency_ns = random.uniform(8_000, 35_000)
            collector.record_latency(max(400.0, simulated_latency_ns))

        # Occasional cancel
        if random.random() < 0.15:
            canc = random.randint(1, 5)
            cancelled += canc
            collector.record_order_cancelled(canc)

        # Rare drop
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
        "--refresh",
        "-r",
        type=float,
        default=0.5,
        help="Refresh interval in seconds (default: 0.5s)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with synthetic order flow simulation",
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

    args = parser.parse_args(argv)

    if args.refresh <= 0:
        sys.stderr.write("Error: --refresh interval must be strictly positive (> 0).\n")
        return 1

    collector = DashboardMetricsCollector(
        source_name="ChronosMatch Demo" if args.demo else "ChronosMatch Standalone"
    )

    import threading

    demo_stop_event = threading.Event()
    demo_thread = None

    if args.demo:
        collector.start()
        demo_thread = threading.Thread(
            target=_run_demo_producer,
            args=(collector, demo_stop_event),
            daemon=True,
        )
        demo_thread.start()

    dashboard = TerminalDashboard(
        collector=collector,
        refresh_interval=args.refresh,
        interactive=not args.plain and not args.once,
        use_colors=not args.no_color,
        use_unicode=False if args.ascii else None,
        clear_screen=not args.plain and not args.once,
    )

    if args.once:
        dashboard.render_once()
        if demo_thread is not None:
            demo_stop_event.set()
        return 0

    try:
        dashboard.start()
        # Keep main thread alive until interrupt
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.stop()
        if demo_thread is not None:
            demo_stop_event.set()
            demo_thread.join(timeout=1.0)
        sys.stdout.write("\nChronosMatch Dashboard stopped gracefully.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
