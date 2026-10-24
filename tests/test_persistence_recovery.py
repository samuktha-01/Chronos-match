"""Comprehensive Test Suite for ChronosMatch Phase 12: Trade Persistence and Crash Recovery.

Validates:
1. WAL append and readback (orders, cancels, trades, checkpoints, sequence numbers).
2. Trade persistence (stable identifiers, in-memory & file-backed TradeStore, volume aggregation).
3. Replay and reconstruction (deterministic LOB reconstruction, FIFO queue priority).
4. Restart recovery (clean shutdown, state restore, continued matching after restart).
5. Truncated final record handling (partial header, partial payload, repair & resume).
6. Corrupted record handling (CRC32 mismatch, bad magic, strict vs salvage recovery).
7. Duplicate replay protection (idempotency, crash before trade record write, trade ID monotonicity).
8. Recovery from an empty log (0-byte file, header-only file, non-existent file).
9. Durability policies & async persistence worker (IMMEDIATE_SYNC, FLUSH_ON_WRITE, ASYNC_BUFFERED).
10. Persistence failures and clean resource shutdown (closed log guards, idempotent close, I/O errors).
11. End-to-end TradingPipeline persistence integration and crash recovery.
"""

from __future__ import annotations

from decimal import Decimal
import os
from pathlib import Path
import struct
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from chronosmatch.matching_engine import MatchingEngine, Trade
from chronosmatch.models import Order, OrderSide
from chronosmatch.persistence import (
    AsyncPersistenceWorker,
    CancelRecord,
    CheckpointRecord,
    DuplicateTradeError,
    DurabilityPolicy,
    PersistedMatchingEngine,
    PersistenceError,
    RecoveryError,
    RecoveryManager,
    RecoveryResult,
    TRADE_PAYLOAD_SIZE,
    TradeRecoveryResult,
    TradeStore,
    WALClosedError,
    WALCorruptedRecordError,
    WALError,
    WALRecord,
    WALRecordType,
    WALTruncatedRecordError,
    WALWriteError,
    WriteAheadLog,
    decode_trade_payload,
    encode_trade_payload,
)
from chronosmatch.pipeline import TradingPipeline, create_pipeline


# ==============================================================================
# 1. WAL Append and Readback
# ==============================================================================


class TestWALAppendAndReadback:
    """Verify append and readback of all WAL record types and sequence integrity."""

    def test_append_order_and_readback(self, tmp_path: Path) -> None:
        """Verify orders of all price types (float, int, Decimal) round-trip faithfully."""
        wal_file = tmp_path / "test_orders.wal"
        orders = [
            Order(order_id="ORD-001", side=OrderSide.BUY, price=150.25, quantity=100, timestamp=1000, sequence_number=1),
            Order(order_id="ORD-002", side=OrderSide.SELL, price=200, quantity=50, timestamp=2000, sequence_number=2),
            Order(order_id="ORD-003", side=OrderSide.BUY, price=Decimal("175.50000000"), quantity=75, timestamp=3000),
        ]

        with WriteAheadLog(wal_file, durability=DurabilityPolicy.FLUSH_ON_WRITE) as wal:
            lsns = [wal.append_order(o) for o in orders]
            assert lsns == [1, 2, 3]
            assert wal.record_count == 3
            assert wal.last_sequence_number == 3

        # Read back in new handle
        with WriteAheadLog(wal_file, create=False) as wal:
            records, warnings, offset = wal.read_records(strict=True)
            assert len(warnings) == 0
            assert len(records) == 3

            for i, rec in enumerate(records):
                assert rec.sequence_number == i + 1
                assert rec.is_order
                assert isinstance(rec.data, Order)
                orig = orders[i]
                assert rec.data.order_id == orig.order_id
                assert rec.data.side == orig.side
                assert rec.data.price == orig.price
                assert type(rec.data.price) is type(orig.price)
                assert rec.data.quantity == orig.quantity
                assert rec.data.timestamp == orig.timestamp
                assert rec.data.sequence_number == orig.sequence_number

    def test_append_cancel_and_readback(self, tmp_path: Path) -> None:
        """Verify order cancellation events are serialized and deserialized accurately."""
        wal_file = tmp_path / "test_cancels.wal"
        with WriteAheadLog(wal_file) as wal:
            lsn1 = wal.append_cancel("ORD-CANCEL-1", timestamp_ns=123456789)
            lsn2 = wal.append_cancel("ORD-CANCEL-2", timestamp_ns=987654321)
            assert lsn1 == 1
            assert lsn2 == 2

        with WriteAheadLog(wal_file, create=False) as wal:
            records, warnings, _ = wal.read_records(strict=True)
            assert len(records) == 2
            assert records[0].is_cancel
            assert isinstance(records[0].data, CancelRecord)
            assert records[0].data.order_id == "ORD-CANCEL-1"
            assert records[0].data.timestamp_ns == 123456789
            assert records[1].data.order_id == "ORD-CANCEL-2"
            assert records[1].data.timestamp_ns == 987654321

    def test_append_trade_and_readback(self, tmp_path: Path) -> None:
        """Verify executed trade records with int and string IDs are preserved."""
        wal_file = tmp_path / "test_trades.wal"
        trades = [
            Trade(trade_id=1, resting_order_id="MAKER-1", incoming_order_id="TAKER-1", price=150.50, quantity=10, timestamp=11111),
            Trade(trade_id="TRD-CUSTOM-99", resting_order_id="MAKER-2", incoming_order_id="TAKER-2", price=Decimal("200.12340000"), quantity=25, timestamp=22222),
        ]

        with WriteAheadLog(wal_file) as wal:
            lsns = wal.append_trades(trades)
            assert lsns == [1, 2]

        with WriteAheadLog(wal_file, create=False) as wal:
            records, warnings, _ = wal.read_records(strict=True)
            assert len(records) == 2
            assert records[0].is_trade
            assert isinstance(records[0].data, Trade)
            assert records[0].data.trade_id == 1
            assert records[0].data.resting_order_id == "MAKER-1"
            assert records[0].data.incoming_order_id == "TAKER-1"
            assert records[0].data.price == 150.50
            assert records[0].data.quantity == 10

            assert records[1].data.trade_id == "TRD-CUSTOM-99"
            assert records[1].data.price == Decimal("200.12340000")

    def test_append_checkpoint_and_readback(self, tmp_path: Path) -> None:
        """Verify checkpoint records store engine high-water marks."""
        wal_file = tmp_path / "test_cp.wal"
        with WriteAheadLog(wal_file) as wal:
            lsn = wal.append_checkpoint(last_order_seq=100, last_trade_id=42)
            assert lsn == 1

        with WriteAheadLog(wal_file, create=False) as wal:
            records, warnings, _ = wal.read_records(strict=True)
            assert len(records) == 1
            assert records[0].is_checkpoint
            assert isinstance(records[0].data, CheckpointRecord)
            assert records[0].data.last_order_seq == 100
            assert records[0].data.last_trade_id == 42

    def test_sequence_number_monotonicity_across_reopen(self, tmp_path: Path) -> None:
        """Verify LSN increases monotonically across multiple process open/close sessions."""
        wal_file = tmp_path / "test_monotonic.wal"

        # Session 1: append 3 records
        with WriteAheadLog(wal_file) as wal:
            wal.append_order(Order(order_id="O1", side=OrderSide.BUY, price=100, quantity=10, timestamp=1))
            wal.append_order(Order(order_id="O2", side=OrderSide.BUY, price=101, quantity=10, timestamp=2))
            wal.append_cancel("O1", 3)

        # Session 2: reopen and append 2 more records
        with WriteAheadLog(wal_file) as wal:
            assert wal.last_sequence_number == 3
            l4 = wal.append_order(Order(order_id="O3", side=OrderSide.SELL, price=105, quantity=5, timestamp=4))
            l5 = wal.append_trade(Trade(trade_id=1, resting_order_id="O2", incoming_order_id="O3", price=101, quantity=5, timestamp=5))
            assert l4 == 4
            assert l5 == 5

        # Verify all 5 records have contiguous LSNs 1..5
        with WriteAheadLog(wal_file, create=False) as wal:
            records, _, _ = wal.read_records(strict=True)
            assert [r.sequence_number for r in records] == [1, 2, 3, 4, 5]


# ==============================================================================
# 2. Trade Persistence
# ==============================================================================


class TestTradePersistence:
    """Verify TradeStore operations, querying, volume aggregation, and persistence."""

    def test_in_memory_trade_store(self) -> None:
        """Verify in-memory TradeStore indexing, lookups, and volume calculation."""
        store = TradeStore()
        t1 = Trade(trade_id=1, resting_order_id="M1", incoming_order_id="T1", price=100.0, quantity=10, timestamp=10)
        t2 = Trade(trade_id=2, resting_order_id="M1", incoming_order_id="T2", price=100.0, quantity=20, timestamp=20)
        t3 = Trade(trade_id=3, resting_order_id="M2", incoming_order_id="T3", price=105.0, quantity=15, timestamp=30)

        store.append_trades([t1, t2, t3])

        assert store.count() == 3
        assert store.total_volume() == 45
        assert store.get_trade(1) == t1
        assert store.get_trade(2) == t2
        assert store.get_trade(999) is None

        # Query by order ID
        assert store.get_trades_for_order("M1") == [t1, t2]
        assert store.get_trades_for_order("T2") == [t2]
        assert store.get_trades_for_order("NONEXISTENT") == []

    def test_disk_backed_trade_store_reload(self, tmp_path: Path) -> None:
        """Verify TradeStore persistence across file close and reload."""
        trade_file = tmp_path / "trades.dat"
        store1 = TradeStore(filepath=trade_file, durability=DurabilityPolicy.FLUSH_ON_WRITE)
        t1 = Trade(trade_id=101, resting_order_id="ORD-A", incoming_order_id="ORD-B", price=50.25, quantity=100, timestamp=500)
        t2 = Trade(trade_id="TRD-BETA", resting_order_id="ORD-C", incoming_order_id="ORD-D", price=Decimal("75.00"), quantity=200, timestamp=600)
        store1.append_trades([t1, t2])
        store1.close()

        # Reopen from disk
        store2 = TradeStore(filepath=trade_file)
        assert store2.count() == 2
        assert store2.total_volume() == 300
        assert store2.get_trade(101) == t1
        assert store2.get_trade("TRD-BETA") == t2
        store2.close()

    def test_trade_store_idempotent_reappend(self) -> None:
        """Verify appending identical trade record is idempotent and does not inflate count or volume."""
        store = TradeStore()
        t1 = Trade(trade_id=1, resting_order_id="M1", incoming_order_id="T1", price=100.0, quantity=10, timestamp=10)
        store.append_trade(t1)
        # Re-append identical trade
        store.append_trade(t1)
        assert store.count() == 1
        assert store.total_volume() == 10

    def test_trade_store_conflicting_id_raises(self) -> None:
        """Verify appending duplicate trade ID with conflicting details raises DuplicateTradeError."""
        store = TradeStore()
        t1 = Trade(trade_id=1, resting_order_id="M1", incoming_order_id="T1", price=100.0, quantity=10, timestamp=10)
        conflicting = Trade(trade_id=1, resting_order_id="M1", incoming_order_id="T1", price=999.0, quantity=10, timestamp=10)
        store.append_trade(t1)
        with pytest.raises(DuplicateTradeError):
            store.append_trade(conflicting)

    def test_trade_store_from_wal_factory(self, tmp_path: Path) -> None:
        """Verify TradeStore can be extracted directly from a WriteAheadLog."""
        wal_file = tmp_path / "combined.wal"
        with WriteAheadLog(wal_file) as wal:
            wal.append_order(Order(order_id="O1", side=OrderSide.BUY, price=100, quantity=10, timestamp=1))
            wal.append_trade(Trade(trade_id=1, resting_order_id="O1", incoming_order_id="O2", price=100, quantity=10, timestamp=2))
            wal.append_cancel("O1", 3)
            wal.append_trade(Trade(trade_id=2, resting_order_id="O3", incoming_order_id="O4", price=105, quantity=20, timestamp=4))

        store = TradeStore.from_wal(wal_file)
        assert store.count() == 2
        assert store.total_volume() == 30
        assert store.get_trade(1).price == 100
        assert store.get_trade(2).price == 105

    def test_trade_store_reopen_with_truncated_trailing_record(self, tmp_path: Path) -> None:
        """Verify TradeStore recovers valid trades and truncates trailing partial writes from crashes."""
        trade_file = tmp_path / "corrupt_trailing.dat"
        t1 = Trade(trade_id=1, resting_order_id="M1", incoming_order_id="T1", price=100.0, quantity=10, timestamp=100)
        t2 = Trade(trade_id=2, resting_order_id="M2", incoming_order_id="T2", price=101.5, quantity=20, timestamp=200)
        t3 = Trade(trade_id=3, resting_order_id="M3", incoming_order_id="T3", price=102.0, quantity=30, timestamp=300)

        # 1. Write 2 trades cleanly and close
        with TradeStore(filepath=trade_file) as store:
            store.append_trades([t1, t2])
        assert trade_file.stat().st_size == 2 * TRADE_PAYLOAD_SIZE

        # 2. Simulate crash mid-write leaving 47 bytes of partial trailing data
        with open(trade_file, "ab") as f:
            f.write(b"X" * 47)
        assert trade_file.stat().st_size == (2 * TRADE_PAYLOAD_SIZE) + 47

        # 3. Reopen TradeStore: should safely recover t1 and t2, truncating trailing corrupt bytes
        with TradeStore(filepath=trade_file) as store:
            assert store.count() == 2
            assert store.get_trade(1) == t1
            assert store.get_trade(2) == t2
            # Append new trade after recovery
            store.append_trade(t3)

        # 4. Verify file on disk is properly aligned to exactly 3 trade payloads (no misaligned offsets)
        assert trade_file.stat().st_size == 3 * TRADE_PAYLOAD_SIZE

        # 5. Reopen store in a new session and verify all 3 trades are cleanly readable
        with TradeStore(filepath=trade_file) as store:
            assert store.count() == 3
            assert store.total_volume() == 60
            assert store.get_trade(1) == t1
            assert store.get_trade(2) == t2
            assert store.get_trade(3) == t3

    def test_trade_store_context_manager(self, tmp_path: Path) -> None:
        """Verify TradeStore properly supports the context manager protocol."""
        trade_file = tmp_path / "cm_test.dat"
        t1 = Trade(trade_id=10, resting_order_id="M10", incoming_order_id="T10", price=50.0, quantity=5, timestamp=10)

        with TradeStore(filepath=trade_file) as store:
            assert not store.is_closed
            store.append_trade(t1)

        assert store.is_closed

        # Reopen and confirm persisted
        with TradeStore(filepath=trade_file) as store:
            assert store.count() == 1
            assert store.get_trade(10) == t1

    def test_persist_single_trade_in_memory_and_disk(self, tmp_path: Path) -> None:
        """Verify persisting a single trade preserves all fields and reloads faithfully."""
        trade_file = tmp_path / "single_trade.dat"
        trade = Trade(
            trade_id=1,
            resting_order_id="MAKER-001",
            incoming_order_id="TAKER-001",
            price=150.75,
            quantity=50,
            timestamp=1234567890,
        )

        # 1. In-memory store
        mem_store = TradeStore()
        mem_store.append_trade(trade)
        assert mem_store.count() == 1
        assert mem_store.total_volume() == 50
        assert mem_store.get_trade(1) == trade
        assert mem_store.get_trades_for_order("MAKER-001") == [trade]
        assert mem_store.get_trades_for_order("TAKER-001") == [trade]

        # 2. Disk-backed store
        with TradeStore(filepath=trade_file, durability=DurabilityPolicy.IMMEDIATE_SYNC) as disk_store:
            disk_store.append_trade(trade)
            assert disk_store.count() == 1
            assert disk_store.total_volume() == 50

        assert trade_file.stat().st_size == TRADE_PAYLOAD_SIZE

        # 3. Reload from disk
        with TradeStore(filepath=trade_file) as disk_store2:
            assert disk_store2.count() == 1
            assert disk_store2.total_volume() == 50
            loaded = disk_store2.get_trade(1)
            assert loaded is not None
            assert loaded.trade_id == 1
            assert loaded.resting_order_id == "MAKER-001"
            assert loaded.incoming_order_id == "TAKER-001"
            assert loaded.maker_order_id == "MAKER-001"
            assert loaded.taker_order_id == "TAKER-001"
            assert loaded.price == 150.75
            assert loaded.execution_price == 150.75
            assert loaded.quantity == 50
            assert loaded.executed_quantity == 50
            assert loaded.execution_qty == 50
            assert loaded.timestamp == 1234567890
            assert loaded.timestamp_ns == 1234567890

    def test_persist_multiple_trades_order_and_volume_preservation(self, tmp_path: Path) -> None:
        """Verify persisting multiple trades preserves strict ordering, volume, and multi-order index."""
        trade_file = tmp_path / "multiple_trades.dat"
        trades = [
            Trade(trade_id=1, resting_order_id="M1", incoming_order_id="T1", price=100.0, quantity=10, timestamp=100),
            Trade(trade_id=2, resting_order_id="M1", incoming_order_id="T2", price=100.5, quantity=25, timestamp=200),
            Trade(trade_id=3, resting_order_id="M2", incoming_order_id="T3", price=101.0, quantity=15, timestamp=300),
            Trade(trade_id=4, resting_order_id="M3", incoming_order_id="T1", price=102.25, quantity=30, timestamp=400),
        ]

        with TradeStore(filepath=trade_file, durability=DurabilityPolicy.FLUSH_ON_WRITE) as store:
            store.append_trades(trades)
            assert store.count() == 4
            assert store.total_volume() == 80
            assert store.get_all_trades() == trades
            assert store.get_trades_for_order("M1") == [trades[0], trades[1]]
            assert store.get_trades_for_order("T1") == [trades[0], trades[3]]

        assert trade_file.stat().st_size == 4 * TRADE_PAYLOAD_SIZE

        # Reload and confirm strict ordering and state consistency
        with TradeStore(filepath=trade_file) as reloaded:
            assert reloaded.count() == 4
            assert reloaded.total_volume() == 80
            reloaded_trades = reloaded.get_all_trades()
            assert len(reloaded_trades) == 4
            for orig, rec in zip(trades, reloaded_trades):
                assert orig == rec
            assert reloaded.get_trades_for_order("M1") == [trades[0], trades[1]]
            assert reloaded.get_trades_for_order("T1") == [trades[0], trades[3]]

    def test_trade_payload_serialization_and_deserialization_fidelity(self) -> None:
        """Verify binary serialization round-trip fidelity across float, int, and Decimal price types."""
        test_cases = [
            Trade(trade_id=101, resting_order_id="ORD-M-1", incoming_order_id="ORD-T-1", price=125.50, quantity=100, timestamp=1000),
            Trade(trade_id="TRD-STR-99", resting_order_id="ORD-M-2", incoming_order_id="ORD-T-2", price=250, quantity=50, timestamp=2000),
            Trade(trade_id=999999, resting_order_id="ORD-M-3", incoming_order_id="ORD-T-3", price=Decimal("175.12345678"), quantity=75, timestamp=3000),
        ]

        for trade in test_cases:
            payload = encode_trade_payload(trade)
            assert len(payload) == TRADE_PAYLOAD_SIZE
            decoded = decode_trade_payload(payload)

            assert decoded.trade_id == trade.trade_id
            assert type(decoded.trade_id) is type(trade.trade_id)
            assert decoded.resting_order_id == trade.resting_order_id
            assert decoded.incoming_order_id == trade.incoming_order_id
            assert decoded.maker_order_id == trade.maker_order_id
            assert decoded.taker_order_id == trade.taker_order_id
            assert decoded.price == trade.price
            assert type(decoded.price) is type(trade.price)
            assert decoded.quantity == trade.quantity
            assert decoded.timestamp == trade.timestamp

    def test_trade_payload_corrupted_payload_handling(self) -> None:
        """Verify decoding corrupted or mis-sized payloads raises appropriate ValueError."""
        # 1. Invalid payload size
        with pytest.raises(ValueError, match="Expected 128 bytes"):
            decode_trade_payload(b"\x00" * 64)

        # 2. Corrupted price ticks
        valid_trade = Trade(trade_id=1, resting_order_id="M1", incoming_order_id="T1", price=100.0, quantity=10, timestamp=10)
        valid_payload = bytearray(encode_trade_payload(valid_trade))

        # Corrupt price ticks to <= 0
        struct.pack_into("<q", valid_payload, 96, -500)
        with pytest.raises(ValueError, match="price ticks must be strictly positive"):
            decode_trade_payload(bytes(valid_payload))

        # Corrupt price type code to invalid value
        struct.pack_into("<B", valid_payload, 108, 99)
        with pytest.raises(ValueError):
            decode_trade_payload(bytes(valid_payload))

    def test_duplicate_trade_timestamp_mismatch_raises(self) -> None:
        """Verify duplicate trade ID with conflicting timestamp raises DuplicateTradeError."""
        store = TradeStore()
        t1 = Trade(trade_id=1, resting_order_id="M1", incoming_order_id="T1", price=100.0, quantity=10, timestamp=1000)
        t1_diff_ts = Trade(trade_id=1, resting_order_id="M1", incoming_order_id="T1", price=100.0, quantity=10, timestamp=2000)

        store.append_trade(t1)
        with pytest.raises(DuplicateTradeError, match="conflicting data"):
            store.append_trade(t1_diff_ts)

    def test_duplicate_write_idempotency_multiple_calls(self, tmp_path: Path) -> None:
        """Verify idempotent repeated writes do not duplicate records on disk or in memory."""
        trade_file = tmp_path / "idempotent_trades.dat"
        t1 = Trade(trade_id=1, resting_order_id="M1", incoming_order_id="T1", price=100.0, quantity=10, timestamp=1000)
        t2 = Trade(trade_id=2, resting_order_id="M2", incoming_order_id="T2", price=105.0, quantity=20, timestamp=2000)

        with TradeStore(filepath=trade_file) as store:
            store.append_trades([t1, t2])
            # Re-append exact same trades
            store.append_trade(t1)
            store.append_trades([t1, t2])

            assert store.count() == 2
            assert store.total_volume() == 30

        # Disk file size must reflect exactly 2 payloads, not 5
        assert trade_file.stat().st_size == 2 * TRADE_PAYLOAD_SIZE


# ==============================================================================
# 3. Replay and State Reconstruction
# ==============================================================================


class TestReplayAndReconstruction:
    """Verify deterministic order book state reconstruction and priority preservation."""

    def test_order_book_depth_and_spread_reconstruction(self, tmp_path: Path) -> None:
        """Verify limit orders placed in WAL reconstruct exact bids, asks, and spread."""
        wal_file = tmp_path / "lob_reconstruct.wal"
        engine = PersistedMatchingEngine(wal_path=wal_file)

        # Place resting bids and asks
        engine.process_order(Order(order_id="B1", side=OrderSide.BUY, price=99.0, quantity=100, timestamp=10))
        engine.process_order(Order(order_id="B2", side=OrderSide.BUY, price=98.5, quantity=200, timestamp=20))
        engine.process_order(Order(order_id="A1", side=OrderSide.SELL, price=101.0, quantity=50, timestamp=30))
        engine.process_order(Order(order_id="A2", side=OrderSide.SELL, price=102.0, quantity=150, timestamp=40))

        assert engine.best_bid == 99.0
        assert engine.best_ask == 101.0
        assert engine.spread == 2.0
        assert engine.total_trades == 0
        engine.close()

        # Recover from crash
        recovery_mgr = RecoveryManager(wal_path=wal_file, strict=True)
        res: RecoveryResult = recovery_mgr.recover()

        assert res.recovered_orders_count == 4
        assert res.recovered_trades_count == 0
        assert res.engine.best_bid == 99.0
        assert res.engine.best_ask == 101.0
        assert res.engine.spread == 2.0
        assert len(res.engine.order_book.bids.depth()) == 2
        assert len(res.engine.order_book.asks.depth()) == 2

    def test_fifo_priority_preservation_on_replay(self, tmp_path: Path) -> None:
        """Verify time priority within the same price level is preserved after recovery."""
        wal_file = tmp_path / "fifo_reconstruct.wal"
        engine = PersistedMatchingEngine(wal_path=wal_file)

        # Three orders at identical price 100.0 in order O1, O2, O3
        engine.process_order(Order(order_id="O1", side=OrderSide.BUY, price=100.0, quantity=10, timestamp=100))
        engine.process_order(Order(order_id="O2", side=OrderSide.BUY, price=100.0, quantity=20, timestamp=200))
        engine.process_order(Order(order_id="O3", side=OrderSide.BUY, price=100.0, quantity=30, timestamp=300))
        engine.close()

        # Reconstruct engine
        recovered_engine = RecoveryManager(wal_path=wal_file, strict=True).recover().engine

        # Match an incoming sell order of quantity 15
        trades = recovered_engine.process_order(Order(order_id="SELL-TAK", side=OrderSide.SELL, price=100.0, quantity=15, timestamp=400))

        # First trade MUST consume O1 fully (10), second trade MUST consume 5 from O2 (FIFO)
        assert len(trades) == 2
        assert trades[0].resting_order_id == "O1"
        assert trades[0].quantity == 10
        assert trades[1].resting_order_id == "O2"
        assert trades[1].quantity == 5

        # Remaining in book: O2 with 15, O3 with 30
        assert recovered_engine.order_book.get_order("O2").quantity == 15
        assert recovered_engine.order_book.get_order("O3").quantity == 30

    def test_cancellation_and_partial_fill_reconstruction(self, tmp_path: Path) -> None:
        """Verify partial fills and order cancellations are reconstructed accurately."""
        wal_file = tmp_path / "cancel_reconstruct.wal"
        engine = PersistedMatchingEngine(wal_path=wal_file)

        engine.process_order(Order(order_id="B1", side=OrderSide.BUY, price=100.0, quantity=100, timestamp=1))
        engine.process_order(Order(order_id="B2", side=OrderSide.BUY, price=99.0, quantity=50, timestamp=2))
        # Partially fill B1 with 40
        engine.process_order(Order(order_id="S1", side=OrderSide.SELL, price=100.0, quantity=40, timestamp=3))
        # Cancel B2
        engine.cancel_order("B2")
        engine.close()

        # Recover
        res = RecoveryManager(wal_path=wal_file).recover()
        assert res.recovered_orders_count == 3
        assert res.recovered_cancels_count == 1
        assert res.recovered_trades_count == 1

        rec_engine = res.engine
        assert "B2" not in rec_engine.order_book
        assert rec_engine.best_bid == 100.0
        assert rec_engine.order_book.get_order("B1").quantity == 60


# ==============================================================================
# 4. Restart Recovery & Continued Matching
# ==============================================================================


class TestRestartRecovery:
    """Verify seamless engine restart, recovery, and post-restart matching."""

    def test_restart_recovery_and_continued_matching(self, tmp_path: Path) -> None:
        """Verify an engine can be restarted, recovered, and continue matching without trade ID reset."""
        wal_file = tmp_path / "restart.wal"

        # Session 1: Place orders and execute 1 trade
        engine1 = PersistedMatchingEngine(wal_path=wal_file)
        engine1.process_order(Order(order_id="BUY-1", side=OrderSide.BUY, price=150.0, quantity=100, timestamp=10))
        t1 = engine1.process_order(Order(order_id="SELL-1", side=OrderSide.SELL, price=150.0, quantity=40, timestamp=20))
        assert len(t1) == 1
        assert t1[0].trade_id == 1
        engine1.close()

        # Session 2: Recover from crash and resume matching
        recovery_mgr = RecoveryManager(wal_path=wal_file)
        res = recovery_mgr.recover()
        assert res.recovered_trades_count == 1

        engine2 = PersistedMatchingEngine(engine=res.engine, wal_path=wal_file)
        # Verify resting order has 60 remaining
        assert engine2.order_book.get_order("BUY-1").quantity == 60

        # Execute another trade against the recovered resting order
        t2 = engine2.process_order(Order(order_id="SELL-2", side=OrderSide.SELL, price=150.0, quantity=25, timestamp=30))
        assert len(t2) == 1
        # Trade ID MUST continue from 2 (no reset to 1)
        assert t2[0].trade_id == 2
        assert t2[0].resting_order_id == "BUY-1"
        assert t2[0].quantity == 25

        assert engine2.order_book.get_order("BUY-1").quantity == 35
        engine2.close()

        # Session 3: Re-recover and verify both trades are preserved
        res3 = RecoveryManager(wal_path=wal_file).recover()
        assert res3.recovered_trades_count == 2
        assert [t.trade_id for t in res3.recovered_trades] == [1, 2]
        assert res3.engine.order_book.get_order("BUY-1").quantity == 35


# ==============================================================================
# 5. Truncated Final Record Handling
# ==============================================================================


class TestTruncatedFinalRecord:
    """Verify resilience against simulated crash mid-write resulting in truncated trailing bytes."""

    def test_truncated_header_recovery_and_warning(self, tmp_path: Path) -> None:
        """Verify incomplete trailing record header (< 24 bytes) is safely handled."""
        wal_file = tmp_path / "trunc_header.wal"

        # Append 3 valid orders
        with WriteAheadLog(wal_file) as wal:
            wal.append_order(Order(order_id="O1", side=OrderSide.BUY, price=100, quantity=10, timestamp=1))
            wal.append_order(Order(order_id="O2", side=OrderSide.BUY, price=101, quantity=10, timestamp=2))
            wal.append_order(Order(order_id="O3", side=OrderSide.BUY, price=102, quantity=10, timestamp=3))

        # Append 11 trailing garbage bytes (simulating crash during frame header write)
        with open(wal_file, "ab") as f:
            f.write(b"\xaa" * 11)

        # Strict recovery should raise WALTruncatedRecordError
        strict_mgr = RecoveryManager(wal_path=wal_file, strict=True)
        with pytest.raises(WALTruncatedRecordError):
            strict_mgr.recover()

        # Non-strict / recovery mode should safely recover all 3 valid records and report warning
        mgr = RecoveryManager(wal_path=wal_file, strict=False)
        res = mgr.recover()
        assert res.recovered_orders_count == 3
        assert res.truncated_bytes_skipped == 11
        assert len(res.warnings) == 1
        assert "Incomplete trailing record header" in res.warnings[0]

    def test_truncated_payload_recovery(self, tmp_path: Path) -> None:
        """Verify crash mid-payload (header intact, payload incomplete) is safely handled."""
        wal_file = tmp_path / "trunc_payload.wal"

        with WriteAheadLog(wal_file) as wal:
            wal.append_order(Order(order_id="O1", side=OrderSide.BUY, price=100, quantity=10, timestamp=1))

        # Write valid 24-byte record header claiming 64 bytes payload, but write only 15 bytes
        fake_header = struct.pack(
            "<4sQBB2sII",
            b"CWAL",
            2,  # seq
            1,  # ORDER
            0,  # flags
            b"\x00\x00",
            64,  # payload length
            123456,  # checksum
        )
        with open(wal_file, "ab") as f:
            f.write(fake_header)
            f.write(b"\x00" * 15)  # only 15 bytes written before crash

        mgr = RecoveryManager(wal_path=wal_file, strict=False)
        res = mgr.recover()
        assert res.recovered_orders_count == 1
        assert res.truncated_bytes_skipped == 24 + 15
        assert any("Truncated payload" in w for w in res.warnings)

    def test_repair_wal_truncates_to_valid_boundary(self, tmp_path: Path) -> None:
        """Verify repair_wal truncates trailing incomplete records so new writes can proceed cleanly."""
        wal_file = tmp_path / "repair.wal"

        with WriteAheadLog(wal_file) as wal:
            wal.append_order(Order(order_id="O1", side=OrderSide.BUY, price=100, quantity=10, timestamp=1))
            valid_size = wal.filepath.stat().st_size

        # Append 15 bytes garbage
        with open(wal_file, "ab") as f:
            f.write(b"CORRUPTED_BYTES")

        assert wal_file.stat().st_size == valid_size + 15

        # Repair log
        mgr = RecoveryManager(wal_path=wal_file)
        repaired_count = mgr.repair_wal()
        assert repaired_count == 1
        assert wal_file.stat().st_size == valid_size

        # Append new records to repaired log
        with WriteAheadLog(wal_file) as wal:
            lsn = wal.append_order(Order(order_id="O2", side=OrderSide.BUY, price=105, quantity=10, timestamp=2))
            assert lsn == 2

        # Verify both records are now intact and clean
        with WriteAheadLog(wal_file, create=False) as wal:
            records, warnings, _ = wal.read_records(strict=True)
            assert len(warnings) == 0
            assert len(records) == 2


# ==============================================================================
# 6. Corrupted Record Handling
# ==============================================================================


class TestCorruptedRecordHandling:
    """Verify CRC32 checksum validation and corrupt record detection."""

    def test_checksum_mismatch_detected(self, tmp_path: Path) -> None:
        """Verify bit flip in payload is detected via CRC32 checksum mismatch."""
        wal_file = tmp_path / "corrupt_crc.wal"

        with WriteAheadLog(wal_file) as wal:
            wal.append_order(Order(order_id="O1", side=OrderSide.BUY, price=100, quantity=10, timestamp=1))
            wal.append_order(Order(order_id="O2", side=OrderSide.BUY, price=105, quantity=20, timestamp=2))

        # Corrupt one byte inside record 2 payload (offset = 32 header + 24 rec1_hdr + 64 rec1_payload + 24 rec2_hdr + 5)
        corrupt_offset = 32 + 24 + 64 + 24 + 5
        with open(wal_file, "r+b") as f:
            f.seek(corrupt_offset)
            original_byte = f.read(1)
            f.seek(corrupt_offset)
            f.write(bytes([original_byte[0] ^ 0xFF]))

        # Strict recovery raises WALCorruptedRecordError
        with pytest.raises(WALCorruptedRecordError) as exc_info:
            RecoveryManager(wal_path=wal_file, strict=True).recover()
        assert "checksum mismatch" in str(exc_info.value).lower()

        # Non-strict recovery halts before the corrupted record, safely recovering record 1
        res = RecoveryManager(wal_path=wal_file, strict=False).recover()
        assert res.recovered_orders_count == 1
        assert any("checksum mismatch" in w.lower() for w in res.warnings)

    def test_invalid_magic_bytes_detected(self, tmp_path: Path) -> None:
        """Verify invalid record frame magic bytes are detected."""
        wal_file = tmp_path / "bad_magic.wal"

        with WriteAheadLog(wal_file) as wal:
            wal.append_order(Order(order_id="O1", side=OrderSide.BUY, price=100, quantity=10, timestamp=1))

        # Tamper with record magic
        with open(wal_file, "r+b") as f:
            f.seek(32)  # record 1 header offset
            f.write(b"BADM")

        with pytest.raises(WALCorruptedRecordError):
            RecoveryManager(wal_path=wal_file, strict=True).recover()

    def test_corrupted_file_header_detected(self, tmp_path: Path) -> None:
        """Verify invalid file-level magic raises WALCorruptedRecordError on open."""
        wal_file = tmp_path / "bad_file_hdr.wal"
        with open(wal_file, "wb") as f:
            f.write(b"XXXX" + b"\x00" * 28)

        with pytest.raises(WALCorruptedRecordError) as exc:
            WriteAheadLog(wal_file, create=False)
        assert "Invalid WAL file magic" in str(exc.value)


# ==============================================================================
# 7. Duplicate Replay Protection
# ==============================================================================


class TestDuplicateReplayProtection:
    """Verify replay does not duplicate trades or produce conflicting trade IDs."""

    def test_replay_does_not_duplicate_persisted_trades(self, tmp_path: Path) -> None:
        """Verify that when WAL already contains TRADE records, replay does not duplicate them."""
        wal_file = tmp_path / "dedup.wal"
        engine = PersistedMatchingEngine(wal_path=wal_file)

        # Place matching orders: BUY 100 @ 100, SELL 100 @ 100 -> Generates Trade 1
        engine.process_order(Order(order_id="BUY-1", side=OrderSide.BUY, price=100.0, quantity=100, timestamp=1))
        trades = engine.process_order(Order(order_id="SELL-1", side=OrderSide.SELL, price=100.0, quantity=100, timestamp=2))
        assert len(trades) == 1
        assert trades[0].trade_id == 1
        engine.close()

        # Replay WAL
        res = RecoveryManager(wal_path=wal_file).recover()
        # Exactly 1 trade recovered (NOT 2)
        assert res.recovered_trades_count == 1
        assert res.uncommitted_trades_count == 0
        assert len(res.engine.trades) == 1
        assert res.engine.trades[0].trade_id == 1

    def test_simulated_crash_between_order_and_trade_write(self, tmp_path: Path) -> None:
        """Verify deterministic recovery when an acknowledged order crashed before trade record was written."""
        wal_file = tmp_path / "crash_before_trade.wal"

        # Manually write ORDER 1 and ORDER 2 to WAL WITHOUT writing the resulting TRADE record
        with WriteAheadLog(wal_file) as wal:
            wal.append_order(Order(order_id="B1", side=OrderSide.BUY, price=100.0, quantity=50, timestamp=1))
            # ORDER 2 is durable in WAL, but system crashes immediately before TRADE record is appended
            wal.append_order(Order(order_id="S1", side=OrderSide.SELL, price=100.0, quantity=50, timestamp=2))

        # Recovery should execute durable ORDER 2 against ORDER 1 deterministically and reconstruct the trade
        res = RecoveryManager(wal_path=wal_file).recover()
        assert res.recovered_orders_count == 2
        assert res.recovered_trades_count == 1
        assert res.uncommitted_trades_count == 1
        assert len(res.engine.trades) == 1
        assert res.engine.trades[0].resting_order_id == "B1"
        assert res.engine.trades[0].incoming_order_id == "S1"
        assert res.engine.trades[0].quantity == 50


# ==============================================================================
# 8. Recovery From an Empty Log
# ==============================================================================


class TestRecoveryFromEmptyLog:
    """Verify recovery handles empty files and missing files gracefully."""

    def test_recovery_from_zero_byte_file(self, tmp_path: Path) -> None:
        """Verify a 0-byte file returns a clean empty engine without raising."""
        wal_file = tmp_path / "zero_bytes.wal"
        wal_file.touch()

        res = RecoveryManager(wal_path=wal_file).recover()
        assert res.recovered_orders_count == 0
        assert res.recovered_trades_count == 0
        assert res.engine.total_trades == 0
        assert len(res.engine.order_book.bids.depth()) == 0
        assert len(res.engine.order_book.asks.depth()) == 0

    def test_recovery_from_header_only_file(self, tmp_path: Path) -> None:
        """Verify a WAL with only the 32-byte file header returns an empty engine."""
        wal_file = tmp_path / "header_only.wal"
        with WriteAheadLog(wal_file):
            pass  # writes header and closes

        res = RecoveryManager(wal_path=wal_file, strict=True).recover()
        assert res.recovered_orders_count == 0
        assert res.recovered_trades_count == 0
        assert res.is_clean

    def test_recovery_from_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Verify non-existent WAL file raises FileNotFoundError."""
        wal_file = tmp_path / "nonexistent.wal"
        with pytest.raises(FileNotFoundError):
            RecoveryManager(wal_path=wal_file).recover()


# ==============================================================================
# 9. Durability Policies & Async Worker
# ==============================================================================


class TestDurabilityPoliciesAndAsyncWorker:
    """Verify durability policy behaviors and decoupled background persistence."""

    def test_immediate_sync_durability(self, tmp_path: Path) -> None:
        """Verify IMMEDIATE_SYNC forces fsync on every write."""
        wal_file = tmp_path / "sync.wal"
        with WriteAheadLog(wal_file, durability=DurabilityPolicy.IMMEDIATE_SYNC) as wal:
            wal.append_order(Order(order_id="O1", side=OrderSide.BUY, price=100, quantity=10, timestamp=1))
            # File size on disk should reflect written header + record
            assert wal_file.stat().st_size == 32 + 24 + 64

    def test_async_buffered_worker_off_path_persistence(self, tmp_path: Path) -> None:
        """Verify AsyncPersistenceWorker processes orders and trades asynchronously."""
        wal_file = tmp_path / "async.wal"
        trade_file = tmp_path / "async_trades.dat"

        wal = WriteAheadLog(wal_file, durability=DurabilityPolicy.ASYNC_BUFFERED)
        trade_store = TradeStore(trade_file)

        worker = AsyncPersistenceWorker(wal=wal, trade_store=trade_store, flush_interval_s=0.01)

        # Submit 100 orders and trades rapidly
        for i in range(100):
            worker.submit_order(Order(order_id=f"O{i}", side=OrderSide.BUY, price=100.0, quantity=10, timestamp=i + 1))
            worker.submit_trade(Trade(trade_id=i + 1, resting_order_id=f"O{i}", incoming_order_id="TAK", price=100.0, quantity=10, timestamp=i + 1))

        # Flush worker to disk
        worker.flush()
        worker.close()
        wal.close()
        trade_store.close()

        # Verify all 200 records reached disk
        with WriteAheadLog(wal_file, create=False) as reader:
            records, _, _ = reader.read_records(strict=True)
            assert len(records) == 200

        # Verify all 100 trades persisted in TradeStore
        loaded_store = TradeStore(trade_file)
        assert loaded_store.count() == 100
        assert loaded_store.total_volume() == 1000
        loaded_store.close()


# ==============================================================================
# 10. Persistence Failures and Clean Resource Shutdown
# ==============================================================================


class TestPersistenceFailuresAndShutdown:
    """Verify failure modes, closed log guards, and clean resource cleanup."""

    def test_operations_on_closed_wal_raise(self, tmp_path: Path) -> None:
        """Verify mutating operations on a closed WAL raise WALClosedError."""
        wal_file = tmp_path / "closed.wal"
        wal = WriteAheadLog(wal_file)
        wal.close()

        order = Order(order_id="O1", side=OrderSide.BUY, price=100, quantity=10, timestamp=1)
        with pytest.raises(WALClosedError):
            wal.append_order(order)

        with pytest.raises(WALClosedError):
            wal.append_cancel("O1")

        with pytest.raises(WALClosedError):
            wal.append_trade(Trade(trade_id=1, resting_order_id="M", incoming_order_id="T", price=100, quantity=1, timestamp=1))

        with pytest.raises(WALClosedError):
            wal.flush()

        with pytest.raises(WALClosedError):
            wal.sync()

    def test_wal_close_is_idempotent(self, tmp_path: Path) -> None:
        """Verify calling close() multiple times is safe and idempotent."""
        wal_file = tmp_path / "idempotent.wal"
        wal = WriteAheadLog(wal_file)
        wal.close()
        wal.close()
        wal.close()
        assert wal.is_closed

    def test_io_write_failure_raises_wal_write_error(self, tmp_path: Path) -> None:
        """Verify simulated I/O failure raises WALWriteError without unhandled crash."""
        wal_file = tmp_path / "io_fail.wal"
        wal = WriteAheadLog(wal_file)

        with patch.object(wal._file, "write", side_effect=OSError("Disk full")):
            with pytest.raises(WALWriteError) as exc:
                wal.append_order(Order(order_id="O1", side=OrderSide.BUY, price=100, quantity=10, timestamp=1))
            assert "Disk full" in str(exc.value)

        wal.close()

    def test_trade_store_write_failure_preserves_atomic_consistency(self, tmp_path: Path) -> None:
        """Verify simulated I/O failure during TradeStore append raises PersistenceError without dirty in-memory state."""
        trade_file = tmp_path / "trade_io_fail.dat"
        t1 = Trade(trade_id=1, resting_order_id="M1", incoming_order_id="T1", price=100.0, quantity=10, timestamp=100)
        t2_failing = Trade(trade_id=2, resting_order_id="M2", incoming_order_id="T2", price=105.0, quantity=20, timestamp=200)
        t3_succeeding = Trade(trade_id=3, resting_order_id="M3", incoming_order_id="T3", price=110.0, quantity=30, timestamp=300)

        store = TradeStore(filepath=trade_file)
        store.append_trade(t1)
        assert store.count() == 1
        assert store.total_volume() == 10
        assert trade_file.stat().st_size == 1 * TRADE_PAYLOAD_SIZE

        # Simulate disk full on second trade write
        with patch.object(store._file, "write", side_effect=OSError("Disk full")):
            with pytest.raises(PersistenceError, match="Failed to append trade to disk"):
                store.append_trade(t2_failing)

        # In-memory state MUST remain clean and uncorrupted
        assert store.count() == 1
        assert store.total_volume() == 10
        assert store.get_trade(2) is None
        assert store.get_trades_for_order("M2") == []
        assert trade_file.stat().st_size == 1 * TRADE_PAYLOAD_SIZE

        # Subsequent valid trade append succeeds cleanly
        store.append_trade(t3_succeeding)
        assert store.count() == 2
        assert store.total_volume() == 40
        assert store.get_trade(3) == t3_succeeding
        assert trade_file.stat().st_size == 2 * TRADE_PAYLOAD_SIZE
        store.close()

        # Reopen from disk and verify exact data integrity
        with TradeStore(filepath=trade_file) as reloaded:
            assert reloaded.count() == 2
            assert reloaded.total_volume() == 40
            assert reloaded.get_trade(1) == t1
            assert reloaded.get_trade(3) == t3_succeeding

    def test_trade_store_flush_and_sync_failures_surfaced(self, tmp_path: Path) -> None:
        """Verify simulated flush and fsync errors raise PersistenceError explicitly."""
        trade_file = tmp_path / "sync_fail.dat"
        store = TradeStore(filepath=trade_file)
        t1 = Trade(trade_id=1, resting_order_id="M1", incoming_order_id="T1", price=100.0, quantity=10, timestamp=100)
        store.append_trade(t1)

        with patch.object(store._file, "flush", side_effect=OSError("Flush I/O error")):
            with pytest.raises(PersistenceError, match="Failed to flush trade store"):
                store.flush()

        with patch("os.fsync", side_effect=OSError("Sync I/O error")):
            with pytest.raises(PersistenceError, match="Failed to sync trade store"):
                store.sync()

        store.close()

    def test_operations_on_closed_trade_store_raise(self, tmp_path: Path) -> None:
        """Verify mutating or flushing operations on a closed TradeStore raise PersistenceError."""
        trade_file = tmp_path / "closed_store.dat"
        store = TradeStore(filepath=trade_file)
        store.close()
        assert store.is_closed

        t = Trade(trade_id=1, resting_order_id="M1", incoming_order_id="T1", price=100.0, quantity=10, timestamp=100)
        with pytest.raises(PersistenceError, match="TradeStore is closed"):
            store.append_trade(t)

        with pytest.raises(PersistenceError, match="TradeStore is closed"):
            store.append_trades([t])

        with pytest.raises(PersistenceError, match="TradeStore is closed"):
            store.flush()

        with pytest.raises(PersistenceError, match="TradeStore is closed"):
            store.sync()

    def test_trade_store_close_is_idempotent(self, tmp_path: Path) -> None:
        """Verify calling close() multiple times on TradeStore is safe and idempotent."""
        trade_file = tmp_path / "idempotent_store.dat"
        store = TradeStore(filepath=trade_file)
        store.close()
        store.close()
        store.close()
        assert store.is_closed

    def test_async_worker_surfaces_write_failure_without_deadlock(self, tmp_path: Path) -> None:
        """Verify AsyncPersistenceWorker surfaces write failure explicitly and does not hang on flush/close."""
        wal_file = tmp_path / "async_fail.wal"
        trade_file = tmp_path / "async_fail_trades.dat"

        wal = WriteAheadLog(wal_file, durability=DurabilityPolicy.ASYNC_BUFFERED)
        trade_store = TradeStore(trade_file)

        worker = AsyncPersistenceWorker(wal=wal, trade_store=trade_store, flush_interval_s=0.01)

        # Force write failure on the underlying WAL
        with patch.object(wal._file, "write", side_effect=OSError("Disk failure in worker")):
            trade = Trade(trade_id=1, resting_order_id="M1", incoming_order_id="T1", price=100.0, quantity=10, timestamp=100)
            worker.submit_trade(trade)

            # flush() must unblock and raise PersistenceError
            with pytest.raises(PersistenceError, match="Async persistence worker background failure"):
                worker.flush()

            # Subsequent submit must also raise
            with pytest.raises(PersistenceError):
                worker.submit_trade(trade)

            # close() should safely shut down without hanging
            with pytest.raises(PersistenceError):
                worker.close()

        wal.close()
        trade_store.close()

    def test_async_worker_normal_shutdown_flushes_pending_writes(self, tmp_path: Path) -> None:
        """Verify AsyncPersistenceWorker.close() cleanly flushes all pending records before stopping."""
        wal_file = tmp_path / "async_shutdown.wal"
        trade_file = tmp_path / "async_shutdown_trades.dat"

        wal = WriteAheadLog(wal_file, durability=DurabilityPolicy.ASYNC_BUFFERED)
        trade_store = TradeStore(trade_file)

        worker = AsyncPersistenceWorker(wal=wal, trade_store=trade_store, flush_interval_s=0.01)

        trades = [
            Trade(trade_id=i, resting_order_id=f"M{i}", incoming_order_id=f"T{i}", price=100.0 + i, quantity=10, timestamp=i * 100)
            for i in range(1, 21)
        ]

        for t in trades:
            worker.submit_trade(t)

        # Immediate close: must drain queue, flush to disk, and join worker thread cleanly
        worker.close()
        assert worker._is_closed
        wal.close()
        trade_store.close()

        # Reopen TradeStore from disk and verify all 20 trades were committed
        with TradeStore(trade_file) as reader:
            assert reader.count() == 20
            assert reader.total_volume() == 200
            for t in trades:
                assert reader.get_trade(t.trade_id) == t

        # Submissions after close must fail
        with pytest.raises(PersistenceError, match="AsyncPersistenceWorker is closed"):
            worker.submit_trade(trades[0])

    def test_persisted_matching_engine_closed_guards_and_clean_shutdown(self, tmp_path: Path) -> None:
        """Verify PersistedMatchingEngine guards closed state and flushes cleanly on shutdown."""
        wal_file = tmp_path / "pme_shutdown.wal"
        trade_file = tmp_path / "pme_shutdown_trades.dat"

        store = TradeStore(trade_file)
        engine = PersistedMatchingEngine(wal_path=wal_file, trade_store=store)

        o1 = Order(order_id="BUY-1", side=OrderSide.BUY, price=100.0, quantity=10, timestamp=100)
        o2 = Order(order_id="SELL-1", side=OrderSide.SELL, price=100.0, quantity=10, timestamp=200)

        trades1 = engine.process_order(o1)
        assert len(trades1) == 0
        trades2 = engine.process_order(o2)
        assert len(trades2) == 1

        engine.close()
        assert engine.is_closed

        # Idempotent close
        engine.close()

        # Mutating operations on closed engine raise PersistenceError
        with pytest.raises(PersistenceError, match="PersistedMatchingEngine is closed"):
            engine.process_order(Order(order_id="BUY-2", side=OrderSide.BUY, price=100.0, quantity=5, timestamp=300))

        with pytest.raises(PersistenceError, match="PersistedMatchingEngine is closed"):
            engine.cancel_order("BUY-1")

        with pytest.raises(PersistenceError, match="PersistedMatchingEngine is closed"):
            engine.flush()

        with pytest.raises(PersistenceError, match="PersistedMatchingEngine is closed"):
            engine.sync()

        # Verify disk files were cleanly committed and can be reloaded
        with TradeStore(trade_file) as reader:
            assert reader.count() == 1
            assert reader.get_trade(trades2[0].trade_id) == trades2[0]


# ==============================================================================
# 11. End-to-End TradingPipeline Integration with Persistence
# ==============================================================================


class TestPipelineIntegrationWithPersistence:
    """Verify end-to-end TradingPipeline with integrated WAL persistence and recovery."""

    def test_pipeline_process_order_and_cancel_persisted(self, tmp_path: Path) -> None:
        """Verify single-step pipeline execution and order cancellation are persisted in WAL."""
        wal_file = tmp_path / "pipeline.wal"

        pipeline = create_pipeline(wal_path=wal_file)
        assert pipeline.wal is not None

        # Process orders: resting bid + matching ask
        pipeline.process_order(Order(order_id="PB-1", side=OrderSide.BUY, price=100.0, quantity=50, timestamp=1))
        pipeline.process_order(Order(order_id="PB-2", side=OrderSide.BUY, price=95.0, quantity=20, timestamp=2))
        trades = pipeline.process_order(Order(order_id="PS-1", side=OrderSide.SELL, price=100.0, quantity=30, timestamp=3))

        assert len(trades) == 1
        assert trades[0].quantity == 30
        assert pipeline.total_trades == 1

        # Cancel remaining PB-2
        cancelled = pipeline.cancel_order("PB-2")
        assert cancelled.order_id == "PB-2"

        pipeline.close()

        # Read back WAL records directly
        with WriteAheadLog(wal_file, create=False) as reader:
            records, warnings, _ = reader.read_records(strict=True)
            # Records: ORDER PB-1, ORDER PB-2, ORDER PS-1, TRADE 1, CANCEL PB-2
            assert len(records) == 5
            types = [r.record_type for r in records]
            assert types == [
                WALRecordType.ORDER,
                WALRecordType.ORDER,
                WALRecordType.ORDER,
                WALRecordType.TRADE,
                WALRecordType.CANCEL,
            ]

    def test_pipeline_crash_and_recover_e2e(self, tmp_path: Path) -> None:
        """Verify full pipeline crash simulation, recovery, and post-recovery continuity."""
        wal_file = tmp_path / "pipe_crash.wal"

        # Session 1: Pipeline runs steps generating trades and resting orders
        pipe1 = create_pipeline(wal_path=wal_file)
        pipe1.process_order(Order(order_id="B1", side=OrderSide.BUY, price=150.0, quantity=100, timestamp=1))
        pipe1.process_order(Order(order_id="B2", side=OrderSide.BUY, price=148.0, quantity=50, timestamp=2))
        pipe1.process_order(Order(order_id="S1", side=OrderSide.SELL, price=150.0, quantity=40, timestamp=3))
        assert pipe1.total_trades == 1
        assert pipe1.best_bid == 150.0
        pipe1.close()

        # Session 2: Recover pipeline via TradingPipeline.recover
        pipe2 = TradingPipeline.recover(wal_path=wal_file)
        assert pipe2.total_trades == 1
        assert pipe2.best_bid == 150.0
        assert pipe2.engine.order_book.get_order("B1").quantity == 60
        assert pipe2.engine.order_book.get_order("B2").quantity == 50

        # Continue execution on recovered pipeline
        t2 = pipe2.process_order(Order(order_id="S2", side=OrderSide.SELL, price=150.0, quantity=60, timestamp=4))
        assert len(t2) == 1
        assert t2[0].trade_id == 2
        assert t2[0].quantity == 60
        # B1 is now fully consumed; best bid drops to B2 (148.0)
        assert pipe2.best_bid == 148.0
        assert "B1" not in pipe2.engine.order_book

        pipe2.close()


# ==============================================================================
# 12. Trade History Recovery and Restart Validation
# ==============================================================================


class TestTradeHistoryRecoveryAndRestartValidation:
    """Validate safe trade-history recovery after application restarts without book pollution."""

    def test_recover_trade_history_reopen_storage(self, tmp_path: Path) -> None:
        """Verify trade history recovery after creating, closing, and reopening TradeStore."""
        trade_file = tmp_path / "hist_reopen.dat"
        trades = [
            Trade(trade_id=1, resting_order_id="M1", incoming_order_id="T1", price=100.0, quantity=10, timestamp=100),
            Trade(trade_id=2, resting_order_id="M1", incoming_order_id="T2", price=100.5, quantity=20, timestamp=200),
            Trade(trade_id=3, resting_order_id="M2", incoming_order_id="T3", price=101.0, quantity=30, timestamp=300),
        ]

        # Session 1: Create and write trades
        with TradeStore(trade_file) as store1:
            store1.append_trades(trades)
            assert store1.count() == 3

        # Session 2: Reopen storage and run recovery
        with TradeStore(trade_file) as store2:
            res: TradeRecoveryResult = store2.recover_trade_history(strict=True)
            assert res.is_clean
            assert res.recovered_count == 3
            assert res.total_volume == 60
            assert res.truncated_bytes_skipped == 0
            assert res.corrupted_records_count == 0
            assert len(res.recovered_trades) == 3
            assert res.recovered_trades == trades

        # Session 3: Initialize PersistedMatchingEngine with recover_trades=True
        wal_file = tmp_path / "hist_reopen.wal"
        with TradeStore(trade_file) as store3:
            pme = PersistedMatchingEngine(
                wal_path=wal_file,
                trade_store=store3,
                recover_trades=True,
            )
            assert pme.total_trades == 3
            assert pme.total_volume_traded == 60
            assert pme.trades == trades
            pme.close()

    def test_recovery_preserves_all_supported_fields(self, tmp_path: Path) -> None:
        """Verify recovered trades faithfully preserve trade IDs, price types, quantities, and timestamps."""
        trade_file = tmp_path / "fields_preservation.dat"
        original_trades = [
            Trade(trade_id=101, resting_order_id="ORD-MAKER-01", incoming_order_id="ORD-TAKER-01", price=99.75, quantity=15, timestamp=111111111),
            Trade(trade_id="TRD-UUID-002", resting_order_id="ORD-MAKER-02", incoming_order_id="ORD-TAKER-02", price=250, quantity=40, timestamp=222222222),
            Trade(trade_id=303, resting_order_id="ORD-MAKER-03", incoming_order_id="ORD-TAKER-03", price=Decimal("175.12345678"), quantity=85, timestamp=333333333),
        ]

        with TradeStore(trade_file) as store:
            store.append_trades(original_trades)

        with TradeStore(trade_file) as reloaded_store:
            res = reloaded_store.recover_trade_history(strict=True)
            assert res.recovered_count == 3
            recovered = res.recovered_trades

            for orig, rec in zip(original_trades, recovered):
                # Trade ID preservation and exact type
                assert rec.trade_id == orig.trade_id
                assert type(rec.trade_id) is type(orig.trade_id)

                # Order identifiers and domain aliases
                assert rec.resting_order_id == orig.resting_order_id
                assert rec.maker_order_id == orig.maker_order_id
                assert rec.incoming_order_id == orig.incoming_order_id
                assert rec.taker_order_id == orig.taker_order_id

                # Execution price and exact numeric type
                assert rec.price == orig.price
                assert rec.execution_price == orig.execution_price
                assert type(rec.price) is type(orig.price)

                # Quantities and aliases
                assert rec.quantity == orig.quantity
                assert rec.executed_quantity == orig.executed_quantity
                assert rec.execution_qty == orig.execution_qty

                # Nanosecond timestamp
                assert rec.timestamp == orig.timestamp
                assert rec.timestamp_ns == orig.timestamp_ns

    def test_repeated_recovery_prevents_duplicate_records(self, tmp_path: Path) -> None:
        """Verify repeated recovery or retries do not duplicate trade records in memory or on disk."""
        trade_file = tmp_path / "repeat_recovery.dat"
        trades = [
            Trade(trade_id=1, resting_order_id="M1", incoming_order_id="T1", price=50.0, quantity=10, timestamp=100),
            Trade(trade_id=2, resting_order_id="M2", incoming_order_id="T2", price=55.0, quantity=20, timestamp=200),
        ]

        with TradeStore(trade_file) as store:
            store.append_trades(trades)
            assert store.count() == 2

            # Repeated recovery on the same store instance
            res1 = store.recover_trade_history(strict=True)
            res2 = store.recover_trade_history(strict=True)
            res3 = store.recover_trade_history(strict=True)

            assert store.count() == 2
            assert store.total_volume() == 30
            assert len(store.get_all_trades()) == 2
            assert res2.duplicate_records_skipped == 2
            assert res3.duplicate_records_skipped == 2

        # Engine level repeated recovery
        engine = MatchingEngine()
        loaded_count_1 = engine.load_trade_history(trades)
        loaded_count_2 = engine.load_trade_history(trades)
        loaded_count_3 = engine.load_trade_history(trades)

        assert loaded_count_1 == 2
        assert loaded_count_2 == 0
        assert loaded_count_3 == 0
        assert engine.total_trades == 2
        assert engine.total_volume_traded == 30
        assert len(engine.trades) == 2

    def test_recovery_incomplete_final_record_strict_and_salvage(self, tmp_path: Path) -> None:
        """Verify incomplete trailing records raise in strict mode and salvage cleanly in non-strict mode."""
        trade_file = tmp_path / "incomplete_final.dat"
        t1 = Trade(trade_id=1, resting_order_id="M1", incoming_order_id="T1", price=100.0, quantity=10, timestamp=100)
        t2 = Trade(trade_id=2, resting_order_id="M2", incoming_order_id="T2", price=102.0, quantity=20, timestamp=200)

        with TradeStore(trade_file) as store:
            store.append_trades([t1, t2])
        assert trade_file.stat().st_size == 2 * TRADE_PAYLOAD_SIZE

        # Simulate trailing crash leaving 53 partial bytes
        with open(trade_file, "ab") as f:
            f.write(b"PARTIAL_TRAILING_BYTES" * 2 + b"123456789")
        assert trade_file.stat().st_size == (2 * TRADE_PAYLOAD_SIZE) + 53

        # Strict recovery MUST explicitly raise PersistenceError on initialization and explicit recovery
        with pytest.raises(PersistenceError, match="Incomplete trailing record at offset 256"):
            TradeStore(trade_file, strict_recovery=True)

        with TradeStore(trade_file) as store_strict:
            with pytest.raises(PersistenceError, match="Incomplete trailing record at offset 256"):
                store_strict.recover_trade_history(strict=True)

        # Salvage mode MUST recover valid prefix, report warnings, and truncate trailing noise
        with TradeStore(trade_file) as store_salvage:
            res = store_salvage.recover_trade_history(strict=False, truncate_corrupted=True)
            assert not res.is_clean
            assert res.recovered_count == 2
            assert res.truncated_bytes_skipped == 53
            assert len(res.warnings) == 1
            assert "Incomplete trailing record" in res.warnings[0]
            assert store_salvage.get_all_trades() == [t1, t2]

        # File on disk must now be cleanly aligned back to 256 bytes
        assert trade_file.stat().st_size == 2 * TRADE_PAYLOAD_SIZE

    def test_recovery_corrupted_record_handling_strict_and_salvage(self, tmp_path: Path) -> None:
        """Verify corrupted records raise in strict mode and halt safely in salvage mode."""
        trade_file = tmp_path / "corrupted_record.dat"
        t1 = Trade(trade_id=1, resting_order_id="M1", incoming_order_id="T1", price=100.0, quantity=10, timestamp=100)
        t2 = Trade(trade_id=2, resting_order_id="M2", incoming_order_id="T2", price=105.0, quantity=20, timestamp=200)

        with TradeStore(trade_file) as store:
            store.append_trades([t1, t2])

        # Append a 128-byte block with corrupted payload (non-positive price ticks)
        corrupted_payload = bytearray(encode_trade_payload(t1))
        struct.pack_into("<q", corrupted_payload, 96, -99999)  # Corrupt price ticks to negative
        with open(trade_file, "ab") as f:
            f.write(bytes(corrupted_payload))

        assert trade_file.stat().st_size == 3 * TRADE_PAYLOAD_SIZE

        # Strict recovery MUST raise PersistenceError reporting corruption
        with pytest.raises(PersistenceError, match="Corrupted trade record at offset 256"):
            TradeStore(trade_file, strict_recovery=True)

        with TradeStore(trade_file) as store_strict:
            with pytest.raises(PersistenceError, match="Corrupted trade record at offset 256"):
                store_strict.recover_trade_history(strict=True)

        # Salvage mode halts at corruption boundary, recovers intact preceding trades, and reports diagnostics
        with TradeStore(trade_file) as store_salvage:
            res = store_salvage.recover_trade_history(strict=False, truncate_corrupted=True)
            assert res.recovered_count == 2
            assert res.corrupted_records_count == 1
            assert len(res.warnings) >= 1
            assert "Corrupted trade record" in res.warnings[0]
            assert store_salvage.get_all_trades() == [t1, t2]

        # Truncation trimmed corrupted block back to 256 bytes
        assert trade_file.stat().st_size == 2 * TRADE_PAYLOAD_SIZE

    def test_recovery_empty_and_missing_storage(self, tmp_path: Path) -> None:
        """Verify empty files recover cleanly and missing files are reported explicitly."""
        # 1. 0-byte storage file
        empty_file = tmp_path / "empty_trades.dat"
        empty_file.touch()

        with TradeStore(empty_file) as store:
            res = store.recover_trade_history(strict=True)
            assert res.is_clean
            assert res.recovered_count == 0
            assert res.total_volume == 0
            assert store.count() == 0

        # 2. Non-existent storage file with create=False raises FileNotFoundError
        missing_file = tmp_path / "non_existent.dat"
        with pytest.raises(FileNotFoundError, match="Trade storage file '.*' does not exist"):
            TradeStore(missing_file, create=False)

        # 3. Non-existent WAL file passed to TradeStore.from_wal raises FileNotFoundError
        missing_wal = tmp_path / "missing.wal"
        with pytest.raises(FileNotFoundError, match="WAL file '.*' does not exist"):
            TradeStore.from_wal(missing_wal)

    def test_recovery_errors_reported_clearly(self, tmp_path: Path) -> None:
        """Verify closed store operations, conflicting trades, and non-Trade inputs raise clear errors."""
        trade_file = tmp_path / "error_reporting.dat"
        store = TradeStore(trade_file)
        store.close()

        # Calling recovery on closed store
        with pytest.raises(PersistenceError, match="TradeStore is closed"):
            store.recover_trade_history()

        # Passing non-Trade objects to engine.load_trade_history
        engine = MatchingEngine()
        with pytest.raises(TypeError, match="Expected Trade instance, got str"):
            engine.load_trade_history(["not_a_trade"])  # type: ignore

        # Conflicting trade ID during load_trade_history
        t1 = Trade(trade_id=10, resting_order_id="M1", incoming_order_id="T1", price=100.0, quantity=10, timestamp=100)
        t1_conflict = Trade(trade_id=10, resting_order_id="M1", incoming_order_id="T1", price=999.0, quantity=10, timestamp=100)
        engine.load_trade_history([t1])
        with pytest.raises(DuplicateTradeError, match="conflicts with existing trade"):
            engine.load_trade_history([t1_conflict])

    def test_matching_engine_behavior_and_trade_ids_after_recovery(self, tmp_path: Path) -> None:
        """Verify matching engine preserves order book isolation and continues monotonic trade IDs after trade recovery."""
        trade_file = tmp_path / "continuity_trades.dat"
        wal_file = tmp_path / "continuity.wal"

        # Step 1: Pre-populate persistent trade store with 3 historical trades (IDs 1, 2, 3)
        historical_trades = [
            Trade(trade_id=1, resting_order_id="HIST-M1", incoming_order_id="HIST-T1", price=100.0, quantity=10, timestamp=10),
            Trade(trade_id=2, resting_order_id="HIST-M2", incoming_order_id="HIST-T2", price=101.0, quantity=20, timestamp=20),
            Trade(trade_id=3, resting_order_id="HIST-M3", incoming_order_id="HIST-T3", price=102.0, quantity=30, timestamp=30),
        ]
        with TradeStore(trade_file) as seed_store:
            seed_store.append_trades(historical_trades)

        # Step 2: Restart engine with trade recovery requested
        store = TradeStore(trade_file)
        pme = PersistedMatchingEngine(
            wal_path=wal_file,
            trade_store=store,
            recover_trades=True,
        )

        # Confirm trade history is recovered
        assert pme.total_trades == 3
        assert pme.total_volume_traded == 60
        assert [t.trade_id for t in pme.trades] == [1, 2, 3]

        # Crucial architectural boundary: Order book state MUST remain empty and unpolluted
        assert pme.order_book.best_bid is None
        assert pme.order_book.best_ask is None
        assert pme.spread is None

        # Step 3: Execute a new trade in the restarted engine
        # Maker order rests in book
        t_maker = pme.process_order(Order(order_id="NEW-BUY", side=OrderSide.BUY, price=150.0, quantity=50, timestamp=100))
        assert len(t_maker) == 0
        assert pme.order_book.best_bid == 150.0

        # Aggressive taker crosses book
        t_taker = pme.process_order(Order(order_id="NEW-SELL", side=OrderSide.SELL, price=150.0, quantity=50, timestamp=200))
        assert len(t_taker) == 1

        # The new trade MUST have trade_id = 4 (strictly monotonic, no collision with historical trades 1, 2, 3)
        new_trade = t_taker[0]
        assert new_trade.trade_id == 4
        assert new_trade.price == 150.0
        assert new_trade.quantity == 50

        # Engine totals now reflect historical + new executions
        assert pme.total_trades == 4
        assert pme.total_volume_traded == 110
        assert [t.trade_id for t in pme.trades] == [1, 2, 3, 4]

        pme.close()
        store.close()
