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
    TradeStore,
    WALClosedError,
    WALCorruptedRecordError,
    WALError,
    WALRecord,
    WALRecordType,
    WALTruncatedRecordError,
    WALWriteError,
    WriteAheadLog,
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
