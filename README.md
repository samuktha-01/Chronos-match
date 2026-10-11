# ChronosMatch: Zero-Copy High-Frequency Trading Engine

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-4%20passed-brightgreen.svg)](tests/)
[![Architecture Status](https://img.shields.io/badge/status-foundation%20phase%201-orange.svg)](docs/architecture.md)

An in-memory, HFT-style simulated order-matching engine and low-latency systems testbed built with mechanical sympathy, zero-copy shared-memory IPC, lock-free ring buffers, and a deterministic limit order book.

> [!IMPORTANT]
> **Simulated Testbed Disclaimer**: ChronosMatch is strictly an educational, benchmarking, and research order-matching simulation engine. It is **not** a production trading system, broker gateway, or live market execution service. It does not connect to live financial exchanges or provide real-world regulatory, margin, or clearing guarantees.

---

## Table of Contents

- [Overview & Problem Statement](#overview--problem-statement)
- [Key Objectives](#key-objectives)
- [Architecture Pipeline](#architecture-pipeline)
- [Implementation Status & Roadmap](#implementation-status--roadmap)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation & Virtual Environment](#installation--virtual-environment)
  - [Running Tests](#running-tests)
- [Development Workflow](#development-workflow)
- [Performance Measurement Approach](#performance-measurement-approach)
- [Project Scope & Limitations](#project-scope--limitations)
- [Architecture Documentation](#architecture-documentation)
- [License](#license)

---

## Overview & Problem Statement

Modern financial trading systems demand deterministic execution and predictable microsecond-level latency profiles. However, standard financial trading simulations and prototype engines implemented in high-level managed runtimes (such as pure Python) encounter severe architectural bottlenecks:

1. **Garbage Collection & Object Allocation Jitter**: Instantiating high volumes of order objects, dictionaries, and tuples on the heap triggers garbage collection cycles, causing catastrophic tail-latency (p99/p99.9) spikes.
2. **Serialization Overhead**: Passing messages across process boundaries using conventional IPC protocols (JSON, Pickle, Protobuf over sockets) causes repeated memory copying and CPU serialization stalls.
3. **Lock Contention & Kernel Transitions**: Mutex-guarded queues and thread synchronization primitives introduce thread preemption, cache-line bouncing, and costly operating system context switches.
4. **Cache Unfriendly Memory Layouts**: Pointer-heavy tree data structures scatter order data across arbitrary memory addresses, incurring frequent CPU L1/L2/L3 cache misses.

**ChronosMatch** addresses these challenges from first principles by designing an HFT-inspired architecture founded on **mechanical sympathy**: cache-aligned memory layouts, zero-copy shared-memory IPC, lock-free single-producer single-consumer (SPSC) ring buffers, and a deterministic price-time priority limit order book executing in compiled Cython/C.

---

## Key Objectives

- **Zero-Copy Ingestion**: Decouple order generation from matching execution using memory-mapped shared-memory buffers and fixed-size 64-byte binary structs without intermediate serialization.
- **Lock-Free Concurrency**: Bridge thread and process boundaries using atomic memory fences (`acquire`/`release`) in tight polling loops, avoiding kernel context switches.
- **Deterministic Order Matching**: Maintain a Level 3 (L3) limit order book executing with strict Price-Time Priority (FIFO) semantics, ensuring replayable and bit-identical executions.
- **Off-Critical-Path Persistence & Telemetry**: Isolate disk logging and terminal dashboard rendering from the matching hot path via non-blocking asynchronous egress queues.
- **Hybrid Performance & Ergonomics**: Pair Python 3.12 developer ergonomics for simulation harnesses and diagnostic dashboards with Cython/C speed on the critical matching path.

---

## Architecture Pipeline

ChronosMatch organizes the order lifecycle into decoupled, cache-conscious processing stages:

```mermaid
flowchart LR
    A[Simulated Market Feed] -->|Fixed 64B Struct| B[Shared-Memory IPC]
    B -->|Zero-Copy Enqueue| C[Lock-Free Ring Buffer]
    C -->|Atomic SPSC Dequeue| D[Cython Matching Engine]
    D <-->|O(1) L3 Book Ops| E[Limit Order Book]
    D -->|Async Trade Events| F[Trade Persistence]
    D -->|Telemetry Ring Buffer| G[Terminal Dashboard]
```

```text
[Simulated Market Feed]
        │  (Fixed-size 64-byte binary structs, high-resolution timestamps)
        ▼
[Shared-Memory IPC]
        │  (OS memory-mapped region: Win32 Named Shared Memory / POSIX shm)
        ▼
[Lock-Free Ring Buffer]
        │  (SPSC circular queue with atomic acquire-release fences)
        ▼
[Cython Matching Engine]  ◄───►  [Limit Order Book (L3)]
        │                         (Price-Time Priority FIFO, contiguous price ladder)
        ├─────────────────────────────┐
        ▼                             ▼
[Trade Persistence (Async)]   [Terminal Dashboard]
(Append-only WAL log)         (Real-time L2 depth, trade tape, latency metrics)
```

### Component Summary

| Component | Responsibility | Interaction Pattern |
| :--- | :--- | :--- |
| **Simulated Market Feed** | Synthesizes realistic, bursty market order flow (New, Cancel, Modify, Limit, Market). | Writes fixed-size binary structs directly into shared memory with nanosecond hardware timestamps. |
| **Shared-Memory IPC** | Eliminates inter-process copying between market feed and matching core. | Memory-mapped shared memory segment accessed via direct pointers / typed memory views. |
| **Lock-Free Ring Buffer** | High-throughput, single-producer single-consumer circular queue. | Atomic memory fences (`memory_order_acquire`/`release`) with power-of-two masking and CPU core pinning. |
| **Cython Matching Engine** | Low-latency execution core executing risk checks and matching logic. | Bypasses Python GIL (`nogil`), matches incoming orders against the resting book, emits execution events. |
| **Limit Order Book (LOB)** | Level 3 (L3) order book maintaining bids (descending) and asks (ascending). | $O(1)$ price level lookup via indexed arrays; $O(1)$ queue insertion, deletion, and cancellation via intrusive lists. |
| **Trade Persistence** | Asynchronous audit logging and deterministic execution replay. | Consumes filled trade events from an off-path ring buffer; writes sequentially to append-only storage. |
| **Terminal Dashboard** | Real-time console visualization of market depth, BBO, throughput, and latency. | Samples telemetry metrics from a non-blocking shared buffer; renders without impacting the matching engine. |

For deep-dive specifications on memory layouts, data structures, and thread contracts, see [docs/architecture.md](file:///c:/Users/keert/OneDrive/Desktop/Chronosmatch-project1/Chronos-match/docs/architecture.md).

---

## Implementation Status & Roadmap

ChronosMatch adopts a staged, test-driven engineering approach. Below is the explicit distinction between currently implemented functionality and planned components:

| Module / Component | Phase | Status | Details |
| :--- | :--- | :--- | :--- |
| **Project Foundation & Packaging** | Phase 1 | **Implemented** | PEP 621 compliant [pyproject.toml](file:///c:/Users/keert/OneDrive/Desktop/Chronosmatch-project1/Chronos-match/pyproject.toml), package imports, versioning (`0.1.0`), automated pytest test suite. |
| **Environment Configuration** | Phase 1 | **Implemented** | [.env.example](file:///c:/Users/keert/OneDrive/Desktop/Chronosmatch-project1/Chronos-match/.env.example) configuration template with deployment settings. |
| **System Architecture Spec** | Phase 1 | **Implemented** | Comprehensive technical specification in [docs/architecture.md](file:///c:/Users/keert/OneDrive/Desktop/Chronosmatch-project1/Chronos-match/docs/architecture.md). |
| **Shared-Memory IPC & Message Schemas** | Phase 2 | *Planned* | 64-byte `OrderMessage` binary struct definitions, cross-platform shared memory (`mmap`). |
| **Lock-Free Ring Buffer** | Phase 3 | *Planned* | SPSC/MPSC circular ring buffer with atomic memory barriers and power-of-two bitmask indexing. |
| **Cython LOB & Matching Core** | Phase 4 | *Planned* | L3 price-time priority order book, intrusive order linked lists, compiled C/Cython execution loop. |
| **Market Feed & Trade Persistence** | Phase 5 | *Planned* | Poisson/burst order feed generator, off-path append-only write-ahead log (WAL) for trade persistence. |
| **Terminal Dashboard & Telemetry** | Phase 6 | *Planned* | Terminal UI (Rich/Textual), tick-to-trade latency histogram (p50/p90/p99/p99.9), benchmarking harness. |

---

## Technology Stack

The technology stack matches the actual repository structure and its planned extensions:

- **Core Runtime**: [Python 3.12+](https://www.python.org/)
- **Packaging & Build**: PEP 621 / PEP 517 standard using `setuptools>=61.0` and `pyproject.toml`
- **Testing Framework**: [pytest 8.0+](https://docs.pytest.org/) configured with `addopts = "-ra -q"` and `pythonpath = ["src"]`
- **Planned Performance Core**: [Cython](https://cython.org/) / C extension modules for lock-free ring buffers and order book hot paths
- **Planned Visualization & Diagnostics**: Terminal UI built with Rich/Textual for real-time market depth and latency telemetry

---

## Repository Structure

```text
Chronos-match/
├── .env.example              # Template environment configuration
├── .gitignore                 # Exclusion rules for caches, builds, and venvs
├── pyproject.toml             # PEP 621 metadata, dependencies, and pytest configuration
├── requirements.txt           # Development dependencies (-e . and pytest)
├── README.md                  # Project overview, architecture, and quickstart guide
├── docs/
│   └── architecture.md        # Comprehensive technical architecture and component specs
├── src/
│   └── chronosmatch/          # Core Python package source tree
│       └── __init__.py        # Package versioning (0.1.0) and public API exports
└── tests/
    └── test_project_setup.py  # Automated tests verifying package structure and imports
```

---

## Getting Started

### Prerequisites

- **Python**: Version 3.12 or higher
- **Git**: For version control
- **C/C++ Compiler** *(for future Cython phases)*: MSVC (Windows) or GCC/Clang (Linux/macOS)

### Installation & Virtual Environment

1. **Clone and enter the repository**:
   ```bash
   git clone https://github.com/samuktha-01/Chronos-match.git
   cd Chronos-match
   ```

2. **Create a Python 3.12 virtual environment**:
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - **Windows (Command Prompt)**:
     ```cmd
     .\.venv\Scripts\activate.bat
     ```
   - **Linux / macOS (Bash / Zsh)**:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies in editable mode**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Alternatively: `pip install -e .[dev]`)*

5. **Configure environment settings**:
   - **Windows (PowerShell)**:
     ```powershell
     Copy-Item .env.example .env
     ```
   - **Linux / macOS**:
     ```bash
     cp .env.example .env
     ```

### Running Tests

Execute the automated test suite using `pytest`:

```bash
python -m pytest -v
```

**Actual Verified Output**:
```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.6.0
rootdir: ...\Chronos-match
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.12.1, asyncio-1.4.0
collected 4 items

tests\test_project_setup.py::test_package_import PASSED                   [ 25%]
tests\test_project_setup.py::test_package_version PASSED                  [ 50%]
tests\test_project_setup.py::test_package_exports PASSED                  [ 75%]
tests\test_project_setup.py::test_package_docstring PASSED                [100%]

============================== 4 passed in 0.06s ==============================
```

---

## Development Workflow

1. **Test-Driven Development (TDD)**:
   - All new features (ring buffers, message parsers, matching logic) are accompanied by unit tests located in [tests/](file:///c:/Users/keert/OneDrive/Desktop/Chronosmatch-project1/Chronos-match/tests/).
   - Run tests frequently during development: `python -m pytest -v`.

2. **Code Quality & Formatting**:
   - Keep codebase compliant with PEP 8 and Python 3.12 type annotations.
   - Preserve non-allocating design patterns in performance-sensitive modules.

3. **Cython Compilation Cycle** *(Planned for Phase 4)*:
   - When Cython modules are introduced, rebuild extensions in-place:
     ```bash
     python setup.py build_ext --inplace
     ```

---

## Performance Measurement Approach

ChronosMatch measures execution speed using hardware-accurate timing rather than high-level wall-clock profilers:

1. **Hardware-Accurate Timestamping**:
   - Order events are stamped with nanosecond CPU timestamps (`rdtsc` instruction on x86_64 or `time.perf_counter_ns()`) at distinct pipeline checkpoints:
     - $T_0$: Market Feed generation timestamp
     - $T_1$: Shared-memory ingestion timestamp
     - $T_2$: Ring buffer dequeue timestamp
     - $T_3$: Order book matching and fill execution timestamp
     - $T_4$: Async persistence egress timestamp
2. **Key Latency Metrics**:
   - **Tick-to-Trade Latency ($T_3 - T_0$)**: Total end-to-end processing transit time.
   - **Core Matching Latency ($T_3 - T_2$)**: Pure L3 limit order book execution duration.
   - **Queue Dwell Latency ($T_2 - T_1$)**: Time spent traversing the lock-free ring buffer.
3. **Statistical Latency Profiling**:
   - Latency distributions are reported via percentiles: **p50 (median)**, **p90**, **p99**, **p99.9**, and **max jitter**.
   - Avoiding averages: Average latency masks microsecond tail spikes caused by cache misses or page faults.
4. **Throughput & Zero-Allocation Auditing**:
   - Benchmarking sustained throughput under synthetic burst rates (target: $10^5$ to $10^6$ orders/second).
   - Zero-allocation validation: Ensuring 0 Python heap allocations occur on the hot matching path during sustained execution.

---

## Project Scope & Limitations

To maintain clear engineering expectations, ChronosMatch operates within defined boundaries:

- **HFT-Style Simulation, Not Live Execution**: ChronosMatch is designed for educational exploration and architecture benchmarking. It does not interface with real exchange gateways (e.g. FIX, ITCH, OUCH, Nasdaq/CME protocols) or handle live capital.
- **Single-Host Architecture**: Shared-memory IPC is optimized for low-latency multicore communication within a single physical workstation or server. It is not a distributed cluster or networked messaging broker.
- **Platform Specifics**: High-resolution shared memory (`mmap`) and thread affinity primitives vary between operating systems (Windows Win32 API vs Linux POSIX `shm_open` and `sched_setaffinity`). Production HFT systems traditionally target Linux bare-metal.
- **Financial Scope**: Focuses strictly on core limit and market orders under Price-Time Priority. Complex institutional order types (icebergs, pegged orders, conditional triggers) and exchange auction uncrossing logic are out of scope for early phases.

---

## Architecture Documentation

For complete technical specifications—including detailed ASCII and Mermaid component flowcharts, 64-byte binary struct layouts, lock-free ring buffer synchronization mechanics, and intrusive order book algorithms—please consult:

👉 **[docs/architecture.md](file:///c:/Users/keert/OneDrive/Desktop/Chronosmatch-project1/Chronos-match/docs/architecture.md)**

---

## License

This project is open-source under the [MIT License](LICENSE).
