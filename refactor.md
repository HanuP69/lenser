# 🏗️ LectureLens Radical Lean Architecture Blueprint (`refactor.md`)
### Modern SDE Reference Architecture: College OS + DBMS + Aryan Mittal LLD + LangGraph + High Recall

> **Goal**: Refactor the codebase from **31,915 LoC down to a radical, textbook ~2,500–3,500 total LoC** (~88–90% reduction).  
> **Key Strategy**: **Zero Wheel-Reinvention.** Use battle-tested industry libraries (**LangGraph**, **FastAPI**, **Redis-py**, **AsyncPG**) to replace thousands of lines of custom spaghetti while showcasing canonical OS, DBMS, and LLD patterns.  
> **Algorithm Invariant**: **Zero degradation in Recall & Accuracy** — preserve the Multimodal Keyframe Superseding Engine (dHash $\le 10$ + OCR novelty $W=6$) and single-pass delimiter note synthesis (`===NOTES_END===`).

---

## 🧭 The Canonical 3-Agent Architecture

We eliminate redundant intermediate processes (like the 720-line `sync_agent.py`) and establish an intuitive, 3-Agent mental model:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        THE 3-AGENT ARCHITECTURE                        │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   1. AudioAgent (The "Ears" - Perception Worker)                       │
│      • Continuous mic/system audio capture                             │
│      • Faster-Whisper ASR transcription                                │
│      • Emits: `stream:transcript` chunks                               │
│      • OS Role: Independent CPU/GPU-bound Producer Process             │
│                                                                        │
│   2. ScreenAgent (The "Eyes" - Perception Worker)                      │
│      • Periodic screen grab (1s interval)                              │
│      • [RECALL INVARIANT]: dHash (<=10) + Running OCR Novelty (W=6)    │
│      • Emits: `stream:slide` (Only on real slide changes - 96.2% recall)│
│      • OS Role: Independent CPU-bound Producer Process                 │
│                                                                        │
│   3. SynthesisAgent (The "Brain" - LangGraph Cognitive Worker)         │
│      • Consumes `stream:transcript` & `stream:slide`                   │
│      • LangGraph Node 1 [AlignContext]: Absorbs time-window sync       │
│      • LangGraph Node 2 [SynthesizeNotes]: Single prompt with delimiter │
│      • LangGraph Node 3 [Validate]: Checks `===NOTES_END===` & JSON     │
│      • Emits: `stream:notes`                                           │
│      • OS Role: Async Consumer Process running LangGraph StateGraph    │
│                                                                        │
│   4. Storage Persister (Service, Not an Agent)                         │
│      • Listens to `stream:notes`                                       │
│      • Commits notes to PostgreSQL using ACID transactions             │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 File-by-File LoC Target (~2,500 Total Lines)

Every file is laser-focused, single-responsibility, and under 200 lines:

```
lecturelens/
├── core/
│   ├── supervisor.py      # [OS] Clean PCB, Process Spawning, Zombie Reap (~120 lines)
│   ├── bus.py             # [OS] Redis Streams Producer-Consumer Buffer (~120 lines)
│   └── rate_limiter.py    # [LLD] Token Bucket Rate Limiter (~80 lines)
├── storage/
│   └── postgres.py        # [DBMS] ACID transactions, Connection pool, 3NF Repositories (~150 lines)
├── agents/
│   ├── audio_agent.py     # [PRODUCER] Audio capture & Faster-Whisper ASR (~120 lines)
│   ├── screen_agent.py    # [RECALL INVARIANT] dHash + OCR novelty filter (~180 lines)
│   └── synthesis_agent.py # [LANGGRAPH] StateGraph for Alignment + Synthesis (~150 lines)
├── gateway/
│   └── app.py             # [FASTAPI] WebSocket endpoint + Replay API (~120 lines)
├── schemas.py             # [PYDANTIC V2] Domain Models & Events (~100 lines)
└── config.py              # Settings (~50 lines)
```

| Component | Before (Custom Spaghetti) | After (Battle-Tested Lean) | LoC Before | LoC Target |
| :--- | :--- | :--- | :---: | :---: |
| **Synthesis Agent** | Custom 1,960-line retry loops in `structuring_agent.py` | **LangGraph `StateGraph`** | 1,960 | **~150** |
| **Sync Logic** | Standalone 720-line process in `sync_agent.py` | Absorbed into LangGraph Node 1 | 720 | **~0 (Absorbed)** |
| **Gateway Server** | Raw `websockets` library in `server.py` | **FastAPI** native WebSockets | 904 | **~120** |
| **Message Bus** | SQLite outbox tables inside `redis_bus.py` | Clean **`redis.asyncio`** Streams | 914 | **~120** |
| **Screen Agent** | Redundant ffmpeg probes & wrappers | Clean dHash + OCR novelty algorithm | 1,021 | **~180** |
| **Audio Agent** | 3 fallback capture layers | Clean Faster-Whisper pipeline | 997 | **~120** |
| **Storage** | Repetitive raw SQL strings in `postgres.py` | Clean **AsyncPG** 3NF Repositories | 598 | **~150** |
| **Supervisor** | Heartbeat files, stall probes in `process_manager.py` | Standard **PCB & `process.poll()`** | 337 | **~120** |
| **Pruned Bloat** | Handwriting, drift filters, offline decoders, 11 CLI files | **Completely Pruned** | ~5,000 | **0** |
| **TOTAL CORE CODE** | | | **~18,200** | **~1,200 lines!** |

*Plus ~1,500 lines of focused, high-yield unit & integration tests $\implies$ **Entire project $\approx$ 2,700–3,000 total lines!***

---

## 🎯 Alignment with College Standards & Interview Walkthrough

### 1. 🖥️ OS Layer (Love Babbar College/Placement Standard)
* **Process vs Thread (CPython GIL Bypass)**:
  * AudioAgent and ScreenAgent run as **independent OS processes** (`subprocess.Popen` / `multiprocessing.Process`).
  * Demonstrates why CPU-bound AI models in Python cannot run on threads due to GIL lock contention.
* **PCB (Process Control Block)**:
  * Implemented as an explicit `ProcessControlBlock` dataclass (`pid`, `name`, `state`, `started_at`, `exit_code`).
* **Zombie & Orphan Process Prevention**:
  * On `SIGINT` (Ctrl+C), supervisor calls `process.poll()` and `process.wait()`, explicitly reaping terminated child processes to prevent **Zombie processes** from leaking in the OS process table.
* **Producer-Consumer Bounded Buffer**:
  * AudioAgent & ScreenAgent (Producers) $\to$ Redis Stream (Bounded Buffer with backpressure) $\to$ SynthesisAgent (Consumer).

---

### 2. 🗄️ DBMS Layer (Love Babbar College/Placement Standard)
* **ACID Transactions**:
  * Explicit transaction context manager for multi-table updates (Atomic note save + session update):
    ```python
    async with db.transaction() as conn:
        await note_repo.insert(conn, note)
        await session_repo.increment_note_count(conn, session_id)
        # Automatic ROLLBACK on exception, COMMIT on success
    ```
* **Relational Normalization & Constraints**:
  * **3NF Schema**:
    * `sessions`: `(session_id PK, title, created_at, status)`
    * `lecture_chunks`: `(chunk_id PK, session_id FK, start_time, end_time, transcript_text)`
    * `notes`: `(note_id PK, session_id FK, sequence_num, title, markdown_content, key_concepts)`
  * Foreign Keys with `ON DELETE CASCADE`.
  * Compound Unique Constraint on `(session_id, sequence_num)`.
* **B+ Tree Compound Indexing**:
  * `CREATE INDEX idx_notes_session_seq ON notes (session_id, sequence_num);`
  * Direct interview explanation: Range queries for replay (`WHERE session_id = ? AND sequence_num >= ?`) traverse the B+ Tree leaf linked-list.
* **Connection Pooling**:
  * Async connection pool (`asyncpg.create_pool(min_size=2, max_size=10)`).

---

### 3. 🧱 LLD Layer (Aryan Mittal Standard)
* **State Pattern via LangGraph `StateGraph`**:
  * Synthesis pipeline modeled as an explicit FSM:
    $$\text{START} \to \text{AlignContext} \to \text{SynthesizeNotes} \to \text{Validate} \xrightarrow[\text{invalid}]{\text{repair}} \text{RepairNode} \to \text{END}$$
* **Strategy Pattern**:
  * `LLMRouter`: Swap between Gemini, OpenAI, and Local Ollama at runtime without modifying synthesis logic.
* **Observer / Pub-Sub Pattern**:
  * Redis Streams pub-sub with Consumer Groups.
* **Repository Pattern**:
  * Clean separation of domain entities from SQL queries (`SessionRepository`, `NoteRepository`).
* **Token Bucket Rate Limiter**:
  * Classic LLD machine coding algorithm guarding the WebSocket gateway.

---

## 🧠 Preserving Core Intelligence, Recall & Accuracy

The algorithms responsible for state-of-the-art benchmarks remain **100% untouched**:

1. **Multimodal Keyframe Superseding Engine (Preserved)**:
   * Difference hash computation (`dHash`, threshold $\le 10$).
   * Running text novelty window ($W = 6$).
   * Triggers LLM synthesis only when cumulative novelty $> 0.35$ or hard slide transition occurs.
   * **Recall Invariant**: Slide change detection recall stays at **$\ge 96.2\%$**.
2. **Text / Transcript Overlap Accuracy (Preserved)**:
   * Word overlap evaluation metric (`Counter(words_of(captured)) & Counter(words_of(truth))`).
   * Word recall stays at **$\ge 0.88$**.
3. **Single-Pass Delimiter Synthesis (Preserved)**:
   * Structured Markdown output separated by `===NOTES_END===` + structured JSON summary.
4. **Monotonic Sequence Ordering (Preserved)**:
   * Strict monotonic integers `sequence_num = 1, 2, 3...` to guarantee replay consistency.

---

## 🚀 5-Step Execution Roadmap

1. **Step 1: Domain Schemas & DBMS Layer (`storage/postgres.py`, `schemas.py`)**
   * Pydantic v2 domain models.
   * AsyncPG pool with `with db.transaction():` context manager.
2. **Step 2: OS Supervisor & Redis Bus (`core/supervisor.py`, `core/bus.py`)**
   * PCB dataclass, multi-process spawning, `SIGINT` zombie reaping.
   * Clean Redis Streams `xadd` / `xreadgroup`.
3. **Step 3: Perception Workers (`agents/audio_agent.py`, `agents/screen_agent.py`)**
   * Faster-Whisper audio ingestion.
   * dHash + OCR novelty filter (Preserving 96.2% trigger recall).
4. **Step 4: LangGraph Cognitive Brain (`agents/synthesis_agent.py`)**
   * StateGraph with `AlignContext` $\to$ `SynthesizeNotes` $\to$ `Validate`.
   * Replaces both `sync_agent.py` and `structuring_agent.py`.
5. **Step 5: FastAPI Gateway & Verification (`gateway/app.py`, `eval/score.py`)**
   * WebSocket live stream + replay API.
   * Run test suite + benchmark evaluation (`eval/score.py`).
