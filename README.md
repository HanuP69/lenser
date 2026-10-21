# LectureLens Live

> **Distributed, multi-agent lecture intelligence system.**  
> A central server captures lecture audio, slides, and transcripts once, synthesizing textbook-quality general notes with monotonic sequence numbers. Thin student clients connect to the gateway, stream live notes, replay missed segments from Postgres, and personalize notes locally using each student's personal LLM (English, Hindi, Hinglish) while maintaining a local SQLite knowledge graph tracking mastery and misconceptions.

---

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Data Flow & Lifecycle of a Session](#data-flow--lifecycle-of-a-session)
3. [Meet the Agents (Personalities & Roles)](#meet-the-agents-personalities--roles)
4. [Redis vs. Postgres: Durability & Replay](#redis-vs-postgres-durability--replay)
5. [Thin Student Client & Personalization Engine](#thin-student-client--personalization-engine)
6. [Student Knowledge Graph (`student_graph.db`)](#student-knowledge-graph-student_graphdb)
7. [Automated Test Suite (412 Tests)](#automated-test-suite-412-tests)
8. [CLI Command Reference](#cli-command-reference)
9. [Configuration & Environment Variables](#configuration--environment-variables)
10. [Docker Compose & Getting Started](#docker-compose--getting-started)

---

## System Architecture

LectureLens Live uses an event-driven, decoupled client-server architecture. The server processes physical inputs once to keep operational costs flat regardless of class size, while students personalize notes and query knowledge locally on their own devices.

```
+---------------------------------------------------------------------------------------+
|                                SERVER CAPTURE & SYNC                                  |
|                                                                                       |
|   +-----------------------+                           +-----------------------+       |
|   |      Audio Agent      |                           |     Screen Agent      |       |
|   |      (Deredere)       |                           |      (Tsundere)       |       |
|   |  Mic / System / ASR   |                           | SlideChangeGate/dHash |       |
|   +-----------+-----------+                           +-----------+-----------+       |
+---------------|---------------------------------------------------|-------------------+
                |                                                   |
                | Redis Stream                                      | Redis Stream
                | stream:transcript                                 | stream:slide
                v                                                   v
+---------------------------------------------------------------------------------------+
|                                    REDIS STREAMS                                      |
|                 Hot in-memory event bus & live fan-out message backbone               |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            | Consumer Group ("group:kuudere")
                                            v
+---------------------------------------------------------------------------------------+
|                                 SYNCHRONIZATION LAYER                                 |
|                                                                                       |
|                                Sync Agent (Kuudere)                                   |
|               Sliding merge window + Cross-modal anchor correlation IDs               |
|                                                                                       |
|                 Outputs: timeline.jsonl & stream:timeline                             |
+-------------------+-----------------------------------------------+-------------------+
                    |                                               |
                    | Batched Slice                                 | stream:timeline
                    v                                               v
+---------------------------------------+       +---------------------------------------+
|          STRUCTURING AGENT            |       |          POSTGRES PERSISTER           |
|              (Dandere)                |       |                                       |
|  LangGraph 6-Node Synthesis Chain     |       | Idempotent write path:                |
|  General notes with sequence numbers  |       | INSERT ... ON CONFLICT DO NOTHING     |
|  Outputs: stream:notes & notes.md     |       | Durability for timeline, notes & tags |
+-------------------+-------------------+       +-------------------+-------------------+
                    |                                               |
                    +-----------------------+-----------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
|                                  GATEWAY SERVER                                       |
|                                                                                       |
|                     WebSocket Fan-Out + Postgres Sequence Replay                      |
|             Reconnect with last_sequence_number -> Replay missed notes                |
+-------------------+-----------------------------------------------+-------------------+
                    |                                               |
                    | WebSocket connection                          | WebSocket connection
                    v                                               v
+---------------------------------------+       +---------------------------------------+
|            STUDENT CLIENT             |       |            STUDENT CLIENT             |
|                                       |       |                                       |
| - Local LLM Personalization           |       | - Local LLM Personalization           |
|   (English, Hindi, Hinglish)          |       |   (English, Hindi, Hinglish)          |
| - Timestamp & Concept Preservation    |       | - Timestamp & Concept Preservation    |
| - Student Chat Agent with citations   |       | - Student Chat Agent with citations   |
| - Local SQLite student_graph.db       |       | - Local SQLite student_graph.db       |
+---------------------------------------+       +---------------------------------------+
```

---

## Data Flow & Lifecycle of a Session

When a lecture is initiated, the system executes an automated, coordinated sequence:

```
[Server Session Launch]
      |
      v
1. Pre-Flight Health Checks
      Verifies Redis connectivity, Postgres connection, and model availability.
      |
      v
2. Capture & Ingestion
      Audio Agent captures spoken audio via VAD and streams CTranslate2 Whisper transcript chunks.
      Screen Agent captures display frames, applying SlideChangeGate settle logic and 64-bit dHash revisit cache.
      |
      v
3. Temporal Synchronization (Sync Agent)
      Consumes transcript and slide streams via consumer group `group:kuudere`.
      Reconciles events within a sliding merge window, assigning correlation IDs.
      Flushes durable chronological entries to `timeline.jsonl` and publishes to `stream:timeline`.
      |
      v
4. Idempotent Postgres Persistence
      Postgres Persister reads `stream:timeline` and `stream:notes`.
      Executes `INSERT ... ON CONFLICT (event_id) DO NOTHING`.
      Guarantees zero dropped or duplicated events across restarts.
      |
      v
5. General Notes Synthesis (Structuring Agent)
      Runs sequential 6-node LangGraph synthesis over timeline slices.
      Produces `GeneralNoteSegment` events with monotonic sequence numbers, key points, and concept tags.
      Streams segments to `stream:notes` and appends to `notes.md`.
      |
      v
6. Gateway Fan-Out & Replay
      Thin clients connect via WebSocket.
      Clients passing `last_sequence_number` receive missed notes replayed directly from Postgres.
      Active connections receive live fanned-out note segments from Redis.
      |
      v
7. Client-Side Personalization & Knowledge Graph
      Student client rewrites notes locally into preferred language (English, Hindi, Hinglish).
      Validates constraint preservation (terms, math formulas, timestamps).
      Maintains private local mastery levels and misconception history in `student_graph.db`.
```

---

## Meet the Agents (Personalities & Roles)

LectureLens models each background subsystem with a distinct anime personality archetype:

### 1. Audio Agent ("Deredere" — The Cheerful Listener)
* **Personality**: Warm, hyper-attentive, and enthusiastic.
* **In Her Own Words**:
  > *"Haaai! I'm Deredere! I listen to everything with all my heart! Whether you're in an auditorium with a microphone or capturing clean system audio, I record sound at 16,000 samples per second. I use adaptive energy VAD so we never waste time on awkward silences. When the teacher speaks, I transcribe it with faster-whisper! If they say 'dhyaan do' or 'this will be on the exam', I tag it for you with bright importance flags!"*
* **Duties**:
  - Mic or system audio capture at 16 kHz mono.
  - Real-time RMS Voice Activity Detection with ambient background adaptation.
  - Streaming CTranslate2 `faster-whisper` transcription with Hinglish cue support.

### 2. Screen Agent ("Tsundere" — The Presentation Gatekeeper)
* **Personality**: Feisty, perfectionist, refuses to waste compute.
* **In Her Own Words**:
  > *"H-Hmph! It's not like I wanted to look at your professor's boring slides or anything, b-baka! But someone has to keep this computer from melting! I downscale the screen to a 640-pixel thumbnail and evaluate SlideChangeGate with settle detection! Unless the slide actually changes and settles, I won't lift a finger. And if the professor flips back to an earlier slide, I fetch it instantly from my revisit cache using dHash without re-running vision OCR. Be grateful, okay?!"*
* **Duties**:
  - Desktop screen grabbing via `mss` frame grabber at 1 FPS.
  - `SlideChangeGate`: Grayscale thumbnailing, changed-pixel fraction thresholding, and settle verification.
  - 64-bit dHash revisit cache (Hamming distance $\le 3$, remembering up to 200 slides).
  - Direct multimodal vision-LLM OCR fallback on settled slide transitions.

### 3. Sync Agent ("Kuudere" — The Emotionless Chrono-Synchronizer)
* **Personality**: Calm, stoic, analytical, speaks strictly in facts.
* **In Her Own Words**:
  > *"Emotion is irrelevant. Temporal alignment is absolute. I consume event streams under consumer group 'group:kuudere'. Audio transcript chunks and slide captures arrive asynchronously. I hold events in a sliding merge window, assign cross-modal correlation IDs linking speech to visible slides, and write a deterministic chronological timeline to timeline.jsonl and Redis. Chaos in. Structured timeline out."*
* **Duties**:
  - Redis Streams consumer group processing with duplicate suppression.
  - Anchor-based cross-modal correlation linking speech chunks to nearby slide changes.
  - Durably flushes timeline entries to `timeline.jsonl` and emits to `stream:timeline`.

### 4. Structuring Agent ("Dandere" — The Quiet Scholar & Note Weaver)
* **Personality**: Shy, soft-spoken, and modest, but crafts textbook-quality notes.
* **In Her Own Words**:
  > *"U-Um... Hello. I work quietly in the background every 120 seconds. While the others collect the events, I take the timeline slice and run our 6-node LangGraph pipeline. I maintain a 20-second historical overlap buffer so sentences never get severed mid-thought. I synthesize chronological notes, extract key technical points and concept tags, and assign monotonic sequence numbers. I hope my notes help everyone study well..."*
* **Duties**:
  - LangGraph 6-node linear synthesis pipeline (Summarize $\to$ Extract Definitions $\to$ Action Items $\to$ Key Points $\to$ Context Update).
  - Sliding slice execution with a 20-second historical overlap continuity buffer.
  - Generates `GeneralNoteSegment` with monotonic sequence numbers.
  - Streams segments to `lecturelens:<session>:stream:notes` and persists to Postgres.

### 5. WebSocket Gateway ("Kamidere" — The Distributed Dispatcher)
* **Personality**: Grand, coordinating distributed client streams across the network.
* **In Her Own Words**:
  > *"Behold! The gateway connects students from across the campus over WebSockets! Connect with your student ID and last sequence number; if you dropped connection, I replay every missed segment from Postgres before streaming live notes. Centralized quality, infinite student fan-out."*
* **Duties**:
  - Authenticated WebSocket server supporting pre-shared API key verification.
  - Sequence-number replay from Postgres for reconnected students.
  - Live fan-out broadcast from Redis `stream:notes` to all active student connections.

---

## Redis vs. Postgres: Durability & Replay

| Layer | Redis Streams | PostgreSQL (with pgvector) |
|---|---|---|
| **Role** | Hot, temporary state & real-time messaging | Durable source of truth & long-term knowledge |
| **Data Held** | Live streams (`transcript`, `slide`, `timeline`, `notes`), consumer group states, rate limiter tokens | Courses, lectures, students, timeline events, general notes with sequence numbers, concepts, embeddings |
| **Recovery** | Ephemeral, buffer for live processing | Durable; clients query missed notes via sequence numbers |
| **Write Pattern** | `XADD` with stream capping | `INSERT ... ON CONFLICT (event_id) DO NOTHING` |

---

## Thin Student Client & Personalization Engine

The thin student client (`lecturelens/client/student_client.py`) connects to the Gateway, receives the general note stream, and personalizes notes locally:
- **Language Rewriting**: Rewrites general notes into English, Hindi, or Hinglish based on student preference.
- **Constraint Preservation**:
  - All core technical terms (e.g., `Binary Search Tree`, `Time Complexity`) are preserved.
  - Section headers and timestamp anchors are strictly preserved.
  - Code blocks and mathematical notation (`$...$` and `$$...$$`) remain exact.
- **Validation Fallback**: If local LLM output fails validation (e.g. hallucinated or omitted all concept tags), the client automatically falls back to the server's clean general note segment.

---

## Student Knowledge Graph (`student_graph.db`)

Each student maintains an isolated, private SQLite database (`student_graph.db`):
- **Concepts**: Tracks concept names, mastery levels ($0.0$ to $1.0$), review counts, and last review timestamps.
- **Misconceptions**: Logs detected error patterns and corrections when the student asks questions or misunderstands concepts.
- **Evidence**: Records timestamps and excerpts linking student errors or successes to specific lecture moments.
- **Student Chat Agent** (`lecturelens/knowledge/student_chat.py`): Answers student questions grounded in lecture context, cites timestamps (`[MM:SS]`), consults the student graph to tailor explanations, and automatically detects new misconceptions.

---

## Automated Test Suite (412 Tests)

LectureLens maintains an automated test suite of **412 unit and integration tests** verifying concurrency, resilience, audio processing, screen capture, Postgres persistence, gateway fan-out, client personalization, and knowledge graphs:

```
================= 410 passed, 2 skipped in 62.21s =================
```

The two skipped tests are the failure injection tests. They need a live Redis
and are only run when it is configured:

```bash
LECTURELENS_INTEGRATION_REDIS=redis://127.0.0.1:6379 pytest tests/test_integration_failures.py
```

### Module Breakdown:

| Test Module | Test Count | Scope & Behaviors Verified |
| :--- | :--- | :--- |
| `test_gateway.py` | **51 tests** | WebSocket gateway server, API key validation, Origin validation, role separation, bounded subscriber queues, reconnect race buffering, two-gateway fan-out groups, capture spool policies. |
| `test_hardening.py` | **45 tests** | Regression coverage: replay-without-spin, duplicate protection, active-slide correlation, notes idempotency and marker, JSON recovery, adaptive VAD scenarios, vision retry, capture timestamps. |
| `test_cli_and_runtime.py` | **41 tests** | CLI commands (`start`, `stop`, `pause`, `resume`, `status`, `export`, `reprocess`, `replay`, `student`, `persister`, `review`, `ask`, `client`), process flag coordination, option validations, event-loop nesting. |
| `test_student_client.py` | **27 tests** | Thin client subscription, Hinglish/Hindi personalization engine, concise English rewrite, gap buffering & draining, sequence persistence across instances, ledger gap recovery, knowledge graph integration. |
| `test_sync_agent.py` | **23 tests** | Redis Streams consumption, sliding merge window, cross-modal correlation IDs, chronological timeline serialization, two-sink flush ordering, crash recovery. |
| `test_redis_bus.py` | **23 tests** | Redis Streams resilient publishing, consumer group acknowledgements, delivery retry discarding with Redis persistence, backlog drain budget. |
| `test_importance.py` | **21 tests** | Verbal cue classification across English, Hindi, and Hinglish ("definition", "exam_relevant", "dhyaan do"), plus protection of lecture content against the ad heuristics. |
| `test_drift_filter.py` | **21 tests** | Context drift detection, sustained topic shifts, refusal handling, retry idempotency, corroborated ad/drift flags, quarantine. |
| `test_audio_agent.py` | **20 tests** | RMS VAD energy gating, speech segmentation, audio mixing, faster-whisper chunking, dropped block timestamp compensation. |
| `test_postgres_persister.py` | **20 tests** | Idempotent timeline events, general note sequence numbers, concept tag extraction and merge with `COALESCE`, crash recovery, pgvector support, embedding reconciliation. |
| `test_screen_agent.py` | **17 tests** | MSS screenshot grabbing, perceptual dHash hashing, Hamming distance slide cache, vision-LLM OCR fallback, mixed-clock gate, model checksum verification. |
| `test_llm_router.py` | **17 tests** | Multimodal task routing, OpenAI-compatible provider, Google GenAI, Ollama, circuit breaker rate-limiting, retries, per-task breaker scoping. |
| `test_clock_and_files.py` | **14 tests** | SessionClock persistence, elapsed duration calculation, accumulated_seconds tracking, atomic file writes. |
| `test_markdown_formatter.py` | **14 tests** | Code block indentation preservation, equation heading protection, unicode math preservation, display math block normalization. |
| `test_structuring_agent.py` | **14 tests** | LangGraph pipeline, rolling window, 20s context overlap, byte-offset tracking, robust JSON extraction, crash rollback, empty-summary retry and raw fallback. |
| `test_qa_engine.py` | **10 tests** | Vector indexing, pgvector integration with Chroma fallback, semantic search, RAG-grounded question answering with source citations. |
| `test_student_chat_and_graph.py` | **9 tests** | Student knowledge graph, concept normalization, mastery decay, misconception recording with confidence, timestamp citations in tutor chat. |
| `test_replay.py` | **8 tests** | ReplayEngine timeline playback, speed scaling (instant to real-time), Redis stream publishing. |
| `test_schemas.py` | **5 tests** | Pydantic event schemas: `TranscriptChunk`, `SlideCaptured`, `TimelineEntry`, `GeneralNoteSegment`, `FlagMoment`. |
| `test_concurrency.py` | **4 tests** | Thread pool and process pool executor allocation, worker limits, safe shutdown. |
| `test_pipeline_integration.py` | **3 tests** | Full integration test from multimodal stream ingestion through sync to general notes synthesis. |
| `test_circuit_breaker.py` | **3 tests** | Redis fixed-window rate limiter, failure count tracking and TTL cooldown. |
| `test_integration_failures.py` | **2 tests** | Opt-in failure injection: a crash between the durable file write and the Redis publish, and a crash between the state update and the release. |

To run the complete test suite:
```bash
pytest
```

---

## CLI Command Reference

### 1. `lecturelens start`
Launches server capture and structuring agents.
```bash
# Start server capturing microphone audio and display slides
lecturelens start --session "cs101_01" --course "cs101" --audio-source mic

# Start in server mode with gateway and persister enabled
lecturelens start --mode server --session "cs101_01" --course "cs101"
```

### 2. `lecturelens student`
Subscribes to live lecture notes from the gateway and personalizes them locally:
```bash
# Connect student client with Hinglish personalization
lecturelens student --server "ws://127.0.0.1:8765" --session "cs101_01" --student-id "student_1" --language "Hinglish" --detail "concise"
```

### 3. `lecturelens replay`
Replays a recorded lecture timeline onto Redis streams at selectable speeds for reproducible testing without live hardware:
```bash
# Replay session at 5x speed
lecturelens replay --session "cs101_01" --speed 5.0

# Instant replay (0 for maximum speed)
lecturelens replay --session "cs101_01" --speed 0
```

### 4. `lecturelens persister`
Runs the durable Postgres persister for a lecture session:
```bash
lecturelens persister --session "cs101_01"
```

### 5. `lecturelens review`
Inspects student mastery scores with time decay or reviews course concepts in Postgres:
```bash
# Review personal mastery and misconceptions for a student
lecturelens review --student-id "student_1"

# Review course concepts in Postgres
lecturelens review --course "cs101"
```

### 6. `lecturelens stop`
Triggers the multi-phase graceful shutdown:
```bash
lecturelens stop --session "cs101_01"
```

### 7. `lecturelens status`
Inspects real-time agent health, heartbeat timestamps, and Redis stream queue backlogs:
```bash
lecturelens status --session "cs101_01"
```

### 8. `lecturelens export`
Exports clean structured Markdown lecture notes:
```bash
lecturelens export --session "cs101_01" --format md --output ./exports
```

### 9. `lecturelens ask`
Interactive question-answering over indexed lecture knowledge with source citations and student misconception tracking:
```bash
# Standard lecture QA
lecturelens ask --session "cs101_01" "How does binary search achieve O(log n) time complexity?"

# Personalized student tutor QA (updates student's personal knowledge graph and tracks misconceptions)
lecturelens ask --session "cs101_01" --student-id "student_1" "Can a binary tree have 0 nodes?"
```

### 10. `lecturelens pause` / `lecturelens resume`
Pauses and resumes the capture agents without ending the session:
```bash
lecturelens pause --session "cs101_01"
lecturelens resume --session "cs101_01"
```

### 11. `lecturelens reprocess`
Rebuilds the general notes from an existing `timeline.jsonl`. The session must not
have live agents, and the previous outputs are restored if the run fails:
```bash
lecturelens reprocess --session "cs101_01" --llm-mode local
```

### 12. `lecturelens client`
Runs the remote capture client: it records audio and slides on this machine and
streams them to a gateway running elsewhere:
```bash
lecturelens client --session "cs101_01" --server "ws://192.168.1.20:8765" --api-key "secret"
```

---

## Configuration & Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `LECTURELENS_REDIS_URL` | `redis://localhost:6379` | Redis connection URL. |
| `LECTURELENS_POSTGRES_URL` | `postgresql://postgres:postgres@localhost:5433/lecturelens` | PostgreSQL connection URL. |
| `LECTURELENS_OLLAMA_URL` | `http://localhost:11434` | Local Ollama base URL. |
| `LECTURELENS_LLM_MODE` | `local` | `local` (Ollama), `hybrid` (Cloud with local fallback), or `cloud`. |
| `LECTURELENS_LOCAL_MODEL` | `gemma4:e4b` | Local model name. Overrides the `model` in `router.yaml`. |
| `LECTURELENS_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name. |
| `LECTURELENS_GATEWAY_HOST` | `127.0.0.1` | Gateway bind host (`0.0.0.0` for remote clients). |
| `LECTURELENS_GATEWAY_PORT` | `8765` | Gateway WebSocket listening port. |
| `LECTURELENS_GATEWAY_API_KEY` | *(generated)* | Pre-shared key for client authentication. When unset the gateway generates a secure key at startup and writes it to `<output_dir>/<session>/gateway_api_key.txt`, which the local `student` and `client` commands pick up automatically. |
| `LECTURELENS_GATEWAY_REQUIRE_API_KEY` | `true` | Require the API key. Set to `false` only for a trusted, isolated network; a non-loopback bind is refused without a key. |
| `LECTURELENS_GATEWAY_ALLOWED_ORIGINS` | *(empty)* | Comma-separated browser origins allowed to open the WebSocket. Browsers do not apply the same-origin policy to WebSockets, so a handshake carrying an `Origin` header is rejected unless it is listed here. Clients that send no `Origin` (the CLI, the Python client) are unaffected. |
| `LECTURELENS_WHISPER_MODEL_SIZE` | `base` | Faster-whisper model size (`tiny`, `base`, `small`, `medium`). |
| `LECTURELENS_OUTPUT_DIR` | `./output` | Root directory for session outputs. |
| `LECTURELENS_ROUTER_CONFIG_PATH` | `./router.yaml` | Provider routing configuration. |
| `LECTURELENS_VISION_MODEL_SHA256` | *(pinned)* | SHA-256 of the downloaded ONNX vision model. Override only for a mirrored copy. |
| `LECTURELENS_SYNC_CONSUMER_GROUP` | `group:kuudere` | Redis consumer group of the sync agent. **Changing it orphans the pending messages of every existing session**, because the old group is never read or acknowledged again. Drain the old group before switching. |
| `LECTURELENS_CAPTURE_SPOOL_MAX_SIZE` | `500` | How many captures the remote client keeps on disk while it is offline. |
| `LECTURELENS_CAPTURE_SPOOL_ON_FULL` | `drop_oldest` | `drop_oldest` keeps the capture running and loses the oldest pending capture. `raise` loses nothing: the new capture is dropped instead and only a log line reports it. |
| `UNOROUTER_API_KEY` / `UNOROUTER_API_KEYS` | *(none)* | Cloud provider credentials. Both the plain and the `LECTURELENS_`-prefixed spellings are accepted, and comma-separated values enable key rotation. |
| `OPENROUTER_API_KEY` / `OPENROUTER_API_KEYS` | *(none)* | As above. |
| `GOOGLE_API_KEY` / `GOOGLE_API_KEYS` | *(none)* | As above. |
| `GEMINI_API_KEY` / `GEMINI_API_KEYS` | *(none)* | As above. |

---

## Docker Compose & Getting Started

### 1. Launch Redis and PostgreSQL
Use the included `docker-compose.yml` to spin up Redis and PostgreSQL (with `pgvector`):
```bash
docker compose up -d
```

### 2. Install Dependencies
```bash
# Core install: capture, sync, structuring, gateway and the student client
pip install -e .

# Everything, including the question answering stack (chromadb) and the
# cloud provider clients
pip install -e ".[full]"
```

### 3. Start Recording
```bash
lecturelens start --session "my_first_lecture" --course "algorithms"
```
