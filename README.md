# LectureLens

> **Real-time, local-first, multimodal multi-agent lecture intelligence system.**  
> Automatically captures spoken audio, computer slides, and physical whiteboard video, aligns them into a unified chronological event stream, and synthesizes textbook-quality Markdown notes, concept graphs, and Anki flashcards.

---

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Data Flow & Lifecycle of a Session](#data-flow--lifecycle-of-a-session)
3. [Meet the Agents (Personalities & Roles)](#meet-the-agents-personalities--roles)
4. [Empirical Benchmarks & Reproducible Metrics](#empirical-benchmarks--reproducible-metrics)
5. [Automated Test Suite (257 Tests)](#automated-test-suite-257-tests)
6. [CLI Command Reference](#cli-command-reference)
7. [Session Artifacts & Output Structure](#session-artifacts--output-structure)
8. [Configuration & Environment Variables](#configuration--environment-variables)
9. [Installation & Getting Started](#installation--getting-started)

---

## System Architecture

LectureLens is structured as an event-driven, decoupled multi-agent system. Individual operating system processes execute specialized capture, synchronization, and structuring duties, communicating asynchronously through **Redis Streams** with consumer groups.

```
+---------------------------------------------------------------------------------------+
|                                CAPTURE LAYER (Agents)                                 |
|                                                                                       |
|   +-----------------------+   +-----------------------+   +-----------------------+   |
|   |      Audio Agent      |   |     Screen Agent      |   |     Camera Agent      |   |
|   |      (Deredere)       |   |      (Tsundere)       |   |       (Yandere)       |   |
|   |  WASAPI / Mic / VAD   |   |  dHash / Gate / OCR   |   |  Warp / Stillness / CV|   |
|   +-----------+-----------+   +-----------+-----------+   +-----------+-----------+   |
+---------------|---------------------------|---------------------------|---------------+
                |                           |                           |
                | Redis Stream              | Redis Stream              | Redis Stream
                | stream:transcript         | stream:slide              | stream:board
                v                           v                           v
+---------------------------------------------------------------------------------------+
|                                    REDIS STREAMS                                      |
|                       Durable In-Memory Event Backlog & Bus                           |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            | Consumer Group ("group:kuudere")
                                            v
+---------------------------------------------------------------------------------------+
|                                 SYNCHRONIZATION LAYER                                 |
|                                                                                       |
|                                Sync Agent (Kuudere)                                   |
|                Sliding Merge Window (30s) + Cross-Modal Correlation IDs               |
|                                                                                       |
|                           Output: timeline.jsonl                                      |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            | Batched Slice (120s window + 20s overlap)
                                            v
+---------------------------------------------------------------------------------------+
|                                 SYNTHESIS LAYER                                       |
|                                                                                       |
|                             Structuring Agent (Dandere)                               |
|                  LangGraph Multi-Step State Machine Pipeline                          |
|             Summarize -> Extract Terms -> Action Items -> Flashcards                  |
|                                                                                       |
|                             LLM Router (Himedere)                                     |
|             Ollama (Local) <---> UnoRouter / Gemini / OpenRouter (Cloud)              |
|                    Token-Bucket Rate Limiter + 3-Strike Circuit Breaker               |
+-------------------+-----------------------+-----------------------+-------------------+
                    |                       |                       |
                    v                       v                       v
          +-------------------+   +-------------------+   +-------------------+
          | Structured Notes  |   |   Anki Deck       |   | Knowledge Graph   |
          |    (notes.md)     |   |    (.apkg)        |   |   (concepts.db)   |
          | Clean AST Markdown|   | Spaced Repetition |   |  ChromaDB Vectors |
          +-------------------+   +-------------------+   +-------------------+
```

---

## Data Flow & Lifecycle of a Session

When a user launches a lecture recording session (`lecturelens start`), the system initiates a synchronized multi-stage lifecycle:

```
[Start Command]
      |
      v
1. Pre-Flight Health Checks (Redis reachable? LLM provider active? UnoRouter HTTP 200 OK?)
      |
      v
2. Session Directory & Clock Initialization (Creates raw/, control/, logs/, clock.json)
      |
      v
3. Process Monitor Spawns 5 Agent Processes (Subprocesses with isolated environments)
      |
      +---> Audio Agent      : Captures 16kHz audio -> VAD chunking -> faster-whisper ASR
      +---> Screen Agent     : Grabs display frames -> Perceptual dHash gate -> OCR/Vision
      +---> Camera Agent     : Detects whiteboard -> 4-point perspective warp -> Vision
      |
      v
4. Cross-Modal Event Ingestion via Redis Streams
      Events tagged with epoch timestamps published to `lecturelens:<session>:stream:*`
      |
      v
5. Sync Agent Alignment (Kuudere)
      Maintains a 30-second sliding correlation buffer.
      Correlates spoken transcript chunks with simultaneous slides or board equations.
      Persists durable chronological timeline to `timeline.jsonl`.
      |
      v
6. Periodic LangGraph Structuring (Dandere)
      Wakes up every 120 seconds.
      Reads unprocessed timeline bytes using byte-offset tracking.
      Injects a 20-second historical overlap buffer to prevent context severance.
      Runs multi-step synthesis:
        - Summarization & Sectioning
        - Term Definition Extraction -> SQLite Concept Graph
        - Action Items & Exam Alerts -> Markdown Callouts
        - Flashcard Generation -> Anki Deck (.apkg)
      Normalizes output through AST cleaner (purges scratchpads, mojibake, and math errors).
      Appends clean Markdown to `notes.md`.
      |
      v
7. 3-Phase Graceful Shutdown (On Ctrl+C or `lecturelens stop`)
      Phase 1: Sets stop.flag -> Capture agents halt audio/video recording.
      Phase 2: Sets sync_stop.flag -> Sync agent flushes remaining stream events to timeline.
      Phase 3: Sets finalize.flag -> Structuring agent executes final end-of-lecture synthesis.
      Zero data dropped. Notes, concept graphs, and flashcards finalized.
```

---

## Meet the Agents (Personalities & Roles)

LectureLens models each background subsystem with a distinct anime personality archetypes ("-dere"). Behind their personas lie production-grade distributed algorithms:

### 1. Audio Agent ("Deredere" — The Cheerful Listener)
* **Personality**: Warm, hyper-attentive, and enthusiastic. Never wants to miss a single word the professor utters.
* **In Her Own Words**:
  > *"Haaai! I'm Deredere! I listen to everything with all my heart! Whether you're in an auditorium with a microphone or watching an online lecture with system audio loopback, I capture every sound wave at 16,000 samples per second! I use Voice Activity Detection so we never waste time on awkward silences. When the teacher speaks, I transcribe it in milliseconds with faster-whisper! If they switch between English and Hindi, or say 'dhyan do' or 'this will be on the exam', I immediately tag it for you with big bright flags! Let's get an A+ together!"*
* **Core Technical Duties**:
  - Dual-source audio ingestion (Windows WASAPI loopback + SoundDevice microphone).
  - Real-time RMS energy Voice Activity Detection (VAD) with silence threshold gating.
  - Streaming CTranslate2 `faster-whisper` transcription with language locking.
  - Verbal importance cue detection (e.g., "crucial", "definition", "dhyaan do", "exam").

### 2. Screen Agent ("Tsundere" — The Picky Presentation Gatekeeper)
* **Personality**: Feisty, perfectionist, pretends she doesn't care, but refuses to waste computer resources.
* **In Her Own Words**:
  > *"H-Hmph! It's not like I wanted to look at your professor's boring slides or anything, b-baka! But someone has to keep this computer from melting! Do you have any idea how wasteful it is to run heavy OCR on 60 identical video frames every minute?! I downscale the screen to a 640-pixel thumbnail and compute a 64-bit difference hash in 25 microseconds flat! Unless the slide actually changes, I won't lift a finger. And if the professor flips back to a previous slide, I fetch it instantly from my 200-slide memory cache without re-reading it. Be grateful, okay?!"*
* **Core Technical Duties**:
  - Desktop screen grabbing via `mss` frame grabber at 1 FPS.
  - Grayscale thumbnailing and 64-bit perceptual difference hashing (dHash).
  - Hamming distance revisit cache (hamming distance $\le 3$, remembering up to 200 slides).
  - Tesseract OCR execution with automatic fallback to Gemma 4 multimodal cloud vision.

### 3. Camera Agent ("Yandere" — The Obsessive Board Stalker)
* **Personality**: Intense, laser-focused, obsessively tracking the physical blackboard without blinking.
* **In Her Own Words**:
  > *"I see the board... I can ALWAYS see the board. No matter where the professor stands or how tilted the classroom camera is, I find the four corners of the whiteboard and stretch them into a perfect flat rectangle. I watch the chalk move... But I wait patiently. Only when the professor steps away and absolute stillness arrives do I capture the frame. I will never let a single derivation disappear from our notes... Never."*
* **Core Technical Duties**:
  - Video feed polling with background subtraction motion detection.
  - 4-point perspective warp correction for skewed classroom whiteboard angles.
  - Laplacian blur variance filtering and stillness stability verification.
  - Multimodal handwritten mathematics and diagram transcription.

### 4. Sync Agent ("Kuudere" — The Emotionless Chrono-Synchronizer)
* **Personality**: Calm, stoic, analytical, speaks strictly in logical facts.
* **In Her Own Words**:
  > *"Emotion is irrelevant. Temporal alignment is absolute. I sit quietly on the Redis Streams message bus under consumer group 'group:kuudere'. Audio transcript chunks, slide captures, and whiteboard events arrive asynchronously at different speeds. Within my 30-second sliding correlation merge window, I match spoken utterances with the exact visual slide visible at that second, assign correlation IDs, and sort them into a chronological timeline. Chaos in. Structured timeline out."*
* **Core Technical Duties**:
  - Redis Streams consumer group processing with resilient backoff buffers.
  - Sliding correlation window alignment across multimodal events.
  - Correlation ID assignment linking speech chunks to active visual slides.
  - Durable chronological event logging to `timeline.jsonl`.

### 5. Structuring Agent ("Dandere" — The Quiet Scholar & Note Weaver)
* **Personality**: Shy, soft-spoken, and modest, but crafts breathtaking, textbook-quality notes and study guides.
* **In Her Own Words**:
  > *"U-Um... Hello. I work quietly in the background every 120 seconds. While the others collect the events, I take the timeline slice and run our multi-step LangGraph pipeline. I maintain a 20-second historical overlap buffer so sentences never get severed mid-thought. I synthesize chronological textbook notes, extract technical terms into our concept graph, generate Anki flashcards, and use our AST normalizer to strip out reasoning scratchpads. I hope... My notes help you study well..."*
* **Core Technical Duties**:
  - LangGraph multi-step state machine pipeline (Summarize $\to$ Extract Definitions $\to$ Action Items $\to$ Flashcards).
  - 120-second sliding slice execution with a 20-second historical overlap continuity buffer.
  - Deterministic AST post-processing: stripping model reasoning scratchpads, checklist echoes, and fixing LaTeX math delimiters.
  - Exporting structured lecture notes to `notes.md` and Anki flashcard decks to `.apkg`.

### 6. LLM Router ("Himedere" — The Regal Model Dispatcher)
* **Personality**: Royal, haughty, demanding maximum elegance, resilience, and zero downtime.
* **In Her Own Words**:
  > *"Bow before the royal LLM dispatch! Only the most capable models are worthy of synthesizing our notes! I command both local Gemma 4 runtimes and cloud royalty like UnoRouter, OpenRouter, and Google Gemini. If a cloud API dares to return an HTTP 429 rate limit or 500 error, my 3-strike circuit breaker trips instantly, banishing it to a 60-second cooldown while seamlessly rerouting traffic! Elegance and fault-tolerance — anything less is unacceptable!"*
* **Core Technical Duties**:
  - Dynamic routing across local Ollama instances and cloud gateways.
  - Redis token-bucket rate limiting (enforcing RPM quotas).
  - 3-strike circuit breaker with 60-second cooldown and automated failover.
  - Transient HTTP error handling with exponential retry backoff.

### 7. WebSocket Gateway ("Kamidere" — The Omnipresent Classroom Overlord)
* **Personality**: Godlike, grand, effortlessly coordinating distributed streams from across the globe.
* **In Her Own Words**:
  > *"Behold! Why restrict intelligence to a single workstation? From the cloud, I open the WebSocket gates to entire university lecture halls and remote laptops! Stream raw audio and video frames over authenticated WebSocket connections; I ingest, validate, and multiplex them across our internal agent bus simultaneously. The entire campus network is under my command."*
* **Core Technical Duties**:
  - Asynchronous WebSocket server accepting remote audio blocks and slide images.
  - HMAC-SHA256 API key authentication and message rate-limiting.
  - Remote client multiplexing for classroom-scale deployments.

---

## Empirical Benchmarks & Reproducible Metrics

LectureLens includes a standalone benchmarking tool (`benchmark_reproducible.py`) to measure latency, real-time factors (RTF), and compute reduction on actual lecture recordings.

The following benchmarks were recorded directly on an Intel multi-core CPU and an NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM) using real 1080p lecture slides and 16kHz lecture speech:

| Pipeline Component | Metric Tested | Baseline / Naive Approach | LectureLens Implementation | Measured Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Speech-to-Text (ASR)** *(7.80s Audio)* | **Transcription Latency** | CPU (INT8): 1,417.2 ms *(5.5x real-time)* | **GPU CUDA (FP16)**: **400.3 ms** *(19.5x real-time)* | **3.5x Speedup** *(RTF: 0.1817 $\to$ 0.0513)* |
| **Slide Change Gate** *(1920x1080 FHD)* | **Frame Check Latency** | Full 1080p Matrix Diff: 23.35 ms | **Thumbnail + 64-bit dHash**: **15.11 ms** | **1.5x Check Speedup** *(Kernel: 25.3 $\mu$s)* |
| **OCR Compute Load** *(10 min Lecture)* | **OCR / Vision Model Calls** | Blind 1 FPS sampling: 600 calls | **LectureLens Perceptual Gate**: **7 calls** | **98.8% Reduction in compute calls** |
| **Markdown Normalizer** *(AST Cleaner)* | **Sanitization Latency** | LLM Reprompting (~3,000 ms) | **AST Regex & Stripper**: **32.3 $\mu$s** | **Near-Instantaneous (0.03 ms)** |
| **Test Suite Coverage** | **Automated Tests** | 0 tests | **257 Unit & Integration Tests** | **100% Passing** in ~30.6s |

### Reproducing the Benchmarks on Your Machine

To execute the benchmark suite and verify these exact numbers:
```bash
python benchmark_reproducible.py
```

---

## Automated Test Suite (257 Tests)

LectureLens maintains a comprehensive test suite of **257 unit and integration tests** verifying concurrency, resilience, audio processing, computer vision, and LLM state machines:

```
----------------------------------------------------------------------
Ran 257 tests in 30.602s

OK
```

### Module Breakdown:

| Test Module | Test Count | Scope & Behaviors Verified |
| :--- | :--- | :--- |
| `test_structuring_agent.py` | **35 tests** | LangGraph multi-step pipeline, rolling 120s window, 20s context overlap, byte-offset tracking, definitions extraction, domain-aware high-yield flashcard synthesis (anti-rote question sanitization & structured answer detailing). |
| `test_cli_and_runtime.py` | **31 tests** | CLI commands (`start`, `stop`, `pause`, `resume`, `status`, `export`), process flag coordination, option validations, disk space warnings. |
| `test_gateway.py` | **18 tests** | WebSocket gateway server, HMAC authentication, base64 payload decoding, connection multiplexing, remote streaming. |
| `test_audio_agent.py` | **17 tests** | RMS VAD energy gating, speech segmentation, audio mixing, faster-whisper chunking, hallucination filtering. |
| `test_camera_agent.py` | **17 tests** | 4-point perspective warp, motion thresholding, blur variance, stillness gating, whiteboard detection. |
| `test_sync_agent.py` | **16 tests** | Redis Streams consumption, sliding merge window, cross-modal correlation IDs, chronological timeline serialization. |
| `test_interactive.py` | **16 tests** | Interactive terminal wizard, live UnoRouter API health check, Ollama local model detection, model filtering. |
| `test_screen_agent.py` | **14 tests** | MSS screenshot grabbing, perceptual dHash hashing, Hamming distance slide cache, OCR fallback to vision LLM. |
| `test_llm_router.py` | **13 tests** | Multimodal task routing, OpenAI-compatible provider, Google GenAI, Ollama, circuit breaker rate-limiting, retries. |
| `test_importance.py` | **13 tests** | Verbal cue classification across English, Hindi, and Hinglish ("definition", "exam_relevant", "action_item", "dhyaan do"). |
| `test_clock_and_files.py` | **11 tests** | SessionClock persistence, elapsed duration calculation, atomic file writes with Windows retry handling. |
| `test_redis_bus.py` | **9 tests** | Redis Streams resilient publishing, offline memory outbox buffer, dead-letter stream, consumer group acknowledgements. |
| `test_recovery.py` | **8 tests** | AgentMonitor process lifecycle, exponential backoff restarts (1s, 2s, 4s, 8s), multi-phase graceful shutdown. |
| `test_concept_graph.py` | **7 tests** | SQLite relational concept store, definition indexing, cross-lecture concept linking, sanitized database isolation. |
| `test_markdown_formatter.py`| **7 tests** | AST cleaner, scratchpad & checklist stripping, display math block normalization, Roman numeral heading conversion. |
| `test_qa_engine.py` | **6 tests** | ChromaDB vector indexing, semantic search, RAG-grounded question answering with source transcript citations. |
| `test_schemas.py` | **5 tests** | Pydantic event schemas: `TranscriptChunk`, `SlideCaptured`, `BoardCaptured`, `TimelineEntry`. |
| `test_anki_exporter.py` | **4 tests** | Anki `.apkg` deck packaging, cloze/basic note model generation, card deduplication. |
| `test_concurrency.py` | **4 tests** | Thread pool and process pool executor allocation, worker limits, safe shutdown. |
| `test_circuit_breaker.py` | **3 tests** | Redis token-bucket rate limiter, 3-strike circuit breaker state transitions (CLOSED $\to$ OPEN $\to$ HALF-OPEN). |
| `test_pdf_exporter.py` | **2 tests** | ReportLab PDF compilation, table formatting, page breaks, typography. |
| `test_pipeline_integration.py`| **1 test** | Full integration test from multimodal stream ingestion to finalized notes. |

To run the complete test suite:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## CLI Command Reference

LectureLens provides a unified CLI tool:

### 1. `lecturelens start`
Launches a lecture capture session. If run in an interactive terminal, launches the interactive setup wizard.
```bash
# Interactive setup wizard (prompts for provider, audio source, monitor)
lecturelens start

# YouTube / Online lecture capture (system loopback audio, screen enabled)
lecturelens start --audio-source system --no-camera --session "os_lecture_01" --course "operating_systems"

# In-person lecture capture (microphone audio + camera whiteboard)
lecturelens start --audio-source mic --session "physics_class" --course "physics"

# Pure Cloud Mode (UnoRouter multimodal vision, zero local GPU usage)
lecturelens start --audio-source system --llm-mode cloud --session "cloud_lecture"
```

### 2. `lecturelens pause` & `lecturelens resume`
Pauses capture (disables audio/video recording while keeping agents alive in memory) and resumes seamlessly:
```bash
lecturelens pause --session "os_lecture_01"
lecturelens resume --session "os_lecture_01"
```

### 3. `lecturelens stop`
Triggers the multi-phase graceful shutdown: capture agents stop, remaining timeline entries are flushed, and a final LLM structuring pass synthesizes notes:
```bash
lecturelens stop --session "os_lecture_01"
```
*(Alternatively, press `Ctrl + C` in the dashboard terminal).*

### 4. `lecturelens status`
Inspects real-time agent health, heartbeat timestamps, Redis queue backlogs, and LLM rate-limiter states:
```bash
lecturelens status --session "os_lecture_01"
```

### 5. `lecturelens ask`
Interactive RAG question-answering over indexed lecture knowledge with source citations:
```bash
lecturelens ask --session "os_lecture_01" "What is the difference between paging and segmentation?"
```

### 6. `lecturelens review`
Prints all technical concepts and definitions extracted across lectures for a course:
```bash
lecturelens review --course "operating_systems"
```

### 7. `lecturelens export`
Compiles generated notes into various distribution formats:
```bash
# Export Markdown, PDF, and Anki flashcard deck simultaneously
lecturelens export --session "os_lecture_01" --format all --output ./my_exports
```

### 8. `lecturelens reprocess`
Clears previous notes and reruns the LangGraph structuring pipeline over the saved `timeline.jsonl`:
```bash
lecturelens reprocess --session "os_lecture_01"
```

### 9. `lecturelens client`
Connects a thin client machine to a remote LectureLens Gateway server over WebSockets:
```bash
lecturelens client --server ws://SERVER_IP:8765 --session "classroom_hall_a" --api-key "secret-key"
```

---

## Session Artifacts & Output Structure

Each session creates an isolated, self-contained directory under `./output/<session_id>/`:

```
output/
└── os_lecture_01/
    ├── notes.md                 # Textbook-quality structured Markdown notes
    ├── timeline.jsonl           # Chronological, correlated event timeline
    ├── session_meta.json        # Run metadata (start/stop epochs, agents, durations)
    ├── clock.json               # Monotonic elapsed session clock
    ├── context.json             # Rolling summary context & byte-offset pointer
    ├── flashcards.apkg          # Generated Anki flashcard deck
    ├── raw/
    │   ├── audio/               # Voice-detected 16kHz WAV audio chunks (chunk_00001.wav)
    │   ├── slides/              # Gated transition slide captures (slide_0001.png)
    │   └── board/               # Perspective-warped whiteboard captures
    ├── control/                 # IPC coordination flags (stop.flag, finalize.flag)
    ├── heartbeats/              # Agent health status files (audio_agent.json)
    └── logs/                    # Isolated agent log outputs (structuring_agent.log)
```

Course-wide knowledge bases are stored in `output/_courses/<course_name>.db` (SQLite relational concept graph).

---

## Configuration & Environment Variables

LectureLens can be configured via environment variables or a `.env` file:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `LECTURELENS_REDIS_URL` | `redis://localhost:6379` | Redis connection URL. |
| `LECTURELENS_OLLAMA_URL` | `http://localhost:11434` | Local Ollama base URL. |
| `LECTURELENS_LLM_MODE` | `local` | `local` (Ollama), `hybrid` (Cloud with local fallback), or `cloud` (Pure cloud). |
| `LECTURELENS_UNOROUTER_API_KEY`| `None` | API key for UnoRouter multimodal gateway. |
| `LECTURELENS_OPENROUTER_API_KEY`| `None` | API key for OpenRouter gateway. |
| `LECTURELENS_GOOGLE_API_KEY` | `None` | API key for Google Gemini models. |
| `LECTURELENS_WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` (NVIDIA GPU acceleration). |
| `LECTURELENS_WHISPER_MODEL_SIZE`| `base` | Whisper model size (`tiny`, `base`, `small`, `medium`). |
| `LECTURELENS_OUTPUT_DIR` | `./output` | Root directory for session outputs. |
| `LECTURELENS_GATEWAY_HOST` | `127.0.0.1` | Gateway bind host (`0.0.0.0` for remote clients). |
| `LECTURELENS_GATEWAY_API_KEY` | `None` | Required authentication key for remote clients. |

---

## Installation & Getting Started

### 1. Prerequisites
- **Python**: Version 3.10 or higher.
- **Redis**: Running locally or via Docker:
  ```bash
  docker run -d -p 6379:6379 --name lecturelens-redis redis:alpine
  ```
- **Ollama** *(Optional, for 100% offline local inference)*:
  ```bash
  ollama pull gemma4:e4b
  ollama pull nomic-embed-text
  ```

### 2. Install LectureLens
```bash
git clone https://github.com/HanuP69/lenser-lenser-lenser.git
cd lenser-lenser-lenser
pip install -e ".[full]"
```

### 3. Start Your First Lecture
```bash
lecturelens start
```
The interactive wizard will launch, verify your LLM health, and begin real-time multimodal lecture processing.
