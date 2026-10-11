# LectureLens

> **Real-time, local-first, multimodal multi-agent lecture intelligence system.**  
> Automatically captures spoken audio, computer slides, and physical whiteboard video, aligns them into a unified chronological event stream, and synthesizes textbook-quality Markdown notes, concept graphs, and Anki flashcards.

---

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Data Flow & Lifecycle of a Session](#data-flow--lifecycle-of-a-session)
3. [Meet the Agents (Personalities & Roles)](#meet-the-agents-personalities--roles)
4. [Empirical Benchmarks & Reproducible Metrics](#empirical-benchmarks--reproducible-metrics)
5. [Automated Test Suite (268 Tests)](#automated-test-suite-268-tests)
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
|   |  WASAPI / Mic / VAD   |   | SlideChangeGate/dHash |   |  Warp / Stillness / CV|   |
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
|                  LangGraph 6-Node Linear Synthesis Pipeline                           |
|             Summarize -> Extract Terms -> Action Items -> Flashcards                  |
|                                                                                       |
|                             LLM Router (Himedere)                                     |
|             Ollama (Local) <---> UnoRouter / Gemini / OpenRouter (Cloud)              |
|                    Fixed-Window Rate Limiter + TTL Failure Breaker                    |
+-------------------+-----------------------+-----------------------+-------------------+
                    |                       |                       |
                    v                       v                       v
          +-------------------+   +-------------------+   +-------------------+
          | Structured Notes  |   |   Anki Deck       |   | Knowledge Graph   |
          |    (notes.md)     |   |    (.apkg)        |   |   (concepts.db)   |
          | Clean Markdown    |   | Basic MathJax Card|   | SQLite Co-occur   |
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
3. Process Monitor Spawns Agent Subprocesses
      |
      +---> Audio Agent      : Captures audio -> VAD energy gating -> faster-whisper ASR
      +---> Screen Agent     : Grabs display frames -> SlideChangeGate pixel diff -> OCR/Vision
      +---> Camera Agent     : Detects whiteboard -> 4-point perspective warp -> Vision
      |
      v
4. Cross-Modal Event Ingestion via Redis Streams
      Events tagged with epoch timestamps published to `lecturelens:<session>:stream:*`
      |
      v
5. Sync Agent Alignment (Kuudere)
      Maintains a 30-second sliding correlation buffer.
      Correlates spoken transcript chunks with nearby slide/board anchors within 5 seconds.
      Persists durable chronological timeline to `timeline.jsonl`.
      |
      v
6. Periodic LangGraph Structuring (Dandere)
      Wakes up every 120 seconds.
      Reads unprocessed timeline bytes using byte-offset checkpointing.
      Injects a 20-second historical overlap buffer to maintain context continuity.
      Runs sequential 6-node synthesis:
        - Summarization & Hierarchical Sectioning
        - Term Definition Extraction -> SQLite Concept Graph
        - Action Items & Exam Alerts -> Markdown Callouts
        - Flashcard Generation -> Anki Deck (.apkg) with MathJax LaTeX formatting
      Normalizes output through regex post-processing (purges model scratchpads, mojibake, and formatting artifacts).
      Appends clean Markdown to `notes.md` with offset tracking to prevent crash duplicates.
      |
      v
7. 3-Phase Graceful Shutdown (On Ctrl+C or `lecturelens stop`)
      Phase 1 (30s timeout): Sets stop.flag -> Capture agents halt audio/video recording.
      Phase 2 (30s timeout): Sets sync_stop.flag -> Sync agent flushes remaining stream events to timeline.
      Phase 3 (120s timeout): Sets finalize.flag -> Structuring agent executes final end-of-lecture synthesis.
      Notes, concept graphs, and flashcards finalized safely.
```

---

## Meet the Agents (Personalities & Roles)

LectureLens models each background subsystem with a distinct anime personality archetype ("-dere"):

### 1. Audio Agent ("Deredere" — The Cheerful Listener)
* **Personality**: Warm, hyper-attentive, and enthusiastic. Never wants to miss a single word the professor utters.
* **In Her Own Words**:
  > *"Haaai! I'm Deredere! I listen to everything with all my heart! Whether you're in an auditorium with a microphone or watching an online lecture with system audio loopback, I capture every sound wave at 16,000 samples per second! I use Voice Activity Detection so we never waste time on awkward silences. When the teacher speaks, I transcribe it with faster-whisper! If they switch between English and Hindi, or say 'dhyaan do' or 'this will be on the exam', I immediately tag it for you with big bright flags! Let's get an A+ together!"*
* **Core Technical Duties**:
  - Dual-source audio ingestion (Windows WASAPI loopback + SoundDevice microphone).
  - Real-time RMS energy Voice Activity Detection (VAD) with silence threshold gating.
  - Streaming CTranslate2 `faster-whisper` transcription with configurable language locking.
  - Contextual importance cue detection (e.g., "crucial", "definition", "dhyaan do", "exam").

### 2. Screen Agent ("Tsundere" — The Picky Presentation Gatekeeper)
* **Personality**: Feisty, perfectionist, pretends she doesn't care, but refuses to waste computer resources.
* **In Her Own Words**:
  > *"H-Hmph! It's not like I wanted to look at your professor's boring slides or anything, b-baka! But someone has to keep this computer from melting! Do you have any idea how wasteful it is to run heavy OCR on 60 identical video frames every minute?! I downscale the screen to a 640-pixel thumbnail and evaluate SlideChangeGate with settle detection! Unless the slide actually changes and settles, I won't lift a finger. And if the professor flips back to an earlier slide, I fetch it instantly from my revisit cache using dHash without re-reading it. Be grateful, okay?!"*
* **Core Technical Duties**:
  - Desktop screen grabbing via `mss` frame grabber at 1 FPS.
  - `SlideChangeGate`: Grayscale thumbnailing, changed-pixel fraction thresholding, and multi-poll settle verification.
  - Hamming distance revisit cache (64-bit dHash, hamming distance $\le 3$, remembering up to 200 slides).
  - Tesseract OCR execution with automatic fallback to multimodal cloud vision (Gemma / Gemini).

### 3. Camera Agent ("Yandere" — The Obsessive Board Stalker)
* **Personality**: Intense, laser-focused, obsessively tracking the physical blackboard without blinking.
* **In Her Own Words**:
  > *"I see the board... I can ALWAYS see the board. No matter where the professor stands or how tilted the classroom camera is, I find the four corners of the whiteboard and stretch them into a perfect flat rectangle. I watch the chalk move... But I wait patiently. Only when the motion settles and blur variance clears do I capture the frame. I will never let a single derivation disappear from our notes... Never."*
* **Core Technical Duties**:
  - Video feed polling with frame-to-frame motion score estimation.
  - 4-point perspective contour detection and warp correction for skewed classroom whiteboard angles.
  - Laplacian blur variance filtering and stillness stability verification.
  - Multimodal handwritten mathematics and diagram transcription.

### 4. Sync Agent ("Kuudere" — The Emotionless Chrono-Synchronizer)
* **Personality**: Calm, stoic, analytical, speaks strictly in logical facts.
* **In Her Own Words**:
  > *"Emotion is irrelevant. Temporal alignment is absolute. I sit quietly on the Redis Streams message bus under consumer group 'group:kuudere'. Audio transcript chunks, slide captures, and whiteboard events arrive asynchronously at different speeds. Within my 30-second sliding correlation merge window, I match spoken utterances with nearby visual slide/board anchors within 5 seconds, assign correlation IDs, and sort them into a chronological timeline. Chaos in. Structured timeline out."*
* **Core Technical Duties**:
  - Redis Streams consumer group processing with resilient backoff buffers.
  - Sliding correlation window alignment across multimodal events.
  - Anchor-based correlation ID assignment linking speech chunks to active visual slides.
  - Chronological event logging to `timeline.jsonl`.

### 5. Structuring Agent ("Dandere" — The Quiet Scholar & Note Weaver)
* **Personality**: Shy, soft-spoken, and modest, but crafts clean, textbook-quality notes and study guides.
* **In Her Own Words**:
  > *"U-Um... Hello. I work quietly in the background every 120 seconds. While the others collect the events, I take the timeline slice and run our 6-node LangGraph pipeline. I maintain a 20-second historical overlap buffer so sentences never get severed mid-thought. I synthesize chronological notes, extract technical terms into our concept graph, generate Anki flashcards with MathJax, and sanitize formatting. I hope... My notes help you study well..."*
* **Core Technical Duties**:
  - LangGraph 6-node linear synthesis pipeline (Summarize $\to$ Extract Definitions $\to$ Action Items $\to$ Flashcards).
  - 120-second sliding slice execution with a 20-second historical overlap continuity buffer.
  - Robust JSON parsing capable of handling prose-wrapped model outputs without data loss.
  - Post-processing: stripping model reasoning scratchpads, checklist echoes, and normalizing LaTeX math blocks.
  - Exporting structured lecture notes to `notes.md` and Anki flashcard decks to `.apkg`.

### 6. LLM Router ("Himedere" — The Regal Model Dispatcher)
* **Personality**: Royal, haughty, demanding maximum elegance, resilience, and zero downtime.
* **In Her Own Words**:
  > *"Bow before the royal LLM dispatch! Only capable models are worthy of synthesizing our notes! I command both local Ollama runtimes and cloud endpoints like UnoRouter, OpenRouter, and Google Gemini. If a cloud API returns an HTTP rate limit or error, my failure circuit breaker records the strike and enforces a cooldown while failover routes take over! Elegance and fault-tolerance — anything less is unacceptable!"*
* **Core Technical Duties**:
  - Dynamic routing across local Ollama instances and cloud gateways.
  - Fixed-window rate limiting (enforcing RPM quotas via Redis INCR).
  - Redis TTL failure circuit breaker with automated provider failover.
  - Transient HTTP error handling with exponential retry backoff.

### 7. WebSocket Gateway ("Kamidere" — The Remote Classroom Dispatcher)
* **Personality**: Grand, coordinating distributed client streams from remote classroom laptops.
* **In Her Own Words**:
  > *"Behold! Why restrict capture to the host workstation? The Gateway accepts remote client streams over WebSockets! Stream raw audio and video frames over authenticated WebSocket connections; I ingest, validate, and multiplex them across our internal agent bus simultaneously."*
* **Core Technical Duties**:
  - Asynchronous WebSocket server accepting remote audio blocks and slide images.
  - Pre-shared API key validation via `hmac.compare_digest` in the hello frame.
  - Fixed session capacity (`gateway_max_sessions = 8`) to bound server memory.

---

## Empirical Benchmarks & Reproducible Metrics

LectureLens includes a standalone benchmarking tool (`benchmark_reproducible.py`) to measure latency, real-time factors (RTF), and compute reduction on actual lecture recordings.

The following benchmarks were recorded directly on an Intel multi-core CPU and an NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM) using real 1080p lecture slides and 16kHz lecture speech:

| Pipeline Component | Metric Tested | Baseline / Naive Approach | LectureLens Implementation | Measured Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Speech-to-Text (ASR)** *(5.0s Audio)* | **Transcription Latency** | CPU (INT8): **802.7 ms** *(6.2x real-time)* | **GPU CUDA (FP16)**: **113.3 ms** *(44.1x real-time)* | **7.1x Speedup** *(RTF: 0.1605 $\to$ 0.0227)* |
| **Slide Change Gate** *(1920x1080 FHD)* | **Frame Check Latency** | Full 1080p Matrix Diff: **34.39 ms** | **Production SlideChangeGate (640px)**: **8.66 ms** | **4.0x Check Speedup** *(Revisit dHash: 73.2 $\mu$s)* |
| **OCR Compute Load** *(10 min Lecture)* | **OCR / Vision Model Calls** | Blind 1 FPS sampling: 600 calls | **LectureLens Gated Invocations**: Dynamic per slide transition | **Significantly reduces vision calls by filtering static frames** |
| **Markdown Post-Processing** | **Sanitization Latency** | Full LLM Reprompting | **Regex Sanitizer & Math Normalizer**: **57.2 $\mu$s** | **Near-Instantaneous (0.05 ms)** |

### Reproducing the Benchmarks on Your Machine

To execute the benchmark suite and verify these numbers:
```bash
python benchmark_reproducible.py
```

---

## Automated Test Suite (268 Tests)

LectureLens maintains an automated test suite of **268 unit and integration tests** verifying concurrency, resilience, audio processing, computer vision, and LLM state machines:

```
----------------------------------------------------------------------
Ran 268 tests in 38.067s

OK
```
*(When running offline without Redis, Redis-dependent tests skip cleanly with 0 failures).*

### Module Breakdown:

| Test Module | Test Count | Scope & Behaviors Verified |
| :--- | :--- | :--- |
| `test_structuring_agent.py` | **32 tests** | LangGraph pipeline, rolling 120s window, 20s context overlap, byte-offset tracking, robust JSON extraction, crash rollback. |
| `test_cli_and_runtime.py` | **31 tests** | CLI commands (`start`, `stop`, `pause`, `resume`, `status`, `export`), process flag coordination, option validations, disk space warnings. |
| `test_gateway.py` | **18 tests** | WebSocket gateway server, API key validation, base64 payload decoding, connection multiplexing, remote streaming. |
| `test_audio_agent.py` | **17 tests** | RMS VAD energy gating, speech segmentation, audio mixing, faster-whisper chunking, language locking. |
| `test_camera_agent.py` | **17 tests** | 4-point perspective warp, motion thresholding, blur variance, stillness gating, whiteboard detection. |
| `test_importance.py` | **17 tests** | Verbal cue classification across English, Hindi, and Hinglish ("definition", "exam_relevant", "dhyaan do", bare test false-positive avoidance). |
| `test_sync_agent.py` | **16 tests** | Redis Streams consumption, sliding merge window, cross-modal correlation IDs, chronological timeline serialization. |
| `test_interactive.py` | **16 tests** | Interactive terminal wizard, live UnoRouter API health check, Ollama local model detection, model filtering. |
| `test_screen_agent.py` | **14 tests** | MSS screenshot grabbing, perceptual dHash hashing, Hamming distance slide cache, OCR fallback to vision LLM. |
| `test_markdown_formatter.py`| **14 tests** | Code block indentation preservation, equation heading protection, unicode math preservation, display math block normalization. |
| `test_llm_router.py` | **13 tests** | Multimodal task routing, OpenAI-compatible provider, Google GenAI, Ollama, circuit breaker rate-limiting, retries. |
| `test_clock_and_files.py` | **11 tests** | SessionClock persistence, elapsed duration calculation, accumulated_seconds tracking, atomic file writes. |
| `test_redis_bus.py` | **9 tests** | Redis Streams resilient publishing, offline memory outbox buffer, dead-letter stream, consumer group acknowledgements. |
| `test_recovery.py` | **8 tests** | AgentMonitor process lifecycle, exponential backoff restarts (1s, 2s, 4s, 8s), multi-phase graceful shutdown with timeouts. |
| `test_concept_graph.py` | **7 tests** | SQLite relational concept store, definition indexing, cross-lecture concept linking, sanitized database isolation. |
| `test_anki_exporter.py` | **6 tests** | Anki `.apkg` deck packaging, Basic note model with MathJax `\(` and `\[` rendering, deck-qualified GUID deduplication, currency preservation. |
| `test_qa_engine.py` | **6 tests** | Vector indexing, semantic search, RAG-grounded question answering with source transcript citations. |
| `test_schemas.py` | **5 tests** | Pydantic event schemas: `TranscriptChunk`, `SlideCaptured`, `BoardCaptured`, `TimelineEntry`. |
| `test_concurrency.py` | **4 tests** | Thread pool and process pool executor allocation, worker limits, safe shutdown. |
| `test_circuit_breaker.py` | **3 tests** | Redis fixed-window rate limiter, failure count tracking and TTL cooldown. |
| `test_pdf_exporter.py` | **3 tests** | ReportLab flowable PDF compilation, table styles, math formatting, typography, callout blocks. |
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

# Cloud Mode (UnoRouter multimodal vision)
lecturelens start --audio-source system --llm-mode cloud --session "cloud_lecture"
```

### 2. `lecturelens pause` & `lecturelens resume`
Pauses capture (disables audio/video recording while keeping agents alive in memory) and resumes seamlessly:
```bash
lecturelens pause --session "os_lecture_01"
lecturelens resume --session "os_lecture_01"
```

### 3. `lecturelens stop`
Triggers the multi-phase graceful shutdown: capture agents stop (30s timeout), remaining timeline entries are flushed (30s timeout), and a final LLM structuring pass synthesizes notes (120s timeout):
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
Safely backs up existing notes and re-runs the LangGraph structuring pipeline over `timeline.jsonl`:
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
    ├── notes.md                 # Structured Markdown lecture notes
    ├── timeline.jsonl           # Chronological, correlated event timeline
    ├── session_meta.json        # Run metadata (start/stop epochs, agents, durations)
    ├── clock.json               # Monotonic elapsed session clock with accumulated_seconds
    ├── context.json             # Rolling summary context, byte-offset pointer & note length
    ├── flashcards.apkg          # Generated Anki flashcard deck (.apkg)
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
| `LECTURELENS_LLM_MODE` | `local` | `local` (Ollama), `hybrid` (Cloud with local fallback), or `cloud` (Cloud gateway). |
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
- **Ollama** *(Optional, for offline local inference & RAG embeddings)*:
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
