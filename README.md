# LectureLens

Real-time multimodal multi-agent lecture note-taking, slide capture, knowledge graph indexing, and Anki export system.

## Overview

LectureLens captures and synchronizes spoken lecture audio, computer slides, and whiteboard video in real time. It uses asynchronous stream processing via Redis Streams and a multi-agent architecture to correlate multimodal events and periodically synthesize cohesive study guides, flashcards, and concept maps.

## Key Features

- **Dual-Source Audio ASR (Deredere)**: Captures in-person microphone audio and online lecture audio via WASAPI system loopback. Applies real-time RMS Voice Activity Detection (VAD) gating and local streaming transcription using `faster-whisper`.
- **Differential Screen Capture (Tsundere)**: Monitors presentation displays, detects slide transitions via normalized image difference gating, and extracts slide text, code, and diagrams using Tesseract OCR or Gemma 4 vision fallback.
- **Computer Vision Whiteboard Capture (Yandere)**: Captures camera frames, checks motion and blur, computes 4-point perspective warp correction on whiteboards, and transcribes handwriting via multimodal vision models.
- **Cross-Modal Synchronization (Kuudere)**: Consumes transcript, slide, and board streams from Redis Streams, matches simultaneous events within a sliding correlation window, and outputs structured `timeline.jsonl`.
- **Periodic Structuring & Synthesis (Dandere)**: Runs a LangGraph pipeline to synthesize chronological lecture notes, extract key technical definitions, identify action items and exam highlights, update rolling lecture context, and build Anki decks (`.apkg`).
- **Knowledge Graph & RAG Q&A**: Indexes lecture transcripts and notes in SQLite (`concepts.db`) and ChromaDB vector store (`nomic-embed-text`) for interactive post-lecture question answering.
- **Fault-Tolerant LLM Router (Himedere)**: Supports local Ollama models (`gemma4:e4b`) with automated fallback to OpenRouter and Google Gemini, guarded by Redis token-bucket rate limiters and circuit breakers.
- **WebSocket Gateway (Kamidere)**: Accepts remote capture streams over WebSockets for distributed or classroom deployments.

## Architecture

```text
               +---------------------------------------------------+
               |             Capture Layer (Agents)                |
               |                                                   |
               |  [Audio Agent]    [Screen Agent]   [Camera Agent] |
               |     (WASAPI/Mic)      (MSS/Diff)     (OpenCV/Warp)|
               +--------+----------------+---------------+---------+
                        |                |               |
                        |   Redis Stream |  Redis Stream |
                        v                v               v
               +---------------------------------------------------+
               |                   Redis Streams                   |
               |        (transcript, slide, board streams)         |
               +-------------------------+-------------------------+
                                         |
                                         v
               +---------------------------------------------------+
               |              Sync Agent (Kuudere)                 |
               |       Timestamp alignment & Correlation IDs       |
               +-------------------------+-------------------------+
                                         |
                                         v timeline.jsonl
               +---------------------------------------------------+
               |          Structuring Agent (Dandere)              |
               |           LangGraph Multi-Step Pipeline           |
               +-------------------------+-------------------------+
                                         |
                        +----------------+----------------+
                        |                                 |
                        v                                 v
               [Structured notes.md]            [Anki Deck .apkg]
                        |                                 |
                        v                                 v
               [SQLite Concept Graph]           [ChromaDB RAG Q&A]
```

## Quick Start

### 1. Requirements

- Python 3.10+
- Redis (running locally or in Docker on `localhost:6379`)
- Ollama (running locally with `gemma4:e4b` and `nomic-embed-text`)

### 2. Installation

```bash
git clone https://github.com/HanuP69/lenser-lenser-lenser.git
cd lenser-lenser-lenser
pip install -e ".[full]"
```

### 3. Usage

Start live lecture capture (online/YouTube mode):
```bash
lecturelens start --audio-source system --no-camera --session "my_lecture" --course "algorithms"
```

Start in-person lecture capture (microphone + camera whiteboard):
```bash
lecturelens start --audio-source mic --session "classroom_lecture"
```

Cloud providers are never used unless you opt in with `--llm-mode hybrid` (local first, cloud fallback) or `--llm-mode cloud`. The default is `local`. API keys are read from `OPENROUTER_API_KEY` and `GOOGLE_API_KEY`.

Starting a session name that already exists is refused unless you pass `--resume`.

Pause, resume and stop (from another terminal):
```bash
lecturelens pause --session "my_lecture"
lecturelens resume --session "my_lecture"
lecturelens stop --session "my_lecture"
```
`stop` (and Ctrl+C in the `start` terminal) stops capture first, flushes the timeline, then runs a final structuring pass over everything that is left.

Inspect agent health, queue backlogs and provider status:
```bash
lecturelens status --session "my_lecture"
```

Review concepts across lectures of a course:
```bash
lecturelens review --course "algorithms"
lecturelens review --session "my_lecture"
```

Ask questions against the lecture knowledge base (answers list their sources):
```bash
lecturelens ask --session "my_lecture" "Explain the main theorem presented in class"
```

Export notes, Anki deck and PDF:
```bash
lecturelens export --session "my_lecture" --format all --output ./export
```

Rebuild notes from the saved timeline (clears previous notes, flashcards and concepts for that session first):
```bash
lecturelens reprocess --session "my_lecture"
```

Server mode, with remote capture clients:
```bash
set LECTURELENS_GATEWAY_API_KEY=change-me
set LECTURELENS_GATEWAY_HOST=0.0.0.0
lecturelens start --mode server --audio-only --session "room1"
lecturelens client --server ws://SERVER:8765 --session "room1" --api-key change-me
```
The gateway refuses to bind to a non-loopback address without an API key, binds each connection to the session it authenticated for, and processes raw audio, screen and camera frames on the server.

## Behaviour notes

- Audio is captured continuously and cut at pauses in speech, so transcription never blocks recording. Every chunk is written to `raw/audio` before transcription.
- Raw files are numbered from the highest existing index, so a restart never overwrites earlier captures.
- `sync_merge_window_seconds` (default 30) is how long entries wait before they are written to `timeline.jsonl`, so slow whiteboard reads can still be correlated with the speech around them.
- A failed structuring pass is retried on the next interval. Progress is stored as a byte offset in `context.json`, only after a batch succeeds.
- Sessions started without `--course` share one default course (`output/_courses/_default.db`), so "Previously covered in ..." notes can point across them. Use `--course` to keep subjects apart.
- `start` warns when less than 2 GB is free on the output drive.
- Agents that keep crashing are restarted with exponential backoff and are abandoned after repeated failures.

## Running Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```
