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
lecturelens start --audio-source system --no-camera --session "my_lecture"
```

Start in-person lecture capture (microphone + camera whiteboard):
```bash
lecturelens start --audio-source mic --session "classroom_lecture"
```

Inspect Redis stream status:
```bash
lecturelens status --session "my_lecture"
```

Review cataloged course concepts:
```bash
lecturelens review --session "my_lecture"
```

Ask questions against the lecture knowledge base:
```bash
lecturelens ask --session "my_lecture" "Explain the main theorem presented in class"
```

Export notes and Anki flashcard deck:
```bash
lecturelens export --session "my_lecture" --format all --destination ./export
```

Reprocess an existing timeline with updated prompts or models:
```bash
lecturelens reprocess --session "my_lecture"
```

## Running Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```
