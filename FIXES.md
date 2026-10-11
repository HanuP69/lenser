# Fix log (v0.2.0)

Numbers refer to the bug list from the code review.

## Critical
1. Importance flags: whole-word regexes, so "example", "latest", "finally", "unimportant" no longer match. (`core/importance.py`)
2. Failed structuring batches are retried: the offset only advances after a batch succeeds. (`structuring_agent.py`)
3. Structuring progress is a byte offset stored in `context.json`; flashcards live in `flashcards.jsonl`, so restarts neither duplicate notes nor drop cards.
4. Only complete lines (ending in a newline) are consumed; a half-written line waits for the next pass.
5. Real `stop` command and graceful shutdown: capture agents, then sync (drains Redis and flushes), then a final structuring pass. Ctrl+C in `start` does the same. (`cli/main.py`, `core/recovery.py`, `core/agent_runtime.py`)
6. Sync holds entries by arrival time (`sync_merge_window_seconds`, default 30 s), orders each batch by event time, and structuring sorts its slice. (`sync_agent.py`)
7. Correlation rewritten around slide/board anchors: nearest anchor wins, older links are never overwritten, anchors persist across flush batches.
8. Camera only publishes when the board changed since the last published capture, with a minimum interval. (`camera_agent.py`)
9. Audio capture is continuous and decoupled from transcription; chunks are cut at pauses, with a bounded transcription queue. (`audio_agent.py`)
10. Raw files are numbered from the highest existing index; starting an existing session needs `--resume`.
11. A broken process pool is rebuilt and the call retried. (`core/concurrency.py`)
12. Concept graph keeps one row per (concept, session), supports courses, migrates the old schema, and `review` reads the right database. "Previously covered in ..." notes are written.
13. Gateway processes raw audio/screen/camera frames server-side, and there is a `lecturelens client` command for remote capture.

## High
14. Redis consumer: stable consumer names, own pending messages replayed on start, idle messages of dead consumers reclaimed, poison messages moved to `<stream>:dead`, backoff instead of a hot loop.
15. Publishing is buffered in a bounded outbox while Redis is down and flushed when it returns.
16. Logging everywhere; errors are no longer silently swallowed.
17. Restart backoff with a failure cap; log handles are closed before reopening.
18. Anki deck id is a CRC32 of the deck name; card text is HTML-escaped; stable note guids.
19. Screen gate: downscaled grayscale, changed-pixel fraction, settle detection, minimum interval; diffing runs off the event loop; default monitor is 1 (primary).
20. OCR fallbacks run in pools/threads; Docker OCR is opt-in via `docker_ocr_url`.
21. Image messages use the `{"url": ...}` form.
22. Fallback only on provider failures; cloud providers are opt-in (`--llm-mode hybrid|cloud`) and skipped without API keys; per-call timeout; `num_ctx` set for Ollama.
23. Whisper: VAD filter, no-speech/logprob filtering, hallucination phrase filter, language locked after a confident detection.
24. `both` mixes mic and loopback into one stream (summed and clipped) instead of two transcripts.
25. Camera stays open across frames with warm-up frames and reopens on failure.
26. Perspective warp validates area, convexity and aspect ratio, else keeps the raw frame.
27. Camera and screen agents tolerate a missing router; provider and confidence come from the model response.
28. Gateway: refuses non-loopback binding without an API key, constant-time key check, session id validation and binding, per-connection rate limit, message size limit, session limit.

## Medium
29. `reprocess` resets outputs first and works in size-bounded batches.
30. Flags are attributed to the specific lines that carry them; definition and exam lines steer extraction and flashcards.
31. QA: batched embeddings, incremental indexing with content-hash ids, stale notes removed, routed through the LLM router, sources returned.
32. CLI: latest session is the default, validation of options, real dashboard (agents, queues, router status, latest transcript, entry count), `--course`, `session_meta.json`, PDF export.
33. Unused settings are now used (`llm_mode`, thread pool size) or removed.
34. See 18.
35. Sync writes before removing entries from its buffer and flushes everything on shutdown.
36. All agents read the session from `--session`.
37. Rate limiter creates its key with a TTL before incrementing; breaker checked before the limiter.
38. Dependencies corrected (`zeroconf` removed, `httpx`, `typing-extensions`, `reportlab` added).
39. Models: `gemini-2.0-flash` was shut down by Google, replaced with `gemini-3.5-flash`; the OpenRouter entry now uses `openrouter/free`. Check both against your accounts.
40. 260 tests, including process-pool recovery, offsets, partial lines, shutdown ordering, gateway security, and markdown formatting resilience.

## Use cases
Pause/resume (`pause`, `resume`), duplicate slide detection, code-friendly OCR settings, course tagging, disk-space warning, Redis stream length cap, notes in the lecture language.

## Not done
Speaker diarization, pre-recorded video files as input, echo cancellation for `--audio-source both`.
