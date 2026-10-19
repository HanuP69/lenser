# Fix log (v0.2.0)

Numbers refer to the bug list from the code review.

## Critical
1. Importance flags: whole-word regexes, so "example", "latest", "finally", "unimportant" no longer match. (`core/importance.py`)
2. Failed structuring batches are retried: the offset only advances after a batch succeeds. (`structuring_agent.py`)
3. Structuring progress is a byte offset stored in `context.json`, so restarts neither duplicate notes nor drop progress.
4. Only complete lines (ending in a newline) are consumed; a half-written line waits for the next pass.
5. Real `stop` command and graceful shutdown: capture agents, then sync (drains Redis and flushes), then a final structuring pass. Ctrl+C in `start` does the same. (`cli/main.py`, `core/agent_runtime.py`)
6. Sync holds entries by arrival time (`sync_merge_window_seconds`, default 30 s), orders each batch by event time, and structuring sorts its slice. (`sync_agent.py`)
7. Correlation rewritten around slide anchors; see v0.3.0 for the active-slide rule.
8. Screen agent only publishes when the slide changed since the last published capture, with a minimum interval. (`screen_agent.py`)
9. Audio capture is continuous and decoupled from transcription; chunks are cut at pauses, with a bounded transcription queue. (`audio_agent.py`)
10. Raw files are numbered from the highest existing index; starting an existing session needs `--resume`.
11. A broken process pool is rebuilt and the call retried. (`core/concurrency.py`)
12. Concept graph keeps one row per (concept, session), supports courses, migrates the old schema, and `review` reads the right database. "Previously covered in ..." notes are written.
13. Gateway processes raw audio/screen frames server-side, and there is a `lecturelens client` command for remote capture.

## High
14. Redis consumer: stable consumer names, own pending messages replayed on start, idle messages of dead consumers reclaimed, poison messages moved to `<stream>:dead`, backoff instead of a hot loop.
15. Publishing is buffered in a bounded outbox while Redis is down and flushed when it returns.
16. Logging everywhere; errors are no longer silently swallowed.
17. Restart backoff with a failure cap; log handles are closed before reopening.
18. Concept extraction with definitions stored into PostgreSQL durable storage.
19. Screen gate: downscaled grayscale, changed-pixel fraction, settle detection, minimum interval; diffing runs off the event loop; default monitor is 1 (primary).
20. OCR vision fallbacks run in pools/threads.
21. Image messages use the `{"url": ...}` form.
22. Fallback only on provider failures; cloud providers are opt-in (`--llm-mode hybrid|cloud`) and skipped without API keys; per-call timeout; `num_ctx` set for Ollama.
23. Whisper: VAD filter, no-speech/logprob filtering, hallucination phrase filter; language locking is opt-in (`whisper_lock_language`).
24. `both` mixes mic and loopback into one stream (summed and clipped) instead of two transcripts.
25. Slide gate stays open across captures with debounce and settle checks.
26. dHash perceptual hashing detects revisited slides to reuse extracted text.
27. Screen agent tolerates a missing router; provider and confidence come from the model response.
28. Gateway: refuses non-loopback binding without an API key, constant-time key check, session id validation and binding, per-connection rate limit, message size limit, session limit.

## Medium
29. `reprocess` resets outputs first and works in size-bounded batches.
30. Flags are attributed to the specific lines that carry them; definition and exam lines steer extraction.
31. QA: batched embeddings, incremental indexing with content-hash ids, stale notes removed, routed through the LLM router, sources returned.
32. CLI: latest session is the default, validation of options, real dashboard (agents, queues, router status, latest transcript, entry count), `--course`, `session_meta.json`, markdown export.
33. Unused settings are now used (`llm_mode`, thread pool size) or removed.
34. Durable Postgres storage persister runs in independent shutdown phase with its own flag.
35. Sync writes before removing entries from its buffer and flushes everything on shutdown.
36. All agents read the session from `--session`.
37. Rate limiter creates its key with a TTL before incrementing; breaker checked before the limiter.
38. Dependencies corrected (`zeroconf` removed, `httpx`, `typing-extensions`, `psycopg` added).
39. Models: `gemini-2.0-flash` was shut down by Google, replaced with `gemini-3.5-flash`; the OpenRouter entry now uses `openrouter/free`. Check both against your accounts.
40. Test suite covering process-pool recovery, offsets, partial lines, shutdown ordering, gateway security, and markdown formatting resilience (see README for the current count).

## Use cases
Pause/resume (`pause`, `resume`), duplicate slide detection, code-friendly OCR settings, course tagging, disk-space warning, Redis stream length cap, notes in the lecture language.

## Not done
Speaker diarization, pre-recorded video files as input, echo cancellation for `--audio-source both`.

# Fix log (v0.3.0)

Second review round. Everything below was reproduced before it was fixed and has a regression test in `tests/test_hardening.py`.

## Data safety
1. Redis consumer: with `auto_ack=False`, replaying the unacked backlog on restart spun forever (thousands of deliveries of the same messages). The backlog is now read with an advancing cursor.
2. Sync agent: acks only after the `timeline.jsonl` write; replays and reclaims are de-duplicated by (stream, message id) and by content fingerprint (restored from the timeline tail on start); the reclaim idle time is `merge window + 60 s` so buffered messages are never reclaimed while they wait.
3. Structuring: notes are guarded by `notes_marker.json` (end offset of the last appended batch) instead of searching for a `### HH:MM:SS` heading. Crashes on the first batch no longer duplicate notes, batches that share a wall-clock second are no longer skipped, and `notes.md` is never truncated on startup, so hand edits survive.
4. `reprocess`: refuses while agents are live, backs up the notes marker as well, and restores both the session and the course concept databases when the run fails.

## Correctness
5. Slide events are stamped with the capture time instead of the time OCR / vision finished.
6. Speech is correlated with the slide that is on screen (latest anchor up to `correlation_active_seconds`), with a short lead (`correlation_lead_seconds`) for speech right before a slide change.
7. Adaptive VAD no longer ratchets up on quiet speech: the floor is a rolling 10th percentile of block energy, it only raises the threshold when the room is louder than `vad_energy_threshold`, and the threshold is capped at 4x.
8. A slide whose OCR and vision both failed is retried in the background and published as a late update with its original timestamp.
9. Screen capture: sudden transient changes are held until settle checks confirm a true slide advance.
10. `(Cont.)` / `(Continued)` markers are removed from every line, not only from lines that became headings.
11. Heading recovery only promotes standalone lines (preceded by a blank line).
12. Gateway: idle sessions are evicted (oldest first) instead of exhausting `gateway_max_sessions` forever; client capture timestamps no longer overwrite the shared session clock (they are range-checked and used per event). The remote client now sends them.
13. Concept definitions extracted by structuring are persisted with uniqueness per course and session.
14. System and mixed audio recording supported with loopback soundcard backend.
15. Concept graph: names are normalized and fuzzy-matched, so cross-lecture links survive rephrasing.
16. `stop --timeout` defaults to 300 s (the worst-case shutdown is 180 s).
17. Configuration: model names come from `config/defaults.py` and settings (`local_model`, `embedding_model`) with router `.env` propagation.

## Benchmarks and docs
18. `benchmark_reproducible.py` warns loudly when it falls back to synthetic data, skips ASR cleanly when `faster-whisper` is missing, includes the thumbnail cost in the gate comparison, no longer assumes a 600 s session, and prints correct ms values.
19. README claims were corrected (privacy, limitations, benchmark meaning, test counts).

## Not done
Speaker diarization, pre-recorded video input, echo cancellation for `--audio-source both`.
