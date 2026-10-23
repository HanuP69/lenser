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

# Fix log (v0.4.0)

Third review round. The default LLM mode changed: cloud providers are asked first and the local model is the fallback.

## Cloud first, local fallback
1. `llm_mode` defaults to `hybrid` instead of `local`: every provider in the chain is allowed, and the chain order decides — the cloud providers are asked first, the local Ollama model last. `local` and `cloud` still exist. (`config/settings.py`, `router.yaml`)
2. The local model is last in every chain of `router.yaml`, so hybrid does not try Ollama before OpenRouter and Gemini.
3. Key rotation rotates on more than 429/5xx: an invalid or expired key (401/403), a timeout and a connection error move on to the next key instead of abandoning the provider on its first one. (`router/llm_router.py`)
4. When Ollama is down in hybrid mode the warning names the fallback honestly (cloud providers carry the work when a key is configured). (`cli/commands/session.py`)

## Data safety
5. Process pool: only `BrokenProcessPool` rebuilds the pool and retries. An exception that the work itself raised arrived in the same catch, so the pool was thrown away under other waiting coroutines and the work was then run a second time in a thread (a write happened twice). The error is handed on untouched. (`core/concurrency.py`)
6. Outbox: the removal of a published row is its own attempt. A failed SQLite delete inside the same try as a successful XADD kept the event in the outbox and delivered it a second time on the next flush. (`core/redis_bus.py`)
7. Outbox single-flight: the wait between two publishers no longer removes a marker that belongs to another call, and an event that was just delivered by one call answers "published_earlier" to a second call instead of being XADDed again. (`core/redis_bus.py`)
8. Dead-letter: the XACK after a death is guarded, so an ack failure cannot re-send the message to the dead stream on every redelivery. (`core/redis_bus.py`)
9. `XAUTOCLAIM` answers that hold a bare message id (deleted messages of some server versions) no longer crash the reclaim with an unpacking error. (`core/redis_bus.py`)
10. Vector database replay includes the `definitions` column; every replayed note lost its term definitions on the way from Postgres. (`storage/postgres.py`)
11. Captures that the server always refuses are counted and dropped after 3 rejected deliveries. A refused head capture used to block every capture behind it for the lifetime of the spool, since the failing send was never acknowledged. (`client/remote_client.py`)
12. A spool append that fails after the eviction (double drop) is counted like a normal drop. (`client/remote_client.py`)
13. Replay budget: the budget of replay requests resets whenever a gap was filled or the connection came back, so ten scattered out-of-order segments across a lecture no longer silence the replay for the rest of it. (`client/student_client.py`)
14. A structuring byte offset past the end of the timeline no longer wedges the agent silently; it is clamped with a warning. (`agents/structuring_agent.py`)
15. `concepts.db` and notes: the structuring heartbeat no longer crashes on a falsy drift filter; a config without the drift filter took the agent down on its first pass. (`agents/structuring_agent.py`)
16. Audio shutdown: the stop sentinel is retried while the transcription task drains, and the task is cancelled if the queue never frees space. The old fallback was a `put_nowait` into the same full queue, and the run then waited forever. (`agents/audio_agent.py`)
17. `slides_awaiting_text` returns to zero when a background extraction ends early at the stop request; a crashed extraction task is logged. (`agents/screen_agent.py`)

## Correctness
18. Session clock: a wall clock step (NTP, laptop sleep) re-anchors the timestamp source. The estimate used to disagree with the wall clock for the rest of the session, so every timestamp came from whichever side of the tolerance it was in and the timeline joggled. The elapsed base of the session is untouched. (`core/clock.py`)
19. Corrupt clock files of more shapes (TypeError on scalar JSON) are survived by `load_or_create`. (`core/clock.py`)
20. Flag files are removed with `unlink(missing_ok=True)` and a logged OSError, so a TOCTOU race or a OneDrive handle cannot crash the agent. (`core/agent_runtime.py`)
21. A phase counts an exit code of 1 as unclean; "exited" is not "finished". (`core/process_manager.py`)
22. A spawn failure (interpreter gone, path moved) is reported instead of crashing the restart loop, and an agent whose process never started is visible to the restart loop again. (`core/process_manager.py`)
23. Gateway: the binding check in `start` uses the key in effect (the generated one), so a LAN bind with an ephemeral key no longer raises after `__init__` accepted it; asset auth accepts an absent token as wrong instead of right; non-ASCII tokens answer 401 instead of raising `TypeError`; `..` is refused in asset paths and the containment check needs the separator. (`gateway/server.py`)
24. `stop` on a session with no live agents says so immediately instead of clearing the finished flag, waiting out the grace period and reporting "still finishing". (`cli/commands/session.py`)
25. The session metadata says `persist` as it was decided after the Postgres check, not as the flag asked for. (`cli/commands/session.py`)
26. The Windows console is reconfigured to UTF-8 for every command; the function existed but was never called, so a cp1252 console could still die on Hindi or Urdu output. (`cli/app.py`)
27. A typo in the window question asks again instead of recording the whole desktop. (`cli/window_picker.py`)
28. A `#` in a value written to `.env` (window title) is quoted, so it is not read back as a comment. (`core/env_file.py`)
29. Citations name `timeline.jsonl`, a file that exists. (`knowledge/qa_engine.py`)
30. Student chat: a duplicated statement removed, the unreachable misconception warning runs when the annotation has no concept name. (`knowledge/student_chat.py`)

## Investigated and not broken
31. The pgvector bindings (`%s::vector` with a Python list) were suspected to fail on every insert; they were tried against a live Postgres 16 with pgvector and both insert and search work. The `definitions` read/write asymmetry above was the real replay loss.

# Fix log (v0.4.1)

Fourth review round. The notes have to be proper and every picture of a slide has to sit at the place it belongs to.

## The picture of a diagram gets its place in the notes
1. The token the model is asked to write was rewritten in the prompt: every `[FIGURE id]` marker in the events has to appear once, on its own line, right after the text that talks about that slide, and never as a pile at the end. (`agents/structuring_agent.py`)
2. An unplaced picture is no longer dumped at the end of the section. Its caption (the title of the slide it was cut from) is compared with every paragraph of the notes, and the picture is inserted after the paragraph that shares its words (two words, or one long word, count as a match). A picture that matches nothing keeps the old place at the end. (`agents/structuring_agent.py`)
3. The token regex takes the colon form too: `[FIGURE: fig_1]` names the picture fig_1 in the old code was left unmatched and the picture wandered to the end of the section. A picture marked twice (the model repeated a token) is now shown once. (`agents/structuring_agent.py`)
4. The pass that writes the mathematics in LaTeX was told nothing about the picture tokens, and a rewrite that swallowed the `[FIGURE id]` lines pushed every picture of the section to the end. The prompt says the lines stay as they are, and a rewrite that loses a token is refused (the section keeps its own text). (`agents/structuring_agent.py`)
5. The markdown formatter promoted a marker of its own line (`[FIGURE fig_9: Insertion Case]`) to a heading; once the token was replaced with the picture a lone `####` stayed behind. Lines in square brackets are not headings any more. (`core/markdown_formatter.py`)

## The slide text and the slide pictures survive the path to the notes
6. Figure deduplication did the counting and none of the work: the stat `slide_figure_deduped` was incremented and the near identical crop was saved anyway, so a diagram that stayed on the screen was put into the notes again for every slide it stayed on. The previous file is reused now. (`agents/screen_agent.py`)
7. A revisited slide whose text was never read (the first visit is still being read in the background when the lecturer returns) was published with no text and no reading was scheduled for it: the notes got a slide of nothing every time the lecturer went back. The retake shows the same screen, so the text is read from it, and the words are stored under the number of the older slide too, so a third visit needs no new reading. The update event carries the revisit number, so the notes group the text with the slide instead of dropping it. (`agents/screen_agent.py`)

## Lecture content is not thrown away
8. The pending quarantine holds the entries of speech that risks to be off topic. When more than twenty of them came, the oldest one was dropped from everywhere — not into the notes and not into the quarantine either. It goes to the quarantine that the structuring agent writes to disk instead. (`core/drift_filter.py`)
9. A deliberate rebuild left the quarantined entries of the run before it in the filter, and the first batch of the new run wrote them into the quarantine file of the new session. (`core/drift_filter.py`)

# Fix log (v0.5.0, improvement plan Phase 0)

The plan for the next rounds (fewer model calls, board capture, offline mode) is judged against numbers, and there were none. This round writes them down.

## The account of every model call
1. Every call through the router is written into the session's logs folder as one line of JSON: the provider, the piece of work (stage), the tokens read and written, how long the call took, and what became of the try. A provider that was refused by its own budget, or that the breaker had closed, is written down too — the burn of the fallback chain is part of the cost, and until now it was invisible. (`router/usage.py`, `router/llm_router.py`)
2. The token counts are read from the three shapes the provider wrappers use (`usage_metadata`, `tokenUsage`, and the Google usage block), and a count that was never sent is left unknown instead of being counted as zero. (`router/usage.py`)
3. Each ask of the structuring graph names its stage in the account (summarize, latex, definitions, action_items, key_points, compress), and the vision reading names its stage (slide_vision). This is how the report says which piece of the work spent the tokens. (`agents/structuring_agent.py`, `agents/screen_agent.py`)
4. The report reads the sessions back and prints the tables: per provider and per stage, and an estimate of the requests one batch of notes takes (a silence longer than the batch gap ends a batch). `python eval/usage_report.py output/<session>` (`eval/usage_report.py`)
5. Regression tests: the sink, the stage window, the totals, and every outcome of the guarded call (ok, limited, breaker open, failed) with their token counts. (`tests/test_router_usage.py`)
