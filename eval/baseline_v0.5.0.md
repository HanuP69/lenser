# Baseline numbers (v0.5.0, Phase 0)

The numbers every later phase of the improvement plan is judged
against. Measured on 23 Oct 2026, on this machine (Windows, Docker
Postgres + Redis up, Ollama down, cloud providers only), with the usage
account of v0.5.0 (`logs/llm_usage.jsonl` per session, read back with
`python eval/usage_report.py output/<session>`).

Both runs are the same recipe: a three-minute clip played through
`play_video.py --seconds 180`, a live session capturing the playback
window with system loopback audio, stopped after the playback ends.

## Run 1: slides (Redis video, 3 min)

One batch of notes at the end of the session. The notes themselves were
proper: headings, modules list, and the "Most Loved Database" figure
inline at the right place (the state of v0.4.1).

| | tries | ok | limited | failed | tokens in | tokens out | s/ok |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| structuring total | 8 | 7 | 0 | 1 | 10 875 | 6 814 | 19.1 |
| -- summarize | 2 | 2 | | | 1 331 avg | 1 173 avg | 29.4 |
| -- definitions | 3 | 2 | | 1 | 1 194 avg | 1 750 avg | 22.0 |
| -- action_items | 1 | 1 | | | 2 089 | 307 | 10.3 |
| -- key_points | 1 | 1 | | | 1 868 | 104 | 5.3 |
| -- compress | 1 | 1 | | | 1 869 | 557 | 15.0 |
| -- latex | 0 | 0 | | | (no mathematics found in the notes) | | |
| screen_ocr (slide_vision) | 1 | 1 | 0 | 0 | 1 431 | 104 | 8.1 |

**Requests for one batch of notes: 8** (7 succeeded, 1 failed on
`unorouter_gemma` and the stage fell through to `unorouter_structuring`).
The plan's estimate ("10k tokens and 5–6 requests per batch") is
confirmed: **10 875 input tokens, 8 tries**. The Phase 1 target is
about 3 000 tokens and 1 request.

Two further observations the plan predicted:

- Every stage resends the whole summary: the input of definitions,
  action_items and key_points (2 089, 1 868, 1 869) is nearly the size
  of the summarize output plus context, instead of working from the
  slice directly.
- A single failed call costs up to its provider timeout (90 s) before
  the chain moves on. That is part of why the notes only land minutes
  after the video ends.

## Run 2: physical board (JEE chemistry, 3 min)

| | tries | ok | limited | failed | tokens in | tokens out | s/ok |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| structuring total | 3 | 2 | 0 | 1 | 1 455 | 4 862 | 54.7 |
| -- summarize | 1 | 1 | | | 689 | 1 876 | 69.9 |
| -- latex | 2 | 1 | | 1 | 766 | 2 986 | 39.6 |
| screen_ocr (slide_vision) | **0** | | | | **no vision call was ever made** | | |

The capture-side column is the finding the plan claims, now measured:

- **1 slide published in 3 minutes, 0 vision calls.** The layout model
  finds nothing on a chalkboard (its labels are made for documents), the
  local OCR returns only window chrome, and because *something* was
  "read", the vision model was never scheduled. The board content never
  reaches any model. This is the hole Phase 3 fills.
- The vision embedder faulted once (`fault_the embedding of the drift
  filter=1`) on board frames, consistent with a gate that is fed floors
  it cannot read.
- The structuring work on the board clip is cheap on the input side
  (only speech) and slow on the wall clock (54.7 s worth of calls,
  including a skipped provider).

The session was stopped before the final structuring pass finished, so
`notes.md` was not written for this run; the plan's Phase 2 (offline
mode, no race against a live video) removes this whole class of rushed
stops.

## Ground truth packs (filled, waiting to be checked)

- `eval/board_pack/chem/`: 20 frames spread over the 14-minute clip,
  each with a hand-written transcription in `truth.jsonl` (uncertain
  words in `[brackets]`, occlusion noted). The frames show the failure
  modes directly: one new line per board state, the teacher's head
  covering the heading, two eraser wipes.
- `eval/audio_pack/chem.wav`: the first five minutes at 16 kHz mono,
  with ten half-minute rows in `chem_truth.jsonl` still to be filled by
  hand (Phase 5).

## Clips still missing from the eval set

Code editor screencast (`C_code` exists in `manifest.json` but the
video lives outside the repository) and tablet ink. Everything else is
covered.

## Numbers to beat

| metric | today | target |
| :--- | :--- | :--- |
| model requests per batch of notes (slides) | 8 | 1 (2 with repair) |
| input tokens per batch (slides) | 10 875 | ~3 000 |
| board frames captured per 3 min (board) | 1 with none of its content | most board states |
| vision calls spent on a board clip | 0 | a reading of every real board state |
