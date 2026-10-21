import asyncio
import json
import logging
import math
import os
from pathlib import Path
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from lecturelens.router.messages import HumanMessage, response_to_text
from lecturelens.schemas.events import GeneralNoteSegment

logger = logging.getLogger(__name__)


@dataclass
class StudentProfile:
    student_id: str = "default_student"
    preferred_language: str = "English"
    detail_level: str = "comprehensive"
    preserved_terms: list[str] = field(default_factory=list)


REWRITE_SYSTEM_PROMPT = """
You are a personal learning assistant customizing lecture notes for a student.
Re-synthesize the provided general lecture note segment according to the student profile:
- Target Language/Style: {preferred_language} (e.g. English, Hindi, Hinglish).
- Detail Level: {detail_level}.

CRITICAL CONSTRAINTS:
1. Preserve all technical terms, formulas ($...$ and $$...$$), and code blocks accurately.
2. Ensure every key point from the source is addressed.
3. Keep the section header with its original timestamp.
4. If writing in Hindi or Hinglish, keep core English technical terminology intact (e.g., 'Binary Search Tree', 'Time Complexity').
5. Output pure markdown notes only.
"""

KEY_POINT_STOPWORDS = frozenset({
    "that", "this", "with", "from", "they", "their", "there", "these", "those",
    "when", "then", "than", "which", "what", "where", "while", "into", "onto",
    "each", "some", "such", "also", "only", "over", "under", "about", "have",
    "will", "would", "could", "should", "because", "these", "those", "very",
})


def highest_contiguous_sequence(applied: Sequence[int]) -> int:
    """The highest sequence number with no gap below it.

    The ledger can be missing a sequence in the middle, for example when a
    note write failed. The cursor must stop at the gap, otherwise the missing
    segments are skipped and never written to the notes.
    """
    if not applied:
        return 0
    ordered = sorted(set(applied))
    contiguous = ordered[0]
    for value in ordered[1:]:
        if value == contiguous + 1:
            contiguous = value
        elif value > contiguous + 1:
            break
    return contiguous


class PersonalizationEngine:
    def __init__(
        self,
        profile: StudentProfile,
        llm_router: Any = None
    ) -> None:
        self.profile = profile
        self.llm_router = llm_router

    def build_prompt(self, segment: GeneralNoteSegment) -> str:
        prompt_intro = REWRITE_SYSTEM_PROMPT.format(
            preferred_language=self.profile.preferred_language,
            detail_level=self.profile.detail_level
        )
        preserved = ""
        if self.profile.preserved_terms:
            preserved = (
                "\nThese terms must appear verbatim in the rewritten section: "
                + ", ".join(self.profile.preserved_terms)
                + "\n"
            )
        return (
            f"{prompt_intro}\n{preserved}\n"
            f"SOURCE SECTION:\n"
            f"Heading: {segment.heading}\n"
            f"Key Points: {json.dumps(segment.key_points)}\n"
            f"Concepts: {', '.join(segment.concept_tags)}\n\n"
            f"SOURCE CONTENT:\n{segment.content}\n"
        )

    def _significant_words(self, text: str) -> list[str]:
        return [
            word.lower()
            for word in re.findall(r"\b\w{4,}\b", text)
            if word.lower() not in KEY_POINT_STOPWORDS
        ]

    @staticmethod
    def _word_matches(word: str, candidates: set[str]) -> bool:
        """Match a word against the rewrite, tolerating simple inflection.

        "halves" has to match "half" and "sorted" has to match "sort",
        otherwise a faithful paraphrase is rejected as a hallucination.
        """
        if word in candidates:
            return True
        if len(word) < 4:
            return False
        stem = word[:3]
        for candidate in candidates:
            if len(candidate) < 3:
                continue
            if candidate[:3] == stem and abs(len(candidate) - len(word)) <= 3:
                return True
        return False

    def validate_rewrite(self, segment: GeneralNoteSegment, rewritten_text: str) -> bool:
        if not rewritten_text or len(rewritten_text.strip()) < 20:
            return False

        lower_rewritten = rewritten_text.lower()
        if segment.concept_tags:
            present_tags = 0
            for tag in segment.concept_tags:
                clean_tag = tag.replace("_", " ").lower()
                if clean_tag in lower_rewritten or tag.lower() in lower_rewritten:
                    present_tags += 1
            if present_tags < math.ceil(len(segment.concept_tags) * 0.5):
                logger.warning("rewritten note omitted over half of the concept tags, rejected")
                return False

        if segment.start_time and segment.start_time in segment.heading:
            if segment.start_time not in rewritten_text:
                logger.warning("rewritten note omitted timestamp %s, rejected", segment.start_time)
                return False

        # Terms the student explicitly asked to keep have to survive the rewrite.
        for term in self.profile.preserved_terms:
            if term.lower() not in lower_rewritten:
                logger.warning("rewritten note omitted the preserved term %r, rejected", term)
                return False

        if segment.key_points:
            rewrite_words = set(self._significant_words(rewritten_text))
            source_words = set(self._significant_words(segment.content))
            shared_with_source = sum(
                1 for word in source_words if self._word_matches(word, rewrite_words)
            )
            if source_words and shared_with_source < 2:
                # The rewrite shares almost no vocabulary with the source, so
                # it is a translation into another script. Word level key
                # point matching cannot judge it, and the concept tag check
                # above already guards against a hallucinated rewrite.
                return True

            # A key point counts as covered when the rewrite reuses most of
            # its significant words. Requiring a single literal word from a
            # single key point was far too lax (a rewrite could drop
            # everything and still pass).
            required = math.ceil(len(segment.key_points) * 0.5)
            covered = 0
            for key_point in segment.key_points:
                words = self._significant_words(key_point)
                if not words:
                    continue
                needed = 1 if len(words) <= 2 else math.ceil(len(words) * 0.5)
                hits = sum(1 for word in words if self._word_matches(word, rewrite_words))
                if hits >= needed:
                    covered += 1
            if covered < required:
                logger.warning(
                    "rewritten note covered %d of %d key points, rejected",
                    covered, len(segment.key_points)
                )
                return False

        return True

    async def rewrite_segment(self, segment: GeneralNoteSegment) -> str:
        if self.llm_router is None:
            return segment.content

        if (
            self.profile.preferred_language.lower() == "english"
            and self.profile.detail_level.lower() == "comprehensive"
            and not self.profile.preserved_terms
        ):
            return segment.content

        prompt = self.build_prompt(segment)
        try:
            response = await self.llm_router.ainvoke([HumanMessage(content=prompt)])
            personalized = response_to_text(response).strip()
            if self.validate_rewrite(segment, personalized):
                return personalized
            logger.warning("personalization validation failed, falling back to original content")
            return segment.content
        except Exception as error:
            logger.error("personalization rewrite failed: %s, falling back to original", error)
            return segment.content


class StudentClient:
    def __init__(
        self,
        server_url: str,
        session_id: str,
        profile: StudentProfile,
        output_dir: Path,
        api_key: Optional[str] = None,
        llm_router: Any = None,
        student_graph: Optional[Any] = None
    ) -> None:
        self.server_url = server_url
        self.session_id = session_id
        self.profile = profile
        self.output_dir = output_dir
        self.api_key = api_key
        self.student_graph = student_graph
        self.rewriter = PersonalizationEngine(profile, llm_router)
        self.state_file = output_dir / f"{session_id}_{profile.student_id}_state.json"
        self.ledger_file = output_dir / f"{session_id}_{profile.student_id}_ledger.db"
        self._ledger_was_new = not self.ledger_file.exists()
        self._init_ledger()
        # The ledger and the cursor are per student, so the rendered notes
        # have to be per student as well. A per-session file name meant two
        # students sharing an output directory overwrote each other's notes.
        self.notes_file = output_dir / f"{session_id}_{profile.student_id}_personalized_notes.md"
        self.legacy_notes_file = output_dir / f"{session_id}_personalized_notes.md"
        if not self.notes_file.exists() and self.legacy_notes_file.exists():
            # Migrate the pre-0.4 per-session notes file so an upgrade does
            # not lose the notes that were already written.
            try:
                self.legacy_notes_file.replace(self.notes_file)
            except OSError as error:
                logger.warning("could not migrate the legacy notes file: %s", error)
        reconciled = self.reconcile_notes_from_ledger()
        self.pending_reconciliation = not reconciled
        self.last_sequence_number = self.load_persisted_sequence()
        self.received_segments: list[GeneralNoteSegment] = []
        self.pending_segments: dict[int, GeneralNoteSegment] = {}
        self.is_running = False
        self.auth_error: Optional[str] = None
        self.replay_request_count = 0
        self.max_replay_requests = 10
        self.last_replay_request_time = 0.0
        self.replay_request_interval_seconds = 2.0

    def reconcile_notes_from_ledger(self) -> bool:
        if not hasattr(self, "ledger_file") or not self.ledger_file.exists():
            self.pending_reconciliation = False
            return True
        conn = None
        try:
            conn = sqlite3.connect(str(self.ledger_file), timeout=5.0)
            cur = conn.cursor()
            cur.execute("SELECT sequence_number, content FROM applied_segments ORDER BY sequence_number ASC;")
            rows = cur.fetchall()
            cur.close()
            conn.close()
            conn = None
            if not rows:
                is_legacy_upgrade = (
                    self.state_file.exists()
                    and getattr(self, "_ledger_was_new", False)
                )
                if is_legacy_upgrade:
                    legacy_seq = 0
                    try:
                        with open(self.state_file, "r", encoding="utf-8") as handle:
                            legacy_seq = int(json.load(handle).get("last_sequence_number", 0))
                    except Exception:
                        pass
                    if legacy_seq > 0:
                        has_usable_notes = (
                            self.notes_file.exists()
                            and bool(self.notes_file.read_text(encoding="utf-8").strip())
                        )
                        if has_usable_notes:
                            legacy_text = self.notes_file.read_text(encoding="utf-8")
                            conn = sqlite3.connect(str(self.ledger_file), timeout=5.0)
                            conn.execute(
                                "INSERT OR IGNORE INTO applied_segments (sequence_number, session_id, heading, content, created_at) VALUES (?, ?, ?, ?, ?)",
                                (legacy_seq, self.session_id, "Legacy Notes", legacy_text.strip(), time.time()),
                            )
                            conn.commit()
                            conn.close()
                            conn = None
                            self.pending_reconciliation = False
                            return True
                        else:
                            try:
                                with open(self.state_file, "w", encoding="utf-8") as handle:
                                    json.dump({"last_sequence_number": 0}, handle)
                            except Exception:
                                pass
                            self.pending_reconciliation = True
                            return False
                    self.pending_reconciliation = False
                    return True
                elif self.notes_file.exists():
                    try:
                        self.notes_file.unlink()
                    except Exception:
                        with open(self.notes_file, "w", encoding="utf-8") as handle:
                            handle.write("")
                self.pending_reconciliation = False
                return True
            segments_text = []
            for _, content in rows:
                if content and content.strip():
                    segments_text.append(f"\n{content.strip()}\n")
            full_markdown = "".join(segments_text)
            existing_text = self.notes_file.read_text(encoding="utf-8") if self.notes_file.exists() else None
            if existing_text != full_markdown:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                tmp_file = self.notes_file.with_suffix(f".tmp_{os.getpid()}")
                with open(tmp_file, "w", encoding="utf-8") as handle:
                    handle.write(full_markdown)
                    handle.flush()
                    try:
                        os.fsync(handle.fileno())
                    except Exception:
                        pass
                tmp_file.replace(self.notes_file)
            self.pending_reconciliation = False
            return True
        except Exception as err:
            logger.warning("failed to reconcile notes from ledger: %s", err)
            self.pending_reconciliation = True
            return False
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def _init_ledger(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        conn = None
        try:
            conn = sqlite3.connect(str(self.ledger_file), timeout=10.0)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS applied_segments (
                    sequence_number INTEGER PRIMARY KEY,
                    session_id TEXT,
                    heading TEXT,
                    content TEXT,
                    created_at REAL
                );
            """)
            conn.commit()
        except Exception as err:
            logger.warning("failed to initialize student ledger: %s", err)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def load_persisted_sequence(self) -> int:
        persisted = 0
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                    persisted = int(data.get("last_sequence_number", 0))
            except Exception:
                pass

        ledger_max: Optional[int] = None
        ledger_contiguous: Optional[int] = None
        if hasattr(self, "ledger_file") and self.ledger_file.exists():
            conn = None
            try:
                conn = sqlite3.connect(str(self.ledger_file), timeout=5.0)
                cur = conn.cursor()
                cur.execute("SELECT MAX(sequence_number) FROM applied_segments;")
                row = cur.fetchone()
                if row and row[0] is not None:
                    ledger_max = int(row[0])
                else:
                    ledger_max = 0
                cur.execute("SELECT sequence_number FROM applied_segments ORDER BY sequence_number ASC;")
                applied = [int(item[0]) for item in cur.fetchall()]
                ledger_contiguous = highest_contiguous_sequence(applied)
            except Exception as err:
                logger.error("reading student ledger failed, marking unverified: %s", err)
                self.pending_reconciliation = True
                return 0
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

        if getattr(self, "pending_reconciliation", False):
            if ledger_contiguous is not None:
                return min(persisted, ledger_contiguous)
            return 0

        if getattr(self, "_ledger_was_new", False) and (ledger_max is None or ledger_max == 0):
            has_usable_notes = (
                self.notes_file.exists()
                and bool(self.notes_file.read_text(encoding="utf-8").strip())
            )
            if not has_usable_notes and persisted > 0:
                self.pending_reconciliation = True
                return 0

        if ledger_contiguous is not None:
            # The ledger is the record of what was actually written to the
            # notes. Taking max(persisted, ledger_max) skipped every
            # sequence that was received but not persisted, so those notes
            # were never written.
            return ledger_contiguous
        return persisted

    def save_persisted_sequence(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.state_file, "w", encoding="utf-8") as handle:
                json.dump({"last_sequence_number": self.last_sequence_number}, handle)
        except Exception as error:
            logger.warning("failed to persist client sequence state: %s", error)

    async def _apply_segment(self, segment: GeneralNoteSegment) -> str:
        personalized_content = await self.rewriter.rewrite_segment(segment)
        conn = None
        try:
            conn = sqlite3.connect(str(self.ledger_file), timeout=10.0)
            cur = conn.cursor()
            cur.execute(
                "INSERT OR IGNORE INTO applied_segments (sequence_number, session_id, heading, content, created_at) VALUES (?, ?, ?, ?, ?)",
                (segment.sequence_number, segment.session_id, segment.heading, personalized_content, time.time())
            )
            conn.commit()
        except Exception as ledger_err:
            logger.error("saving to student ledger failed: %s", ledger_err)
            raise RuntimeError(f"saving to student ledger failed: {ledger_err}") from ledger_err
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

        if not self.reconcile_notes_from_ledger():
            self.pending_reconciliation = True
            raise RuntimeError("reconciling notes markdown file failed")

        self.pending_reconciliation = False
        self.received_segments.append(segment)
        self.last_sequence_number = max(self.last_sequence_number, segment.sequence_number)
        self.save_persisted_sequence()

        if self.student_graph is not None:
            for tag in segment.concept_tags:
                try:
                    self.student_graph.record_concept(tag)
                except Exception as graph_err:
                    logger.debug("failed to record concept in student graph: %s", graph_err)

        return personalized_content

    async def on_note_segment_received(
        self,
        segment: GeneralNoteSegment,
        websocket: Optional[Any] = None
    ) -> str:
        seq = segment.sequence_number
        if seq <= self.last_sequence_number and not getattr(self, "pending_reconciliation", False):
            return ""

        if seq > self.last_sequence_number + 1:
            self.pending_segments[seq] = segment
            if websocket is not None:
                await self.request_replay(websocket)
            return ""

        last_applied = await self._apply_segment(segment)
        next_expected = self.last_sequence_number + 1
        while next_expected in self.pending_segments:
            next_seg = self.pending_segments.pop(next_expected)
            last_applied = await self._apply_segment(next_seg)
            next_expected = self.last_sequence_number + 1

        return last_applied

    async def request_replay(self, websocket: Any) -> None:
        """Ask the gateway for the notes that are missing.

        The request used to be sent for every out-of-order segment with no
        cap and no backoff, so a gap that could never be filled (a note that
        never reached the durable store) turned into an endless request loop.
        """
        if self.replay_request_count >= self.max_replay_requests:
            if self.replay_request_count == self.max_replay_requests:
                logger.error(
                    "sequence gap at %d could not be filled after %d replay requests; "
                    "giving up on the replay",
                    self.last_sequence_number + 1, self.max_replay_requests
                )
                self.replay_request_count += 1
            return
        now = time.monotonic()
        if (now - self.last_replay_request_time) < self.replay_request_interval_seconds:
            return
        self.last_replay_request_time = now
        self.replay_request_count += 1
        gap_request = {
            "type": "request_replay",
            "last_sequence_number": self.last_sequence_number
        }
        try:
            await websocket.send(json.dumps(gap_request))
        except Exception:
            pass

    async def handle_server_frame(self, raw_frame: str, websocket: Optional[Any] = None) -> None:
        try:
            payload = json.loads(raw_frame)
            frame_type = payload.get("type")
            if frame_type == "note_segment":
                segment_data = payload.get("payload", {})
                segment = GeneralNoteSegment(**segment_data)
                await self.on_note_segment_received(segment, websocket)
                return
            status = payload.get("status")
            if status == "unauthorized":
                # The gateway rejected the handshake. Retrying every second
                # with the same credentials is pointless, so the error is
                # recorded and the reconnect loop stops.
                self.auth_error = str(payload.get("error", "unauthorized"))
                logger.error("gateway rejected the connection: %s", self.auth_error)
        except Exception as error:
            logger.warning("error parsing server frame: %s", error)

    async def run(
        self,
        stop_event: Optional[asyncio.Event] = None
    ) -> None:
        try:
            from websockets.asyncio.client import connect
        except ImportError:
            from websockets import connect

        self.is_running = True
        backoff = 1.0

        while self.is_running:
            if stop_event is not None and stop_event.is_set():
                break

            if getattr(self, "pending_reconciliation", False):
                if self.reconcile_notes_from_ledger():
                    self.pending_reconciliation = False
                    self.last_sequence_number = self.load_persisted_sequence()
                    self.save_persisted_sequence()

            hello_payload = {
                "type": "hello",
                "role": "student",
                "session_id": self.session_id,
                "client_id": self.profile.student_id,
                "last_sequence_number": self.last_sequence_number,
                "api_key": self.api_key
            }

            try:
                async with connect(self.server_url) as websocket:
                    await websocket.send(json.dumps(hello_payload))
                    backoff = 1.0
                    while self.is_running:
                        if stop_event is not None and stop_event.is_set():
                            return
                        try:
                            frame = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                            await self.handle_server_frame(frame, websocket)
                            if self.auth_error is not None:
                                logger.error(
                                    "stopping the student client: the gateway rejected the "
                                    "credentials (%s)", self.auth_error
                                )
                                return
                        except asyncio.TimeoutError:
                            if getattr(self, "pending_reconciliation", False):
                                if self.reconcile_notes_from_ledger():
                                    self.pending_reconciliation = False
                                    self.last_sequence_number = self.load_persisted_sequence()
                                    self.save_persisted_sequence()
                            continue
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if stop_event is not None and stop_event.is_set():
                    break
                logger.info("connection failed or dropped (%s), reconnecting in %.1fs", error, backoff)
                await asyncio.sleep(backoff)
                backoff = min(10.0, backoff * 1.5)
