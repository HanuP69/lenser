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
from typing import Any, Callable, Optional

from lecturelens.config.settings import Settings
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
        return (
            f"{prompt_intro}\n\n"
            f"SOURCE SECTION:\n"
            f"Heading: {segment.heading}\n"
            f"Key Points: {json.dumps(segment.key_points)}\n"
            f"Concepts: {', '.join(segment.concept_tags)}\n\n"
            f"SOURCE CONTENT:\n{segment.content}\n"
        )

    def validate_rewrite(self, segment: GeneralNoteSegment, rewritten_text: str) -> bool:
        if not rewritten_text or len(rewritten_text.strip()) < 20:
            return False

        lower_rewritten = rewritten_text.lower()
        if segment.concept_tags:
            present_tags = 0
            for tag in segment.concept_tags:
                clean_tag = tag.replace("_", " ").lower()
                if clean_tag in lower_rewritten:
                    present_tags += 1
            if present_tags < math.ceil(len(segment.concept_tags) * 0.5):
                logger.warning("rewritten note omitted over half of the concept tags, rejected")
                return False

        if segment.start_time and segment.start_time in segment.heading:
            if segment.start_time not in rewritten_text:
                logger.warning("rewritten note omitted timestamp %s, rejected", segment.start_time)
                return False

        if segment.key_points:
            found_kp = False
            for kp in segment.key_points:
                words = [w.lower() for w in re.findall(r"\b\w{4,}\b", kp)]
                if words and any(w in lower_rewritten for w in words):
                    found_kp = True
                    break
            if not found_kp:
                logger.warning("rewritten note omitted all key points content, rejected")
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
        self._init_ledger()
        self.notes_file = output_dir / f"{session_id}_personalized_notes.md"
        reconciled = self.reconcile_notes_from_ledger()
        self.pending_reconciliation = not reconciled
        self.last_sequence_number = self.load_persisted_sequence()
        self.received_segments: list[GeneralNoteSegment] = []
        self.pending_segments: dict[int, GeneralNoteSegment] = {}
        self.is_running = False

    def reconcile_notes_from_ledger(self) -> bool:
        if not hasattr(self, "ledger_file") or not self.ledger_file.exists():
            self.pending_reconciliation = False
            return True
        try:
            conn = sqlite3.connect(str(self.ledger_file), timeout=5.0)
            cur = conn.cursor()
            cur.execute("SELECT sequence_number, content FROM applied_segments ORDER BY sequence_number ASC;")
            rows = cur.fetchall()
            conn.close()
            if not rows:
                if self.notes_file.exists():
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

    def _init_ledger(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
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
            conn.close()
        except Exception as err:
            logger.warning("failed to initialize student ledger: %s", err)

    def load_persisted_sequence(self) -> int:
        persisted = 0
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                    persisted = int(data.get("last_sequence_number", 0))
            except Exception:
                pass
        if getattr(self, "pending_reconciliation", False):
            return persisted
        if hasattr(self, "ledger_file") and self.ledger_file.exists():
            try:
                conn = sqlite3.connect(str(self.ledger_file), timeout=5.0)
                cur = conn.cursor()
                cur.execute("SELECT MAX(sequence_number) FROM applied_segments;")
                row = cur.fetchone()
                conn.close()
                if row and row[0] is not None:
                    persisted = max(persisted, int(row[0]))
            except Exception:
                pass
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
        try:
            conn = sqlite3.connect(str(self.ledger_file), timeout=10.0)
            cur = conn.cursor()
            cur.execute(
                "INSERT OR IGNORE INTO applied_segments (sequence_number, session_id, heading, content, created_at) VALUES (?, ?, ?, ?, ?)",
                (segment.sequence_number, segment.session_id, segment.heading, personalized_content, time.time())
            )
            conn.commit()
            conn.close()
        except Exception as ledger_err:
            logger.error("saving to student ledger failed: %s", ledger_err)
            raise RuntimeError(f"saving to student ledger failed: {ledger_err}") from ledger_err

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
                gap_request = {
                    "type": "request_replay",
                    "last_sequence_number": self.last_sequence_number
                }
                try:
                    await websocket.send(json.dumps(gap_request))
                except Exception:
                    pass
            return ""

        last_applied = await self._apply_segment(segment)
        next_expected = self.last_sequence_number + 1
        while next_expected in self.pending_segments:
            next_seg = self.pending_segments.pop(next_expected)
            last_applied = await self._apply_segment(next_seg)
            next_expected = self.last_sequence_number + 1

        return last_applied

    async def handle_server_frame(self, raw_frame: str, websocket: Optional[Any] = None) -> None:
        try:
            payload = json.loads(raw_frame)
            frame_type = payload.get("type")
            if frame_type == "note_segment":
                segment_data = payload.get("payload", {})
                segment = GeneralNoteSegment(**segment_data)
                await self.on_note_segment_received(segment, websocket)
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
                        except asyncio.TimeoutError:
                            continue
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if stop_event is not None and stop_event.is_set():
                    break
                logger.info("connection failed or dropped (%s), reconnecting in %.1fs", error, backoff)
                await asyncio.sleep(backoff)
                backoff = min(10.0, backoff * 1.5)
