import asyncio
import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
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
        llm_router: Any = None
    ) -> None:
        self.server_url = server_url
        self.session_id = session_id
        self.profile = profile
        self.output_dir = output_dir
        self.api_key = api_key
        self.rewriter = PersonalizationEngine(profile, llm_router)
        self.state_file = output_dir / f"{session_id}_{profile.student_id}_state.json"
        self.last_sequence_number = self.load_persisted_sequence()
        self.notes_file = output_dir / f"{session_id}_personalized_notes.md"
        self.received_segments: list[GeneralNoteSegment] = []
        self.pending_segments: dict[int, GeneralNoteSegment] = {}
        self.is_running = False

    def load_persisted_sequence(self) -> int:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                    return int(data.get("last_sequence_number", 0))
            except Exception:
                pass
        return 0

    def save_persisted_sequence(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.state_file, "w", encoding="utf-8") as handle:
                json.dump({"last_sequence_number": self.last_sequence_number}, handle)
        except Exception as error:
            logger.warning("failed to persist client sequence state: %s", error)

    async def _apply_segment(self, segment: GeneralNoteSegment) -> str:
        personalized_content = await self.rewriter.rewrite_segment(segment)
        self.received_segments.append(segment)
        self.last_sequence_number = segment.sequence_number
        self.save_persisted_sequence()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.notes_file, "a", encoding="utf-8") as handle:
            handle.write(f"\n{personalized_content}\n")
        return personalized_content

    async def on_note_segment_received(
        self,
        segment: GeneralNoteSegment,
        websocket: Optional[Any] = None
    ) -> str:
        seq = segment.sequence_number
        if seq <= self.last_sequence_number:
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
