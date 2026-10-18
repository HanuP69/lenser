import asyncio
import json
import logging
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
        missing_count = 0
        for tag in segment.concept_tags:
            clean_tag = tag.replace("_", " ").lower()
            if clean_tag not in lower_rewritten:
                missing_count += 1

        if segment.concept_tags and missing_count == len(segment.concept_tags):
            logger.warning("rewritten note omitted all concept tags, validation rejected")
            return False

        return True

    async def rewrite_segment(self, segment: GeneralNoteSegment) -> str:
        if self.llm_router is None or self.profile.preferred_language.lower() == "english":
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
        self.last_sequence_number = 0
        self.notes_file = output_dir / f"{session_id}_personalized_notes.md"
        self.received_segments: list[GeneralNoteSegment] = []
        self.is_running = False

    async def on_note_segment_received(self, segment: GeneralNoteSegment) -> str:
        if segment.sequence_number <= self.last_sequence_number:
            return ""

        personalized_content = await self.rewriter.rewrite_segment(segment)
        self.received_segments.append(segment)
        self.last_sequence_number = segment.sequence_number

        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.notes_file, "a", encoding="utf-8") as handle:
            handle.write(f"\n{personalized_content}\n")

        return personalized_content

    async def handle_server_frame(self, raw_frame: str) -> None:
        try:
            payload = json.loads(raw_frame)
            frame_type = payload.get("type")
            if frame_type == "note_segment":
                segment_data = payload.get("payload", {})
                segment = GeneralNoteSegment(**segment_data)
                await self.on_note_segment_received(segment)
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
        hello_payload = {
            "type": "hello",
            "role": "student",
            "session_id": self.session_id,
            "client_id": self.profile.student_id,
            "last_sequence_number": self.last_sequence_number,
            "api_key": self.api_key
        }

        async with connect(self.server_url) as websocket:
            await websocket.send(json.dumps(hello_payload))
            while self.is_running:
                if stop_event is not None and stop_event.is_set():
                    break
                try:
                    frame = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    await self.handle_server_frame(frame)
                except asyncio.TimeoutError:
                    continue
