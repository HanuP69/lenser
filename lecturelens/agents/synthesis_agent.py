"""SynthesisAgent powered by LangGraph StateGraph.

Synthesizes multimodal lecture events (transcripts, slides, figures) into
textbook-quality markdown notes with single-pass delimiter parsing (===NOTES_END===).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Optional, TypedDict

from langgraph.graph import StateGraph, END

from lecturelens.agents.base_agent import BaseAgent
from lecturelens.config.settings import Settings
from lecturelens.core.clock import SessionClock
from lecturelens.core.fileio import atomic_write_text
from lecturelens.core.logging_setup import get_logger
from lecturelens.core.redis_bus import RedisBus
from lecturelens.schemas.events import GeneralNoteSegment

logger = get_logger("lecturelens.synthesis_agent")

STRUCTURED_DELIMITER = "===NOTES_END==="
STRUCTURED_KEYS = ("definitions", "action_items", "key_points")
OVERLAP_WINDOW_SECONDS = 15.0

SYNTHESIS_PROMPT = """
You are an expert academic note taker synthesizing live lecture streams.
Turn the lecture events below into textbook-quality notes.

Write the answer only. Never write your reasoning, your self-checks, or
any commentary on these instructions; the reader sees nothing except
the notes and the structured block.

HOW THE ANSWER MUST LOOK (a program reads it, so follow it exactly):
1. The notes first, in markdown: headings (##), short paragraphs, bullets.
   Synthesize the spoken explanation with the text of the slides. Keep
   technical depth, and never invent facts that are not in the events.
2. Every marker such as [FIGURE fig_12: caption] names a picture of a
   diagram. The marker must appear once, on its own line, right after
   the paragraph or bullet that talks about that slide. Never omit one,
   never repeat one, never pile several together.
3. Every mathematical expression is written in LaTeX: inline between
   single dollar signs ($O(1)$, $H(x)$, $n \\log n$, $64.8\\%$), and
   display math on its own lines between double dollar signs:
   $$
   \\text{Index} = H(\\text{Key}) \\mod M
   $$
   A bare O(1), H(x) or 2^n left as plain text is wrong. Words that are
   not mathematics stay outside the dollar signs.
4. After the notes, one line that holds only this:
   ===NOTES_END===
5. After that line, one JSON object and nothing else:
   {"definitions": [{"term": "Term Name", "definition": "clear definition"}],
    "action_items": ["an action item, exam tip or reminder"],
    "key_points": ["a key point of this part of the lecture"]}
   Any of the three lists may be empty when the section has none.

PREVIOUS CONTEXT:
%%PREVIOUS_CONTEXT%%

CURRENT LECTURE EVENTS:
%%SLICE_TEXT%%
"""

REPAIR_PROMPT = """
The lecture notes below are good, but the structured block that was
supposed to follow them came out broken or is missing. Read the notes
and answer with one JSON object and nothing else - no commentary, no
code fence, no markdown:

{"definitions": [{"term": "Term Name", "definition": "clear definition"}],
 "action_items": ["an action item"],
 "key_points": ["a key point"]}

An empty list counts when the notes hold none.

NOTES:
%%NOTES%%
"""


class SynthesisState(TypedDict, total=False):
    session_id: str
    sequence_number: int
    running_context: str
    entries: list[dict[str, Any]]
    slice_text: str
    raw_response: str
    summary: str
    definitions: list[dict[str, str]]
    action_items: list[str]
    key_points: list[str]
    concept_tags: list[str]
    repair_attempts: int
    is_valid: bool
    empty_slice: bool


def structured_from_payload(payload: Any) -> Optional[dict[str, Any]]:
    if not isinstance(payload, dict) or not any(k in payload for k in STRUCTURED_KEYS):
        return None
    defs: list[dict[str, str]] = []
    for item in payload.get("definitions") or []:
        if isinstance(item, dict) and "term" in item and "definition" in item:
            defs.append({"term": str(item["term"]).strip(), "definition": str(item["definition"]).strip()})
    action_items = [str(i).strip() for i in payload.get("action_items") or [] if str(i).strip()]
    key_points = [str(i).strip() for i in payload.get("key_points") or [] if str(i).strip()]
    return {
        "definitions": defs,
        "action_items": action_items,
        "key_points": key_points,
        "concept_tags": [d["term"] for d in defs],
    }


def extract_structured_dict(raw: str) -> Optional[dict[str, Any]]:
    cleaned = raw.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()
    try:
        data = json.loads(cleaned)
        return structured_from_payload(data)
    except Exception:
        pass
    decoder = json.JSONDecoder()
    for m in reversed(list(re.finditer(r"\{", cleaned))):
        try:
            cand, _ = decoder.raw_decode(cleaned[m.start():])
            res = structured_from_payload(cand)
            if res:
                return res
        except ValueError:
            continue
    return None


def split_notes_and_structured(raw: str) -> tuple[str, Optional[dict[str, Any]]]:
    text = raw.strip()
    # Strip LLM think blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    marker = re.search(rf"^\s*{re.escape(STRUCTURED_DELIMITER)}\s*$", text, re.MULTILINE)
    if marker:
        return text[:marker.start()].strip(), extract_structured_dict(text[marker.end():])
    # Fallback search for trailing JSON block
    decoder = json.JSONDecoder()
    for match in reversed(list(re.finditer(r"\{", text))):
        try:
            cand, _ = decoder.raw_decode(text[match.start():])
            res = structured_from_payload(cand)
            if res is not None:
                notes_part = re.sub(rf"\s*{re.escape(STRUCTURED_DELIMITER)}\s*$", "", text[:match.start()]).strip()
                return notes_part, res
        except ValueError:
            continue
    return text, None


def build_running_context(previous: str, summary: str, max_chars: int = 1500, max_lines: int = 30) -> str:
    combined = (f"{previous}\n\n{summary}" if previous else summary).strip()
    lines = [ln for ln in combined.splitlines() if ln.strip()]
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    result = "\n".join(lines)
    if len(result) > max_chars:
        result = result[-max_chars:].lstrip()
    return result


def build_synthesis_graph(llm_router: Any) -> Any:
    """Build LangGraph StateGraph FSM for note synthesis."""

    async def align_context_node(state: SynthesisState) -> dict:
        entries = state.get("entries", [])
        lines: list[str] = []
        for e in entries:
            st = e.get("source_type", "")
            cnt = str(e.get("content", "")).strip()
            if not cnt:
                continue
            if st == "transcript":
                lines.append(f"[SPOKEN]: {cnt}")
            elif st == "slide":
                lines.append(f"[SLIDE]: {cnt}")
                fig = e.get("metadata", {}).get("figure_id")
                cap = e.get("metadata", {}).get("figure_caption", "")
                if fig:
                    lines.append(f"[FIGURE {fig}: {cap}]")
        if not lines:
            return {"empty_slice": True, "is_valid": True, "summary": ""}
        return {"slice_text": "\n".join(lines), "empty_slice": False}

    async def synthesize_notes_node(state: SynthesisState) -> dict:
        if state.get("empty_slice"):
            return {}
        prompt = SYNTHESIS_PROMPT.replace(
            "%%PREVIOUS_CONTEXT%%", state.get("running_context", "") or "None (beginning of lecture)"
        ).replace("%%SLICE_TEXT%%", state.get("slice_text", ""))
        try:
            from langchain_core.messages import HumanMessage
            response = await llm_router.ainvoke([HumanMessage(content=prompt)])
            raw_text = response.content if hasattr(response, "content") else str(response)
        except Exception as err:
            logger.warning("LLM synthesis invocation error: %s", err)
            raw_text = state.get("slice_text", "")
        return {"raw_response": raw_text}

    async def validate_node(state: SynthesisState) -> dict:
        if state.get("empty_slice"):
            return {"is_valid": True}
        raw = state.get("raw_response", "")
        notes_part, structured = split_notes_and_structured(raw)
        if structured is not None:
            return {
                "summary": notes_part,
                "definitions": structured["definitions"],
                "action_items": structured["action_items"],
                "key_points": structured["key_points"],
                "concept_tags": structured["concept_tags"],
                "is_valid": True,
            }
        return {
            "summary": notes_part,
            "is_valid": False,
            "repair_attempts": state.get("repair_attempts", 0) + 1,
        }

    async def repair_node(state: SynthesisState) -> dict:
        notes = state.get("summary", "")
        repair_prompt = REPAIR_PROMPT.replace("%%NOTES%%", notes)
        try:
            from langchain_core.messages import HumanMessage
            rep_res = await llm_router.ainvoke([HumanMessage(content=repair_prompt)])
            rep_text = rep_res.content if hasattr(rep_res, "content") else str(rep_res)
            parsed = extract_structured_dict(rep_text)
            if parsed is not None:
                return {
                    "definitions": parsed["definitions"],
                    "action_items": parsed["action_items"],
                    "key_points": parsed["key_points"],
                    "concept_tags": parsed["concept_tags"],
                    "is_valid": True,
                }
        except Exception as err:
            logger.warning("Repair call failed: %s", err)

        if state.get("repair_attempts", 0) >= 2:
            return {"definitions": [], "action_items": [], "key_points": [], "concept_tags": [], "is_valid": True}
        return {}

    def route_after_validate(state: SynthesisState) -> str:
        if state.get("empty_slice") or state.get("is_valid"):
            return END
        if state.get("repair_attempts", 0) < 2:
            return "repair"
        return END

    workflow = StateGraph(SynthesisState)
    workflow.add_node("align_context", align_context_node)
    workflow.add_node("synthesize_notes", synthesize_notes_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("repair", repair_node)

    workflow.set_entry_point("align_context")
    workflow.add_edge("align_context", "synthesize_notes")
    workflow.add_edge("synthesize_notes", "validate")
    workflow.add_conditional_edges("validate", route_after_validate, {"repair": "repair", END: END})
    workflow.add_edge("repair", "validate")

    return workflow.compile()


class SynthesisAgent(BaseAgent):
    """Note synthesizer coordinating StateGraph FSM, Markdown/HTML persistence, and Redis streams."""

    def __init__(
        self,
        settings: Settings,
        redis_bus: Optional[RedisBus] = None,
        session_clock: Optional[SessionClock] = None,
        session_id: str = "default",
        llm_router: Any = None,
    ) -> None:
        clock = session_clock or SessionClock()
        super().__init__(settings, redis_bus, clock, session_id, "structuring_agent")
        self.llm_router = llm_router
        self.notes_path = self.session_dir / "notes.md"
        self.notes_html_path = self.session_dir / "notes.html"
        self.notes_marker_path = self.session_dir / "control" / "notes_marker.txt"
        self.state_path = self.session_dir / "control" / "structuring_state.json"
        self.timeline_path = self.session_dir / "timeline.jsonl"
        self.quarantine_path = self.session_dir / "control" / "timeline_quarantine.jsonl"
        self.sequence_number = 0
        self.running_context = ""
        self.current_overlap: list[dict[str, Any]] = []
        self.synthesis_graph = build_synthesis_graph(llm_router) if llm_router else None
        self.load_state()

    def reset_outputs(self) -> None:
        for p in (self.notes_path, self.notes_html_path, self.notes_marker_path, self.state_path, self.quarantine_path):
            if p.exists():
                p.unlink(missing_ok=True)
        self.sequence_number = 0
        self.running_context = ""
        self.current_overlap = []

    def load_state(self) -> None:
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                self.running_context = data.get("running_context", "")
                self.sequence_number = data.get("sequence_number", 0)
            except Exception:
                pass

    def save_state(self, running_context: str, marker_offset: int, overlap: list[dict[str, Any]]) -> None:
        self.running_context = running_context
        atomic_write_text(
            self.state_path,
            json.dumps({"running_context": running_context, "sequence_number": self.sequence_number, "offset": marker_offset}),
        )
        atomic_write_text(self.notes_marker_path, str(marker_offset))

    async def synthesis(self, entries: list[dict[str, Any]], overlap: list[dict[str, Any]], running_context: str) -> dict[str, Any]:
        if not self.synthesis_graph:
            self.synthesis_graph = build_synthesis_graph(self.llm_router)
        state_in: SynthesisState = {
            "session_id": self.session_id,
            "sequence_number": self.sequence_number,
            "running_context": running_context,
            "entries": entries,
            "repair_attempts": 0,
        }
        res: SynthesisState = await self.synthesis_graph.ainvoke(state_in)
        summary = res.get("summary", "")
        new_context = build_running_context(running_context, summary)
        return {
            "summary": summary,
            "definitions": res.get("definitions", []),
            "action_items": res.get("action_items", []),
            "key_points": res.get("key_points", []),
            "concept_tags": res.get("concept_tags", []),
            "running_context": new_context,
            "empty_slice": res.get("empty_slice", False),
        }

    async def process_single_batch(self, batch: Any, running_context: str) -> str:
        entries = getattr(batch, "entries", batch)
        if not entries:
            return running_context
        res = await self.synthesis(entries, self.current_overlap, running_context)
        summary = res.get("summary", "")
        if not summary:
            return running_context

        # Section formatting
        t0 = float(entries[0].get("timestamp_epoch", time.time()))
        t1 = float(entries[-1].get("timestamp_epoch", t0))
        start_str = self.session_clock.format_epoch_as_elapsed(t0)
        end_str = self.session_clock.format_epoch_as_elapsed(t1)
        section = f"\n\n### {start_str} - {end_str}\n\n{summary}\n"

        with open(self.notes_path, "a", encoding="utf-8") as f:
            f.write(section)

        self.sequence_number += 1
        end_offset = getattr(batch, "end_offset", 0)
        self.save_state(res.get("running_context", running_context), end_offset, [])

        # Emit to Redis notes stream
        if self.redis_bus:
            heading_match = re.search(r"^##\s+(.+)$", summary, re.MULTILINE)
            heading = heading_match.group(1).strip() if heading_match else f"Section {self.sequence_number}"
            segment = GeneralNoteSegment(
                session_id=self.session_id,
                lecture_id=self.settings.course_name,
                sequence_number=self.sequence_number,
                timestamp_epoch=t0,
                start_time=start_str,
                end_time=end_str,
                heading=heading,
                content=section,
                key_points=res.get("key_points", []),
                concept_tags=res.get("concept_tags", []),
                slide_references=[],
                definitions=res.get("definitions", []),
            )
            await self.redis_bus.publish_resilient(f"lecturelens:{self.session_id}:stream:notes", segment)

        return str(res.get("running_context", running_context))

    async def process_all_pending(self) -> int:
        if not self.timeline_path.exists():
            return 0
        entries: list[dict[str, Any]] = []
        with open(self.timeline_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        pass
        if not entries:
            return 0
        from dataclasses import dataclass

        @dataclass
        class SimpleBatch:
            entries: list[dict[str, Any]]
            end_offset: int
            start_offset: int = 0

        batch = SimpleBatch(entries=entries, end_offset=self.timeline_path.stat().st_size)
        await self.process_single_batch(batch, self.running_context)
        return len(entries)
