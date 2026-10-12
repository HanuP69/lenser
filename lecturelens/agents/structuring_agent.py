import asyncio
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from lecturelens.config.settings import Settings
from lecturelens.core.agent_runtime import (
    AgentControl,
    FINALIZE_FLAG_NAME,
    RuntimeContext,
    execute_agent_main,
    run_standard_agent
)
from lecturelens.core.fileio import atomic_write_text
from lecturelens.core.logging_setup import get_logger
from lecturelens.core.markdown_formatter import format_clean_markdown
from lecturelens.exports.anki_exporter import create_anki_deck
from lecturelens.knowledge.concept_graph import ConceptGraph
from lecturelens.router.messages import response_to_text

logger = get_logger("lecturelens.structuring_agent")

REVISIT_TEXT_LIMIT = 200
FLAG_EXCERPT_LIMIT = 160
MAX_FLAG_EXCERPTS = 3
OVERLAP_WINDOW_SECONDS = 20.0
MAX_OVERLAP_ITEMS = 15


class StructuringState(TypedDict):
    timeline_slice: list[dict[str, Any]]
    running_context: str
    correlated_groups: list[dict[str, Any]]
    summary: str
    definitions: list[dict[str, str]]
    action_items: list[str]
    flashcards: list[dict[str, str]]
    importance_flags: list[str]
    overlap_entries: list[dict[str, Any]]


def extract_overlap_entries(
    entries: list[dict[str, Any]],
    window_seconds: float = OVERLAP_WINDOW_SECONDS
) -> list[dict[str, Any]]:
    if len(entries) == 0:
        return []
    max_timestamp = 0.0
    for entry in entries:
        ts = float(entry.get("timestamp_epoch", 0.0))
        if ts > max_timestamp:
            max_timestamp = ts

    cutoff = max_timestamp - window_seconds
    overlap: list[dict[str, Any]] = []
    for entry in entries:
        ts = float(entry.get("timestamp_epoch", 0.0))
        if ts >= cutoff:
            overlap.append(entry)
    if len(overlap) > MAX_OVERLAP_ITEMS:
        overlap = overlap[-MAX_OVERLAP_ITEMS:]
    return overlap


@dataclass
class SliceBatch:
    entries: list[dict[str, Any]]
    end_offset: int


def extract_json_payload(raw_output: str, expect_list: bool = True) -> Any:
    clean_json = raw_output.strip()
    if "```json" in clean_json:
        clean_json = clean_json.split("```json")[1].split("```")[0].strip()
    elif "```" in clean_json:
        clean_json = clean_json.split("```")[1].split("```")[0].strip()

    clean_no_trailing = re.sub(r",\s*([\]}])", r"\1", clean_json)
    for text in (clean_json, clean_no_trailing):
        try:
            val = json.loads(text)
            if expect_list and isinstance(val, dict):
                for k in ("flashcards", "definitions", "action_items", "items", "cards"):
                    if k in val and isinstance(val[k], list):
                        return val[k]
                if any(k in val for k in ("question", "term", "action_item")):
                    return [val]
            return val
        except (ValueError, TypeError):
            pass

    repaired = re.sub(r",\s*([\]}])", r"\1", raw_output)
    decoder = json.JSONDecoder()
    candidates: list[Any] = []
    for source in (raw_output, repaired):
        for i in range(len(source)):
            if source[i] in ("[", "{"):
                try:
                    obj, _ = decoder.raw_decode(source[i:])
                    if obj not in candidates:
                        candidates.append(obj)
                except Exception:
                    pass

    if expect_list:
        for c in candidates:
            if isinstance(c, list) and len(c) > 0 and isinstance(c[0], (dict, str)):
                return c
        for c in candidates:
            if isinstance(c, dict):
                for k in ("flashcards", "definitions", "action_items", "items", "cards"):
                    if k in c and isinstance(c[k], list):
                        return c[k]
                if any(k in c for k in ("question", "term", "action_item")):
                    return [c]

    for c in candidates:
        if isinstance(c, (list, dict)):
            return c

    if len(raw_output.strip()) > 0:
        logger.warning("failed to parse JSON payload from LLM response: %s", raw_output[:300])
    return None


def truncate_text(text: str, limit: int) -> str:
    flattened = " ".join(text.split())
    if len(flattened) <= limit:
        return flattened
    return flattened[:limit].rstrip() + "..."


def format_entry_line(item: dict[str, Any]) -> str:
    source = str(item.get("source_type", "unknown")).upper()
    text = str(item.get("content", "")).strip()
    metadata = item.get("metadata", {}) or {}
    if len(text) == 0:
        return ""
    revisited_number = int(metadata.get("revisited_slide_number", 0) or 0)
    if source == "SLIDE" and revisited_number > 0:
        text = f"(returned to earlier slide {revisited_number}) " + truncate_text(text, REVISIT_TEXT_LIMIT)
    flags = item.get("importance_flags", []) or []
    flag_text = ""
    if len(flags) > 0:
        flag_text = "[" + ",".join(flags) + "]"
    return f"[{source}]{flag_text} {text}"


def split_entries_into_batches(
    records: list[tuple[Optional[dict[str, Any]], int]],
    max_characters: int,
    final_offset: int
) -> list[SliceBatch]:
    batches: list[SliceBatch] = []
    current_entries: list[dict[str, Any]] = []
    current_characters = 0
    current_end_offset = 0
    for entry, end_offset in records:
        if entry is None:
            current_end_offset = end_offset
            continue
        entry_characters = len(str(entry.get("content", ""))) + 40
        if len(current_entries) > 0 and (current_characters + entry_characters) > max_characters:
            batches.append(SliceBatch(entries=current_entries, end_offset=current_end_offset))
            current_entries = []
            current_characters = 0
        current_entries.append(entry)
        current_characters += entry_characters
        current_end_offset = end_offset
    if len(current_entries) > 0:
        batches.append(SliceBatch(entries=current_entries, end_offset=final_offset))
    elif len(batches) > 0:
        batches[-1].end_offset = final_offset
    return batches


def create_structuring_graph(llm_router: Any) -> Any:
    async def call_llm(prompt: str) -> str:
        message = HumanMessage(content=prompt)
        response = await llm_router.ainvoke([message])
        return response_to_text(response)

    async def group_correlated_events_node(state: StructuringState) -> dict[str, Any]:
        slice_items = state.get("timeline_slice", [])
        groups_map: dict[str, list[dict[str, Any]]] = {}
        group_order: list[str] = []
        aggregated_groups: list[dict[str, Any]] = []

        for item in slice_items:
            correlation_id = item.get("correlation_id")
            if correlation_id is not None and len(correlation_id) > 0:
                if correlation_id not in groups_map:
                    groups_map[correlation_id] = []
                    group_order.append(correlation_id)
                    aggregated_groups.append({"correlation_id": correlation_id, "items": groups_map[correlation_id]})
                groups_map[correlation_id].append(item)
            else:
                aggregated_groups.append({"correlation_id": None, "items": [item]})

        all_flags: list[str] = []
        for item in slice_items:
            for flag in item.get("importance_flags", []) or []:
                if flag not in all_flags:
                    all_flags.append(flag)

        return {"correlated_groups": aggregated_groups, "importance_flags": all_flags}

    async def summarize_slice_node(state: StructuringState) -> dict[str, Any]:
        running_context = state.get("running_context", "")
        groups = state.get("correlated_groups", [])
        overlap_items = state.get("overlap_entries", [])

        overlap_lines: list[str] = []
        for item in overlap_items:
            line = format_entry_line(item)
            if len(line) > 0:
                overlap_lines.append(line)
        overlap_text = "\n".join(overlap_lines)

        content_lines: list[str] = []
        for group in groups:
            group_texts: list[str] = []
            for item in group.get("items", []):
                line = format_entry_line(item)
                if len(line) > 0:
                    group_texts.append(line)
            if len(group_texts) > 0:
                content_lines.append(" | ".join(group_texts))

        new_timeline_text = "\n".join(content_lines)
        if len(new_timeline_text.strip()) == 0:
            return {"summary": ""}

        overlap_block = ""
        if len(overlap_text.strip()) > 0:
            overlap_block = f"""
PREVIOUS OVERLAP BUFFER (tail of previous slice, already recorded):
{overlap_text}
(Note: Use this overlap only to complete cut-off thoughts and maintain continuity. Do not duplicate notes for content in this overlap.)
"""

        prompt = f"""
You are an expert academic note taker creating textbook-quality markdown lecture notes.
PREVIOUS LECTURE CONTEXT:
{running_context}
{overlap_block}
NEW LECTURE TIMELINE SLICE (transcript, slide OCR, board captures; lines tagged with importance flags):
{new_timeline_text}

FORMATTING GUIDELINES:
1. HEADINGS: Organize concepts and subtopics logically with clean Markdown headings (#### for major concepts, ##### for subsections) so they nest properly under the section header.
2. CONTINUITY: Seamless narrative from earlier notes. Never repeat titles with (Cont.) or (Continued).
3. NO EMOJIS: Never output emojis or icons.
4. MATH: Single dollar inline ($formula$). Display math on its own lines:
   $$
   \\text{{formula}}
   $$
   Never write $$...$$ inline with text. Group parameters into bullet lists.
5. EXAM RELEVANCE: Emphasize points flagged exam_relevant in bold.
6. LANGUAGE & TERMINOLOGY: Let the language of the lecture guide your writing. If the lecture is conducted in Hindi, Hinglish, Spanish, French, or another language, produce structured notes naturally in that language with technical accuracy, preserving relevant academic terminology.
7. ACCURACY: Do not invent facts not present in the content.
8. OUTPUT ONLY final markdown notes. Do NOT output scratchpads, drafting notes, or checklists. Start with the first heading.
"""
        summary_text = await call_llm(prompt)
        cleaned_summary = format_clean_markdown(summary_text).strip()
        return {"summary": cleaned_summary}

    async def extract_definitions_node(state: StructuringState) -> dict[str, Any]:
        summary_text = state.get("summary", "")
        if len(summary_text) == 0:
            return {"definitions": []}

        definition_lines: list[str] = []
        for item in state.get("timeline_slice", []):
            if "definition" in (item.get("importance_flags", []) or []):
                definition_lines.append(truncate_text(str(item.get("content", "")), 300))
        priority_text = "\n".join(definition_lines)

        prompt = f"""
From the following lecture note summary, extract all key technical terms, foundational concepts, and core analytical entities with their rigorous definitions and contextual explanations.
Prioritize terms from these spoken lines flagged as definitions:
{priority_text}

SUMMARY:
{summary_text}

Respond strictly in valid JSON format as a list of objects with 'term' and 'definition' keys:
[
  {{"term": "term or concept name", "definition": "detailed definition and academic context"}}
]
"""
        raw_output = await call_llm(prompt)
        parsed = extract_json_payload(raw_output)
        definitions_list: list[dict[str, str]] = []
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "term" in item and "definition" in item:
                    definitions_list.append({
                        "term": str(item["term"]),
                        "definition": str(item["definition"])
                    })
        return {"definitions": definitions_list}

    async def extract_action_items_node(state: StructuringState) -> dict[str, Any]:
        summary_text = state.get("summary", "")
        if len(summary_text) == 0:
            return {"action_items": []}

        prompt = f"""
From the following lecture summary, extract any student action items, homework assignments, upcoming exam dates, or deadlines mentioned.
SUMMARY:
{summary_text}

Respond strictly in valid JSON format as a list of strings:
["action item 1", "action item 2"]
If none, return [].
"""
        raw_output = await call_llm(prompt)
        parsed = extract_json_payload(raw_output)
        action_items: list[str] = []
        if isinstance(parsed, list):
            for item in parsed:
                action_items.append(str(item))
        return {"action_items": action_items}

    async def generate_flashcards_node(state: StructuringState) -> dict[str, Any]:
        definitions = state.get("definitions", [])
        summary_text = state.get("summary", "")
        flashcards: list[dict[str, str]] = []

        if len(summary_text) == 0 and len(definitions) == 0:
            return {"flashcards": []}

        exam_lines: list[str] = []
        for item in state.get("timeline_slice", []):
            if "exam_relevant" in (item.get("importance_flags", []) or []):
                exam_lines.append(truncate_text(str(item.get("content", "")), 300))
        exam_text = "\n".join(exam_lines) if len(exam_lines) > 0 else "None explicitly flagged in this slice."

        definitions_context_lines: list[str] = []
        for defn in definitions:
            term = str(defn.get("term", "")).strip()
            explanation = str(defn.get("definition", "")).strip()
            if len(term) > 0 and len(explanation) > 0:
                definitions_context_lines.append(f"- {term}: {explanation}")
        definitions_text = "\n".join(definitions_context_lines) if len(definitions_context_lines) > 0 else "None extracted in this slice."

        prompt = f"""
Generate 3-5 high-yield study flashcards based on this lecture summary and concepts.

LECTURE SUMMARY:
{summary_text}

KEY CONCEPTS & DEFINITIONS DISCUSSED:
{definitions_text}

EXAM-RELEVANT POINTS FLAGGED BY INSTRUCTOR:
{exam_text}

FLASHCARD QUALITY & FORMULATION REQUIREMENTS:
1. STRICT PROHIBITION ON ROTE DEFINITION QUESTIONS:
   - Do NOT write generic "What is the definition of [X]?", "Define [X]", or "What is [Person/Term]?" questions.
   - Adapt question formulations to the subject domain and concept type:
     * Humanities & Literature: Analyze character dynamics, authorial intent, thematic conflicts, narrative techniques, and textual subtext.
     * Science & Engineering: Probe underlying physical/computational mechanisms, governing laws, boundary conditions, trade-offs, and failure modes.
     * Mathematics & Formal Logic: Focus on proof techniques, invariant preservation, edge cases, and structural properties.
   - Frame questions that test active recall, analytical comprehension, cause-and-effect, and practical/exam application.

2. COMPREHENSIVE, DEEP, DETAILED ANSWERS:
   - Answers must NOT be shallow one-liners, single-word responses, or superficial fragments.
   - Each answer should provide a thorough, multi-sentence breakdown (typically 2 to 4 substantive sentences or clear bullet points) covering:
     * Direct Core Answer: The clear, direct explanation answering the question.
     * Context & Mechanism / Textual Evidence: The underlying reasoning, mathematical derivation, or narrative evidence.
     * Analytical Significance / Takeaway: Why this concept matters, common traps, or exam significance.

3. ACCURACY & FORMATTING:
   - Use standard LaTeX math ($inline$ or $$display$$) where appropriate.
   - Do NOT invent facts not supported by the lecture summary or context.

Respond strictly in valid JSON format as a list of objects with 'question' and 'answer':
[
  {{"question": "Deep, context-rich active recall question", "answer": "Comprehensive multi-sentence answer with core explanation, context/mechanism, and analytical takeaway."}}
]
"""
        raw_output = await call_llm(prompt)
        parsed = extract_json_payload(raw_output)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "question" in item and "answer" in item:
                    q = str(item["question"]).strip()
                    a = str(item["answer"]).strip()
                    if len(q) > 0 and len(a) > 0:
                        flashcards.append({
                            "question": q,
                            "answer": a
                        })

        return {"flashcards": flashcards}

    async def update_running_context_node(state: StructuringState) -> dict[str, Any]:
        previous_context = state.get("running_context", "")
        summary_text = state.get("summary", "")
        if len(summary_text) == 0:
            return {"running_context": previous_context}

        prompt = f"""
Compress the following lecture notes into a dense running context of at most 300 words.
Keep key topics, definitions, and the current thread of the lecture.

EARLIER CONTEXT:
{previous_context}

NEW NOTES:
{summary_text}
"""
        compressed = await call_llm(prompt)
        return {"running_context": compressed}

    graph_builder = StateGraph(StructuringState)
    graph_builder.add_node("group_correlated_events", group_correlated_events_node)
    graph_builder.add_node("summarize_slice", summarize_slice_node)
    graph_builder.add_node("extract_definitions", extract_definitions_node)
    graph_builder.add_node("extract_action_items", extract_action_items_node)
    graph_builder.add_node("generate_flashcards", generate_flashcards_node)
    graph_builder.add_node("update_running_context", update_running_context_node)

    graph_builder.add_edge(START, "group_correlated_events")
    graph_builder.add_edge("group_correlated_events", "summarize_slice")
    graph_builder.add_edge("summarize_slice", "extract_definitions")
    graph_builder.add_edge("extract_definitions", "extract_action_items")
    graph_builder.add_edge("extract_action_items", "generate_flashcards")
    graph_builder.add_edge("generate_flashcards", "update_running_context")
    graph_builder.add_edge("update_running_context", END)
    return graph_builder.compile()


class StructuringAgent:
    def __init__(
        self,
        settings: Settings,
        session_id: str,
        llm_router: Any,
        redis_bus: Any = None
    ) -> None:
        self.settings: Settings = settings
        self.session_id: str = session_id
        self.llm_router: Any = llm_router
        self.redis_bus: Any = redis_bus
        self.is_running: bool = False

        session_dir = settings.create_session_directories(session_id)
        self.session_dir: Path = session_dir
        self.timeline_path: Path = session_dir / "timeline.jsonl"
        self.notes_path: Path = session_dir / "notes.md"
        self.context_path: Path = session_dir / "context.json"
        self.notes_marker_path: Path = session_dir / "notes_marker.json"
        self.flashcards_path: Path = session_dir / "flashcards.jsonl"
        self.anki_path: Path = session_dir / "flashcards.apkg"
        self.control: AgentControl = AgentControl(session_dir, "structuring_agent", FINALIZE_FLAG_NAME)

        self.concept_graph: ConceptGraph = ConceptGraph(session_dir / "concepts.db")
        self.course_graph: ConceptGraph = ConceptGraph(
            settings.get_course_database_path(settings.course_name)
        )
        self.graph: Any = create_structuring_graph(llm_router)
        self.consecutive_failures: int = 0
        initial_state_data = self.load_state()
        self.current_overlap: list[dict[str, Any]] = list(initial_state_data.get("overlap_entries", []))

    def load_notes_marker(self) -> int:
        if not self.notes_marker_path.exists():
            return -1
        try:
            with open(self.notes_marker_path, "r", encoding="utf-8") as file_handle:
                return int(json.load(file_handle).get("last_end_offset", -1))
        except (ValueError, OSError, TypeError, AttributeError):
            return -1

    def save_notes_marker(self, end_offset: int) -> None:
        atomic_write_text(self.notes_marker_path, json.dumps({"last_end_offset": end_offset}))

    def load_state(self) -> dict[str, Any]:
        if not self.context_path.exists():
            return {"running_context": "", "processed_byte_offset": 0, "overlap_entries": []}
        try:
            with open(self.context_path, "r", encoding="utf-8") as file_handle:
                data = json.load(file_handle)
        except (ValueError, OSError):
            return {"running_context": "", "processed_byte_offset": 0, "overlap_entries": []}
        return {
            "running_context": str(data.get("running_context", "")),
            "processed_byte_offset": int(data.get("processed_byte_offset", 0)),
            "overlap_entries": list(data.get("overlap_entries", []))
        }

    def save_state(
        self,
        running_context: str,
        processed_byte_offset: int,
        overlap_entries: Optional[list[dict[str, Any]]] = None
    ) -> None:
        payload = {
            "running_context": running_context[: self.settings.context_max_characters],
            "processed_byte_offset": processed_byte_offset,
            "overlap_entries": overlap_entries or []
        }
        atomic_write_text(self.context_path, json.dumps(payload, ensure_ascii=False))

    def read_pending_records(self, start_offset: int) -> tuple[list[tuple[Optional[dict[str, Any]], int]], int]:
        if not self.timeline_path.exists():
            return [], start_offset
        with open(self.timeline_path, "rb") as file_handle:
            file_handle.seek(start_offset)
            raw_bytes = file_handle.read()
        last_newline = raw_bytes.rfind(b"\n")
        if last_newline < 0:
            return [], start_offset
        complete_bytes = raw_bytes[: last_newline + 1]
        records: list[tuple[Optional[dict[str, Any]], int]] = []
        cursor = start_offset
        for raw_line in complete_bytes[:-1].split(b"\n"):
            if len(raw_line) == 0:
                cursor += 1
                continue
            cursor_after = cursor + len(raw_line) + 1
            parsed: Optional[dict[str, Any]] = None
            try:
                candidate = json.loads(raw_line.decode("utf-8"))
                if isinstance(candidate, dict):
                    parsed = candidate
            except (ValueError, UnicodeDecodeError):
                logger.warning("skipping corrupt timeline line at byte %d", cursor)
            records.append((parsed, cursor_after))
            cursor = cursor_after
        return records, start_offset + len(complete_bytes)

    def read_pending_batches(self) -> list[SliceBatch]:
        state = self.load_state()
        start_offset = state["processed_byte_offset"]
        records, final_offset = self.read_pending_records(start_offset)
        if final_offset == start_offset:
            return []
        batches = split_entries_into_batches(
            records, self.settings.max_slice_characters, final_offset
        )
        if len(batches) == 0:
            self.save_state(state["running_context"], final_offset, self.current_overlap)
        for batch in batches:
            batch.entries.sort(key=lambda entry: float(entry.get("timestamp_epoch", 0.0)))
        return batches

    def collect_flag_excerpts(self, entries: list[dict[str, Any]], flag_name: str) -> list[str]:
        excerpts: list[str] = []
        for entry in entries:
            if flag_name not in (entry.get("importance_flags", []) or []):
                continue
            if entry.get("source_type") != "transcript":
                continue
            excerpts.append(truncate_text(str(entry.get("content", "")), FLAG_EXCERPT_LIMIT))
            if len(excerpts) >= MAX_FLAG_EXCERPTS:
                break
        return excerpts

    def build_notes_section(
        self,
        batch: SliceBatch,
        result_state: dict[str, Any],
        previous_mentions: list[str]
    ) -> str:
        summary = str(result_state.get("summary", "")).strip()
        if len(summary) == 0:
            return ""
        cleaned_summary = format_clean_markdown(summary).strip()
        first_timestamp = float(batch.entries[0].get("timestamp_epoch", 0.0))
        heading_time = time.strftime("%H:%M:%S", time.localtime(first_timestamp))
        divider = "\n---\n\n" if (self.notes_path.exists() and self.notes_path.stat().st_size > 0) else ""
        section_lines: list[str] = [f"{divider}### {heading_time}", ""]

        exam_excerpts = self.collect_flag_excerpts(batch.entries, "exam_relevant")
        if len(exam_excerpts) > 0:
            section_lines.append("> **Important Exam Topic**")
            for excerpt in exam_excerpts:
                section_lines.append(f"> - {excerpt}")
            section_lines.append("")

        confusion_excerpts = self.collect_flag_excerpts(batch.entries, "confusion")
        for excerpt in confusion_excerpts:
            section_lines.append(f"> **Review**: {excerpt}")
        if len(confusion_excerpts) > 0:
            section_lines.append("")

        section_lines.append(cleaned_summary)
        section_lines.append("")

        for mention in previous_mentions:
            section_lines.append(f"> Previously covered in {mention}")
        if len(previous_mentions) > 0:
            section_lines.append("")

        action_items = result_state.get("action_items", []) or []
        if len(action_items) > 0:
            section_lines.append("**Action Items**")
            for action_item in action_items:
                section_lines.append(f"- {action_item}")
            section_lines.append("")
        return "\n".join(section_lines) + "\n"

    def find_previous_mentions(self, definitions: list[dict[str, str]]) -> list[str]:
        mentions: list[str] = []
        for definition_item in definitions:
            term = definition_item["term"]
            earlier = self.course_graph.find_concept_in_other_sessions(term, self.session_id)
            if len(earlier) == 0:
                continue
            sessions: list[str] = []
            for record in earlier:
                if record["session_id"] not in sessions:
                    sessions.append(record["session_id"])
            mentions.append(f"{', '.join(sessions)}: {term}")
        return mentions

    def record_concepts(self, definitions: list[dict[str, str]], timestamp_epoch: float) -> None:
        term_names: list[str] = []
        for definition_item in definitions:
            term_names.append(definition_item["term"])
        for definition_item in definitions:
            related: list[str] = []
            for other_term in term_names:
                if other_term != definition_item["term"]:
                    related.append(other_term)
            self.concept_graph.add_or_update_concept(
                concept_name=definition_item["term"],
                session_id=self.session_id,
                timestamp_epoch=timestamp_epoch,
                definition=definition_item["definition"],
                related_concepts=related,
                course_name=self.settings.course_name
            )
            self.course_graph.add_or_update_concept(
                concept_name=definition_item["term"],
                session_id=self.session_id,
                timestamp_epoch=timestamp_epoch,
                definition=definition_item["definition"],
                related_concepts=related,
                course_name=self.settings.course_name
            )

    def load_flashcards(self) -> list[dict[str, str]]:
        cards: list[dict[str, str]] = []
        if not self.flashcards_path.exists():
            return cards
        with open(self.flashcards_path, "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                stripped = line.strip()
                if len(stripped) == 0:
                    continue
                try:
                    cards.append(json.loads(stripped))
                except ValueError:
                    continue
        return cards

    def append_flashcards(self, new_cards: list[dict[str, str]]) -> None:
        existing_questions: set[str] = set()
        for card in self.load_flashcards():
            existing_questions.add(str(card.get("question", "")))
        lines: list[str] = []
        for card in new_cards:
            question = str(card.get("question", ""))
            if question in existing_questions:
                continue
            existing_questions.add(question)
            lines.append(json.dumps(card, ensure_ascii=False))
        if len(lines) == 0:
            return
        with open(self.flashcards_path, "a", encoding="utf-8", newline="\n") as file_handle:
            file_handle.write("\n".join(lines) + "\n")

    def regenerate_anki_deck(self) -> None:
        cards = self.load_flashcards()
        if len(cards) == 0:
            return
        create_anki_deck(
            deck_name=f"LectureLens {self.session_id}",
            flashcards=cards,
            output_file_path=self.anki_path
        )

    async def process_single_batch(self, batch: SliceBatch, running_context: str) -> str:
        initial_state: StructuringState = {
            "timeline_slice": batch.entries,
            "running_context": running_context,
            "correlated_groups": [],
            "summary": "",
            "definitions": [],
            "action_items": [],
            "flashcards": [],
            "importance_flags": [],
            "overlap_entries": self.current_overlap
        }
        result_state = await self.graph.ainvoke(initial_state)

        definitions = result_state.get("definitions", []) or []
        previous_mentions = self.find_previous_mentions(definitions)
        section_text = self.build_notes_section(batch, result_state, previous_mentions)
        if len(section_text) > 0:
            if self.notes_path.exists() and self.load_notes_marker() >= batch.end_offset:
                logger.info(
                    "batch ending at byte %d already written to notes, skipping append", batch.end_offset
                )
            else:
                with open(self.notes_path, "a", encoding="utf-8", newline="\n") as file_handle:
                    file_handle.write(section_text)
                self.save_notes_marker(batch.end_offset)

        self.append_flashcards(result_state.get("flashcards", []) or [])
        first_timestamp = float(batch.entries[0].get("timestamp_epoch", 0.0))
        self.record_concepts(definitions, first_timestamp)
        self.regenerate_anki_deck()

        next_overlap = extract_overlap_entries(batch.entries, OVERLAP_WINDOW_SECONDS)
        self.current_overlap = next_overlap

        new_context = str(result_state.get("running_context", running_context))
        self.save_state(new_context, batch.end_offset, next_overlap)
        return new_context

    async def process_all_pending(self) -> int:
        processed_batches = 0
        for _ in range(1000):
            batches = self.read_pending_batches()
            if len(batches) == 0:
                break
            state = self.load_state()
            running_context = state["running_context"]
            for batch in batches:
                running_context = await self.process_single_batch(batch, running_context)
                processed_batches += 1
                self.control.set_stat("batches_processed", processed_batches)
        return processed_batches

    async def process_batch(self) -> int:
        return await self.process_all_pending()

    def reset_outputs(self) -> None:
        for path in (self.notes_path, self.context_path, self.flashcards_path, self.anki_path, self.notes_marker_path):
            if path.exists():
                path.unlink()
        self.current_overlap = []
        self.concept_graph.delete_session(self.session_id)
        self.course_graph.delete_session(self.session_id)

    async def run_guarded_pass(self) -> None:
        try:
            await self.process_all_pending()
            self.consecutive_failures = 0
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.consecutive_failures += 1
            self.control.set_stat("consecutive_failures", self.consecutive_failures)
            logger.error("structuring pass failed (%d), will retry: %s", self.consecutive_failures, error)

    async def run(self) -> None:
        self.is_running = True
        interval = float(self.settings.structuring_interval_seconds)
        last_pass_time = time.monotonic()
        heartbeat_interval = 2.0
        last_heartbeat_time = 0.0
        while self.is_running:
            now = time.monotonic()
            if (now - last_heartbeat_time) >= heartbeat_interval:
                self.control.write_heartbeat("live")
                last_heartbeat_time = now
            if self.control.stop_requested():
                logger.info("finalize requested, running final structuring pass")
                await self.run_guarded_pass()
                return
            if (now - last_pass_time) >= interval:
                await self.run_guarded_pass()
                last_pass_time = time.monotonic()
            await asyncio.sleep(1.0)

    def stop(self) -> None:
        self.is_running = False


async def main() -> None:
    def build_agent(context: RuntimeContext) -> StructuringAgent:
        from lecturelens.router.llm_router import build_router_for_task, load_router_config
        router_config = load_router_config(context.settings.router_config_path)
        router = build_router_for_task(
            "structuring",
            context.redis_bus.client,
            router_config,
            context.settings.ollama_url,
            context.settings.llm_mode,
            context.settings.llm_timeout_seconds
        )
        return StructuringAgent(
            settings=context.settings,
            session_id=context.session_id,
            llm_router=router,
            redis_bus=context.redis_bus
        )

    await run_standard_agent(build_agent)


if __name__ == "__main__":
    execute_agent_main(main)
