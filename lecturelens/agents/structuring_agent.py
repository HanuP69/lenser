import asyncio
import hashlib
import json
import os
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
from lecturelens.core.clock import SessionClock
from lecturelens.core.concurrency import run_in_thread_pool
from lecturelens.core.fileio import append_text_durably, atomic_write_text
from lecturelens.core.drift_filter import ContextDriftFilter
from lecturelens.core.logging_setup import get_logger
from lecturelens.core.markdown_formatter import format_clean_markdown
from lecturelens.router.messages import response_to_text
from lecturelens.schemas.events import GeneralNoteSegment

logger = get_logger("lecturelens.structuring_agent")

REVISIT_TEXT_LIMIT = 200
FLAG_EXCERPT_LIMIT = 160
MAX_FLAG_EXCERPTS = 3
OVERLAP_WINDOW_SECONDS = 20.0
MAX_OVERLAP_ITEMS = 15
MAX_BATCH_RETRIES = 3
MAX_CONSECUTIVE_PASS_FAILURES = 10
# How often a batch is split in half when the provider says that the
# message is too long. The limit of a message is different for every
# provider, so the batch is made smaller until it fits.
MAX_BATCH_SPLITS = 6


def is_message_too_long_error(error: BaseException) -> bool:
    """Whether the provider refused the message because it was too long.

    The free providers put a limit on the length of a single message. A
    long batch is refused with a plain error, and the whole lecture would
    be left without notes. The batch is split instead.
    """
    message = str(error).lower()
    if "limits the length of a single message" in message:
        return True
    if "shorten your last message" in message:
        return True
    for signature in (
        "context length",
        "maximum context",
        "too many tokens",
        "request too large",
        "payload too large",
        "input is too long",
        "prompt is too long",
        "string too long",
    ):
        if signature in message:
            return True
    return False


class StructuringBatchError(RuntimeError):
    """Raised when a batch could not be synthesised into notes.

    The batch is retried; the byte offset is not advanced until the batch
    either succeeds or falls back to the raw timeline content.
    """


class StructuringState(TypedDict):
    timeline_slice: list[dict[str, Any]]
    running_context: str
    correlated_groups: list[dict[str, Any]]
    summary: str
    definitions: list[dict[str, str]]
    action_items: list[str]
    key_points: list[str]
    concept_tags: list[str]
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
    # Where the batch starts in the timeline file. It does not change when
    # more events arrive, so it identifies the batch across retries. The end
    # offset does change, because the last event of the batch may be a new
    # one, and a counter keyed by it never reached the retry limit.
    start_offset: int = 0


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
                for k in ("definitions", "action_items", "key_points", "concept_tags", "items", "flashcards"):
                    if k in val and isinstance(val[k], list):
                        return val[k]
                if any(k in val for k in ("term", "action_item", "concept", "question")):
                    return [val]
            return val
        except (ValueError, TypeError):
            pass

    repaired = re.sub(r",\s*([\]}])", r"\1", raw_output)
    decoder = json.JSONDecoder()
    pos = 0
    best_candidate: Any = None
    while pos < len(repaired):
        m = re.search(r"(\[|\{)", repaired[pos:])
        if not m:
            break
        start_idx = pos + m.start()
        pos = start_idx + 1
        try:
            obj, _ = decoder.raw_decode(repaired, idx=start_idx)
            if isinstance(obj, list):
                if any(isinstance(x, (dict, str)) for x in obj):
                    return obj
                if best_candidate is None:
                    best_candidate = obj
            elif isinstance(obj, dict):
                if expect_list:
                    for k in ("definitions", "action_items", "key_points", "concept_tags", "items", "flashcards"):
                        if k in obj and isinstance(obj[k], list):
                            return obj[k]
                    if any(k in obj for k in ("term", "action_item", "concept", "question")):
                        return [obj]
                    if best_candidate is None:
                        best_candidate = [obj]
                else:
                    return obj
        except ValueError:
            continue
    if best_candidate is not None:
        return best_candidate
    return None


def truncate_text(text: str, limit: int) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "..."


def format_group_content(group: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in group.get("items", []):
        source_type = item.get("source_type", "")
        content = item.get("content", "")
        metadata = item.get("metadata", {}) or {}
        revisited = int(metadata.get("revisited_slide_number", 0) or 0)
        flags = item.get("importance_flags", [])
        flag_prefix = f"[{','.join(flags)}]" if flags else ""

        if source_type == "transcript":
            parts.append(f"[TRANSCRIPT]{flag_prefix} {content}")
        elif source_type == "slide":
            if revisited > 0:
                truncated = truncate_text(content, REVISIT_TEXT_LIMIT)
                parts.append(f"[SLIDE:REVISIT#{revisited}] {truncated}")
            else:
                parts.append(f"[SLIDE] {content}")
    return " | ".join(parts)


def format_timeline_slice(groups: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for group in groups:
        formatted = format_group_content(group)
        if len(formatted) > 0:
            lines.append(formatted)
    return "\n".join(lines)


def split_entries_into_batches(
    entries: list[dict[str, Any]],
    max_characters: int,
    final_offset: int
) -> list[SliceBatch]:
    batches: list[SliceBatch] = []
    current_entries: list[dict[str, Any]] = []
    current_length = 0

    for entry in entries:
        content_length = len(str(entry.get("content", "")))
        if current_length + content_length > max_characters and len(current_entries) > 0:
            batches.append(SliceBatch(
                entries=current_entries,
                end_offset=int(current_entries[-1].get("_line_end_offset", final_offset)),
                start_offset=int(current_entries[0].get("_line_start_offset", 0))
            ))
            current_entries = []
            current_length = 0
        current_entries.append(entry)
        current_length += content_length

    if len(current_entries) > 0:
        batches.append(SliceBatch(
            entries=current_entries,
            end_offset=final_offset,
            start_offset=int(current_entries[0].get("_line_start_offset", 0))
        ))
    return batches


def create_structuring_graph(llm_router: Any) -> Any:
    async def call_llm(prompt: str) -> str:
        response = await llm_router.ainvoke([HumanMessage(content=prompt)])
        return response_to_text(response)

    async def group_correlated_events_node(state: StructuringState) -> dict[str, Any]:
        slice_entries = state.get("timeline_slice", [])
        overlap_entries = state.get("overlap_entries", [])
        combined = list(overlap_entries) + list(slice_entries)

        by_correlation: dict[str, list[dict[str, Any]]] = {}
        ordered_keys: list[str] = []
        unassigned_index = 0

        for entry in combined:
            correlation_id = entry.get("correlation_id")
            if correlation_id is not None and len(str(correlation_id).strip()) > 0:
                key = str(correlation_id).strip()
            else:
                key = f"unassigned_{unassigned_index}"
                unassigned_index += 1
            if key not in by_correlation:
                by_correlation[key] = []
                ordered_keys.append(key)
            by_correlation[key].append(entry)

        groups: list[dict[str, Any]] = []
        for key in ordered_keys:
            groups.append({
                "correlation_id": key,
                "items": by_correlation[key]
            })
        return {"correlated_groups": groups}

    async def summarize_slice_node(state: StructuringState) -> dict[str, Any]:
        groups = state.get("correlated_groups", [])
        formatted_slice = format_timeline_slice(groups)
        previous_context = state.get("running_context", "")
        prompt = f"""
You are an expert academic note taker synthesizing live lecture streams.
Format the following lecture events into structured, textbook-quality lecture notes.

GUIDELINES:
- Synthesize spoken explanations with corresponding slide content.
- Prioritize technical depth, mathematical accuracy, and clarity.
- Structure using markdown headers, bullet points, and code/math blocks ($...$ for inline, $$...$$ for display).
- Do not repeat or invent facts not grounded in the source text.

PREVIOUS CONTEXT:
{previous_context}

CURRENT LECTURE EVENTS:
{formatted_slice}
"""
        summary_text = await call_llm(prompt)
        return {"summary": summary_text}

    async def extract_definitions_node(state: StructuringState) -> dict[str, Any]:
        summary_text = state.get("summary", "")
        prompt = f"""
Extract all technical terms and concepts defined or introduced in this lecture section.
Respond strictly in valid JSON format as a list of objects with 'term' and 'definition'.

LECTURE SUMMARY:
{summary_text}

[
  {{"term": "Term Name", "definition": "Clear concise academic definition."}}
]
"""
        raw_output = await call_llm(prompt)
        parsed = extract_json_payload(raw_output)
        definitions: list[dict[str, str]] = []
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "term" in item and "definition" in item:
                    # The term used to be lowercased, which wrecked acronyms
                    # and proper nouns such as "TCP" or "Binary Search Tree".
                    # The term is matched case insensitively where it is
                    # looked up, so the original spelling is kept here.
                    definitions.append({
                        "term": str(item["term"]).strip(),
                        "definition": str(item["definition"]).strip()
                    })
        return {"definitions": definitions}

    async def extract_action_items_node(state: StructuringState) -> dict[str, Any]:
        summary_text = state.get("summary", "")
        prompt = f"""
Identify all action items, exam tips, homework reminders, or reading assignments from this section.
Respond strictly in valid JSON format as a list of strings. If none exist, return an empty array [].

LECTURE SUMMARY:
{summary_text}

["Action item 1", "Action item 2"]
"""
        raw_output = await call_llm(prompt)
        parsed = extract_json_payload(raw_output)
        action_items: list[str] = []
        if isinstance(parsed, list):
            for item in parsed:
                action_items.append(str(item))
        return {"action_items": action_items}

    async def extract_key_points_node(state: StructuringState) -> dict[str, Any]:
        summary_text = state.get("summary", "")
        definitions = state.get("definitions", [])
        concept_tags = [d["term"] for d in definitions if "term" in d]

        prompt = f"""
Extract 3-5 core key points from this lecture section.
Respond strictly in valid JSON format as a list of strings:
["Key point 1", "Key point 2"]

LECTURE SUMMARY:
{summary_text}
"""
        raw_output = await call_llm(prompt)
        parsed = extract_json_payload(raw_output)
        key_points: list[str] = []
        if isinstance(parsed, list):
            for item in parsed:
                key_points.append(str(item).strip())
        return {"key_points": key_points, "concept_tags": concept_tags}

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
    graph_builder.add_node("extract_key_points", extract_key_points_node)
    graph_builder.add_node("update_running_context", update_running_context_node)

    graph_builder.add_edge(START, "group_correlated_events")
    graph_builder.add_edge("group_correlated_events", "summarize_slice")
    graph_builder.add_edge("summarize_slice", "extract_definitions")
    graph_builder.add_edge("extract_definitions", "extract_action_items")
    graph_builder.add_edge("extract_action_items", "extract_key_points")
    graph_builder.add_edge("extract_key_points", "update_running_context")
    graph_builder.add_edge("update_running_context", END)
    return graph_builder.compile()


class StructuringAgent:
    def __init__(
        self,
        settings: Settings,
        session_id: str,
        llm_router: Any,
        redis_bus: Any = None,
        storage: Any = None,
        drift_filter: Optional[ContextDriftFilter] = None
    ) -> None:
        self.settings: Settings = settings
        self.session_id: str = session_id
        self.llm_router: Any = llm_router
        self.redis_bus: Any = redis_bus
        self.storage: Any = storage
        self.is_running: bool = False

        if drift_filter is not None:
            self.drift_filter: Optional[ContextDriftFilter] = drift_filter
        elif settings.enable_drift_filter:
            resolved_ollama_url = settings.ollama_url or os.environ.get("LECTURELENS_OLLAMA_URL", "http://127.0.0.1:11434")
            self.drift_filter = ContextDriftFilter(
                ollama_url=resolved_ollama_url,
                embed_model=settings.drift_embed_model or settings.embedding_model,
                drift_threshold=settings.drift_similarity_threshold,
                sustain_consecutive=settings.drift_consecutive_shift,
                min_words=settings.drift_min_content_words,
                session_topic=settings.course_name
            )
        else:
            self.drift_filter = None

        session_dir = settings.create_session_directories(session_id)
        self.session_dir: Path = session_dir
        self.timeline_path: Path = session_dir / "timeline.jsonl"
        self.notes_path: Path = session_dir / "notes.md"
        self.context_path: Path = session_dir / "context.json"
        self.notes_marker_path: Path = session_dir / "notes_marker.json"
        self.batch_attempts_path: Path = session_dir / "batch_attempts.json"
        self.quarantine_path: Path = session_dir / "timeline_quarantine.jsonl"
        self.control: AgentControl = AgentControl(session_dir, "structuring_agent", FINALIZE_FLAG_NAME)

        self.graph: Any = create_structuring_graph(llm_router)
        self.consecutive_failures: int = 0
        self.max_batch_retries: int = MAX_BATCH_RETRIES
        self.max_consecutive_failures: int = MAX_CONSECUTIVE_PASS_FAILURES
        # The notes are timestamped with the elapsed lecture time, not with
        # the wall clock time of the moment.
        self.session_clock: SessionClock = SessionClock.load_or_create(self.session_dir / "clock.json")
        initial_state_data = self.load_state()
        self._batch_attempts: dict[int, int] = self.load_batch_attempts()
        self.current_overlap: list[dict[str, Any]] = list(initial_state_data.get("overlap_entries", []))
        self.sequence_number: int = int(initial_state_data.get("sequence_number", 0))
        # How often the current batch was already split, so a provider with
        # a very small limit cannot make it split forever.
        self._split_depth: int = 0

    def load_notes_marker(self) -> int:
        if not self.notes_marker_path.exists():
            return -1
        try:
            payload = json.loads(self.notes_marker_path.read_text(encoding="utf-8"))
            return int(payload.get("written_end_offset", -1))
        except (ValueError, OSError):
            return -1

    def save_notes_marker(self, written_end_offset: int) -> None:
        payload = {"written_end_offset": written_end_offset}
        atomic_write_text(self.notes_marker_path, json.dumps(payload))

    def load_state(self) -> dict[str, Any]:
        if not self.context_path.exists():
            return {
                "running_context": "",
                "timeline_byte_offset": 0,
                "overlap_entries": [],
                "sequence_number": 0
            }
        try:
            with open(self.context_path, "r", encoding="utf-8") as file_handle:
                data = json.load(file_handle)
            if self.drift_filter is not None and "drift_filter_state" in data:
                self.drift_filter.load_dict(data["drift_filter_state"])
            return {
                "running_context": str(data.get("running_context", "")),
                "timeline_byte_offset": int(data.get("timeline_byte_offset", 0)),
                "overlap_entries": list(data.get("overlap_entries", [])),
                "sequence_number": int(data.get("sequence_number", 0))
            }
        except (ValueError, OSError) as error:
            logger.warning("failed to load structuring state: %s", error)
            return {
                "running_context": "",
                "timeline_byte_offset": 0,
                "overlap_entries": [],
                "sequence_number": 0
            }

    def save_state(self, running_context: str, timeline_byte_offset: int, overlap_entries: list[dict[str, Any]]) -> None:
        max_context = self.settings.context_max_characters
        trimmed_context = running_context
        if max_context > 0 and len(trimmed_context) > max_context:
            trimmed_context = trimmed_context[-max_context:]
        payload = {
            "running_context": trimmed_context,
            "timeline_byte_offset": timeline_byte_offset,
            "overlap_entries": overlap_entries,
            "sequence_number": self.sequence_number
        }
        if self.drift_filter is not None:
            payload["drift_filter_state"] = self.drift_filter.to_dict()
        atomic_write_text(self.context_path, json.dumps(payload, indent=2))

    def persist_batch_attempts(self) -> None:
        """Persist the attempt counter without advancing the byte offset.

        The counter lives in its own file: the reprocess command restores
        context.json from a backup when a run fails, which would wipe the
        counter and make the retry cap restart from zero every time.
        """
        atomic_write_text(
            self.batch_attempts_path,
            json.dumps({str(key): value for key, value in self._batch_attempts.items()})
        )

    def load_batch_attempts(self) -> dict[int, int]:
        if not self.batch_attempts_path.exists():
            return {}
        try:
            raw = json.loads(self.batch_attempts_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {}
        if not isinstance(raw, dict):
            return {}
        attempts: dict[int, int] = {}
        for key, value in raw.items():
            try:
                attempts[int(key)] = int(value)
            except (TypeError, ValueError):
                continue
        return attempts

    def read_pending_batches(self) -> list[SliceBatch]:
        if not self.timeline_path.exists():
            return []
        state = self.load_state()
        offset = state["timeline_byte_offset"]

        records: list[dict[str, Any]] = []
        final_offset = offset

        with open(self.timeline_path, "r", encoding="utf-8") as file_handle:
            file_handle.seek(offset)
            while True:
                line_start = file_handle.tell()
                line = file_handle.readline()
                if not line:
                    break
                if not line.endswith("\n"):
                    file_handle.seek(line_start)
                    break
                line_end = file_handle.tell()
                stripped = line.strip()
                if len(stripped) == 0:
                    continue
                try:
                    record = json.loads(stripped)
                    record["_line_end_offset"] = line_end
                    record["_line_start_offset"] = line_start
                    records.append(record)
                    final_offset = line_end
                except ValueError:
                    logger.warning("skipping corrupt timeline line at byte %d", line_start)
                    final_offset = line_end
                    continue

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

    def section_start_timestamp(self, entries: list[dict[str, Any]]) -> float:
        """When the section of the lecture starts.

        The speech of the lecturer is what the time mark points at. A
        slide that was published late can be much older than the rest of
        the section, because the text of a slide is read by a model that
        takes a while, and then the slide sits in front of a section that
        is about something else entirely. That put a mark of 00:00:22 on
        a section that was about the fifth minute of the lecture.
        """
        if len(entries) == 0:
            return 0.0
        for entry in entries:
            if str(entry.get("source_type", "transcript")) == "transcript":
                return float(entry.get("timestamp_epoch", 0.0))
        return float(entries[0].get("timestamp_epoch", 0.0))

    def build_notes_section(
        self,
        batch: SliceBatch,
        result_state: dict[str, Any]
    ) -> str:
        summary = str(result_state.get("summary", "")).strip()
        if len(summary) == 0:
            return ""
        cleaned_summary = format_clean_markdown(summary).strip()
        first_timestamp = self.section_start_timestamp(batch.entries)
        heading_time = self.session_clock.format_epoch_as_elapsed(first_timestamp)
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

        action_items = result_state.get("action_items", []) or []
        if len(action_items) > 0:
            section_lines.append("**Action Items**")
            for action_item in action_items:
                section_lines.append(f"- {action_item}")
            section_lines.append("")
        return "\n".join(section_lines) + "\n"

    def build_raw_section(self, batch: SliceBatch) -> str:
        """Build a section from the raw timeline content.

        Used when the language model keeps returning an empty summary. The
        lecture material is kept verbatim instead of being dropped.
        """
        first_timestamp = self.section_start_timestamp(batch.entries)
        heading_time = self.session_clock.format_epoch_as_elapsed(first_timestamp)
        divider = "\n---\n\n" if (self.notes_path.exists() and self.notes_path.stat().st_size > 0) else ""
        section_lines: list[str] = [f"{divider}### {heading_time}", ""]
        for entry in batch.entries:
            content = str(entry.get("content", "")).strip()
            if len(content) == 0:
                continue
            section_lines.append(content)
            section_lines.append("")
        return "\n".join(section_lines) + "\n"

    @staticmethod
    def _quarantine_key(entry: dict[str, Any]) -> str:
        """A stable key for a quarantined entry.

        The event id is used when there is one. The timeline records do not
        always carry one, so the content is used as the fallback.
        """
        event_id = str(entry.get("event_id") or "")
        if len(event_id) > 0:
            return event_id
        raw = (
            f"{entry.get('source_type', '')}|{entry.get('timestamp_epoch', '')}|"
            f"{entry.get('content', '')}"
        )
        return hashlib.sha1(raw.encode("utf-8", errors="replace"), usedforsecurity=False).hexdigest()

    async def persist_quarantined_entries(self) -> None:
        """Write the entries that the ad or drift filter excluded to disk.

        They are kept out of the notes, but they are not deleted either, so a
        false positive on the heuristics does not lose the content. The write
        happens once the batch is finished, so a batch that is retried does
        not append the same entries again, and the entries are keyed so a
        repeated write is skipped.
        """
        if self.drift_filter is None:
            return
        quarantined = list(self.drift_filter.quarantined_entries)
        if len(quarantined) == 0:
            return
        self.drift_filter.quarantined_entries.clear()
        known_keys = self._read_quarantine_keys()
        lines = []
        for entry in quarantined:
            key = self._quarantine_key(entry)
            if key in known_keys:
                continue
            try:
                lines.append(json.dumps(entry, ensure_ascii=False, default=str))
            except (TypeError, ValueError):
                continue
            known_keys.add(key)
        if len(lines) == 0:
            return
        try:
            await run_in_thread_pool(
                append_text_durably,
                self.quarantine_path,
                "\n".join(lines) + "\n"
            )
            self.control.increment_stat("quarantined_entries", len(lines))
        except OSError as error:
            logger.warning("could not write the quarantined entries: %s", error)

    def _read_quarantine_keys(self) -> set[str]:
        """The keys that are already in the quarantine file."""
        if not self.quarantine_path.exists():
            return set()
        keys: set[str] = set()
        try:
            with open(self.quarantine_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    try:
                        payload = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(payload, dict):
                        keys.add(self._quarantine_key(payload))
        except OSError:
            return keys
        return keys

    async def process_split_batch(
        self,
        batch: SliceBatch,
        entries: list[dict[str, Any]],
        running_context: str
    ) -> Optional[str]:
        """Do a batch in two halves, because the message was too long.

        Returns the running context, or None when the batch cannot be
        split any further.
        """
        if len(entries) < 2:
            return None
        if self._split_depth >= MAX_BATCH_SPLITS:
            logger.error(
                "the batch was split %d times and the provider still refuses the "
                "message, falling back to the raw content",
                self._split_depth
            )
            return None
        self._split_depth += 1
        try:
            split_index = max(1, len(entries) // 2)
            first_entries = entries[:split_index]
            second_entries = entries[split_index:]

            # The halves share the start, so the retry counter of the batch
            # stays the same. The end of the first half is the end of its
            # last entry, so the marker of the notes moves forward in order
            # and the second half is not skipped as already written.
            first_end = int(first_entries[-1].get("_line_end_offset", batch.end_offset))
            first_half = SliceBatch(
                entries=first_entries,
                end_offset=first_end,
                start_offset=batch.start_offset
            )
            second_half = SliceBatch(
                entries=second_entries,
                end_offset=batch.end_offset,
                start_offset=batch.start_offset
            )

            logger.warning(
                "the provider cannot take a message of %d entries, splitting the batch",
                len(entries)
            )
            context = await self.process_single_batch(first_half, running_context)
            return await self.process_single_batch(second_half, context)
        finally:
            self._split_depth -= 1

    async def process_single_batch(self, batch: SliceBatch, running_context: str) -> str:
        entries = batch.entries
        if self.drift_filter is not None:
            entries = await run_in_thread_pool(self.drift_filter.filter_batch_entries, batch.entries)

        if len(entries) == 0:
            # Everything was filtered out, so the batch is finished and the
            # quarantine can be written.
            await self.persist_quarantined_entries()
            logger.info("all entries in batch ending at offset %d filtered as drift/ad; skipping synthesis", batch.end_offset)
            self.save_notes_marker(batch.end_offset)
            self.save_state(running_context, batch.end_offset, self.current_overlap)
            return running_context

        active_batch = SliceBatch(
            entries=entries,
            end_offset=batch.end_offset,
            start_offset=batch.start_offset
        )
        initial_state: StructuringState = {
            "timeline_slice": entries,
            "running_context": running_context,
            "correlated_groups": [],
            "summary": "",
            "definitions": [],
            "action_items": [],
            "key_points": [],
            "concept_tags": [],
            "importance_flags": [],
            "overlap_entries": self.current_overlap
        }

        synthesis_error: Optional[BaseException] = None
        result_state: dict[str, Any] = dict(initial_state)
        try:
            result_state = await self.graph.ainvoke(initial_state)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            synthesis_error = error

        if synthesis_error is not None and is_message_too_long_error(synthesis_error):
            # The provider cannot take a message this long. The batch is
            # split in half and each half is done on its own, which gives
            # two sections with a time mark each instead of no notes at
            # all. A whole lecture of eleven minutes is one batch when
            # nothing was processed while it was recorded, and the
            # provider refused it.
            split_result = await self.process_split_batch(
                batch, entries, running_context
            )
            if split_result is not None:
                return split_result

        section_text = "" if synthesis_error is not None else self.build_notes_section(active_batch, result_state)

        if len(section_text) == 0:
            # The model produced nothing usable for this slice. Publishing an
            # empty note would burn a sequence number and writing nothing
            # would drop the batch, so the batch is retried instead. The
            # attempt counter bounds the retries: once it is exhausted the raw
            # timeline content is used so the lecture material is never lost.
            # The counter is keyed by where the batch starts, so it keeps
            # counting across retries even when more events arrived and the
            # end of the batch moved. Keyed by the end offset it always
            # looked like the first attempt, and the raw content fallback
            # never happened.
            attempt_key = batch.start_offset
            attempts = self._batch_attempts.get(attempt_key, 0) + 1
            self._batch_attempts[attempt_key] = attempts
            # The attempt counter is persisted, so the cap also applies across
            # restarts and across separate reprocess invocations. The byte
            # offset is deliberately left untouched.
            self.persist_batch_attempts()
            reason = synthesis_error if synthesis_error is not None else "empty summary"
            if attempts <= self.max_batch_retries:
                logger.error(
                    "structuring batch ending at offset %d produced no notes (%s); "
                    "attempt %d/%d, will retry",
                    batch.end_offset, reason, attempts, self.max_batch_retries
                )
                raise StructuringBatchError(
                    f"batch ending at offset {batch.end_offset} produced no notes: {reason}"
                ) from synthesis_error
            logger.error(
                "structuring batch ending at offset %d produced no notes after %d attempts; "
                "falling back to the raw timeline content",
                batch.end_offset, attempts
            )
            # The raw content is a fallback, not a synthesis. It is worth
            # pointing out, because the notes for that part of the lecture
            # are unedited transcript until the batch is reprocessed.
            self.control.increment_stat("raw_fallback_batches")
            self.control.set_stat("raw_fallback_offset", batch.end_offset)
            section_text = self.build_raw_section(active_batch)
            result_state = dict(result_state)
            result_state["summary"] = section_text
            result_state["key_points"] = []
            result_state["concept_tags"] = []
            result_state["definitions"] = []
            result_state["action_items"] = []

        definitions = result_state.get("definitions", []) or []
        key_points = result_state.get("key_points", []) or []
        concept_tags = result_state.get("concept_tags", []) or [d["term"] for d in definitions if "term" in d]

        first_timestamp = self.section_start_timestamp(entries)
        last_timestamp = float(entries[-1].get("timestamp_epoch", first_timestamp))
        start_time_str = self.session_clock.format_epoch_as_elapsed(first_timestamp)
        end_time_str = self.session_clock.format_epoch_as_elapsed(last_timestamp)

        if self.notes_path.exists() and self.load_notes_marker() >= batch.end_offset:
            logger.info(
                "batch ending at byte %d already written to notes, skipping append", batch.end_offset
            )
        else:
            with open(self.notes_path, "a", encoding="utf-8", newline="\n") as file_handle:
                file_handle.write(section_text)
            self.save_notes_marker(batch.end_offset)

        self.sequence_number += 1
        self._batch_attempts.pop(batch.start_offset, None)
        self._batch_attempts.pop(batch.end_offset, None)
        # The batch is finished, so the quarantined entries are written now.
        # Writing them before the synthesis would append them again on every
        # retry of a failed batch.
        await self.persist_quarantined_entries()

        slide_references: list[int] = []
        for entry in entries:
            if entry.get("source_type") == "slide":
                metadata = entry.get("metadata", {}) or {}
                slide_num = metadata.get("slide_number")
                if slide_num is not None:
                    try:
                        slide_references.append(int(slide_num))
                    except ValueError:
                        pass

        note_segment = GeneralNoteSegment(
            session_id=self.session_id,
            lecture_id=self.settings.course_name,
            sequence_number=self.sequence_number,
            timestamp_epoch=first_timestamp,
            start_time=start_time_str,
            end_time=end_time_str,
            heading=f"Section {self.sequence_number} ({start_time_str})",
            content=section_text,
            key_points=key_points,
            concept_tags=concept_tags,
            slide_references=slide_references,
            definitions=result_state.get("definitions", [])
        )

        if self.redis_bus is not None:
            stream_name = f"lecturelens:{self.session_id}:stream:notes"
            if hasattr(self.redis_bus, "publish_resilient"):
                await self.redis_bus.publish_resilient(stream_name, note_segment)
            else:
                try:
                    await self.redis_bus.publish(stream_name, note_segment)
                except Exception as error:
                    logger.warning("failed to publish note segment to redis: %s", error)

        next_overlap = extract_overlap_entries(entries, OVERLAP_WINDOW_SECONDS)
        self.current_overlap = next_overlap

        new_context = str(result_state.get("running_context", running_context))
        max_context = self.settings.context_max_characters
        if max_context > 0 and len(new_context) > max_context:
            new_context = new_context[-max_context:]
        self.save_state(new_context, batch.end_offset, next_overlap)
        return new_context

    async def process_all_pending(self) -> int:
        processed_batches = 0
        max_batches = self.settings.structuring_max_batches_per_run
        while processed_batches < max_batches:
            batches = self.read_pending_batches()
            if len(batches) == 0:
                break
            state = self.load_state()
            running_context = state["running_context"]
            for batch in batches:
                if processed_batches >= max_batches:
                    break
                running_context = await self.process_single_batch(batch, running_context)
                processed_batches += 1
                self.control.set_stat("batches_processed", processed_batches)
        return processed_batches

    async def process_batch(self) -> int:
        return await self.process_all_pending()

    def reset_outputs(self) -> None:
        for path in (
            self.notes_path,
            self.context_path,
            self.notes_marker_path,
            self.batch_attempts_path,
            # The quarantine belongs to the run. Leaving it behind made a
            # reprocess append the same entries to the old file again.
            self.quarantine_path,
        ):
            if path.exists():
                path.unlink()
        self.current_overlap = []
        self.sequence_number = 0
        # A deliberate rebuild starts from scratch, so the retry counter is
        # cleared as well.
        self._batch_attempts = {}
        if self.drift_filter is not None:
            self.drift_filter.reset()

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
            if self.consecutive_failures >= self.max_consecutive_failures:
                # The counter used to be written but never read, so the same
                # failing call was retried forever. Surface the condition in
                # the heartbeat instead of retrying silently; the per-batch
                # retry cap still guarantees forward progress.
                logger.error(
                    "structuring pass failed %d times in a row; the failing batch is "
                    "retried up to %d times before falling back to the raw timeline content",
                    self.consecutive_failures, self.max_batch_retries
                )
                self.control.write_heartbeat("failed")

    async def run(self) -> None:
        self.is_running = True
        interval = float(self.settings.structuring_interval_seconds)
        last_pass_time = time.monotonic()
        heartbeat_interval = 2.0
        last_heartbeat_time = 0.0
        while self.is_running:
            now = time.monotonic()
            if (now - last_heartbeat_time) >= heartbeat_interval:
                if self.consecutive_failures >= self.max_consecutive_failures:
                    # Keep reporting the failure instead of flipping straight
                    # back to "live" on the next heartbeat.
                    self.control.write_heartbeat("failed")
                else:
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
