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

from lecturelens.config.settings import Settings
from lecturelens.router.usage import stage as usage_stage
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
MAX_BATCH_SPLITS = 6


def is_message_too_long_error(error: BaseException) -> bool:
    message = str(error).lower()
    if "limits the length of a single message" in message or "shorten your last message" in message:
        return True
    return any(
        sig in message
        for sig in (
            "context length", "maximum context", "too many tokens", "request too large",
            "payload too large", "input is too long", "prompt is too long", "string too long"
        )
    )


class StructuringBatchError(RuntimeError):
    pass


def extract_overlap_entries(
    entries: list[dict[str, Any]],
    window_seconds: float = OVERLAP_WINDOW_SECONDS
) -> list[dict[str, Any]]:
    if not entries:
        return []
    max_ts = max((float(e.get("timestamp_epoch", 0.0)) for e in entries), default=0.0)
    cutoff = max_ts - window_seconds
    overlap = [e for e in entries if float(e.get("timestamp_epoch", 0.0)) >= cutoff]
    return overlap[-MAX_OVERLAP_ITEMS:] if len(overlap) > MAX_OVERLAP_ITEMS else overlap


@dataclass
class SliceBatch:
    entries: list[dict[str, Any]]
    end_offset: int
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
    return best_candidate


def truncate_text(text: str, limit: int) -> str:
    cleaned = text.strip()
    return cleaned if len(cleaned) <= limit else cleaned[:limit] + "..."


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
            fig_path = str(metadata.get("figure_file_path", "") or "")
            fig_token = ""
            if fig_path:
                fig_id = str(metadata.get("figure_id", "") or Path(fig_path).stem)
                fig_cap = str(metadata.get("figure_caption", "") or "").strip()
                fig_token = f" [FIGURE {fig_id}: {fig_cap}]" if fig_cap else f" [FIGURE {fig_id}]"
            if revisited > 0:
                parts.append(f"[SLIDE:REVISIT#{revisited}] {truncate_text(content, REVISIT_TEXT_LIMIT)}{fig_token}")
            else:
                parts.append(f"[SLIDE] {content}{fig_token}")
    return " | ".join(parts)


def format_timeline_slice(groups: list[dict[str, Any]]) -> str:
    return "\n".join(filter(None, (format_group_content(g) for g in groups)))


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
        if current_length + content_length > max_characters and current_entries:
            batches.append(SliceBatch(
                entries=current_entries,
                end_offset=int(current_entries[-1].get("_line_end_offset", final_offset)),
                start_offset=int(current_entries[0].get("_line_start_offset", 0))
            ))
            current_entries = []
            current_length = 0
        current_entries.append(entry)
        current_length += content_length

    if current_entries:
        batches.append(SliceBatch(
            entries=current_entries,
            end_offset=final_offset,
            start_offset=int(current_entries[0].get("_line_start_offset", 0))
        ))
    return batches


LATEX_ENFORCEMENT_PROMPT = """
You are a LaTeX formatter for lecture notes. Rewrite the markdown below so that
every mathematical expression is written in LaTeX.

RULES:
- Inline math goes between single dollar signs: $O(1)$, $H(x)$, $n \\log n$, $64.8\\%$.
- Display math goes between double dollar signs, each on its own line:
  $$
  \\text{Index} = H(\\text{Key}) \\mod M
  $$
- A bare expression such as O(1), O(log n), H(x), 2^n, n^2, 64.8% or a symbol
  used as a variable has to be written in LaTeX.
- A word that is not mathematics stays as it is. Do not put a word into
  dollar signs.
- Keep every heading, list, table, code block and every word of the text
  exactly as it is. Change only the mathematics.
- A line that marks where a picture goes, written as [FIGURE id] or
  [FIGURE id: caption], must stay exactly as it is, word for word.
- Do not add anything that is not in the text.
- Reply with the rewritten markdown only. No commentary, no code fence around
  the whole answer.

TEXT:
%%SUMMARY%%
"""

MATHEMATICS_MARKERS: tuple[str, ...] = (
    "O(", "$", "^", "%", "∫", "≤", "≥", "≠", "≈", "×", "÷", "π", "∈", "⊂", "∞", "λ", "α", "β", "θ", "Δ", "Σ"
)
MATHEMATICS_WORDS = re.compile(r"\b(log|mod|sqrt|ln)\b|\bsum\s+(of|over)\b", re.IGNORECASE)
FUNCTION_MARKER = re.compile(r"\b[A-Za-z]\([A-Za-z0-9_]")


def text_has_mathematics(text: str) -> bool:
    return bool(FUNCTION_MARKER.search(text) or MATHEMATICS_WORDS.search(text) or any(m in text for m in MATHEMATICS_MARKERS))


HEADING_PATTERN = re.compile(r"^#{1,6}\s", re.MULTILINE)
FIGURE_TOKEN_PATTERN = re.compile(r"\[FIGURE\s*:?\s*([a-zA-Z0-9_\-]+)", re.IGNORECASE)
FIGURE_CAPTION_STOP_WORDS = frozenset({
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "at", "by", "for",
    "with", "from", "as", "is", "are", "was", "were", "be", "into", "about",
    "it", "its", "this", "that", "these", "those", "how", "why", "what",
    "when", "image", "picture", "diagram", "slide", "lecture", "chapter",
    "week", "part", "introduction",
})


def caption_keywords(caption: str) -> list[str]:
    words = re.findall(r"\w+", caption.lower(), re.UNICODE)
    return [w for w in words if len(w) >= 3 and w not in FIGURE_CAPTION_STOP_WORDS]


def best_paragraph_for_caption(paragraphs: list[str], caption: str) -> Optional[int]:
    keywords = caption_keywords(caption)
    if not keywords or not paragraphs:
        return None
    unique_keywords = list(dict.fromkeys(keywords))
    long_words = [w for w in unique_keywords if len(w) >= 6]
    best_index: Optional[int] = None
    best_hits = 0
    best_long_hit = False
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            continue
        lowered = paragraph.lower()
        hits = sum(1 for w in unique_keywords if re.search(rf"\b{re.escape(w)}", lowered))
        if hits == 0:
            continue
        long_hit = any(re.search(rf"\b{re.escape(w)}", lowered) for w in long_words)
        if not (hits >= 2 or long_hit):
            continue
        if hits > best_hits or (hits == best_hits and long_hit and not best_long_hit):
            best_hits, best_long_hit, best_index = hits, long_hit, index
    return best_index if best_hits > 0 else None


def figure_token_ids(text: str) -> set[str]:
    return {m.group(1).lower() for m in FIGURE_TOKEN_PATTERN.finditer(text)}


def latex_result_is_usable(original: str, rewritten: str) -> bool:
    candidate = rewritten.strip()
    if len(candidate) < 20:
        return False
    original_length = len(original.strip())
    if original_length == 0 or len(candidate) < original_length * 0.6 or len(candidate) > original_length * 2.0:
        return False
    if len(HEADING_PATTERN.findall(candidate)) < len(HEADING_PATTERN.findall(original)):
        return False
    return not (figure_token_ids(original) - figure_token_ids(candidate))


STRUCTURED_DELIMITER = "===NOTES_END==="
STRUCTURED_KEYS = ("definitions", "action_items", "key_points")
RUNNING_CONTEXT_TARGET = 1200
FILLER_WORDS = re.compile(r"\b(?:um+|uh+|hm+|hmm+|err+)\b[,:;.!?]*\s*", re.IGNORECASE)
STAMMERED_WORD = re.compile(r"\b(\w+)((?:[,\s]+)\1\b){2,}", re.IGNORECASE)
HEADING_TEXT_PATTERN = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def tail_head_overlap(first: str, second: str, maximum: int = 96, minimum: int = 20) -> int:
    scan = min(maximum, len(first), len(second))
    f_low, s_low = first.lower(), second.lower()
    for size in range(scan, minimum - 1, -1):
        if f_low[-size:] == s_low[:size]:
            return size
    return 0


def correlate_events(
    slice_entries: list[dict[str, Any]],
    overlap_entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    combined = list(overlap_entries) + list(slice_entries)
    by_corr: dict[str, list[dict[str, Any]]] = {}
    ordered_keys: list[str] = []
    unassigned = 0

    for entry in combined:
        cid = entry.get("correlation_id")
        key = str(cid).strip() if cid and str(cid).strip() else f"unassigned_{unassigned}"
        if key.startswith("unassigned_"):
            unassigned += 1
        if key not in by_corr:
            by_corr[key] = []
            ordered_keys.append(key)
        by_corr[key].append(entry)

    return [{"correlation_id": k, "items": by_corr[k]} for k in ordered_keys]


def trim_prompt_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full_slide_texts: set[str] = set()
    for group in groups:
        for item in group.get("items", []):
            if str(item.get("source_type", "")) == "slide":
                if int((item.get("metadata", {}) or {}).get("revisited_slide_number", 0) or 0) == 0:
                    normal = normalize_for_match(str(item.get("content", "")))
                    if normal:
                        full_slide_texts.add(normal)

    seen_transcripts: set[str] = set()
    trimmed: list[dict[str, Any]] = []
    for group in groups:
        items: list[dict[str, Any]] = []
        for item in group.get("items", []):
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            source_type = str(item.get("source_type", ""))
            normal = normalize_for_match(content)
            if source_type == "transcript":
                cleaned = FILLER_WORDS.sub(" ", content)
                cleaned = STAMMERED_WORD.sub(r"\1", cleaned)
                cleaned = re.sub(r"\s+", " ", cleaned).strip()
                if not cleaned:
                    continue
                normal = normalize_for_match(cleaned)
                if normal in seen_transcripts:
                    continue
                last = items[-1] if items else None
                if last and str(last.get("source_type", "")) == "transcript":
                    shared = tail_head_overlap(str(last.get("content", "")), cleaned)
                    if shared:
                        merged = f"{str(last.get('content', ''))} {cleaned[shared:]}".strip()
                        last["content"] = merged
                        seen_transcripts.add(normalize_for_match(merged))
                        continue
                seen_transcripts.add(normal)
                items.append({**item, "content": cleaned})
            elif source_type == "slide":
                revisited = int((item.get("metadata", {}) or {}).get("revisited_slide_number", 0) or 0)
                if revisited > 0 and normal in full_slide_texts:
                    continue
                items.append({**item, "content": content})
            else:
                items.append({**item, "content": content})
        if items:
            trimmed.append({"correlation_id": group.get("correlation_id"), "items": items})
    return trimmed


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


def structured_from_payload(payload: Any) -> Optional[dict[str, Any]]:
    if not isinstance(payload, dict) or not any(k in payload for k in STRUCTURED_KEYS):
        return None
    defs: list[dict[str, str]] = [
        {"term": str(i["term"]).strip(), "definition": str(i["definition"]).strip()}
        for i in (payload.get("definitions") or [])
        if isinstance(i, dict) and "term" in i and "definition" in i
    ]
    return {
        "definitions": defs,
        "action_items": [str(i).strip() for i in (payload.get("action_items") or []) if str(i).strip()],
        "key_points": [str(i).strip() for i in (payload.get("key_points") or []) if str(i).strip()],
        "concept_tags": [i["term"] for i in defs],
    }


def extract_structured_dict(raw_output: str) -> Optional[dict[str, Any]]:
    cleaned = raw_output.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()
    return structured_from_payload(extract_json_payload(cleaned, expect_list=False))


def split_notes_and_structured(raw_output: str) -> tuple[str, Optional[dict[str, Any]]]:
    text = raw_output.strip()
    marker = re.search(rf"^\s*{re.escape(STRUCTURED_DELIMITER)}\s*$", text, re.MULTILINE)
    if marker:
        return text[:marker.start()].strip(), extract_structured_dict(text[marker.end():])

    decoder = json.JSONDecoder()
    for match in reversed(list(re.finditer(r"\{", text))):
        try:
            candidate, _ = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        structured = structured_from_payload(candidate)
        if structured is not None:
            notes_before = re.sub(rf"\s*{re.escape(STRUCTURED_DELIMITER)}\s*$", "", text[:match.start()])
            return notes_before.strip(), structured
    return text, None


def build_running_context(
    previous: str,
    summary: str,
    definitions: list[dict[str, str]],
    key_points: list[str],
    max_lines: int = 50
) -> str:
    topics: list[str] = []
    seen_topics: set[str] = set()
    for line in (previous or "").splitlines():
        if line.startswith("Topics so far:"):
            for topic in line[len("Topics so far:"):].split(";"):
                t = topic.strip()
                norm = normalize_for_match(t)
                if t and norm and norm not in seen_topics:
                    topics.append(t)
                    seen_topics.add(norm)
    for heading in HEADING_TEXT_PATTERN.findall(summary or ""):
        h = heading.strip()
        norm = normalize_for_match(h)
        if h and norm and norm not in seen_topics:
            topics.append(h)
            seen_topics.add(norm)
    topics = topics[-12:]

    lines: list[str] = []
    if topics:
        lines.append("Topics so far: " + "; ".join(topics))
    terms = [
        f"{str(d['term']).strip()} = {truncate_text(str(d.get('definition', '')), 70)}"
        if d.get("definition") else str(d["term"]).strip()
        for d in (definitions or [])[:6] if d.get("term")
    ]
    if terms:
        lines.append("Terms: " + "; ".join(terms))
    points = [truncate_text(str(p), 110) for p in (key_points or [])[:4] if str(p).strip()]
    if points:
        lines.append("Recent key points:\n" + "\n".join(f"- {p}" for p in points))

    if not lines:
        text = truncate_text(previous or "", RUNNING_CONTEXT_TARGET)
    else:
        text = "\n".join(lines)
    if max_lines > 0:
        split_lines = text.splitlines()
        if len(split_lines) > max_lines:
            text = "\n".join(split_lines[-max_lines:])
    if len(text) > RUNNING_CONTEXT_TARGET:
        text = text[:RUNNING_CONTEXT_TARGET - 3].rstrip() + "..."
    return text


def math_left_unformatted(text: str) -> bool:
    sliced = re.sub(r"```.*?```", " ", text, flags=re.S)
    sliced = re.sub(r"\$\$.*?\$\$", " ", sliced, flags=re.S)
    sliced = re.sub(r"\$[^$\n]*\$", " ", sliced)
    sliced = re.sub(r"\[FIGURE[^\]]*\]", " ", sliced, flags=re.IGNORECASE)
    return bool(FUNCTION_MARKER.search(sliced) or re.search(r"[A-Za-z0-9]\^\(?[0-9A-Za-z(]", sliced) or MATHEMATICS_WORDS.search(sliced))


SELF_TALK_STARTERS = re.compile(
    r"^\s*\*?\s*(?:wait\b|hm+\b|let'?s\b|let me\b|note to self\b|self-correction\b|double check\b|one more check\b|final check\b|finali[sz]\b|json check\b|the prompt says\b)",
    re.IGNORECASE,
)
ITALIC_LABEL_LINE = re.compile(r"^\s*\*[^*\n]{0,90}:?\*\s*$")
RESTATED_STRUCTURED_LIST = re.compile(r"^\s*\*?\s*`?(?:definitions|action_items|key_points)`?\s*[:=]", re.IGNORECASE)
STRUCTURED_LIST_DUMP = re.compile(r'\{"(?:definitions|action_items|key_points)"\s*:', re.IGNORECASE)
CHECKLIST_TICK = re.compile(r"^\s*\d+\.\s+.{0,140}-\s*Yes\b")


def is_self_talk_line(line: str) -> bool:
    return bool(
        SELF_TALK_STARTERS.match(line)
        or ITALIC_LABEL_LINE.match(line)
        or RESTATED_STRUCTURED_LIST.match(line)
        or CHECKLIST_TICK.match(line)
        or STRUCTURED_LIST_DUMP.search(line)
        or STRUCTURED_DELIMITER in line
    )


def scrub_reasoning_chatter(notes: str) -> str:
    kept: list[str] = []
    skipping_body = False
    for line in notes.splitlines():
        if skipping_body:
            stripped = line.strip()
            if stripped == "" or stripped.startswith("#") or stripped.startswith("[FIGURE"):
                skipping_body = False
                kept.append(line)
            continue
        if is_self_talk_line(line):
            skipping_body = True
            continue
        kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept))


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
        self.settings = settings
        self.session_id = session_id
        self.llm_router = llm_router
        self.redis_bus = redis_bus
        self.storage = storage
        self.is_running = False

        if drift_filter is not None:
            self.drift_filter = drift_filter
        elif settings.enable_drift_filter:
            ollama_url = settings.ollama_url or os.environ.get("LECTURELENS_OLLAMA_URL", "http://127.0.0.1:11434")
            self.drift_filter = ContextDriftFilter(
                ollama_url=ollama_url,
                embed_model=settings.drift_embed_model or settings.embedding_model,
                drift_threshold=settings.drift_similarity_threshold,
                sustain_consecutive=settings.drift_consecutive_shift,
                min_words=settings.drift_min_content_words,
                session_topic=settings.course_name
            )
        else:
            self.drift_filter = None

        session_dir = settings.create_session_directories(session_id)
        self.session_dir = session_dir
        self.timeline_path = session_dir / "timeline.jsonl"
        self.notes_path = session_dir / "notes.md"
        self.notes_html_path = session_dir / "notes.html"
        self.context_path = session_dir / "context.json"
        self.notes_marker_path = session_dir / "notes_marker.json"
        self.batch_attempts_path = session_dir / "batch_attempts.json"
        self.quarantine_path = session_dir / "timeline_quarantine.jsonl"
        self.control = AgentControl(session_dir, "structuring_agent", FINALIZE_FLAG_NAME)

        self.latex_pass_enabled = bool(getattr(settings, "structuring_latex_pass", True))
        self.synthesis = self.run_synthesis
        self.consecutive_failures = 0
        self.max_batch_retries = MAX_BATCH_RETRIES
        self.max_consecutive_failures = MAX_CONSECUTIVE_PASS_FAILURES
        self.session_clock = SessionClock.load_or_create(self.session_dir / "clock.json")
        initial_state = self.load_state()
        self._batch_attempts = self.load_batch_attempts()
        self.current_overlap = list(initial_state.get("overlap_entries", []))
        self.sequence_number = int(initial_state.get("sequence_number", 0))
        self._split_depth = 0

    def load_notes_marker(self) -> int:
        if not self.notes_marker_path.exists():
            return -1
        try:
            return int(json.loads(self.notes_marker_path.read_text(encoding="utf-8")).get("written_end_offset", -1))
        except (ValueError, OSError):
            return -1

    def save_notes_marker(self, written_end_offset: int) -> None:
        atomic_write_text(self.notes_marker_path, json.dumps({"written_end_offset": written_end_offset}))

    def load_state(self) -> dict[str, Any]:
        if not self.context_path.exists():
            return {"running_context": "", "timeline_byte_offset": 0, "overlap_entries": [], "sequence_number": 0}
        try:
            with open(self.context_path, "r", encoding="utf-8") as f:
                data = json.load(f)
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
            return {"running_context": "", "timeline_byte_offset": 0, "overlap_entries": [], "sequence_number": 0}

    def save_state(self, running_context: str, timeline_byte_offset: int, overlap_entries: list[dict[str, Any]]) -> None:
        state: dict[str, Any] = {
            "running_context": running_context,
            "timeline_byte_offset": timeline_byte_offset,
            "overlap_entries": overlap_entries,
            "sequence_number": self.sequence_number
        }
        if self.drift_filter is not None:
            state["drift_filter_state"] = self.drift_filter.to_dict()
        atomic_write_text(self.context_path, json.dumps(state, indent=2, ensure_ascii=False))

    def persist_batch_attempts(self) -> None:
        try:
            atomic_write_text(self.batch_attempts_path, json.dumps({str(k): v for k, v in self._batch_attempts.items()}))
        except OSError as error:
            logger.warning("could not persist batch retry attempts: %s", error)

    def load_batch_attempts(self) -> dict[int, int]:
        if not self.batch_attempts_path.exists():
            return {}
        try:
            with open(self.batch_attempts_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {int(k): int(v) for k, v in data.items()} if isinstance(data, dict) else {}
        except (ValueError, OSError) as error:
            logger.warning("could not load batch retry attempts: %s", error)
            return {}

    def read_pending_batches(self) -> list[SliceBatch]:
        if not self.timeline_path.exists():
            return []
        state = self.load_state()
        offset = state["timeline_byte_offset"]
        file_size = self.timeline_path.stat().st_size
        if offset > file_size:
            offset = file_size

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
                if not stripped:
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

        batches = split_entries_into_batches(records, self.settings.max_slice_characters, final_offset)
        if not batches:
            self.save_state(state["running_context"], final_offset, self.current_overlap)
        for batch in batches:
            batch.entries.sort(key=lambda entry: float(entry.get("timestamp_epoch", 0.0)))
        return batches

    def collect_flag_excerpts(self, entries: list[dict[str, Any]], flag_name: str) -> list[str]:
        excerpts: list[str] = []
        for entry in entries:
            if flag_name in (entry.get("importance_flags", []) or []) and entry.get("source_type") == "transcript":
                excerpts.append(truncate_text(str(entry.get("content", "")), FLAG_EXCERPT_LIMIT))
                if len(excerpts) >= MAX_FLAG_EXCERPTS:
                    break
        return excerpts

    def section_start_timestamp(self, entries: list[dict[str, Any]]) -> float:
        if not entries:
            return 0.0
        for entry in entries:
            if str(entry.get("source_type", "transcript")) == "transcript":
                return float(entry.get("timestamp_epoch", 0.0))
        return float(entries[0].get("timestamp_epoch", 0.0))

    def collect_figure_references(self, entries: list[dict[str, Any]]) -> list[str]:
        lines: list[str] = []
        seen: set[str] = set()
        for entry in entries:
            if str(entry.get("source_type", "")) == "slide":
                metadata = entry.get("metadata", {}) or {}
                fig_path = str(metadata.get("figure_file_path", "") or "")
                if fig_path and fig_path not in seen:
                    seen.add(fig_path)
                    caption = str(metadata.get("figure_caption", "") or "").strip()
                    rel = self.relative_figure_path(fig_path)
                    if caption:
                        lines.extend([f"![{caption}]({rel})", f"*{caption}*"])
                    else:
                        lines.append(f"![the picture on the screen]({rel})")
        return lines

    def relative_figure_path(self, figure_path: str) -> str:
        try:
            return Path(figure_path).resolve().relative_to(self.session_dir.resolve()).as_posix()
        except (ValueError, OSError):
            return Path(figure_path).name

    def build_notes_section(self, batch: SliceBatch, result_state: dict[str, Any]) -> str:
        summary = str(result_state.get("summary", "")).strip()
        if not summary:
            return ""
        cleaned = format_clean_markdown(summary).strip()
        first_ts = self.section_start_timestamp(batch.entries)
        heading_time = self.session_clock.format_epoch_as_elapsed(first_ts)
        divider = "\n---\n\n" if (self.notes_path.exists() and self.notes_path.stat().st_size > 0) else ""
        section_lines = [f"{divider}### {heading_time}", ""]

        exam = self.collect_flag_excerpts(batch.entries, "exam_relevant")
        if exam:
            section_lines.append("> **Important Exam Topic**")
            for e in exam:
                section_lines.append(f"> - {e}")
            section_lines.append("")

        confusion = self.collect_flag_excerpts(batch.entries, "confusion")
        for c in confusion:
            section_lines.append(f"> **Review**: {c}")
        if confusion:
            section_lines.append("")

        valid_figures: dict[str, dict[str, str]] = {}
        for entry in batch.entries:
            if str(entry.get("source_type", "")) == "slide":
                metadata = entry.get("metadata", {}) or {}
                fig_path = str(metadata.get("figure_file_path", "") or "")
                if fig_path:
                    fig_id = str(metadata.get("figure_id", "") or Path(fig_path).stem)
                    raw_cap = str(metadata.get("figure_caption", "") or "").strip()
                    caption = re.sub(r"[\[\]\r\n]", " ", raw_cap).strip()
                    info = {"path": fig_path, "relative": self.relative_figure_path(fig_path), "caption": caption}
                    valid_figures[fig_id] = info
                    valid_figures[Path(fig_path).stem] = info

        ref_figures: set[str] = set()

        def replace_token(match: re.Match) -> str:
            raw_id = match.group(1).strip()
            if raw_id in valid_figures:
                info = valid_figures[raw_id]
                if info["path"] in ref_figures:
                    return ""
                ref_figures.add(info["path"])
                cap, rel = info["caption"], info["relative"]
                return f"\n\n![{cap}]({rel})\n*{cap}*\n\n" if cap else f"\n\n![the picture on the screen]({rel})\n\n"
            return ""

        token_re = re.compile(r"\[FIGURE\s*:?\s*([a-zA-Z0-9_\-]+)(?::\s*[^\]]*)?\]", re.IGNORECASE)
        processed = re.sub(r"\n{3,}", "\n\n", token_re.sub(replace_token, cleaned)).strip()

        ordered_figs = [info for fid, info in valid_figures.items() if info["path"] not in ref_figures and fid == info["caption"]]
        # Deduplicate by path
        seen_paths: set[str] = set()
        unique_figs = []
        for info in valid_figures.values():
            if info["path"] not in ref_figures and info["path"] not in seen_paths:
                seen_paths.add(info["path"])
                unique_figs.append(info)

        def fig_md(info: dict[str, str]) -> list[str]:
            return [f"![{info['caption']}]({info['relative']})", f"*{info['caption']}*"] if info["caption"] else [f"![the picture on the screen]({info['relative']})"]

        paragraphs = processed.split("\n\n") if processed else []
        insertions: dict[int, list[list[str]]] = {}
        leftovers: list[dict[str, str]] = []
        for info in unique_figs:
            idx = best_paragraph_for_caption(paragraphs, info["caption"])
            if idx is None:
                leftovers.append(info)
            else:
                insertions.setdefault(idx, []).append(fig_md(info))

        if insertions:
            rebuilt = []
            for i, p in enumerate(paragraphs):
                rebuilt.append(p)
                if i in insertions:
                    for lines in insertions[i]:
                        rebuilt.extend(lines)
            processed = re.sub(r"\n{3,}", "\n\n", "\n\n".join(rebuilt)).strip()

        section_lines.append(processed)
        section_lines.append("")
        for info in leftovers:
            section_lines.extend(fig_md(info))
            section_lines.append("")

        action_items = result_state.get("action_items", []) or []
        if action_items:
            section_lines.append("**Action Items**")
            for a in action_items:
                section_lines.append(f"- {a}")
            section_lines.append("")
        return "\n".join(section_lines) + "\n"

    def build_raw_section(self, batch: SliceBatch) -> str:
        first_ts = self.section_start_timestamp(batch.entries)
        heading_time = self.session_clock.format_epoch_as_elapsed(first_ts)
        divider = "\n---\n\n" if (self.notes_path.exists() and self.notes_path.stat().st_size > 0) else ""
        lines = [f"{divider}### {heading_time}", ""]
        for entry in batch.entries:
            c = str(entry.get("content", "")).strip()
            if c:
                lines.extend([c, ""])
        return "\n".join(lines) + "\n"

    async def _call_llm(self, prompt: str, stage_name: str) -> str:
        with usage_stage(stage_name):
            response = await self.llm_router.ainvoke([HumanMessage(content=prompt)])
            return response_to_text(response)

    async def run_synthesis(
        self,
        slice_entries: list[dict[str, Any]],
        overlap_entries: list[dict[str, Any]],
        running_context: str
    ) -> dict[str, Any]:
        groups = correlate_events(slice_entries, overlap_entries)
        slice_text = format_timeline_slice(trim_prompt_groups(groups))
        if not slice_text.strip():
            return {
                "summary": "", "definitions": [], "action_items": [], "key_points": [],
                "concept_tags": [], "running_context": running_context, "empty_slice": True
            }

        prompt = (
            SYNTHESIS_PROMPT
            .replace("%%PREVIOUS_CONTEXT%%", running_context or "(the lecture just started)")
            .replace("%%SLICE_TEXT%%", slice_text)
        )
        notes_raw = await self._call_llm(prompt, "notes")
        notes_text, structured = split_notes_and_structured(notes_raw)
        notes_text = scrub_reasoning_chatter(notes_text)

        if structured is None and notes_text.strip():
            try:
                repair_raw = await self._call_llm(REPAIR_PROMPT.replace("%%NOTES%%", notes_text), "repair")
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.warning("structure repair failed: %s", error)
                repair_raw = ""
            structured = extract_structured_dict(repair_raw)
        if structured is None:
            structured = {"definitions": [], "action_items": [], "key_points": [], "concept_tags": []}

        notes_text = await self._rewrite_latex_if_needed(notes_text)
        max_lines = int(getattr(self.settings, "context_max_lines", 50))
        new_context = build_running_context(
            running_context, notes_text, structured["definitions"], structured["key_points"], max_lines=max_lines
        )
        return {
            "summary": notes_text,
            "definitions": structured["definitions"],
            "action_items": structured["action_items"],
            "key_points": structured["key_points"],
            "concept_tags": structured["concept_tags"],
            "running_context": new_context,
        }

    async def _rewrite_latex_if_needed(self, notes_text: str) -> str:
        if not self.latex_pass_enabled or not notes_text.strip():
            return notes_text
        if not text_has_mathematics(notes_text) or not math_left_unformatted(notes_text):
            return notes_text
        try:
            rewritten = await self._call_llm(LATEX_ENFORCEMENT_PROMPT.replace("%%SUMMARY%%", notes_text), "latex")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning("latex pass failed: %s", error)
            return notes_text
        cleaned = rewritten.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].rstrip()
        return cleaned if latex_result_is_usable(notes_text, cleaned) else notes_text

    @staticmethod
    def _quarantine_key(entry: dict[str, Any]) -> str:
        eid = str(entry.get("event_id") or "")
        if eid:
            return eid
        raw = f"{entry.get('source_type', '')}|{entry.get('timestamp_epoch', '')}|{entry.get('content', '')}"
        return hashlib.sha1(raw.encode("utf-8", errors="replace"), usedforsecurity=False).hexdigest()

    async def persist_quarantined_entries(self) -> None:
        if self.drift_filter is None or not self.drift_filter.quarantined_entries:
            return
        quarantined = list(self.drift_filter.quarantined_entries)
        self.drift_filter.quarantined_entries.clear()
        known = self._read_quarantine_keys()
        lines = []
        for entry in quarantined:
            key = self._quarantine_key(entry)
            if key not in known:
                try:
                    lines.append(json.dumps(entry, ensure_ascii=False, default=str))
                    known.add(key)
                except (TypeError, ValueError):
                    continue
        if lines:
            try:
                await run_in_thread_pool(append_text_durably, self.quarantine_path, "\n".join(lines) + "\n")
                self.control.increment_stat("quarantined_entries", len(lines))
            except OSError as error:
                logger.warning("could not write quarantined entries: %s", error)

    def _read_quarantine_keys(self) -> set[str]:
        if not self.quarantine_path.exists():
            return set()
        keys = set()
        try:
            with open(self.quarantine_path, "r", encoding="utf-8") as h:
                for line in h:
                    line = line.strip()
                    if line:
                        try:
                            payload = json.loads(line)
                            if isinstance(payload, dict):
                                keys.add(self._quarantine_key(payload))
                        except ValueError:
                            continue
        except OSError:
            return keys
        return keys

    async def process_split_batch(
        self,
        batch: SliceBatch,
        entries: list[dict[str, Any]],
        running_context: str
    ) -> Optional[str]:
        if len(entries) < 2 or self._split_depth >= MAX_BATCH_SPLITS:
            return None
        self._split_depth += 1
        try:
            split_idx = max(1, len(entries) // 2)
            first_entries, second_entries = entries[:split_idx], entries[split_idx:]
            first_end = int(first_entries[-1].get("_line_end_offset", batch.end_offset))
            first_half = SliceBatch(entries=first_entries, end_offset=first_end, start_offset=batch.start_offset)
            second_half = SliceBatch(entries=second_entries, end_offset=batch.end_offset, start_offset=batch.start_offset)
            context = await self.process_single_batch(first_half, running_context)
            return await self.process_single_batch(second_half, context)
        finally:
            self._split_depth -= 1

    def write_html_notes(self) -> None:
        if not self.notes_path.exists():
            return
        try:
            from lecturelens.core.notes_html import render_notes_html
            notes_text = self.notes_path.read_text(encoding="utf-8")
            self.notes_html_path.write_text(
                render_notes_html(notes_text, title=f"Lecture Notes - {self.session_id}", base_dir=self.session_dir),
                encoding="utf-8"
            )
        except Exception as error:
            logger.warning("the html notes could not be written: %s", error)

    async def process_single_batch(self, batch: SliceBatch, running_context: str) -> str:
        entries = batch.entries
        if self.drift_filter is not None:
            entries = await run_in_thread_pool(self.drift_filter.filter_batch_entries, batch.entries)

        if not entries:
            await self.persist_quarantined_entries()
            self.save_notes_marker(batch.end_offset)
            self.save_state(running_context, batch.end_offset, self.current_overlap)
            return running_context

        if self.notes_path.exists() and self.load_notes_marker() >= batch.end_offset:
            first_ts = self.section_start_timestamp(entries)
            last_ts = float(entries[-1].get("timestamp_epoch", first_ts))
            start_str = self.session_clock.format_epoch_as_elapsed(first_ts)
            end_str = self.session_clock.format_epoch_as_elapsed(last_ts)
            slide_refs = [
                int(e.get("metadata", {}).get("slide_number"))
                for e in entries
                if str(e.get("source_type", "")) == "slide" and (e.get("metadata", {}) or {}).get("slide_number") is not None
            ]
            saved_notes = self.notes_path.read_text(encoding="utf-8")
            segment = GeneralNoteSegment(
                session_id=self.session_id,
                lecture_id=self.settings.course_name,
                sequence_number=self.sequence_number,
                timestamp_epoch=first_ts,
                start_time=start_str,
                end_time=end_str,
                heading=f"Section {self.sequence_number} ({start_str})",
                content=saved_notes,
                key_points=[],
                concept_tags=[],
                slide_references=slide_refs,
                definitions=[]
            )
            if self.redis_bus is not None:
                stream_name = f"lecturelens:{self.session_id}:stream:notes"
                if hasattr(self.redis_bus, "publish_resilient"):
                    await self.redis_bus.publish_resilient(stream_name, segment)
                else:
                    try:
                        await self.redis_bus.publish(stream_name, segment)
                    except Exception:
                        pass
            self.sequence_number += 1
            self._batch_attempts.pop(batch.start_offset, None)
            self._batch_attempts.pop(batch.end_offset, None)
            next_overlap = extract_overlap_entries(entries, OVERLAP_WINDOW_SECONDS)
            self.current_overlap = next_overlap
            self.save_state(running_context, batch.end_offset, next_overlap)
            return running_context

        active_batch = SliceBatch(entries=entries, end_offset=batch.end_offset, start_offset=batch.start_offset)
        synthesis_error: Optional[BaseException] = None
        result_state: dict[str, Any] = {
            "summary": "", "definitions": [], "action_items": [], "key_points": [],
            "concept_tags": [], "running_context": running_context
        }
        try:
            result_state = await self.synthesis(entries, self.current_overlap, running_context)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            synthesis_error = error

        if synthesis_error is None and result_state.get("empty_slice"):
            self.control.increment_stat("empty_batches_skipped")
            self.control.set_stat("empty_batches_offset", batch.end_offset)
            await self.persist_quarantined_entries()
            self.save_notes_marker(batch.end_offset)
            next_overlap = extract_overlap_entries(entries, OVERLAP_WINDOW_SECONDS)
            self.current_overlap = next_overlap
            ctx = str(result_state.get("running_context", running_context))
            self.save_state(ctx, batch.end_offset, next_overlap)
            return ctx

        if synthesis_error is not None and is_message_too_long_error(synthesis_error):
            split_result = await self.process_split_batch(batch, entries, running_context)
            if split_result is not None:
                return split_result

        section_text = "" if synthesis_error is not None else self.build_notes_section(active_batch, result_state)

        if not section_text:
            attempt_key = batch.start_offset
            attempts = self._batch_attempts.get(attempt_key, 0) + 1
            self._batch_attempts[attempt_key] = attempts
            self.persist_batch_attempts()
            reason = synthesis_error if synthesis_error is not None else "empty summary"
            if attempts <= self.max_batch_retries:
                raise StructuringBatchError(f"batch ending at offset {batch.end_offset} produced no notes: {reason}") from synthesis_error
            self.control.increment_stat("raw_fallback_batches")
            self.control.set_stat("raw_fallback_offset", batch.end_offset)
            section_text = self.build_raw_section(active_batch)
            result_state = {**result_state, "summary": section_text, "key_points": [], "concept_tags": [], "definitions": [], "action_items": []}

        definitions = result_state.get("definitions", []) or []
        key_points = result_state.get("key_points", []) or []
        concept_tags = result_state.get("concept_tags", []) or [d["term"] for d in definitions if "term" in d]

        first_ts = self.section_start_timestamp(entries)
        last_ts = float(entries[-1].get("timestamp_epoch", first_ts))
        start_str = self.session_clock.format_epoch_as_elapsed(first_ts)
        end_str = self.session_clock.format_epoch_as_elapsed(last_ts)

        if not (self.notes_path.exists() and self.load_notes_marker() >= batch.end_offset):
            with open(self.notes_path, "a", encoding="utf-8", newline="\n") as f:
                f.write(section_text)
            self.save_notes_marker(batch.end_offset)
            self.write_html_notes()

        self.sequence_number += 1
        self._batch_attempts.pop(batch.start_offset, None)
        self._batch_attempts.pop(batch.end_offset, None)
        await self.persist_quarantined_entries()

        slide_refs: list[int] = []
        for e in entries:
            if e.get("source_type") == "slide":
                snum = (e.get("metadata", {}) or {}).get("slide_number")
                if snum is not None:
                    try:
                        slide_refs.append(int(snum))
                    except ValueError:
                        pass

        note_segment = GeneralNoteSegment(
            session_id=self.session_id,
            lecture_id=self.settings.course_name,
            sequence_number=self.sequence_number,
            timestamp_epoch=first_ts,
            start_time=start_str,
            end_time=end_str,
            heading=f"Section {self.sequence_number} ({start_str})",
            content=section_text,
            key_points=key_points,
            concept_tags=concept_tags,
            slide_references=slide_refs,
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
                    logger.warning("failed to publish note segment: %s", error)

        next_overlap = extract_overlap_entries(entries, OVERLAP_WINDOW_SECONDS)
        self.current_overlap = next_overlap

        new_context = str(result_state.get("running_context", running_context))
        max_context = self.settings.context_max_characters
        max_lines = getattr(self.settings, "context_max_lines", 0)
        if max_lines > 0:
            lines = new_context.splitlines()
            if len(lines) > max_lines:
                new_context = "\n".join(lines[-max_lines:])
        if max_context > 0 and len(new_context) > max_context:
            new_context = new_context[-max_context:]
        self.save_state(new_context, batch.end_offset, next_overlap)
        return new_context

    async def process_all_pending(self) -> int:
        processed = 0
        max_batches = self.settings.structuring_max_batches_per_run
        while processed < max_batches:
            batches = self.read_pending_batches()
            if not batches:
                break
            running_context = self.load_state()["running_context"]
            for b in batches:
                if processed >= max_batches:
                    break
                running_context = await self.process_single_batch(b, running_context)
                processed += 1
                self.control.set_stat("batches_processed", processed)
        return processed

    async def process_batch(self) -> int:
        return await self.process_all_pending()

    def reset_outputs(self) -> None:
        for path in (
            self.notes_path, self.notes_html_path, self.context_path,
            self.notes_marker_path, self.batch_attempts_path, self.quarantine_path
        ):
            if path.exists():
                path.unlink()
        self.current_overlap = []
        self.sequence_number = 0
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
                self.control.write_heartbeat("failed")

    async def run(self) -> None:
        self.is_running = True
        interval = float(self.settings.structuring_interval_seconds)
        last_pass = time.monotonic()
        heartbeat_interval = 2.0
        last_heartbeat = 0.0
        while self.is_running:
            now = time.monotonic()
            if (now - last_heartbeat) >= heartbeat_interval:
                if self.drift_filter is not None:
                    for name, count in self.drift_filter.guards().counts().items():
                        self.control.set_stat(f"fault_{name}", count)
                state = "failed" if self.consecutive_failures >= self.max_consecutive_failures else "live"
                self.control.write_heartbeat(state)
                last_heartbeat = now
            if self.control.stop_requested():
                logger.info("finalize requested, running final structuring pass")
                await self.run_guarded_pass()
                return
            if (now - last_pass) >= interval:
                await self.run_guarded_pass()
                last_pass = time.monotonic()
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
            context.settings.llm_timeout_seconds,
            context.settings.local_model
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
