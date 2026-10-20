import json
import logging
import re
import time
from typing import Any, Optional

from lecturelens.config.defaults import DEFAULT_LOCAL_MODEL
from lecturelens.knowledge.student_graph import StudentKnowledgeGraph
from lecturelens.router.messages import HumanMessage, response_to_text

logger = logging.getLogger(__name__)

TIMESTAMP_PATTERN = re.compile(r"\[\d\d:\d\d(?::\d\d)?\]")
# The annotation holds a JSON object, which may itself contain braces (for
# example a correction that mentions a set literal). A non greedy \{.*?\}
# stopped at the first closing brace, so the tag was never matched: the
# misconception was not recorded and the raw annotation leaked into the
# answer shown to the student.
MISCONCEPTION_TAG_START = re.compile(r"\[MISCONCEPTION:\s*", re.IGNORECASE)


def extract_misconception_payload(text: str) -> tuple[Optional[str], str]:
    """Split a misconception annotation off the answer.

    Returns the JSON payload and the answer with the annotation removed.
    """
    match = MISCONCEPTION_TAG_START.search(text)
    if match is None:
        return None, text.strip()

    brace_index = text.find("{", match.end())
    if brace_index < 0:
        return None, text.strip()

    depth = 0
    in_string = False
    escaped = False
    closing_index = -1
    for index in range(brace_index, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                closing_index = index
                break
    if closing_index < 0:
        return None, text.strip()

    payload = text[brace_index:closing_index + 1]
    # The annotation ends with a closing bracket right after the payload. Only
    # whitespace may separate them, otherwise a later bracket (for example a
    # timestamp citation) would be swallowed.
    head = text[:match.start()]
    tail = text[closing_index + 1:]
    stripped_tail = tail.lstrip()
    if stripped_tail.startswith("]"):
        tail = tail[len(tail) - len(stripped_tail) + 1:]
    remainder = (head.rstrip() + " " + tail.lstrip()).strip() if head.strip() and tail.strip() else (head + tail).strip()
    return payload, remainder

STUDENT_CHAT_SYSTEM_PROMPT = """
You are an intelligent, empathetic academic tutor helping a student understand their lecture.
Answer the student's question based strictly on the provided lecture context and timeline excerpts.

GUIDELINES:
1. Always ground your explanations in the lecture moments.
2. Cite specific timestamps in square brackets, e.g. [01:23] or [01:02:05], wherever referencing what was taught.
3. Be aware of the student's known misconceptions and address them if relevant.
4. If the student's question or statement reveals a fundamental misunderstanding or false assumption, gently clarify it and output an annotation at the very end in this exact format:
   [MISCONCEPTION: {"concept": "concept_name", "error": "what the student mistakenly thought", "correction": "correct explanation"}]
5. Keep explanations direct, technically rigorous, and encouraging.
"""


class StudentChatAgent:
    def __init__(
        self,
        llm_router: Any = None,
        student_graph: Optional[StudentKnowledgeGraph] = None,
        ollama_url: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> None:
        self.llm_router = llm_router
        self.student_graph = student_graph
        self.ollama_url = ollama_url.rstrip("/") if ollama_url else None
        self.model_name = model_name

    def format_context_excerpts(
        self,
        timeline_events: list[dict[str, Any]],
        notes_segments: list[dict[str, Any]],
        max_events: int = 40
    ) -> str:
        lines: list[str] = []
        if notes_segments:
            lines.append("LECTURE NOTES SECTIONS:")
            for note in notes_segments[-5:]:
                start_str = note.get("start_time") or "00:00"
                lines.append(f"[{start_str}] {note.get('heading', '')}: {note.get('content', '')}")

        if timeline_events:
            lines.append("\nLECTURE TIMELINE MOMENTS:")
            first_epoch = float(timeline_events[0].get("timestamp_epoch", 0.0)) if timeline_events else 0.0
            for event in timeline_events[-max_events:]:
                raw_elapsed = event.get("elapsed_seconds")
                try:
                    num_elapsed = float(raw_elapsed) if raw_elapsed is not None else 0.0
                except (ValueError, TypeError):
                    num_elapsed = 0.0

                if num_elapsed <= 0.0 and isinstance(event.get("metadata"), dict):
                    meta_start = event["metadata"].get("start_seconds")
                    if meta_start is not None:
                        try:
                            meta_val = float(meta_start)
                            if meta_val > 0.0:
                                num_elapsed = meta_val
                        except (ValueError, TypeError):
                            pass

                if num_elapsed <= 0.0:
                    cur_epoch = float(event.get("timestamp_epoch", 0.0))
                    if cur_epoch >= first_epoch > 0.0:
                        diff = cur_epoch - first_epoch
                        if diff > 0.0:
                            num_elapsed = diff

                mins = int(num_elapsed // 60)
                secs = int(num_elapsed % 60)
                stamp = f"[{mins:02d}:{secs:02d}]"
                source = event.get("source_type", "")
                content = event.get("content", "")
                lines.append(f"{stamp} ({source}) {content}")

        return "\n".join(lines)

    async def ask(
        self,
        question: str,
        timeline_events: Optional[list[dict[str, Any]]] = None,
        notes_segments: Optional[list[dict[str, Any]]] = None
    ) -> tuple[str, list[str]]:
        events = timeline_events or []
        notes = notes_segments or []
        context_str = self.format_context_excerpts(events, notes)

        misconception_str = ""
        if self.student_graph is not None:
            known = self.student_graph.get_known_misconceptions()
            if known:
                misconception_items = [f"- {m['concept_name']}: {m['error_pattern']}" for m in known[:5]]
                misconception_str = "\nKNOWN STUDENT MISCONCEPTIONS:\n" + "\n".join(misconception_items)

        prompt = (
            f"{STUDENT_CHAT_SYSTEM_PROMPT}\n\n"
            f"{misconception_str}\n\n"
            f"LECTURE CONTEXT:\n{context_str}\n\n"
            f"STUDENT QUESTION:\n{question}\n\n"
            f"TUTOR RESPONSE:"
        )

        if self.llm_router is not None:
            response = await self.llm_router.ainvoke([HumanMessage(content=prompt)])
            raw_answer = response_to_text(response)
        elif self.ollama_url:
            import httpx
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={"model": self.model_name or DEFAULT_LOCAL_MODEL, "prompt": prompt, "stream": False}
                )
                resp.raise_for_status()
                raw_answer = str(resp.json().get("response", "")).strip()
        else:
            raw_answer = "No LLM router or Ollama endpoint configured for student chat."

        tag_payload, clean_answer = extract_misconception_payload(raw_answer)

        tag_payload, clean_answer = extract_misconception_payload(raw_answer)
        if tag_payload and self.student_graph is not None:
            try:
                data = json.loads(tag_payload)
                concept = str(data.get("concept", "")).strip()
                error = str(data.get("error", ""))
                correction = str(data.get("correction", ""))
                if concept:
                    self.student_graph.record_misconception(concept, error, correction)
                    self.student_graph.record_evidence(
                        concept_name=concept,
                        lecture_id="live_session",
                        timestamp_epoch=time.time(),
                        evidence_type="student_chat_misconception",
                        excerpt=f"Q: {question} | Error: {error}"
                    )
            except Exception as parse_error:
                logger.warning("could not parse misconception tag: %s", parse_error)
        elif tag_payload and self.student_graph is not None:
            logger.warning("misconception annotation without a concept name was ignored")

        citations = TIMESTAMP_PATTERN.findall(clean_answer)
        return clean_answer, citations
