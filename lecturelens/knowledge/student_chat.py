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
MISCONCEPTION_TAG_START = re.compile(r"\[MISCONCEPTION:\s*", re.IGNORECASE)


def extract_misconception_payload(text: str) -> tuple[Optional[str], str]:
    """Split a misconception annotation off the answer, supporting nested braces."""
    match = MISCONCEPTION_TAG_START.search(text)
    if not match:
        return None, text.strip()
    idx = text.find("{", match.end())
    if idx < 0:
        return None, text.strip()

    depth, in_str, escaped, closing = 0, False, False, -1
    for i in range(idx, len(text)):
        ch = text[i]
        if in_str:
            if escaped: escaped = False
            elif ch == "\\": escaped = True
            elif ch == '"': in_str = False
            continue
        if ch == '"': in_str = True
        elif ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                closing = i
                break

    if closing < 0:
        return None, text.strip()

    payload = text[idx:closing + 1]
    head = text[:match.start()].rstrip()
    tail = text[closing + 1:].lstrip()
    if tail.startswith("]"):
        tail = tail[1:].lstrip()
    clean = f"{head} {tail}".strip() if head and tail else f"{head}{tail}".strip()
    return payload, clean


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

    def format_context_excerpts(self, timeline_events: list[dict[str, Any]], notes_segments: list[dict[str, Any]], max_events: int = 40) -> str:
        lines: list[str] = []
        if notes_segments:
            lines.append("LECTURE NOTES SECTIONS:")
            for note in notes_segments[-5:]:
                lines.append(f"[{note.get('start_time') or '00:00'}] {note.get('heading', '')}: {note.get('content', '')}")
        if timeline_events:
            lines.append("\nLECTURE TIMELINE MOMENTS:")
            first = float(timeline_events[0].get("timestamp_epoch", 0.0))
            for ev in timeline_events[-max_events:]:
                sec = float(ev.get("elapsed_seconds") or 0.0)
                if sec <= 0.0 and isinstance(ev.get("metadata"), dict) and "start_seconds" in ev["metadata"]:
                    sec = float(ev["metadata"]["start_seconds"])
                if sec <= 0.0 and float(ev.get("timestamp_epoch", 0.0)) >= first > 0.0:
                    sec = float(ev.get("timestamp_epoch", 0.0)) - first
                lines.append(f"[{int(sec // 60):02d}:{int(sec % 60):02d}] ({ev.get('source_type', '')}) {ev.get('content', '')}")
        return "\n".join(lines)

    async def ask(self, question: str, timeline_events: Optional[list[dict[str, Any]]] = None, notes_segments: Optional[list[dict[str, Any]]] = None) -> tuple[str, list[str]]:
        context_str = self.format_context_excerpts(timeline_events or [], notes_segments or [])
        misc_str = ""
        if self.student_graph:
            known = self.student_graph.get_known_misconceptions()
            if known:
                misc_str = "\nKNOWN STUDENT MISCONCEPTIONS:\n" + "\n".join(f"- {m['concept_name']}: {m['error_pattern']}" for m in known[:5])

        prompt = f"{STUDENT_CHAT_SYSTEM_PROMPT}\n\n{misc_str}\n\nLECTURE CONTEXT:\n{context_str}\n\nSTUDENT QUESTION:\n{question}\n\nTUTOR RESPONSE:"

        if self.llm_router is not None:
            resp = await self.llm_router.ainvoke([HumanMessage(content=prompt)])
            raw_answer = response_to_text(resp)
        elif self.ollama_url:
            import httpx
            async with httpx.AsyncClient(timeout=60.0) as client:
                res = await client.post(f"{self.ollama_url}/api/generate", json={"model": self.model_name or DEFAULT_LOCAL_MODEL, "prompt": prompt, "stream": False})
                res.raise_for_status()
                raw_answer = str(res.json().get("response", "")).strip()
        else:
            raw_answer = "No LLM router or Ollama endpoint configured for student chat."

        tag_payload, clean_answer = extract_misconception_payload(raw_answer)
        if tag_payload and self.student_graph is not None:
            try:
                data = json.loads(tag_payload)
                concept = str(data.get("concept", "")).strip()
                if concept:
                    self.student_graph.record_misconception(concept, str(data.get("error", "")), str(data.get("correction", "")))
                    self.student_graph.record_evidence(concept, "live_session", time.time(), "student_chat_misconception", f"Q: {question} | Error: {data.get('error', '')}")
                else:
                    logger.warning("misconception annotation without concept name ignored")
            except Exception as e:
                logger.warning("could not parse misconception: %s", e)

        return clean_answer, TIMESTAMP_PATTERN.findall(clean_answer)
