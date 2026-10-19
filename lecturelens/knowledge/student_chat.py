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
MISCONCEPTION_TAG_PATTERN = re.compile(r"\[MISCONCEPTION:\s*(\{.*?\})\s*\]", re.DOTALL)

STUDENT_CHAT_SYSTEM_PROMPT = """
You are an intelligent, empathetic academic tutor helping a student understand their lecture.
Answer the student's question based strictly on the provided lecture context and timeline excerpts.

GUIDELINES:
1. Always ground your explanations in the lecture moments.
2. Cite specific timestamps in square brackets, e.g. [01:23] or [01:02:05], wherever referencing what was taught.
3. Be aware of the student's known misconceptions and address them if relevant.
4. If the student's question or statement reveals a fundamental misunderstanding or false assumption, gently clarify it and output an annotation at the very end in this exact format:
   [MISCONCEPTION: {{"concept": "concept_name", "error": "what the student mistakenly thought", "correction": "correct explanation"}}]
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
            for event in timeline_events[-max_events:]:
                elapsed = float(event.get("elapsed_seconds", 0.0))
                mins = int(elapsed // 60)
                secs = int(elapsed % 60)
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

        citations = TIMESTAMP_PATTERN.findall(raw_answer)

        tag_match = MISCONCEPTION_TAG_PATTERN.search(raw_answer)
        if tag_match and self.student_graph is not None:
            try:
                data = json.loads(tag_match.group(1))
                concept = str(data.get("concept", "general"))
                error = str(data.get("error", ""))
                correction = str(data.get("correction", ""))
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
        elif self.student_graph is not None:
            self.student_graph.record_evidence(
                concept_name="general",
                lecture_id="live_session",
                timestamp_epoch=time.time(),
                evidence_type="student_chat_inquiry",
                excerpt=question[:200]
            )

        clean_answer = MISCONCEPTION_TAG_PATTERN.sub("", raw_answer).strip()
        return clean_answer, citations
