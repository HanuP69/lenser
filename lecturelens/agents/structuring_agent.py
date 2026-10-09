import asyncio
import json
from pathlib import Path
from typing import Any, Optional
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from lecturelens.config.settings import Settings
from lecturelens.core.redis_bus import RedisBus
from lecturelens.exports.anki_exporter import create_anki_deck
from lecturelens.knowledge.concept_graph import ConceptGraph
from lecturelens.router.llm_router import build_router_for_task, load_router_config


class StructuringState(TypedDict):
    timeline_slice: list[dict[str, Any]]
    running_context: str
    correlated_groups: list[dict[str, Any]]
    summary: str
    definitions: list[dict[str, str]]
    action_items: list[str]
    flashcards: list[dict[str, str]]
    importance_flags: list[str]


def create_structuring_graph(llm_router: Any) -> Any:
    async def group_correlated_events_node(state: StructuringState) -> dict[str, Any]:
        slice_items = state.get("timeline_slice", [])
        groups_map: dict[str, list[dict[str, Any]]] = {}
        ungrouped_items: list[dict[str, Any]] = []

        for item in slice_items:
            correlation_id = item.get("correlation_id")
            if correlation_id is not None and len(correlation_id) > 0:
                if correlation_id not in groups_map:
                    groups_map[correlation_id] = []
                groups_map[correlation_id].append(item)
            else:
                ungrouped_items.append(item)

        aggregated_groups: list[dict[str, Any]] = []
        for corr_id, group_elements in groups_map.items():
            aggregated_groups.append({
                "correlation_id": corr_id,
                "items": group_elements
            })

        for single_item in ungrouped_items:
            aggregated_groups.append({
                "correlation_id": None,
                "items": [single_item]
            })

        all_flags: list[str] = []
        for item in slice_items:
            flags = item.get("importance_flags", [])
            for flag in flags:
                if flag not in all_flags:
                    all_flags.append(flag)

        return {
            "correlated_groups": aggregated_groups,
            "importance_flags": all_flags
        }

    async def summarize_slice_node(state: StructuringState) -> dict[str, Any]:
        running_context = state.get("running_context", "")
        groups = state.get("correlated_groups", [])

        content_lines: list[str] = []
        for group in groups:
            items = group.get("items", [])
            group_texts: list[str] = []
            for item in items:
                source = item.get("source_type", "unknown").upper()
                text = item.get("content", "").strip()
                if len(text) > 0:
                    group_texts.append(f"[{source}] {text}")
            if len(group_texts) > 0:
                combined_group_text = " | ".join(group_texts)
                content_lines.append(combined_group_text)

        new_timeline_text = "\n".join(content_lines)
        if len(new_timeline_text.strip()) == 0:
            return {"summary": ""}

        prompt = f"""
You are an expert academic note taker.
PREVIOUS LECTURE CONTEXT:
{running_context}

NEW LECTURE TIMELINE SLICE (transcript, slide OCR, board captures):
{new_timeline_text}

Generate clear, structured, cohesive markdown notes for this slice.
Incorporate definitions, key formulas, and explanations.
Maintain continuous narrative flow from the previous lecture context.
Do not invent facts not present in the content.
"""
        message = HumanMessage(content=prompt)
        response = await llm_router.ainvoke([message])
        summary_content = getattr(response, "content", str(response))
        return {"summary": str(summary_content).strip()}

    async def extract_definitions_node(state: StructuringState) -> dict[str, Any]:
        summary_text = state.get("summary", "")
        if len(summary_text) == 0:
            return {"definitions": []}

        prompt = f"""
From the following lecture note summary, extract all key technical terms and their definitions.
SUMMARY:
{summary_text}

Respond strictly in valid JSON format as a list of objects with 'term' and 'definition' keys:
[
  {{"term": "term name", "definition": "detailed definition"}}
]
"""
        message = HumanMessage(content=prompt)
        response = await llm_router.ainvoke([message])
        raw_output = getattr(response, "content", str(response)).strip()

        clean_json = raw_output
        if "```json" in clean_json:
            clean_json = clean_json.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_json:
            clean_json = clean_json.split("```")[1].split("```")[0].strip()

        definitions_list: list[dict[str, str]] = []
        try:
            parsed = json.loads(clean_json)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "term" in item and "definition" in item:
                        definitions_list.append({
                            "term": str(item["term"]),
                            "definition": str(item["definition"])
                        })
        except Exception:
            definitions_list = []

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
        message = HumanMessage(content=prompt)
        response = await llm_router.ainvoke([message])
        raw_output = getattr(response, "content", str(response)).strip()

        clean_json = raw_output
        if "```json" in clean_json:
            clean_json = clean_json.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_json:
            clean_json = clean_json.split("```")[1].split("```")[0].strip()

        action_items: list[str] = []
        try:
            parsed = json.loads(clean_json)
            if isinstance(parsed, list):
                for item in parsed:
                    action_items.append(str(item))
        except Exception:
            action_items = []

        return {"action_items": action_items}

    async def generate_flashcards_node(state: StructuringState) -> dict[str, Any]:
        definitions = state.get("definitions", [])
        summary_text = state.get("summary", "")
        flashcards: list[dict[str, str]] = []

        for d in definitions:
            flashcards.append({
                "question": f"What is the definition of {d['term']}?",
                "answer": d["definition"]
            })

        if len(summary_text) > 0 and len(definitions) == 0:
            prompt = f"""
Generate 2-3 high-yield study flashcards based on this lecture summary.
SUMMARY:
{summary_text}

Respond strictly in valid JSON format as a list of objects with 'question' and 'answer':
[
  {{"question": "...", "answer": "..."}}
]
"""
            message = HumanMessage(content=prompt)
            response = await llm_router.ainvoke([message])
            raw_output = getattr(response, "content", str(response)).strip()

            clean_json = raw_output
            if "```json" in clean_json:
                clean_json = clean_json.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_json:
                clean_json = clean_json.split("```")[1].split("```")[0].strip()

            try:
                parsed = json.loads(clean_json)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "question" in item and "answer" in item:
                            flashcards.append({
                                "question": str(item["question"]),
                                "answer": str(item["answer"])
                            })
            except Exception:
                pass

        return {"flashcards": flashcards}

    async def update_running_context_node(state: StructuringState) -> dict[str, Any]:
        previous_context = state.get("running_context", "")
        summary_text = state.get("summary", "")
        if len(summary_text) == 0:
            return {"running_context": previous_context}

        prompt = f"""
Compress the following cumulative lecture summary so that it maintains all critical concepts, theories, and progression in under 400 words.
PREVIOUS CONTEXT:
{previous_context}

LATEST NOTES ADDITION:
{summary_text}

Output only the concise, cumulative context summary:
"""
        message = HumanMessage(content=prompt)
        response = await llm_router.ainvoke([message])
        updated_context = getattr(response, "content", str(response))
        return {"running_context": str(updated_context).strip()}

    workflow = StateGraph(StructuringState)
    workflow.add_node("group_correlated_events", group_correlated_events_node)
    workflow.add_node("summarize_slice", summarize_slice_node)
    workflow.add_node("extract_definitions", extract_definitions_node)
    workflow.add_node("extract_action_items", extract_action_items_node)
    workflow.add_node("generate_flashcards", generate_flashcards_node)
    workflow.add_node("update_running_context", update_running_context_node)

    workflow.add_edge(START, "group_correlated_events")
    workflow.add_edge("group_correlated_events", "summarize_slice")
    workflow.add_edge("summarize_slice", "extract_definitions")
    workflow.add_edge("extract_definitions", "extract_action_items")
    workflow.add_edge("extract_action_items", "generate_flashcards")
    workflow.add_edge("generate_flashcards", "update_running_context")
    workflow.add_edge("update_running_context", END)

    compiled_graph = workflow.compile()
    return compiled_graph


class StructuringAgent:
    def __init__(
        self,
        settings: Settings,
        session_id: str,
        llm_router: Any
    ) -> None:
        self.settings: Settings = settings
        self.session_id: str = session_id
        self.llm_router: Any = llm_router
        self.compiled_graph: Any = create_structuring_graph(llm_router)

        session_dir = self.settings.get_session_output_dir(self.session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        self.timeline_file_path: Path = session_dir / "timeline.jsonl"
        self.notes_file_path: Path = session_dir / "notes.md"
        self.context_file_path: Path = session_dir / "context.json"
        self.flashcards_file_path: Path = session_dir / "flashcards.apkg"
        self.concepts_db_path: Path = session_dir / "concepts.db"

        self.concept_graph: ConceptGraph = ConceptGraph(self.concepts_db_path)
        self.processed_line_offset: int = 0
        self.accumulated_flashcards: list[dict[str, str]] = []
        self.is_running: bool = False

    def load_running_context(self) -> str:
        if self.context_file_path.exists():
            try:
                with open(self.context_file_path, "r", encoding="utf-8") as file_handle:
                    data = json.load(file_handle)
                    return data.get("running_context", "")
            except Exception:
                return ""
        return ""

    def save_running_context(self, context_text: str) -> None:
        with open(self.context_file_path, "w", encoding="utf-8") as file_handle:
            json.dump({"running_context": context_text}, file_handle)

    def read_new_timeline_slice(self) -> list[dict[str, Any]]:
        if not self.timeline_file_path.exists():
            return []

        new_entries: list[dict[str, Any]] = []
        current_line_index = 0

        with open(self.timeline_file_path, "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                if current_line_index >= self.processed_line_offset:
                    line_clean = line.strip()
                    if len(line_clean) > 0:
                        try:
                            item = json.loads(line_clean)
                            new_entries.append(item)
                        except Exception:
                            pass
                current_line_index += 1

        self.processed_line_offset = current_line_index
        return new_entries

    async def process_batch(self) -> None:
        timeline_slice = self.read_new_timeline_slice()
        if len(timeline_slice) == 0:
            return

        running_context = self.load_running_context()
        initial_state: StructuringState = {
            "timeline_slice": timeline_slice,
            "running_context": running_context,
            "correlated_groups": [],
            "summary": "",
            "definitions": [],
            "action_items": [],
            "flashcards": [],
            "importance_flags": []
        }

        result_state = await self.compiled_graph.ainvoke(initial_state)

        summary = result_state.get("summary", "").strip()
        new_context = result_state.get("running_context", "")
        definitions = result_state.get("definitions", [])
        action_items = result_state.get("action_items", [])
        flashcards = result_state.get("flashcards", [])
        flags = result_state.get("importance_flags", [])

        if len(summary) > 0:
            with open(self.notes_file_path, "a", encoding="utf-8") as notes_file:
                if "exam_relevant" in flags:
                    notes_file.write("\n> **Important Exam Topic**\n\n")
                if "confusion" in flags:
                    notes_file.write("\n> ⚠️ *Review point: Concept identified as needing reinforcement.*\n\n")

                notes_file.write(summary + "\n\n")

                if len(action_items) > 0:
                    notes_file.write("### Action Items & Deadlines\n")
                    for action in action_items:
                        notes_file.write(f"- [ ] {action}\n")
                    notes_file.write("\n")

        if len(new_context) > 0:
            self.save_running_context(new_context)

        for definition in definitions:
            term = definition.get("term", "")
            meaning = definition.get("definition", "")
            self.concept_graph.add_or_update_concept(
                concept_name=term,
                session_id=self.session_id,
                timestamp_epoch=0.0,
                definition=meaning,
                related_concepts=[]
            )

        if len(flashcards) > 0:
            for card in flashcards:
                self.accumulated_flashcards.append(card)
            create_anki_deck(
                deck_name=f"LectureLens_{self.session_id}",
                flashcards=self.accumulated_flashcards,
                output_file_path=self.flashcards_file_path
            )

    async def run(self) -> None:
        self.is_running = True
        while self.is_running:
            await asyncio.sleep(self.settings.structuring_interval_seconds)
            try:
                await self.process_batch()
            except Exception:
                pass

    def stop(self) -> None:
        self.is_running = False


async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default=None)
    args, _ = parser.parse_known_args()

    settings = Settings()
    if args.session is not None and len(args.session) > 0:
        settings.session_name = args.session

    session_id = settings.session_name or "default_session"
    redis_bus = RedisBus(settings.redis_url)

    router_config = load_router_config(settings.router_config_path)
    llm_router = build_router_for_task(
        task_type="structuring",
        redis_client=redis_bus.client,
        router_config=router_config,
        ollama_url=settings.ollama_url
    )

    agent = StructuringAgent(
        settings=settings,
        session_id=session_id,
        llm_router=llm_router
    )
    try:
        await agent.run()
    finally:
        await redis_bus.close()


if __name__ == "__main__":
    asyncio.run(main())
