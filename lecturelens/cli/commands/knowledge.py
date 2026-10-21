"""The commands that ask about a lecture that was recorded."""

import asyncio
from typing import Any, Optional

import typer

from lecturelens.cli.app import (
    app,
    console,
    ensure_session_exists,
    fail,
    resolve_session,
    safe_for_console
)
from lecturelens.config.settings import Settings
from lecturelens.core.agent_runtime import create_postgres_storage, create_redis_bus
from lecturelens.core.validation import sanitize_file_stem


@app.command()
def ask(
    question: str = typer.Argument(...),
    session: Optional[str] = typer.Option(None, "--session"),
    llm_mode: Optional[str] = typer.Option(None, "--llm-mode"),
    student_id: Optional[str] = typer.Option(None, "--student-id", help="Student ID for personalized tutoring and misconception tracking"),
) -> None:
    settings = Settings()
    settings.llm_mode = llm_mode if llm_mode is not None else settings.llm_mode
    session_id = resolve_session(settings, session)
    session_dir = ensure_session_exists(settings, session_id)

    async def run() -> tuple[str, list[str]]:
        from lecturelens.knowledge.qa_engine import QAEngine
        from lecturelens.router.llm_router import build_router_for_task, load_router_config

        bus = create_redis_bus(settings, component_name="qa_cli")
        try:
            router = None
            try:
                router = build_router_for_task(
                    "qa", bus.client, load_router_config(settings.router_config_path),
                    settings.ollama_url, settings.llm_mode, settings.llm_timeout_seconds
                )
            except Exception as error:
                console.print(f"[yellow]Router unavailable, using Ollama directly: {error}[/yellow]")
            storage = None
            if settings.postgres_url:
                try:
                    storage = create_postgres_storage(settings)
                except Exception:
                    storage = None

            if student_id and isinstance(student_id, str):
                import json
                from lecturelens.knowledge.student_chat import StudentChatAgent
                from lecturelens.knowledge.student_graph import StudentKnowledgeGraph

                safe_id = sanitize_file_stem(student_id)
                graph_db = settings.output_dir / safe_id / "student_graph.db"
                graph = StudentKnowledgeGraph(graph_db)
                chat_agent = StudentChatAgent(
                    llm_router=router,
                    student_graph=graph,
                    ollama_url=settings.ollama_url,
                    model_name=settings.local_model
                )

                timeline_events: list[dict[str, Any]] = []
                notes_segments: list[dict[str, Any]] = []
                if storage is not None:
                    try:
                        timeline_events = await storage.get_timeline_events(session_id, limit=40, descending=True)
                        notes_segments = await storage.get_general_notes_since(session_id, after_sequence_number=0)
                    except Exception:
                        pass

                if not timeline_events and (session_dir / "timeline.jsonl").exists():
                    try:
                        with open(session_dir / "timeline.jsonl", "r", encoding="utf-8") as file_handle:
                            for line in file_handle:
                                if line.strip():
                                    timeline_events.append(json.loads(line))
                        timeline_events = timeline_events[-40:]
                    except Exception:
                        pass

                try:
                    return await chat_agent.ask(question, timeline_events=timeline_events, notes_segments=notes_segments)
                finally:
                    if storage is not None:
                        await storage.close()

            engine = QAEngine(
                session_dir,
                settings.ollama_url,
                router,
                embedding_model=settings.embedding_model,
                fallback_model=settings.local_model,
                storage=storage,
                embed_batch_size=settings.qa_embed_batch_size
            )
            try:
                return await engine.ask_with_sources(question)
            finally:
                if storage is not None:
                    await storage.close()
        finally:
            await bus.close()

    try:
        answer, references = asyncio.run(run())
    except Exception as error:
        fail(f"Could not answer: {error}")
    console.print(safe_for_console(answer))
    if len(references) > 0:
        console.print("\n[bold]Sources[/bold]")
        for reference in references:
            console.print(safe_for_console(f"- {reference}"))


@app.command()
def review(
    course: Optional[str] = typer.Option(None, "--course"),
    session: Optional[str] = typer.Option(None, "--session"),
    student_id: Optional[str] = typer.Option(None, "--student-id"),
) -> None:
    settings = Settings()
    if student_id and isinstance(student_id, str):
        from lecturelens.knowledge.student_graph import StudentKnowledgeGraph
        safe_id = sanitize_file_stem(student_id)
        graph_db = settings.output_dir / safe_id / "student_graph.db"
        if graph_db.exists():
            graph = StudentKnowledgeGraph(graph_db)
            summary = graph.get_student_summary()
            console.print(f"[bold]Student {student_id} Mastery Summary[/bold]")
            console.print(f"Concepts tracked: {summary['concepts_tracked']}, Misconceptions: {summary['total_misconceptions']}")
            for c in summary["concepts"][:10]:
                decayed = c.get("decayed_mastery", c["mastery_level"])
                console.print(safe_for_console(f"  - {c['concept_name']}: mastery {c['mastery_level']:.2f} (decayed: {decayed:.2f})"))
            return
    if settings.postgres_url:

        course_val = course if isinstance(course, str) else None
        session_val = session if isinstance(session, str) else None

        async def fetch():
            storage = create_postgres_storage(settings)
            try:
                if course_val:
                    return await storage.get_concepts_for_course(course_val)
                elif session_val:
                    return await storage.get_all_concepts_for_session(session_val)
                return []
            finally:
                await storage.close()

        try:
            concepts = asyncio.run(fetch())
            if concepts:
                console.print(f"[bold]Concepts ({len(concepts)})[/bold]")
                for c in concepts:
                    console.print(safe_for_console(f"  - {c['concept_name']}: {c.get('definition') or 'No definition'} (Related: {', '.join(c.get('related_concepts') or [])})"))
                return
        except Exception as err:
            console.print(f"[yellow]Could not query Postgres concepts: {err}[/yellow]")
    console.print("No review records found.")
