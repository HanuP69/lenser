"""The commands that reach other machines."""

import asyncio
import os
from typing import Optional

import typer

from lecturelens.cli.app import (
    app,
    check_redis,
    console,
    ensure_session_exists,
    fail,
    resolve_session,
    validate_start_options
)
from lecturelens.config.settings import Settings
from lecturelens.core.agent_runtime import create_postgres_storage, create_redis_bus, read_gateway_api_key
from lecturelens.core.validation import is_valid_session_id, sanitize_file_stem


@app.command()
def student(
    server: Optional[str] = typer.Option(None, "--server", help="WebSocket URL of lecture gateway"),
    session: str = typer.Option(..., "--session", help="Session ID to subscribe to"),
    student_id: str = typer.Option(
        os.environ.get("USERNAME") or os.environ.get("USER") or "student",
        "--student-id",
        help="Unique student ID"
    ),
    language: str = typer.Option("English", "--language", help="Target language/style (e.g. English, Hindi, Hinglish)"),
    detail: str = typer.Option("comprehensive", "--detail", help="Detail level (comprehensive, concise, beginner)"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="LECTURELENS_GATEWAY_API_KEY"),
) -> None:
    settings = Settings()
    client_host = "127.0.0.1" if settings.gateway_host in ("0.0.0.0", "::") else settings.gateway_host
    resolved_server = server or f"ws://{client_host}:{settings.gateway_port}"
    from lecturelens.client.student_client import StudentClient, StudentProfile
    from lecturelens.knowledge.student_graph import StudentKnowledgeGraph

    safe_id = sanitize_file_stem(student_id)
    profile = StudentProfile(
        student_id=safe_id,
        preferred_language=language,
        detail_level=detail
    )
    if not api_key:
        # The gateway generates an ephemeral key when none is configured and
        # writes it next to the session, so a local client can pick it up.
        api_key = read_gateway_api_key(settings.get_session_output_dir(session))
    student_output_dir = (settings.output_dir / safe_id).resolve()
    try:
        student_output_dir.relative_to(settings.output_dir.resolve())
    except ValueError:
        fail("Invalid student ID path")
    graph_db = student_output_dir / "student_graph.db"
    graph = StudentKnowledgeGraph(graph_db)

    router = None
    bus = None
    try:
        from lecturelens.router.llm_router import build_router_for_task, load_router_config
        config = load_router_config(settings.router_config_path)
        redis_client = None
        if check_redis(settings):
            bus = create_redis_bus(settings, component_name=f"student_{safe_id}")
            redis_client = bus.client
        router = build_router_for_task(
            "structuring",
            redis_client,
            config,
            settings.ollama_url,
            settings.llm_mode,
            settings.llm_timeout_seconds
        )
    except Exception:
        try:
            from langchain_ollama import ChatOllama
            router = ChatOllama(
                base_url=settings.ollama_url,
                model=settings.local_model,
                num_ctx=8192,
                temperature=0.2
            )
        except Exception:
            router = None

    client = StudentClient(
        server_url=resolved_server,
        session_id=session,
        profile=profile,
        output_dir=student_output_dir,
        api_key=api_key,
        llm_router=router,
        student_graph=graph
    )

    console.print(f"[green]Connecting student client '{student_id}' to {resolved_server} (session: {session}, language: {language})...[/green]")
    async def _run_student() -> None:
        try:
            await client.run()
        finally:
            if bus is not None:
                try:
                    await bus.close()
                except Exception:
                    pass

    try:
        asyncio.run(_run_student())
    except KeyboardInterrupt:
        console.print("[yellow]Student client stopped.[/yellow]")


@app.command()
def persister(
    session: Optional[str] = typer.Option(None, "--session", help="Session ID to persist"),
) -> None:
    settings = Settings()
    if not settings.postgres_url:
        fail("LECTURELENS_POSTGRES_URL is not configured.")
    session_id = resolve_session(settings, session)
    from lecturelens.storage.persister import PostgresPersister
    async def run() -> None:
        redis_bus = create_redis_bus(settings, component_name="persister")
        storage = create_postgres_storage(settings)
        await storage.init_schema()
        persister_worker = PostgresPersister(settings, redis_bus, storage)
        console.print(f"[green]Persister listening on streams for session {session_id}...[/green]")
        try:
            await persister_worker.run_for_session(session_id)
        finally:
            await storage.close()
            await redis_bus.close()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        console.print("[yellow]Persister stopped.[/yellow]")


@app.command()
def client(
    session: str = typer.Option(..., "--session"),
    server: Optional[str] = typer.Option(None, "--server", help="WebSocket URL of lecture gateway"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="LECTURELENS_GATEWAY_API_KEY"),
    audio_source: Optional[str] = typer.Option(None, "--audio-source"),
    no_screen: bool = typer.Option(False, "--no-screen"),
    audio_only: bool = typer.Option(False, "--audio-only"),
    monitor_index: Optional[int] = typer.Option(None, "--monitor"),
) -> None:
    settings = Settings()
    # The option defaults used to be exported unconditionally, so the values
    # configured in .env were always overridden.
    resolved_audio_source = audio_source if audio_source is not None else settings.audio_source
    resolved_monitor_index = monitor_index if monitor_index is not None else settings.monitor_index
    try:
        validate_start_options(resolved_audio_source, "local", "local")
    except ValueError as error:
        fail(str(error))
    if resolved_audio_source == "system":
        from lecturelens.agents.audio_agent import find_loopback_microphone
        _, loopback_err = find_loopback_microphone()
        if loopback_err:
            fail(f"Cannot use '--audio-source system': {loopback_err}")
    if not is_valid_session_id(session):
        fail(f"Invalid session name: {session}")
    from lecturelens.client.remote_client import run_remote_client

    settings.audio_source = resolved_audio_source
    settings.monitor_index = resolved_monitor_index
    client_host = "127.0.0.1" if settings.gateway_host in ("0.0.0.0", "::") else settings.gateway_host
    resolved_server = server or f"ws://{client_host}:{settings.gateway_port}"
    if not api_key:
        api_key = read_gateway_api_key(settings.get_session_output_dir(session))

    async def run() -> None:
        stop_event = asyncio.Event()
        await run_remote_client(
            settings, resolved_server, api_key, session,
            True, (not audio_only and not no_screen),
            stop_event
        )

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        console.print("Client stopped.")


@app.command()
def replay(
    session: str = typer.Option(..., "--session"),
    speed: float = typer.Option(1.0, "--speed", help="Playback speed multiplier (0 for instant)"),
    target_session: Optional[str] = typer.Option(None, "--target-session", help="Session ID to emit events into"),
) -> None:
    from lecturelens.core.replay import ReplayEngine

    settings = Settings()
    session_id = resolve_session(settings, session)
    session_dir = ensure_session_exists(settings, session_id)
    timeline_path = session_dir / "timeline.jsonl"
    if not timeline_path.exists():
        fail(f"No timeline.jsonl found in {session_dir}")

    async def run() -> tuple[int, int, int]:
        outbox_file = session_dir / "redis_outbox_replay.db"
        bus = create_redis_bus(
            settings,
            outbox_path=outbox_file,
            component_name=f"replay_{session_id}"
        )
        try:
            engine = ReplayEngine(timeline_path, redis_bus=bus, speed=speed)
            total = await engine.replay(target_session_id=target_session or session_id)
            return total, engine.published_count, engine.buffered_count
        finally:
            await bus.close()

    try:
        total, published, buffered = asyncio.run(run())
        if buffered > 0:
            console.print(f"[yellow]Replayed {total} event(s) at {speed}x speed ({published} published to Redis, {buffered} buffered in outbox).[/yellow]")
        else:
            console.print(f"[green]Replayed {total} event(s) at {speed}x speed.[/green]")
    except Exception as error:
        fail(f"Replay failed: {error}")
