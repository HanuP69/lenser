import asyncio
import datetime
import os
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from lecturelens.config.settings import Settings
from lecturelens.core.clock import SessionClock
from lecturelens.core.recovery import AgentMonitor
from lecturelens.core.redis_bus import RedisBus
from lecturelens.knowledge.concept_graph import ConceptGraph
from lecturelens.knowledge.qa_engine import QAEngine

app = typer.Typer(help="LectureLens: Real-Time Multimodal Lecture Notes System")
console = Console()


def generate_default_session_id() -> str:
    now = datetime.datetime.now()
    formatted = now.strftime("%Y-%m-%d_%H%M%S")
    return formatted


def render_dashboard_table(
    session_id: str,
    clock: SessionClock,
    monitor: AgentMonitor,
    pending_stats: dict[str, int]
) -> Table:
    elapsed_seconds = int(clock.get_elapsed_seconds())
    minutes = elapsed_seconds // 60
    seconds = elapsed_seconds % 60
    elapsed_str = f"{minutes:02d}:{seconds:02d}"

    table = Table(title=f"LectureLens Live Monitor — Session: [bold cyan]{session_id}[/bold cyan] (Elapsed: {elapsed_str})")
    table.add_column("Agent / Component", style="bold white")
    table.add_column("Process Status", style="bold")
    table.add_column("Pending Stream Lag", style="yellow")

    health_map = monitor.check_health()
    for agent_name, status in health_map.items():
        status_color = "green" if status == "RUNNING" else "red"
        status_display = f"[{status_color}]{status}[/{status_color}]"

        lag = pending_stats.get(agent_name, 0)
        lag_display = str(lag)

        table.add_row(agent_name, status_display, lag_display)

    return table


@app.command()
def start(
    mode: str = typer.Option("local", help="Deployment mode: local or server"),
    audio_source: str = typer.Option("mic", help="Audio source: mic, system, or both"),
    no_camera: bool = typer.Option(False, help="Disable camera agent"),
    no_screen: bool = typer.Option(False, help="Disable screen agent"),
    audio_only: bool = typer.Option(False, help="Capture audio only"),
    session: Optional[str] = typer.Option(None, help="Custom session identifier"),
    monitor_index: int = typer.Option(0, help="Screen display monitor index")
) -> None:
    settings = Settings()
    session_id = session if session is not None else generate_default_session_id()

    session_dir = settings.create_session_directories(session_id)
    clock = SessionClock()
    clock.save_to_file(session_dir / "clock.json")

    base_env = os.environ.copy()
    base_env["LECTURELENS_SESSION_NAME"] = session_id
    base_env["LECTURELENS_AUDIO_SOURCE"] = audio_source
    base_env["LECTURELENS_MONITOR_INDEX"] = str(monitor_index)

    monitor = AgentMonitor()

    python_executable = sys.executable
    logs_dir = session_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    monitor.register_agent(
        "sync_agent",
        [python_executable, "-m", "lecturelens.agents.sync_agent", "--session", session_id],
        env=base_env,
        log_file_path=logs_dir / "sync_agent.log"
    )
    monitor.register_agent(
        "structuring_agent",
        [python_executable, "-m", "lecturelens.agents.structuring_agent", "--session", session_id],
        env=base_env,
        log_file_path=logs_dir / "structuring_agent.log"
    )
    monitor.register_agent(
        "audio_agent",
        [python_executable, "-m", "lecturelens.agents.audio_agent", "--session", session_id, "--audio-source", audio_source],
        env=base_env,
        log_file_path=logs_dir / "audio_agent.log"
    )

    if not audio_only and not no_screen:
        monitor.register_agent(
            "screen_agent",
            [python_executable, "-m", "lecturelens.agents.screen_agent", "--session", session_id],
            env=base_env,
            log_file_path=logs_dir / "screen_agent.log"
        )

    if not audio_only and not no_camera:
        monitor.register_agent(
            "camera_agent",
            [python_executable, "-m", "lecturelens.agents.camera_agent", "--session", session_id],
            env=base_env,
            log_file_path=logs_dir / "camera_agent.log"
        )

    if mode == "server":
        monitor.register_agent(
            "gateway_server",
            [python_executable, "-m", "lecturelens.gateway.server"],
            env=base_env,
            log_file_path=logs_dir / "gateway_server.log"
        )

    console.print(f"[bold green]Starting LectureLens session: {session_id}[/bold green]")
    for agent_name in monitor.command_templates.keys():
        monitor.start_agent(agent_name)

    async def run_live_loop() -> None:
        redis_bus = RedisBus(settings.redis_url)
        with Live(console=console, refresh_per_second=1) as live:
            try:
                while True:
                    monitor.restart_failed_agents()

                    pending_stats = {
                        "sync_agent": await redis_bus.get_pending_count(
                            f"lecturelens:{session_id}:stream:transcript",
                            "group:kuudere"
                        ),
                        "audio_agent": 0,
                        "screen_agent": 0,
                        "camera_agent": 0,
                        "structuring_agent": 0,
                        "gateway_server": 0
                    }

                    table = render_dashboard_table(
                        session_id=session_id,
                        clock=clock,
                        monitor=monitor,
                        pending_stats=pending_stats
                    )
                    live.update(table)
                    await asyncio.sleep(1.0)
            except (KeyboardInterrupt, asyncio.CancelledError):
                pass
            finally:
                console.print("[yellow]Stopping all agents cleanly...[/yellow]")
                monitor.stop_all()
                await redis_bus.close()
                console.print("[green]Session finished cleanly.[/green]")

    try:
        asyncio.run(run_live_loop())
    except KeyboardInterrupt:
        monitor.stop_all()


@app.command()
def status(
    session: Optional[str] = typer.Option(None, help="Session identifier")
) -> None:
    settings = Settings()
    session_id = session if session is not None else "default_session"

    async def fetch_status() -> None:
        redis_bus = RedisBus(settings.redis_url)
        streams = [
            f"lecturelens:{session_id}:stream:transcript",
            f"lecturelens:{session_id}:stream:slide",
            f"lecturelens:{session_id}:stream:board"
        ]

        table = Table(title=f"Redis Streams Status — Session: {session_id}")
        table.add_column("Stream Key", style="cyan")
        table.add_column("Message Length", style="green")
        table.add_column("First Entry ID", style="dim")
        table.add_column("Last Entry ID", style="bold")

        for stream_key in streams:
            info = await redis_bus.get_stream_info(stream_key)
            length = str(info.get("length", 0))
            first_id = str(info.get("first-entry", ["None"])[0] if info.get("first-entry") else "None")
            last_id = str(info.get("last-entry", ["None"])[0] if info.get("last-entry") else "None")
            table.add_row(stream_key, length, first_id, last_id)

        console.print(table)
        await redis_bus.close()

    asyncio.run(fetch_status())


@app.command()
def export(
    session: str = typer.Option(..., help="Session identifier to export"),
    format_type: str = typer.Option("all", "--format", "-f", help="Export format: md, anki, or all"),
    destination: Path = typer.Option(Path("./export_output"), help="Destination directory")
) -> None:
    settings = Settings()
    session_dir = settings.get_session_output_dir(session)
    if not session_dir.exists():
        console.print(f"[red]Error: Session directory does not exist: {session_dir}[/red]")
        raise typer.Exit(code=1)

    destination.mkdir(parents=True, exist_ok=True)

    if format_type in ("md", "all"):
        source_notes = session_dir / "notes.md"
        if source_notes.exists():
            target_notes = destination / f"{session}_notes.md"
            shutil.copyfile(source_notes, target_notes)
            console.print(f"[green]Exported markdown notes to: {target_notes}[/green]")
        else:
            console.print("[yellow]Notice: notes.md not found in session.[/yellow]")

    if format_type in ("anki", "all"):
        source_anki = session_dir / "flashcards.apkg"
        if source_anki.exists():
            target_anki = destination / f"{session}_flashcards.apkg"
            shutil.copyfile(source_anki, target_anki)
            console.print(f"[green]Exported Anki deck to: {target_anki}[/green]")
        else:
            console.print("[yellow]Notice: flashcards.apkg not found in session.[/yellow]")


@app.command()
def ask(
    session: str = typer.Option(..., help="Session identifier"),
    question: str = typer.Argument(..., help="Question to ask about the lecture")
) -> None:
    settings = Settings()
    session_dir = settings.get_session_output_dir(session)
    if not session_dir.exists():
        console.print(f"[red]Error: Session directory does not exist: {session_dir}[/red]")
        raise typer.Exit(code=1)

    qa_engine = QAEngine(session_dir=session_dir, ollama_url=settings.ollama_url)

    async def run_query() -> None:
        console.print("[cyan]Indexing lecture timeline and notes...[/cyan]")
        indexed_count = await qa_engine.build_index()
        console.print(f"[cyan]Indexed {indexed_count} items. Querying lecture knowledge base...[/cyan]")
        answer = await qa_engine.ask(question)
        console.print(Panel(answer, title="LectureLens Answer", subtitle=f"Session: {session}"))

    asyncio.run(run_query())


@app.command()
def review(
    session: Optional[str] = typer.Option(None, help="Specific session identifier"),
    course_db: Optional[Path] = typer.Option(None, help="Custom concepts.db path")
) -> None:
    settings = Settings()
    if course_db is not None:
        db_path = course_db
    elif session is not None:
        db_path = settings.get_session_output_dir(session) / "concepts.db"
    else:
        db_path = settings.output_dir / "concepts.db"

    if not db_path.exists():
        console.print(f"[yellow]No concepts database found at: {db_path}[/yellow]")
        return

    graph = ConceptGraph(db_path)
    if session is not None:
        concepts = graph.get_all_concepts_for_session(session)
    else:
        concepts = graph.get_all_concepts()

    table = Table(title=f"Course Concepts Review ({len(concepts)} concepts found)")
    table.add_column("Concept / Term", style="bold cyan")
    table.add_column("Session", style="dim")
    table.add_column("Definition", style="white")

    for concept in concepts:
        table.add_row(
            concept["concept_name"],
            concept["session_id"],
            concept["definition"]
        )

    console.print(table)


@app.command()
def reprocess(
    session: str = typer.Option(..., help="Session identifier to reprocess")
) -> None:
    settings = Settings()
    session_dir = settings.get_session_output_dir(session)
    timeline_path = session_dir / "timeline.jsonl"

    if not timeline_path.exists():
        console.print(f"[red]Error: timeline.jsonl not found in {session_dir}[/red]")
        raise typer.Exit(code=1)

    from lecturelens.router.llm_router import build_router_for_task, load_router_config
    from lecturelens.agents.structuring_agent import StructuringAgent

    async def run_reprocess() -> None:
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
            session_id=session,
            llm_router=llm_router
        )
        console.print(f"[cyan]Reprocessing timeline for session: {session}...[/cyan]")
        await agent.process_batch()
        await redis_bus.close()
        console.print(f"[green]Reprocessing complete. Updated notes.md and flashcards.apkg.[/green]")

    asyncio.run(run_reprocess())


if __name__ == "__main__":
    app()
