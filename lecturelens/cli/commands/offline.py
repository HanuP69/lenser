"""The command that turns a recorded lecture file into notes."""

import asyncio
from pathlib import Path
from typing import Optional

import typer

from lecturelens.cli.app import app, console, fail, logger, option_value
from lecturelens.config.settings import Settings
from lecturelens.core.offline_pipeline import run_offline_video
from lecturelens.core.validation import is_valid_session_id


@app.command()
def offline(
    video: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="The lecture video file to play through the pipeline."
    ),
    session: str = typer.Option(..., "--session", help="The name of the session folder."),
    course: Optional[str] = typer.Option(None, "--course", help="Overrides the course name for the prompts."),
    fps: Optional[float] = typer.Option(None, "--fps", help="Frames per second sampled from the video."),
    llm_mode: Optional[str] = typer.Option(None, "--llm-mode", help="local, hybrid or cloud")
) -> None:
    """Notes from a recording, without any live devices.

    The file is decoded as fast as the machine can: the speech runs
    through the same segmentation and transcription agents, the frames
    through the same slide gate, and the notes are structured in this
    process without Redis or Postgres.
    """
    settings = Settings()
    if course:
        settings.course_name = course
    resolved_llm_mode = option_value(llm_mode)
    if resolved_llm_mode:
        settings.llm_mode = resolved_llm_mode
    if not is_valid_session_id(session):
        fail(f"Invalid session name: {session}")

    console.print(f"Playing [bold]{video.name}[/bold] for session [bold]{session}[/bold]")
    try:
        result = asyncio.run(run_offline_video(video, settings, session, fps))
    except Exception as error:
        logger.exception("the offline run failed")
        fail(f"The offline run failed: {error}")

    duration = ""
    if result.duration_seconds is not None:
        duration = f"{result.duration_seconds:.0f} s of file time became "
    console.print(
        f"{duration}{result.transcript_events} transcript events and "
        f"{result.slide_events} slides on the timeline."
    )
    if result.note_segments == 0:
        console.print("[yellow]No notes were written. The model answers nothing here, "
                      "so check the logs of the run above.[/yellow]")
    console.print(f"Notes: {result.session_dir / 'notes.md'}")
    console.print(f"Timeline: {result.session_dir / 'timeline.jsonl'}")
