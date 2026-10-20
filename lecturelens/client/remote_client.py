import asyncio
import base64
import json
import time
from typing import Any, Optional

import numpy as np

from lecturelens.agents.audio_agent import (
    SpeechSegmenter,
    build_block_source,
    convert_numpy_to_wav_bytes
)
from lecturelens.agents.screen_agent import capture_screen_frame, encode_frame_as_png
from lecturelens.config.settings import Settings
from lecturelens.core.concurrency import run_in_thread_pool
from lecturelens.core.image_utils import SlideChangeGate
from lecturelens.core.logging_setup import get_logger
from lecturelens.core.vision_embedder import get_shared_vision_embedder

logger = get_logger("lecturelens.client")


def encode_message(
    message_type: str,
    field_name: str,
    payload: bytes,
    capture_epoch: Optional[float] = None
) -> str:
    if capture_epoch is None:
        capture_epoch = time.time()
    encoded = base64.b64encode(payload).decode("ascii")
    return json.dumps({"type": message_type, field_name: encoded, "timestamp_epoch": capture_epoch})


async def send_and_wait(websocket: Any, text_frame: str) -> None:
    await websocket.send(text_frame)
    reply = await websocket.recv()
    parsed = json.loads(reply)
    if parsed.get("status") != "acknowledged":
        logger.warning("server replied: %s", parsed)


async def stream_audio(websocket: Any, settings: Settings, stop_event: asyncio.Event) -> None:
    source = build_block_source(settings)
    await run_in_thread_pool(source.start)
    segmenter = SpeechSegmenter(
        sample_rate=settings.mic_sample_rate,
        energy_threshold=settings.vad_energy_threshold,
        min_chunk_seconds=settings.mic_chunk_duration_seconds,
        max_chunk_seconds=settings.audio_max_chunk_seconds,
        silence_gap_seconds=settings.audio_silence_gap_seconds
    )
    started = time.monotonic()
    try:
        while not stop_event.is_set():
            block = await run_in_thread_pool(source.read_block, 0.5)
            if block is None:
                continue
            for chunk in segmenter.push(block, time.monotonic() - started):
                wav_bytes = convert_numpy_to_wav_bytes(chunk.samples, settings.mic_sample_rate)
                try:
                    await send_and_wait(
                        websocket, encode_message("audio", "wav_b64", wav_bytes, time.time())
                    )
                except Exception as stream_err:
                    logger.warning("audio streaming connection closed: %s", stream_err)
                    return
    finally:
        await run_in_thread_pool(source.stop)


async def stream_screen(websocket: Any, settings: Settings, stop_event: asyncio.Event) -> None:
    gate = SlideChangeGate(
        settings.slide_diff_threshold,
        settings.slide_pixel_delta,
        settings.slide_min_interval_seconds,
        embedder=get_shared_vision_embedder()
    )
    while not stop_event.is_set():
        try:
            frame = await run_in_thread_pool(
                capture_screen_frame, settings.monitor_index, settings.slide_thumbnail_width
            )
            gate_accepted = await run_in_thread_pool(gate.evaluate, frame.thumbnail, time.monotonic())
            if gate_accepted:
                captured_at = time.time()
                png_bytes = await run_in_thread_pool(encode_frame_as_png, frame)
                await send_and_wait(
                    websocket, encode_message("screen", "png_b64", png_bytes, captured_at)
                )
        except Exception as error:
            logger.error("screen streaming error: %s", error)
            if "close" in str(error).lower() or getattr(websocket, "closed", False):
                break
        await asyncio.sleep(settings.screenshot_interval_seconds)


async def wait_with_stop(coro: Any, stop_event: asyncio.Event, timeout: float = 10.0) -> Any:
    if stop_event.is_set():
        raise asyncio.CancelledError("Stopped before starting operation")
    stop_task = asyncio.create_task(stop_event.wait())
    op_task = asyncio.create_task(coro)
    done, pending = await asyncio.wait([op_task, stop_task], timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
    for p in pending:
        p.cancel()
    if stop_event.is_set():
        raise asyncio.CancelledError("Operation interrupted by stop_event")
    if op_task not in done:
        raise TimeoutError("Operation timed out")
    return op_task.result()


async def connect_and_auth(
    server_url: str,
    max_bytes: int,
    api_key: Optional[str],
    session_id: str,
    stop_event: asyncio.Event
) -> tuple[Any, Any]:
    from websockets import connect
    ctx = connect(server_url, max_size=max_bytes)
    socket = await wait_with_stop(ctx.__aenter__(), stop_event, timeout=10.0)
    try:
        hello_payload = json.dumps({"type": "hello", "api_key": api_key, "session_id": session_id})
        await wait_with_stop(socket.send(hello_payload), stop_event, timeout=5.0)
        raw_reply = await wait_with_stop(socket.recv(), stop_event, timeout=10.0)
        reply = json.loads(raw_reply)
        if reply.get("status") != "authenticated":
            raise PermissionError(str(reply))
        return ctx, socket
    except Exception:
        await ctx.__aexit__(None, None, None)
        raise


async def run_remote_client(
    settings: Settings,
    server_url: str,
    api_key: Optional[str],
    session_id: str,
    use_audio: bool,
    use_screen: bool,
    stop_event: Optional[asyncio.Event] = None
) -> None:
    if stop_event is None:
        stop_event = asyncio.Event()

    while not stop_event.is_set():
        tasks: list[asyncio.Task] = []
        contexts: list[Any] = []
        session_stop_event = asyncio.Event()

        async def watch_stop() -> None:
            await stop_event.wait()
            session_stop_event.set()

        stop_watcher = asyncio.create_task(watch_stop())
        tasks.append(stop_watcher)

        try:
            audio_ctx, audio_socket = await connect_and_auth(
                server_url, settings.gateway_max_message_bytes, api_key, session_id, stop_event
            )
            contexts.append(audio_ctx)

            if use_audio:
                tasks.append(asyncio.create_task(stream_audio(audio_socket, settings, session_stop_event)))

            if use_screen:
                screen_ctx, screen_socket = await connect_and_auth(
                    server_url, settings.gateway_max_message_bytes, api_key, session_id, stop_event
                )
                contexts.append(screen_ctx)
                tasks.append(asyncio.create_task(stream_screen(screen_socket, settings, session_stop_event)))

            if len(tasks) == 1 and tasks[0] is stop_watcher:
                return
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            session_stop_event.set()
        except PermissionError:
            raise
        except (asyncio.CancelledError, Exception) as conn_error:
            if stop_event.is_set():
                break
            logger.warning("remote client connection lost (%s), reconnecting in 2s...", conn_error)
            await asyncio.sleep(2.0)
        finally:
            session_stop_event.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for ctx in contexts:
                try:
                    await ctx.__aexit__(None, None, None)
                except Exception:
                    pass
