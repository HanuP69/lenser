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
from lecturelens.agents.camera_agent import CameraSource, compute_blur_score, compute_motion_score
from lecturelens.agents.screen_agent import capture_screen_frame, encode_frame_as_png
from lecturelens.config.settings import Settings
from lecturelens.core.concurrency import run_in_thread_pool
from lecturelens.core.image_utils import SlideChangeGate
from lecturelens.core.logging_setup import get_logger

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
                await send_and_wait(
                    websocket, encode_message("audio", "wav_b64", wav_bytes, time.time())
                )
    finally:
        await run_in_thread_pool(source.stop)


async def stream_screen(websocket: Any, settings: Settings, stop_event: asyncio.Event) -> None:
    gate = SlideChangeGate(
        settings.slide_diff_threshold,
        settings.slide_pixel_delta,
        settings.slide_min_interval_seconds
    )
    while not stop_event.is_set():
        try:
            frame = await run_in_thread_pool(
                capture_screen_frame, settings.monitor_index, settings.slide_thumbnail_width
            )
            if gate.evaluate(frame.thumbnail, time.monotonic()):
                captured_at = time.time()
                png_bytes = await run_in_thread_pool(encode_frame_as_png, frame)
                await send_and_wait(
                    websocket, encode_message("screen", "png_b64", png_bytes, captured_at)
                )
        except Exception as error:
            logger.error("screen streaming error: %s", error)
        await asyncio.sleep(settings.screenshot_interval_seconds)


async def stream_camera(websocket: Any, settings: Settings, stop_event: asyncio.Event) -> None:
    camera = CameraSource(settings.camera_device_index, settings.camera_warmup_frames)
    previous_frame: Optional[bytes] = None
    still_count = 0
    while not stop_event.is_set():
        try:
            frame_bytes = await run_in_thread_pool(camera.read_jpeg)
            frame_captured_at = time.time()
            if previous_frame is not None:
                motion = await run_in_thread_pool(compute_motion_score, previous_frame, frame_bytes)
                if motion > settings.motion_threshold:
                    still_count = 0
                else:
                    still_count += 1
                if still_count >= settings.motion_stillness_required_frames:
                    blur = await run_in_thread_pool(compute_blur_score, frame_bytes)
                    if blur >= settings.blur_variance_threshold:
                        await send_and_wait(
                            websocket, encode_message("camera", "jpeg_b64", frame_bytes, frame_captured_at)
                        )
                        still_count = 0
            previous_frame = frame_bytes
        except Exception as error:
            logger.error("camera streaming error: %s", error)
        await asyncio.sleep(settings.camera_poll_interval_seconds)
    await run_in_thread_pool(camera.close)


async def run_remote_client(
    settings: Settings,
    server_url: str,
    api_key: Optional[str],
    session_id: str,
    use_audio: bool,
    use_screen: bool,
    use_camera: bool,
    stop_event: asyncio.Event
) -> None:
    from websockets import connect
    async with connect(server_url, max_size=settings.gateway_max_message_bytes) as audio_socket:
        await audio_socket.send(json.dumps({"type": "hello", "api_key": api_key, "session_id": session_id}))
        hello_reply = json.loads(await audio_socket.recv())
        if hello_reply.get("status") != "authenticated":
            raise PermissionError(str(hello_reply))

        tasks: list[asyncio.Task] = []
        extra_sockets: list[Any] = []
        if use_audio:
            tasks.append(asyncio.create_task(stream_audio(audio_socket, settings, stop_event)))

        for enabled, streamer in ((use_screen, stream_screen), (use_camera, stream_camera)):
            if not enabled:
                continue
            extra_context = connect(server_url, max_size=settings.gateway_max_message_bytes)
            extra_socket = await extra_context.__aenter__()
            extra_sockets.append(extra_context)
            await extra_socket.send(json.dumps({"type": "hello", "api_key": api_key, "session_id": session_id}))
            reply = json.loads(await extra_socket.recv())
            if reply.get("status") != "authenticated":
                raise PermissionError(str(reply))
            tasks.append(asyncio.create_task(streamer(extra_socket, settings, stop_event)))

        try:
            if len(tasks) == 0:
                return
            await asyncio.gather(*tasks)
        finally:
            stop_event.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for extra_context in extra_sockets:
                await extra_context.__aexit__(None, None, None)
