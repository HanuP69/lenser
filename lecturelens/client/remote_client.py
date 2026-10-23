import asyncio
import base64
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Optional


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
    capture_epoch: Optional[float] = None,
    capture_id: Optional[str] = None
) -> str:
    if capture_epoch is None:
        capture_epoch = time.time()
    if capture_id is None:
        capture_id = uuid.uuid4().hex
    encoded = base64.b64encode(payload).decode("ascii")
    return json.dumps({
        "type": message_type,
        field_name: encoded,
        "timestamp_epoch": capture_epoch,
        "capture_id": capture_id
    })


class CaptureSpoolFull(RuntimeError):
    """Raised when the capture spool is full and the policy is to fail."""


class CaptureSpool:
    def __init__(
        self,
        spool_path: Path,
        max_size: int = 500,
        append_timeout_seconds: float = 5.0,
        on_full: str = "drop_oldest"
    ) -> None:
        self.spool_path = spool_path
        self.max_size = max_size
        self.append_timeout_seconds = append_timeout_seconds
        # "drop_oldest" keeps the capture running and loses the oldest
        # pending capture. "raise" loses nothing: the capture itself is
        # dropped and only a log line reports it, so the client never
        # receives that capture.
        self.on_full = on_full
        self.dropped_count = 0
        # How often each capture failed to be delivered. It lives on the
        # spool, which survives the reconnects in the same process, so a
        # capture that the server always refuses is noticed across the
        # sessions.
        self.delivery_failures: dict[str, int] = {}
        self.spool_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self.spool_path), timeout=5.0)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pending_captures (
                    capture_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
            """)
            conn.commit()
        finally:
            conn.close()

    def _try_append(self, capture_id: str, payload_text: str) -> bool:
        """Insert a capture. Returns False when the spool is full.

        The blocking wait for space happens in the async wrapper, so the
        thread pool is never blocked. A blocked append used to delay the
        acknowledgements that would have freed the space.
        """
        conn = sqlite3.connect(str(self.spool_path), timeout=5.0)
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM pending_captures;")
            count = int(cur.fetchone()[0])
            if count >= self.max_size:
                return False
            cur.execute(
                "INSERT OR REPLACE INTO pending_captures (capture_id, payload, created_at) VALUES (?, ?, ?);",
                (capture_id, payload_text, time.time())
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def _evict_oldest(self) -> None:
        conn = sqlite3.connect(str(self.spool_path), timeout=5.0)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT capture_id FROM pending_captures ORDER BY created_at ASC LIMIT 1;"
            )
            old_rows = cur.fetchall()
            if old_rows:
                cur.execute(
                    "DELETE FROM pending_captures WHERE capture_id = ?;", (old_rows[0][0],)
                )
                conn.commit()
        finally:
            conn.close()

    async def append(self, capture_id: str, payload_text: str) -> None:
        """Add a capture to the spool, applying backpressure when it is full.

        The wait happens on the event loop, not in a thread pool worker, so
        the acknowledgements that free the space can still run.
        """
        deadline = time.monotonic() + self.append_timeout_seconds
        while True:
            inserted = await run_in_thread_pool(self._try_append, capture_id, payload_text)
            if inserted:
                return
            if time.monotonic() >= deadline:
                break
            # Backpressure: wait for the sender to acknowledge something
            # instead of dropping the capture.
            await asyncio.sleep(0.05)

        # The spool stayed full.
        if self.on_full == "raise":
            # Nothing is dropped: the capture itself fails, so the caller can
            # decide whether to keep the audio or stop the capture.
            raise CaptureSpoolFull(
                f"capture spool {self.spool_path} is full ({self.max_size} captures) "
                f"after {self.append_timeout_seconds:.1f}s"
            )

        # Dropping the oldest undelivered capture would lose lecture audio,
        # so the failure is explicit and counted.
        self.dropped_count += 1
        logger.error(
            "capture spool %s is full (%d captures) after %.1fs; dropping the "
            "oldest pending capture (%d dropped so far)",
            self.spool_path, self.max_size, self.append_timeout_seconds,
            self.dropped_count
        )
        await run_in_thread_pool(self._evict_oldest)
        inserted = await run_in_thread_pool(self._try_append, capture_id, payload_text)
        if not inserted:
            # The spool filled up again between the eviction and the
            # second insert. That is two captures lost instead of one,
            # which the "drop the oldest" policy does not promise, so it
            # is counted like a normal drop.
            self.dropped_count += 1
            logger.error(
                "capture spool %s filled up again while appending; the capture "
                "was dropped as well (%d dropped so far)",
                self.spool_path, self.dropped_count
            )

    def acknowledge(self, capture_id: str) -> None:
        conn = sqlite3.connect(str(self.spool_path), timeout=5.0)
        try:
            conn.execute("DELETE FROM pending_captures WHERE capture_id = ?;", (capture_id,))
            conn.commit()
        finally:
            conn.close()

    def get_pending(self, limit: int = 50) -> list[tuple[str, str]]:
        conn = sqlite3.connect(str(self.spool_path), timeout=5.0)
        try:
            cur = conn.cursor()
            cur.execute("SELECT capture_id, payload FROM pending_captures ORDER BY created_at ASC LIMIT ?;", (limit,))
            return cur.fetchall()
        finally:
            conn.close()

    def count(self) -> int:
        conn = sqlite3.connect(str(self.spool_path), timeout=5.0)
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM pending_captures;")
            return int(cur.fetchone()[0])
        finally:
            conn.close()


def is_connection_closed(error: BaseException, websocket: Any = None) -> bool:
    """Decide whether a send failure means the connection is gone.

    The check used to look for the substring "close" in the error message and
    for a ``websocket.closed`` attribute. The modern websockets client has no
    ``closed`` attribute and a transport level reset ("connection reset by
    peer") contains neither, so the loop kept running on a dead socket and
    the client never reconnected.
    """
    try:
        from websockets.exceptions import ConnectionClosed
    except ImportError:
        ConnectionClosed = ()  # type: ignore[assignment]
    if ConnectionClosed and isinstance(error, ConnectionClosed):
        return True
    if isinstance(error, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, EOFError)):
        return True
    closed_state = getattr(websocket, "closed", None)
    if closed_state is True:
        return True
    closed_property = getattr(websocket, "close_code", None)
    if closed_property is not None:
        return True
    return False


# How often a capture may be refused by the server before it is given up
# on and counted, so it cannot block the rest of the spool forever. Each
# refusal makes the client reconnect, so the number is small.
SPOOL_MAX_DELIVERY_ATTEMPTS: int = 3


def is_rejection_error(error: BaseException) -> bool:
    """Whether a failed delivery was the server refusing the capture.

    The rejection is a RuntimeError with the answer of the server in it,
    which send_and_wait raises after its retries. A timeout or a closed
    connection says nothing about the capture and is left alone.
    """
    return isinstance(error, RuntimeError) and "server rejected" in str(error).lower()


async def drain_spool(
    spool: CaptureSpool,
    websocket: Any,
    stop_event: Optional[asyncio.Event] = None,
    batch_size: int = 25
) -> int:
    """Send every pending capture.

    The drain used to send a single batch of 25 per connection, so a client
    that had been offline longer than that never delivered the rest of the
    backlog.
    """
    delivered = 0
    while True:
        if stop_event is not None and stop_event.is_set():
            break
        pending = await run_in_thread_pool(spool.get_pending, batch_size)
        if len(pending) == 0:
            break
        progressed = False
        for cap_id, frame_str in pending:
            if stop_event is not None and stop_event.is_set():
                break
            try:
                await send_and_wait(websocket, frame_str, max_retries=2, stop_event=stop_event, timeout=5.0)
            except Exception as send_error:
                spool.delivery_failures[cap_id] = spool.delivery_failures.get(cap_id, 0) + 1
                attempts = spool.delivery_failures[cap_id]
                logger.warning("could not send a spooled capture: %s", send_error)
                if attempts >= SPOOL_MAX_DELIVERY_ATTEMPTS and is_rejection_error(send_error):
                    # The server refuses this capture every time it sees
                    # it, and it is the head of the spool, so a capture
                    # that stays pending blocks every capture behind it
                    # for the lifetime of the spool. After enough
                    # rejections it is given up on and counted, so the
                    # rest of the backlog can be delivered. A capture
                    # that failed because the connection was gone is not
                    # given up on: the network changes, the server's
                    # answer to the same bytes does not.
                    await run_in_thread_pool(spool.acknowledge, cap_id)
                    spool.delivery_failures.pop(cap_id, None)
                    spool.dropped_count += 1
                    progressed = True
                    logger.error(
                        "capture %s was rejected %d times and is given up on "
                        "(%d dropped so far); the rest of the spool can drain",
                        cap_id, attempts, spool.dropped_count
                    )
                break
            spool.delivery_failures.pop(cap_id, None)
            await run_in_thread_pool(spool.acknowledge, cap_id)
            delivered += 1
            progressed = True
        if not progressed:
            break
    return delivered


async def wait_with_stop(coro: Any, stop_event: Optional[asyncio.Event] = None, timeout: float = 10.0) -> Any:
    if stop_event is not None and stop_event.is_set():
        raise asyncio.CancelledError("Stopped before starting operation")
    op_task = asyncio.create_task(coro)
    tasks: list[asyncio.Task] = [op_task]
    if stop_event is not None:
        stop_task = asyncio.create_task(stop_event.wait())
        tasks.append(stop_task)
    try:
        done, pending = await asyncio.wait(tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        for p in pending:
            p.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if stop_event is not None and stop_event.is_set():
            raise asyncio.CancelledError("Operation interrupted by stop_event")
        if op_task not in done:
            raise TimeoutError("Operation timed out")
        return op_task.result()
    except BaseException:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def send_and_wait(
    websocket: Any,
    text_frame: str,
    max_retries: int = 3,
    initial_backoff: float = 0.5,
    stop_event: Optional[asyncio.Event] = None,
    timeout: float = 10.0
) -> None:
    backoff = initial_backoff
    last_reply = None
    for attempt in range(max_retries):
        if stop_event is not None and stop_event.is_set():
            raise asyncio.CancelledError("Stopped during frame send")
        await wait_with_stop(websocket.send(text_frame), stop_event, timeout=timeout)
        raw_reply = await wait_with_stop(websocket.recv(), stop_event, timeout=timeout)
        reply = json.loads(raw_reply)
        if reply.get("status") == "acknowledged":
            return
        last_reply = reply
        logger.warning(
            "server rejected frame (attempt %d/%d): %s",
            attempt + 1,
            max_retries,
            reply
        )
        if attempt < max_retries - 1:
            if stop_event is not None:
                await wait_with_stop(asyncio.sleep(backoff), stop_event, timeout=backoff + 1.0)
            else:
                await asyncio.sleep(backoff)
            backoff *= 2.0
    raise RuntimeError(f"server rejected frame after {max_retries} attempts: {last_reply}")


async def stream_audio(
    websocket: Any,
    settings: Settings,
    stop_event: asyncio.Event,
    spool: Optional[CaptureSpool] = None
) -> None:
    source = build_block_source(settings)
    try:
        await run_in_thread_pool(source.start)
        segmenter = SpeechSegmenter(
            sample_rate=settings.mic_sample_rate,
            energy_threshold=settings.vad_energy_threshold,
            min_chunk_seconds=settings.mic_chunk_duration_seconds,
            max_chunk_seconds=settings.audio_max_chunk_seconds,
            silence_gap_seconds=settings.audio_silence_gap_seconds
        )
        started = time.monotonic()
        if spool is not None:
            try:
                await drain_spool(spool, websocket, stop_event)
            except Exception as drain_err:
                logger.warning("draining audio spool encountered error: %s", drain_err)
        while not stop_event.is_set():
            block = await run_in_thread_pool(source.read_block, 0.5)
            if block is None:
                continue
            for chunk in segmenter.push(block, time.monotonic() - started):
                wav_bytes = convert_numpy_to_wav_bytes(chunk.samples, settings.mic_sample_rate)
                cap_id = uuid.uuid4().hex
                msg_text = encode_message("audio", "wav_b64", wav_bytes, time.time(), capture_id=cap_id)
                if spool is not None:
                    try:
                        await spool.append(cap_id, msg_text)
                    except CaptureSpoolFull as spool_error:
                        # The spool is full and the policy is to lose
                        # nothing, so the capture is dropped instead of the
                        # oldest pending one.
                        logger.error("dropping a capture, the spool is full: %s", spool_error)
                        continue
                try:
                    await send_and_wait(websocket, msg_text, stop_event=stop_event)
                    if spool is not None:
                        await run_in_thread_pool(spool.acknowledge, cap_id)
                except Exception as stream_err:
                    logger.error("audio streaming error: %s", stream_err)
                    raise
    finally:
        await run_in_thread_pool(source.stop)


async def stream_screen(
    websocket: Any,
    settings: Settings,
    stop_event: asyncio.Event,
    spool: Optional[CaptureSpool] = None
) -> None:
    gate = SlideChangeGate(
        settings.slide_diff_threshold,
        settings.slide_pixel_delta,
        settings.slide_min_interval_seconds,
        embedder=get_shared_vision_embedder()
    )
    if spool is not None:
        try:
            await drain_spool(spool, websocket, stop_event)
        except Exception as drain_err:
            logger.warning("draining screen spool encountered error: %s", drain_err)
    while not stop_event.is_set():
        try:
            frame = await run_in_thread_pool(
                capture_screen_frame, settings.monitor_index, settings.slide_thumbnail_width
            )
            gate_accepted = await run_in_thread_pool(gate.evaluate, frame.thumbnail, time.monotonic())
            if gate_accepted:
                captured_at = time.time()
                png_bytes = await run_in_thread_pool(encode_frame_as_png, frame)
                cap_id = uuid.uuid4().hex
                msg_text = encode_message("screen", "png_b64", png_bytes, captured_at, capture_id=cap_id)
                if spool is not None:
                    try:
                        await spool.append(cap_id, msg_text)
                    except CaptureSpoolFull as spool_error:
                        # The spool is full and the policy is to lose
                        # nothing, so the capture is dropped instead of the
                        # oldest pending one.
                        logger.error("dropping a capture, the spool is full: %s", spool_error)
                        continue
                await send_and_wait(websocket, msg_text, stop_event=stop_event)
                if spool is not None:
                    await run_in_thread_pool(spool.acknowledge, cap_id)
        except Exception as error:
            logger.error("screen streaming error: %s", error)
            if is_connection_closed(error, websocket):
                break
        await asyncio.sleep(settings.screenshot_interval_seconds)


async def connect_and_auth(
    server_url: str,
    max_bytes: int,
    api_key: Optional[str],
    session_id: str,
    stop_event: asyncio.Event
) -> tuple[Any, Any]:
    from websockets import connect
    ctx = connect(server_url, max_size=max_bytes)
    entered = False
    try:
        socket = await wait_with_stop(ctx.__aenter__(), stop_event, timeout=10.0)
        entered = True
        hello_payload = json.dumps({"type": "hello", "api_key": api_key, "session_id": session_id})
        await wait_with_stop(socket.send(hello_payload), stop_event, timeout=5.0)
        raw_reply = await wait_with_stop(socket.recv(), stop_event, timeout=10.0)
        reply = json.loads(raw_reply)
        if reply.get("status") != "authenticated":
            raise PermissionError(str(reply))
        return ctx, socket
    except BaseException:
        if entered:
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:
                pass
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

    session_dir = settings.get_session_output_dir(session_id)
    audio_spool = CaptureSpool(
        session_dir / "capture_spool_audio.db",
        max_size=settings.capture_spool_max_size,
        append_timeout_seconds=settings.capture_spool_append_timeout_seconds,
        on_full=settings.capture_spool_on_full
    )
    screen_spool = CaptureSpool(
        session_dir / "capture_spool_screen.db",
        max_size=settings.capture_spool_max_size,
        append_timeout_seconds=settings.capture_spool_append_timeout_seconds,
        on_full=settings.capture_spool_on_full
    )

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
                tasks.append(asyncio.create_task(stream_audio(audio_socket, settings, session_stop_event, spool=audio_spool)))

            if use_screen:
                screen_ctx, screen_socket = await connect_and_auth(
                    server_url, settings.gateway_max_message_bytes, api_key, session_id, stop_event
                )
                contexts.append(screen_ctx)
                tasks.append(asyncio.create_task(stream_screen(screen_socket, settings, session_stop_event, spool=screen_spool)))

            if len(tasks) == 1 and tasks[0] is stop_watcher:
                return
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            session_stop_event.set()
            if not stop_event.is_set():
                await asyncio.sleep(1.0)
        except PermissionError:
            raise
        except asyncio.CancelledError:
            break
        except Exception as conn_error:
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
