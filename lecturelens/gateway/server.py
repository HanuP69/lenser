import asyncio
import base64
import binascii
import hmac
import inspect
import json
import os
import re
import secrets
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from lecturelens.core.event_loop import ensure_event_loop_policy

ensure_event_loop_policy()

from lecturelens.agents.audio_agent import AudioAgent
from lecturelens.agents.screen_agent import ScreenAgent
from lecturelens.config.settings import Settings
from lecturelens.core.agent_runtime import (
    AgentControl,
    create_postgres_storage,
    create_redis_bus,
    execute_agent_main,
    parse_runtime_arguments,
    write_gateway_api_key
)
from lecturelens.core.clock import SessionClock
from lecturelens.core.concurrency import configure_pools, shutdown_pools
from lecturelens.core.logging_setup import get_logger
from lecturelens.core.repeat_guard import Guards
from lecturelens.core.redis_bus import RedisBus
from lecturelens.core.validation import is_valid_session_id
from lecturelens.schemas.events import GeneralNoteSegment, SlideCaptured, TranscriptChunk

logger = get_logger("lecturelens.gateway")

LOOPBACK_HOSTS: tuple[str, ...] = ("127.0.0.1", "localhost", "::1")
CAPTURE_PAST_TOLERANCE_SECONDS: float = 600.0
CAPTURE_FUTURE_TOLERANCE_SECONDS: float = 5.0
EVENT_SOURCE_TYPES: tuple[str, ...] = ("transcript", "slide")
EVENT_CLASSES: dict[str, Any] = {
    "transcript": TranscriptChunk,
    "slide": SlideCaptured
}


class GatewayConfigurationError(RuntimeError):
    pass


class MessageRateLimiter:
    def __init__(self, messages_per_second: float, burst: Optional[float] = None) -> None:
        self.rate: float = messages_per_second
        self.capacity: float = burst if burst is not None else max(1.0, messages_per_second * 2.0)
        self.tokens: float = self.capacity
        self.last_refill: float = time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.last_refill) * self.rate)
        self.last_refill = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


@dataclass
class SessionRuntime:
    session_id: str
    clock: SessionClock
    audio_agent: AudioAgent
    screen_agent: ScreenAgent
    connection_count: int = 0
    last_active: float = field(default_factory=time.monotonic)


def validate_gateway_binding(host: str, api_key: Optional[str]) -> None:
    if host not in LOOPBACK_HOSTS and not api_key:
        raise GatewayConfigurationError(
            "gateway bound to a non-loopback address requires LECTURELENS_GATEWAY_API_KEY"
        )


def keys_match(provided_key: str, expected_key: str) -> bool:
    return hmac.compare_digest(provided_key.encode("utf-8"), expected_key.encode("utf-8"))


def extract_origin(websocket: Any) -> Optional[str]:
    """Read the Origin header from a WebSocket handshake.

    Supports both the modern ``websockets.asyncio`` connection object and the
    legacy protocol object.
    """
    request = getattr(websocket, "request", None)
    if request is not None:
        headers = getattr(request, "headers", None)
        if headers is not None:
            origin = headers.get("Origin")
            if origin:
                return str(origin)
    origin = getattr(websocket, "origin", None)
    if origin:
        return str(origin)
    request_headers = getattr(websocket, "request_headers", None)
    if request_headers is not None:
        origin = request_headers.get("Origin")
        if origin:
            return str(origin)
    return None


def is_origin_allowed(origin: Optional[str], allowed_origins: Sequence[str]) -> bool:
    """Decide whether a handshake Origin may be accepted.

    A missing Origin means a non-browser client (the CLI, the Python client
    or a remote capture agent), which is allowed. A present Origin means a
    browser, and browsers neither enforce the same-origin policy for
    WebSockets nor can be trusted, so the origin has to be listed explicitly.
    """
    if origin is None or origin == "":
        return True
    return origin in list(allowed_origins or [])


def resolve_capture_epoch(message: dict[str, Any], clock: SessionClock) -> Optional[float]:
    raw_value = message.get("timestamp_epoch")
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        return None
    candidate = float(raw_value)
    server_now = clock.get_timestamp()
    if server_now - CAPTURE_PAST_TOLERANCE_SECONDS <= candidate <= server_now + CAPTURE_FUTURE_TOLERANCE_SECONDS:
        return candidate
    return None


def decode_base64_field(message: dict[str, Any], field_name: str) -> bytes:
    raw_value = message.get(field_name)
    if not isinstance(raw_value, str) or len(raw_value) == 0:
        raise ValueError(f"missing {field_name}")
    try:
        return base64.b64decode(raw_value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"invalid base64 in {field_name}") from error


class GatewayServer:
    def __init__(
        self,
        settings: Settings,
        redis_bus: RedisBus,
        llm_router: Any = None,
        storage: Any = None,
        instance_id: Optional[str] = None
    ) -> None:
        configured_key = settings.gateway_api_key
        self.api_key: str = ""
        if isinstance(configured_key, str) and len(configured_key.strip()) > 0:
            self.api_key = configured_key
        elif settings.gateway_require_api_key:
            # Without a key every local process could read the lecture notes
            # and inject audio, screen frames and events. Generate one so the
            # gateway is never open by default.
            self.api_key = secrets.token_urlsafe(32)
            logger.warning(
                "no LECTURELENS_GATEWAY_API_KEY configured; generated an ephemeral "
                "gateway API key. Set the variable explicitly so clients can connect."
            )
        validate_gateway_binding(settings.gateway_host, self.api_key)
        self.settings: Settings = settings
        self.redis_bus: RedisBus = redis_bus
        self.llm_router: Any = llm_router
        self.storage: Any = storage
        self.instance_id: str = instance_id or uuid.uuid4().hex[:8]
        self.sessions: dict[str, SessionRuntime] = {}
        self.subscribers: dict[str, Any] = {}
        self.catching_up: dict[str, dict[Any, list[dict[str, Any]]]] = {}
        self.subscriber_writers: dict[str, dict[Any, asyncio.Task]] = {}
        self.fanout_tasks: dict[str, asyncio.Task] = {}
        self.server: Any = None
        self.accepted_capture_ids: set[str] = set()
        self.in_flight_captures: dict[str, asyncio.Event] = {}
        self._accepted_capture_queue: list[tuple[float, str]] = []
        # The faults that are expected but should not be silent.
        self._guards = Guards(logger)

    def _mark_capture_accepted(self, capture_id: str) -> None:
        self.accepted_capture_ids.add(capture_id)
        self._accepted_capture_queue.append((time.monotonic(), capture_id))
        capacity = self.settings.gateway_capture_id_capacity
        ttl = self.settings.gateway_capture_id_ttl_seconds
        # The expiry was only enforced when the queue grew past the capacity,
        # so a capture id could stay accepted for the whole session. The
        # ttl is now enforced on every call.
        cutoff = time.monotonic() - ttl
        while self._accepted_capture_queue and self._accepted_capture_queue[0][0] < cutoff:
            _, old_id = self._accepted_capture_queue.pop(0)
            self.accepted_capture_ids.discard(old_id)
        while len(self._accepted_capture_queue) > capacity:
            _, old_id = self._accepted_capture_queue.pop(0)
            self.accepted_capture_ids.discard(old_id)

    async def broadcast_note_segment(self, session_id: str, note_data: dict[str, Any]) -> None:
        if session_id in self.catching_up:
            for pending_list in list(self.catching_up[session_id].values()):
                pending_list.append(note_data)

        subs = self.subscribers.get(session_id)
        if not subs:
            return

        payload = {"type": "note_segment", "payload": note_data}
        if isinstance(subs, dict):
            dead_sockets: list[Any] = []
            for ws, q in list(subs.items()):
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    logger.warning("subscriber queue full for %s, closing slow subscriber socket to trigger replay", session_id)
                    dead_sockets.append(ws)
                    try:
                        asyncio.create_task(ws.close(code=1008, reason="subscriber queue overflow"))
                    except Exception:
                        pass
                except Exception:
                    dead_sockets.append(ws)
            for dead in dead_sockets:
                subs.pop(dead, None)
        elif isinstance(subs, (set, list)):
            dead_sockets = []
            for ws in list(subs):
                try:
                    await self.send_json(ws, payload)
                except Exception:
                    dead_sockets.append(ws)
            if dead_sockets and session_id in self.subscribers:
                for dead in dead_sockets:
                    current = self.subscribers[session_id]
                    if isinstance(current, set):
                        current.discard(dead)
                    elif isinstance(current, list) and dead in current:
                        current.remove(dead)

    async def run_notes_fanout(self, session_id: str) -> None:
        stream_name = f"lecturelens:{session_id}:stream:notes"
        group_name = f"group:gateway_{session_id}_{self.instance_id}"
        # The consumer group has to start at the end of the stream. The
        # shared helper creates the group at offset 0, which made the
        # fanout replay the entire note history as if it were live and
        # overflowed the bounded subscriber queue.
        await self.redis_bus.ensure_consumer_group(stream_name, group_name, start_id="$")

        async def handler(data: dict[str, Any]) -> None:
            await self.broadcast_note_segment(session_id, data)

        while True:
            try:
                await self.redis_bus.consume(
                    stream_name=stream_name,
                    group_name=group_name,
                    consumer_name=f"gateway_{self.instance_id}_{session_id}",
                    handler=handler,
                    batch_size=5,
                    block_ms=500
                )
                return
            except asyncio.CancelledError:
                raise
            except Exception as error:
                # A dead fanout task silently stopped the live note delivery
                # for every connected subscriber, and nothing ever restarted
                # it.
                logger.error("fanout loop error for %s: %s", session_id, error)
                await asyncio.sleep(1.0)

    def get_or_create_session(self, session_id: str) -> SessionRuntime:
        existing = self.sessions.get(session_id)
        if existing is not None:
            return existing
        if len(self.sessions) >= self.settings.gateway_max_sessions:
            self.evict_idle_sessions(len(self.sessions) - self.settings.gateway_max_sessions + 1)
        if len(self.sessions) >= self.settings.gateway_max_sessions:
            raise ValueError("session limit reached")
        session_dir = self.settings.create_session_directories(session_id)
        clock = SessionClock.load_or_create(session_dir / "clock.json")
        runtime = SessionRuntime(
            session_id=session_id,
            clock=clock,
            audio_agent=AudioAgent(self.settings, self.redis_bus, clock, session_id),
            screen_agent=ScreenAgent(self.settings, self.redis_bus, clock, session_id, self.llm_router)
        )
        self.sessions[session_id] = runtime
        return runtime

    def evict_idle_sessions(self, count: int) -> int:
        idle_runtimes = sorted(
            (
                runtime for runtime in self.sessions.values()
                if runtime.connection_count <= 0 and not self.subscribers.get(runtime.session_id)
            ),
            key=lambda runtime: runtime.last_active
        )
        evicted = 0
        for runtime in idle_runtimes[:max(0, count)]:
            self.release_session(runtime.session_id)
            evicted += 1
        return evicted

    def release_session(self, session_id: str) -> None:
        runtime = self.sessions.pop(session_id, None)
        if runtime is None:
            return
        runtime.audio_agent.stop()
        runtime.screen_agent.stop()
        fanout = self.fanout_tasks.pop(session_id, None)
        if fanout is not None and not fanout.done():
            fanout.cancel()
        self.subscribers.pop(session_id, None)
        writers = self.subscriber_writers.pop(session_id, None)
        if writers:
            for wtask in writers.values():
                if not wtask.done():
                    wtask.cancel()
        self.catching_up.pop(session_id, None)
        logger.info("released idle gateway session %s", session_id)

    def authenticate(self, hello_message: dict[str, Any]) -> str:
        if hello_message.get("type") != "hello":
            raise PermissionError("expected hello message")
        expected_key = self.api_key
        if expected_key:
            provided_key = hello_message.get("api_key")
            if not isinstance(provided_key, str) or not keys_match(provided_key, expected_key):
                raise PermissionError("invalid api key")
        session_id = hello_message.get("session_id")
        if not isinstance(session_id, str) or not is_valid_session_id(session_id):
            raise PermissionError("invalid session id")
        return session_id

    def check_origin(self, websocket: Any) -> None:
        """Reject browser handshakes from an unknown origin.

        A web page can open ws://127.0.0.1:8765 even though the page itself
        is served from another site, because browsers do not apply the
        same-origin policy to WebSockets. Any handshake that carries an
        Origin header therefore has to match the configured allowlist.
        """
        origin = extract_origin(websocket)
        allowed = self.settings.gateway_allowed_origins
        if not is_origin_allowed(origin, allowed):
            raise PermissionError(f"origin not allowed: {origin}")

    async def handle_message(self, runtime: SessionRuntime, message: dict[str, Any]) -> dict[str, Any]:
        runtime.last_active = time.monotonic()
        capture_id = message.get("capture_id")
        is_tracked_capture = isinstance(capture_id, str) and bool(capture_id)
        in_flight_event: Optional[asyncio.Event] = None
        if is_tracked_capture:
            if capture_id in self.accepted_capture_ids:
                return {"status": "acknowledged", "duplicate": True}
            while capture_id in self.in_flight_captures:
                waiter = self.in_flight_captures.get(capture_id)
                if waiter is None:
                    break
                try:
                    await asyncio.wait_for(waiter.wait(), timeout=15.0)
                except Exception:
                    # The barrier timed out. The capture is still being
                    # processed by the other holder, so the wait is retried
                    # instead of falling open and processing the same capture
                    # twice. The client's ack timeout is shorter than this
                    # barrier, so a timeout here means the server is slow,
                    # not that the capture was lost.
                    if capture_id in self.accepted_capture_ids:
                        return {"status": "acknowledged", "duplicate": True}
                    continue
                if capture_id in self.accepted_capture_ids:
                    return {"status": "acknowledged", "duplicate": True}
            if capture_id in self.accepted_capture_ids:
                return {"status": "acknowledged", "duplicate": True}
            in_flight_event = asyncio.Event()
            self.in_flight_captures[capture_id] = in_flight_event
        try:
            capture_epoch = resolve_capture_epoch(message, runtime.clock)
            message_type = message.get("type")
            if message_type == "audio":
                wav_bytes = decode_base64_field(message, "wav_b64")
                await runtime.audio_agent.ingest_wav_bytes(wav_bytes, capture_epoch)
                if is_tracked_capture:
                    self._mark_capture_accepted(capture_id)
                return {"status": "acknowledged"}
            if message_type == "screen":
                png_bytes = decode_base64_field(message, "png_b64")
                sig = inspect.signature(runtime.screen_agent.ingest_png_bytes)
                if "enforce_gate" in sig.parameters:
                    await runtime.screen_agent.ingest_png_bytes(png_bytes, capture_epoch, enforce_gate=True)
                else:
                    await runtime.screen_agent.ingest_png_bytes(png_bytes, capture_epoch)
                if is_tracked_capture:
                    self._mark_capture_accepted(capture_id)
                return {"status": "acknowledged"}
            if message_type == "event":
                source_type = message.get("source_type")
                if source_type not in EVENT_SOURCE_TYPES:
                    raise ValueError("unknown source_type")
                payload = message.get("payload")
                if not isinstance(payload, dict):
                    raise ValueError("payload must be an object")
                string_payload: dict[str, str] = {}
                for key, value in payload.items():
                    if isinstance(value, (dict, list)):
                        string_payload[str(key)] = json.dumps(value)
                    else:
                        string_payload[str(key)] = "" if value is None else str(value)
                string_payload["session_id"] = runtime.session_id
                event = EVENT_CLASSES[source_type].from_redis_dict(string_payload)
                stream_name = f"lecturelens:{runtime.session_id}:stream:{source_type}"
                await self.redis_bus.publish_resilient(stream_name, event)
                if is_tracked_capture:
                    self._mark_capture_accepted(capture_id)
                return {"status": "acknowledged"}
            raise ValueError("unknown message type")
        finally:
            if is_tracked_capture and in_flight_event is not None:
                # Only release the marker that this handler installed. The
                # finally block used to pop whatever event happened to be in
                # the dictionary, which released another holder's waiters.
                if self.in_flight_captures.get(capture_id) is in_flight_event:
                    self.in_flight_captures.pop(capture_id, None)
                    in_flight_event.set()

    async def check_session_exists(self, session_id: str) -> bool:
        if session_id in self.sessions:
            return True
        session_dir = self.settings.get_session_output_dir(session_id)
        if session_dir.exists():
            return True
        if self.storage is not None:
            if hasattr(self.storage, "session_exists"):
                try:
                    if await self.storage.session_exists(session_id):
                        return True
                except Exception as error:
                    # The storage cannot answer, so the session cannot be
                    # ruled out either. The other branch below does the
                    # same, and a storage that is down used to shut every
                    # student out of their session.
                    self._guards.saw("the check of a session in the storage", error)
                    return True
            elif hasattr(self.storage, "get_general_notes_since"):
                # The fallback has to verify the session instead of
                # authorising every session id: returning True unconditionally
                # let any authenticated client attach to any session.
                try:
                    notes = await self.storage.get_general_notes_since(session_id, 0)
                    return len(notes) > 0
                except Exception as error:
                    # The storage cannot answer, so the session cannot be
                    # ruled out either.
                    self._guards.saw("the check of a session in the storage", error)
                    return True
        return False

    async def _subscriber_writer(self, websocket: Any, queue: asyncio.Queue) -> None:
        try:
            while True:
                payload = await queue.get()
                if payload is None:
                    break
                await self.send_json(websocket, payload)
                queue.task_done()
        except Exception:
            pass

    async def send_json(self, websocket: Any, payload: dict[str, Any]) -> None:
        await websocket.send(json.dumps(payload, default=str))

    async def handle_connection(self, websocket: Any) -> None:
        runtime: Optional[SessionRuntime] = None
        limiter = MessageRateLimiter(self.settings.gateway_messages_per_second)
        session_id: Optional[str] = None
        writer_task: Optional[asyncio.Task] = None
        client_queue: Optional[asyncio.Queue] = None
        is_subscriber = False
        try:
            try:
                self.check_origin(websocket)
            except PermissionError as error:
                await self.send_json(websocket, {"status": "unauthorized", "error": str(error)})
                await websocket.close()
                return
            first_frame = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            try:
                hello_message = json.loads(first_frame)
                if not isinstance(hello_message, dict):
                    raise PermissionError("malformed hello")
                session_id = self.authenticate(hello_message)
                is_subscriber = (
                    hello_message.get("role") in ("subscriber", "student")
                    or hello_message.get("subscribe") is True
                )
                if is_subscriber:
                    session_found = await self.check_session_exists(session_id)
                    if not session_found:
                        await self.send_json(websocket, {"status": "unauthorized", "error": "session does not exist"})
                        await websocket.close()
                        return
                else:
                    runtime = self.get_or_create_session(session_id)
                    runtime.connection_count += 1
            except (ValueError, PermissionError) as error:
                await self.send_json(websocket, {"status": "unauthorized", "error": str(error)})
                await websocket.close()
                return

            await self.send_json(websocket, {"status": "authenticated", "session_id": session_id})

            if is_subscriber:
                if session_id in self.sessions:
                    self.sessions[session_id].last_active = time.monotonic()
                client_queue = asyncio.Queue(maxsize=200)
                writer_task = asyncio.create_task(self._subscriber_writer(websocket, client_queue))

                if session_id not in self.catching_up:
                    self.catching_up[session_id] = {}
                pending_live: list[dict[str, Any]] = []
                self.catching_up[session_id][websocket] = pending_live

                if session_id not in self.fanout_tasks or self.fanout_tasks[session_id].done():
                    self.fanout_tasks[session_id] = asyncio.create_task(self.run_notes_fanout(session_id))

                last_seq = int(hello_message.get("last_sequence_number", 0))
                highest_seq = last_seq
                if last_seq >= 0:
                    missed = await self._replay_notes(session_id, last_seq)
                    for note in missed:
                        note_seq = int(note.get("sequence_number", 0))
                        if note_seq > highest_seq:
                            highest_seq = note_seq
                        await client_queue.put({"type": "note_segment", "payload": note})

                for live_note in pending_live:
                    note_seq = int(live_note.get("sequence_number", 0))
                    if note_seq > highest_seq:
                        highest_seq = note_seq
                        await client_queue.put({"type": "note_segment", "payload": live_note})

                # The high water mark used to be computed and thrown away, so
                # a client could never tell a temporary gap from a permanent
                # one.
                await client_queue.put({
                    "type": "session_state",
                    "highest_sequence_number": highest_seq
                })

                if session_id in self.catching_up and websocket in self.catching_up[session_id]:
                    del self.catching_up[session_id][websocket]

                if session_id not in self.subscribers:
                    self.subscribers[session_id] = {}
                if isinstance(self.subscribers[session_id], dict):
                    self.subscribers[session_id][websocket] = client_queue
                elif isinstance(self.subscribers[session_id], set):
                    self.subscribers[session_id].add(websocket)

                if session_id not in self.subscriber_writers:
                    self.subscriber_writers[session_id] = {}
                self.subscriber_writers[session_id][websocket] = writer_task

            async for raw_frame in websocket:
                if not limiter.allow():
                    await self.send_json(websocket, {"status": "error", "error": "rate limited"})
                    continue
                try:
                    message = json.loads(raw_frame)
                    if not isinstance(message, dict):
                        raise ValueError("message must be an object")

                    if is_subscriber:
                        msg_type = message.get("type")
                        if msg_type in ("audio", "screen", "event"):
                            await self.send_json(websocket, {"status": "error", "error": "subscribers are read-only"})
                            continue
                        if msg_type == "ping":
                            await self.send_json(websocket, {"type": "pong"})
                            continue
                        if msg_type == "request_replay":
                            req_last_seq = int(message.get("last_sequence_number", 0))
                            if client_queue is not None:
                                missed = await self._replay_notes(session_id, req_last_seq)
                                for note in missed:
                                    await client_queue.put({"type": "note_segment", "payload": note})
                            await self.send_json(websocket, {"status": "acknowledged"})
                            continue
                        await self.send_json(websocket, {"status": "error", "error": "unsupported subscriber message"})
                        continue

                    reply = await self.handle_message(runtime, message)
                except (ValueError, KeyError) as error:
                    reply = {"status": "error", "error": str(error)}
                except Exception as error:
                    logger.error("gateway processing error: %s", error)
                    reply = {"status": "error", "error": "processing failed"}
                await self.send_json(websocket, reply)
        except asyncio.TimeoutError:
            logger.warning("connection closed: no hello within timeout")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.info("connection ended: %s", error)
        finally:
            if session_id and session_id in self.catching_up and websocket in self.catching_up[session_id]:
                del self.catching_up[session_id][websocket]
            if session_id and session_id in self.subscribers:
                if isinstance(self.subscribers[session_id], dict):
                    self.subscribers[session_id].pop(websocket, None)
                elif isinstance(self.subscribers[session_id], set):
                    self.subscribers[session_id].discard(websocket)
            if session_id and session_id in self.subscriber_writers:
                self.subscriber_writers[session_id].pop(websocket, None)
            if client_queue is not None:
                try:
                    await asyncio.wait_for(client_queue.join(), timeout=0.5)
                except Exception:
                    pass
            if writer_task is not None and not writer_task.done():
                writer_task.cancel()
            if runtime is not None:
                runtime.connection_count -= 1
                runtime.last_active = time.monotonic()

    def handle_http_request(self, connection: Any, request: Any) -> Optional[Any]:
        try:
            from websockets.http11 import Response
            from websockets.datastructures import Headers
        except ImportError:
            return None

        path_with_query = getattr(request, "path", "") or ""
        parsed_url = urllib.parse.urlparse(path_with_query)
        path = parsed_url.path

        if not path.startswith("/api/assets/"):
            return None

        if self.api_key:
            auth_header = ""
            req_headers = getattr(request, "headers", None)
            if req_headers is not None:
                auth_header = req_headers.get("Authorization", "") or req_headers.get("x-api-key", "")
            query_params = urllib.parse.parse_qs(parsed_url.query)
            token = ""
            if auth_header.startswith("Bearer "):
                token = auth_header[7:].strip()
            elif auth_header:
                token = auth_header.strip()
            elif "key" in query_params:
                token = query_params["key"][0].strip()
            elif "token" in query_params:
                token = query_params["token"][0].strip()

            try:
                # An absent token is not a correct token. Rewriting the
                # condition as "not token or compare" gave an empty token
                # the same answer as the right key.
                token_ok = bool(token) and secrets.compare_digest(token, self.api_key)
            except TypeError:
                # compare_digest cannot look at non-ASCII strings. A token
                # with such characters can never equal the generated key,
                # so it is simply not accepted instead of raising.
                token_ok = False
            if not token_ok:
                return Response(
                    401,
                    "Unauthorized",
                    Headers([("Content-Type", "text/plain")]),
                    b"unauthorized"
                )

        parts = path.strip("/").split("/")
        if len(parts) < 4:
            return Response(
                400,
                "Bad Request",
                Headers([("Content-Type", "text/plain")]),
                b"invalid asset path"
            )

        session_id = parts[2]
        asset_id = parts[3]
        if not re.match(r"^[a-zA-Z0-9_\-\.]+$", session_id) or not re.match(r"^[a-zA-Z0-9_\-\.]+$", asset_id) or ".." in session_id or ".." in asset_id:
            return Response(
                400,
                "Bad Request",
                Headers([("Content-Type", "text/plain")]),
                b"invalid path parameters"
            )

        session_dir = self.settings.get_session_output_dir(session_id)
        assets_dir = (session_dir / "assets").resolve()
        target_path = (assets_dir / asset_id).resolve()

        # The containment check needs the separator, otherwise
        # "assets_evil" starts with "assets" and passes it.
        assets_prefix = str(assets_dir).rstrip("\\/") + ("\\\\" if os.name == "nt" else "/")
        if not str(target_path).startswith(assets_prefix) or not target_path.is_file():
            if not asset_id.endswith(".png"):
                alt_path = (assets_dir / f"{asset_id}.png").resolve()
                if str(alt_path).startswith(assets_prefix) and alt_path.is_file():
                    target_path = alt_path

        if not target_path.is_file():
            return Response(
                404,
                "Not Found",
                Headers([("Content-Type", "text/plain")]),
                b"asset not found"
            )

        try:
            body = target_path.read_bytes()
            content_type = "image/png"
            if target_path.suffix.lower() in (".jpg", ".jpeg"):
                content_type = "image/jpeg"
            return Response(
                200,
                "OK",
                Headers([("Content-Type", content_type), ("Content-Length", str(len(body)))]),
                body
            )
        except Exception as read_err:
            logger.warning("failed to serve asset %s: %s", target_path, read_err)
            return Response(
                500,
                "Internal Server Error",
                Headers([("Content-Type", "text/plain")]),
                b"error reading asset"
            )

    async def start(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        try:
            from websockets.asyncio.server import serve
        except ImportError:
            from websockets import serve
        bind_host = host if host is not None else self.settings.gateway_host
        bind_port = port if port is not None else self.settings.gateway_port
        # The key that is actually in effect decides whether a non-local
        # bind is safe: when none was configured and the gateway generated
        # an ephemeral one in __init__, the setting itself is still empty
        # and re-checking with it raised for a bind that was accepted
        # before.
        validate_gateway_binding(bind_host, self.api_key if self.api_key else None)
        self.server = await serve(
            self.handle_connection,
            bind_host,
            bind_port,
            max_size=self.settings.gateway_max_message_bytes,
            process_request=self.handle_http_request
        )
        logger.info("gateway listening on %s:%d", bind_host, bind_port)

    async def _replay_notes(self, session_id: str, after_seq: int = 0) -> list[dict[str, Any]]:
        notes_by_seq: dict[int, dict[str, Any]] = {}
        if self.storage is not None:
            try:
                pg_notes = await self.storage.get_general_notes_since(session_id, after_seq)
                for note in pg_notes:
                    seq = int(note.get("sequence_number", 0))
                    if seq > after_seq:
                        notes_by_seq[seq] = note
            except Exception as replay_err:
                logger.warning("postgres replay failed for %s: %s; falling back to redis", session_id, replay_err)

        try:
            redis_notes = await self._replay_notes_from_redis(session_id, after_seq)
            for note in redis_notes:
                seq = int(note.get("sequence_number", 0))
                if seq > after_seq and seq not in notes_by_seq:
                    notes_by_seq[seq] = note
        except Exception as redis_err:
            logger.warning("redis replay failed for %s: %s", session_id, redis_err)

        return [notes_by_seq[s] for s in sorted(notes_by_seq.keys())]

    async def _replay_notes_from_redis(self, session_id: str, after_seq: int = 0) -> list[dict[str, Any]]:
        stream_name = f"lecturelens:{session_id}:stream:notes"
        scan_limit = int(self.settings.gateway_replay_scan_limit)
        try:
            # The scan is bounded: the notes stream can hold up to
            # stream_max_length entries and reading all of them on every
            # replay was a client triggerable memory spike that stalled the
            # event loop. The newest entries are scanned first because the
            # notes are appended in sequence order.
            raw_entries = await self.redis_bus.client.xrevrange(
                stream_name, "+", "-", count=scan_limit
            )
            raw_entries = list(reversed(raw_entries))
        except Exception as err:
            logger.warning("xrevrange on %s failed: %s", stream_name, err)
            return []
        by_seq: dict[int, dict[str, Any]] = {}
        for _, data in raw_entries:
            try:
                segment = GeneralNoteSegment.from_redis_dict(data)
                seq = segment.sequence_number
                if seq > after_seq:
                    by_seq[seq] = segment.model_dump()
            except Exception:
                pass
        notes = [by_seq[k] for k in sorted(by_seq.keys())]
        return notes

    async def serve_forever(self, should_stop: Optional[Callable[[], bool]] = None) -> None:
        while True:
            if should_stop is not None and should_stop():
                break
            await asyncio.sleep(0.5)

    async def stop(self) -> None:
        for task in self.fanout_tasks.values():
            if not task.done():
                task.cancel()
        for session_writers in self.subscriber_writers.values():
            for wtask in session_writers.values():
                if not wtask.done():
                    wtask.cancel()
        for session_id in list(self.fanout_tasks.keys()):
            stream_name = f"lecturelens:{session_id}:stream:notes"
            group_name = f"group:gateway_{session_id}_{self.instance_id}"
            try:
                await self.redis_bus.client.xgroup_destroy(stream_name, group_name)
            except Exception:
                pass
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None


async def main() -> None:
    settings = Settings()
    configure_pools(settings.thread_pool_max_workers, settings.process_pool_max_workers)
    namespace = parse_runtime_arguments()
    session_id = namespace.session or settings.session_name or "default_session"
    session_dir = settings.get_session_output_dir(session_id)
    control = AgentControl(session_dir, "gateway")
    control.write_heartbeat("live")
    heartbeat_task = asyncio.create_task(control.heartbeat_loop())
    redis_bus = create_redis_bus(settings, component_name="gateway")
    router = None
    try:
        from lecturelens.router.llm_router import build_router_for_task, load_router_config
        router_config = load_router_config(settings.router_config_path)
        router = build_router_for_task(
            "screen_ocr", redis_bus.client, router_config,
            settings.ollama_url, settings.llm_mode, settings.llm_timeout_seconds,
            settings.local_model
        )
    except Exception as error:
        logger.warning("gateway vision router unavailable: %s", error)

    storage = None
    if settings.postgres_url:
        try:
            storage = create_postgres_storage(settings)
            await storage.init_schema()
        except Exception as error:
            logger.warning("gateway postgres storage initialization failed: %s", error)
            if storage is not None:
                # The connection was opened by init_schema, so it has to be
                # closed again. Dropping the reference leaked the socket for
                # the lifetime of the gateway process.
                try:
                    await storage.close()
                except Exception:
                    pass
            storage = None

    gateway = GatewayServer(settings, redis_bus, router, storage=storage)
    try:
        write_gateway_api_key(session_dir, gateway.api_key)
        await gateway.start()
        await gateway.serve_forever(should_stop=control.stop_requested)
    finally:
        control.write_heartbeat("stopping")
        heartbeat_task.cancel()
        await gateway.stop()
        if storage is not None:
            await storage.close()
        await redis_bus.close()
        shutdown_pools()
        control.write_heartbeat("stopped")


if __name__ == "__main__":
    execute_agent_main(main)
