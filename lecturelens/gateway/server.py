import asyncio
import base64
import binascii
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from lecturelens.agents.audio_agent import AudioAgent
from lecturelens.agents.camera_agent import CameraAgent
from lecturelens.agents.screen_agent import ScreenAgent
from lecturelens.config.settings import Settings
from lecturelens.core.agent_runtime import execute_agent_main
from lecturelens.core.clock import SessionClock
from lecturelens.core.concurrency import configure_pools, shutdown_pools
from lecturelens.core.logging_setup import get_logger
from lecturelens.core.redis_bus import RedisBus
from lecturelens.core.validation import is_valid_session_id
from lecturelens.schemas.events import BoardCaptured, SlideCaptured, TranscriptChunk

logger = get_logger("lecturelens.gateway")

LOOPBACK_HOSTS: tuple[str, ...] = ("127.0.0.1", "localhost", "::1")
EVENT_SOURCE_TYPES: tuple[str, ...] = ("transcript", "slide", "board")
EVENT_CLASSES: dict[str, Any] = {
    "transcript": TranscriptChunk,
    "slide": SlideCaptured,
    "board": BoardCaptured
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
    camera_agent: CameraAgent
    connection_count: int = 0


def validate_gateway_binding(host: str, api_key: Optional[str]) -> None:
    if host not in LOOPBACK_HOSTS and not api_key:
        raise GatewayConfigurationError(
            "gateway bound to a non-loopback address requires LECTURELENS_GATEWAY_API_KEY"
        )


def keys_match(provided_key: str, expected_key: str) -> bool:
    return hmac.compare_digest(provided_key.encode("utf-8"), expected_key.encode("utf-8"))


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
        llm_router: Any = None
    ) -> None:
        validate_gateway_binding(settings.gateway_host, settings.gateway_api_key)
        self.settings: Settings = settings
        self.redis_bus: RedisBus = redis_bus
        self.llm_router: Any = llm_router
        self.sessions: dict[str, SessionRuntime] = {}
        self.server: Any = None

    def get_or_create_session(self, session_id: str) -> SessionRuntime:
        existing = self.sessions.get(session_id)
        if existing is not None:
            return existing
        if len(self.sessions) >= self.settings.gateway_max_sessions:
            raise ValueError("session limit reached")
        session_dir = self.settings.create_session_directories(session_id)
        clock = SessionClock.load_or_create(session_dir / "clock.json")
        runtime = SessionRuntime(
            session_id=session_id,
            clock=clock,
            audio_agent=AudioAgent(self.settings, self.redis_bus, clock, session_id),
            screen_agent=ScreenAgent(self.settings, self.redis_bus, clock, session_id, self.llm_router),
            camera_agent=CameraAgent(self.settings, self.redis_bus, clock, session_id, self.llm_router)
        )
        self.sessions[session_id] = runtime
        return runtime

    def authenticate(self, hello_message: dict[str, Any]) -> str:
        if hello_message.get("type") != "hello":
            raise PermissionError("expected hello message")
        expected_key = self.settings.gateway_api_key
        if expected_key:
            provided_key = hello_message.get("api_key")
            if not isinstance(provided_key, str) or not keys_match(provided_key, expected_key):
                raise PermissionError("invalid api key")
        session_id = hello_message.get("session_id")
        if not isinstance(session_id, str) or not is_valid_session_id(session_id):
            raise PermissionError("invalid session id")
        return session_id

    async def handle_message(self, runtime: SessionRuntime, message: dict[str, Any]) -> dict[str, Any]:
        message_type = message.get("type")
        if message_type == "audio":
            wav_bytes = decode_base64_field(message, "wav_b64")
            await runtime.audio_agent.ingest_wav_bytes(wav_bytes)
            return {"status": "acknowledged"}
        if message_type == "screen":
            png_bytes = decode_base64_field(message, "png_b64")
            await runtime.screen_agent.ingest_png_bytes(png_bytes)
            return {"status": "acknowledged"}
        if message_type == "camera":
            jpeg_bytes = decode_base64_field(message, "jpeg_b64")
            await runtime.camera_agent.ingest_jpeg_bytes(jpeg_bytes)
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
            return {"status": "acknowledged"}
        raise ValueError("unknown message type")

    async def send_json(self, websocket: Any, payload: dict[str, Any]) -> None:
        await websocket.send(json.dumps(payload))

    async def handle_connection(self, websocket: Any) -> None:
        runtime: Optional[SessionRuntime] = None
        limiter = MessageRateLimiter(self.settings.gateway_messages_per_second)
        try:
            first_frame = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            try:
                hello_message = json.loads(first_frame)
                if not isinstance(hello_message, dict):
                    raise PermissionError("malformed hello")
                session_id = self.authenticate(hello_message)
                runtime = self.get_or_create_session(session_id)
            except (ValueError, PermissionError) as error:
                await self.send_json(websocket, {"status": "unauthorized", "error": str(error)})
                await websocket.close()
                return
            runtime.connection_count += 1
            await self.send_json(websocket, {"status": "authenticated", "session_id": session_id})

            async for raw_frame in websocket:
                if not limiter.allow():
                    await self.send_json(websocket, {"status": "error", "error": "rate limited"})
                    continue
                try:
                    message = json.loads(raw_frame)
                    if not isinstance(message, dict):
                        raise ValueError("message must be an object")
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
            if runtime is not None:
                runtime.connection_count -= 1

    async def start(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        try:
            from websockets.asyncio.server import serve
        except ImportError:
            from websockets import serve
        bind_host = host if host is not None else self.settings.gateway_host
        bind_port = port if port is not None else self.settings.gateway_port
        validate_gateway_binding(bind_host, self.settings.gateway_api_key)
        self.server = await serve(
            self.handle_connection,
            bind_host,
            bind_port,
            max_size=self.settings.gateway_max_message_bytes
        )
        logger.info("gateway listening on %s:%d", bind_host, bind_port)

    async def serve_forever(self) -> None:
        while True:
            await asyncio.sleep(1.0)

    async def stop(self) -> None:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None


async def main() -> None:
    settings = Settings()
    configure_pools(settings.thread_pool_max_workers, settings.process_pool_max_workers)
    redis_bus = RedisBus(settings.redis_url)
    router = None
    try:
        from lecturelens.router.llm_router import build_router_for_task, load_router_config
        router_config = load_router_config(settings.router_config_path)
        router = build_router_for_task(
            "board_ocr", redis_bus.client, router_config,
            settings.ollama_url, settings.llm_mode, settings.llm_timeout_seconds
        )
    except Exception as error:
        logger.warning("gateway vision router unavailable: %s", error)
    gateway = GatewayServer(settings, redis_bus, router)
    try:
        await gateway.start()
        await gateway.serve_forever()
    finally:
        await gateway.stop()
        await redis_bus.close()
        shutdown_pools()


if __name__ == "__main__":
    execute_agent_main(main)
