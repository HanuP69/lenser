import asyncio
import base64
import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Optional
from unittest import mock

from lecturelens.config.settings import Settings
from lecturelens.gateway import server as gateway_server
from lecturelens.gateway.server import (
    GatewayConfigurationError,
    GatewayServer,
    MessageRateLimiter,
    keys_match,
    validate_gateway_binding
)
from tests.support import pluck, pluck_optional


class FakeBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, Any]] = []

    async def publish_resilient(self, stream_name: str, event: Any) -> str:
        self.published.append((stream_name, event))
        return "1-1"


class FakeWebSocket:
    def __init__(self, frames: list[str]) -> None:
        self.incoming = list(frames)
        self.sent: list[dict] = []
        self.closed = False

    async def recv(self) -> str:
        if len(self.incoming) == 0:
            await asyncio.sleep(30)
        return self.incoming.pop(0)

    async def send(self, text: str) -> None:
        self.sent.append(json.loads(text))

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self) -> "FakeWebSocket":
        return self

    async def __anext__(self) -> str:
        if len(self.incoming) == 0:
            raise StopAsyncIteration
        return self.incoming.pop(0)


def hello(session: str = "room1", api_key: Optional[str] = "secret") -> str:
    return json.dumps({"type": "hello", "api_key": api_key, "session_id": session})


class GatewayTestBase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings = Settings()
        self.settings.output_dir = Path(self.temp_dir.name)
        self.settings.gateway_api_key = "secret"
        self.settings.gateway_messages_per_second = 1000.0
        self.bus = FakeBus()
        self.gateway = GatewayServer(self.settings, self.bus)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()


class TestGatewaySecurity(GatewayTestBase):
    def test_non_loopback_without_key_is_refused(self) -> None:
        with self.assertRaises(GatewayConfigurationError):
            validate_gateway_binding("0.0.0.0", None)
        validate_gateway_binding("0.0.0.0", "key")
        validate_gateway_binding("127.0.0.1", None)

    def test_constructor_enforces_binding_rule(self) -> None:
        settings = Settings()
        settings.output_dir = self.settings.output_dir
        settings.gateway_host = "0.0.0.0"
        settings.gateway_api_key = None
        with self.assertRaises(GatewayConfigurationError):
            GatewayServer(settings, self.bus)

    def test_constant_time_key_compare(self) -> None:
        self.assertTrue(keys_match("abc", "abc"))
        self.assertFalse(keys_match("abc", "abd"))
        self.assertFalse(keys_match("", "abc"))

    async def test_wrong_key_is_rejected_and_closed(self) -> None:
        socket = FakeWebSocket([hello(api_key="wrong")])
        await self.gateway.handle_connection(socket)
        self.assertEqual(socket.sent[0]["status"], "unauthorized")
        self.assertTrue(socket.closed)
        self.assertEqual(self.gateway.sessions, {})

    async def test_missing_hello_is_rejected(self) -> None:
        socket = FakeWebSocket([json.dumps({"type": "event"})])
        await self.gateway.handle_connection(socket)
        self.assertEqual(socket.sent[0]["status"], "unauthorized")

    async def test_non_json_hello_is_rejected(self) -> None:
        socket = FakeWebSocket(["not json at all"])
        await self.gateway.handle_connection(socket)
        self.assertEqual(socket.sent[0]["status"], "unauthorized")

    async def test_path_traversal_session_ids_are_rejected(self) -> None:
        for bad_session in ("../evil", "a/b", "", "x" * 100, ".."):
            socket = FakeWebSocket([hello(session=bad_session)])
            await self.gateway.handle_connection(socket)
            self.assertEqual(socket.sent[0]["status"], "unauthorized", bad_session)
        self.assertEqual(self.gateway.sessions, {})

    async def test_session_limit_is_enforced_while_sessions_are_connected(self) -> None:
        self.settings.gateway_max_sessions = 1
        busy_runtime = self.gateway.get_or_create_session("one")
        busy_runtime.connection_count = 1
        second = FakeWebSocket([hello(session="two")])
        await self.gateway.handle_connection(second)
        self.assertEqual(second.sent[0]["status"], "unauthorized")
        self.assertIn("one", self.gateway.sessions)

    async def test_idle_sessions_are_evicted_to_make_room(self) -> None:
        self.settings.gateway_max_sessions = 2
        for session_name in ("a", "b", "c", "d"):
            socket = FakeWebSocket([hello(session=session_name)])
            await self.gateway.handle_connection(socket)
            self.assertEqual(socket.sent[0]["status"], "authenticated", session_name)
        self.assertLessEqual(len(self.gateway.sessions), 2)
        self.assertIn("d", self.gateway.sessions)

    async def test_oldest_idle_session_is_evicted_first(self) -> None:
        self.settings.gateway_max_sessions = 2
        older = self.gateway.get_or_create_session("older")
        newer = self.gateway.get_or_create_session("newer")
        older.last_active = 1.0
        newer.last_active = 2.0
        self.gateway.get_or_create_session("third")
        self.assertNotIn("older", self.gateway.sessions)
        self.assertIn("newer", self.gateway.sessions)

    def test_capture_epoch_is_accepted_only_when_plausible(self) -> None:
        from lecturelens.core.clock import SessionClock
        from lecturelens.gateway.server import resolve_capture_epoch
        clock = SessionClock()
        now = clock.get_timestamp()
        self.assertAlmostEqual(resolve_capture_epoch({"timestamp_epoch": now - 2.0}, clock), now - 2.0, places=3)
        self.assertIsNone(resolve_capture_epoch({"timestamp_epoch": now + 3600.0}, clock))
        self.assertIsNone(resolve_capture_epoch({"timestamp_epoch": now - 86400.0}, clock))
        self.assertIsNone(resolve_capture_epoch({"timestamp_epoch": "soon"}, clock))
        self.assertIsNone(resolve_capture_epoch({"timestamp_epoch": True}, clock))
        self.assertIsNone(resolve_capture_epoch({}, clock))

    async def test_capture_epoch_does_not_reset_the_shared_session_clock(self) -> None:
        runtime = self.gateway.get_or_create_session("clockroom")
        before_reference = runtime.clock.reference_wall_time
        captured: list[object] = []

        async def fake_screen(data: bytes, capture_epoch: Any = None) -> None:
            captured.append(capture_epoch)

        runtime.screen_agent.ingest_png_bytes = fake_screen
        stamp = runtime.clock.get_timestamp() - 1.5
        message = {"type": "screen", "png_b64": base64.b64encode(b"png").decode(), "timestamp_epoch": stamp}
        await self.gateway.handle_message(runtime, message)
        self.assertEqual(runtime.clock.reference_wall_time, before_reference)
        self.assertEqual(captured, [stamp])

    async def test_silent_connection_times_out(self) -> None:
        socket = FakeWebSocket([])
        original_wait_for = asyncio.wait_for

        async def quick_wait_for(awaitable: Any, timeout: float) -> Any:
            return await original_wait_for(awaitable, 0.1)

        with mock.patch.object(gateway_server.asyncio, "wait_for", quick_wait_for):
            await self.gateway.handle_connection(socket)
        self.assertEqual(socket.sent, [])


class TestGatewayMessages(GatewayTestBase):
    async def test_event_is_forced_into_authenticated_session(self) -> None:
        payload = {
            "session_id": "someone_elses_session",
            "timestamp_epoch": 5.0,
            "start_seconds": 0.0,
            "end_seconds": 1.0,
            "text": "hello",
            "confidence": 0.9
        }
        socket = FakeWebSocket([
            hello(session="room1"),
            json.dumps({"type": "event", "source_type": "transcript", "payload": payload})
        ])
        await self.gateway.handle_connection(socket)
        self.assertEqual(socket.sent[0]["status"], "authenticated")
        self.assertEqual(socket.sent[1]["status"], "acknowledged")
        stream_name, event = self.bus.published[0]
        self.assertEqual(stream_name, "lecturelens:room1:stream:transcript")
        self.assertEqual(event.session_id, "room1")

    async def test_unknown_source_type_and_message_type_are_errors(self) -> None:
        socket = FakeWebSocket([
            hello(),
            json.dumps({"type": "event", "source_type": "../../x", "payload": {}}),
            json.dumps({"type": "teleport"}),
            json.dumps([1, 2, 3])
        ])
        await self.gateway.handle_connection(socket)
        statuses = pluck(socket.sent[1:], "status")
        self.assertEqual(statuses, ["error", "error", "error"])
        self.assertEqual(self.bus.published, [])

    async def test_malformed_event_payload_does_not_kill_connection(self) -> None:
        socket = FakeWebSocket([
            hello(),
            json.dumps({"type": "event", "source_type": "transcript", "payload": {"text": "missing fields"}}),
            json.dumps({"type": "event", "source_type": "transcript", "payload": "nope"})
        ])
        await self.gateway.handle_connection(socket)
        self.assertEqual(pluck(socket.sent[1:], "status"), ["error", "error"])

    async def test_invalid_base64_is_rejected(self) -> None:
        socket = FakeWebSocket([hello(), json.dumps({"type": "audio", "wav_b64": "***not base64***"})])
        await self.gateway.handle_connection(socket)
        self.assertEqual(socket.sent[1]["status"], "error")

    async def test_audio_screen_frames_are_processed_by_session_agents(self) -> None:
        socket = FakeWebSocket([
            hello(),
            json.dumps({"type": "audio", "wav_b64": base64.b64encode(b"wav").decode()}),
            json.dumps({"type": "screen", "png_b64": base64.b64encode(b"png").decode()})
        ])
        calls: list[tuple[str, bytes]] = []
        runtime = self.gateway.get_or_create_session("room1")

        async def fake_audio(data: bytes, capture_epoch: Any = None) -> None:
            calls.append(("audio", data))

        async def fake_screen(data: bytes, capture_epoch: Any = None) -> None:
            calls.append(("screen", data))

        runtime.audio_agent.ingest_wav_bytes = fake_audio
        runtime.screen_agent.ingest_png_bytes = fake_screen
        await self.gateway.handle_connection(socket)
        self.assertEqual(calls, [("audio", b"wav"), ("screen", b"png")])
        self.assertEqual(pluck(socket.sent[1:], "status"), ["acknowledged"] * 2)

    async def test_processing_exception_returns_generic_error(self) -> None:
        runtime = self.gateway.get_or_create_session("room1")

        async def exploding(data: bytes) -> None:
            raise RuntimeError("secret internal detail")

        runtime.audio_agent.ingest_wav_bytes = exploding
        socket = FakeWebSocket([hello(), json.dumps({"type": "audio", "wav_b64": base64.b64encode(b"x").decode()})])
        await self.gateway.handle_connection(socket)
        self.assertEqual(socket.sent[1], {"status": "error", "error": "processing failed"})

    async def test_rate_limit_rejects_bursts(self) -> None:
        self.settings.gateway_messages_per_second = 1.0
        frames = [hello()]
        for _ in range(10):
            frames.append(json.dumps({"type": "teleport"}))
        socket = FakeWebSocket(frames)
        await self.gateway.handle_connection(socket)
        errors = pluck_optional(socket.sent[1:], "error")
        self.assertIn("rate limited", errors)

    async def test_connection_count_is_released(self) -> None:
        socket = FakeWebSocket([hello()])
        await self.gateway.handle_connection(socket)
        self.assertEqual(self.gateway.sessions["room1"].connection_count, 0)

    async def test_subscriber_receives_missed_notes_from_storage_replay(self) -> None:
        class FakeNotesStorage:
            async def get_general_notes_since(self, session_id: str, after_seq: int) -> list[dict]:
                return [
                    {"sequence_number": 2, "heading": "Part 2", "content": "Text 2"},
                    {"sequence_number": 3, "heading": "Part 3", "content": "Text 3"}
                ]

        self.gateway.storage = FakeNotesStorage()
        sub_hello = json.dumps({
            "type": "hello",
            "api_key": "secret",
            "session_id": "room1",
            "role": "subscriber",
            "last_sequence_number": 1
        })
        socket = FakeWebSocket([sub_hello])
        await self.gateway.handle_connection(socket)
        sent_types = [msg.get("type") for msg in socket.sent]
        self.assertIn("note_segment", sent_types)
        note_payloads = [msg["payload"] for msg in socket.sent if msg.get("type") == "note_segment"]
        self.assertEqual(len(note_payloads), 2)
        self.assertEqual(note_payloads[0]["sequence_number"], 2)
        self.assertEqual(note_payloads[1]["sequence_number"], 3)

    async def test_broadcast_note_segment_fans_out_to_subscribers(self) -> None:
        socket_a = FakeWebSocket([])
        socket_b = FakeWebSocket([])
        self.gateway.subscribers["room1"] = {socket_a, socket_b}
        await self.gateway.broadcast_note_segment("room1", {"sequence_number": 5, "content": "Live Note"})
        self.assertEqual(len(socket_a.sent), 1)
        self.assertEqual(len(socket_b.sent), 1)
        self.assertEqual(socket_a.sent[0]["payload"]["sequence_number"], 5)
        self.assertEqual(socket_b.sent[0]["payload"]["sequence_number"], 5)


class TestRateLimiter(unittest.TestCase):
    def test_burst_then_refill(self) -> None:
        limiter = MessageRateLimiter(10.0, burst=3.0)
        allowed = []
        for _ in range(5):
            allowed.append(limiter.allow())
        self.assertEqual(allowed, [True, True, True, False, False])
        time.sleep(0.25)
        self.assertTrue(limiter.allow())


if __name__ == "__main__":
    unittest.main()
