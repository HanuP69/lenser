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

    async def ensure_consumer_group(self, stream_name: str, group_name: str, start_id: str = "0") -> None:
        pass

    async def consume(self, *args: Any, **kwargs: Any) -> None:
        pass


class FakeWebSocket:
    def __init__(self, frames: list[str]) -> None:
        self.incoming = list(frames)
        self.sent: list[dict] = []
        self.closed = False
        # Mirrors the handshake header a browser would send. None means a
        # non-browser client, which sends no Origin at all.
        self.origin: Optional[str] = None

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
        settings.gateway_require_api_key = False
        with self.assertRaises(GatewayConfigurationError):
            GatewayServer(settings, self.bus)

    def test_gateway_generates_a_key_when_none_is_configured(self) -> None:
        """The gateway used to accept every connection when no API key was
        configured, which let any local process and any web page read the
        notes and inject events."""
        settings = Settings()
        settings.output_dir = self.settings.output_dir
        settings.gateway_host = "127.0.0.1"
        settings.gateway_api_key = None
        gateway = GatewayServer(settings, self.bus)
        self.assertTrue(gateway.api_key)
        self.assertGreaterEqual(len(gateway.api_key), 32)

    async def test_handshake_from_an_unknown_origin_is_rejected(self) -> None:
        """Browsers do not apply the same-origin policy to WebSockets, so a
        page on any site could open ws://127.0.0.1:8765."""
        socket = FakeWebSocket([hello(api_key="secret")])
        socket.origin = "https://evil.example"
        await self.gateway.handle_connection(socket)
        self.assertEqual(socket.sent[0]["status"], "unauthorized")
        self.assertTrue(socket.closed)

    async def test_handshake_without_an_origin_is_allowed(self) -> None:
        socket = FakeWebSocket([hello(api_key="secret")])
        socket.origin = None
        await self.gateway.handle_connection(socket)
        self.assertEqual(socket.sent[0]["status"], "authenticated")

    async def test_handshake_from_an_allowed_origin_is_accepted(self) -> None:
        self.settings.gateway_allowed_origins = ["https://dashboard.example"]
        gateway = GatewayServer(self.settings, self.bus)
        socket = FakeWebSocket([hello(api_key="secret")])
        socket.origin = "https://dashboard.example"
        await gateway.handle_connection(socket)
        self.assertEqual(socket.sent[0]["status"], "authenticated")

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

    async def test_release_session_cleans_up_fanout_tasks(self) -> None:
        self.gateway.get_or_create_session("cleanup_room")
        dummy_task = asyncio.create_task(asyncio.sleep(10.0))
        self.gateway.fanout_tasks["cleanup_room"] = dummy_task
        self.gateway.subscribers["cleanup_room"] = set()
        self.gateway.release_session("cleanup_room")
        await asyncio.sleep(0)
        self.assertNotIn("cleanup_room", self.gateway.sessions)
        self.assertNotIn("cleanup_room", self.gateway.fanout_tasks)
        self.assertNotIn("cleanup_room", self.gateway.subscribers)
        self.assertTrue(dummy_task.cancelled() or dummy_task.done())

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


    async def test_reconnect_race_buffers_live_notes_and_delivers_in_order(self) -> None:
        class DelayNotesStorage:
            def __init__(self, gateway_ref: Any) -> None:
                self.gateway_ref = gateway_ref

            async def get_general_notes_since(self, session_id: str, after_seq: int) -> list[dict]:
                await self.gateway_ref.broadcast_note_segment(
                    session_id, {"sequence_number": 4, "content": "Live Note 4"}
                )
                await asyncio.sleep(0.02)
                return [
                    {"sequence_number": 2, "heading": "Part 2", "content": "Text 2"},
                    {"sequence_number": 3, "heading": "Part 3", "content": "Text 3"}
                ]

        self.gateway.storage = DelayNotesStorage(self.gateway)
        sub_hello = json.dumps({
            "type": "hello",
            "api_key": "secret",
            "session_id": "room1",
            "role": "subscriber",
            "last_sequence_number": 1
        })
        socket = FakeWebSocket([sub_hello])
        await self.gateway.handle_connection(socket)

        note_payloads = [msg["payload"] for msg in socket.sent if msg.get("type") == "note_segment"]
        self.assertEqual(len(note_payloads), 3)
        self.assertEqual(note_payloads[0]["sequence_number"], 2)
        self.assertEqual(note_payloads[1]["sequence_number"], 3)
        self.assertEqual(note_payloads[2]["sequence_number"], 4)

    async def test_subscribers_are_read_only_and_cannot_push_capture_events(self) -> None:
        class DummyStorage:
            async def get_general_notes_since(self, session_id: str, after_seq: int) -> list[dict]:
                # The session has to exist, otherwise the subscriber is
                # rejected before the read-only check is reached.
                return [{
                    "session_id": session_id,
                    "sequence_number": 1,
                    "timestamp_epoch": 100.0,
                    "start_time": "00:01:00",
                    "end_time": "00:02:00",
                    "heading": "Existing note",
                    "content": "Existing content",
                    "key_points": [],
                    "concept_tags": []
                }]

        self.gateway.storage = DummyStorage()
        sub_hello = json.dumps({
            "type": "hello",
            "api_key": "secret",
            "session_id": "room1",
            "role": "subscriber"
        })
        push_audio = json.dumps({"type": "audio", "wav_b64": base64.b64encode(b"dummy").decode()})
        socket = FakeWebSocket([sub_hello, push_audio])
        await self.gateway.handle_connection(socket)

        replies = socket.sent
        self.assertTrue(any(r.get("error") == "subscribers are read-only" for r in replies))

    async def test_subscriber_rejected_if_session_does_not_exist(self) -> None:
        self.gateway.storage = None
        sub_hello = json.dumps({
            "type": "hello",
            "api_key": "secret",
            "session_id": "nonexistent_room_999",
            "role": "subscriber"
        })
        socket = FakeWebSocket([sub_hello])
        await self.gateway.handle_connection(socket)

        self.assertEqual(socket.sent[0]["status"], "unauthorized")
        self.assertEqual(socket.sent[0]["error"], "session does not exist")
        self.assertNotIn("nonexistent_room_999", self.gateway.sessions)

    async def test_subscriber_is_not_authorized_for_an_unknown_session(self) -> None:
        """A storage without session_exists used to authorise every session
        id, so any authenticated client could read any session's notes."""
        class EmptyStorage:
            async def get_general_notes_since(self, session_id: str, after_seq: int) -> list[dict]:
                return []

        self.gateway.storage = EmptyStorage()
        sub_hello = json.dumps({
            "type": "hello",
            "api_key": "secret",
            "session_id": "room_without_notes",
            "role": "subscriber"
        })
        socket = FakeWebSocket([sub_hello])
        await self.gateway.handle_connection(socket)

        self.assertEqual(socket.sent[0]["status"], "unauthorized")
        self.assertEqual(socket.sent[0]["error"], "session does not exist")

    def test_two_gateway_fanout_distinct_groups(self) -> None:
        gw1 = GatewayServer(self.settings, self.bus, instance_id="gw_alpha")
        gw2 = GatewayServer(self.settings, self.bus, instance_id="gw_beta")
        self.assertNotEqual(gw1.instance_id, gw2.instance_id)

        group1 = f"group:gateway_room1_{gw1.instance_id}"
        group2 = f"group:gateway_room1_{gw2.instance_id}"
        self.assertNotEqual(group1, group2)

    async def test_datetime_serialization_in_replay(self) -> None:
        from datetime import datetime, timezone
        now_dt = datetime.now(timezone.utc)

        class DateTimeStorage:
            async def get_general_notes_since(self, session_id: str, after_seq: int) -> list[dict]:
                return [{
                    "sequence_number": 2,
                    "heading": "Part 2",
                    "content": "Text 2",
                    "created_at": now_dt
                }]

        self.gateway.storage = DateTimeStorage()
        sub_hello = json.dumps({
            "type": "hello",
            "api_key": "secret",
            "session_id": "room1",
            "role": "subscriber",
            "last_sequence_number": 1
        })
        socket = FakeWebSocket([sub_hello])
        await self.gateway.handle_connection(socket)

        note_payloads = [msg["payload"] for msg in socket.sent if msg.get("type") == "note_segment"]
        self.assertEqual(len(note_payloads), 1)
        self.assertEqual(note_payloads[0]["sequence_number"], 2)

    async def test_subscriber_replay_falls_back_to_redis_when_postgres_fails(self) -> None:
        class FailingPostgresStorage:
            async def get_general_notes_since(self, session_id: str, after_seq: int) -> list[dict[str, Any]]:
                raise RuntimeError("Postgres connection dropped")

        self.gateway.storage = FailingPostgresStorage()
        redis_notes = [{
            "session_id": "room1",
            "sequence_number": 2,
            "timestamp_epoch": 100.0,
            "start_time": "00:01:00",
            "end_time": "00:02:00",
            "heading": "From Redis",
            "content": "Redis note content",
            "key_points": [],
            "concept_tags": []
        }]
        self.gateway._replay_notes_from_redis = mock.AsyncMock(return_value=redis_notes)

        sub_hello = json.dumps({
            "type": "hello",
            "api_key": "secret",
            "session_id": "room1",
            "role": "subscriber",
            "last_sequence_number": 1
        })
        socket = FakeWebSocket([sub_hello])
        await self.gateway.handle_connection(socket)

        note_payloads = [msg["payload"] for msg in socket.sent if msg.get("type") == "note_segment"]
        self.assertEqual(len(note_payloads), 1)
        self.assertEqual(note_payloads[0]["heading"], "From Redis")
        self.assertEqual(note_payloads[0]["sequence_number"], 2)

    async def test_subscriber_replay_merges_postgres_and_uncommitted_redis_notes(self) -> None:
        class PartialPostgresStorage:
            async def get_general_notes_since(self, session_id: str, after_seq: int) -> list[dict[str, Any]]:
                return [{
                    "session_id": "room1",
                    "sequence_number": 2,
                    "timestamp_epoch": 100.0,
                    "start_time": "00:01:00",
                    "end_time": "00:02:00",
                    "heading": "From Postgres",
                    "content": "Postgres note content",
                    "key_points": [],
                    "concept_tags": []
                }]

        self.gateway.storage = PartialPostgresStorage()
        redis_notes = [
            {
                "session_id": "room1",
                "sequence_number": 2,
                "timestamp_epoch": 100.0,
                "start_time": "00:01:00",
                "end_time": "00:02:00",
                "heading": "From Postgres",
                "content": "Postgres note content",
                "key_points": [],
                "concept_tags": []
            },
            {
                "session_id": "room1",
                "sequence_number": 3,
                "timestamp_epoch": 110.0,
                "start_time": "00:02:00",
                "end_time": "00:03:00",
                "heading": "From Redis New",
                "content": "Unpersisted Redis note content",
                "key_points": [],
                "concept_tags": []
            }
        ]
        self.gateway._replay_notes_from_redis = mock.AsyncMock(return_value=redis_notes)

        sub_hello = json.dumps({
            "type": "hello",
            "api_key": "secret",
            "session_id": "room1",
            "role": "subscriber",
            "last_sequence_number": 1
        })
        socket = FakeWebSocket([sub_hello])
        await self.gateway.handle_connection(socket)

        note_payloads = [msg["payload"] for msg in socket.sent if msg.get("type") == "note_segment"]
        self.assertEqual(len(note_payloads), 2)
        self.assertEqual(note_payloads[0]["sequence_number"], 2)
        self.assertEqual(note_payloads[1]["sequence_number"], 3)
        self.assertEqual(note_payloads[1]["heading"], "From Redis New")

    def test_active_subscribers_prevent_session_eviction(self) -> None:
        runtime = self.gateway.get_or_create_session("active_sub_session")
        runtime.connection_count = 0
        self.gateway.subscribers["active_sub_session"] = {mock.MagicMock(): asyncio.Queue()}
        evicted = self.gateway.evict_idle_sessions(1)
        self.assertEqual(evicted, 0)
        self.assertIn("active_sub_session", self.gateway.sessions)

    def test_idle_session_evicted_when_no_subscribers(self) -> None:
        runtime = self.gateway.get_or_create_session("idle_session")
        runtime.connection_count = 0
        self.gateway.subscribers.pop("idle_session", None)
        evicted = self.gateway.evict_idle_sessions(1)
        self.assertEqual(evicted, 1)
        self.assertNotIn("idle_session", self.gateway.sessions)


class TestRemoteClientShutdown(unittest.IsolatedAsyncioTestCase):
    async def test_remote_client_terminates_when_stop_event_set(self) -> None:
        from lecturelens.client.remote_client import run_remote_client

        class FakeSocket:
            async def send(self, data: str) -> None:
                pass

            async def recv(self) -> str:
                return json.dumps({"status": "authenticated"})

        class FakeConnectContext:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            async def __aenter__(self) -> FakeSocket:
                return FakeSocket()

            async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
                pass

        stop_event = asyncio.Event()
        settings = Settings()

        async def stop_after_delay() -> None:
            await asyncio.sleep(0.05)
            stop_event.set()

        asyncio.create_task(stop_after_delay())

        async def mock_stream(s: Any, setts: Any, stop_ev: Any) -> None:
            await stop_ev.wait()

        with mock.patch("websockets.connect", side_effect=lambda *a, **kw: FakeConnectContext(*a, **kw)):
            with mock.patch("lecturelens.client.remote_client.stream_audio", side_effect=mock_stream):
                await asyncio.wait_for(
                    run_remote_client(
                        settings,
                        "ws://localhost:8765",
                        api_key=None,
                        session_id="test_remote_sess",
                        use_audio=True,
                        use_screen=False,
                        stop_event=stop_event
                    ),
                    timeout=2.0
                )

    async def test_remote_client_terminates_immediately_on_cancellation(self) -> None:
        from lecturelens.client.remote_client import run_remote_client

        class HangingSocket:
            async def send(self, data: str) -> None:
                pass

            async def recv(self) -> str:
                await asyncio.sleep(60.0)
                return "{}"

        class FakeConnectContext:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.exited = False

            async def __aenter__(self) -> HangingSocket:
                return HangingSocket()

            async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
                self.exited = True

        settings = Settings()
        ctx_instance = FakeConnectContext()

        with mock.patch("websockets.connect", return_value=ctx_instance):
            task = asyncio.create_task(
                run_remote_client(
                    settings,
                    "ws://localhost:8765",
                    api_key=None,
                    session_id="test_cancel_sess",
                    use_audio=False,
                    use_screen=False,
                    stop_event=None
                )
            )
            await asyncio.sleep(0.05)
            task.cancel()
            start_t = time.monotonic()
            await asyncio.gather(task, return_exceptions=True)
            elapsed = time.monotonic() - start_t
            # The socket hangs for a minute, so a return in a few seconds
            # still shows that the cancellation was not waited for. The
            # budget has to be generous, because the event loop of a
            # machine that is busy with the other tests is slow.
            self.assertLess(elapsed, 5.0)
            self.assertTrue(ctx_instance.exited)


class TestRateLimiter(unittest.TestCase):
    def test_burst_then_refill(self) -> None:
        limiter = MessageRateLimiter(10.0, burst=3.0)
        allowed = []
        for _ in range(5):
            allowed.append(limiter.allow())
        self.assertEqual(allowed, [True, True, True, False, False])
        time.sleep(0.25)
        self.assertTrue(limiter.allow())


class TestRemoteClientSendAndWait(unittest.IsolatedAsyncioTestCase):
    async def test_send_and_wait_succeeds_on_acknowledged(self) -> None:
        from lecturelens.client.remote_client import send_and_wait
        mock_ws = mock.AsyncMock()
        mock_ws.recv.return_value = json.dumps({"status": "acknowledged"})
        await send_and_wait(mock_ws, '{"type": "audio"}')
        mock_ws.send.assert_awaited_once_with('{"type": "audio"}')

    async def test_send_and_wait_retries_transient_error_and_succeeds(self) -> None:
        from lecturelens.client.remote_client import send_and_wait
        mock_ws = mock.AsyncMock()
        mock_ws.recv.side_effect = [
            json.dumps({"status": "error", "error": "busy"}),
            json.dumps({"status": "acknowledged"})
        ]
        await send_and_wait(mock_ws, '{"type": "screen"}', max_retries=3, initial_backoff=0.01)
        self.assertEqual(mock_ws.send.await_count, 2)

    async def test_send_and_wait_raises_runtime_error_when_retries_exhausted(self) -> None:
        from lecturelens.client.remote_client import send_and_wait
        mock_ws = mock.AsyncMock()
        mock_ws.recv.return_value = json.dumps({"status": "error", "error": "outbox full"})
        with self.assertRaises(RuntimeError) as ctx:
            await send_and_wait(mock_ws, '{"type": "audio"}', max_retries=2, initial_backoff=0.01)
        self.assertIn("server rejected frame", str(ctx.exception))
        self.assertEqual(mock_ws.send.await_count, 2)

    async def test_failed_ingestion_remains_retryable_with_same_capture_id(self) -> None:
        settings = Settings()
        bus = FakeBus()
        server = GatewayServer(settings, bus)
        runtime = server.get_or_create_session("sess_retry")
        msg = {
            "type": "audio",
            "wav_b64": base64.b64encode(b"dummy_wav_content").decode("ascii"),
            "timestamp_epoch": time.time(),
            "capture_id": "cap_retry_123"
        }

        with mock.patch.object(runtime.audio_agent, "ingest_wav_bytes", side_effect=RuntimeError("transient db error")):
            with self.assertRaises(RuntimeError):
                await server.handle_message(runtime, msg)

        self.assertNotIn("cap_retry_123", server.accepted_capture_ids)

        with mock.patch.object(runtime.audio_agent, "ingest_wav_bytes", return_value=None) as mock_ingest:
            reply2 = await server.handle_message(runtime, msg)
            self.assertEqual(reply2.get("status"), "acknowledged")
            self.assertNotIn("duplicate", reply2)
            mock_ingest.assert_awaited_once()

        self.assertIn("cap_retry_123", server.accepted_capture_ids)

        with mock.patch.object(runtime.audio_agent, "ingest_wav_bytes") as mock_ingest2:
            reply3 = await server.handle_message(runtime, msg)
            self.assertEqual(reply3.get("status"), "acknowledged")
            self.assertTrue(reply3.get("duplicate"))
            mock_ingest2.assert_not_called()

    async def test_capture_spool_durability_and_drain(self) -> None:
        from lecturelens.client.remote_client import CaptureSpool, drain_spool
        with tempfile.TemporaryDirectory() as td:
            spool_file = Path(td) / "spool.db"
            spool = CaptureSpool(spool_file, max_size=10)
            await spool.append("c1", '{"capture_id": "c1", "type": "audio"}')
            await spool.append("c2", '{"capture_id": "c2", "type": "screen"}')
            self.assertEqual(spool.count(), 2)

            mock_ws = mock.AsyncMock()
            mock_ws.recv.return_value = json.dumps({"status": "acknowledged"})
            stop_evt = asyncio.Event()

            await drain_spool(spool, mock_ws, stop_evt)
            self.assertEqual(spool.count(), 0)
            self.assertEqual(mock_ws.send.await_count, 2)

    async def test_capture_spool_with_a_large_backlog(self) -> None:
        """A spool with more than 500 pending captures has to behave the same
        way in both policies."""
        from lecturelens.client.remote_client import CaptureSpool, CaptureSpoolFull
        with tempfile.TemporaryDirectory() as td:
            spool_file = Path(td) / "spool.db"

            # drop_oldest: the oldest captures are evicted, the newest survive.
            spool = CaptureSpool(
                spool_file, max_size=500, append_timeout_seconds=0.1, on_full="drop_oldest"
            )
            for index in range(520):
                await spool.append(f"c{index}", f"payload {index}")
            self.assertEqual(spool.count(), 500)
            self.assertGreater(spool.dropped_count, 0)
            pending = [capture_id for capture_id, _ in spool.get_pending(limit=1000)]
            self.assertEqual(len(pending), 500)
            self.assertNotIn("c0", pending)
            self.assertIn("c519", pending)

            # raise: nothing is evicted, the capture itself fails.
            spool_file2 = Path(td) / "spool2.db"
            strict = CaptureSpool(
                spool_file2, max_size=500, append_timeout_seconds=0.1, on_full="raise"
            )
            for index in range(500):
                await strict.append(f"c{index}", f"payload {index}")
            self.assertEqual(strict.count(), 500)
            with self.assertRaises(CaptureSpoolFull):
                await strict.append("overflow", "payload overflow")
            self.assertEqual(strict.count(), 500)
            self.assertEqual(strict.dropped_count, 0)
            pending = [capture_id for capture_id, _ in strict.get_pending(limit=1000)]
            self.assertEqual(pending[0], "c0")

    async def test_capture_spool_raise_policy_drops_nothing(self) -> None:
        """With the raise policy nothing is evicted, the capture fails
        instead."""
        from lecturelens.client.remote_client import CaptureSpool, CaptureSpoolFull
        with tempfile.TemporaryDirectory() as td:
            spool_file = Path(td) / "spool.db"
            spool = CaptureSpool(
                spool_file, max_size=2, append_timeout_seconds=0.1, on_full="raise"
            )
            await spool.append("c1", "one")
            await spool.append("c2", "two")

            with self.assertRaises(CaptureSpoolFull):
                await spool.append("c3", "three")

            # Nothing was evicted.
            self.assertEqual(spool.dropped_count, 0)
            self.assertEqual(spool.count(), 2)
            self.assertEqual(spool.get_pending(limit=10)[0][0], "c1")

    async def test_capture_spool_drop_policy_evicts_the_oldest(self) -> None:
        from lecturelens.client.remote_client import CaptureSpool
        with tempfile.TemporaryDirectory() as td:
            spool_file = Path(td) / "spool.db"
            spool = CaptureSpool(
                spool_file, max_size=2, append_timeout_seconds=0.1, on_full="drop_oldest"
            )
            await spool.append("c1", "one")
            await spool.append("c2", "two")
            await spool.append("c3", "three")

            self.assertEqual(spool.dropped_count, 1)
            pending = [capture_id for capture_id, _ in spool.get_pending(limit=10)]
            self.assertEqual(pending, ["c2", "c3"])

    async def test_capture_spool_backpressure_does_not_drop_when_space_frees(self) -> None:
        """The backpressure wait used to block a thread pool worker, so the
        acknowledgements that free the space could not run."""
        from lecturelens.client.remote_client import CaptureSpool
        with tempfile.TemporaryDirectory() as td:
            spool_file = Path(td) / "spool.db"
            spool = CaptureSpool(spool_file, max_size=2, append_timeout_seconds=5.0)
            await spool.append("c1", "one")
            await spool.append("c2", "two")

            async def free_space() -> None:
                await asyncio.sleep(0.2)
                spool.acknowledge("c1")

            freer = asyncio.create_task(free_space())
            await spool.append("c3", "three")
            await freer

            self.assertEqual(spool.dropped_count, 0)
            self.assertEqual(spool.count(), 2)

    async def test_send_and_wait_times_out_on_unresponsive_server(self) -> None:
        from lecturelens.client.remote_client import send_and_wait
        mock_ws = mock.AsyncMock()

        async def hung_recv():
            await asyncio.sleep(10.0)
            return json.dumps({"status": "acknowledged"})

        mock_ws.recv = hung_recv
        with self.assertRaises(Exception):
            await send_and_wait(mock_ws, '{"type": "audio"}', max_retries=1, timeout=0.05)


if __name__ == "__main__":
    unittest.main()
