import asyncio
import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import redis.asyncio as redis_module

from lecturelens.config.settings import Settings
from lecturelens.core.redis_bus import RedisBus
from lecturelens.schemas.events import TranscriptChunk


class TestRedisBus(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.bus = RedisBus(Settings().redis_url)
        try:
            await self.bus.client.ping()
            self.redis_available = True
        except Exception:
            self.redis_available = False

    async def asyncTearDown(self) -> None:
        if self.redis_available:
            await self.bus.client.delete("lecturelens:test_stream:transcript")
            await self.bus.close()

    async def test_publish_and_stream_info(self) -> None:
        if not self.redis_available:
            self.skipTest("Redis not available on localhost:6379")

        stream_name = "lecturelens:test_stream:transcript"
        chunk = TranscriptChunk(
            session_id="test_session",
            timestamp_epoch=1000.0,
            start_seconds=0.0,
            end_seconds=2.0,
            text="Hello world test",
            confidence=0.9,
            language="en",
            audio_source="mic",
            speaker_id=None,
            importance_flags=[],
            raw_audio_path=None
        )

        entry_id = await self.bus.publish(stream_name, chunk)
        self.assertIsNotNone(entry_id)
        self.assertIn("-", entry_id)

        info = await self.bus.get_stream_info(stream_name)
        self.assertGreaterEqual(info.get("length", 0), 1)

    async def test_consumer_group_creation(self) -> None:
        if not self.redis_available:
            self.skipTest("Redis not available on localhost:6379")

        stream_name = "lecturelens:test_stream:transcript"
        group_name = "test_group"

        await self.bus.ensure_consumer_group(stream_name, group_name)
        await self.bus.ensure_consumer_group(stream_name, group_name)

        pending_count = await self.bus.get_pending_count(stream_name, group_name)
        self.assertEqual(pending_count, 0)


def make_chunk(text: str) -> TranscriptChunk:
    return TranscriptChunk(
        session_id="test_session",
        timestamp_epoch=1000.0,
        start_seconds=0.0,
        end_seconds=2.0,
        text=text,
        confidence=0.9
    )


class TestRedisBusReliability(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.bus = RedisBus(Settings().redis_url, reclaim_interval_seconds=0.0, reclaim_idle_ms=0)
        try:
            await self.bus.client.ping()
            self.redis_available = True
        except Exception:
            self.redis_available = False
        self.stream = "lecturelens:test_reliability:transcript"
        self.group = "test_group_reliability"
        if self.redis_available:
            await self.bus.client.delete(self.stream)

    async def asyncTearDown(self) -> None:
        if self.redis_available:
            await self.bus.client.delete(self.stream)
            await self.bus.close()

    async def run_consumer(self, handler, stop_flag: dict, consumer_name: str = "consumer_a") -> None:
        await self.bus.consume(
            self.stream, self.group, consumer_name, handler,
            batch_size=5, block_ms=50, should_stop=lambda: stop_flag["stop"]
        )

    async def test_all_messages_arrive_and_are_acknowledged(self) -> None:
        if not self.redis_available:
            self.skipTest("Redis not available on localhost:6379")
        for index in range(10):
            await self.bus.publish(self.stream, make_chunk(f"line {index}"))
        received: list[str] = []

        async def handler(data: dict) -> None:
            received.append(data["text"])

        stop_flag = {"stop": True}
        await asyncio.wait_for(self.run_consumer(handler, stop_flag), timeout=10)
        self.assertEqual(len(received), 10)
        self.assertEqual(await self.bus.get_pending_count(self.stream, self.group), 0)

    async def test_failing_message_discarded_after_max_attempts(self) -> None:
        if not self.redis_available:
            self.skipTest("Redis not available on localhost:6379")
        self.bus.max_delivery_attempts = 3
        await self.bus.publish(self.stream, make_chunk("poison"))
        await self.bus.publish(self.stream, make_chunk("fine"))
        received: list[str] = []

        async def handler(data: dict) -> None:
            if data["text"] == "poison":
                raise ValueError("bad message")
            received.append(data["text"])

        stop_flag = {"stop": False}
        consumer_task = asyncio.create_task(self.run_consumer(handler, stop_flag))
        for _ in range(50):
            await asyncio.sleep(0.05)
            if "fine" in received and (await self.bus.get_pending_count(self.stream, self.group)) == 0:
                break
        stop_flag["stop"] = True
        await asyncio.wait_for(consumer_task, timeout=10)
        self.assertEqual(received, ["fine"])
        self.assertEqual(await self.bus.get_pending_count(self.stream, self.group), 0)

    async def test_restarted_consumer_reprocesses_its_unacknowledged_messages(self) -> None:
        if not self.redis_available:
            self.skipTest("Redis not available on localhost:6379")
        await self.bus.publish(self.stream, make_chunk("one"))
        await self.bus.ensure_consumer_group(self.stream, self.group)
        await self.bus.read_group_messages(self.stream, self.group, "consumer_a", ">", 5, None)
        self.assertEqual(await self.bus.get_pending_count(self.stream, self.group), 1)
        received: list[str] = []

        async def handler(data: dict) -> None:
            received.append(data["text"])

        stop_flag = {"stop": True}
        await asyncio.wait_for(self.run_consumer(handler, stop_flag, "consumer_a"), timeout=10)
        self.assertEqual(received, ["one"])
        self.assertEqual(await self.bus.get_pending_count(self.stream, self.group), 0)

    async def test_other_consumers_stuck_messages_are_reclaimed(self) -> None:
        if not self.redis_available:
            self.skipTest("Redis not available on localhost:6379")
        await self.bus.publish(self.stream, make_chunk("orphan"))
        await self.bus.ensure_consumer_group(self.stream, self.group)
        await self.bus.read_group_messages(self.stream, self.group, "dead_consumer", ">", 5, None)
        received: list[str] = []

        async def handler(data: dict) -> None:
            received.append(data["text"])

        stop_flag = {"stop": False}
        consumer_task = asyncio.create_task(self.run_consumer(handler, stop_flag, "live_consumer"))
        for _ in range(100):
            await asyncio.sleep(0.05)
            if received:
                break
        stop_flag["stop"] = True
        await asyncio.wait_for(consumer_task, timeout=10)
        self.assertEqual(received, ["orphan"])

    async def test_publish_resilient_returns_none_on_connection_error(self) -> None:
        bus = RedisBus(Settings().redis_url)

        async def flaky_publish(stream_name, event):
            raise redis_module.ConnectionError("down")

        bus.publish = flaky_publish
        result = await bus.publish_resilient("lecturelens:test_buffer:stream", make_chunk("a"))
        self.assertIsNone(result)
        await bus.close()

    async def test_persistent_outbox_saves_and_reloads_on_restart(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            outbox_file = Path(td) / "outbox.jsonl"
            bus1 = RedisBus(Settings().redis_url, outbox_path=outbox_file)

            async def failing_publish(stream_name, event):
                raise redis_module.ConnectionError("offline")

            bus1.publish = failing_publish
            res = await bus1.publish_resilient("stream1", make_chunk("event 1"))
            self.assertIsNone(res)
            self.assertTrue(outbox_file.exists())
            await bus1.close()

            bus2 = RedisBus(Settings().redis_url, outbox_path=outbox_file)
            self.assertEqual(len(bus2.outbox), 1)
            self.assertEqual(bus2.outbox[0][0], "stream1")

            published_events = []
            async def ok_publish(stream_name, event):
                published_events.append((stream_name, event))
                return "1-0"

            bus2.publish = ok_publish
            flushed = await bus2.flush_outbox()
            self.assertEqual(flushed, 1)
            self.assertEqual(len(bus2.outbox), 0)
            self.assertEqual(len(published_events), 1)
            await bus2.close()

    async def test_publish_resilient_raises_when_outbox_persistence_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            outbox_file = Path(td) / "outbox.db"
            bus = RedisBus(Settings().redis_url, outbox_path=outbox_file)
            try:
                async def fail_pub(s, e):
                    raise redis_module.ConnectionError("redis unreachable")

                bus.publish = fail_pub
                bus._append_to_outbox = mock.Mock(side_effect=IOError("disk full"))
                with self.assertRaises(RuntimeError):
                    await bus.publish_resilient("stream1", make_chunk("fail disk"))
            finally:
                await bus.close()

    async def test_flush_outbox_stops_on_first_error_without_quadratic_retry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            outbox_file = Path(td) / "outbox.db"
            bus = RedisBus(Settings().redis_url, outbox_path=outbox_file)
            try:
                for i in range(5):
                    bus.outbox.append(("stream1", {"text": f"item {i}"}))
                    bus._outbox_row_ids.append(i + 1)
                publish_attempts = 0

                async def failing_publish(s, e):
                    nonlocal publish_attempts
                    publish_attempts += 1
                    raise redis_module.ConnectionError("still down")

                bus.publish = failing_publish
                flushed = await bus.flush_outbox()
                self.assertEqual(flushed, 0)
                self.assertEqual(publish_attempts, 1)
                self.assertEqual(len(bus.outbox), 5)
            finally:
                await bus.close()

    async def test_outbox_max_size_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            outbox_file = Path(td) / "outbox.db"
            bus = RedisBus(Settings().redis_url, outbox_max_size=2, outbox_path=outbox_file)
            try:
                async def failing_pub(s, e):
                    raise redis_module.ConnectionError("down")

                bus.publish = failing_pub
                await bus.publish_resilient("s", make_chunk("1"))
                await bus.publish_resilient("s", make_chunk("2"))
                with self.assertRaises(RuntimeError):
                    await bus.publish_resilient("s", make_chunk("3"))
            finally:
                await bus.close()

    async def test_consume_survives_read_errors_with_backoff(self) -> None:
        calls = {"count": 0}

        async def flaky_read(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] <= 2:
                raise redis_module.ConnectionError("down")
            return []

        self.bus.read_group_messages = flaky_read

        async def noop(data: dict) -> None:
            return None

        async def fake_ensure(stream_name, group_name):
            return None
        self.bus.ensure_consumer_group = fake_ensure
        stop_flag = {"stop": True}
        started = time.monotonic()
        await asyncio.wait_for(self.run_consumer(noop, stop_flag), timeout=10)
        self.assertTrue(calls["count"] >= 1)
        self.assertLess(time.monotonic() - started, 10)
    async def test_unfinished_legacy_migration_file_recovered_on_startup(self) -> None:
        import json
        with tempfile.TemporaryDirectory() as td:
            outbox_file = Path(td) / "outbox.db"
            legacy_file = outbox_file.with_suffix(".legacy_migration")
            legacy_entry = {"stream": "stream_legacy", "data": {"session_id": "s1", "text": "legacy item"}}
            legacy_file.write_text(json.dumps(legacy_entry) + "\n", encoding="utf-8")
            self.assertTrue(legacy_file.exists())
            self.assertFalse(outbox_file.exists())

            bus = RedisBus(Settings().redis_url, outbox_path=outbox_file)
            self.assertEqual(len(bus.outbox), 1)
            self.assertEqual(bus.outbox[0][0], "stream_legacy")
            self.assertEqual(bus.outbox[0][1]["text"], "legacy item")
            self.assertFalse(legacy_file.exists())
            self.assertTrue(outbox_file.exists())
            await bus.close()

    async def test_crash_after_publish_before_sqlite_delete_delivers_same_event_id(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            outbox_file = Path(td) / "outbox.db"
            bus1 = RedisBus(Settings().redis_url, outbox_path=outbox_file)

            async def fail_pub(s, e):
                raise redis_module.ConnectionError("offline")

            bus1.publish = fail_pub
            chunk = make_chunk("idempotent event")
            chunk.event_id = "stable_evt_123"
            await bus1.publish_resilient("stream_idem", chunk)
            await bus1.close()

            published_payloads: list[dict] = []
            bus_recovering = RedisBus(Settings().redis_url, outbox_path=outbox_file)
            self.assertEqual(len(bus_recovering.outbox), 1)

            async def ok_pub_and_crash(s, event):
                published_payloads.append(dict(event))
                raise OSError("simulated process crash before sqlite delete commit")

            bus_recovering.publish = ok_pub_and_crash
            flushed = await bus_recovering.flush_outbox()
            self.assertEqual(flushed, 0)
            self.assertEqual(len(published_payloads), 1)
            first_event_id = published_payloads[0].get("event_id")
            self.assertEqual(first_event_id, "stable_evt_123")
            if bus_recovering._db_conn is not None:
                bus_recovering._db_conn.close()
                bus_recovering._db_conn = None
            bus_recovering.outbox = []

            bus_post_recovery = RedisBus(Settings().redis_url, outbox_path=outbox_file)
            try:
                self.assertEqual(len(bus_post_recovery.outbox), 1)

                async def ok_pub(s, event):
                    published_payloads.append(dict(event))
                    return "1-0"

                bus_post_recovery.publish = ok_pub
                flushed2 = await bus_post_recovery.flush_outbox()
                self.assertEqual(flushed2, 1)
                self.assertEqual(len(published_payloads), 2)
                self.assertEqual(published_payloads[0].get("event_id"), published_payloads[1].get("event_id"))
                self.assertEqual(published_payloads[1].get("event_id"), "stable_evt_123")
            finally:
                await bus_post_recovery.close()

    async def test_legacy_migration_idempotent_if_process_crashes_before_unlinking(self) -> None:
        import json
        with tempfile.TemporaryDirectory() as td:
            outbox_file = Path(td) / "outbox.db"
            legacy_file = outbox_file.with_suffix(".legacy_migration")
            legacy_entry = {"stream": "stream_legacy", "data": {"session_id": "s1", "text": "legacy item"}}
            legacy_file.write_text(json.dumps(legacy_entry) + "\n", encoding="utf-8")

            bus1 = RedisBus(Settings().redis_url, outbox_path=outbox_file)
            self.assertEqual(len(bus1.outbox), 1)
            await bus1.close()

            legacy_file.write_text(json.dumps(legacy_entry) + "\n", encoding="utf-8")
            bus2 = RedisBus(Settings().redis_url, outbox_path=outbox_file)
            self.assertEqual(len(bus2.outbox), 1)
            self.assertFalse(legacy_file.exists())
            await bus2.close()

    async def test_outbox_starts_background_worker_when_pending_items_exist(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            outbox_file = Path(td) / "outbox.db"
            bus1 = RedisBus(Settings().redis_url, outbox_path=outbox_file)

            async def fail_pub(s, e):
                raise redis_module.ConnectionError("offline")

            bus1.publish = fail_pub
            chunk = make_chunk("pending event")
            await bus1.publish_resilient("stream_worker", chunk)
            await bus1.close()

            bus2 = RedisBus(Settings().redis_url, outbox_path=outbox_file)
            self.assertEqual(len(bus2.outbox), 1)
            bus2.ensure_background_worker()
            self.assertIsNotNone(bus2._retry_task)
            self.assertFalse(bus2._retry_task.done())
            await bus2.close()

    async def test_outbox_can_buffer_same_event_id_to_different_streams_during_outage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            outbox_file = Path(td) / "outbox.db"
            bus = RedisBus(Settings().redis_url, outbox_path=outbox_file)

            async def offline_pub(stream, event):
                raise redis_module.ConnectionError("offline")

            bus.publish = offline_pub
            chunk = make_chunk("shared event")
            chunk.event_id = "shared_evt_42"

            res1 = await bus.publish_resilient("stream_timeline", chunk)
            res2 = await bus.publish_resilient("stream_transcript", chunk)

            self.assertIsNone(res1)
            self.assertIsNone(res2)
            self.assertEqual(len(bus.outbox), 2)
            self.assertEqual(bus.outbox[0][0], "stream_timeline")
            self.assertEqual(bus.outbox[1][0], "stream_transcript")
            self.assertEqual(bus.outbox[0][1]["event_id"], "shared_evt_42")
            self.assertEqual(bus.outbox[1][1]["event_id"], "shared_evt_42")
            await bus.close()

    async def test_close_prevents_retry_worker_restart_and_cleans_up_task(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            outbox_file = Path(td) / "outbox.db"
            bus = RedisBus(Settings().redis_url, outbox_path=outbox_file)

            async def offline_pub(stream, event):
                raise redis_module.ConnectionError("offline")

            bus.publish = offline_pub
            chunk = make_chunk("failing event")
            await bus.publish_resilient("stream_fail", chunk)
            self.assertEqual(len(bus.outbox), 1)

            await bus.close()
            self.assertTrue(bus._closing)
            self.assertIsNone(bus._retry_task)
            self.assertIsNone(bus._db_conn)

    async def test_buffering_same_event_twice_deduplicates_without_duplicate_row_ids(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            outbox_file = Path(td) / "outbox.db"
            bus = RedisBus(Settings().redis_url, outbox_path=outbox_file)

            async def offline_pub(stream, event):
                raise redis_module.ConnectionError("offline")

            bus.publish = offline_pub
            chunk = make_chunk("idempotent event")
            chunk.event_id = "test_dup_event_1"

            await bus.publish_resilient("stream_dedup", chunk)
            await bus.publish_resilient("stream_dedup", chunk)

            self.assertEqual(len(bus.outbox), 1)
            self.assertEqual(len(bus._outbox_row_ids), 1)

            cur = bus._db_conn.cursor()
            cur.execute("SELECT COUNT(*) FROM outbox_entries WHERE stream_name = ? AND event_id = ?", ("stream_dedup", "test_dup_event_1"))
            self.assertEqual(cur.fetchone()[0], 1)

            published_events = []

            async def online_pub(stream, event):
                published_events.append((stream, event))
                return "1-0"

            bus.publish = online_pub
            flushed = await bus.flush_outbox()
            self.assertEqual(flushed, 1)
            self.assertEqual(len(published_events), 1)
            self.assertEqual(len(bus.outbox), 0)
            self.assertEqual(len(bus._outbox_row_ids), 0)

            cur.execute("SELECT COUNT(*) FROM outbox_entries")
            self.assertEqual(cur.fetchone()[0], 0)
            await bus.close()

    async def test_outbox_migration_recovers_from_stranded_migrated_table_and_validates_composite_index(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            outbox_file = Path(td) / "outbox.db"
            conn = sqlite3.connect(str(outbox_file))
            conn.execute("""
                CREATE TABLE outbox_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE,
                    stream_name TEXT,
                    payload TEXT,
                    created_at REAL
                );
            """)
            conn.execute("""
                CREATE TABLE outbox_entries_migrated (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT,
                    stream_name TEXT,
                    payload TEXT,
                    created_at REAL,
                    UNIQUE(stream_name, event_id)
                );
            """)
            conn.commit()
            conn.close()

            bus = RedisBus(Settings().redis_url, outbox_path=outbox_file)
            self.assertIsNotNone(bus._db_conn)
            cur = bus._db_conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='outbox_entries_migrated';")
            self.assertIsNone(cur.fetchone())
            cur.execute("PRAGMA index_list(outbox_entries);")
            indexes = cur.fetchall()
            has_composite = False
            for idx in indexes:
                if idx[2]:
                    cur.execute(f"PRAGMA index_info('{idx[1]}');")
                    cols = sorted([c[2] for c in cur.fetchall()])
                    if cols == ["event_id", "stream_name"]:
                        has_composite = True
                        break
            self.assertTrue(has_composite)
            await bus.close()

    async def test_publish_resilient_does_not_duplicate_when_redis_recovers_before_backoff_expires(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            outbox_file = Path(td) / "outbox.db"
            bus = RedisBus(Settings().redis_url, outbox_path=outbox_file)

            async def offline_pub(stream, event):
                raise redis_module.ConnectionError("offline")

            bus.publish = offline_pub
            chunk = make_chunk("backoff recovery event")
            chunk.event_id = "rec_backoff_1"

            res1 = await bus.publish_resilient("stream_backoff", chunk)
            self.assertIsNone(res1)
            self.assertEqual(len(bus.outbox), 1)

            bus._flush_backoff_seconds = 1000.0
            bus._last_flush_attempt = time.monotonic()

            direct_pub_calls = []

            async def online_pub(stream, event):
                direct_pub_calls.append((stream, event))
                return "1-0"

            bus.publish = online_pub

            res2 = await bus.publish_resilient("stream_backoff", chunk)
            self.assertIsNone(res2)
            self.assertEqual(len(direct_pub_calls), 0)
            self.assertEqual(len(bus.outbox), 1)

            bus._flush_backoff_seconds = 0.0
            flushed = await bus.flush_outbox()
            self.assertEqual(flushed, 1)
            self.assertEqual(len(direct_pub_calls), 1)
            self.assertEqual(len(bus.outbox), 0)
            await bus.close()

    async def test_legacy_outbox_migration_with_malformed_json_does_not_delete_legacy_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            outbox_file = Path(td) / "outbox.db"
            legacy_file = Path(td) / "outbox.db.legacy.jsonl"
            lines = [
                json.dumps({"stream": "s1", "data": {"event_id": "ok1", "text": "valid"}}),
                "MALFORMED_JSON_LINE_HERE",
                json.dumps({"stream": "s1", "data": {"event_id": "ok2", "text": "valid2"}}),
            ]
            legacy_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

            bus = RedisBus(Settings().redis_url, outbox_path=outbox_file)
            self.assertTrue(legacy_file.exists())
            await bus.close()

    async def test_stranded_migration_table_recovery_without_main_table(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            outbox_file = Path(td) / "outbox.db"
            conn = sqlite3.connect(str(outbox_file))
            conn.execute("""
                CREATE TABLE outbox_entries_migrated (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    stream_name TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE(stream_name, event_id)
                );
            """)
            conn.execute(
                "INSERT INTO outbox_entries_migrated (event_id, stream_name, payload, created_at) VALUES (?, ?, ?, ?);",
                ("mig_1", "s_mig", '{"test": 1}', 100.0)
            )
            conn.commit()
            conn.close()

            bus = RedisBus(Settings().redis_url, outbox_path=outbox_file)
            cur = bus._db_conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='outbox_entries';")
            self.assertIsNotNone(cur.fetchone())
            cur.execute("SELECT COUNT(*) FROM outbox_entries WHERE event_id = 'mig_1';")
            self.assertEqual(cur.fetchone()[0], 1)
            await bus.close()


if __name__ == "__main__":
    unittest.main()
