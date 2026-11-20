"""Redis Stream Event Bus for real-time lecture event streaming.

Implements high-throughput, at-least-once message delivery using Redis Streams
consumer groups (XREADGROUP, XACK, XAUTOCLAIM) with an ACID SQLite outbox fallback.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from collections import OrderedDict
from pathlib import Path
import sqlite3
import time
from typing import Any, Awaitable, Callable, Optional
import uuid

import redis.asyncio as redis
from pydantic import BaseModel

from lecturelens.core.logging_setup import get_logger
from lecturelens.core.repeat_guard import Guards

logger = get_logger("lecturelens.redis_bus")


def event_to_redis_fields(event: BaseModel | dict[str, Any]) -> dict[str, str]:
    """Serialize a Pydantic model or dict to Redis Stream string key-value pairs."""
    if hasattr(event, "to_redis_dict"):
        return event.to_redis_dict()
    raw = event.model_dump() if isinstance(event, BaseModel) else event
    result: dict[str, str] = {}
    for k, v in raw.items():
        if v is None:
            result[k] = ""
        elif isinstance(v, (dict, list)):
            result[k] = json.dumps(v)
        else:
            result[k] = str(v)
    return result


def handler_accepts_message_id(handler: Callable[..., Any]) -> bool:
    """Return True if handler accepts message_id as a second argument."""
    try:
        params = list(inspect.signature(handler).parameters.values())
        pos = [p for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        has_var = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
        return len(pos) >= 2 or has_var
    except (TypeError, ValueError):
        return False


class RedisBus:
    """Async Redis Streams message bus with consumer groups and persistent outbox."""

    def __init__(
        self,
        redis_url: str,
        stream_max_length: int = 200000,
        max_delivery_attempts: int = 5,
        reclaim_idle_ms: int = 30000,
        reclaim_interval_seconds: float = 15.0,
        retry_ttl_seconds: int = 86400,
        outbox_max_size: int = 1000,
        outbox_path: Optional[Path] = None,
        backlog_drain_budget_seconds: float = 0.0,
        socket_connect_timeout: float = 5.0,
        socket_timeout: float = 30.0,
    ) -> None:
        self.redis_url = redis_url
        self.client: redis.Redis = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=socket_connect_timeout,
            socket_timeout=socket_timeout,
        )
        self.stream_max_length = stream_max_length
        self.max_delivery_attempts = max_delivery_attempts
        self.reclaim_idle_ms = reclaim_idle_ms
        self.reclaim_interval_seconds = reclaim_interval_seconds
        self.retry_ttl_seconds = retry_ttl_seconds
        self.outbox_max_size = outbox_max_size
        self.outbox_path = Path(outbox_path) if outbox_path else None
        self.backlog_drain_budget_seconds = backlog_drain_budget_seconds

        self.outbox: list[tuple[str, Any]] = []
        self._outbox_row_ids: list[int] = []
        self._outbox_lock = asyncio.Lock()
        self._inflight_event_ids: dict[str, object] = {}
        self._inflight_wait_seconds: float = 5.0
        self._recently_published: OrderedDict[str, None] = OrderedDict()
        self._recently_published_capacity: int = 4096
        self._reclaim_cursors: dict[str, str] = {}
        self.failure_counts: dict[str, int] = {}
        self._last_flush_attempt: float = 0.0
        self._flush_backoff_seconds: float = 1.0
        self._db_conn: Optional[sqlite3.Connection] = None
        self._retry_task: Optional[asyncio.Task] = None
        self._closing: bool = False
        self._guards = Guards(logger)

        if self.outbox_path:
            self._init_db()
            self._load_outbox()

    def _init_db(self) -> None:
        if not self.outbox_path:
            return
        self.outbox_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._init_db_inner()
        except Exception as error:
            broken = self.outbox_path.with_suffix(".broken")
            try:
                if self.outbox_path.exists():
                    self.outbox_path.replace(broken)
            except OSError:
                pass
            logger.error("Outbox %s could not be initialised (%s). Recreating.", self.outbox_path, error)
            self._init_db_inner()

    def _init_db_inner(self) -> None:
        assert self.outbox_path is not None
        legacy_path = self.outbox_path.with_suffix(".legacy_migration")
        legacy_file = self.outbox_path.parent / f"{self.outbox_path.name}.legacy.jsonl"
        legacy_source = legacy_path if legacy_path.exists() else (legacy_file if legacy_file.exists() else None)

        legacy_items: list[tuple[str, Any, str]] = []
        parsing_ok = True
        if legacy_source and legacy_source.exists():
            try:
                with open(legacy_source, "r", encoding="utf-8") as f:
                    for line in f:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        try:
                            item = json.loads(stripped)
                            s, p = item.get("stream"), item.get("data")
                            if s and p:
                                if isinstance(p, dict) and p.get("event_id"):
                                    ev_id = str(p["event_id"])
                                else:
                                    payload_str = json.dumps(p, sort_keys=True)
                                    ev_id = hashlib.sha256(f"{s}:{payload_str}".encode()).hexdigest()[:32]
                                legacy_items.append((s, p, ev_id))
                            else:
                                parsing_ok = False
                        except Exception:
                            parsing_ok = False
            except Exception:
                parsing_ok = False

        self._db_conn = sqlite3.connect(str(self.outbox_path), timeout=10.0, check_same_thread=False)
        self._db_conn.execute("PRAGMA journal_mode = WAL;")
        self._db_conn.execute("PRAGMA busy_timeout = 5000;")

        cur = self._db_conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='outbox_entries';")
        has_main = bool(cur.fetchone())
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='outbox_entries_migrated';")
        has_migrated = bool(cur.fetchone())

        if has_migrated and not has_main:
            self._db_conn.execute("ALTER TABLE outbox_entries_migrated RENAME TO outbox_entries;")
            self._db_conn.commit()
        elif has_migrated and has_main:
            self._db_conn.execute("""
                INSERT OR IGNORE INTO outbox_entries (event_id, stream_name, payload, created_at)
                SELECT event_id, stream_name, payload, created_at FROM outbox_entries_migrated;
            """)
            self._db_conn.execute("DROP TABLE outbox_entries_migrated;")
            self._db_conn.commit()

        self._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS outbox_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                stream_name TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at REAL NOT NULL,
                UNIQUE (stream_name, event_id)
            );
        """)

        # Schema migration check: ensure UNIQUE(stream_name, event_id) composite constraint exists
        cur.execute("PRAGMA index_list(outbox_entries);")
        indexes = cur.fetchall()
        needs_migration = any(idx[2] and [c[2] for c in cur.execute(f"PRAGMA index_info('{idx[1]}');").fetchall()] == ["event_id"] for idx in indexes)
        if needs_migration:
            self._db_conn.execute("""
                CREATE TABLE outbox_entries_migrated (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    stream_name TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE (stream_name, event_id)
                );
            """)
            self._db_conn.execute("INSERT INTO outbox_entries_migrated (id, event_id, stream_name, payload, created_at) SELECT id, event_id, stream_name, payload, created_at FROM outbox_entries;")
            self._db_conn.execute("DROP TABLE outbox_entries;")
            self._db_conn.execute("ALTER TABLE outbox_entries_migrated RENAME TO outbox_entries;")
            self._db_conn.commit()

        # Import legacy items
        if legacy_items:
            for s, p, ev_id in legacy_items:
                self._db_conn.execute(
                    "INSERT OR IGNORE INTO outbox_entries (event_id, stream_name, payload, created_at) VALUES (?, ?, ?, ?)",
                    (ev_id, s, json.dumps(p), time.time()),
                )
            self._db_conn.commit()
            if parsing_ok and legacy_source and legacy_source.exists():
                try:
                    legacy_source.unlink(missing_ok=True)
                except OSError:
                    pass

    def _load_outbox(self) -> None:
        if self._db_conn is None:
            return
        try:
            cur = self._db_conn.cursor()
            cur.execute("SELECT id, stream_name, payload FROM outbox_entries ORDER BY id ASC;")
            rows = cur.fetchall()
            self.outbox.clear()
            self._outbox_row_ids.clear()
            for row_id, s_name, p_raw in rows:
                try:
                    p = json.loads(p_raw)
                except Exception:
                    p = p_raw
                self.outbox.append((s_name, p))
                self._outbox_row_ids.append(row_id)
        except Exception as err:
            logger.warning("Loading outbox failed: %s", err)

    def _append_to_outbox(self, stream_name: str, event: Any) -> None:
        data = event_to_redis_fields(event) if isinstance(event, (BaseModel, dict)) else event
        if isinstance(data, dict):
            ev_id = str(data.get("event_id") or getattr(event, "event_id", "") or uuid.uuid4().hex)
            data["event_id"] = ev_id
        else:
            ev_id = uuid.uuid4().hex

        existing_idx = next(
            (i for i, (s, p) in enumerate(self.outbox) if s == stream_name and (p.get("event_id") if isinstance(p, dict) else getattr(p, "event_id", "")) == ev_id),
            None
        )

        existing_row_id = None
        if self._db_conn is not None:
            try:
                cur = self._db_conn.cursor()
                cur.execute("SELECT id FROM outbox_entries WHERE stream_name = ? AND event_id = ?;", (stream_name, ev_id))
                row = cur.fetchone()
                if row:
                    existing_row_id = row[0]
            except Exception as e:
                self._guards.saw("outbox dup check", e)

        is_dup = (existing_idx is not None) or (existing_row_id is not None)
        if not is_dup:
            total = len(self.outbox)
            if self._db_conn is not None:
                try:
                    cur = self._db_conn.cursor()
                    cur.execute("SELECT COUNT(*) FROM outbox_entries;")
                    r = cur.fetchone()
                    total = int(r[0]) if r else total
                except Exception:
                    pass
            if total >= self.outbox_max_size:
                raise RuntimeError(f"Persistent outbox capacity exceeded ({self.outbox_max_size})")

        assigned_row_id = None
        if self._db_conn is not None:
            try:
                payload_str = json.dumps(data)
                cur = self._db_conn.cursor()
                if existing_row_id is not None:
                    cur.execute("UPDATE outbox_entries SET payload = ?, created_at = ? WHERE id = ?", (payload_str, time.time(), existing_row_id))
                    assigned_row_id = existing_row_id
                else:
                    cur.execute("INSERT OR REPLACE INTO outbox_entries (event_id, stream_name, payload, created_at) VALUES (?, ?, ?, ?)", (ev_id, stream_name, payload_str, time.time()))
                    assigned_row_id = cur.lastrowid
                self._db_conn.commit()
            except Exception as err:
                raise RuntimeError(f"Persisting to outbox failed: {err}") from err

        if existing_idx is not None:
            self.outbox[existing_idx] = (stream_name, data)
            if assigned_row_id is not None and existing_idx < len(self._outbox_row_ids):
                self._outbox_row_ids[existing_idx] = assigned_row_id
        else:
            self.outbox.append((stream_name, data))
            if assigned_row_id is not None:
                self._outbox_row_ids.append(assigned_row_id)

    def has_pending_items(self) -> bool:
        if self._db_conn is not None:
            try:
                cur = self._db_conn.cursor()
                cur.execute("SELECT 1 FROM outbox_entries LIMIT 1;")
                return cur.fetchone() is not None
            except Exception:
                pass
        return bool(self.outbox)

    def start_background_worker(self) -> None:
        if self._closing:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if loop.is_running() and (self._retry_task is None or self._retry_task.done()):
            self._retry_task = loop.create_task(self._retry_worker_loop())

    def ensure_background_worker(self) -> None:
        if not self._closing and self.has_pending_items() and (self._retry_task is None or self._retry_task.done()):
            self.start_background_worker()

    async def _retry_worker_loop(self) -> None:
        while not self._closing:
            try:
                await asyncio.sleep(self._flush_backoff_seconds)
                if self.has_pending_items():
                    await self.flush_outbox()
            except asyncio.CancelledError:
                break
            except Exception as err:
                logger.debug("Outbox retry error: %s", err)

    async def flush_outbox(self) -> int:
        async with self._outbox_lock:
            self._last_flush_attempt = time.monotonic()
            if not self.outbox and self._db_conn is not None:
                self._load_outbox()
            if not self.outbox:
                return 0

            flushed = 0
            fail_idx = None
            for idx, (stream_name, event) in enumerate(list(self.outbox)):
                try:
                    await self.publish(stream_name, event)
                    flushed += 1
                except (redis.RedisError, OSError):
                    fail_idx = idx
                    self._flush_backoff_seconds = min(60.0, max(2.0, self._flush_backoff_seconds * 2.0))
                    break

                if self._db_conn is not None and idx < len(self._outbox_row_ids):
                    row_id = self._outbox_row_ids[idx]
                    try:
                        self._db_conn.execute("DELETE FROM outbox_entries WHERE id = ?", (row_id,))
                        self._db_conn.commit()
                    except Exception as err:
                        logger.warning("Failed to delete outbox row %s: %s", row_id, err)

            if fail_idx is not None:
                self.outbox = self.outbox[fail_idx:]
                self._outbox_row_ids = self._outbox_row_ids[fail_idx:]
            else:
                self.outbox = []
                self._outbox_row_ids = []
                self._flush_backoff_seconds = 1.0

            if flushed > 0:
                logger.info("Flushed %d events from outbox (%d remaining)", flushed, len(self.outbox))
            return flushed

    async def publish(self, stream_name: str, event: BaseModel | dict[str, Any]) -> str:
        self.ensure_background_worker()
        fields = event_to_redis_fields(event)
        return await self.client.xadd(stream_name, fields, maxlen=self.stream_max_length, approximate=True)

    async def publish_resilient(self, stream_name: str, event: BaseModel | dict[str, Any]) -> Optional[str]:
        self.start_background_worker()
        ev_id = getattr(event, "event_id", None) or (event.get("event_id") if isinstance(event, dict) else None)
        if not ev_id and hasattr(event, "session_id") and hasattr(event, "sequence_number"):
            ev_id = f"note_{getattr(event, 'session_id')}_{getattr(event, 'sequence_number')}"

        token = object()
        if ev_id:
            waited = 0.0
            while True:
                async with self._outbox_lock:
                    if ev_id in self._recently_published:
                        return "published_earlier"
                    if ev_id not in self._inflight_event_ids:
                        self._inflight_event_ids[ev_id] = token
                        break
                if waited >= self._inflight_wait_seconds:
                    break
                await asyncio.sleep(0.01)
                waited += 0.01

        try:
            res = await self._publish_resilient_inner(stream_name, event, ev_id)
            if ev_id and res is not None:
                async with self._outbox_lock:
                    self._recently_published[ev_id] = None
                    while len(self._recently_published) > self._recently_published_capacity:
                        self._recently_published.popitem(last=False)
            return res
        finally:
            if ev_id:
                async with self._outbox_lock:
                    if self._inflight_event_ids.get(ev_id) is token:
                        self._inflight_event_ids.pop(ev_id, None)

    async def _publish_resilient_inner(self, stream_name: str, event: Any, ev_id: Optional[str]) -> Optional[str]:
        was_queued = False
        async with self._outbox_lock:
            if ev_id:
                was_queued = any(s == stream_name and (p.get("event_id") if isinstance(p, dict) else getattr(p, "event_id", "")) == ev_id for s, p in self.outbox)

        if time.monotonic() - self._last_flush_attempt >= self._flush_backoff_seconds:
            await self.flush_outbox()

        if was_queued and ev_id:
            async with self._outbox_lock:
                still_queued = any(s == stream_name and (p.get("event_id") if isinstance(p, dict) else getattr(p, "event_id", "")) == ev_id for s, p in self.outbox)
            return None if still_queued else "flushed_from_outbox"

        async with self._outbox_lock:
            has_pending = bool(self.outbox)

        if has_pending:
            async with self._outbox_lock:
                self._append_to_outbox(stream_name, event)
            return None

        try:
            return await self.publish(stream_name, event)
        except (redis.RedisError, OSError) as error:
            logger.warning("Publish failed for %s (%s); buffering to outbox", stream_name, error)
            async with self._outbox_lock:
                try:
                    self._append_to_outbox(stream_name, event)
                except Exception as save_err:
                    raise RuntimeError(f"Publish to redis failed ({error}) and outbox persistence failed: {save_err}") from save_err
            return None

    async def ensure_consumer_group(self, stream_name: str, group_name: str, start_id: str = "0") -> None:
        try:
            await self.client.xgroup_create(name=stream_name, groupname=group_name, id=start_id, mkstream=True)
        except redis.ResponseError as error:
            if "BUSYGROUP" not in str(error):
                raise error

    async def read_group_messages(
        self,
        stream_name: str,
        group_name: str,
        consumer_name: str,
        start_id: str,
        batch_size: int,
        block_ms: Optional[int],
    ) -> list[tuple[str, dict[str, str]]]:
        res = await self.client.xreadgroup(groupname=group_name, consumername=consumer_name, streams={stream_name: start_id}, count=batch_size, block=block_ms)
        messages: list[tuple[str, dict[str, str]]] = []
        if res:
            for _, msg_list in res:
                for msg_id, data in msg_list:
                    messages.append((msg_id, data))
        return messages

    async def reclaim_idle_messages(
        self,
        stream_name: str,
        group_name: str,
        consumer_name: str,
        batch_size: int,
        idle_ms: Optional[int] = None,
    ) -> list[tuple[str, dict[str, str]]]:
        cursor_key = f"{stream_name}|{group_name}"
        start_id = self._reclaim_cursors.get(cursor_key, "0-0")
        res = await self.client.xautoclaim(
            stream_name,
            group_name,
            consumer_name,
            min_idle_time=idle_ms if idle_ms is not None else self.reclaim_idle_ms,
            start_id=start_id,
            count=batch_size,
        )
        claimed: list[tuple[str, dict[str, str]]] = []
        if not res or len(res) < 2:
            return claimed
        next_cursor = res[0]
        if isinstance(next_cursor, str) and next_cursor:
            self._reclaim_cursors[cursor_key] = next_cursor
        for entry in res[1]:
            if isinstance(entry, (tuple, list)) and len(entry) >= 2 and entry[1] is not None:
                claimed.append((entry[0], entry[1]))
        return claimed

    async def dispatch_messages(
        self,
        stream_name: str,
        group_name: str,
        messages: list[tuple[str, dict[str, str]]],
        handler: Callable[..., Awaitable[Any]],
        auto_ack: bool = True,
    ) -> None:
        pass_id = handler_accepts_message_id(handler)
        for msg_id, data in messages:
            fail_key = f"{stream_name}|{msg_id}"
            retry_key = f"lecturelens:retry:{stream_name}:{group_name}:{msg_id}"
            try:
                if pass_id:
                    await handler(data, msg_id)
                else:
                    await handler(data)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                total_fails = self.failure_counts.get(fail_key, 0) + 1
                try:
                    redis_count = await self.client.incr(retry_key)
                    await self.client.expire(retry_key, self.retry_ttl_seconds)
                    total_fails = max(total_fails, redis_count)
                except Exception as c_err:
                    self._guards.saw("retry counter", c_err)
                self.failure_counts[fail_key] = total_fails
                logger.error("Handler failed for %s on %s (attempt %d): %s", msg_id, stream_name, total_fails, error)

                if total_fails >= self.max_delivery_attempts:
                    dead_stream = f"{stream_name}:dead"
                    dead_payload = dict(data)
                    dead_payload["_original_id"] = str(msg_id)
                    dead_payload["_error"] = str(error)
                    dead_payload["_failed_at"] = str(time.time())
                    try:
                        await self.client.xadd(dead_stream, dead_payload)
                        await self.client.xack(stream_name, group_name, msg_id)
                        await self.client.delete(retry_key)
                        self.failure_counts.pop(fail_key, None)
                    except Exception as dead_err:
                        logger.warning("DLQ handling failed for %s: %s", msg_id, dead_err)
                continue

            if auto_ack:
                await self.client.xack(stream_name, group_name, msg_id)
                try:
                    await self.client.delete(retry_key)
                except Exception:
                    pass
            self.failure_counts.pop(fail_key, None)

    async def acknowledge(self, stream_name: str, group_name: str, message_id: str) -> None:
        await self.client.xack(stream_name, group_name, message_id)
        try:
            await self.client.delete(f"lecturelens:retry:{stream_name}:{group_name}:{message_id}")
        except Exception:
            pass

    async def consume(
        self,
        stream_name: str,
        group_name: str,
        consumer_name: str,
        handler: Callable[..., Awaitable[Any]],
        batch_size: int = 10,
        block_ms: int = 2000,
        should_stop: Optional[Callable[[], bool]] = None,
        auto_ack: bool = True,
        reclaim_idle_ms: Optional[int] = None,
    ) -> None:
        backoff = 0.5
        backlog_drained = False
        backlog_cursor = "0"
        drain_start = time.monotonic()
        last_reclaim = 0.0
        group_ready = False
        self.ensure_background_worker()

        while True:
            try:
                if not group_ready:
                    await self.ensure_consumer_group(stream_name, group_name)
                    group_ready = True

                if not backlog_drained:
                    if self.backlog_drain_budget_seconds > 0 and (time.monotonic() - drain_start) >= self.backlog_drain_budget_seconds:
                        backlog_drained = True
                        continue
                    msgs = await self.read_group_messages(stream_name, group_name, consumer_name, backlog_cursor, batch_size, None)
                    if not msgs:
                        backlog_drained = True
                        continue
                    await self.dispatch_messages(stream_name, group_name, msgs, handler, auto_ack)
                    backlog_cursor = msgs[-1][0]
                    continue

                now = time.monotonic()
                if now - last_reclaim >= self.reclaim_interval_seconds:
                    last_reclaim = now
                    reclaimed = await self.reclaim_idle_messages(stream_name, group_name, consumer_name, batch_size, reclaim_idle_ms)
                    if reclaimed:
                        await self.dispatch_messages(stream_name, group_name, reclaimed, handler, auto_ack)

                msgs = await self.read_group_messages(stream_name, group_name, consumer_name, ">", batch_size, block_ms)
                backoff = 0.5
                if not msgs:
                    if should_stop and should_stop():
                        return
                    continue
                await self.dispatch_messages(stream_name, group_name, msgs, handler, auto_ack)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.error("Consume loop error on %s: %s", stream_name, error)
                group_ready = False
                backlog_drained = False
                backlog_cursor = "0"
                if should_stop and should_stop():
                    return
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 10.0)

    async def get_stream_info(self, stream_name: str) -> dict[str, Any]:
        try:
            return await self.client.xinfo_stream(stream_name)
        except Exception:
            return {}

    async def get_pending_count(self, stream_name: str, group_name: str) -> int:
        try:
            summary = await self.client.xpending(stream_name, group_name)
            return int(summary.get("pending", 0)) if summary else 0
        except Exception:
            return 0

    async def get_group_backlog(self, stream_name: str, group_name: str) -> int:
        try:
            groups = await self.client.xinfo_groups(stream_name)
            for g in groups:
                if g.get("name") == group_name:
                    lag = g.get("lag")
                    pending = int(g.get("pending", 0))
                    return pending if lag is None else int(lag) + pending
        except Exception:
            pass
        return 0

    async def ping(self) -> bool:
        try:
            await self.client.ping()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        self._closing = True
        try:
            await self.flush_outbox()
        except Exception:
            pass
        if self._retry_task and not self._retry_task.done():
            self._retry_task.cancel()
            await asyncio.gather(self._retry_task, return_exceptions=True)
            self._retry_task = None
        if self._db_conn is not None:
            try:
                self._db_conn.close()
            except Exception:
                pass
            self._db_conn = None
        await self.client.aclose()
