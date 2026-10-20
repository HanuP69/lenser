import asyncio
import inspect
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Awaitable, Callable, Optional
import uuid

import redis.asyncio as redis
from pydantic import BaseModel

from lecturelens.core.logging_setup import get_logger

logger = get_logger("lecturelens.redis_bus")


def event_to_redis_fields(event: BaseModel) -> dict[str, str]:
    if hasattr(event, "to_redis_dict"):
        return event.to_redis_dict()
    raw_dict = event.model_dump()
    event_data: dict[str, str] = {}
    for key, value in raw_dict.items():
        if value is None:
            event_data[key] = ""
        elif isinstance(value, (dict, list)):
            event_data[key] = json.dumps(value)
        else:
            event_data[key] = str(value)
    return event_data


def handler_accepts_message_id(handler: Callable[..., Any]) -> bool:
    try:
        parameters = inspect.signature(handler).parameters.values()
    except (TypeError, ValueError):
        return False
    positional = [
        parameter for parameter in parameters
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_var_positional = any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in inspect.signature(handler).parameters.values()
    )
    return len(positional) >= 2 or has_var_positional


class RedisBus:
    def __init__(
        self,
        redis_url: str,
        stream_max_length: int = 200000,
        max_delivery_attempts: int = 5,
        reclaim_idle_ms: int = 30000,
        reclaim_interval_seconds: float = 15.0,
        retry_ttl_seconds: int = 86400,
        outbox_max_size: int = 1000,
        outbox_path: Optional[Path] = None
    ) -> None:
        self.redis_url: str = redis_url
        self.client: redis.Redis = redis.from_url(redis_url, decode_responses=True)
        self.stream_max_length: int = stream_max_length
        self.max_delivery_attempts: int = max_delivery_attempts
        self.reclaim_idle_ms: int = reclaim_idle_ms
        self.reclaim_interval_seconds: float = reclaim_interval_seconds
        self.retry_ttl_seconds: int = retry_ttl_seconds
        self.outbox_max_size: int = outbox_max_size
        self.outbox_path: Optional[Path] = Path(outbox_path) if outbox_path else None
        self.outbox: list[tuple[str, Any]] = []
        self._outbox_row_ids: list[int] = []
        self._outbox_lock: asyncio.Lock = asyncio.Lock()
        self.failure_counts: dict[str, int] = {}
        self._last_flush_attempt: float = 0.0
        self._flush_backoff_seconds: float = 1.0
        self._db_conn: Optional[sqlite3.Connection] = None
        self._retry_task: Optional[asyncio.Task] = None
        if self.outbox_path:
            self._init_db()
            self._load_outbox()

    def _init_db(self) -> None:
        if not self.outbox_path:
            return
        self.outbox_path.parent.mkdir(parents=True, exist_ok=True)
        is_legacy = False
        legacy_path = None
        if self.outbox_path.exists():
            try:
                with open(self.outbox_path, "rb") as check_f:
                    header = check_f.read(16)
                    if header and not header.startswith(b"SQLite format 3"):
                        is_legacy = True
            except Exception:
                pass

        legacy_items: list[tuple[str, Any]] = []
        if is_legacy:
            try:
                legacy_path = self.outbox_path.with_suffix(".legacy_migration")
                self.outbox_path.rename(legacy_path)
                with open(legacy_path, "r", encoding="utf-8") as f:
                    for line in f:
                        stripped = line.strip()
                        if stripped:
                            item = json.loads(stripped)
                            s = item.get("stream")
                            p = item.get("data")
                            if s and p:
                                legacy_items.append((s, p))
            except Exception as err:
                logger.warning("reading legacy outbox file failed: %s", err)

        self._db_conn = sqlite3.connect(str(self.outbox_path), timeout=10.0, check_same_thread=False)
        self._db_conn.execute("PRAGMA journal_mode = WAL;")
        self._db_conn.execute("PRAGMA busy_timeout = 5000;")
        self._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS outbox_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE,
                stream_name TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at REAL NOT NULL
            );
        """)
        self._db_conn.commit()

        if legacy_items:
            for s, p in legacy_items:
                ev_id = uuid.uuid4().hex
                self._db_conn.execute(
                    "INSERT OR IGNORE INTO outbox_entries (event_id, stream_name, payload, created_at) VALUES (?, ?, ?, ?)",
                    (ev_id, s, json.dumps(p), time.time())
                )
            self._db_conn.commit()
            if legacy_path and legacy_path.exists():
                legacy_path.unlink(missing_ok=True)

    def _load_outbox(self) -> None:
        if self._db_conn is None:
            return
        try:
            cursor = self._db_conn.cursor()
            cursor.execute("SELECT id, stream_name, payload FROM outbox_entries ORDER BY id ASC;")
            rows = cursor.fetchall()
            self.outbox.clear()
            self._outbox_row_ids.clear()
            for row_id, stream_name, payload_raw in rows:
                try:
                    payload = json.loads(payload_raw)
                except Exception:
                    payload = payload_raw
                self.outbox.append((stream_name, payload))
                self._outbox_row_ids.append(row_id)
        except Exception as err:
            logger.warning("loading persistent outbox failed: %s", err)

    def _append_to_outbox(self, stream_name: str, event: Any) -> None:
        if self._db_conn is not None:
            try:
                cur = self._db_conn.cursor()
                cur.execute("SELECT COUNT(*) FROM outbox_entries;")
                count_row = cur.fetchone()
                total_in_db = int(count_row[0]) if count_row else 0
            except Exception:
                total_in_db = len(self.outbox)
        else:
            total_in_db = len(self.outbox)

        if total_in_db >= self.outbox_max_size:
            raise RuntimeError(f"Persistent outbox capacity exceeded ({self.outbox_max_size})")

        event_data = event_to_redis_fields(event) if isinstance(event, BaseModel) else event
        payload_str = json.dumps(event_data)
        event_id = uuid.uuid4().hex
        if self._db_conn is not None:
            try:
                cursor = self._db_conn.cursor()
                cursor.execute(
                    "INSERT INTO outbox_entries (event_id, stream_name, payload, created_at) VALUES (?, ?, ?, ?)",
                    (event_id, stream_name, payload_str, time.time())
                )
                self._db_conn.commit()
                self._outbox_row_ids.append(cursor.lastrowid)
            except Exception as err:
                logger.error("saving persistent outbox failed: %s", err)
                raise RuntimeError(f"Persisting to outbox failed: {err}") from err
        self.outbox.append((stream_name, event_data))

    def start_background_worker(self) -> None:
        if self._retry_task is None or self._retry_task.done():
            self._retry_task = asyncio.create_task(self._retry_worker_loop())

    async def _retry_worker_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._flush_backoff_seconds)
                if self.outbox:
                    await self.flush_outbox()
            except asyncio.CancelledError:
                break
            except Exception as err:
                logger.debug("outbox retry worker error: %s", err)

    async def flush_outbox(self) -> int:
        async with self._outbox_lock:
            if not self.outbox:
                return 0
            self._last_flush_attempt = time.monotonic()
            flushed = 0
            fail_idx = None
            for idx, (stream_name, event) in enumerate(list(self.outbox)):
                try:
                    await self.publish(stream_name, event)
                    flushed += 1
                    if self._db_conn is not None and idx < len(self._outbox_row_ids):
                        row_id = self._outbox_row_ids[idx]
                        self._db_conn.execute("DELETE FROM outbox_entries WHERE id = ?", (row_id,))
                        self._db_conn.commit()
                except (redis.RedisError, OSError):
                    fail_idx = idx
                    self._flush_backoff_seconds = min(60.0, max(2.0, self._flush_backoff_seconds * 2.0))
                    break

            if fail_idx is not None:
                self.outbox = self.outbox[fail_idx:]
                self._outbox_row_ids = self._outbox_row_ids[fail_idx:]
            else:
                self.outbox = []
                self._outbox_row_ids = []
                self._flush_backoff_seconds = 1.0

            if flushed > 0:
                logger.info("flushed %d events from outbox (%d remaining)", flushed, len(self.outbox))
            return flushed

    async def publish(self, stream_name: str, event: BaseModel | dict[str, Any]) -> str:
        if isinstance(event, BaseModel):
            event_data = event_to_redis_fields(event)
        elif isinstance(event, dict):
            event_data = {str(k): str(v) for k, v in event.items()}
        else:
            event_data = event_to_redis_fields(event)
        entry_id: str = await self.client.xadd(
            stream_name,
            event_data,
            maxlen=self.stream_max_length,
            approximate=True
        )
        return entry_id

    async def publish_resilient(self, stream_name: str, event: BaseModel) -> Optional[str]:
        self.start_background_worker()
        now = time.monotonic()
        if now - self._last_flush_attempt >= self._flush_backoff_seconds:
            await self.flush_outbox()
        try:
            return await self.publish(stream_name, event)
        except (redis.RedisError, OSError) as error:
            logger.warning("publish failed for %s: %s, buffering in persistent outbox", stream_name, error)
            async with self._outbox_lock:
                try:
                    self._append_to_outbox(stream_name, event)
                except Exception as save_err:
                    raise RuntimeError(f"Publish to redis failed ({error}) and outbox persistence failed: {save_err}") from save_err
            return None

    async def ensure_consumer_group(self, stream_name: str, group_name: str) -> None:
        try:
            await self.client.xgroup_create(
                name=stream_name,
                groupname=group_name,
                id="0",
                mkstream=True
            )
        except redis.ResponseError as error:
            error_message = str(error)
            if "BUSYGROUP" not in error_message:
                raise error

    async def read_group_messages(
        self,
        stream_name: str,
        group_name: str,
        consumer_name: str,
        start_id: str,
        batch_size: int,
        block_ms: Optional[int]
    ) -> list[tuple[str, dict[str, str]]]:
        streams_response = await self.client.xreadgroup(
            groupname=group_name,
            consumername=consumer_name,
            streams={stream_name: start_id},
            count=batch_size,
            block=block_ms
        )
        collected: list[tuple[str, dict[str, str]]] = []
        if not streams_response:
            return collected
        for current_stream_name, message_list in streams_response:
            for message_id, message_data in message_list:
                collected.append((message_id, message_data))
        return collected

    async def reclaim_idle_messages(
        self,
        stream_name: str,
        group_name: str,
        consumer_name: str,
        batch_size: int,
        idle_ms: Optional[int] = None
    ) -> list[tuple[str, dict[str, str]]]:
        result = await self.client.xautoclaim(
            stream_name,
            group_name,
            consumer_name,
            min_idle_time=idle_ms if idle_ms is not None else self.reclaim_idle_ms,
            start_id="0-0",
            count=batch_size
        )
        claimed: list[tuple[str, dict[str, str]]] = []
        if result is None or len(result) < 2:
            return claimed
        for message_id, message_data in result[1]:
            if message_data is None:
                continue
            claimed.append((message_id, message_data))
        return claimed

    async def dispatch_messages(
        self,
        stream_name: str,
        group_name: str,
        messages: list[tuple[str, dict[str, str]]],
        handler: Callable[..., Awaitable[Any]],
        auto_ack: bool = True
    ) -> None:
        pass_message_id = handler_accepts_message_id(handler)
        for message_id, message_data in messages:
            failure_key = f"{stream_name}|{message_id}"
            redis_retry_key = f"lecturelens:retry:{stream_name}:{group_name}:{message_id}"
            try:
                if pass_message_id:
                    await handler(message_data, message_id)
                else:
                    await handler(message_data)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                failure_total = self.failure_counts.get(failure_key, 0) + 1
                try:
                    redis_count = await self.client.incr(redis_retry_key)
                    await self.client.expire(redis_retry_key, self.retry_ttl_seconds)
                    failure_total = max(failure_total, redis_count)
                except Exception:
                    pass
                self.failure_counts[failure_key] = failure_total
                logger.error(
                    "handler failed for %s on %s (attempt %d): %s",
                    message_id,
                    stream_name,
                    failure_total,
                    error
                )
                if failure_total >= self.max_delivery_attempts:
                    dead_stream = f"{stream_name}:dead"
                    logger.error(
                        "message %s on %s reached max attempts (%d), routing to %s: %s",
                        message_id,
                        stream_name,
                        self.max_delivery_attempts,
                        dead_stream,
                        error
                    )
                    dead_written = False
                    try:
                        dead_payload = dict(message_data)
                        dead_payload["_original_id"] = str(message_id)
                        dead_payload["_error"] = str(error)
                        dead_payload["_failed_at"] = str(time.time())
                        await self.client.xadd(dead_stream, dead_payload)
                        dead_written = True
                    except Exception as dead_err:
                        logger.warning("failed to write to dead letter stream %s: %s", dead_stream, dead_err)
                    if dead_written:
                        await self.client.xack(stream_name, group_name, message_id)
                        try:
                            await self.client.delete(redis_retry_key)
                        except Exception:
                            pass
                        if failure_key in self.failure_counts:
                            del self.failure_counts[failure_key]
                continue
            if auto_ack:
                await self.client.xack(stream_name, group_name, message_id)
                try:
                    await self.client.delete(redis_retry_key)
                except Exception:
                    pass
            if failure_key in self.failure_counts:
                del self.failure_counts[failure_key]

    async def acknowledge(self, stream_name: str, group_name: str, message_id: str) -> None:
        await self.client.xack(stream_name, group_name, message_id)
        redis_retry_key = f"lecturelens:retry:{stream_name}:{group_name}:{message_id}"
        try:
            await self.client.delete(redis_retry_key)
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
        reclaim_idle_ms: Optional[int] = None
    ) -> None:
        backoff_seconds = 0.5
        backlog_drained = False
        backlog_cursor = "0"
        last_reclaim_time = 0.0
        group_ready = False
        while True:
            try:
                if not group_ready:
                    await self.ensure_consumer_group(stream_name, group_name)
                    group_ready = True

                if not backlog_drained:
                    messages = await self.read_group_messages(
                        stream_name, group_name, consumer_name, backlog_cursor, batch_size, None
                    )
                    if len(messages) == 0:
                        backlog_drained = True
                        continue
                    await self.dispatch_messages(stream_name, group_name, messages, handler, auto_ack)
                    backlog_cursor = messages[-1][0]
                    continue

                current_time = time.monotonic()
                if current_time - last_reclaim_time >= self.reclaim_interval_seconds:
                    last_reclaim_time = current_time
                    reclaimed = await self.reclaim_idle_messages(
                        stream_name, group_name, consumer_name, batch_size, reclaim_idle_ms
                    )
                    if len(reclaimed) > 0:
                        await self.dispatch_messages(stream_name, group_name, reclaimed, handler, auto_ack)

                messages = await self.read_group_messages(
                    stream_name, group_name, consumer_name, ">", batch_size, block_ms
                )
                backoff_seconds = 0.5
                if len(messages) == 0:
                    if should_stop is not None and should_stop():
                        return
                    continue
                await self.dispatch_messages(stream_name, group_name, messages, handler, auto_ack)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.error("consume loop error on %s: %s", stream_name, error)
                group_ready = False
                backlog_drained = False
                backlog_cursor = "0"
                if should_stop is not None and should_stop():
                    return
                await asyncio.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2.0, 10.0)

    async def get_stream_info(self, stream_name: str) -> dict[str, Any]:
        try:
            info = await self.client.xinfo_stream(stream_name)
            return info
        except Exception:
            return {}

    async def get_pending_count(self, stream_name: str, group_name: str) -> int:
        try:
            pending_summary = await self.client.xpending(stream_name, group_name)
            if pending_summary and "pending" in pending_summary:
                return int(pending_summary["pending"])
            return 0
        except Exception:
            return 0

    async def get_group_backlog(self, stream_name: str, group_name: str) -> int:
        try:
            groups = await self.client.xinfo_groups(stream_name)
        except Exception:
            return 0
        for group_info in groups:
            if group_info.get("name") != group_name:
                continue
            lag_value = group_info.get("lag")
            pending_value = int(group_info.get("pending", 0))
            if lag_value is None:
                return pending_value
            return int(lag_value) + pending_value
        return 0

    async def ping(self) -> bool:
        try:
            await self.client.ping()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        if self._retry_task is not None and not self._retry_task.done():
            self._retry_task.cancel()
            await asyncio.gather(self._retry_task, return_exceptions=True)
            self._retry_task = None
        try:
            await self.flush_outbox()
        except Exception:
            pass
        if self._db_conn is not None:
            try:
                self._db_conn.close()
            except Exception:
                pass
            self._db_conn = None
        await self.client.aclose()
