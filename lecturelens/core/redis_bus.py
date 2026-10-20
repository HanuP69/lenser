import asyncio
import inspect
import json
import time
from typing import Any, Awaitable, Callable, Optional

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
        outbox_max_size: int = 1000
    ) -> None:
        self.redis_url: str = redis_url
        self.client: redis.Redis = redis.from_url(redis_url, decode_responses=True)
        self.stream_max_length: int = stream_max_length
        self.max_delivery_attempts: int = max_delivery_attempts
        self.reclaim_idle_ms: int = reclaim_idle_ms
        self.reclaim_interval_seconds: float = reclaim_interval_seconds
        self.retry_ttl_seconds: int = retry_ttl_seconds
        self.outbox_max_size: int = outbox_max_size
        self.outbox: list[tuple[str, BaseModel]] = []
        self._outbox_lock: asyncio.Lock = asyncio.Lock()
        self.failure_counts: dict[str, int] = {}

    async def flush_outbox(self) -> int:
        async with self._outbox_lock:
            if not self.outbox:
                return 0
            flushed = 0
            remaining: list[tuple[str, BaseModel]] = []
            for stream_name, event in self.outbox:
                try:
                    await self.publish(stream_name, event)
                    flushed += 1
                except Exception:
                    remaining.append((stream_name, event))
            self.outbox = remaining
            if flushed > 0:
                logger.info("flushed %d events from outbox (%d remaining)", flushed, len(self.outbox))
            return flushed

    async def publish(self, stream_name: str, event: BaseModel) -> str:
        event_data = event_to_redis_fields(event)
        entry_id: str = await self.client.xadd(
            stream_name,
            event_data,
            maxlen=self.stream_max_length,
            approximate=True
        )
        return entry_id

    async def publish_resilient(self, stream_name: str, event: BaseModel) -> Optional[str]:
        await self.flush_outbox()
        try:
            return await self.publish(stream_name, event)
        except (redis.RedisError, OSError) as error:
            logger.warning("publish failed for %s: %s, buffering in outbox", stream_name, error)
            async with self._outbox_lock:
                if len(self.outbox) < self.outbox_max_size:
                    self.outbox.append((stream_name, event))
                else:
                    logger.error("outbox full (%d items), dropping oldest event", self.outbox_max_size)
                    self.outbox.pop(0)
                    self.outbox.append((stream_name, event))
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
        await self.client.aclose()
