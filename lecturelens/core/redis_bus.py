import asyncio
import inspect
import json
import time
from collections import deque
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
        outbox_capacity: int = 2000,
        stream_max_length: int = 200000,
        max_delivery_attempts: int = 5,
        reclaim_idle_ms: int = 30000,
        reclaim_interval_seconds: float = 15.0
    ) -> None:
        self.redis_url: str = redis_url
        self.client: redis.Redis = redis.from_url(redis_url, decode_responses=True)
        self.outbox: deque = deque()
        self.outbox_capacity: int = outbox_capacity
        self.stream_max_length: int = stream_max_length
        self.max_delivery_attempts: int = max_delivery_attempts
        self.reclaim_idle_ms: int = reclaim_idle_ms
        self.reclaim_interval_seconds: float = reclaim_interval_seconds
        self.failure_counts: dict[str, int] = {}

    async def publish(self, stream_name: str, event: BaseModel) -> str:
        event_data = event_to_redis_fields(event)
        entry_id: str = await self.client.xadd(
            stream_name,
            event_data,
            maxlen=self.stream_max_length,
            approximate=True
        )
        return entry_id

    async def flush_outbox(self) -> int:
        flushed_count = 0
        while len(self.outbox) > 0:
            stream_name, event = self.outbox[0]
            try:
                await self.publish(stream_name, event)
            except (redis.RedisError, OSError) as error:
                logger.warning("outbox flush paused: %s", error)
                return flushed_count
            self.outbox.popleft()
            flushed_count += 1
        return flushed_count

    async def publish_resilient(self, stream_name: str, event: BaseModel) -> Optional[str]:
        if len(self.outbox) > 0:
            await self.flush_outbox()
        if len(self.outbox) > 0:
            self.append_to_outbox(stream_name, event)
            return None
        try:
            return await self.publish(stream_name, event)
        except (redis.RedisError, OSError) as error:
            logger.warning("publish failed, buffering event for %s: %s", stream_name, error)
            self.append_to_outbox(stream_name, event)
            return None

    def append_to_outbox(self, stream_name: str, event: BaseModel) -> None:
        if len(self.outbox) >= self.outbox_capacity:
            self.outbox.popleft()
            logger.error("outbox full, oldest buffered event dropped")
        self.outbox.append((stream_name, event))

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

    async def move_to_dead_letter(
        self,
        stream_name: str,
        group_name: str,
        message_id: str,
        message_data: dict[str, str],
        error: Exception
    ) -> None:
        dead_stream_name = f"{stream_name}:dead"
        dead_fields = {
            "original_id": str(message_id),
            "error": str(error)[:500],
            "fields": json.dumps(message_data)
        }
        await self.client.xadd(dead_stream_name, dead_fields, maxlen=10000, approximate=True)
        await self.client.xack(stream_name, group_name, message_id)
        logger.error("message %s on %s moved to dead letter stream", message_id, stream_name)

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
            try:
                if pass_message_id:
                    await handler(message_data, message_id)
                else:
                    await handler(message_data)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                failure_total = self.failure_counts.get(failure_key, 0) + 1
                self.failure_counts[failure_key] = failure_total
                logger.error(
                    "handler failed for %s on %s (attempt %d): %s",
                    message_id,
                    stream_name,
                    failure_total,
                    error
                )
                if failure_total >= self.max_delivery_attempts:
                    await self.move_to_dead_letter(
                        stream_name, group_name, message_id, message_data, error
                    )
                    del self.failure_counts[failure_key]
                continue
            if auto_ack:
                await self.client.xack(stream_name, group_name, message_id)
            if failure_key in self.failure_counts:
                del self.failure_counts[failure_key]

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
