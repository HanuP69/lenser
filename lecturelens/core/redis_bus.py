import json
from typing import Any, Callable
import redis.asyncio as redis
from pydantic import BaseModel


class RedisBus:
    def __init__(self, redis_url: str) -> None:
        self.redis_url: str = redis_url
        self.client: redis.Redis = redis.from_url(redis_url, decode_responses=True)

    async def publish(self, stream_name: str, event: BaseModel) -> str:
        if hasattr(event, "to_redis_dict"):
            event_data: dict[str, str] = event.to_redis_dict()
        else:
            raw_dict = event.model_dump()
            event_data = {}
            for key, value in raw_dict.items():
                if value is None:
                    event_data[key] = ""
                elif isinstance(value, (dict, list)):
                    event_data[key] = json.dumps(value)
                else:
                    event_data[key] = str(value)
        entry_id: str = await self.client.xadd(stream_name, event_data)
        return entry_id

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

    async def consume(
        self,
        stream_name: str,
        group_name: str,
        consumer_name: str,
        handler: Callable[[dict[str, str]], Any],
        batch_size: int = 10,
        block_ms: int = 2000
    ) -> None:
        await self.ensure_consumer_group(stream_name, group_name)
        while True:
            try:
                streams_response = await self.client.xreadgroup(
                    groupname=group_name,
                    consumername=consumer_name,
                    streams={stream_name: ">"},
                    count=batch_size,
                    block=block_ms
                )
                if not streams_response:
                    continue

                for current_stream_name, message_list in streams_response:
                    for message_id, message_data in message_list:
                        try:
                            await handler(message_data)
                            await self.client.xack(stream_name, group_name, message_id)
                        except Exception:
                            continue
            except Exception:
                continue

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

    async def close(self) -> None:
        await self.client.aclose()
