import time
from typing import Callable, Optional

import redis.asyncio as redis


class RedisRateLimiter:
    def __init__(
        self,
        redis_client: redis.Redis,
        provider_name: str,
        max_requests_per_window: int = 60,
        window_seconds: int = 60,
        time_provider: Optional[Callable[[], float]] = None
    ) -> None:
        self.redis_client: redis.Redis = redis_client
        self.provider_name: str = provider_name
        self.max_requests_per_window: int = max_requests_per_window
        self.window_seconds: int = window_seconds
        # Injectable so the window bucket can be pinned in tests; a test that
        # straddles a window boundary is otherwise flaky.
        self.time_provider: Callable[[], float] = time_provider or time.time

    def generate_window_key(self) -> str:
        current_time_epoch = int(self.time_provider())
        window_bucket = current_time_epoch // self.window_seconds
        key_name = f"lecturelens:ratelimit:{self.provider_name}:{window_bucket}"
        return key_name

    async def is_allowed(self) -> bool:
        key_name = self.generate_window_key()
        await self.redis_client.set(key_name, 0, ex=self.window_seconds + 1, nx=True)
        current_count = await self.redis_client.incr(key_name)
        if current_count <= self.max_requests_per_window:
            return True
        return False

    async def get_current_count(self) -> int:
        key_name = self.generate_window_key()
        value = await self.redis_client.get(key_name)
        if value is None:
            return 0
        return int(value)

    async def get_remaining(self) -> int:
        current_count = await self.get_current_count()
        remaining = self.max_requests_per_window - current_count
        if remaining < 0:
            return 0
        return remaining


class RedisCircuitBreaker:
    def __init__(
        self,
        redis_client: redis.Redis,
        provider_name: str,
        failure_threshold: int = 5,
        cooldown_seconds: int = 60
    ) -> None:
        self.redis_client: redis.Redis = redis_client
        self.provider_name: str = provider_name
        self.failure_threshold: int = failure_threshold
        self.cooldown_seconds: int = cooldown_seconds

    def get_breaker_key(self) -> str:
        return f"lecturelens:breaker:{self.provider_name}:open"

    def get_failure_key(self) -> str:
        return f"lecturelens:breaker:{self.provider_name}:failures"

    async def is_open(self) -> bool:
        breaker_key = self.get_breaker_key()
        exists = await self.redis_client.get(breaker_key)
        if exists is not None:
            return True
        return False

    async def get_failure_count(self) -> int:
        value = await self.redis_client.get(self.get_failure_key())
        if value is None:
            return 0
        return int(value)

    async def record_success(self) -> None:
        failure_key = self.get_failure_key()
        breaker_key = self.get_breaker_key()
        await self.redis_client.delete(failure_key)
        await self.redis_client.delete(breaker_key)

    async def record_failure(self) -> None:
        failure_key = self.get_failure_key()
        failure_count = await self.redis_client.incr(failure_key)
        await self.redis_client.expire(failure_key, self.cooldown_seconds * 2)

        if failure_count >= self.failure_threshold:
            breaker_key = self.get_breaker_key()
            await self.redis_client.set(breaker_key, "1", ex=self.cooldown_seconds)
