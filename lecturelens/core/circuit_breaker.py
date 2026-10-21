import logging
import time
import uuid
from typing import Callable, Optional

import redis.asyncio as redis

logger = logging.getLogger(__name__)


# A true sliding window limiter.
#
# The fixed window version counted the requests in the wall clock aligned
# bucket of the current minute, so two calls one second apart around a
# bucket boundary were both allowed. Providers that limit "one request per
# minute" measure the window from the last request, so the burst slipped
# through and the provider answered with a rate limit error.
#
# The requests are kept in a sorted set, scored by their timestamp. An entry
# is removed once it falls out of the window, so the count is always the
# number of requests in the last window_seconds, no matter where the wall
# clock boundaries are.
SLIDING_WINDOW_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]

redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)
local count = redis.call('ZCARD', key)
if count >= limit then
    return 0
end
redis.call('ZADD', key, now, member)
redis.call('PEXPIRE', key, math.ceil(window * 1000) + 1000)
return 1
"""

SLIDING_WINDOW_COUNT_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)
return redis.call('ZCARD', key)
"""


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
        # Injectable so the window can be pinned in tests.
        self.time_provider: Callable[[], float] = time_provider or time.time
        self._scripts: dict[str, object] = {}

    def _script(self, source: str) -> object:
        """Register the Lua script once and reuse it."""
        script = self._scripts.get(source)
        if script is None:
            script = self.redis_client.register_script(source)
            self._scripts[source] = script
        return script

    def generate_window_key(self) -> str:
        """The key that holds the request timestamps.

        It used to be a key per wall clock bucket. There is one key now, the
        entries are scored by their own timestamp, so the window slides.
        """
        return f"lecturelens:ratelimit:{self.provider_name}"

    async def is_allowed(self) -> bool:
        key_name = self.generate_window_key()
        now = float(self.time_provider())
        # The member has to be unique, two requests in the same millisecond
        # would otherwise collapse into one entry.
        member = f"{now:.6f}:{uuid.uuid4().hex[:8]}"
        try:
            script = self._script(SLIDING_WINDOW_SCRIPT)
            allowed = await script(
                keys=[key_name],
                args=[now, float(self.window_seconds), int(self.max_requests_per_window), member],
            )
            return int(allowed) == 1
        except Exception as error:
            # The guard store is unavailable. Fail open, the provider is
            # still the thing that decides whether a call is allowed.
            logger.warning("rate limiter unavailable for %s: %s", self.provider_name, error)
            return True

    async def get_current_count(self) -> int:
        key_name = self.generate_window_key()
        now = float(self.time_provider())
        try:
            script = self._script(SLIDING_WINDOW_COUNT_SCRIPT)
            count = await script(
                keys=[key_name],
                args=[now, float(self.window_seconds)],
            )
            return int(count)
        except Exception:
            return 0

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
        breaker_key = self.get_breaker_key()
        failure_count = await self.redis_client.incr(failure_key)
        await self.redis_client.expire(failure_key, self.cooldown_seconds * 2)

        if failure_count >= self.failure_threshold:
            await self.redis_client.set(breaker_key, "1", ex=self.cooldown_seconds)
