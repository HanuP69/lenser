import unittest
import redis.asyncio as redis
from lecturelens.config.settings import Settings
from lecturelens.core.circuit_breaker import RedisRateLimiter, RedisCircuitBreaker


class TestCircuitBreaker(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.redis_client = redis.from_url(Settings().redis_url, decode_responses=True)
        try:
            await self.redis_client.ping()
            self.redis_available = True
        except Exception:
            self.redis_available = False

    async def asyncTearDown(self) -> None:
        if self.redis_available:
            keys = await self.redis_client.keys("lecturelens:ratelimit:test_provider:*")
            for k in keys:
                await self.redis_client.delete(k)
            keys = await self.redis_client.keys("lecturelens:breaker:test_provider:*")
            for k in keys:
                await self.redis_client.delete(k)
            await self.redis_client.aclose()

    async def test_rate_limiter_boundary(self) -> None:
        if not self.redis_available:
            self.skipTest("Redis not available on localhost:6379")

        limiter = RedisRateLimiter(
            redis_client=self.redis_client,
            provider_name="test_provider",
            max_requests_per_window=2,
            window_seconds=10
        )

        allowed_first = await limiter.is_allowed()
        allowed_second = await limiter.is_allowed()
        allowed_third = await limiter.is_allowed()

        self.assertTrue(allowed_first)
        self.assertTrue(allowed_second)
        self.assertFalse(allowed_third)

    async def test_circuit_breaker_tripping(self) -> None:
        if not self.redis_available:
            self.skipTest("Redis not available on localhost:6379")

        breaker = RedisCircuitBreaker(
            redis_client=self.redis_client,
            provider_name="test_provider",
            failure_threshold=2,
            cooldown_seconds=5
        )

        initial_state = await breaker.is_open()
        self.assertFalse(initial_state)

        await breaker.record_failure()
        still_closed = await breaker.is_open()
        self.assertFalse(still_closed)

        await breaker.record_failure()
        tripped_open = await breaker.is_open()
        self.assertTrue(tripped_open)

        await breaker.record_success()
        reset_closed = await breaker.is_open()
        self.assertFalse(reset_closed)

    async def test_rate_limiter_remaining_and_ttl_is_set(self) -> None:
        if not self.redis_available:
            self.skipTest("Redis not available on localhost:6379")
        limiter = RedisRateLimiter(
            redis_client=self.redis_client,
            provider_name="test_provider",
            max_requests_per_window=3,
            window_seconds=10
        )
        await limiter.is_allowed()
        self.assertEqual(await limiter.get_current_count(), 1)
        self.assertEqual(await limiter.get_remaining(), 2)
        ttl = await self.redis_client.ttl(limiter.generate_window_key()) if hasattr(self.redis_client, "ttl") else 1
        self.assertGreater(ttl, 0)


if __name__ == "__main__":
    unittest.main()
