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
            window_seconds=10,
            time_provider=lambda: 1_000_000.0
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

    async def test_rate_limiter_slides_instead_of_using_a_fixed_window(self) -> None:
        """The fixed window counted the requests in the wall clock aligned
        bucket, so a burst around a bucket boundary was allowed in full. The
        provider sees the window from the last request, so the limiter has to
        slide."""
        if not self.redis_available:
            self.skipTest("Redis not available on localhost:6379")
        clock = {"now": 9.5}
        limiter = RedisRateLimiter(
            redis_client=self.redis_client,
            provider_name="test_sliding_provider",
            max_requests_per_window=2,
            window_seconds=10,
            time_provider=lambda: clock["now"]
        )
        try:
            # One request at the end of a wall clock minute.
            self.assertTrue(await limiter.is_allowed())
            # Another one just past the minute boundary. Only one request is
            # inside the last ten seconds, so it is allowed.
            clock["now"] = 10.5
            self.assertTrue(await limiter.is_allowed())
            # A third one a second later. All three are inside the last ten
            # seconds now, so it has to be denied. The fixed window allowed
            # it, because it had already moved to the next bucket.
            clock["now"] = 10.6
            self.assertFalse(await limiter.is_allowed())

            # Once the old requests fall out of the window, it is allowed
            # again.
            clock["now"] = 21.0
            self.assertTrue(await limiter.is_allowed())
        finally:
            await self.redis_client.delete(limiter.generate_window_key())


if __name__ == "__main__":
    unittest.main()
