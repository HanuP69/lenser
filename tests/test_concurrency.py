import asyncio
import time
import unittest
from lecturelens.core.concurrency import (
    run_in_thread_pool,
    run_in_process_pool,
    shutdown_pools
)


def sample_cpu_task(number: int) -> int:
    accumulator = 0
    for value in range(number):
        accumulator += value
    return accumulator


def sample_io_task(duration: float) -> str:
    time.sleep(duration)
    return "completed"


class TestConcurrency(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        shutdown_pools()

    async def test_run_in_thread_pool(self) -> None:
        result = await run_in_thread_pool(sample_io_task, 0.05)
        self.assertEqual(result, "completed")

    async def test_run_in_process_pool(self) -> None:
        result = await run_in_process_pool(sample_cpu_task, 100)
        expected = sum(range(100))
        self.assertEqual(result, expected)


def crash_worker_process() -> None:
    import os
    os._exit(1)


class TestProcessPoolRecovery(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        shutdown_pools()

    async def test_pool_is_rebuilt_after_worker_crash(self) -> None:
        await run_in_process_pool(sample_cpu_task, 10)
        with self.assertRaises(Exception):
            await run_in_process_pool(crash_worker_process, max_attempts=1)
        result = await run_in_process_pool(sample_cpu_task, 100)
        self.assertEqual(result, sum(range(100)))

    async def test_configured_thread_pool_size_is_used(self) -> None:
        from lecturelens.core import concurrency
        shutdown_pools()
        concurrency.configure_pools(3, 1)
        await run_in_thread_pool(sample_io_task, 0.0)
        self.assertEqual(concurrency.get_thread_pool()._max_workers, 3)
        shutdown_pools()
        concurrency.configure_pools(4, 2)


if __name__ == "__main__":
    unittest.main()
