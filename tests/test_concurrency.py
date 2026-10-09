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


if __name__ == "__main__":
    unittest.main()
