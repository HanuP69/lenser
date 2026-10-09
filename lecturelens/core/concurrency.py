import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Any, Callable


_thread_pool: ThreadPoolExecutor | None = None
_process_pool: ProcessPoolExecutor | None = None


def get_thread_pool(max_workers: int = 4) -> ThreadPoolExecutor:
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(max_workers=max_workers)
    return _thread_pool


def get_process_pool(max_workers: int = 2) -> ProcessPoolExecutor:
    global _process_pool
    if _process_pool is None:
        _process_pool = ProcessPoolExecutor(max_workers=max_workers)
    return _process_pool


async def run_in_thread_pool(
    callable_function: Callable,
    *args: Any,
    max_workers: int = 4
) -> Any:
    loop = asyncio.get_running_loop()
    pool = get_thread_pool(max_workers)
    result = await loop.run_in_executor(pool, callable_function, *args)
    return result


async def run_in_process_pool(
    callable_function: Callable,
    *args: Any,
    max_workers: int = 2
) -> Any:
    loop = asyncio.get_running_loop()
    pool = get_process_pool(max_workers)
    result = await loop.run_in_executor(pool, callable_function, *args)
    return result


def shutdown_pools() -> None:
    global _thread_pool
    global _process_pool

    if _thread_pool is not None:
        _thread_pool.shutdown(wait=True, cancel_futures=False)
        _thread_pool = None

    if _process_pool is not None:
        _process_pool.shutdown(wait=True, cancel_futures=False)
        _process_pool = None
