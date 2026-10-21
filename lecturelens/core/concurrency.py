import asyncio
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable, Optional

from lecturelens.core.logging_setup import get_logger

logger = get_logger("lecturelens.concurrency")

_thread_pool: Optional[ThreadPoolExecutor] = None
_process_pool: Optional[ProcessPoolExecutor] = None
_configured_thread_workers: int = 4
_configured_process_workers: int = 2
_process_pool_initializer: Optional[Callable[..., Any]] = None
_process_pool_initargs: tuple = ()


def configure_pools(
    thread_workers: int,
    process_workers: int,
    process_initializer: Optional[Callable[..., Any]] = None,
    process_initargs: tuple = ()
) -> None:
    global _configured_thread_workers
    global _configured_process_workers
    global _process_pool_initializer
    global _process_pool_initargs
    if thread_workers > 0:
        _configured_thread_workers = thread_workers
    if process_workers > 0:
        _configured_process_workers = process_workers
    if process_initializer is not None:
        _process_pool_initializer = process_initializer
        _process_pool_initargs = tuple(process_initargs)
        # The initializer only applies to a newly created pool.
        reset_process_pool()


def get_thread_pool(max_workers: Optional[int] = None) -> ThreadPoolExecutor:
    global _thread_pool
    if _thread_pool is None:
        worker_count = max_workers if max_workers is not None else _configured_thread_workers
        _thread_pool = ThreadPoolExecutor(max_workers=worker_count)
    return _thread_pool


def get_process_pool(max_workers: Optional[int] = None) -> ProcessPoolExecutor:
    global _process_pool
    if _process_pool is None:
        worker_count = max_workers if max_workers is not None else _configured_process_workers
        # The default start method on Linux is "fork", which copies a process
        # that has running threads. The agents run an event loop and a thread
        # pool, so forking can deadlock on a lock that another thread held at
        # fork time. "spawn" is safe.
        try:
            context = multiprocessing.get_context("spawn")
        except ValueError:
            context = multiprocessing.get_context()
        # The initializer runs in every worker, so a model that is expensive to
        # load is only loaded once per worker. Without it the model was
        # preloaded in a single worker and the other one had to load it again
        # on its first use.
        if _process_pool_initializer is not None:
            _process_pool = ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=context,
                initializer=_process_pool_initializer,
                initargs=_process_pool_initargs,
            )
        else:
            _process_pool = ProcessPoolExecutor(max_workers=worker_count, mp_context=context)
    return _process_pool


def reset_process_pool() -> None:
    global _process_pool
    broken_pool = _process_pool
    _process_pool = None
    if broken_pool is not None:
        try:
            broken_pool.shutdown(wait=False, cancel_futures=True)
        except Exception as error:
            logger.warning("process pool shutdown during reset failed: %s", error)


async def run_in_thread_pool(
    callable_function: Callable,
    *args: Any,
    max_workers: Optional[int] = None
) -> Any:
    loop = asyncio.get_running_loop()
    pool = get_thread_pool(max_workers)
    result = await loop.run_in_executor(pool, callable_function, *args)
    return result


async def run_in_process_pool(
    callable_function: Callable,
    *args: Any,
    max_workers: Optional[int] = None,
    max_attempts: int = 2
) -> Any:
    loop = asyncio.get_running_loop()
    last_error: Optional[BaseException] = None
    for attempt_number in range(max_attempts):
        pool = get_process_pool(max_workers)
        try:
            result = await loop.run_in_executor(pool, callable_function, *args)
            return result
        except BrokenProcessPool as error:
            last_error = error
            logger.error(
                "process pool broke on attempt %d, rebuilding: %s",
                attempt_number + 1,
                error
            )
            reset_process_pool()
    raise RuntimeError("process pool unavailable after rebuild attempts") from last_error


def shutdown_pools() -> None:
    global _thread_pool
    global _process_pool

    if _thread_pool is not None:
        _thread_pool.shutdown(wait=False, cancel_futures=True)
        _thread_pool = None

    if _process_pool is not None:
        _process_pool.shutdown(wait=False, cancel_futures=True)
        _process_pool = None
