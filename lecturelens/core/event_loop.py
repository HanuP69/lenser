import asyncio
import sys
from typing import Awaitable, TypeVar

T = TypeVar("T")


def ensure_event_loop_policy() -> None:
    if sys.platform == "win32":
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass


def run_async(coro: Awaitable[T]) -> T:
    ensure_event_loop_policy()
    if sys.version_info >= (3, 12) and sys.platform == "win32":
        try:
            return asyncio.run(coro, loop_factory=asyncio.SelectorEventLoop)
        except TypeError:
            return asyncio.run(coro)
    return asyncio.run(coro)


ensure_event_loop_policy()
