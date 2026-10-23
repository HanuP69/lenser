"""The account of every question the system asks a model.

Every call through the router writes one line to a file inside the
session folder. The line carries the provider, the piece of work the
model was asked to do, the tokens it read and wrote, and the time the
call took. The improvement plan (Phase 0) is judged against these
numbers, so they are written even when nobody asks for them.

The recorder holds no lock that the router waits on and talks to no
outside service: one totals dictionary guarded by a lock, and one
append of a small line of JSON to the sink file per call.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Optional

_lock = threading.Lock()
_sink: Optional[Path] = None if os.environ.get("LECTURELENS_USAGE_SINK", "") == "" else Path(os.environ["LECTURELENS_USAGE_SINK"])
# Totals are kept per (task, provider). Attempts is every try of a
# provider in a chain, including the ones that were refused before any
# request left the machine, because the burn of the fallback chain is
# part of the cost.
_totals: dict[tuple[str, str], dict[str, Any]] = {}
_stage: ContextVar[str] = ContextVar("router_usage_stage", default="")


def set_sink(path: Any) -> None:
    """Point the account at a file, usually the session's logs folder."""
    global _sink
    with _lock:
        _sink = Path(path) if path is not None else None


def sink_path() -> Optional[Path]:
    with _lock:
        return _sink


def reset_for_tests() -> None:
    """Clear everything, so a test starts from an empty account."""
    global _sink
    with _lock:
        _sink = None
        _totals.clear()


@contextmanager
def stage(name: str) -> Any:
    """Name the piece of work the next calls in this task belong to.

    A single task of the structuring agent asks the model several
    times (summarize, definitions, ...). The stage lets the report say
    which of those asks costs the tokens, without the router having to
    know anything about the callers.
    """
    token = _stage.set(name or "")
    try:
        yield
    finally:
        _stage.reset(token)


def _as_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number


def tokens_from_response(response: Any) -> tuple[Optional[int], Optional[int]]:
    """Read the token counts a provider put in its answer.

    Every provider wrapper reports the usage under its own name, so
    three shapes are checked in turn, and missing counts are left as
    unknown instead of zero, because a zero would lie in the totals.
    """
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        return _as_int(usage.get("input_tokens")), _as_int(usage.get("output_tokens"))

    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, dict):
        token_usage = metadata.get("tokenUsage")
        if isinstance(token_usage, dict):
            return (
                _as_int(token_usage.get("prompt_tokens")),
                _as_int(token_usage.get("completion_tokens")),
            )

    extra = getattr(response, "additional_kwargs", None)
    if isinstance(extra, dict):
        usage = extra.get("usage")
        if isinstance(usage, dict):
            input_tokens = usage.get("prompt_token_count", usage.get("prompt_tokens"))
            output_tokens = usage.get(
                "candidates_token_count", usage.get("completion_tokens")
            )
            return _as_int(input_tokens), _as_int(output_tokens)
        token_usage = extra.get("token_usage")
        if isinstance(token_usage, dict):
            return (
                _as_int(token_usage.get("prompt_tokens")),
                _as_int(token_usage.get("completion_tokens")),
            )
    return None, None


def _empty_account() -> dict[str, Any]:
    return {
        "attempts": 0,
        "ok": 0,
        "limited": 0,
        "breaker_open": 0,
        "failed": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "request_seconds": 0.0,
        "tokens_known": 0,
    }


def record(
    task_type: str,
    provider_name: str,
    outcome: str,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    request_seconds: Optional[float] = None,
    total_seconds: Optional[float] = None,
    model: str = "",
    error: Optional[str] = None,
) -> None:
    """One try of one provider in one task. Nothing here raises."""
    current_stage = _stage.get()
    with _lock:
        account = _totals.setdefault((task_type, provider_name), _empty_account())
        account["attempts"] += 1
        if outcome in ("ok", "limited", "breaker_open", "failed"):
            account[outcome] += 1
        if input_tokens is not None:
            account["input_tokens"] += input_tokens
        if output_tokens is not None:
            account["output_tokens"] += output_tokens
            account["tokens_known"] += 1
        if request_seconds is not None:
            account["request_seconds"] += request_seconds

        if _sink is not None:
            row = {
                "ts": round(time.time(), 3),
                "task": task_type,
                "stage": current_stage,
                "provider": provider_name,
                "model": model,
                "outcome": outcome,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "request_seconds": round(request_seconds, 4) if request_seconds is not None else None,
                "total_seconds": round(total_seconds, 4) if total_seconds is not None else None,
                "error": (error or "")[:300],
            }
            try:
                with open(_sink, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception:
                # A full disk or a closed folder must never take a
                # lecture recording down. The totals in memory still
                # carry the call.
                pass


def snapshot() -> dict[str, Any]:
    """The totals per task, for a heartbeat or a report."""
    with _lock:
        copied = {key: dict(value) for key, value in _totals.items()}
    tasks: dict[str, dict[str, Any]] = {}
    for (task_type, provider_name), account in copied.items():
        per_task = tasks.setdefault(task_type, {
            "attempts": 0, "ok": 0, "limited": 0, "breaker_open": 0, "failed": 0,
            "input_tokens": 0, "output_tokens": 0, "request_seconds": 0.0,
            "providers": {},
        })
        for key in ("attempts", "ok", "limited", "breaker_open", "failed"):
            per_task[key] += account[key]
        per_task["input_tokens"] += account["input_tokens"]
        per_task["output_tokens"] += account["output_tokens"]
        per_task["request_seconds"] += account["request_seconds"]
        per_task["providers"][provider_name] = dict(account)
    return {"tasks": tasks}


def snapshot_for_tasks(task_types: list[str]) -> dict[str, Any]:
    """The totals of the named tasks only."""
    everything = snapshot()
    return {"tasks": {name: value for name, value in everything["tasks"].items() if name in task_types}}
