import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from lecturelens.config.settings import Settings
from lecturelens.core.clock import SessionClock
from lecturelens.core.concurrency import configure_pools, shutdown_pools
from lecturelens.core.event_loop import ensure_event_loop_policy, run_async
from lecturelens.core.fileio import atomic_write_text
from lecturelens.core.logging_setup import get_logger
from lecturelens.core.redis_bus import RedisBus
from lecturelens.core.validation import is_valid_session_id, sanitize_file_stem

logger = get_logger("lecturelens.runtime")

STOP_FLAG_NAME = "stop.flag"
SYNC_STOP_FLAG_NAME = "stop_sync.flag"
FINALIZE_FLAG_NAME = "finalize.flag"
PERSISTER_STOP_FLAG_NAME = "stop_persister.flag"
PAUSE_FLAG_NAME = "pause.flag"
SHUTDOWN_REQUEST_FLAG_NAME = "shutdown_request.flag"
SESSION_FINISHED_FLAG_NAME = "session_finished.flag"


def get_control_directory(session_dir: Path) -> Path:
    return session_dir / "control"


def get_heartbeat_directory(session_dir: Path) -> Path:
    return session_dir / "heartbeats"


def set_flag(session_dir: Path, flag_name: str) -> None:
    control_dir = get_control_directory(session_dir)
    control_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(control_dir / flag_name, str(time.time()))


def clear_flag(session_dir: Path, flag_name: str) -> None:
    flag_path = get_control_directory(session_dir) / flag_name
    try:
        flag_path.unlink(missing_ok=True)
    except OSError as error:
        # Another process can hold the flag open on Windows (OneDrive and
        # virus scanners do it too). The flag being unread for one call is
        # harmless; crashing the agent over it is not.
        logger.warning("flag %s could not be removed: %s", flag_name, error)


def flag_exists(session_dir: Path, flag_name: str) -> bool:
    return (get_control_directory(session_dir) / flag_name).exists()


def clear_all_flags(session_dir: Path) -> None:
    control_dir = get_control_directory(session_dir)
    if not control_dir.exists():
        return
    for entry in control_dir.iterdir():
        if entry.name.endswith(".flag"):
            try:
                entry.unlink(missing_ok=True)
            except OSError as error:
                logger.warning("flag %s could not be removed: %s", entry.name, error)


def read_heartbeats(session_dir: Path) -> dict[str, dict[str, Any]]:
    heartbeat_dir = get_heartbeat_directory(session_dir)
    heartbeats: dict[str, dict[str, Any]] = {}
    if not heartbeat_dir.exists():
        return heartbeats
    for entry in sorted(heartbeat_dir.iterdir()):
        if not entry.name.endswith(".json"):
            continue
        try:
            with open(entry, "r", encoding="utf-8") as file_handle:
                heartbeats[entry.stem] = json.load(file_handle)
        except (ValueError, OSError):
            continue
    return heartbeats


class AgentControl:
    def __init__(
        self,
        session_dir: Path,
        agent_name: str,
        stop_flag_name: str = STOP_FLAG_NAME
    ) -> None:
        self.session_dir: Path = session_dir
        self.agent_name: str = agent_name
        self.stop_flag_name: str = stop_flag_name
        self.stats: dict[str, Any] = {}
        self.state: str = "starting"

    def stop_requested(self) -> bool:
        return flag_exists(self.session_dir, self.stop_flag_name)

    def is_paused(self) -> bool:
        return flag_exists(self.session_dir, PAUSE_FLAG_NAME)

    def set_stat(self, key: str, value: Any) -> None:
        self.stats[key] = value

    def increment_stat(self, key: str, amount: int = 1) -> None:
        self.stats[key] = int(self.stats.get(key, 0)) + amount

    def write_heartbeat(self, state: Optional[str] = None) -> None:
        if state is not None:
            self.state = state
        heartbeat_dir = get_heartbeat_directory(self.session_dir)
        payload = {
            "agent": self.agent_name,
            "pid": os.getpid(),
            "time": time.time(),
            "state": self.state,
            "stats": self.stats
        }
        atomic_write_text(heartbeat_dir / f"{self.agent_name}.json", json.dumps(payload))

    async def heartbeat_loop(self, interval_seconds: float = 2.0) -> None:
        while True:
            try:
                if self.is_paused() and self.state == "live":
                    self.state = "paused"
                elif not self.is_paused() and self.state == "paused":
                    self.state = "live"
                self.write_heartbeat()
            except Exception as error:
                logger.warning("heartbeat write failed: %s", error)
            await asyncio.sleep(interval_seconds)


@dataclass
class RuntimeContext:
    settings: Settings
    session_id: str
    session_clock: SessionClock
    redis_bus: RedisBus


def parse_runtime_arguments(argument_list: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default=None)
    parser.add_argument("--audio-source", default=None)
    namespace, _ = parser.parse_known_args(argument_list)
    return namespace


def create_redis_bus(
    settings: Settings,
    outbox_path: Optional[Path] = None,
    component_name: Optional[str] = None
) -> RedisBus:
    if outbox_path is not None:
        resolved_outbox = outbox_path
    else:
        # The component name reaches this function from user input (for
        # example "--student-id"). It is sanitised so the outbox file can
        # never be written outside the output directory.
        comp = sanitize_file_stem(component_name or "default")
        resolved_outbox = settings.output_dir / f"redis_outbox_{comp}.db"
    return RedisBus(
        settings.redis_url,
        stream_max_length=settings.stream_max_length,
        max_delivery_attempts=settings.redis_max_delivery_attempts,
        reclaim_idle_ms=settings.reclaim_idle_ms,
        reclaim_interval_seconds=settings.redis_reclaim_interval_seconds,
        retry_ttl_seconds=settings.retry_key_ttl_seconds,
        outbox_max_size=settings.redis_outbox_max_size,
        outbox_path=resolved_outbox,
        backlog_drain_budget_seconds=settings.redis_backlog_drain_budget_seconds,
        socket_connect_timeout=settings.redis_socket_connect_timeout_seconds,
        socket_timeout=settings.redis_socket_timeout_seconds
    )


def create_postgres_storage(settings: Settings) -> Any:
    from lecturelens.storage.postgres import PostgresStorage
    return PostgresStorage(
        settings.postgres_url,
        query_limit=settings.postgres_query_limit
    )


GATEWAY_API_KEY_FILENAME = "gateway_api_key.txt"


def write_gateway_api_key(session_dir: Path, api_key: str) -> None:
    """Persist the effective gateway API key for local clients.

    The gateway generates an ephemeral key when none is configured, so the
    key has to be discoverable by the student client running on the same
    machine.
    """
    path = session_dir / GATEWAY_API_KEY_FILENAME
    try:
        session_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, api_key)
        os.chmod(path, 0o600)
    except OSError as error:
        logger.warning("could not persist the gateway api key: %s", error)


def read_gateway_api_key(session_dir: Path) -> Optional[str]:
    path = session_dir / GATEWAY_API_KEY_FILENAME
    if not path.exists():
        return None
    try:
        key = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return key or None


def build_runtime_context(
    namespace: argparse.Namespace,
    agent_name: Optional[str] = None
) -> RuntimeContext:
    settings = Settings()
    if namespace.session is not None and len(namespace.session) > 0:
        settings.session_name = namespace.session
    if namespace.audio_source is not None and len(namespace.audio_source) > 0:
        settings.audio_source = namespace.audio_source

    session_id = settings.session_name or "default_session"
    if not is_valid_session_id(session_id):
        raise ValueError(f"invalid session id: {session_id}")

    session_dir = settings.create_session_directories(session_id)
    session_clock = SessionClock.load_or_create(session_dir / "clock.json")
    comp = agent_name or "runtime"
    redis_bus = create_redis_bus(
        settings,
        outbox_path=session_dir / f"redis_outbox_{comp}.db",
        component_name=comp
    )
    configure_pools(settings.thread_pool_max_workers, settings.process_pool_max_workers)
    return RuntimeContext(
        settings=settings,
        session_id=session_id,
        session_clock=session_clock,
        redis_bus=redis_bus
    )


def configure_agent_pools(settings: Settings) -> None:
    """Configure the pools for an agent process.

    The audio agent preloads the Whisper model in every process pool worker,
    because the model was previously only loaded in whichever worker the
    preload task happened to run in. The other worker then had to load it
    again on its first transcription.
    """
    configure_pools(settings.thread_pool_max_workers, settings.process_pool_max_workers)
    if not settings.whisper_preload_in_every_worker:
        return
    try:
        from lecturelens.agents.audio_agent import preload_whisper_model
    except Exception:
        return
    configure_pools(
        settings.thread_pool_max_workers,
        settings.process_pool_max_workers,
        process_initializer=preload_whisper_model,
        process_initargs=(
            settings.whisper_model_size,
            settings.whisper_device,
            settings.whisper_compute_type,
        ),
    )


async def run_agent_with_heartbeat(agent: Any) -> None:
    control: AgentControl = agent.control
    control.write_heartbeat("live")
    heartbeat_task = asyncio.create_task(control.heartbeat_loop())
    try:
        await agent.run()
    finally:
        heartbeat_task.cancel()
        await asyncio.gather(heartbeat_task, return_exceptions=True)
        try:
            control.write_heartbeat("stopped")
        except Exception as error:
            logger.warning("final heartbeat failed: %s", error)


async def run_standard_agent(
    build_agent: Callable[[RuntimeContext], Any],
    agent_name: Optional[str] = None
) -> None:
    namespace = parse_runtime_arguments()
    inferred_name = agent_name or Path(sys.argv[0]).stem.replace("_agent", "")
    context = build_runtime_context(namespace, agent_name=inferred_name)
    agent = build_agent(context)
    try:
        await run_agent_with_heartbeat(agent)
    finally:
        await context.redis_bus.close()
        shutdown_pools()


def execute_agent_main(main_function: Callable[[], Awaitable[None]]) -> None:
    ensure_event_loop_policy()
    try:
        run_async(main_function())
    except KeyboardInterrupt:
        pass
