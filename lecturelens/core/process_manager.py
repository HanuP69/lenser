"""Process Supervisor & Process Control Block (PCB) Management.

Implements College OS & Placement Standards (Love Babbar):
1. Process Control Block (PCB): Explicit dataclass tracking PID, state, execution flags, and exit status.
2. Independent Address Spaces: Spawns CPU-bound workers as OS processes to circumvent CPython GIL.
3. Zombie & Orphan Prevention: Enforces process.poll() and process.wait() reaping on SIGINT/SIGTERM.
4. Graceful Phased Teardown: Sequential draining of in-flight queues before SIGKILL escalation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Optional

from lecturelens.core.agent_runtime import set_flag

logger = logging.getLogger("lecturelens.core.process_manager")

PROGRESS_STALL_SECONDS: Optional[float] = None


class ProcessState(str, Enum):
    """OS Process Lifecycle States."""
    NEW = "NEW"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    FAILED_SPAWN = "FAILED_SPAWN"
    GAVE_UP = "GAVE_UP"


@dataclass
class ProcessControlBlock:
    """Process Control Block (PCB) representing an isolated OS process."""
    name: str
    command: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    log_file_path: Optional[Path] = None
    process: Optional[subprocess.Popen[bytes]] = None
    started_at: Optional[float] = None
    consecutive_failures: int = 0
    next_restart_at: Optional[float] = None
    gave_up: bool = False

    @property
    def pid(self) -> Optional[int]:
        """Return the OS Process Identifier (PID)."""
        return getattr(self.process, "pid", None)

    @property
    def is_alive(self) -> bool:
        """Check if process is actively executing (prevents Zombie state)."""
        return self.process is not None and getattr(self.process, "poll", lambda: None)() is None

    @property
    def exit_code(self) -> Optional[int]:
        """Inspect exit code via OS waitpid / poll syscall."""
        return getattr(self.process, "poll", lambda: None)() if self.process else None


class AgentProcessManager:
    """OS Process Supervisor managing agent lifecycles, health checks, and shutdown."""

    def __init__(
        self,
        base_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 30.0,
        max_consecutive_failures: int = 5,
        stable_runtime_seconds: float = 60.0
    ) -> None:
        self.pcb_table: dict[str, ProcessControlBlock] = {}
        self.processes: dict[str, Any] = {}
        self.command_templates: dict[str, list[str]] = {}
        self.environments: dict[str, dict[str, str]] = {}
        self.log_file_paths: dict[str, Path] = {}
        self.log_handles: dict[str, Any] = {}
        self.started_at: dict[str, float] = {}
        self.consecutive_failures: dict[str, int] = {}
        self.next_restart_at: dict[str, float] = {}
        self.gave_up: set[str] = set()

        self.base_backoff_seconds: float = base_backoff_seconds
        self.max_backoff_seconds: float = max_backoff_seconds
        self.max_consecutive_failures: int = max_consecutive_failures
        self.stable_runtime_seconds: float = stable_runtime_seconds
        self.shutting_down: bool = False

    def register_agent(
        self,
        agent_name: str,
        command: list[str],
        env: Optional[dict[str, str]] = None,
        log_file_path: Optional[Path] = None
    ) -> None:
        """Register a new process in the supervisor's PCB table."""
        env_dict = env if env is not None else os.environ.copy()
        self.command_templates[agent_name] = command
        self.environments[agent_name] = env_dict
        if log_file_path:
            self.log_file_paths[agent_name] = log_file_path

        self.pcb_table[agent_name] = ProcessControlBlock(
            name=agent_name,
            command=command,
            env=env_dict,
            log_file_path=log_file_path
        )

    def close_log_handle(self, agent_name: str) -> None:
        handle = self.log_handles.pop(agent_name, None)
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass

    def start_agent(self, agent_name: str) -> bool:
        """Spawn agent as an independent OS process with isolated address space."""
        command = self.command_templates.get(agent_name)
        if command is None:
            return False

        env_vars = self.environments.get(agent_name, os.environ.copy())
        self.close_log_handle(agent_name)
        stdout_dest = subprocess.DEVNULL
        stderr_dest = subprocess.DEVNULL

        log_path = self.log_file_paths.get(agent_name)
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = open(log_path, "a", encoding="utf-8")
            self.log_handles[agent_name] = log_file
            stdout_dest = log_file
            stderr_dest = log_file

        creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0

        try:
            process = subprocess.Popen(
                command,
                env=env_vars,
                stdout=stdout_dest,
                stderr=stderr_dest,
                creationflags=creation_flags
            )
            now = time.monotonic()
            self.processes[agent_name] = process
            self.started_at[agent_name] = now
            if agent_name in self.pcb_table:
                self.pcb_table[agent_name].process = process
                self.pcb_table[agent_name].started_at = now
            return True
        except Exception as spawn_error:
            logger.error("agent %s could not be started: %s", agent_name, spawn_error)
            self.close_log_handle(agent_name)
            return False

    def check_health(self) -> dict[str, str]:
        """Query exit codes for all registered processes to monitor states."""
        status_map: dict[str, str] = {}
        for name, process in list(self.processes.items()):
            if name in self.gave_up:
                status_map[name] = ProcessState.GAVE_UP.value
                continue
            poll_func = getattr(process, "poll", lambda: None)
            code = poll_func()
            if code is None:
                status_map[name] = ProcessState.RUNNING.value
            elif code == 0:
                status_map[name] = ProcessState.STOPPED.value
            else:
                status_map[name] = f"FAILED_EXIT_{code}"

        for name in self.command_templates:
            if name not in status_map and name not in self.gave_up:
                if name not in self.processes:
                    status_map[name] = ProcessState.FAILED_SPAWN.value
        return status_map

    def compute_backoff_seconds(self, failure_count: int) -> float:
        exponent = max(0, failure_count - 1)
        return min(self.max_backoff_seconds, self.base_backoff_seconds * (2 ** exponent))

    def restart_failed_agents(self, now: Optional[float] = None) -> list[str]:
        """Automatically restart failed child processes with exponential backoff."""
        if self.shutting_down:
            return []
        current_time = now if now is not None else time.monotonic()
        restarted: list[str] = []
        health = self.check_health()

        for name, status in health.items():
            if not status.startswith("FAILED"):
                continue
            if name in self.gave_up:
                continue

            scheduled_time = self.next_restart_at.get(name)
            if scheduled_time is None:
                runtime = current_time - self.started_at.get(name, current_time)
                if runtime >= self.stable_runtime_seconds:
                    self.consecutive_failures[name] = 0
                count = self.consecutive_failures.get(name, 0) + 1
                self.consecutive_failures[name] = count
                if count > self.max_consecutive_failures:
                    self.gave_up.add(name)
                    logger.error("agent %s failed %d times, giving up", name, count)
                    continue
                backoff = self.compute_backoff_seconds(count)
                self.next_restart_at[name] = current_time + backoff
                logger.warning("agent %s failed (%s), restart in %.1fs", name, status, backoff)
                continue

            if current_time >= scheduled_time:
                del self.next_restart_at[name]
                if self.start_agent(name):
                    restarted.append(name)
        return restarted

    def wait_for_exit(
        self,
        agent_names: list[str],
        timeout_seconds: Optional[float] = None,
        progress_probe: Optional[Callable[[], bool]] = None,
        progress_timeout_seconds: Optional[float] = None
    ) -> bool:
        """Wait for child processes to exit while reaping Zombie entries."""
        deadline = (time.monotonic() + timeout_seconds) if (timeout_seconds and timeout_seconds > 0) else None
        last_progress = time.monotonic()
        last_state: Any = None

        while True:
            all_exited = True
            for name in agent_names:
                proc = self.processes.get(name)
                poll_func = getattr(proc, "poll", lambda: None) if proc else lambda: 0
                if proc is not None and poll_func() is None:
                    all_exited = False
                    break
            if all_exited:
                return True

            if progress_probe is not None:
                state = progress_probe()
                if state != last_state:
                    last_state = state
                    last_progress = time.monotonic()
                elif progress_timeout_seconds and (time.monotonic() - last_progress) >= progress_timeout_seconds:
                    return False

            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.2)

    def run_shutdown_sequence(
        self,
        session_dir: Path,
        phases: list[tuple[list[str], str, Optional[float]]]
    ) -> dict[str, bool]:
        """Execute graceful phased shutdown on SIGINT/SIGTERM to drain buffers."""
        self.shutting_down = True
        outcomes: dict[str, bool] = {}

        for agent_names, flag_name, timeout_seconds in phases:
            running = [name for name in agent_names if name in self.processes]
            if not running:
                continue

            set_flag(session_dir, flag_name)
            try:
                exited_cleanly = self.wait_for_exit(
                    running,
                    timeout_seconds,
                    progress_probe=self._phase_progress_probe(session_dir) if timeout_seconds else None,
                    progress_timeout_seconds=PROGRESS_STALL_SECONDS if timeout_seconds else None
                )
            except KeyboardInterrupt:
                logger.warning("interrupted during %s wait; agents finishing in background", flag_name)
                exited_cleanly = False

            for name in running:
                proc = self.processes[name]
                poll_func = getattr(proc, "poll", lambda: None)
                code = poll_func()
                clean = (code == 0)
                outcomes[name] = clean
                if not clean and timeout_seconds:
                    logger.warning("agent %s incomplete (code %s); terminating", name, code)
                    self.terminate_agent(name)

        for name in list(self.processes.keys()):
            self.close_log_handle(name)
        return outcomes

    @staticmethod
    def _phase_progress_probe(session_dir: Path) -> Callable[[], Any]:
        watched = [session_dir / "timeline.jsonl", session_dir / "notes.md", session_dir / "context.json"]
        heartbeat_dir = session_dir / "heartbeats"

        def probe() -> Any:
            total = sum(p.stat().st_size for p in watched if p.exists())
            newest = 0.0
            try:
                for path in heartbeat_dir.glob("*.json"):
                    newest = max(newest, path.stat().st_mtime)
            except OSError:
                pass
            return (total, newest)

        return probe

    def terminate_agent(self, agent_name: str, grace_seconds: float = 3.0) -> None:
        """Send SIGTERM followed by SIGKILL, and waitpid to reap Zombie processes."""
        proc = self.processes.get(agent_name)
        if proc is None:
            return
        poll_func = getattr(proc, "poll", lambda: None)
        if poll_func() is None:
            if hasattr(proc, "terminate"):
                proc.terminate()
            if hasattr(proc, "wait"):
                try:
                    proc.wait(timeout=grace_seconds)
                except subprocess.TimeoutExpired:
                    if hasattr(proc, "kill"):
                        proc.kill()
                    try:
                        proc.wait(timeout=grace_seconds)
                    except subprocess.TimeoutExpired:
                        logger.error("agent %s did not terminate after SIGKILL", agent_name)


AgentMonitor = AgentProcessManager
