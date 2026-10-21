import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Optional

from lecturelens.core.agent_runtime import set_flag

logger = logging.getLogger(__name__)

# A shutdown phase waits as long as the agent keeps producing output. This is
# how long it may stall without any progress before the wait gives up.
PROGRESS_STALL_SECONDS: float = 45.0


class AgentProcessManager:
    def __init__(
        self,
        base_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 30.0,
        max_consecutive_failures: int = 5,
        stable_runtime_seconds: float = 60.0
    ) -> None:
        self.processes: dict[str, subprocess.Popen[bytes]] = {}
        self.command_templates: dict[str, list[str]] = {}
        self.environments: dict[str, dict[str, str]] = {}
        self.log_file_paths: dict[str, Path] = {}
        self.log_handles: dict[str, Any] = {}
        self.base_backoff_seconds: float = base_backoff_seconds
        self.max_backoff_seconds: float = max_backoff_seconds
        self.max_consecutive_failures: int = max_consecutive_failures
        self.stable_runtime_seconds: float = stable_runtime_seconds
        self.consecutive_failures: dict[str, int] = {}
        self.started_at: dict[str, float] = {}
        self.next_restart_at: dict[str, float] = {}
        self.gave_up: set[str] = set()
        self.shutting_down: bool = False

    def register_agent(
        self,
        agent_name: str,
        command: list[str],
        env: Optional[dict[str, str]] = None,
        log_file_path: Optional[Path] = None
    ) -> None:
        self.command_templates[agent_name] = command
        if env is not None:
            self.environments[agent_name] = env
        else:
            self.environments[agent_name] = os.environ.copy()

        if log_file_path is not None:
            self.log_file_paths[agent_name] = log_file_path

    def close_log_handle(self, agent_name: str) -> None:
        handle = self.log_handles.pop(agent_name, None)
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass

    def start_agent(self, agent_name: str) -> bool:
        if agent_name not in self.command_templates:
            return False

        command = self.command_templates[agent_name]
        env_vars = self.environments.get(agent_name, os.environ.copy())

        self.close_log_handle(agent_name)
        stdout_destination = subprocess.DEVNULL
        stderr_destination = subprocess.DEVNULL
        if agent_name in self.log_file_paths:
            log_path = self.log_file_paths[agent_name]
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = open(log_path, "a", encoding="utf-8")
            self.log_handles[agent_name] = log_file
            stdout_destination = log_file
            stderr_destination = log_file

        creation_flags = 0
        if os.name == "nt":
            creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

        process = subprocess.Popen(
            command,
            env=env_vars,
            stdout=stdout_destination,
            stderr=stderr_destination,
            creationflags=creation_flags
        )
        self.processes[agent_name] = process
        self.started_at[agent_name] = time.monotonic()
        return True

    def check_health(self) -> dict[str, str]:
        status_map: dict[str, str] = {}
        for agent_name, process in list(self.processes.items()):
            if agent_name in self.gave_up:
                status_map[agent_name] = "GAVE_UP"
                continue
            exit_code = process.poll()
            if exit_code is None:
                status_map[agent_name] = "RUNNING"
            elif exit_code == 0:
                status_map[agent_name] = "STOPPED"
            else:
                status_map[agent_name] = f"FAILED_EXIT_{exit_code}"
        return status_map

    def compute_backoff_seconds(self, failure_count: int) -> float:
        exponent = failure_count - 1
        if exponent < 0:
            exponent = 0
        backoff = self.base_backoff_seconds * (2 ** exponent)
        if backoff > self.max_backoff_seconds:
            return self.max_backoff_seconds
        return backoff

    def restart_failed_agents(self, now: Optional[float] = None) -> list[str]:
        if self.shutting_down:
            return []
        current_time = now if now is not None else time.monotonic()
        restarted_agents: list[str] = []
        health = self.check_health()
        for agent_name, status in health.items():
            if not status.startswith("FAILED"):
                continue
            if agent_name in self.gave_up:
                continue

            scheduled_time = self.next_restart_at.get(agent_name)
            if scheduled_time is None:
                runtime_seconds = current_time - self.started_at.get(agent_name, current_time)
                if runtime_seconds >= self.stable_runtime_seconds:
                    self.consecutive_failures[agent_name] = 0
                failure_count = self.consecutive_failures.get(agent_name, 0) + 1
                self.consecutive_failures[agent_name] = failure_count
                if failure_count > self.max_consecutive_failures:
                    self.gave_up.add(agent_name)
                    logger.error("agent %s failed %d times, giving up", agent_name, failure_count)
                    continue
                backoff = self.compute_backoff_seconds(failure_count)
                self.next_restart_at[agent_name] = current_time + backoff
                logger.warning("agent %s failed (%s), restart in %.1fs", agent_name, status, backoff)
                continue

            if current_time < scheduled_time:
                continue

            del self.next_restart_at[agent_name]
            if self.start_agent(agent_name):
                restarted_agents.append(agent_name)
        return restarted_agents

    def wait_for_exit(
        self,
        agent_names: list[str],
        timeout_seconds: Optional[float] = None,
        progress_probe: Optional[Callable[[], bool]] = None,
        progress_timeout_seconds: Optional[float] = None
    ) -> bool:
        """Wait for the agents to exit.

        The wait continues while the agent still makes progress, so a large
        backlog is flushed instead of being killed at a fixed deadline. The
        overall timeout still applies as a hard limit.
        """
        deadline = (
            time.monotonic() + timeout_seconds
            if (timeout_seconds is not None and timeout_seconds > 0)
            else None
        )
        last_progress = time.monotonic()
        last_state: Any = None
        while True:
            all_exited = True
            for agent_name in agent_names:
                process = self.processes.get(agent_name)
                if process is not None and process.poll() is None:
                    all_exited = False
                    break
            if all_exited:
                return True
            if progress_probe is not None:
                state = progress_probe()
                if state != last_state:
                    last_state = state
                    last_progress = time.monotonic()
                elif (
                    progress_timeout_seconds is not None
                    and progress_timeout_seconds > 0
                    and (time.monotonic() - last_progress) >= progress_timeout_seconds
                ):
                    # The agent stopped making progress, so waiting longer
                    # will not help.
                    return False
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.2)

    def run_shutdown_sequence(
        self,
        session_dir: Path,
        phases: list[tuple[list[str], str, Optional[float]]]
    ) -> dict[str, bool]:
        self.shutting_down = True
        outcomes: dict[str, bool] = {}
        for agent_names, flag_name, timeout_seconds in phases:
            running_names: list[str] = []
            for agent_name in agent_names:
                if agent_name in self.processes:
                    running_names.append(agent_name)
            if len(running_names) == 0:
                continue
            set_flag(session_dir, flag_name)
            exited_cleanly = self.wait_for_exit(
                running_names,
                timeout_seconds,
                progress_probe=self._phase_progress_probe(session_dir),
                progress_timeout_seconds=PROGRESS_STALL_SECONDS,
            )
            for agent_name in running_names:
                process = self.processes[agent_name]
                clean = process.poll() is not None
                outcomes[agent_name] = clean
                if not clean and (timeout_seconds is not None and timeout_seconds > 0):
                    logger.warning("agent %s ignored stop flag, terminating", agent_name)
                    self.terminate_agent(agent_name)
            if not exited_cleanly and (timeout_seconds is not None and timeout_seconds > 0):
                logger.warning(
                    "phase %s did not finish within %.0fs; the remaining work of the "
                    "agents was discarded", flag_name, timeout_seconds
                )
        for agent_name in list(self.processes.keys()):
            self.close_log_handle(agent_name)
        return outcomes

    @staticmethod
    def _phase_progress_probe(session_dir: Path) -> Callable[[], Any]:
        """Report whether the session output is still growing.

        A phase that keeps producing output is making progress, so the
        shutdown waits for it instead of killing it at a fixed deadline.
        """
        watched = [
            session_dir / "timeline.jsonl",
            session_dir / "notes.md",
            session_dir / "context.json",
        ]

        def probe() -> Any:
            total = 0
            for path in watched:
                try:
                    if path.exists():
                        total += path.stat().st_size
                except OSError:
                    continue
            return total

        return probe

    def terminate_agent(self, agent_name: str, grace_seconds: float = 3.0) -> None:
        process = self.processes.get(agent_name)
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=grace_seconds)
                except subprocess.TimeoutExpired:
                    logger.error("agent %s did not die after kill", agent_name)

    def restart_agent(self, agent_name: str) -> bool:
        self.terminate_agent(agent_name, grace_seconds=2.0)
        return self.start_agent(agent_name)

    def stop_agent(self, agent_name: str) -> None:
        self.terminate_agent(agent_name)
        self.close_log_handle(agent_name)

    def stop_all(self) -> None:
        self.shutting_down = True
        for agent_name in list(self.processes.keys()):
            self.stop_agent(agent_name)


AgentMonitor = AgentProcessManager
