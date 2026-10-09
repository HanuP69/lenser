import os
from pathlib import Path
import subprocess
from typing import Optional


class AgentMonitor:
    def __init__(self) -> None:
        self.processes: dict[str, subprocess.Popen] = {}
        self.command_templates: dict[str, list[str]] = {}
        self.environments: dict[str, dict[str, str]] = {}
        self.log_file_paths: dict[str, Path] = {}

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

    def start_agent(self, agent_name: str) -> bool:
        if agent_name not in self.command_templates:
            return False

        command = self.command_templates[agent_name]
        env_vars = self.environments.get(agent_name, os.environ.copy())

        stdout_destination = subprocess.DEVNULL
        stderr_destination = subprocess.DEVNULL

        if agent_name in self.log_file_paths:
            log_path = self.log_file_paths[agent_name]
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = open(log_path, "a", encoding="utf-8")
            stdout_destination = log_file
            stderr_destination = log_file

        process = subprocess.Popen(
            command,
            env=env_vars,
            stdout=stdout_destination,
            stderr=stderr_destination,
            text=True
        )
        self.processes[agent_name] = process
        return True

    def check_health(self) -> dict[str, str]:
        status_map: dict[str, str] = {}
        for agent_name, process in list(self.processes.items()):
            exit_code = process.poll()
            if exit_code is None:
                status_map[agent_name] = "RUNNING"
            elif exit_code == 0:
                status_map[agent_name] = "STOPPED"
            else:
                status_map[agent_name] = f"FAILED_EXIT_{exit_code}"
        return status_map

    def restart_agent(self, agent_name: str) -> bool:
        if agent_name in self.processes:
            existing_process = self.processes[agent_name]
            if existing_process.poll() is None:
                existing_process.terminate()
                try:
                    existing_process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    existing_process.kill()
        return self.start_agent(agent_name)

    def restart_failed_agents(self) -> list[str]:
        restarted_agents: list[str] = []
        health = self.check_health()
        for agent_name, status in health.items():
            if status.startswith("FAILED"):
                success = self.restart_agent(agent_name)
                if success:
                    restarted_agents.append(agent_name)
        return restarted_agents

    def stop_agent(self, agent_name: str) -> None:
        if agent_name in self.processes:
            process = self.processes[agent_name]
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    process.kill()

    def stop_all(self) -> None:
        for agent_name in list(self.processes.keys()):
            self.stop_agent(agent_name)
