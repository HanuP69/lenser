"""Base agent abstraction for LectureLens workers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lecturelens.config.settings import Settings
from lecturelens.core.agent_runtime import AgentControl
from lecturelens.core.clock import SessionClock
from lecturelens.core.redis_bus import RedisBus


class BaseAgent:
    """Base class implementing common lifecycle and control for background agents."""

    def __init__(
        self,
        settings: Settings,
        redis_bus: RedisBus,
        session_clock: SessionClock,
        session_id: str,
        component_name: str,
    ) -> None:
        self.settings = settings
        self.redis_bus = redis_bus
        self.session_clock = session_clock
        self.session_id = session_id
        self.component_name = component_name
        self.session_dir: Path = settings.create_session_directories(session_id)
        self.control = AgentControl(self.session_dir, component_name)
        self.is_running: bool = False

    async def run(self) -> None:
        """Execute agent lifecycle: setup -> process_loop -> teardown."""
        self.is_running = True
        self.control.write_heartbeat("running")
        try:
            await self.setup()
            await self.process_loop()
        finally:
            self.is_running = False
            await self.teardown()
            self.control.write_heartbeat("stopped")

    async def setup(self) -> None:
        """Initialize models, subscriptions, and directories."""
        pass

    async def process_loop(self) -> None:
        """Main processing loop executed while not stop_requested()."""
        pass

    async def teardown(self) -> None:
        """Clean up background tasks, flush queues, and release hardware resources."""
        pass

    def stop_requested(self) -> bool:
        """Check if external stop flag has been signaled."""
        return self.control.stop_requested()
