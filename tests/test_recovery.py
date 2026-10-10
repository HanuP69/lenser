import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

from lecturelens.core.agent_runtime import flag_exists, set_flag
from lecturelens.core.recovery import AgentMonitor

FLAG_WAITER = (
    "import sys, time, os\n"
    "flag = sys.argv[1]\n"
    "while not os.path.exists(flag):\n"
    "    time.sleep(0.05)\n"
)
CRASHER = "import sys; sys.exit(3)"
STUBBORN = "import time\nwhile True:\n    time.sleep(0.1)\n"


def python_command(code: str, *args: str) -> list[str]:
    return [sys.executable, "-c", code, *args]


class TestRestartBackoff(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.monitor = AgentMonitor(
            base_backoff_seconds=1.0,
            max_backoff_seconds=8.0,
            max_consecutive_failures=4,
            stable_runtime_seconds=10**9
        )

    def tearDown(self) -> None:
        self.monitor.stop_all()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def wait_until_failed(self, name: str) -> None:
        for _ in range(100):
            if self.monitor.check_health().get(name, "").startswith("FAILED"):
                return
            time.sleep(0.05)
        self.fail("agent never exited")

    def test_restart_waits_for_exponential_backoff(self) -> None:
        self.monitor.register_agent("crasher", python_command(CRASHER), log_file_path=Path(self.temp_dir.name) / "c.log")
        self.monitor.start_agent("crasher")
        self.wait_until_failed("crasher")

        now = time.monotonic()
        self.assertEqual(self.monitor.restart_failed_agents(now), [])
        self.assertEqual(self.monitor.restart_failed_agents(now + 0.5), [])
        self.assertEqual(self.monitor.restart_failed_agents(now + 1.1), ["crasher"])

        self.wait_until_failed("crasher")
        later = now + 2.0
        self.assertEqual(self.monitor.restart_failed_agents(later), [])
        self.assertEqual(self.monitor.restart_failed_agents(later + 1.5), [])
        self.assertEqual(self.monitor.restart_failed_agents(later + 2.1), ["crasher"])

    def test_agent_is_abandoned_after_repeated_failures(self) -> None:
        self.monitor.register_agent("crasher", python_command(CRASHER))
        self.monitor.start_agent("crasher")
        clock = time.monotonic()
        restarts = 0
        for _ in range(12):
            self.wait_until_failed("crasher")
            clock += 100.0
            self.monitor.restart_failed_agents(clock)
            clock += 100.0
            restarted = self.monitor.restart_failed_agents(clock)
            restarts += len(restarted)
            if "crasher" in self.monitor.gave_up:
                break
        self.assertIn("crasher", self.monitor.gave_up)
        self.assertLessEqual(restarts, 4)
        self.assertEqual(self.monitor.check_health()["crasher"], "GAVE_UP")

    def test_stable_run_resets_failure_count(self) -> None:
        self.monitor.stable_runtime_seconds = 0.0
        self.monitor.register_agent("crasher", python_command(CRASHER))
        self.monitor.start_agent("crasher")
        clock = time.monotonic()
        for _ in range(6):
            self.wait_until_failed("crasher")
            clock += 100.0
            self.monitor.restart_failed_agents(clock)
            clock += 100.0
            self.monitor.restart_failed_agents(clock)
        self.assertNotIn("crasher", self.monitor.gave_up)

    def test_log_handles_do_not_accumulate_across_restarts(self) -> None:
        log_path = Path(self.temp_dir.name) / "agent.log"
        self.monitor.register_agent("crasher", python_command(CRASHER), log_file_path=log_path)
        for _ in range(5):
            self.monitor.start_agent("crasher")
        self.assertEqual(len(self.monitor.log_handles), 1)

    def test_no_restarts_during_shutdown(self) -> None:
        self.monitor.register_agent("crasher", python_command(CRASHER))
        self.monitor.start_agent("crasher")
        self.wait_until_failed("crasher")
        self.monitor.shutting_down = True
        self.assertEqual(self.monitor.restart_failed_agents(10**9), [])


class TestShutdownSequence(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.session_dir = Path(self.temp_dir.name)
        (self.session_dir / "control").mkdir()
        self.monitor = AgentMonitor()

    def tearDown(self) -> None:
        self.monitor.stop_all()
        self.temp_dir.cleanup()

    def flag_path(self, name: str) -> str:
        return str(self.session_dir / "control" / name)

    def test_phases_run_in_order_and_agents_exit_cleanly(self) -> None:
        for name, flag in (("audio_agent", "stop.flag"), ("sync_agent", "stop_sync.flag"), ("structuring_agent", "finalize.flag")):
            self.monitor.register_agent(name, python_command(FLAG_WAITER, self.flag_path(flag)))
            self.monitor.start_agent(name)
        time.sleep(0.3)

        order: list[str] = []
        original_wait = self.monitor.wait_for_exit

        def recording_wait(names: list[str], timeout_seconds: float) -> bool:
            order.append(",".join(names))
            return original_wait(names, timeout_seconds)

        self.monitor.wait_for_exit = recording_wait
        outcomes = self.monitor.run_shutdown_sequence(
            self.session_dir,
            [(["audio_agent"], "stop.flag", 10.0), (["sync_agent"], "stop_sync.flag", 10.0), (["structuring_agent"], "finalize.flag", 10.0)]
        )
        self.assertEqual(order, ["audio_agent", "sync_agent", "structuring_agent"])
        self.assertEqual(outcomes, {"audio_agent": True, "sync_agent": True, "structuring_agent": True})
        for process in self.monitor.processes.values():
            self.assertEqual(process.returncode, 0)

    def test_later_phase_flag_is_not_set_early(self) -> None:
        self.monitor.register_agent("audio_agent", python_command(FLAG_WAITER, self.flag_path("stop.flag")))
        self.monitor.register_agent("sync_agent", python_command(FLAG_WAITER, self.flag_path("stop_sync.flag")))
        self.monitor.start_agent("audio_agent")
        self.monitor.start_agent("sync_agent")
        time.sleep(0.3)
        seen: dict[str, bool] = {}
        original_wait = self.monitor.wait_for_exit

        def spying_wait(names: list[str], timeout_seconds: float) -> bool:
            if names == ["audio_agent"]:
                seen["sync_flag_during_capture_phase"] = flag_exists(self.session_dir, "stop_sync.flag")
            return original_wait(names, timeout_seconds)

        self.monitor.wait_for_exit = spying_wait
        self.monitor.run_shutdown_sequence(
            self.session_dir, [(["audio_agent"], "stop.flag", 10.0), (["sync_agent"], "stop_sync.flag", 10.0)]
        )
        self.assertFalse(seen["sync_flag_during_capture_phase"])

    def test_agent_ignoring_flag_is_terminated(self) -> None:
        self.monitor.register_agent("stubborn", python_command(STUBBORN))
        self.monitor.start_agent("stubborn")
        time.sleep(0.2)
        started = time.monotonic()
        outcomes = self.monitor.run_shutdown_sequence(self.session_dir, [(["stubborn"], "stop.flag", 1.0)])
        self.assertFalse(outcomes["stubborn"])
        self.assertIsNotNone(self.monitor.processes["stubborn"].poll())
        self.assertLess(time.monotonic() - started, 15.0)


if __name__ == "__main__":
    unittest.main()
