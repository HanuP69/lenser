import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lecturelens.cli import helpers

from lecturelens.cli.app import (
    run_shutdown,
)
from lecturelens.cli import app as cli_app
from lecturelens.cli.commands.session import (
    agents_are_gone,
)
from lecturelens.cli.commands.windows import (
    doctor_command,
)


class TestConsoleSafety(unittest.TestCase):
    def test_a_character_the_console_cannot_show_is_dropped(self) -> None:
        # The console of a Windows machine uses cp1252, which has no
        # characters for Urdu or Hindi.
        with mock.patch.object(sys, "stdout") as fake_stdout:
            fake_stdout.encoding = "cp1252"
            cleaned = helpers.safe_for_console("hello \u0628\u0633 \u2028 world")
        self.assertIn("hello", cleaned)
        self.assertNotIn("\u0628", cleaned)

    def test_text_that_the_console_can_show_is_kept(self) -> None:
        with mock.patch.object(sys, "stdout") as fake_stdout:
            fake_stdout.encoding = "utf-8"
            cleaned = helpers.safe_for_console("hello \u0628\u0633 world")
        self.assertIn("\u0628\u0633", cleaned)

    def test_the_length_is_limited(self) -> None:
        self.assertTrue(helpers.safe_for_console("abcdefgh", 3).startswith("abc"))

    def test_an_empty_text_stays_empty(self) -> None:
        self.assertEqual(helpers.safe_for_console(""), "")

    def test_the_snippet_of_a_lecture_in_urdu_is_usable(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            timeline = Path(folder) / "timeline.jsonl"
            entry = {
                "source_type": "transcript",
                "content": "\u0622\u062c \u06a9\u06cc \u0631\u06cc\u0686\u06cc\u0633 \u06a9\u06cc\u0627 \u06c1\u06d2"
            }
            timeline.write_text(json.dumps(entry) + "\n", encoding="utf-8")
            with mock.patch.object(sys, "stdout") as fake_stdout:
                fake_stdout.encoding = "cp1252"
                snippet = helpers.last_transcript_snippet(timeline)
        self.assertIsInstance(snippet, str)
        # It must be printable on the console that asked for it.
        snippet.encode("cp1252")


class TestAgentLiveness(unittest.TestCase):
    def test_a_session_without_heartbeats_is_not_given_up_on(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            session_dir = Path(folder)
            self.assertFalse(agents_are_gone(session_dir))

    def test_a_session_with_running_agents_is_not_given_up_on(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            session_dir = Path(folder)
            heartbeat_dir = session_dir / "heartbeats"
            heartbeat_dir.mkdir(parents=True)
            (heartbeat_dir / "audio_agent.json").write_text(
                json.dumps({"agent": "audio_agent", "pid": os.getpid()}),
                encoding="utf-8"
            )
            self.assertFalse(agents_are_gone(session_dir))

    def test_a_session_whose_agents_are_gone_is_given_up_on(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            session_dir = Path(folder)
            heartbeat_dir = session_dir / "heartbeats"
            heartbeat_dir.mkdir(parents=True)
            # A process id that is not running.
            (heartbeat_dir / "audio_agent.json").write_text(
                json.dumps({"agent": "audio_agent", "pid": 0x7FFFFFF0}),
                encoding="utf-8"
            )
            if os.name == "nt":
                self.assertTrue(agents_are_gone(session_dir))
            else:
                self.assertTrue(agents_are_gone(session_dir))


class TestInterruptedShutdown(unittest.TestCase):
    def test_the_phases_still_run_after_a_second_interrupt(self) -> None:
        """A second Ctrl+C must not throw the queue of the session away.

        The user presses the key again while the session is closing,
        because the closing takes a while when the provider is on a free
        tier. The whole shutdown used to be thrown away, so the events
        that were waiting were never written into the notes.
        """
        from lecturelens.core.process_manager import AgentProcessManager

        class FakeProcess:
            def poll(self) -> int:
                return 0

        manager = AgentProcessManager()
        manager.processes = {
            "sync_agent": FakeProcess(),
            "structuring_agent": FakeProcess(),
        }
        manager.started_at = {"sync_agent": 0.0, "structuring_agent": 0.0}
        manager.next_restart_at = {}

        flags_written: list[str] = []

        def fake_set_flag(session_dir: Path, flag_name: str) -> None:
            flags_written.append(flag_name)

        def interrupted_wait(*args: object, **kwargs: object) -> bool:
            raise KeyboardInterrupt

        from lecturelens.core import process_manager

        with mock.patch.object(process_manager, "set_flag", side_effect=fake_set_flag):
            with mock.patch.object(process_manager.AgentProcessManager, "wait_for_exit", side_effect=interrupted_wait):
                manager.run_shutdown_sequence(
                    Path("."),
                    [
                        (["sync_agent"], "sync_stop.flag", 60.0),
                        (["structuring_agent"], "finalize.flag", 300.0),
                    ]
                )

        # The phase that was interrupted and the phase after it both ran.
        self.assertIn("sync_stop.flag", flags_written)
        self.assertIn("finalize.flag", flags_written)

    def test_the_session_is_marked_as_finished_after_an_interrupt(self) -> None:
        from lecturelens.core.agent_runtime import SESSION_FINISHED_FLAG_NAME, flag_exists

        class BrokenConsole:
            def __enter__(self) -> "BrokenConsole":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def print(self, *args: object, **kwargs: object) -> None:
                raise UnicodeEncodeError("charmap", "\u0628", 0, 1, "not encodable")

        class InterruptedMonitor:
            def run_shutdown_sequence(self, *args: object, **kwargs: object) -> dict:
                raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as folder:
            session_dir = Path(folder)
            (session_dir / "control").mkdir(parents=True, exist_ok=True)
            with mock.patch.object(cli_app, "console", BrokenConsole()):
                run_shutdown(InterruptedMonitor(), session_dir, ["audio_agent"])
            self.assertTrue(flag_exists(session_dir, SESSION_FINISHED_FLAG_NAME))


class TestShutdownSurvivesABrokenConsole(unittest.TestCase):
    def test_the_session_is_marked_as_finished_even_when_the_console_fails(self) -> None:
        """A console that cannot show a character must not keep the session
        from finishing.

        The shutdown is the cleanup path. When it failed the agents were
        left running and the session was never marked as finished, so the
        stop command waited for the whole timeout for nothing.
        """
        from lecturelens.core.agent_runtime import SESSION_FINISHED_FLAG_NAME, flag_exists

        class BrokenConsole:
            def __enter__(self) -> "BrokenConsole":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def print(self, *args: object, **kwargs: object) -> None:
                raise UnicodeEncodeError("charmap", "\u0628", 0, 1, "not encodable")

        class BrokenMonitor:
            def run_shutdown_sequence(self, *args: object, **kwargs: object) -> dict:
                raise RuntimeError("the console failed")

        with tempfile.TemporaryDirectory() as folder:
            session_dir = Path(folder)
            (session_dir / "control").mkdir(parents=True, exist_ok=True)
            with mock.patch.object(cli_app, "console", BrokenConsole()):
                run_shutdown(BrokenMonitor(), session_dir, ["audio_agent"])
            self.assertTrue(flag_exists(session_dir, SESSION_FINISHED_FLAG_NAME))


class TestAgentFailureReason(unittest.TestCase):
    def test_a_missing_module_is_found_in_the_log(self) -> None:
        """The reason an agent died was only in its log.

        The capture went on and everything was recorded, but the notes
        were never written, because the agent that writes them was dead.
        """
        import tempfile
        from pathlib import Path

        from lecturelens.cli.helpers import agent_failure_reason

        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "structuring_agent.log"
            log.write_text(
                "Traceback (most recent call last):\n"
                "  File \"x.py\", line 12, in <module>\n"
                "    from langgraph.graph import END\n"
                "ModuleNotFoundError: No module named 'langgraph'\n",
                encoding="utf-8"
            )
            reason = agent_failure_reason(log)
        self.assertIsNotNone(reason)
        self.assertIn("langgraph", reason)

    def test_a_log_without_an_error_has_no_reason(self) -> None:
        import tempfile
        from pathlib import Path

        from lecturelens.cli.helpers import agent_failure_reason

        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "audio_agent.log"
            log.write_text("2026-10-21 21:00:00 INFO everything is fine\n", encoding="utf-8")
            self.assertIsNone(agent_failure_reason(log))


class TestInstallHint(unittest.TestCase):
    def test_the_hint_says_how_to_install(self) -> None:
        """A folder made with uv has no pip in it.

        The command that people know then does nothing at all, or worse,
        it installs into the python of the machine and the folder stays
        as it was.
        """
        from lecturelens.cli.helpers import install_hint

        hint = install_hint()
        self.assertIn("install", hint)
        self.assertIn(".[full]", hint)

    def test_a_folder_without_pip_says_uv(self) -> None:
        import sys
        from unittest import mock

        from lecturelens.cli import helpers

        with mock.patch.object(sys, "prefix", "C:\\somewhere\\.venv"):
            with mock.patch.object(sys, "base_prefix", "C:\\Python"):
                with mock.patch(
                    "importlib.util.find_spec", return_value=None
                ):
                    hint = helpers.install_hint()
        self.assertIn("uv pip install", hint)


class TestAgentDependencies(unittest.TestCase):
    def test_the_agents_that_are_there_are_found(self) -> None:
        from lecturelens.cli.helpers import check_agent_dependencies

        problems = check_agent_dependencies(["audio_agent", "screen_agent", "sync_agent"])
        self.assertEqual(problems, [])

    def test_an_agent_that_is_not_there_is_reported(self) -> None:
        from lecturelens.cli.helpers import check_agent_dependencies

        problems = check_agent_dependencies(["not_an_agent"])
        self.assertEqual(len(problems), 1)
        self.assertIn("not_an_agent", problems[0])


class TestDependencyVersions(unittest.TestCase):
    def test_the_requirement_lines_are_read(self) -> None:
        from lecturelens.cli.helpers import _read_requirement_lines

        text = (
            "[project]\n"
            "dependencies = [\n"
            '    "pydantic>=2.0",\n'
            "    # a comment\n"
            '    "redis>=5.0"\n'
            "]\n"
            "\n"
            "[project.optional-dependencies]\n"
            "full = [\n"
            '    "mss>=9.0"\n'
            "]\n"
        )
        lines = _read_requirement_lines(text)
        self.assertIn("pydantic>=2.0", lines)
        self.assertIn("redis>=5.0", lines)
        self.assertIn("mss>=9.0", lines)

    def test_the_installed_versions_fit(self) -> None:
        """The versions of the langchain packages have to agree.

        A machine with an old langgraph and a new langchain-core could
        not start the agent that writes the notes, and the reason was a
        confusing message about a module that is installed.
        """
        from lecturelens.cli.helpers import check_dependency_versions

        self.assertEqual(check_dependency_versions(), [])


class TestDoctorCommand(unittest.TestCase):
    def test_the_doctor_command_runs(self) -> None:
        import typer


        try:
            doctor_command()
        except typer.Exit as error:
            self.assertNotEqual(error.exit_code, 1)


if __name__ == "__main__":
    unittest.main()
