import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lecturelens.cli import helpers
from lecturelens.cli import main as cli_main


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
            self.assertFalse(cli_main.agents_are_gone(session_dir))

    def test_a_session_with_running_agents_is_not_given_up_on(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            session_dir = Path(folder)
            heartbeat_dir = session_dir / "heartbeats"
            heartbeat_dir.mkdir(parents=True)
            (heartbeat_dir / "audio_agent.json").write_text(
                json.dumps({"agent": "audio_agent", "pid": os.getpid()}),
                encoding="utf-8"
            )
            self.assertFalse(cli_main.agents_are_gone(session_dir))

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
                self.assertTrue(cli_main.agents_are_gone(session_dir))
            else:
                self.assertTrue(cli_main.agents_are_gone(session_dir))


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
            with mock.patch.object(cli_main, "console", BrokenConsole()):
                cli_main.run_shutdown(BrokenMonitor(), session_dir, ["audio_agent"])
            self.assertTrue(flag_exists(session_dir, SESSION_FINISHED_FLAG_NAME))


if __name__ == "__main__":
    unittest.main()
