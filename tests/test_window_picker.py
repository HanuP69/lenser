import unittest
from typing import Any
from unittest import mock

from lecturelens.cli import window_picker
from lecturelens.core.window_capture import WindowInfo



from lecturelens.cli.commands.session import (
    _remember_window,
)
def make_window(handle: int, title: str) -> WindowInfo:
    return WindowInfo(
        handle=handle,
        title=title,
        left=0,
        top=0,
        width=800,
        height=600,
        process_id=1000 + handle,
        is_minimized=False
    )


class FakeConsole:
    """A console that answers the question with a fixed answer."""

    def __init__(self, answer: str = "") -> None:
        self.answer = answer
        self.printed: list[str] = []

    def print(self, *args: object, **kwargs: object) -> None:
        self.printed.append(" ".join(str(item) for item in args))

    def input(self, prompt: str = "") -> str:
        return self.answer


WINDOWS = [make_window(10, "Redis Talk"), make_window(11, "Other App")]


class TestWindowPicker(unittest.TestCase):
    def test_the_number_picks_the_window(self) -> None:
        console = FakeConsole(answer="2")
        with mock.patch("lecturelens.core.window_capture.list_windows", return_value=WINDOWS):
            window = window_picker.prompt_for_window(console)
        self.assertIsNotNone(window)
        self.assertEqual(window.handle, 11)

    def test_an_empty_answer_records_the_whole_screen(self) -> None:
        console = FakeConsole(answer="")
        with mock.patch("lecturelens.core.window_capture.list_windows", return_value=WINDOWS):
            window = window_picker.prompt_for_window(console)
        self.assertIsNone(window)

    def test_a_number_that_does_not_exist_is_asked_again(self) -> None:
        """A wrong number is a typo, not a choice.

        The command used to fall through to recording the whole screen, so
        a mistyped number recorded something the user never asked for. The
        question is asked again instead, and the loop only ends when an
        answer that fits is given, the whole screen is chosen with an
        empty answer, or the input ends.
        """
        console = FakeConsole(answer="")
        console.answers = ["99", "1"]
        with mock.patch("lecturelens.core.window_capture.list_windows", return_value=WINDOWS), \
                mock.patch.object(console, "input", side_effect=lambda prompt="": console.answers.pop(0)):
            window = window_picker.prompt_for_window(console)
        self.assertIsNotNone(window)
        self.assertEqual(window.handle, WINDOWS[0].handle)
        # The typo was answered with an explanation, not with a decision.
        self.assertTrue(
            any("no window number 99" in line for line in console.printed)
        )

    def test_a_title_picks_the_window(self) -> None:
        console = FakeConsole(answer="Redis Talk")
        with mock.patch("lecturelens.core.window_capture.list_windows", return_value=WINDOWS):
            window = window_picker.prompt_for_window(console)
        self.assertIsNotNone(window)
        self.assertEqual(window.handle, 10)

    def test_the_query_is_used_without_asking(self) -> None:
        console = FakeConsole(answer="")
        with mock.patch("lecturelens.core.window_capture.list_windows", return_value=WINDOWS):
            target, handle, title = window_picker.resolve_capture_target(
                console, window_query="Redis Talk", ask=True
            )
        self.assertEqual(target, "window")
        self.assertEqual(handle, 10)
        self.assertEqual(title, "Redis Talk")

    def test_a_configured_window_is_used_without_asking(self) -> None:
        console = FakeConsole(answer="")
        with mock.patch("lecturelens.core.window_capture.is_window_alive", return_value=True):
            target, handle, title = window_picker.resolve_capture_target(
                console, configured_handle=10, configured_title="Redis Talk", ask=True
            )
        self.assertEqual(target, "window")
        self.assertEqual(handle, 10)
        self.assertEqual(title, "Redis Talk")

    def test_nothing_is_asked_when_the_question_is_not_allowed(self) -> None:
        console = FakeConsole(answer="1")
        target, handle, title = window_picker.resolve_capture_target(console, ask=False)
        self.assertEqual(target, "screen")
        self.assertIsNone(handle)
        self.assertIsNone(title)

    def test_a_window_that_matches_nothing_falls_back_to_the_screen(self) -> None:
        console = FakeConsole(answer="")
        with mock.patch("lecturelens.core.window_capture.list_windows", return_value=WINDOWS):
            target, handle, title = window_picker.resolve_capture_target(
                console, window_query="does not exist", ask=True
            )
        self.assertEqual(target, "screen")
        self.assertIsNone(handle)


class TestRememberingTheWindow(unittest.TestCase):
    def test_the_chosen_window_is_kept_for_the_next_run(self) -> None:
        """The question is only asked the first time."""
        import tempfile
        from pathlib import Path

        from lecturelens.core.env_file import read_env_value

        with tempfile.TemporaryDirectory() as folder:
            env_path = Path(folder) / ".env"

            def remember(name: str, value: str, path: Any = None) -> bool:
                with open(env_path, "a", encoding="utf-8") as handle:
                    handle.write(f"{name}={value}\n")
                return True

            with mock.patch(
                "lecturelens.core.env_file.set_env_value", side_effect=remember
            ):
                _remember_window(4321, "Redis Talk")

            self.assertEqual(read_env_value("LECTURELENS_WINDOW_HANDLE", env_path), "4321")
            self.assertEqual(read_env_value("LECTURELENS_WINDOW_TITLE", env_path), "Redis Talk")
            self.assertEqual(read_env_value("LECTURELENS_CAPTURE_TARGET", env_path), "window")


if __name__ == "__main__":
    unittest.main()
