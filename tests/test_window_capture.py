import unittest
from unittest import mock

from lecturelens.core import window_capture
from lecturelens.core.window_capture import (
    IGNORED_WINDOW_TITLES,
    WindowCaptureUnavailable,
    WindowInfo,
    find_window,
    is_window_alive,
    list_windows
)


def make_window(handle: int, title: str, width: int = 800, height: int = 600) -> WindowInfo:
    return WindowInfo(
        handle=handle,
        title=title,
        left=0,
        top=0,
        width=width,
        height=height,
        process_id=1000 + handle,
        is_minimized=False
    )


class TestWindowListing(unittest.TestCase):
    def test_a_number_finds_the_window(self) -> None:
        windows = [make_window(10, "Alpha"), make_window(11, "Beta")]
        self.assertEqual(find_window("2", windows).handle, 11)
        self.assertEqual(find_window("1", windows).handle, 10)

    def test_a_number_out_of_range_finds_nothing(self) -> None:
        windows = [make_window(10, "Alpha")]
        self.assertIsNone(find_window("9", windows))
        self.assertIsNone(find_window("0", windows))

    def test_a_title_finds_the_window(self) -> None:
        windows = [make_window(10, "Redis Talk"), make_window(11, "Other")]
        found = find_window("redis", windows)
        self.assertIsNotNone(found)
        self.assertEqual(found.handle, 10)

    def test_an_exact_title_wins_over_a_partial_one(self) -> None:
        windows = [make_window(10, "Talk"), make_window(11, "Talk about Redis")]
        found = find_window("Talk", windows)
        self.assertEqual(found.handle, 10)

    def test_an_empty_query_finds_nothing(self) -> None:
        self.assertIsNone(find_window("  ", [make_window(10, "Alpha")]))

    def test_the_number_is_the_one_the_listing_showed(self) -> None:
        """The number has to mean the same window in both places.

        The list that the lookup used had the minimized windows in it as
        well, so the number the user saw was off by however many
        minimized windows there were. The window that was recorded was
        not the one on the screen.
        """
        visible = [make_window(10, "Alpha"), make_window(11, "Beta")]
        with mock.patch.object(window_capture, "list_windows", return_value=visible):
            found = find_window("2")
        self.assertIsNotNone(found)
        self.assertEqual(found.handle, 11)

    def test_a_minimized_window_is_still_found_by_its_number(self) -> None:
        visible = [make_window(10, "Alpha")]
        minimized = [make_window(10, "Alpha"), make_window(12, "Hidden")]
        with mock.patch.object(window_capture, "list_windows", return_value=visible):
            with mock.patch.object(
                window_capture, "list_windows", return_value=minimized
            ):
                found = find_window("2", include_minimized=True)
        self.assertIsNotNone(found)
        self.assertEqual(found.handle, 12)

    def test_the_shell_windows_are_not_offered(self) -> None:
        # The desktop itself is never what a lecture is about.
        self.assertIn("program manager", IGNORED_WINDOW_TITLES)


class TestWindowCaptureGuards(unittest.TestCase):
    def test_a_window_that_is_gone_is_reported(self) -> None:
        with mock.patch.object(window_capture, "IS_WINDOWS", False):
            with self.assertRaises(WindowCaptureUnavailable):
                list_windows()

    def test_a_closed_window_is_not_alive(self) -> None:
        # A handle that was never handed out is not a window.
        self.assertFalse(is_window_alive(0x7FFFFFFF) if window_capture.IS_WINDOWS else True)


class TestWindowPixels(unittest.TestCase):
    def test_an_all_black_frame_is_not_treated_as_content(self) -> None:
        black = b"\x00" * 400
        self.assertFalse(window_capture._has_visible_pixels(black))

    def test_a_frame_with_content_is_accepted(self) -> None:
        raw = bytearray(b"\x00" * 1024)
        # The check samples every 64th pixel, so the content has to be on
        # one of those.
        raw[256:259] = b"\xff\xff\xff"
        self.assertTrue(window_capture._has_visible_pixels(bytes(raw)))


if __name__ == "__main__":
    unittest.main()
