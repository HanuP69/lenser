import tempfile
import unittest
from pathlib import Path

from lecturelens.core.env_file import (
    read_env_value,
    remove_env_value,
    set_env_value
)


class TestEnvFile(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name) / ".env"
        self.addCleanup(self.folder.cleanup)

    def test_a_value_is_added(self) -> None:
        self.assertTrue(set_env_value("LECTURELENS_WINDOW_HANDLE", "1234", self.path))
        self.assertEqual(read_env_value("LECTURELENS_WINDOW_HANDLE", self.path), "1234")

    def test_an_existing_value_is_replaced(self) -> None:
        self.path.write_text("A=1\nLECTURELENS_WINDOW_HANDLE=99\nB=2\n", encoding="utf-8")
        set_env_value("LECTURELENS_WINDOW_HANDLE", "1234", self.path)
        self.assertEqual(read_env_value("LECTURELENS_WINDOW_HANDLE", self.path), "1234")
        # The rest of the file is kept.
        self.assertEqual(read_env_value("A", self.path), "1")
        self.assertEqual(read_env_value("B", self.path), "2")

    def test_a_commented_out_value_is_not_replaced(self) -> None:
        self.path.write_text("# LECTURELENS_WINDOW_HANDLE=99\n", encoding="utf-8")
        set_env_value("LECTURELENS_WINDOW_HANDLE", "1234", self.path)
        self.assertEqual(read_env_value("LECTURELENS_WINDOW_HANDLE", self.path), "1234")

    def test_a_value_is_removed(self) -> None:
        self.path.write_text("A=1\nLECTURELENS_WINDOW_HANDLE=99\n", encoding="utf-8")
        self.assertTrue(remove_env_value("LECTURELENS_WINDOW_HANDLE", self.path))
        self.assertIsNone(read_env_value("LECTURELENS_WINDOW_HANDLE", self.path))
        self.assertEqual(read_env_value("A", self.path), "1")

    def test_removing_a_value_that_is_not_there_does_nothing(self) -> None:
        self.path.write_text("A=1\n", encoding="utf-8")
        self.assertFalse(remove_env_value("MISSING", self.path))
        self.assertEqual(read_env_value("A", self.path), "1")

    def test_a_value_with_an_equal_sign_in_it_is_kept(self) -> None:
        self.path.write_text("A=b=c\n", encoding="utf-8")
        self.assertEqual(read_env_value("A", self.path), "b=c")


if __name__ == "__main__":
    unittest.main()
