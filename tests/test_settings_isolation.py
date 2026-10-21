"""Check that the .env file of the project cannot influence the tests."""

import os
import unittest

from lecturelens.config.settings import Settings


class TestTheMachineDoesNotChangeTheTests(unittest.TestCase):
    def test_the_env_file_is_not_read(self) -> None:
        # The .env file of the developer has a language in it. The settings
        # must not see it.
        self.assertIsNone(Settings().whisper_language)
        self.assertFalse(Settings().whisper_lock_language)

    def test_the_key_of_the_developer_is_not_in_the_environment(self) -> None:
        for name in os.environ:
            self.assertFalse(name.startswith("LECTURELENS_"), name)

    def test_the_settings_still_work(self) -> None:
        settings = Settings()
        self.assertEqual(settings.screenshot_interval_seconds, 1.0)
        self.assertEqual(settings.monitor_index, 1)

    def test_the_output_is_the_output_of_the_project(self) -> None:
        """A relative path is resolved against the project.

        The same command wrote its notes somewhere else every time it
        was started from a different folder, and the notes of a session
        could not be found again.
        """
        settings = Settings()
        self.assertTrue(settings.output_dir.is_absolute())
        self.assertIn("virt", str(settings.output_dir))
        self.assertEqual(settings.output_dir.name, "output")

    def test_the_router_file_is_the_one_of_the_project(self) -> None:
        settings = Settings()
        self.assertTrue(settings.router_config_path.is_absolute())
        self.assertTrue(settings.router_config_path.exists())


if __name__ == "__main__":
    unittest.main()
