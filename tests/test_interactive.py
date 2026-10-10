import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rich.console import Console

from lecturelens.cli import interactive
from lecturelens.config.settings import Settings


class TestInteractiveHelpers(unittest.TestCase):
    def test_filter_gemma4_models(self) -> None:
        raw_models = [
            "gemma4:e4b",
            "llama3.2:3b",
            "gemma-4-26b:latest",
            "qwen2.5:7b",
            "nomic-embed-text:latest"
        ]
        compatible = interactive.filter_gemma4_models(raw_models)
        self.assertEqual(compatible, ["gemma4:e4b", "gemma-4-26b:latest"])

    def test_filter_gemma4_models_empty(self) -> None:
        raw_models = ["llama3:latest", "phi3:mini"]
        compatible = interactive.filter_gemma4_models(raw_models)
        self.assertEqual(compatible, [])

    def test_mask_api_key(self) -> None:
        self.assertEqual(interactive.mask_api_key("short"), "****")
        masked = interactive.mask_api_key("sk-uno-1234567890abcdef")
        self.assertTrue(masked.startswith("sk-u"))
        self.assertTrue(masked.endswith("cdef"))
        self.assertIn("...", masked)

    def test_env_api_key_load_and_persist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertIsNone(interactive.load_env_api_key(env_file))
                interactive.persist_api_key_to_env("secret_uno_key_123", env_file)
                loaded = interactive.load_env_api_key(env_file)
                self.assertEqual(loaded, "secret_uno_key_123")
                interactive.persist_api_key_to_env("updated_uno_key_456", env_file)
                updated = interactive.load_env_api_key(env_file)
                self.assertEqual(updated, "updated_uno_key_456")

    def test_fetch_ollama_models_success(self) -> None:
        fake_payload = json.dumps({"models": [{"name": "gemma4:e4b"}, {"name": "nomic-embed-text"}]}).encode("utf-8")
        fake_response = mock.MagicMock()
        fake_response.status = 200
        fake_response.read.return_value = fake_payload
        fake_response.__enter__.return_value = fake_response

        with mock.patch("urllib.request.urlopen", return_value=fake_response):
            models = interactive.fetch_ollama_models("http://localhost:11434")
            self.assertEqual(models, ["gemma4:e4b", "nomic-embed-text"])

    def test_fetch_ollama_models_failure(self) -> None:
        with mock.patch("urllib.request.urlopen", side_effect=Exception("network down")):
            models = interactive.fetch_ollama_models("http://localhost:11434")
            self.assertEqual(models, [])


class TestInteractiveFlows(unittest.TestCase):
    def setUp(self) -> None:
        self.console = Console(record=True)
        self.settings = Settings()

    def test_configure_ollama_single_compatible(self) -> None:
        with mock.patch.object(interactive, "fetch_ollama_models", return_value=["gemma4:e4b", "nomic-embed-text"]):
            provider, model, llm_mode, primary = interactive.configure_ollama_flow(self.settings, self.console)
            self.assertEqual(provider, "gemma")
            self.assertEqual(model, "gemma4:e4b")
            self.assertEqual(llm_mode, "local")
            self.assertEqual(primary, "gemma")

    def test_configure_ollama_multiple_compatible(self) -> None:
        with mock.patch.object(interactive, "fetch_ollama_models", return_value=["gemma4:e4b", "gemma4:26b"]):
            with mock.patch("rich.prompt.Prompt.ask", return_value="2"):
                provider, model, llm_mode, primary = interactive.configure_ollama_flow(self.settings, self.console)
                self.assertEqual(provider, "gemma")
                self.assertEqual(model, "gemma4:26b")
                self.assertEqual(llm_mode, "local")

    def test_configure_unorouter_flow_new_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch.object(interactive, "load_env_api_key", return_value=None):
                    with mock.patch.object(interactive, "persist_api_key_to_env") as mock_persist:
                        with mock.patch("rich.prompt.Prompt.ask", return_value="my_new_uno_key"):
                            provider, model, llm_mode, primary = interactive.configure_unorouter_flow(self.console)
                            self.assertEqual(provider, "unorouter")
                            self.assertEqual(model, "gemma-4-26b:free")
                            self.assertEqual(llm_mode, "hybrid")
                            self.assertEqual(primary, "unorouter")
                            self.assertEqual(os.environ.get("UNOROUTER_API_KEY"), "my_new_uno_key")
                            mock_persist.assert_called_once_with("my_new_uno_key")

    def test_run_interactive_wizard_start_immediately(self) -> None:
        initial = {
            "session_id": "test_sess",
            "audio_source": "system",
            "no_camera": True,
            "no_screen": False,
            "audio_only": False,
            "course": None,
            "monitor_index": 1,
            "llm_mode": "local"
        }
        with mock.patch.object(interactive, "select_llm_provider", return_value=("unorouter", "gemma-4-26b:free", "hybrid", "unorouter")):
            with mock.patch("rich.prompt.Prompt.ask", side_effect=["1"]):
                result = interactive.run_interactive_wizard(initial, self.settings, self.console)
                self.assertIsNotNone(result)
                self.assertEqual(result["session_id"], "test_sess")
                self.assertEqual(result["provider"], "unorouter")
                self.assertEqual(result["model"], "gemma-4-26b:free")
                self.assertEqual(result["llm_mode"], "hybrid")

    def test_run_interactive_wizard_cancel(self) -> None:
        initial = {
            "session_id": "test_sess",
            "audio_source": "system",
            "no_camera": True,
            "no_screen": False,
            "audio_only": False,
            "course": None,
            "monitor_index": 1,
            "llm_mode": "local"
        }
        with mock.patch.object(interactive, "select_llm_provider", return_value=("gemma", "gemma4:e4b", "local", "gemma")):
            with mock.patch("rich.prompt.Prompt.ask", return_value="q"):
                result = interactive.run_interactive_wizard(initial, self.settings, self.console)
                self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
