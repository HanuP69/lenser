import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import typer

from lecturelens.cli import helpers, main as cli_main
from lecturelens.config.settings import Settings
from lecturelens.core import agent_runtime
from lecturelens.core.agent_runtime import AgentControl, read_heartbeats
from tests.support import index_by


class TestHelpers(unittest.TestCase):
    def test_option_validation(self) -> None:
        helpers.validate_start_options("mic", "local", "local")
        with self.assertRaises(ValueError):
            helpers.validate_start_options("micc", "local", "local")
        with self.assertRaises(ValueError):
            helpers.validate_start_options("mic", "everywhere", "local")
        with self.assertRaises(ValueError):
            helpers.validate_start_options("mic", "local", "cluster")
        with self.assertRaises(ValueError):
            helpers.validate_export_format("docx")
        self.assertEqual(helpers.validate_export_format("pdf"), "pdf")

    def test_format_elapsed(self) -> None:
        self.assertEqual(helpers.format_elapsed(3725.9), "01:02:05")
        self.assertEqual(helpers.format_elapsed(-5), "00:00:00")

    def test_heartbeat_classification(self) -> None:
        now = time.time()
        self.assertEqual(helpers.classify_heartbeat(None, now), "NO DATA")
        self.assertEqual(helpers.classify_heartbeat({"state": "live", "time": now}, now), "LIVE")
        self.assertEqual(helpers.classify_heartbeat({"state": "paused", "time": now}, now), "PAUSED")
        self.assertEqual(helpers.classify_heartbeat({"state": "live", "time": now - 60}, now), "ERROR")
        self.assertEqual(helpers.classify_heartbeat({"state": "stopped", "time": now - 60}, now), "STOPPED")

    def test_agent_rows_combine_process_and_heartbeat(self) -> None:
        now = time.time()
        rows = helpers.build_agent_rows(
            {"audio_agent": "RUNNING", "screen_agent": "FAILED_EXIT_1"},
            {"audio_agent": {"state": "live", "time": now, "stats": {"transcripts_published": 4}},
             "screen_agent": {"state": "live", "time": now, "stats": {}}},
            now
        )
        by_name = index_by(rows, "agent")
        self.assertEqual(by_name["audio_agent"]["status"], "LIVE")
        self.assertIn("transcripts_published=4", by_name["audio_agent"]["stats"])
        self.assertEqual(by_name["screen_agent"]["status"], "ERROR")

    def test_tail_and_snippet_and_line_counter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "timeline.jsonl"
            self.assertIsNone(helpers.tail_last_line(path))
            self.assertEqual(helpers.last_transcript_snippet(path), "")
            counter = helpers.LineCounter(path)
            self.assertEqual(counter.refresh(), 0)
            lines = [
                {"source_type": "transcript", "content": "first words"},
                {"source_type": "slide", "content": "slide text"},
                {"source_type": "transcript", "content": "x" * 300}
            ]
            encoded_lines: list[str] = []
            for line in lines:
                encoded_lines.append(json.dumps(line))
            path.write_text("\n".join(encoded_lines) + "\n", encoding="utf-8")
            self.assertEqual(counter.refresh(), 3)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"source_type": "board", "content": "b"}) + "\n")
            self.assertEqual(counter.refresh(), 4)
            self.assertTrue(helpers.last_transcript_snippet(path).endswith("..."))
            self.assertIn("board", helpers.tail_last_line(path))
            path.write_text("", encoding="utf-8")
            self.assertEqual(counter.refresh(), 0)

    def test_low_disk_warning(self) -> None:
        self.assertIsNone(helpers.format_low_disk_warning(10 * 1024 ** 3))
        message = helpers.format_low_disk_warning(1024 ** 3)
        self.assertIn("1.0 GB", message)
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertGreater(helpers.get_free_disk_bytes(Path(temp_dir) / "not" / "yet" / "created"), 0)

    def test_find_latest_session_ignores_internal_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            (output / "_courses").mkdir()
            (output / "old").mkdir()
            (output / "old" / "session_meta.json").write_text("{}")
            time.sleep(0.02)
            (output / "new").mkdir()
            (output / "new" / "session_meta.json").write_text("{}")
            (output / "no_meta").mkdir()
            self.assertEqual(helpers.find_latest_session(output), "new")
            self.assertIsNone(helpers.find_latest_session(output / "missing"))

    def test_format_file_size(self) -> None:
        self.assertEqual(helpers.format_file_size(500), "500 B")
        self.assertEqual(helpers.format_file_size(2048), "2.0 KB")
        self.assertEqual(helpers.format_file_size(5 * 1024 * 1024), "5.0 MB")

    def test_count_file_lines_and_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.txt"
            file_path.write_text("line1\nline2\nline3\n", encoding="utf-8")
            self.assertEqual(helpers.count_file_lines(file_path), 3)
            self.assertEqual(helpers.count_file_lines(Path(temp_dir) / "missing.txt"), 0)

            img_dir = Path(temp_dir) / "raw" / "screen"
            img_dir.mkdir(parents=True)
            (img_dir / "frame_001.jpg").write_bytes(b"dummy_image_bytes")
            count, size = helpers.count_images_in_directory(img_dir)
            self.assertEqual(count, 1)
            self.assertGreater(size, 0)

    def test_build_session_output_table_and_display(self) -> None:
        from rich.console import Console
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir) / "test_sess"
            session_dir.mkdir(parents=True)
            (session_dir / "notes.md").write_text("# Notes\n- item 1\n", encoding="utf-8")
            (session_dir / "timeline.jsonl").write_text('{"event": 1}\n{"event": 2}\n', encoding="utf-8")
            (session_dir / "session_meta.json").write_text('{"runs": []}', encoding="utf-8")

            table = helpers.build_session_output_table(session_dir)
            self.assertIsNotNone(table)

            console = Console(record=True)
            helpers.display_session_output_summary(session_dir, console)
            output = console.export_text()
            self.assertIn("notes.md", output)
            self.assertIn("timeline.jsonl", output)
            self.assertIn("test_sess", output)


class TestRuntimeControl(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.session_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_flags_and_pause_state(self) -> None:
        control = AgentControl(self.session_dir, "audio_agent")
        self.assertFalse(control.stop_requested())
        self.assertFalse(control.is_paused())
        agent_runtime.set_flag(self.session_dir, "stop.flag")
        agent_runtime.set_flag(self.session_dir, agent_runtime.PAUSE_FLAG_NAME)
        self.assertTrue(control.stop_requested())
        self.assertTrue(control.is_paused())
        agent_runtime.clear_all_flags(self.session_dir)
        self.assertFalse(control.stop_requested())

    def test_each_agent_uses_its_own_stop_flag(self) -> None:
        sync_control = AgentControl(self.session_dir, "sync_agent", agent_runtime.SYNC_STOP_FLAG_NAME)
        agent_runtime.set_flag(self.session_dir, "stop.flag")
        self.assertFalse(sync_control.stop_requested())
        agent_runtime.set_flag(self.session_dir, agent_runtime.SYNC_STOP_FLAG_NAME)
        self.assertTrue(sync_control.stop_requested())

    def test_heartbeat_round_trip(self) -> None:
        (self.session_dir / "heartbeats").mkdir()
        control = AgentControl(self.session_dir, "audio_agent")
        control.increment_stat("transcripts_published", 2)
        control.write_heartbeat("live")
        beats = read_heartbeats(self.session_dir)
        self.assertEqual(beats["audio_agent"]["state"], "live")
        self.assertEqual(beats["audio_agent"]["stats"]["transcripts_published"], 2)
        self.assertEqual(beats["audio_agent"]["pid"], os.getpid())

    def test_corrupt_heartbeat_is_ignored(self) -> None:
        (self.session_dir / "heartbeats").mkdir()
        (self.session_dir / "heartbeats" / "bad.json").write_text("{oops")
        self.assertEqual(read_heartbeats(self.session_dir), {})

    def test_runtime_context_uses_session_argument_and_validates(self) -> None:
        with mock.patch.dict(os.environ, {"LECTURELENS_OUTPUT_DIR": self.temp_dir.name}):
            namespace = agent_runtime.parse_runtime_arguments(["--session", "lecture1", "--audio-source", "both", "--unknown", "x"])
            self.assertEqual(namespace.session, "lecture1")
            context = agent_runtime.build_runtime_context(namespace)
            self.assertEqual(context.session_id, "lecture1")
            self.assertEqual(context.settings.audio_source, "both")
            self.assertTrue((Path(self.temp_dir.name) / "lecture1" / "clock.json").exists())
            bad = agent_runtime.parse_runtime_arguments(["--session", "../escape"])
            with self.assertRaises(ValueError):
                agent_runtime.build_runtime_context(bad)


class TestCliCommands(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output = Path(self.temp_dir.name)
        self.environment = mock.patch.dict(os.environ, {"LECTURELENS_OUTPUT_DIR": self.temp_dir.name})
        self.environment.start()
        self.session_dir = self.output / "lec1"
        for sub in ("control", "heartbeats", "logs"):
            (self.session_dir / sub).mkdir(parents=True)
        (self.session_dir / "session_meta.json").write_text(json.dumps({"runs": []}))
        (self.session_dir / "notes.md").write_text("### 10:00:00\n\nBinary search is **fast**.\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.environment.stop()
        self.temp_dir.cleanup()

    def test_plan_agents(self) -> None:
        full = cli_main.plan_agents("local", False, True, False)
        self.assertEqual(sorted(full), ["audio_agent", "screen_agent", "structuring_agent", "sync_agent"])
        self.assertNotIn("screen_agent", cli_main.plan_agents("local", False, True, True))
        self.assertEqual(
            sorted(cli_main.plan_agents("local", True, True, False)),
            ["audio_agent", "structuring_agent", "sync_agent"]
        )
        self.assertIn("gateway", cli_main.plan_agents("server", True, True, False))

    def test_environment_passes_absolute_paths_and_course(self) -> None:
        settings = Settings()
        environment = cli_main.build_agent_environment(settings, "lec1", "system", 2, "hybrid", "algorithms")
        self.assertTrue(Path(environment["LECTURELENS_OUTPUT_DIR"]).is_absolute())
        self.assertTrue(Path(environment["LECTURELENS_ROUTER_CONFIG_PATH"]).is_absolute())
        self.assertEqual(environment["LECTURELENS_AUDIO_SOURCE"], "system")
        self.assertEqual(environment["LECTURELENS_LLM_MODE"], "hybrid")
        self.assertEqual(environment["LECTURELENS_COURSE_NAME"], "algorithms")
        self.assertEqual(environment["LECTURELENS_MONITOR_INDEX"], "2")

    def test_environment_passes_unorouter_api_key(self) -> None:
        settings = Settings()
        with mock.patch.dict(os.environ, {"UNOROUTER_API_KEY": "test_uno_key"}):
            environment = cli_main.build_agent_environment(settings, "lec1", "system", 1, "hybrid", None)
            self.assertEqual(environment["UNOROUTER_API_KEY"], "test_uno_key")
            self.assertEqual(environment["LECTURELENS_UNOROUTER_API_KEY"], "test_uno_key")

    def test_module_names(self) -> None:
        self.assertEqual(cli_main.module_for_agent("gateway"), "lecturelens.gateway.server")
        self.assertEqual(cli_main.module_for_agent("audio_agent"), "lecturelens.agents.audio_agent")

    def test_pause_and_resume_toggle_flag(self) -> None:
        cli_main.pause(session="lec1")
        self.assertTrue(agent_runtime.flag_exists(self.session_dir, agent_runtime.PAUSE_FLAG_NAME))
        cli_main.resume(session="lec1")
        self.assertFalse(agent_runtime.flag_exists(self.session_dir, agent_runtime.PAUSE_FLAG_NAME))

    def test_stop_sets_shutdown_request_without_waiting(self) -> None:
        cli_main.stop(session="lec1", wait=False, timeout_seconds=1.0)
        self.assertTrue(agent_runtime.flag_exists(self.session_dir, agent_runtime.SHUTDOWN_REQUEST_FLAG_NAME))

    def test_stop_waits_for_finished_flag(self) -> None:
        agent_runtime.set_flag(self.session_dir, agent_runtime.SESSION_FINISHED_FLAG_NAME)
        with mock.patch.object(agent_runtime, "clear_flag"):
            with mock.patch.object(cli_main, "clear_flag"):
                cli_main.stop(session="lec1", wait=True, timeout_seconds=3.0)

    def test_stop_times_out_when_session_never_finishes(self) -> None:
        with self.assertRaises(typer.Exit):
            cli_main.stop(session="lec1", wait=True, timeout_seconds=0.3)

    def test_unknown_or_invalid_sessions_exit_with_error(self) -> None:
        with self.assertRaises(typer.Exit):
            cli_main.pause(session="does_not_exist")
        with self.assertRaises(typer.Exit):
            cli_main.pause(session="../etc")

    def test_session_defaults_to_latest_instead_of_default_session(self) -> None:
        cli_main.pause(session=None)
        self.assertTrue(agent_runtime.flag_exists(self.session_dir, agent_runtime.PAUSE_FLAG_NAME))

    def test_export_md_and_rejects_unknown_format(self) -> None:
        destination = self.output / "out"
        cli_main.export(export_format="md", session="lec1", output=destination)
        self.assertTrue((destination / "lec1_notes.md").exists())
        with self.assertRaises(typer.Exit):
            cli_main.export(export_format="docx", session="lec1", output=destination)

    def test_review_command_executes(self) -> None:
        cli_main.review(course="algorithms", session=None)
        cli_main.review(course=None, session="lec1")

    def test_ask_command_accepts_student_id_argument(self) -> None:
        import inspect
        sig = inspect.signature(cli_main.ask)
        self.assertIn("student_id", sig.parameters)

    def test_start_refuses_existing_session_without_resume(self) -> None:
        with self.assertRaises(typer.Exit):
            cli_main.start(
                mode="local", audio_source="mic", no_camera=False, no_screen=False, audio_only=False,
                session="lec1", course=None, monitor_index=1, llm_mode="local", resume=False
            )

    def test_start_rejects_bad_options_before_touching_anything(self) -> None:
        with self.assertRaises(typer.Exit):
            cli_main.start(
                mode="local", audio_source="loudspeaker", no_camera=False, no_screen=False, audio_only=False,
                session="fresh", course=None, monitor_index=1, llm_mode="local", resume=False
            )
        with self.assertRaises(typer.Exit):
            cli_main.start(
                mode="local", audio_source="mic", no_camera=False, no_screen=False, audio_only=False,
                session="../x", course=None, monitor_index=1, llm_mode="local", resume=False
            )
        self.assertFalse((self.output / "fresh").exists())

    def test_start_requires_redis(self) -> None:
        with mock.patch.object(cli_main, "check_redis", return_value=False):
            with self.assertRaises(typer.Exit):
                cli_main.start(
                    mode="local", audio_source="mic", no_camera=False, no_screen=False, audio_only=False,
                    session="fresh2", course=None, monitor_index=1, llm_mode="local", resume=False
                )
        self.assertFalse((self.output / "fresh2").exists())


if __name__ == "__main__":
    unittest.main()
