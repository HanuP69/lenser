import asyncio
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lecturelens.client.student_client import (
    PersonalizationEngine,
    StudentClient,
    StudentProfile
)
from lecturelens.schemas.events import GeneralNoteSegment


class MockLLMRouter:
    def __init__(self, output: str) -> None:
        self.output = output
        self.invoked_prompts: list[str] = []

    async def ainvoke(self, messages: list) -> str:
        self.invoked_prompts.append(messages[0].content)
        return self.output


class TestPersonalizationEngine(unittest.IsolatedAsyncioTestCase):
    def make_sample_segment(self) -> GeneralNoteSegment:
        return GeneralNoteSegment(
            session_id="cs101_lec",
            sequence_number=1,
            timestamp_epoch=1700000000.0,
            start_time="00:01:00",
            end_time="00:02:00",
            heading="Binary Search Analysis",
            content="Binary search runs in O(log n) time by halving search space.",
            key_points=["Search space halves every step", "Requires sorted array"],
            concept_tags=["binary_search", "complexity"]
        )

    async def test_english_profile_returns_original_without_extra_llm_call(self) -> None:
        profile = StudentProfile(preferred_language="English")
        router = MockLLMRouter("rewritten")
        engine = PersonalizationEngine(profile, router)
        segment = self.make_sample_segment()
        result = await engine.rewrite_segment(segment)
        self.assertEqual(result, segment.content)
        self.assertEqual(len(router.invoked_prompts), 0)

    async def test_hinglish_profile_invokes_llm_and_preserves_concepts(self) -> None:
        profile = StudentProfile(preferred_language="Hinglish")
        hinglish_output = "Binary search har step pe array ko half kar deta hai, isliye iska time complexity O(log n) hota hai."
        router = MockLLMRouter(hinglish_output)
        engine = PersonalizationEngine(profile, router)
        segment = self.make_sample_segment()
        result = await engine.rewrite_segment(segment)
        self.assertEqual(result, hinglish_output)
        self.assertEqual(len(router.invoked_prompts), 1)
        self.assertIn("Hinglish", router.invoked_prompts[0])

    async def test_rewrite_validation_fallback_on_empty_or_hallucinated(self) -> None:
        profile = StudentProfile(preferred_language="Hindi")
        router = MockLLMRouter("Too short")
        engine = PersonalizationEngine(profile, router)
        segment = self.make_sample_segment()
        result = await engine.rewrite_segment(segment)
        self.assertEqual(result, segment.content)

    async def test_english_profile_with_concise_detail_rewrites(self) -> None:
        profile = StudentProfile(preferred_language="English", detail_level="concise")
        router = MockLLMRouter("Binary search halves search space in O(log n) time. [00:01:00] binary search complexity")
        engine = PersonalizationEngine(profile, router)
        segment = self.make_sample_segment()
        result = await engine.rewrite_segment(segment)
        self.assertIn("halves search space", result)
        self.assertEqual(len(router.invoked_prompts), 1)


class TestStudentClient(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    async def test_on_note_segment_received_appends_and_updates_sequence(self) -> None:
        profile = StudentProfile(student_id="student_42", preferred_language="English")
        client = StudentClient(
            server_url="ws://localhost:8765",
            session_id="lec_1",
            profile=profile,
            output_dir=self.output_dir
        )
        segment = GeneralNoteSegment(
            session_id="lec_1",
            sequence_number=1,
            timestamp_epoch=1700000000.0,
            start_time="00:00:10",
            end_time="00:00:30",
            heading="Recursion",
            content="A function calling itself.",
            key_points=["Base case needed"],
            concept_tags=["recursion"]
        )
        content = await client.on_note_segment_received(segment)
        self.assertEqual(content, "A function calling itself.")
        self.assertEqual(client.last_sequence_number, 1)
        self.assertTrue(client.notes_file.exists())
        written_text = client.notes_file.read_text(encoding="utf-8")
        self.assertIn("A function calling itself.", written_text)

    async def test_older_or_duplicate_sequence_is_ignored(self) -> None:
        profile = StudentProfile(preferred_language="English")
        client = StudentClient("ws://localhost:8765", "lec_1", profile, self.output_dir)
        client.last_sequence_number = 5
        segment = GeneralNoteSegment(
            session_id="lec_1",
            sequence_number=3,
            timestamp_epoch=1700000000.0,
            start_time="00:00:10",
            end_time="00:00:30",
            heading="Old Section",
            content="Old content",
            key_points=[],
            concept_tags=[]
        )
        result = await client.on_note_segment_received(segment)
        self.assertEqual(result, "")
        self.assertEqual(client.last_sequence_number, 5)

    async def test_client_buffers_gap_and_drains_when_missing_segment_arrives(self) -> None:
        profile = StudentProfile(student_id="student_gap", preferred_language="English")
        client = StudentClient("ws://localhost:8765", "lec_gap", profile, self.output_dir)
        client.last_sequence_number = 1

        seg3 = GeneralNoteSegment(
            session_id="lec_gap",
            sequence_number=3,
            timestamp_epoch=1700000020.0,
            start_time="00:00:20",
            end_time="00:00:30",
            heading="Section 3",
            content="Third section content.",
            key_points=[],
            concept_tags=[]
        )
        seg2 = GeneralNoteSegment(
            session_id="lec_gap",
            sequence_number=2,
            timestamp_epoch=1700000010.0,
            start_time="00:00:10",
            end_time="00:00:20",
            heading="Section 2",
            content="Second section content.",
            key_points=[],
            concept_tags=[]
        )

        res3 = await client.on_note_segment_received(seg3)
        self.assertEqual(res3, "")
        self.assertEqual(client.last_sequence_number, 1)
        self.assertIn(3, client.pending_segments)

        res2 = await client.on_note_segment_received(seg2)
        self.assertEqual(client.last_sequence_number, 3)
        self.assertEqual(len(client.pending_segments), 0)
        written_text = client.notes_file.read_text(encoding="utf-8")
        self.assertIn("Second section content.", written_text)
        self.assertIn("Third section content.", written_text)

    async def test_client_persists_sequence_number_across_instances(self) -> None:
        profile = StudentProfile(student_id="student_persist", preferred_language="English")
        client1 = StudentClient("ws://localhost:8765", "lec_p", profile, self.output_dir)
        seg1 = GeneralNoteSegment(
            session_id="lec_p",
            sequence_number=1,
            timestamp_epoch=1700000010.0,
            start_time="00:00:10",
            end_time="00:00:20",
            heading="Section 1",
            content="Content 1",
            key_points=[],
            concept_tags=[]
        )
        await client1.on_note_segment_received(seg1)
        self.assertEqual(client1.last_sequence_number, 1)

        client2 = StudentClient("ws://localhost:8765", "lec_p", profile, self.output_dir)
        self.assertEqual(client2.last_sequence_number, 1)


    async def test_client_updates_student_graph_concepts(self) -> None:
        from lecturelens.knowledge.student_graph import StudentKnowledgeGraph
        profile = StudentProfile(student_id="student_graph_test", preferred_language="English")
        graph_db = self.output_dir / "test_graph.db"
        graph = StudentKnowledgeGraph(graph_db)
        client = StudentClient(
            server_url="ws://localhost:8765",
            session_id="lec_g",
            profile=profile,
            output_dir=self.output_dir,
            student_graph=graph
        )
        segment = GeneralNoteSegment(
            session_id="lec_g",
            sequence_number=1,
            timestamp_epoch=1700000000.0,
            start_time="00:00:10",
            end_time="00:00:30",
            heading="Graph Theory",
            content="Nodes and edges form a graph.",
            key_points=[],
            concept_tags=["graph_theory", "nodes"]
        )
        await client.on_note_segment_received(segment)
        summary = graph.get_student_summary()
        self.assertEqual(summary["concepts_tracked"], 2)
        names = [c["concept_name"] for c in summary["concepts"]]
        self.assertIn("graph_theory", names)
        self.assertIn("nodes", names)

    async def test_client_ledger_prevents_duplicate_markdown_and_persists(self) -> None:
        profile = StudentProfile(student_id="student_ledger_test", preferred_language="English")
        client = StudentClient("ws://localhost:8765", "lec_ledger", profile, self.output_dir)
        seg = GeneralNoteSegment(
            session_id="lec_ledger",
            sequence_number=1,
            timestamp_epoch=1700000010.0,
            start_time="00:00:10",
            end_time="00:00:20",
            heading="Ledger Section",
            content="Ledger content should appear once.",
            key_points=[],
            concept_tags=[]
        )
        await client._apply_segment(seg)
        await client._apply_segment(seg)
        written = client.notes_file.read_text(encoding="utf-8")
        self.assertEqual(written.count("Ledger content should appear once."), 1)
        self.assertTrue(client.ledger_file.exists())

    async def test_startup_reconciliation_appends_missing_markdown(self) -> None:
        import sqlite3
        profile = StudentProfile(student_id="student_reconcile", preferred_language="English")
        ledger_file = self.output_dir / "lec_recon_student_reconcile_ledger.db"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(ledger_file))
        conn.execute("""
            CREATE TABLE applied_segments (
                sequence_number INTEGER PRIMARY KEY,
                session_id TEXT,
                heading TEXT,
                content TEXT,
                created_at REAL
            );
        """)
        conn.execute(
            "INSERT INTO applied_segments VALUES (1, 'lec_recon', 'Head 1', 'Content segment 1', 1000.0)"
        )
        conn.execute(
            "INSERT INTO applied_segments VALUES (2, 'lec_recon', 'Head 2', 'Content segment 2', 1001.0)"
        )
        conn.commit()
        conn.close()

        notes_file = self.output_dir / "lec_recon_student_reconcile_personalized_notes.md"
        notes_file.write_text("\nContent segment 1\n", encoding="utf-8")

        client = StudentClient("ws://localhost:8765", "lec_recon", profile, self.output_dir)
        written = client.notes_file.read_text(encoding="utf-8")
        self.assertIn("Content segment 1", written)
        self.assertIn("Content segment 2", written)
        self.assertEqual(client.last_sequence_number, 2)

    async def test_unauthorized_reply_is_recorded(self) -> None:
        """The client used to ignore the unauthorized reply and reconnect
        every second with the same rejected credentials."""
        profile = StudentProfile(student_id="student_auth", preferred_language="English")
        client = StudentClient("ws://localhost:8765", "lec_auth", profile, self.output_dir)
        self.assertIsNone(client.auth_error)

        await client.handle_server_frame(json.dumps({"status": "unauthorized", "error": "invalid api key"}))
        self.assertEqual(client.auth_error, "invalid api key")

    async def test_replay_requests_are_capped(self) -> None:
        """A gap that can never be filled used to trigger an endless stream
        of request_replay messages."""
        profile = StudentProfile(student_id="student_gap", preferred_language="English")
        client = StudentClient("ws://localhost:8765", "lec_gap", profile, self.output_dir)
        sent: list[str] = []

        class FakeSocket:
            async def send(self, payload: str) -> None:
                sent.append(payload)

        socket = FakeSocket()
        segment = GeneralNoteSegment(
            session_id="lec_gap",
            sequence_number=5,
            timestamp_epoch=1700000000.0,
            start_time="00:00:10",
            end_time="00:00:30",
            heading="Section 5",
            content="Gap content",
            key_points=[],
            concept_tags=[]
        )
        for _ in range(40):
            client.last_replay_request_time = 0.0
            await client.on_note_segment_received(segment, socket)

        self.assertLessEqual(len(sent), client.max_replay_requests + 1)
        self.assertGreater(len(sent), 0)

    async def test_two_students_in_one_directory_do_not_overwrite_each_other(self) -> None:
        """The notes file used to be named after the session only, so two
        students sharing an output directory clobbered each other's notes."""
        profile_a = StudentProfile(student_id="student_a", preferred_language="English")
        profile_b = StudentProfile(student_id="student_b", preferred_language="English")
        client_a = StudentClient("ws://localhost:8765", "lec_shared", profile_a, self.output_dir)
        client_b = StudentClient("ws://localhost:8765", "lec_shared", profile_b, self.output_dir)
        self.assertNotEqual(client_a.notes_file, client_b.notes_file)

        seg_a = GeneralNoteSegment(
            session_id="lec_shared",
            sequence_number=1,
            timestamp_epoch=1700000000.0,
            start_time="00:00:10",
            end_time="00:00:30",
            heading="Section 1",
            content="Content written by student A",
            key_points=[],
            concept_tags=[]
        )
        seg_b = GeneralNoteSegment(
            session_id="lec_shared",
            sequence_number=1,
            timestamp_epoch=1700000000.0,
            start_time="00:00:10",
            end_time="00:00:30",
            heading="Section 1",
            content="Content written by student B",
            key_points=[],
            concept_tags=[]
        )
        await client_a.on_note_segment_received(seg_a)
        await client_b.on_note_segment_received(seg_b)

        self.assertIn("Content written by student A", client_a.notes_file.read_text(encoding="utf-8"))
        self.assertIn("Content written by student B", client_b.notes_file.read_text(encoding="utf-8"))

    async def test_apply_segment_raises_and_preserves_cursor_on_ledger_error(self) -> None:
        profile = StudentProfile(student_id="student_err", preferred_language="English")
        client = StudentClient("ws://localhost:8765", "lec_err", profile, self.output_dir)
        self.assertEqual(client.last_sequence_number, 0)
        seg = GeneralNoteSegment(
            session_id="lec_err",
            sequence_number=1,
            timestamp_epoch=1700000010.0,
            start_time="00:00:10",
            end_time="00:00:20",
            heading="Err Section",
            content="Content should not persist on error.",
            key_points=[],
            concept_tags=[]
        )
        with mock.patch("sqlite3.connect", side_effect=sqlite3.OperationalError("disk failure")):
            with self.assertRaises(RuntimeError):
                await client._apply_segment(seg)

        self.assertEqual(client.last_sequence_number, 0)
        if client.notes_file.exists():
            self.assertNotIn("Content should not persist on error.", client.notes_file.read_text(encoding="utf-8"))

    async def test_reconciliation_orders_segments_strictly_by_sequence(self) -> None:
        profile = StudentProfile(student_id="student_order", preferred_language="English")
        client = StudentClient("ws://localhost:8765", "lec_order", profile, self.output_dir)
        seg13 = GeneralNoteSegment(
            session_id="lec_order",
            sequence_number=13,
            timestamp_epoch=1700000020.0,
            start_time="00:02:00",
            end_time="00:03:00",
            heading="Section 13",
            content="Content for section 13",
            key_points=[],
            concept_tags=[]
        )
        seg12 = GeneralNoteSegment(
            session_id="lec_order",
            sequence_number=12,
            timestamp_epoch=1700000010.0,
            start_time="00:01:00",
            end_time="00:02:00",
            heading="Section 12",
            content="Content for section 12",
            key_points=[],
            concept_tags=[]
        )
        await client._apply_segment(seg13)
        await client._apply_segment(seg12)

        written = client.notes_file.read_text(encoding="utf-8")
        pos12 = written.find("Content for section 12")
        pos13 = written.find("Content for section 13")
        self.assertGreaterEqual(pos12, 0)
        self.assertGreaterEqual(pos13, 0)
        self.assertLess(pos12, pos13)

    async def test_substring_collision_does_not_skip_segment(self) -> None:
        profile = StudentProfile(student_id="student_sub", preferred_language="English")
        client = StudentClient("ws://localhost:8765", "lec_sub", profile, self.output_dir)
        seg1 = GeneralNoteSegment(
            session_id="lec_sub",
            sequence_number=1,
            timestamp_epoch=1700000010.0,
            start_time="00:01:00",
            end_time="00:02:00",
            heading="Fruit",
            content="Apple",
            key_points=[],
            concept_tags=[]
        )
        seg2 = GeneralNoteSegment(
            session_id="lec_sub",
            sequence_number=2,
            timestamp_epoch=1700000020.0,
            start_time="00:02:00",
            end_time="00:03:00",
            heading="Bigger Fruit",
            content="Pineapple",
            key_points=[],
            concept_tags=[]
        )
        await client._apply_segment(seg1)
        await client._apply_segment(seg2)

        written = client.notes_file.read_text(encoding="utf-8")
        self.assertIn("Apple", written)
        self.assertIn("Pineapple", written)

    async def test_retry_reconciles_markdown_even_if_already_in_ledger(self) -> None:
        profile = StudentProfile(student_id="student_retry", preferred_language="English")
        client = StudentClient("ws://localhost:8765", "lec_retry", profile, self.output_dir)
        seg = GeneralNoteSegment(
            session_id="lec_retry",
            sequence_number=1,
            timestamp_epoch=1700000010.0,
            start_time="00:01:00",
            end_time="00:02:00",
            heading="Section 1",
            content="Durable content",
            key_points=[],
            concept_tags=[]
        )
        await client._apply_segment(seg)
        client.notes_file.unlink()
        self.assertFalse(client.notes_file.exists())

        await client._apply_segment(seg)
        self.assertTrue(client.notes_file.exists())
        self.assertIn("Durable content", client.notes_file.read_text(encoding="utf-8"))

    async def test_empty_ledger_clears_stale_markdown_file(self) -> None:
        profile = StudentProfile(student_id="student_stale", preferred_language="English")
        stale_file = self.output_dir / "lec_stale_personalized_notes.md"
        stale_file.write_text("Stale notes from previous run\n", encoding="utf-8")
        self.assertTrue(stale_file.exists())

        client = StudentClient("ws://localhost:8765", "lec_stale", profile, self.output_dir)
        self.assertFalse(stale_file.exists())

    async def test_apply_segment_raises_and_preserves_cursor_on_reconciliation_failure(self) -> None:
        profile = StudentProfile(student_id="student_fail", preferred_language="English")
        client = StudentClient("ws://localhost:8765", "lec_fail", profile, self.output_dir)
        self.assertEqual(client.last_sequence_number, 0)

        seg = GeneralNoteSegment(
            session_id="lec_fail",
            sequence_number=1,
            timestamp_epoch=1700000010.0,
            start_time="00:01:00",
            end_time="00:02:00",
            heading="Section 1",
            content="Important content",
            key_points=[],
            concept_tags=[]
        )

        with mock.patch.object(client, "reconcile_notes_from_ledger", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                await client._apply_segment(seg)
            self.assertIn("reconciling notes markdown file failed", str(ctx.exception))

        self.assertEqual(client.last_sequence_number, 0)
        self.assertTrue(client.pending_reconciliation)
        self.assertEqual(client.load_persisted_sequence(), 0)

    async def test_load_persisted_sequence_bounds_stale_state_file_by_ledger_max_when_pending(self) -> None:
        profile = StudentProfile(student_id="student_stale_state", preferred_language="English")
        client = StudentClient("ws://localhost:8765", "lec_stale_state", profile, self.output_dir)

        client.state_file.write_text(json.dumps({"last_sequence_number": 100}), encoding="utf-8")

        conn = sqlite3.connect(str(client.ledger_file))
        for seq in [1, 2, 95]:
            conn.execute(
                "INSERT INTO applied_segments (sequence_number, session_id, heading, content, created_at) VALUES (?, ?, ?, ?, ?)",
                (seq, "lec_stale_state", f"Head {seq}", f"Content {seq}", 1700000000.0)
            )
        conn.commit()
        conn.close()

        client.pending_reconciliation = True
        self.assertEqual(client.load_persisted_sequence(), 95)

        client.pending_reconciliation = False
        self.assertEqual(client.load_persisted_sequence(), 100)

    async def test_reconcile_notes_from_ledger_cleans_empty_ledger_with_zero_cursor(self) -> None:
        profile = StudentProfile(student_id="student_empty_state", preferred_language="English")
        client = StudentClient("ws://localhost:8765", "lec_empty_state", profile, self.output_dir)
        client.state_file.write_text(json.dumps({"last_sequence_number": 50}), encoding="utf-8")
        client.pending_reconciliation = True
        self.assertEqual(client.load_persisted_sequence(), 0)

    async def test_legacy_upgrade_preserves_existing_notes_and_populates_ledger(self) -> None:
        profile = StudentProfile(student_id="student_legacy", preferred_language="English")
        state_file = self.output_dir / "lec_legacy_student_legacy_state.json"
        state_file.write_text(json.dumps({"last_sequence_number": 7}), encoding="utf-8")
        legacy_notes_file = self.output_dir / "lec_legacy_personalized_notes.md"
        legacy_notes_file.write_text("Preserved legacy notes content\n", encoding="utf-8")

        client = StudentClient("ws://localhost:8765", "lec_legacy", profile, self.output_dir)
        # The legacy per-session notes file is migrated to the per-student
        # name so an upgrade does not lose the notes.
        self.assertFalse(legacy_notes_file.exists())
        self.assertIn(
            "Preserved legacy notes content",
            client.notes_file.read_text(encoding="utf-8")
        )
        self.assertEqual(client.last_sequence_number, 7)
        self.assertFalse(client.pending_reconciliation)

        conn = sqlite3.connect(str(client.ledger_file))
        cur = conn.cursor()
        cur.execute("SELECT sequence_number, content FROM applied_segments;")
        rows = cur.fetchall()
        conn.close()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], 7)
        self.assertEqual(rows[0][1], "Preserved legacy notes content")

    async def test_corrupt_ledger_marks_unverified_and_resets_cursor(self) -> None:
        profile = StudentProfile(student_id="student_corrupt", preferred_language="English")
        client = StudentClient("ws://localhost:8765", "lec_corrupt", profile, self.output_dir)
        client.state_file.write_text(json.dumps({"last_sequence_number": 42}), encoding="utf-8")

        client.ledger_file.write_bytes(b"corrupted sqlite file header")
        seq = client.load_persisted_sequence()
        self.assertEqual(seq, 0)
        self.assertTrue(client.pending_reconciliation)

    async def test_legacy_upgrade_with_missing_notes_resets_cursor(self) -> None:
        profile = StudentProfile(student_id="student_missing_notes", preferred_language="English")
        state_file = self.output_dir / "lec_missing_student_missing_notes_state.json"
        state_file.write_text(json.dumps({"last_sequence_number": 100}), encoding="utf-8")

        client = StudentClient("ws://localhost:8765", "lec_missing", profile, self.output_dir)
        self.assertEqual(client.last_sequence_number, 0)
        self.assertTrue(client.pending_reconciliation)
        data = json.loads(state_file.read_text(encoding="utf-8"))
        self.assertEqual(data.get("last_sequence_number"), 0)


if __name__ == "__main__":
    unittest.main()
