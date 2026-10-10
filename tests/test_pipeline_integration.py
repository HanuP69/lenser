import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path

from lecturelens.agents.structuring_agent import StructuringAgent
from lecturelens.agents.sync_agent import SyncAgent
from lecturelens.config.settings import Settings
from lecturelens.core.agent_runtime import FINALIZE_FLAG_NAME, SYNC_STOP_FLAG_NAME, set_flag
from lecturelens.core.clock import SessionClock
from lecturelens.core.redis_bus import RedisBus
from lecturelens.schemas.events import SlideCaptured, TranscriptChunk
from tests.support import pluck


class RecordingRouter:
    def __init__(self) -> None:
        self.summary_prompts: list[str] = []

    async def ainvoke(self, messages: list) -> str:
        text = str(messages[0].content)
        if "NEW LECTURE TIMELINE SLICE" in text:
            self.summary_prompts.append(text)
            return "Notes about trees."
        if "key technical terms" in text:
            return json.dumps([{"term": "Tree", "definition": "A connected acyclic graph."}])
        if "action items" in text:
            return "[]"
        if "study flashcards" in text:
            return json.dumps([{"question": "What is a tree?", "answer": "Connected acyclic graph."}])
        return "Trees were discussed."


class TestPipeline(unittest.IsolatedAsyncioTestCase):
    async def test_events_flow_to_notes_through_the_shutdown_protocol(self) -> None:
        bus = RedisBus("redis://localhost:6379")
        try:
            await bus.client.ping()
        except Exception:
            self.skipTest("Redis not available")
        session = "pipeline_session"
        for suffix in ("transcript", "slide", "board"):
            await bus.client.delete(f"lecturelens:{session}:stream:{suffix}")

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings()
            settings.output_dir = Path(temp_dir)
            settings.sync_merge_window_seconds = 0.2
            clock = SessionClock()
            router = RecordingRouter()
            sync = SyncAgent(settings, bus, clock, session)
            structuring = StructuringAgent(settings, session, router)
            session_dir = settings.get_session_output_dir(session)

            base = time.time()
            await bus.publish(
                f"lecturelens:{session}:stream:transcript",
                TranscriptChunk(
                    session_id=session, timestamp_epoch=base, start_seconds=0.0, end_seconds=3.0,
                    text="A tree is defined as a connected acyclic graph.", confidence=0.9,
                    importance_flags=["definition"]
                )
            )
            await bus.publish(
                f"lecturelens:{session}:stream:slide",
                SlideCaptured(
                    session_id=session, timestamp_epoch=base + 1.0, slide_number=1, monitor_index=1,
                    image_file_path="s.png", extracted_text="Trees: connected, acyclic"
                )
            )

            async def shutdown_sequence() -> None:
                await asyncio.sleep(1.0)
                set_flag(session_dir, SYNC_STOP_FLAG_NAME)
                await sync_task
                set_flag(session_dir, FINALIZE_FLAG_NAME)

            sync_task = asyncio.create_task(sync.run())
            structuring_task = asyncio.create_task(structuring.run())
            await asyncio.wait_for(asyncio.gather(shutdown_sequence(), structuring_task), timeout=60)

            notes = (session_dir / "notes.md").read_text(encoding="utf-8")
            self.assertIn("Notes about trees.", notes)
            self.assertEqual(len(router.summary_prompts), 1)
            prompt = router.summary_prompts[0]
            self.assertIn("[TRANSCRIPT][definition] A tree is defined as a connected acyclic graph. | [SLIDE] Trees: connected, acyclic", prompt)
            self.assertTrue((session_dir / "flashcards.apkg").exists())
            concepts = structuring.concept_graph.get_all_concepts_for_session(session)
            self.assertEqual(pluck(concepts, "concept_name"), ["tree"])

        for suffix in ("transcript", "slide", "board"):
            await bus.client.delete(f"lecturelens:{session}:stream:{suffix}")
        await bus.close()


if __name__ == "__main__":
    unittest.main()
