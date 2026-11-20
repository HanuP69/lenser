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


class RecordingRouter:
    def __init__(self) -> None:
        self.summary_prompts: list[str] = []

    async def ainvoke(self, messages: list) -> str:
        text = str(messages[0].content)
        if "CURRENT LECTURE EVENTS" in text or "NEW LECTURE TIMELINE SLICE" in text:
            self.summary_prompts.append(text)
            return "Notes about trees."
        if "key technical terms" in text:
            return json.dumps([{"term": "Tree", "definition": "A connected acyclic graph."}])
        if "action items" in text:
            return "[]"
        if "key points" in text:
            return json.dumps(["A connected acyclic graph."])
        if "concept tags" in text:
            return json.dumps(["tree", "graph"])
        return "Trees were discussed."


class TestPipeline(unittest.IsolatedAsyncioTestCase):
    async def test_events_flow_to_notes_through_the_shutdown_protocol(self) -> None:
        settings = Settings()
        bus = RedisBus(settings.redis_url)
        try:
            await bus.client.ping()
        except Exception:
            self.skipTest("Redis not available")
        session = "pipeline_session"
        for suffix in ("transcript", "slide", "notes"):
            await bus.client.delete(f"lecturelens:{session}:stream:{suffix}")

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings()
            settings.output_dir = Path(temp_dir)
            settings.sync_merge_window_seconds = 0.2
            clock = SessionClock()
            router = RecordingRouter()
            sync = SyncAgent(settings, bus, clock, session)
            structuring = StructuringAgent(settings, session, router, redis_bus=bus)
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

        for suffix in ("transcript", "slide", "notes"):
            await bus.client.delete(f"lecturelens:{session}:stream:{suffix}")
        await bus.close()

    async def test_pipeline_filters_drift_and_ads(self) -> None:
        settings = Settings()
        bus = RedisBus(settings.redis_url)
        try:
            await bus.client.ping()
        except Exception:
            self.skipTest("Redis not available")
        session = "pipeline_drift_session"
        for suffix in ("transcript", "slide", "notes"):
            await bus.client.delete(f"lecturelens:{session}:stream:{suffix}")

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings()
            settings.output_dir = Path(temp_dir)
            settings.sync_merge_window_seconds = 0.2
            settings.enable_drift_filter = True
            clock = SessionClock()
            router = RecordingRouter()

            from lecturelens.core.drift_filter import ContextDriftFilter
            class IntegrationMockEmbedder:
                def embed(self, text: str):
                    import numpy as np
                    clean = text.lower()
                    if "quadratic" in clean or "math" in clean or "roots" in clean:
                        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                    if "communication" in clean or "ad" in clean or "commercial" in clean:
                        return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
                    return np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)

            custom_drift = ContextDriftFilter(
                drift_threshold=0.50,
                min_words=4,
                embedder=IntegrationMockEmbedder()
            )

            sync = SyncAgent(settings, bus, clock, session)
            structuring = StructuringAgent(settings, session, router, redis_bus=bus, drift_filter=custom_drift)
            session_dir = settings.get_session_output_dir(session)

            base = time.time()
            await bus.publish(
                f"lecturelens:{session}:stream:transcript",
                TranscriptChunk(
                    session_id=session, timestamp_epoch=base, start_seconds=0.0, end_seconds=3.0,
                    text="In mathematics today we solve quadratic equations for roots.", confidence=0.9,
                    importance_flags=["definition"]
                )
            )
            await bus.publish(
                f"lecturelens:{session}:stream:slide",
                SlideCaptured(
                    session_id=session, timestamp_epoch=base + 1.0, slide_number=1, monitor_index=1,
                    image_file_path="s1.png", extracted_text="Quadratic Equations: ax^2 + bx + c = 0"
                )
            )
            await bus.publish(
                f"lecturelens:{session}:stream:slide",
                SlideCaptured(
                    session_id=session, timestamp_epoch=base + 2.0, slide_number=2, monitor_index=1,
                    image_file_path="s2.png", extracted_text="The Importance of Effective Communication Presented by: [Name] Build relationships."
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
            self.assertIn("quadratic", prompt.lower())
            self.assertNotIn("Effective Communication", prompt)

        for suffix in ("transcript", "slide", "notes"):
            await bus.client.delete(f"lecturelens:{session}:stream:{suffix}")
        await bus.close()

    async def test_screen_agent_with_vision_embedder_pipeline(self) -> None:
        import numpy as np
        from lecturelens.agents.screen_agent import ScreenAgent
        from lecturelens.core.image_utils import SlideChangeGate

        settings = Settings()
        bus = RedisBus(settings.redis_url)
        try:
            await bus.client.ping()
        except Exception:
            self.skipTest("Redis not available")

        session = "screen_embed_session"
        stream_name = f"lecturelens:{session}:stream:slide"
        await bus.client.delete(stream_name)

        with tempfile.TemporaryDirectory() as temp_dir:
            settings.output_dir = Path(temp_dir)
            clock = SessionClock()
            agent = ScreenAgent(settings, bus, clock, session)

            class IntegrationVisionMock:
                def __init__(self) -> None:
                    self.sim = 0.99

                def embed(self, arr: np.ndarray) -> np.ndarray:
                    return np.ones(10, dtype=np.float32) / np.sqrt(10.0)

                def similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
                    return self.sim

            mock_emb = IntegrationVisionMock()
            agent.gate = SlideChangeGate(
                change_threshold=0.01,
                pixel_delta=20,
                min_interval_seconds=0.1,
                embedder=mock_emb,
                similarity_threshold=0.95
            )

            frame1 = np.zeros((100, 100), dtype=np.uint8)
            accepted1 = agent.gate.evaluate(frame1, 1.0)
            self.assertTrue(accepted1)

            frame2_motion = np.zeros((100, 100), dtype=np.uint8)
            frame2_motion[:40, :40] = 200
            mock_emb.sim = 0.98
            accepted2 = agent.gate.evaluate(frame2_motion, 2.0)
            self.assertFalse(accepted2)

            mock_emb.sim = 0.70
            accepted3 = agent.gate.evaluate(frame2_motion, 3.0)
            self.assertTrue(accepted3)

        await bus.client.delete(stream_name)
        await bus.close()


if __name__ == "__main__":
    unittest.main()
