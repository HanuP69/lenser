import io
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image

from lecturelens.agents.screen_agent import ScreenAgent, build_thumbnail_from_png
from lecturelens.config.settings import Settings
from lecturelens.core.clock import SessionClock
from lecturelens.core.image_utils import (
    compute_edge_count,
    is_content_building_on_top
)


class DummyBus:
    def __init__(self) -> None:
        self.published: list = []

    async def publish_resilient(self, stream_name: str, event: object) -> str:
        self.published.append(event)
        return "msg-1"


def make_png_with_text(text: str, width: int = 640, height: int = 360) -> bytes:
    from PIL import ImageDraw
    img = Image.new("RGB", (width, height), (30, 80, 40))
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), text, fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestKeyframeSuperseding(unittest.IsolatedAsyncioTestCase):
    def test_fuzzy_ocr_incremental_expansion(self) -> None:
        building, reason = is_content_building_on_top(
            "Substitution v/s elim",
            "Substitution v/s elimination",
            prev_edges=1000,
            curr_edges=1500
        )
        self.assertTrue(building)
        self.assertIn("expansion", reason)

    def test_diagram_growth_without_text(self) -> None:
        building, reason = is_content_building_on_top(
            "",
            "",
            prev_edges=2000,
            curr_edges=2500
        )
        self.assertTrue(building)
        self.assertEqual(reason, "diagram_edges_growing")

    def test_duster_wipe_text_drop(self) -> None:
        building, reason = is_content_building_on_top(
            "Substitution v/s elimination Sn1 Solvolysis moist Ag2O KOH AgNO3",
            "Substitution v/s",
            prev_edges=20000,
            curr_edges=8000
        )
        self.assertFalse(building)
        self.assertEqual(reason, "text_erased_len_drop")

    def test_duster_wipe_diagram_edge_drop(self) -> None:
        building, reason = is_content_building_on_top(
            "",
            "",
            prev_edges=15000,
            curr_edges=5000
        )
        self.assertFalse(building)
        self.assertEqual(reason, "diagram_edges_dropped")

    def test_edge_count_computation(self) -> None:
        blank = np.zeros((100, 100), dtype=np.uint8)
        self.assertEqual(compute_edge_count(blank), 0)
        pattern = np.zeros((100, 100), dtype=np.uint8)
        pattern[::10, :] = 255
        self.assertGreater(compute_edge_count(pattern), 50)

    async def test_screen_agent_supersedes_incremental_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings()
            settings.output_dir = Path(temp_dir)
            bus = DummyBus()
            agent = ScreenAgent(
                settings=settings,
                redis_bus=bus,
                session_clock=SessionClock(),
                session_id="test_supersede",
                llm_router=None
            )

            png1 = make_png_with_text("Substitution v/s elim")
            png2 = make_png_with_text("Substitution v/s elimination")

            event1 = await agent.ingest_png_bytes(png1, capture_epoch=1.0, enforce_gate=True, gate_now=1.0)
            self.assertIsNotNone(event1)
            self.assertEqual(agent.slide_counter, 1)

            event2 = await agent.ingest_png_bytes(png2, capture_epoch=6.0, enforce_gate=True, gate_now=6.0)
            self.assertIsNone(event2)
            self.assertEqual(agent.slide_counter, 1)

            slide1_file = agent.slides_dir / "slide_0001.png"
            self.assertTrue(slide1_file.exists())
