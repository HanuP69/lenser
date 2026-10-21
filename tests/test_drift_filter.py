import json
import unittest
from pathlib import Path

import numpy as np

from lecturelens.core.drift_filter import (
    ContextDriftFilter,
    cosine_similarity,
    is_refusal_text,
    normalize_vector
)


class MockTextEmbedder:
    def __init__(self) -> None:
        self.vectors: dict[str, np.ndarray] = {}

    def set_vector(self, text: str, vec: np.ndarray) -> None:
        self.vectors[text.strip()] = normalize_vector(np.asarray(vec, dtype=np.float32))

    def embed(self, text: str) -> np.ndarray:
        clean = text.strip()
        if clean in self.vectors:
            return self.vectors[clean]
        for key, vec in self.vectors.items():
            if key in clean or clean in key:
                return vec
        return np.ones(8, dtype=np.float32) / np.sqrt(8.0)


class TestDriftFilterMath(unittest.TestCase):
    def test_normalize_vector(self) -> None:
        zero = np.zeros(4, dtype=np.float32)
        self.assertTrue(np.array_equal(normalize_vector(zero), zero))
        vec = np.array([3.0, 4.0], dtype=np.float32)
        normed = normalize_vector(vec)
        self.assertAlmostEqual(float(np.linalg.norm(normed)), 1.0, places=5)

    def test_cosine_similarity(self) -> None:
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)
        self.assertAlmostEqual(cosine_similarity(v1, v2), 0.0, places=5)
        self.assertAlmostEqual(cosine_similarity(v1, v1), 1.0, places=5)
        self.assertAlmostEqual(cosine_similarity(v1, np.zeros(2, dtype=np.float32)), 0.0, places=5)

    def test_is_refusal_text(self) -> None:
        self.assertTrue(is_refusal_text("Please provide the presentation slide image"))
        self.assertTrue(is_refusal_text("no text or diagram was visible"))
        self.assertTrue(is_refusal_text("I need an image of the presentation slide to transcribe"))
        self.assertFalse(is_refusal_text("Quadratic equation roots and discriminant"))
        # A vision model that never got the image says so in its own words.
        # The gemini free tier drops the image, and the reply used to be
        # written into the notes as if it were the slide text.
        self.assertTrue(is_refusal_text(
            "It looks like you didn't attach or upload an image! Please provide the image"
        ))
        self.assertTrue(is_refusal_text(
            "It looks like you forgot to attach an image, and I would be happy to help"
        ))
        self.assertTrue(is_refusal_text("I can't see any image in your message."))
        self.assertTrue(is_refusal_text("No image was provided."))
        # A real transcription is not a refusal.
        self.assertFalse(is_refusal_text(
            "LectureLens Slide 1 Binary search divides the search space in half"
        ))


class TestContextDriftFilterBehavior(unittest.TestCase):
    def setUp(self) -> None:
        self.embedder = MockTextEmbedder()
        self.filter = ContextDriftFilter(
            drift_threshold=0.50,
            sustain_consecutive=3,
            min_words=4,
            embedder=self.embedder
        )

        math_topic = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        math_related = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
        ad_topic = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        new_chapter = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        self.embedder.set_vector("Chapter 4 Quadratic Equation Master Plan", math_topic)
        self.embedder.set_vector("Roots and discriminant of quadratic equation", math_related)
        self.embedder.set_vector("Buy our revolutionary online communication course now", ad_topic)
        self.embedder.set_vector("Welcome to chapter 5 arithmetic progressions", new_chapter)
        self.embedder.set_vector("Arithmetic progressions common difference and nth term", new_chapter)
        self.embedder.set_vector("Sum of first n terms of arithmetic progression", new_chapter)
        self.embedder.set_vector("Next example of arithmetic progression", new_chapter)

    def test_initial_anchor_and_on_topic_acceptance(self) -> None:
        is_drift, score = self.filter.check_entry("Chapter 4 Quadratic Equation Master Plan")
        self.assertFalse(is_drift)
        self.assertEqual(score, 1.0)
        self.assertEqual(self.filter.accepted_count, 1)

        is_drift, score = self.filter.check_entry("Roots and discriminant of quadratic equation")
        self.assertFalse(is_drift)
        self.assertGreaterEqual(score, 0.50)
        self.assertEqual(self.filter.accepted_count, 2)

    def test_isolated_ad_is_filtered(self) -> None:
        self.filter.check_entry("Chapter 4 Quadratic Equation Master Plan")
        is_drift, score = self.filter.check_entry("Buy our revolutionary online communication course now")
        self.assertTrue(is_drift)
        self.assertLess(score, 0.50)

        is_drift, score = self.filter.check_entry("Roots and discriminant of quadratic equation")
        self.assertFalse(is_drift)
        self.assertGreaterEqual(score, 0.50)

    def test_sustained_shift_updates_anchor(self) -> None:
        self.filter.check_entry("Chapter 4 Quadratic Equation Master Plan")

        is_drift, _ = self.filter.check_entry("Welcome to chapter 5 arithmetic progressions")
        self.assertTrue(is_drift)

        is_drift, _ = self.filter.check_entry("Arithmetic progressions common difference and nth term")
        self.assertTrue(is_drift)

        is_drift, _ = self.filter.check_entry("Sum of first n terms of arithmetic progression")
        self.assertFalse(is_drift)

    def test_short_banter_bypasses_drift(self) -> None:
        self.filter.check_entry("Chapter 4 Quadratic Equation Master Plan")
        is_drift, score = self.filter.check_entry("Yes okay")
        self.assertFalse(is_drift)
        self.assertEqual(score, 1.0)

    def test_filter_batch_entries_drops_drift(self) -> None:
        self.filter.check_entry("Chapter 4 Quadratic Equation Master Plan")
        entries = [
            {"content": "Roots and discriminant of quadratic equation", "source_type": "transcript"},
            {"content": "Buy our revolutionary online communication course now", "source_type": "slide"},
            {"content": "Please provide the presentation slide image", "source_type": "slide"}
        ]
        kept = self.filter.filter_batch_entries(entries)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["content"], "Roots and discriminant of quadratic equation")

    def test_exam_and_definition_bypass_drift(self) -> None:
        self.filter.check_entry("Chapter 4 Quadratic Equation Master Plan")
        entries = [
            {
                "content": "Exam is on Friday, Chapter 4 is extremely important",
                "importance_flags": ["exam_relevant"],
                "source_type": "transcript"
            },
            {
                "content": "Buy our revolutionary online communication course now",
                "importance_flags": [],
                "source_type": "slide"
            }
        ]
        kept = self.filter.filter_batch_entries(entries)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["content"], "Exam is on Friday, Chapter 4 is extremely important")

    def test_topic_shift_releases_all_quarantined_entries(self) -> None:
        self.filter.check_entry("Chapter 4 Quadratic Equation Master Plan")
        entries = [
            {"content": "Welcome to chapter 5 arithmetic progressions", "source_type": "transcript"},
            {"content": "Arithmetic progressions common difference and nth term", "source_type": "transcript"},
            {"content": "Sum of first n terms of arithmetic progression", "source_type": "transcript"},
            {"content": "Next example of arithmetic progression", "source_type": "transcript"}
        ]
        kept = self.filter.filter_batch_entries(entries)
        self.assertEqual(len(kept), 4)

    def test_ad_flag_alone_does_not_delete_important_content(self) -> None:
        """A single heuristic flag used to delete the entry outright. The
        content is now only dropped when no important marker contradicts it."""
        self.filter.check_entry("Chapter 4 Quadratic Equation Master Plan")
        entries = [
            {
                "event_id": "important-ad",
                "content": "the audible range of human hearing is twenty hertz",
                "importance_flags": ["ad", "definition"],
                "source_type": "transcript"
            },
            {
                "event_id": "plain-ad",
                "content": "Buy our revolutionary online communication course now",
                "importance_flags": ["ad"],
                "source_type": "slide"
            }
        ]
        kept = self.filter.filter_batch_entries(entries)
        contents = [entry["content"] for entry in kept]
        self.assertIn("the audible range of human hearing is twenty hertz", contents)
        self.assertNotIn("Buy our revolutionary online communication course now", contents)

    def test_refiltering_the_same_batch_is_idempotent(self) -> None:
        """A retried batch used to advance the running anchor and the counters
        a second time, which fabricated a topic shift."""
        self.filter.check_entry("Chapter 4 Quadratic Equation Master Plan")
        batch = [
            {
                "event_id": "a",
                "content": "Roots and discriminant of quadratic equation",
                "source_type": "transcript"
            },
            {
                "event_id": "b",
                "content": "Sum of first n terms of arithmetic progression",
                "source_type": "transcript"
            }
        ]
        first = self.filter.filter_batch_entries(batch)
        anchor_after_first = self.filter.running_anchor.copy()
        accepted_after_first = self.filter.accepted_count
        drift_after_first = self.filter.drift_count

        second = self.filter.filter_batch_entries(batch)

        self.assertEqual(
            [entry["event_id"] for entry in first],
            [entry["event_id"] for entry in second]
        )
        self.assertTrue(np.array_equal(anchor_after_first, self.filter.running_anchor))
        self.assertEqual(accepted_after_first, self.filter.accepted_count)
        self.assertEqual(drift_after_first, self.filter.drift_count)

    def test_ad_flagged_entries_are_quarantined_not_deleted(self) -> None:
        """A false positive on the ad heuristics must not lose the content, so
        the entry is quarantined instead of being dropped."""
        entries = [
            {
                "event_id": "ad-1",
                "content": "This video is sponsored by our friends",
                "importance_flags": ["ad"],
                "source_type": "transcript"
            },
            {
                "event_id": "keep-1",
                "content": "The definition of a binary search tree",
                "importance_flags": ["definition"],
                "source_type": "transcript"
            }
        ]
        kept = self.filter.filter_batch_entries(entries)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["event_id"], "keep-1")
        quarantined = self.filter.quarantined_entries
        self.assertEqual(len(quarantined), 1)
        self.assertEqual(quarantined[0]["event_id"], "ad-1")
        self.assertIn("sponsored by", quarantined[0]["content"])

    def test_refusal_text_is_dropped_outright(self) -> None:
        """A refusal is a vision-model artefact. It used to be quarantined
        like a topic drift and could be released back into the notes when an
        important entry arrived."""
        entries = [
            {
                "event_id": "refusal",
                "content": "Please provide the presentation slide image",
                "importance_flags": [],
                "source_type": "slide"
            },
            {
                "event_id": "important",
                "content": "Here is the definition of a binary search tree",
                "importance_flags": ["definition"],
                "source_type": "transcript"
            }
        ]
        kept = self.filter.filter_batch_entries(entries)
        contents = [entry["content"] for entry in kept]
        self.assertNotIn("Please provide the presentation slide image", contents)
        self.assertIn("Here is the definition of a binary search tree", contents)

    def test_topic_shift_across_batch_boundaries(self) -> None:
        self.filter.check_entry("Chapter 4 Quadratic Equation Master Plan")
        batch1 = [
            {"content": "Welcome to chapter 5 arithmetic progressions", "source_type": "transcript"},
            {"content": "Arithmetic progressions common difference and nth term", "source_type": "transcript"}
        ]
        kept1 = self.filter.filter_batch_entries(batch1)
        self.assertEqual(len(kept1), 0)

        batch2 = [
            {"content": "Sum of first n terms of arithmetic progression", "source_type": "transcript"},
            {"content": "Next example of arithmetic progression", "source_type": "transcript"}
        ]
        kept2 = self.filter.filter_batch_entries(batch2)
        self.assertEqual(len(kept2), 4)
        self.assertEqual(kept2[0]["content"], "Welcome to chapter 5 arithmetic progressions")
        self.assertEqual(kept2[3]["content"], "Next example of arithmetic progression")

    def test_code_not_refusal_false_positive(self) -> None:
        self.assertFalse(is_refusal_text("result = data[name]"))
        self.assertFalse(is_refusal_text("mapping = {'k': [name]}"))
        self.assertFalse(is_refusal_text("template = '{{ [title] }}'"))
        self.assertTrue(is_refusal_text("Presented by: [Name]"))
        self.assertTrue(is_refusal_text("Presented by: [Your Name]"))

    def test_state_persistence_dict(self) -> None:
        self.filter.check_entry("Chapter 4 Quadratic Equation Master Plan")
        state = self.filter.to_dict()
        self.assertIsNotNone(state.get("running_anchor"))
        self.assertEqual(state.get("accepted_count"), 1)

        new_filter = ContextDriftFilter(embedder=self.embedder)
        new_filter.load_dict(state)
        self.assertEqual(new_filter.accepted_count, 1)
        self.assertTrue(np.array_equal(new_filter.running_anchor, self.filter.running_anchor))

    def test_short_statement_preserves_topic_shift_quarantine(self) -> None:
        self.filter.filter_batch_entries([
            {"content": "Chapter 4 Quadratic Equation Master Plan", "source_type": "transcript", "importance_flags": []}
        ])
        entries = [
            {"content": "Welcome to chapter 5 arithmetic progressions", "source_type": "transcript", "importance_flags": []},
            {"content": "Arithmetic progressions common difference and nth term", "source_type": "transcript", "importance_flags": []},
            {"content": "Yes okay", "source_type": "transcript", "importance_flags": []},
            {"content": "Sum of first n terms of arithmetic progression", "source_type": "transcript", "importance_flags": []},
        ]
        kept = self.filter.filter_batch_entries(entries)
        contents = [e["content"] for e in kept]
        self.assertIn("Welcome to chapter 5 arithmetic progressions", contents)
        self.assertIn("Arithmetic progressions common difference and nth term", contents)
        self.assertIn("Yes okay", contents)
        self.assertIn("Sum of first n terms of arithmetic progression", contents)

    def test_drift_filter_reset(self) -> None:
        self.filter.check_entry("Chapter 4 Quadratic Equation Master Plan")
        self.assertIsNotNone(self.filter.running_anchor)
        self.filter.reset()
        self.assertIsNone(self.filter.running_anchor)
        self.assertEqual(self.filter.accepted_count, 0)
        self.assertEqual(len(self.filter.pending_quarantine_entries), 0)

    def test_quarantine_flushed_in_order_before_important_entry(self) -> None:
        self.filter.check_entry("Chapter 4 Quadratic Equation Master Plan")
        entries = [
            {"content": "Welcome to chapter 5 arithmetic progressions", "source_type": "transcript", "importance_flags": []},
            {"content": "Arithmetic progressions common difference and nth term", "source_type": "transcript", "importance_flags": []},
            {"content": "Exam will cover all problems next week", "source_type": "transcript", "importance_flags": ["exam_relevant"]}
        ]
        kept = self.filter.filter_batch_entries(entries)
        self.assertEqual(len(kept), 3)
        self.assertEqual(kept[0]["content"], "Welcome to chapter 5 arithmetic progressions")
        self.assertEqual(kept[1]["content"], "Arithmetic progressions common difference and nth term")
        self.assertEqual(kept[2]["content"], "Exam will cover all problems next week")
        self.assertEqual(len(self.filter.pending_quarantine_entries), 0)


class TestRecordedTimelineFiltering(unittest.TestCase):
    def test_yt_test_timeline_filtering(self) -> None:
        candidates = [
            Path.home() / "output" / "yt_test" / "timeline.jsonl",
            Path.cwd() / "output" / "yt_test" / "timeline.jsonl",
        ]
        timeline_path = next((p for p in candidates if p.exists()), None)
        if timeline_path is None:
            self.skipTest("yt_test output not available on disk")

        filt = ContextDriftFilter(
            drift_threshold=0.50,
            sustain_consecutive=3,
            min_words=8
        )

        entries = []
        with open(timeline_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    entries.append(json.loads(stripped))

        kept = filt.filter_batch_entries(entries)
        contents = [e.get("content", "") for e in kept]

        for c in contents:
            self.assertNotIn("The Importance of Effective Communication", c)
            self.assertFalse(is_refusal_text(c))

        has_math = any("quadratic equation" in c.lower() or "factorization" in c.lower() for c in contents)
        self.assertTrue(has_math)


if __name__ == "__main__":
    unittest.main()
