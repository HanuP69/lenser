import json
import unittest
from pathlib import Path

from eval import score
from eval.extract_readings import frame_at, read_frame


class TestTheManifest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = score.load_manifest()

    def test_the_manifest_is_readable(self) -> None:
        self.assertIn("videos", self.manifest)
        self.assertGreater(len(self.manifest["videos"]), 3)

    def test_every_video_says_what_it_is_for(self) -> None:
        for entry in self.manifest["videos"]:
            self.assertIn("archetype", entry, entry["id"])
            self.assertIn("ground_truth", entry, entry["id"])
            self.assertIn("why", entry, entry["id"])

    def test_every_video_is_at_the_same_height(self) -> None:
        heights = {entry["height"] for entry in self.manifest["videos"]}
        self.assertEqual(
            len(heights),
            1,
            f"the videos are at different heights {heights}. The size of the text in "
            f"the picture decides how well it can be read, so a video at 360p and "
            f"one at 1080p differ more than the systems do."
        )

    def test_the_talking_head_expects_nothing(self) -> None:
        talking_heads = [
            entry for entry in self.manifest["videos"]
            if entry["archetype"] == "E_talking_head"
        ]
        self.assertTrue(talking_heads, "there is no video that expects nothing")
        for entry in talking_heads:
            self.assertEqual(entry["ground_truth"]["kind"], "expect_nothing")

    def test_the_notes_are_where_they_are_said(self) -> None:
        root = Path(self.manifest["data_root"])
        for entry in self.manifest["videos"]:
            for source in entry["ground_truth"].get("sources", []):
                self.assertTrue(
                    (root / source).exists(),
                    f"the notes of {entry['id']} are not there: {source}"
                )


class TestTheWords(unittest.TestCase):
    def test_the_words_are_counted_in_both_directions(self) -> None:
        precision, recall, f1 = score.word_scores("alpha beta gamma", "alpha beta delta")
        self.assertAlmostEqual(precision, 2 / 3, places=3)
        self.assertAlmostEqual(recall, 2 / 3, places=3)
        self.assertGreater(f1, 0.6)

    def test_a_slide_that_shows_half_of_its_text_still_counts(self) -> None:
        """A slide is often shown bit by bit while it is being animated.

        The character error rate punishes that heavily, so the share of
        the words is used.
        """
        precision, recall, f1 = score.word_scores("title", "title bullet one bullet two")
        self.assertAlmostEqual(precision, 1.0, places=3)
        self.assertAlmostEqual(recall, 0.2, places=3)
        self.assertGreater(f1, 0.3)

    def test_the_noise_is_left_out(self) -> None:
        precision, _, _ = score.word_scores("title the of and", "title")
        self.assertAlmostEqual(precision, 1.0, places=3)

    def test_two_empty_texts_are_the_same(self) -> None:
        self.assertEqual(score.word_scores("", ""), (1.0, 1.0, 1.0))


class TestTheSignal(unittest.TestCase):
    def test_text_that_is_new_is_new(self) -> None:
        readings = [
            {"t": 0.0, "text": "alpha beta"},
            {"t": 1.0, "text": "alpha beta"},
            {"t": 2.0, "text": "gamma delta"},
        ]
        signal = score.novelty_signal(readings, window=2)
        self.assertEqual(signal[0], 1.0)
        self.assertEqual(signal[1], 0.0)
        self.assertAlmostEqual(signal[2], 1.0)

    def test_a_picture_without_text_is_not_new(self) -> None:
        readings = [{"t": 0.0, "text": ""}, {"t": 1.0, "text": "alpha"}]
        signal = score.novelty_signal(readings)
        self.assertEqual(signal[0], 0.0)

    def test_the_moments_are_kept_apart(self) -> None:
        readings = [
            {"t": 0.0, "text": "alpha beta gamma delta"},
            {"t": 1.0, "text": "epsilon zeta eta theta"},
            {"t": 2.0, "text": "iota kappa lambda mu"},
        ]
        # With no gap between them, every new text is a moment.
        fired = score.fire_times(readings, threshold=0.5, min_gap_seconds=0.0)
        self.assertEqual(fired, [0.0, 1.0, 2.0])
        # With a gap, the texts that come too soon after a pick are left.
        fired = score.fire_times(readings, threshold=0.5, min_gap_seconds=5.0)
        self.assertEqual(fired, [0.0])


class TestTheChanges(unittest.TestCase):
    def test_the_moments_where_the_page_changes_are_found(self) -> None:
        readings = [
            {"t": 0.0, "text": "a"},
            {"t": 1.0, "text": "a"},
            {"t": 2.0, "text": "b"},
            {"t": 3.0, "text": "b"},
            {"t": 4.0, "text": "c"},
        ]
        matches = [0, 0, 1, 1, 2]
        changes = score.change_times(readings, matches)
        self.assertEqual(changes, [2.0, 4.0])

    def test_a_picture_without_a_page_is_not_a_change(self) -> None:
        readings = [{"t": 0.0, "text": "a"}, {"t": 1.0, "text": "b"}]
        changes = score.change_times(readings, [0, None])
        self.assertEqual(changes, [])

    def test_the_moments_are_scored(self) -> None:
        result = score.score_trigger([2.1, 4.2], [2.0, 4.0], tolerance=1.0)
        self.assertAlmostEqual(result["recall"], 1.0, places=3)
        self.assertAlmostEqual(result["precision"], 1.0, places=3)

    def test_a_pick_that_is_not_a_change_costs(self) -> None:
        result = score.score_trigger([2.0, 10.0, 11.0], [2.0], tolerance=1.0)
        self.assertAlmostEqual(result["precision"], 1 / 3, places=3)
        self.assertAlmostEqual(result["recall"], 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
