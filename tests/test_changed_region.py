import unittest

import numpy as np
from PIL import Image

from lecturelens.agents.screen_agent import ScreenFrame, crop_of_the_change
from lecturelens.core.image_utils import changed_region, compute_block_similarities


def make_picture(seed: int, width: int = 1920, height: int = 1080) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 60, size=(height, width), dtype=np.uint8)


def small_of(picture: np.ndarray) -> np.ndarray:
    return np.array(
        Image.fromarray(picture).convert("L").resize((640, 360), Image.BILINEAR)
    )


def make_frame(picture: np.ndarray) -> ScreenFrame:
    height, width = picture.shape[:2]
    raw = Image.fromarray(picture).convert("RGBA").tobytes()
    return ScreenFrame(
        raw_bgra=raw,
        width=width,
        height=height,
        thumbnail=small_of(picture)
    )


class TestTheChangedRegion(unittest.TestCase):
    def test_nothing_that_changed_has_no_region(self) -> None:
        picture = make_picture(1)
        self.assertIsNone(changed_region(picture, picture.copy()))

    def test_a_part_that_changed_is_found(self) -> None:
        first = make_picture(1)
        second = first.copy()
        second[400:700, 1200:1800] = 200
        region = changed_region(first, second)
        self.assertIsNotNone(region)
        left, top, right, bottom = region
        self.assertLessEqual(left, 1200)
        self.assertLessEqual(top, 400)
        self.assertGreaterEqual(right, 1800)
        self.assertGreaterEqual(bottom, 700)

    def test_a_part_that_changed_is_a_small_part_of_the_screen(self) -> None:
        first = make_picture(1)
        second = first.copy()
        second[500:560, 900:960] = 255
        region = changed_region(first, second)
        self.assertIsNotNone(region)
        left, top, right, bottom = region
        share = ((right - left) * (bottom - top)) / (1920 * 1080)
        self.assertLess(share, 0.2)

    def test_a_screen_that_changed_completely_has_no_small_part(self) -> None:
        first = make_picture(1)
        second = make_picture(2)
        self.assertIsNone(changed_region(first, second))

    def test_the_parts_are_the_same_when_nothing_changed(self) -> None:
        picture = make_picture(3)
        values = compute_block_similarities(picture, picture.copy())
        self.assertTrue(all(value > 0.99 for value in values))


class TestTheCrop(unittest.TestCase):
    def test_the_crop_is_much_smaller_than_the_screen(self) -> None:
        first = make_picture(1)
        second = first.copy()
        second[400:700, 1200:1800] = 200
        region = changed_region(small_of(first), small_of(second))
        self.assertIsNotNone(region)

        frame = make_frame(second)
        crop = crop_of_the_change(frame, region)
        self.assertIsNotNone(crop)
        whole = len(Image.fromarray(second).convert("RGB").tobytes())
        self.assertLess(len(crop), whole // 2)

    def test_the_crop_holds_the_part_that_changed(self) -> None:
        first = make_picture(1)
        second = first.copy()
        second[400:700, 1200:1800] = 200
        region = changed_region(small_of(first), small_of(second))
        self.assertIsNotNone(region)

        frame = make_frame(second)
        crop = crop_of_the_change(frame, region)
        self.assertIsNotNone(crop)
        crop_image = Image.open(__import__("io").BytesIO(crop)).convert("L")
        array = np.array(crop_image)
        # The part that changed is bright, the rest of the screen is dark.
        self.assertGreater(float(array.max()), 150)

    def test_no_region_means_no_crop(self) -> None:
        frame = make_frame(make_picture(1))
        self.assertIsNone(crop_of_the_change(frame, None))


if __name__ == "__main__":
    unittest.main()
