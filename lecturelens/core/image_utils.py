from typing import Optional

import numpy as np


def compute_changed_fraction(
    first_gray: np.ndarray,
    second_gray: np.ndarray,
    pixel_delta: int = 25
) -> float:
    if first_gray.shape != second_gray.shape:
        return 1.0
    difference = np.abs(first_gray.astype(np.int16) - second_gray.astype(np.int16))
    if difference.size == 0:
        return 0.0
    changed_pixels = int(np.count_nonzero(difference > pixel_delta))
    return changed_pixels / float(difference.size)


def compute_difference_hash(gray_thumbnail: np.ndarray) -> int:
    height, width = gray_thumbnail.shape
    row_indices = np.linspace(0, height - 1, 8).astype(int)
    column_indices = np.linspace(0, width - 1, 9).astype(int)
    sampled = gray_thumbnail[np.ix_(row_indices, column_indices)].astype(np.int16)
    hash_value = 0
    for row_index in range(8):
        for column_index in range(8):
            hash_value = hash_value << 1
            if sampled[row_index, column_index + 1] > sampled[row_index, column_index]:
                hash_value = hash_value | 1
    return hash_value


def hamming_distance(first_hash: int, second_hash: int) -> int:
    return bin(first_hash ^ second_hash).count("1")


class SlideChangeGate:
    def __init__(
        self,
        change_threshold: float,
        pixel_delta: int,
        min_interval_seconds: float,
        max_settle_polls: int = 4
    ) -> None:
        self.change_threshold: float = change_threshold
        self.pixel_delta: int = pixel_delta
        self.min_interval_seconds: float = min_interval_seconds
        self.max_settle_polls: int = max_settle_polls
        self.last_accepted: Optional[np.ndarray] = None
        self.previous_poll: Optional[np.ndarray] = None
        self.last_accept_time: float = 0.0
        self.settle_polls: int = 0

    def evaluate(self, thumbnail: np.ndarray, now_seconds: float) -> bool:
        previous_poll = self.previous_poll
        self.previous_poll = thumbnail

        if self.last_accepted is None:
            return self.accept(thumbnail, now_seconds)

        changed_fraction = compute_changed_fraction(
            self.last_accepted, thumbnail, self.pixel_delta
        )
        if changed_fraction < self.change_threshold:
            self.settle_polls = 0
            return False

        self.settle_polls += 1
        poll_motion = 0.0
        if previous_poll is not None:
            poll_motion = compute_changed_fraction(previous_poll, thumbnail, self.pixel_delta)

        is_stable = poll_motion < (self.change_threshold / 2.0)
        has_settled = is_stable or self.settle_polls >= self.max_settle_polls
        interval_elapsed = (now_seconds - self.last_accept_time) >= self.min_interval_seconds
        if has_settled and interval_elapsed:
            return self.accept(thumbnail, now_seconds)
        return False

    def accept(self, thumbnail: np.ndarray, now_seconds: float) -> bool:
        self.last_accepted = thumbnail
        self.last_accept_time = now_seconds
        self.settle_polls = 0
        return True
