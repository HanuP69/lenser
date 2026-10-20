from typing import Any, Optional

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


def compute_spatial_block_similarity(
    first_gray: np.ndarray,
    second_gray: np.ndarray,
    grid: tuple[int, int] = (6, 8)
) -> float:
    if first_gray.shape != second_gray.shape or first_gray.size == 0:
        return 0.0
    grid_rows, grid_cols = grid
    height, width = first_gray.shape
    block_h = height // grid_rows
    block_w = width // grid_cols
    if block_h == 0 or block_w == 0:
        b1 = first_gray.astype(np.float32).flatten()
        b2 = second_gray.astype(np.float32).flatten()
        norm1 = float(np.linalg.norm(b1))
        norm2 = float(np.linalg.norm(b2))
        if norm1 == 0.0 and norm2 == 0.0:
            return 1.0
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return float(np.dot(b1, b2) / (norm1 * norm2))
    similarities: list[float] = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            b1 = first_gray[r * block_h:(r + 1) * block_h, c * block_w:(c + 1) * block_w].astype(np.float32).flatten()
            b2 = second_gray[r * block_h:(r + 1) * block_h, c * block_w:(c + 1) * block_w].astype(np.float32).flatten()
            norm1 = float(np.linalg.norm(b1))
            norm2 = float(np.linalg.norm(b2))
            if norm1 == 0.0 and norm2 == 0.0:
                similarities.append(1.0)
            elif norm1 == 0.0 or norm2 == 0.0:
                similarities.append(0.0)
            else:
                similarities.append(float(np.dot(b1, b2) / (norm1 * norm2)))
    return float(np.median(similarities)) if similarities else 1.0


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
        max_settle_polls: int = 4,
        embedder: Optional[Any] = None,
        similarity_threshold: float = 0.95
    ) -> None:
        self.change_threshold: float = change_threshold
        self.pixel_delta: int = pixel_delta
        self.min_interval_seconds: float = min_interval_seconds
        self.max_settle_polls: int = max_settle_polls
        self.embedder: Optional[Any] = embedder
        self.similarity_threshold: float = similarity_threshold
        self.last_accepted: Optional[np.ndarray] = None
        self.last_accepted_embedding: Optional[np.ndarray] = None
        self.previous_poll: Optional[np.ndarray] = None
        self.last_accept_time: float = 0.0
        self.settle_polls: int = 0
        self.detection_timestamp: float = 0.0
        self._last_observed_time: Optional[float] = None

    def evaluate(self, thumbnail: np.ndarray, now_seconds: float) -> bool:
        previous_poll = self.previous_poll
        self.previous_poll = thumbnail

        if self._last_observed_time is not None and now_seconds < self._last_observed_time:
            # The caller switched clock base (for example a wall-clock epoch
            # after a monotonic reading). Rebase onto the new base instead of
            # computing a negative interval, which would otherwise reject every
            # later frame for the rest of the session.
            self.last_accept_time = now_seconds
        self._last_observed_time = now_seconds

        if self.last_accepted is None:
            return self.accept(thumbnail, now_seconds)

        changed_fraction = compute_changed_fraction(
            self.last_accepted, thumbnail, self.pixel_delta
        )
        if changed_fraction < self.change_threshold:
            self.settle_polls = 0
            return False

        current_embedding: Optional[np.ndarray] = None
        if self.embedder is not None and (not hasattr(self.embedder, "is_available") or self.embedder.is_available()):
            current_embedding = self.embedder.embed(thumbnail)
            if current_embedding is not None and self.last_accepted_embedding is not None:
                sim = self.embedder.similarity(self.last_accepted_embedding, current_embedding)
                if sim >= self.similarity_threshold:
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
            return self.accept(thumbnail, now_seconds, current_embedding)
        return False

    def accept(
        self,
        thumbnail: np.ndarray,
        now_seconds: float,
        embedding: Optional[np.ndarray] = None
    ) -> bool:
        self.last_accepted = thumbnail
        if embedding is not None:
            self.last_accepted_embedding = embedding
        elif self.embedder is not None:
            self.last_accepted_embedding = self.embedder.embed(thumbnail)
        else:
            self.last_accepted_embedding = None
        self.last_accept_time = now_seconds
        self.detection_timestamp = now_seconds
        self.settle_polls = 0
        return True
