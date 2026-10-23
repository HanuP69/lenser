from typing import Any, Optional

import numpy as np

from lecturelens.core.handwriting import detect_board_surface

# The parts of the screen that are compared to find out where it
# changed. More parts than the ones that decide whether it changed, so
# that the part is found more closely.
CHANGED_REGION_GRID: tuple[int, int] = (12, 16)


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


def compute_block_similarities(
    first_gray: np.ndarray,
    second_gray: np.ndarray,
    grid: tuple[int, int] = (6, 8)
) -> list[float]:
    """How alike every part of two pictures is.

    The middle value of the parts is what tells whether the pictures
    changed at all. The parts themselves say **where** they changed,
    which is what the model needs, because a small part of a picture is
    enough to tell it about the part that changed.
    """
    if first_gray.shape != second_gray.shape or first_gray.size == 0:
        return []
    if first_gray.ndim == 3:
        first_gray = first_gray.mean(axis=2).astype(np.uint8)
        second_gray = second_gray.mean(axis=2).astype(np.uint8)
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
            return [1.0]
        if norm1 == 0.0 or norm2 == 0.0:
            return [0.0]
        return [float(np.dot(b1, b2) / (norm1 * norm2))]
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
    return similarities


def compute_spatial_block_similarity(
    first_gray: np.ndarray,
    second_gray: np.ndarray,
    grid: tuple[int, int] = (6, 8)
) -> float:
    similarities = compute_block_similarities(first_gray, second_gray, grid)
    if not similarities:
        return 0.0
    return float(np.median(similarities)) if similarities else 1.0


def changed_region(
    first_gray: np.ndarray,
    second_gray: np.ndarray,
    grid: tuple[int, int] = (12, 16),
    similarity_limit: float = 0.98,
    padding_blocks: int = 1
) -> Optional[tuple[int, int, int, int]]:
    """The part of the picture that changed, as a box.

    The box is in the coordinates of the picture that was given, so a
    small box can be cut out of the whole picture. The model is then shown
    that part and not the whole screen, which is what makes the request
    small and the small text readable.

    The parts of the picture are found first, and then the exact pixels
    that changed are looked for inside them. A part on its own is big,
    and the request would still be a large part of the screen.

    None is returned when nothing changed, or when the whole picture
    changed, because then the whole picture is the part that changed.
    """
    if first_gray.shape != second_gray.shape or first_gray.size == 0:
        return None
    if first_gray.ndim == 3:
        first_gray = first_gray.mean(axis=2).astype(np.uint8)
        second_gray = second_gray.mean(axis=2).astype(np.uint8)
    similarities = compute_block_similarities(first_gray, second_gray, grid)
    if not similarities:
        return None
    grid_rows, grid_cols = grid
    changed = [
        index for index, value in enumerate(similarities) if value < similarity_limit
    ]
    if not changed:
        return None
    if len(changed) == len(similarities):
        # Everything changed, so the whole picture is the part that
        # changed.
        return None

    height, width = first_gray.shape
    block_h = max(1, height // grid_rows)
    block_w = max(1, width // grid_cols)

    rows = [index // grid_cols for index in changed]
    columns = [index % grid_cols for index in changed]
    top = max(0, (min(rows) - padding_blocks) * block_h)
    bottom = min(height, (max(rows) + padding_blocks + 1) * block_h)
    left = max(0, (min(columns) - padding_blocks) * block_w)
    right = min(width, (max(columns) + padding_blocks + 1) * block_w)

    exact = _exact_changed_box(first_gray, second_gray, top, bottom, left, right)
    if exact is not None:
        return exact
    return (left, top, right, bottom)


def _exact_changed_box(
    first_gray: np.ndarray,
    second_gray: np.ndarray,
    top: int,
    bottom: int,
    left: int,
    right: int
) -> Optional[tuple[int, int, int, int]]:
    """The exact pixels that changed, as a box.

    The pixels are looked for inside the parts that changed, so the box
    is as small as the change and not as big as the part it is in.
    """
    first = first_gray[top:bottom, left:right].astype(np.int16)
    second = second_gray[top:bottom, left:right].astype(np.int16)
    if first.size == 0:
        return None
    difference = np.abs(first - second)
    rows = np.any(difference > 8, axis=1)
    columns = np.any(difference > 8, axis=0)
    if not rows.any() or not columns.any():
        return None
    row_indexes = np.flatnonzero(rows)
    column_indexes = np.flatnonzero(columns)
    return (
        left + int(column_indexes[0]),
        top + int(row_indexes[0]),
        left + int(column_indexes[-1]) + 1,
        top + int(row_indexes[-1]) + 1
    )



def compute_difference_hash(gray_thumbnail: np.ndarray) -> int:
    if gray_thumbnail.ndim == 3:
        gray_thumbnail = gray_thumbnail.mean(axis=2).astype(np.uint8)
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


def compute_edge_count(image_array: np.ndarray) -> int:
    if image_array is None or image_array.size == 0:
        return 0
    if image_array.ndim == 3:
        if image_array.shape[2] == 4:
            gray = image_array[:, :, :3].mean(axis=2).astype(np.uint8)
        else:
            gray = image_array.mean(axis=2).astype(np.uint8)
    else:
        gray = image_array
    try:
        import cv2
        edges = cv2.Canny(gray, 50, 150)
        return int(np.count_nonzero(edges))
    except Exception:
        diff_y = np.abs(gray[1:, :] - gray[:-1, :])
        diff_x = np.abs(gray[:, 1:] - gray[:, :-1])
        return int(np.count_nonzero(diff_y > 20) + np.count_nonzero(diff_x > 20))


def is_content_building_on_top(
    prev_text: str,
    curr_text: str,
    prev_edges: int,
    curr_edges: int
) -> tuple[bool, str]:
    prev_clean = " ".join(prev_text.lower().split())
    curr_clean = " ".join(curr_text.lower().split())

    if len(prev_clean) < 6 and len(curr_clean) < 6:
        if curr_edges >= int(prev_edges * 0.85):
            return True, "diagram_edges_growing"
        return False, "diagram_edges_dropped"

    if len(prev_clean) < 6 and len(curr_clean) >= 6:
        return True, "new_text_introduced"

    if len(curr_clean) < int(len(prev_clean) * 0.60):
        return False, "text_erased_len_drop"

    if prev_clean in curr_clean or curr_clean.startswith(prev_clean[:min(len(prev_clean), 25)]):
        return True, "substring_prefix_expansion"

    import difflib
    ratio = difflib.SequenceMatcher(None, prev_clean, curr_clean).ratio()
    prev_words = set(prev_clean.split())
    curr_words = set(curr_clean.split())
    token_overlap = len(prev_words.intersection(curr_words)) / max(1, len(prev_words))

    if ratio >= 0.60 or token_overlap >= 0.60:
        return True, "fuzzy_text_match"

    return False, "unrelated_content_change"


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
        self.last_accepted_edges: int = 0
        self.previous_poll: Optional[np.ndarray] = None
        self.last_accept_time: float = 0.0
        self.settle_polls: int = 0
        self.detection_timestamp: float = 0.0
        self._last_observed_time: Optional[float] = None
        # The part of the picture that changed, in the coordinates of the
        # small picture. The model is shown that part and not the whole
        # screen.
        self.last_changed_region: Optional[tuple[int, int, int, int]] = None

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

        is_board, _ = detect_board_surface(thumbnail)
        current_embedding: Optional[np.ndarray] = None
        if not is_board and self.embedder is not None and (not hasattr(self.embedder, "is_available") or self.embedder.is_available()):
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
        if self.last_accepted is not None:
            self.last_changed_region = changed_region(
                self.last_accepted, thumbnail, CHANGED_REGION_GRID
            )
        else:
            self.last_changed_region = None
        self.last_accepted = thumbnail
        self.last_accepted_edges = compute_edge_count(thumbnail)
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
