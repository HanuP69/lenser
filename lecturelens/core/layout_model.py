"""Finding the parts of a screen.

The screen of a lecture is not only text. It has a title, a list, a
table, a diagram, a formula, and the parts of the window around them
that have nothing to do with the lecture.

The model that is used here was made for the layout of documents. A
screen is not a document, but a slide is close enough to one that the
parts are found well. It runs on the processor and takes a second or
two for a picture of a screen.
"""

import hashlib
import os
import threading
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from lecturelens.core.logging_setup import get_logger
from lecturelens.utils.hashing import sha256_of_file

logger = get_logger("lecturelens.layout_model")

# The model of the layout of a document, in the form of ONNX. It is
# published under the Apache licence, so it can be used in a product
# that is closed.
MODEL_REPOSITORY = "Bei0001/PP-DocLayoutV3-ONNX"
MODEL_FILES = ("PP-DocLayoutV3.onnx", "PP-DocLayoutV3.onnx.data")
MODEL_URL = "https://huggingface.co/{repository}/resolve/main/{name}"

# The digests of the files that are downloaded. The file is run inside
# the process, so an unverified download would be a hole.
EXPECTED_MODEL_DIGESTS = {
    "PP-DocLayoutV3.onnx": "c0721928ff08741bb208ebed539c77170db5234a68cb7e546e6cc9bc172a695b",
    "PP-DocLayoutV3.onnx.data": "34df3e4b79d7bbbf82abce1b4f3cde3d540fa57ad42ec8905c352b97c408d437",
}

# The size of the picture the model takes.
MODEL_INPUT_SIZE = (800, 800)

# The kinds of part, and what is done with them.
KIND_FIGURE = "figure"
KIND_CODE = "code"
KIND_FORMULA = "formula"
KIND_TEXT = "text"
KIND_TITLE = "title"
# The label that is written under a picture. It is not the title of the
# slide, so it is kept apart from it.
KIND_FIGURE_TITLE = "figure_title"
KIND_CHROME = "chrome"
KIND_OTHER = "other"

# The names of the parts of the model, and what they mean here.
KIND_BY_LABEL = {
    "image": KIND_FIGURE,
    "chart": KIND_FIGURE,
    "seal": KIND_CHROME,
    "algorithm": KIND_CODE,
    "display_formula": KIND_FORMULA,
    "inline_formula": KIND_FORMULA,
    "formula_number": KIND_FORMULA,
    "doc_title": KIND_TITLE,
    "paragraph_title": KIND_TITLE,
    "figure_title": KIND_FIGURE_TITLE,
    "abstract": KIND_TEXT,
    "text": KIND_TEXT,
    "table": KIND_TEXT,
    "aside_text": KIND_TEXT,
    "content": KIND_TEXT,
    "reference_content": KIND_TEXT,
    "reference": KIND_TEXT,
    "footnote": KIND_TEXT,
    "vision_footnote": KIND_TEXT,
    "number": KIND_TEXT,
    "vertical_text": KIND_TEXT,
    "header": KIND_CHROME,
    "header_image": KIND_CHROME,
    "footer": KIND_CHROME,
    "footer_image": KIND_CHROME,
}


@dataclass(frozen=True)
class Region:
    """One part of a picture."""

    kind: str
    box: tuple[int, int, int, int]
    score: float
    label: str

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)


def expected_model_digest(name: str) -> str:
    """The digest of a file, overridable for users who mirror it."""
    override = os.environ.get("LECTURELENS_LAYOUT_MODEL_SHA256", "").strip().lower()
    if override:
        return override
    return EXPECTED_MODEL_DIGESTS.get(name, "")



class LayoutModel:
    """The model that finds the parts of a picture."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        auto_download: bool = True
    ) -> None:
        self.session: Optional[Any] = None
        self.input_name: Optional[str] = None
        self.auto_download = auto_download
        self.labels: list[str] = []
        if model_path is None:
            cache_dir = Path.home() / ".cache" / "lecturelens" / "models"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.model_path = cache_dir / "PP-DocLayoutV3.onnx"
        else:
            self.model_path = Path(model_path)
        self._initialize()

    # The model is made of two files, and the second one holds the
    # weights.
    @property
    def data_path(self) -> Path:
        return self.model_path.with_suffix(self.model_path.suffix + ".data")

    def _ensure_model_files(self) -> bool:
        if self.model_path.exists() and self.data_path.exists():
            if self._verify(self.model_path) and self._verify(self.data_path):
                return True
        if not self.auto_download:
            return False
        for name in MODEL_FILES:
            target = self.model_path if name.endswith(".onnx") else self.data_path
            if not self._download(name, target):
                return False
        return self._verify(self.model_path) and self._verify(self.data_path)

    def _verify(self, path: Path) -> bool:
        expected = expected_model_digest(path.name)
        if not expected:
            return True
        try:
            return sha256_of_file(path) == expected
        except OSError:
            return False

    def _download(self, name: str, target: Path) -> bool:
        url = MODEL_URL.format(repository=MODEL_REPOSITORY, name=name)
        temp_path = target.with_name(f"{target.name}.{uuid.uuid4().hex[:8]}.part")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                with open(temp_path, "wb") as file_handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        file_handle.write(chunk)
            temp_path.replace(target)
            return True
        except Exception as error:
            logger.warning("the layout model could not be downloaded: %s", error)
            temp_path.unlink(missing_ok=True)
            return False

    def _initialize(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            logger.warning("onnxruntime is not installed, the parts of the screen are not found")
            return
        if not self._ensure_model_files():
            return
        try:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(
                str(self.model_path),
                options,
                providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            self.labels = _read_labels()
        except Exception as error:
            logger.warning("the layout model could not be started: %s", error)
            self.session = None

    def is_available(self) -> bool:
        return self.session is not None

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """The picture as the model takes it."""
        from PIL import Image

        if image.ndim == 2:
            picture = Image.fromarray(image).convert("RGB")
        elif image.shape[2] == 4:
            picture = Image.fromarray(image).convert("RGB")
        else:
            picture = Image.fromarray(image)
        resized = picture.resize(MODEL_INPUT_SIZE, Image.BILINEAR)
        array = np.array(resized, dtype=np.float32) / 255.0
        tensor = np.transpose(array, (2, 0, 1))[None, ...]
        return tensor.astype(np.float32)

    def detect(
        self,
        image: np.ndarray,
        threshold: float = 0.4,
        figure_threshold: float = 0.65
    ) -> list[Region]:
        """The parts of a picture."""
        if self.session is None:
            return []
        try:
            tensor = self.preprocess(image)
            outputs = self.session.run(None, {self.input_name: tensor})
        except Exception as error:
            logger.warning("the layout model failed: %s", error)
            return []
        logits, boxes = outputs[0], outputs[1]
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits[0], -30, 30)))
        labels = np.argmax(scores, axis=-1)
        confidence = np.max(scores, axis=-1)
        height, width = image.shape[:2]
        regions: list[Region] = []
        for index in range(labels.shape[0]):
            if confidence[index] < threshold:
                continue
            label_index = int(labels[index])
            label = self.labels[label_index] if label_index < len(self.labels) else "other"
            kind = KIND_BY_LABEL.get(label, KIND_OTHER)
            if kind == KIND_FIGURE and confidence[index] < figure_threshold:
                continue
            cx, cy, box_width, box_height = [float(value) for value in boxes[0][index]]
            left = int((cx - box_width / 2) * width)
            top = int((cy - box_height / 2) * height)
            right = int((cx + box_width / 2) * width)
            bottom = int((cy + box_height / 2) * height)
            left = max(0, min(width - 1, left))
            top = max(0, min(height - 1, top))
            right = max(left + 1, min(width, right))
            bottom = max(top + 1, min(height, bottom))
            regions.append(
                Region(
                    kind=kind,
                    box=(left, top, right, bottom),
                    score=round(float(confidence[index]), 4),
                    label=label
                )
            )
        regions.sort(key=lambda region: (region.box[1], region.box[0]))
        return regions


def _read_labels() -> list[str]:
    """The names of the parts, in the order the model gives them."""
    names = [
        "abstract", "algorithm", "aside_text", "chart", "content",
        "display_formula", "doc_title", "figure_title", "footer",
        "footer_image", "footnote", "formula_number", "header",
        "header_image", "image", "inline_formula", "number",
        "paragraph_title", "reference", "reference_content", "seal",
        "table", "text", "vertical_text", "vision_footnote"
    ]
    return names


_SHARED: Optional[LayoutModel] = None
_SHARED_LOCK = threading.Lock()


def get_shared_layout_model() -> LayoutModel:
    """The model, made once."""
    global _SHARED
    if _SHARED is None:
        with _SHARED_LOCK:
            if _SHARED is None:
                _SHARED = LayoutModel()
    return _SHARED
