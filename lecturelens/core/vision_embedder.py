from pathlib import Path
from typing import Any, Optional
import urllib.request
import numpy as np

from lecturelens.core.logging_setup import get_logger

logger = get_logger("lecturelens.vision_embedder")

DEFAULT_MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx"


class VisionEmbedder:
    def __init__(
        self,
        model_path: Optional[str] = None,
        auto_download: bool = True
    ) -> None:
        self.session: Optional[Any] = None
        self.input_name: Optional[str] = None
        self.auto_download: bool = auto_download

        if model_path is None:
            cache_dir = Path.home() / ".cache" / "lecturelens" / "models"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.model_path = cache_dir / "squeezenet1.1.onnx"
        else:
            self.model_path = Path(model_path)

        self._initialize_session()

    def _ensure_model_file(self) -> bool:
        if self.model_path.exists() and self.model_path.stat().st_size > 1000000:
            return True
        if not self.auto_download:
            return False
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.model_path.with_suffix(".tmp")
            urllib.request.urlretrieve(DEFAULT_MODEL_URL, temp_path)
            temp_path.replace(self.model_path)
            return True
        except Exception as error:
            logger.warning("Failed to download vision embedder model: %s", error)
            return False

    def _initialize_session(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            logger.warning("onnxruntime not installed, vision embedder disabled")
            return

        if not self._ensure_model_file():
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
        except Exception as error:
            logger.warning("Failed to initialize vision embedder session: %s", error)
            self.session = None

    def is_available(self) -> bool:
        return self.session is not None

    def preprocess(self, image_array: np.ndarray) -> np.ndarray:
        try:
            from PIL import Image
            if image_array.ndim == 2:
                pil_img = Image.fromarray(image_array).convert("RGB")
            elif image_array.shape[2] == 4:
                pil_img = Image.fromarray(image_array).convert("RGB")
            else:
                pil_img = Image.fromarray(image_array)
            resized = pil_img.resize((224, 224))
            arr = np.array(resized, dtype=np.float32) / 255.0
        except Exception:
            h, w = image_array.shape[:2]
            r_idx = np.linspace(0, h - 1, 224).astype(int)
            c_idx = np.linspace(0, w - 1, 224).astype(int)
            if image_array.ndim == 2:
                sampled = image_array[np.ix_(r_idx, c_idx)]
                arr = np.repeat(sampled[:, :, np.newaxis], 3, axis=2).astype(np.float32) / 255.0
            else:
                sampled = image_array[np.ix_(r_idx, c_idx, np.arange(min(3, image_array.shape[2])))]
                arr = sampled.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (arr - mean) / std
        transposed = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(transposed, axis=0).astype(np.float32)

    def embed(self, image_array: np.ndarray) -> Optional[np.ndarray]:
        if self.session is None or self.input_name is None:
            return None
        try:
            tensor = self.preprocess(image_array)
            raw_output = self.session.run(None, {self.input_name: tensor})[0].flatten()
            norm = np.linalg.norm(raw_output)
            if norm == 0.0:
                return raw_output
            return raw_output / norm
        except Exception as error:
            logger.warning("Vision embedding extraction failed: %s", error)
            return None

    def similarity(self, first_vec: np.ndarray, second_vec: np.ndarray) -> float:
        if first_vec.shape != second_vec.shape:
            return 0.0
        dot_product = float(np.dot(first_vec, second_vec))
        return max(-1.0, min(1.0, dot_product))
