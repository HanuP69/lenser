import json
import re
import urllib.request
from typing import Any, Optional, Sequence

import numpy as np

from lecturelens.core.logging_setup import get_logger

logger = get_logger("lecturelens.drift_filter")

REFUSAL_PHRASES: tuple[str, ...] = (
    "please provide the presentation slide",
    "please provide the presentation slide image",
    "no text or diagram was visible",
    "i need an image of the presentation slide",
    "no text is currently visible",
    "please provide an image",
    "[name]",
    "[insert",
    "[presentation",
    "[title",
    "presented by: [",
)


def is_refusal_text(text: str) -> bool:
    lower = text.lower().strip()
    for phrase in REFUSAL_PHRASES:
        if phrase in lower:
            return True
    return False


def normalize_vector(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm <= 1e-9:
        return v
    return v / norm


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= 1e-9 or n2 <= 1e-9:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


class ContextDriftFilter:
    def __init__(
        self,
        ollama_url: str = "http://127.0.0.1:11434",
        embed_model: str = "nomic-embed-text",
        drift_threshold: float = 0.50,
        sustain_consecutive: int = 3,
        min_words: int = 8,
        alpha: float = 0.85,
        embedder: Optional[Any] = None,
        session_topic: Optional[str] = None
    ) -> None:
        self.ollama_url: str = ollama_url.rstrip("/")
        self.embed_model: str = embed_model
        self.drift_threshold: float = drift_threshold
        self.sustain_consecutive: int = sustain_consecutive
        self.min_words: int = min_words
        self.alpha: float = alpha
        self.embedder: Optional[Any] = embedder
        self.session_topic: Optional[str] = session_topic

        self.running_anchor: Optional[np.ndarray] = None
        self.quarantine_buffer: list[tuple[str, np.ndarray]] = []
        self.accepted_count: int = 0
        self.drift_count: int = 0

        if session_topic is not None and len(session_topic.strip()) > 0:
            initial_vec = self.embed_text(session_topic.strip())
            if initial_vec is not None:
                self.running_anchor = initial_vec

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        clean_text = text.strip()
        if len(clean_text) == 0:
            return None

        if self.embedder is not None and hasattr(self.embedder, "embed"):
            try:
                vec = self.embedder.embed(clean_text)
                return normalize_vector(np.asarray(vec, dtype=np.float32))
            except Exception as embed_err:
                logger.warning("custom embedder failed: %s", embed_err)
                return None

        endpoint = f"{self.ollama_url}/api/embeddings"
        prompt_text = clean_text
        if "nomic" in self.embed_model.lower() and not clean_text.startswith("search_"):
            prompt_text = f"search_document: {clean_text}"
        payload = json.dumps({"model": self.embed_model, "prompt": prompt_text}).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                embedding = data.get("embedding")
                if embedding and isinstance(embedding, list):
                    return normalize_vector(np.asarray(embedding, dtype=np.float32))
        except Exception as error:
            logger.debug("ollama embedding call failed for text: %s", error)
        return None

    def check_entry(self, content: str, source_type: str = "transcript") -> tuple[bool, float]:
        clean = content.strip()
        if len(clean) == 0:
            return False, 1.0

        if is_refusal_text(clean):
            self.drift_count += 1
            return True, 0.0

        words = clean.split()
        if len(words) < self.min_words:
            return False, 1.0

        vector = self.embed_text(clean)
        if vector is None:
            return False, 1.0

        if self.running_anchor is None:
            self.running_anchor = vector
            self.accepted_count = 1
            return False, 1.0

        similarity = cosine_similarity(vector, self.running_anchor)

        if similarity >= self.drift_threshold:
            self.running_anchor = normalize_vector(
                self.alpha * self.running_anchor + (1.0 - self.alpha) * vector
            )
            self.quarantine_buffer.clear()
            self.accepted_count += 1
            return False, similarity

        self.quarantine_buffer.append((clean, vector))
        if len(self.quarantine_buffer) >= self.sustain_consecutive:
            recent_vectors = [v for _, v in self.quarantine_buffer[-self.sustain_consecutive:]]
            consistent = True
            for i in range(len(recent_vectors)):
                for j in range(i + 1, len(recent_vectors)):
                    if cosine_similarity(recent_vectors[i], recent_vectors[j]) < 0.60:
                        consistent = False
                        break
                if not consistent:
                    break
            if consistent:
                new_mean = np.mean(recent_vectors, axis=0)
                self.running_anchor = normalize_vector(new_mean)
                self.quarantine_buffer.clear()
                self.accepted_count += self.sustain_consecutive
                logger.info("sustained lecture topic shift detected; anchor updated")
                return False, similarity

        self.drift_count += 1
        return True, similarity

    def filter_batch_entries(self, entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        for entry in entries:
            flags = list(entry.get("importance_flags", []) or [])
            if "drift" in flags or "ad" in flags:
                continue
            content = str(entry.get("content", ""))
            source_type = str(entry.get("source_type", "transcript"))
            is_drift, score = self.check_entry(content, source_type=source_type)
            if is_drift:
                continue
            filtered.append(entry)
        return filtered
