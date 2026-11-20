import hashlib
import json
import re
import urllib.request
from typing import Any, Optional, Sequence

import numpy as np

from lecturelens.core.logging_setup import get_logger
from lecturelens.core.repeat_guard import Guards

logger = get_logger("lecturelens.drift_filter")

MAX_BATCH_SNAPSHOTS = 16
# How many embedding failures are tolerated before the endpoint is treated as
# unreachable.
EMBEDDING_FAILURE_LIMIT = 3

REFUSAL_SENTENCE_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"^\s*please\s+provide\s+(?:the\s+)?presentation\s+slide", re.IGNORECASE),
    re.compile(r"^\s*no\s+text\s+or\s+diagram\s+was\s+visible", re.IGNORECASE),
    re.compile(r"^\s*i\s+need\s+an\s+image\s+of\s+the\s+presentation\s+slide", re.IGNORECASE),
    re.compile(r"^\s*no\s+text\s+is\s+currently\s+visible", re.IGNORECASE),
    re.compile(r"^\s*please\s+provide\s+an\s+image", re.IGNORECASE),
    # A vision model that never received the image says so in its own words.
    # The gemini free tier drops the image and answers like this, and the
    # reply used to be written into the notes as if it were the slide text.
    re.compile(r"^\s*it\s+looks\s+like\s+you\s+(?:didn'?t|did\s+not)\s+(?:attach|upload)", re.IGNORECASE),
    re.compile(r"^\s*it\s+looks\s+like\s+you\s+forgot\s+to\s+attach", re.IGNORECASE),
    re.compile(r"^\s*it\s+looks\s+like\s+there\s+(?:is|'?s)\s+no\s+image", re.IGNORECASE),
    re.compile(r"^\s*no\s+image\s+(?:was\s+)?(?:provided|attached|uploaded)", re.IGNORECASE),
    re.compile(r"^\s*the\s+image\s+(?:was\s+not|isn'?t)\s+(?:provided|attached|uploaded|received)", re.IGNORECASE),
    re.compile(r"^\s*i\s+(?:can'?t|cannot)\s+see\s+(?:any\s+)?image", re.IGNORECASE),
    re.compile(r"^\s*i\s+do\s+not\s+see\s+(?:an?\s+)?image", re.IGNORECASE),
)

TEMPLATE_PLACEHOLDER_REGEX: re.Pattern = re.compile(
    r"(?:presented|prepared)\s+by\s*:\s*\[\s*(?:name|your\s+name|insert\s+name|author|presenter)\s*\]",
    re.IGNORECASE
)


def is_refusal_text(text: str) -> bool:
    clean = text.strip()
    if len(clean) == 0:
        return False
    for pattern in REFUSAL_SENTENCE_PATTERNS:
        if pattern.search(clean) is not None:
            return True
    if TEMPLATE_PLACEHOLDER_REGEX.search(clean) is not None:
        return True
    return False


def normalize_vector(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm <= 1e-9:
        return v
    return v / norm


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    if v1.shape != v2.shape:
        return 0.0
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
        self.pending_quarantine_entries: list[dict[str, Any]] = []
        # The entries that were excluded from the notes because of the ad or
        # drift flag. They are kept so a false positive does not lose the
        # content.
        self.quarantined_entries: list[dict[str, Any]] = []
        # The embedding endpoint is often not running, because the cloud
        # providers are used instead of the local model. After a few failures
        # the drift filter stops trying, so it does not cost seconds per
        # entry.
        self._embedding_failures: int = 0
        self._embedding_endpoint_unavailable: bool = False
        self.accepted_count: int = 0
        self.drift_count: int = 0
        self._batch_snapshots: dict[str, dict[str, Any]] = {}
        # The faults that are expected but should not be silent.
        self._guards = Guards(logger)

        if session_topic is not None and len(session_topic.strip()) > 0:
            clean_topic = session_topic.strip()
            if len(clean_topic.split()) < 3:
                clean_topic = f"Course lecture on academic subject {clean_topic}"
            initial_vec = self.embed_text(clean_topic)
            if initial_vec is not None:
                self.running_anchor = initial_vec

    def reset(self) -> None:
        self.running_anchor = None
        self.accepted_count = 0
        self.drift_count = 0
        self.pending_quarantine_entries = []
        # The quarantine belongs to the run, like the pending buffer: a
        # deliberate rebuild used to keep the quarantined entries of the
        # run before it, and the first batch of the new run wrote them
        # into the quarantine file of the new session.
        self.quarantined_entries = []
        self.quarantine_buffer.clear()
        self._batch_snapshots.clear()
        if self.session_topic is not None and len(self.session_topic.strip()) > 0:
            clean_topic = self.session_topic.strip()
            if len(clean_topic.split()) < 3:
                clean_topic = f"Course lecture on academic subject {clean_topic}"
            initial_vec = self.embed_text(clean_topic)
            if initial_vec is not None:
                self.running_anchor = initial_vec

    def snapshot_state(self) -> dict[str, Any]:
        """Capture the mutable filter state so a batch can be re-filtered
        idempotently after a failed synthesis pass."""
        return {
            "running_anchor": None if self.running_anchor is None else self.running_anchor.copy(),
            "quarantine_buffer": list(self.quarantine_buffer),
            "pending_quarantine_entries": list(self.pending_quarantine_entries),
            # The quarantined entries are part of the state, so a retried batch
            # does not quarantine the same entries a second time.
            "quarantined_entries": list(self.quarantined_entries),
            "accepted_count": self.accepted_count,
            "drift_count": self.drift_count,
        }

    def restore_state(self, snapshot: dict[str, Any]) -> None:
        anchor = snapshot.get("running_anchor")
        self.running_anchor = None if anchor is None else np.asarray(anchor, dtype=np.float32)
        self.quarantine_buffer = list(snapshot.get("quarantine_buffer", []))
        self.pending_quarantine_entries = list(snapshot.get("pending_quarantine_entries", []))
        self.quarantined_entries = list(snapshot.get("quarantined_entries", []))
        self.accepted_count = int(snapshot.get("accepted_count", 0))
        self.drift_count = int(snapshot.get("drift_count", 0))

    @staticmethod
    def _batch_identity(entries: Sequence[dict[str, Any]]) -> str:
        parts: list[str] = []
        for entry in entries:
            identifier = entry.get("event_id")
            if not identifier:
                identifier = f"{entry.get('timestamp_epoch')}:{entry.get('content', '')}"
            parts.append(str(identifier))
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "running_anchor": self.running_anchor.tolist() if self.running_anchor is not None else None,
            "accepted_count": self.accepted_count,
            "drift_count": self.drift_count,
            "session_topic": self.session_topic,
            "pending_quarantine_entries": self.pending_quarantine_entries
        }

    def guards(self) -> Guards:
        """The guards of the faults, for the heartbeat of the agent."""
        return self._guards

    def load_dict(self, data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return
        anchor_list = data.get("running_anchor")
        if anchor_list and isinstance(anchor_list, list):
            self.running_anchor = np.asarray(anchor_list, dtype=np.float32)
        self.accepted_count = int(data.get("accepted_count", self.accepted_count))
        self.drift_count = int(data.get("drift_count", self.drift_count))
        if "session_topic" in data:
            self.session_topic = data.get("session_topic")
        if "pending_quarantine_entries" in data and isinstance(data["pending_quarantine_entries"], list):
            self.pending_quarantine_entries = list(data["pending_quarantine_entries"])

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

        if self._embedding_endpoint_unavailable:
            # The endpoint was already found to be unreachable. Trying it
            # again for every entry cost seconds each time and stalled the
            # shutdown, because the local model is often not running at all
            # when the cloud providers are used.
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
            with urllib.request.urlopen(req, timeout=6.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                embedding = data.get("embedding")
                if embedding and isinstance(embedding, list):
                    return normalize_vector(np.asarray(embedding, dtype=np.float32))
        except Exception as error:
            self._embedding_failures += 1
            if self._embedding_failures >= EMBEDDING_FAILURE_LIMIT:
                if not self._embedding_endpoint_unavailable:
                    logger.warning(
                        "the embedding endpoint %s is unreachable (%s); the drift filter "
                        "will not try again, so the entries are kept as they are",
                        endpoint, error
                    )
                self._embedding_endpoint_unavailable = True
            else:
                # The fault is said once and then counted, so that a
                # model that is not running does not fill the log.
                self._guards.saw("the embedding of the drift filter", error)
        return None

    def check_entry_with_shift(
        self,
        content: str,
        source_type: str = "transcript"
    ) -> tuple[bool, float, bool, bool]:
        clean = content.strip()
        if len(clean) == 0:
            return False, 1.0, False, False

        if is_refusal_text(clean):
            self.drift_count += 1
            return True, 0.0, False, False

        words = clean.split()
        if len(words) < self.min_words:
            return False, 1.0, False, False

        vector = self.embed_text(clean)
        if vector is None:
            return False, 1.0, False, False

        if self.running_anchor is None:
            self.running_anchor = vector
            self.accepted_count = 1
            return False, 1.0, False, True

        effective_threshold = self.drift_threshold
        if self.accepted_count < 3:
            effective_threshold = max(0.40, self.drift_threshold - 0.08)

        similarity = cosine_similarity(vector, self.running_anchor)

        if similarity >= effective_threshold:
            alpha = 0.50 if self.accepted_count < 3 else self.alpha
            self.running_anchor = normalize_vector(
                alpha * self.running_anchor + (1.0 - alpha) * vector
            )
            self.quarantine_buffer.clear()
            self.accepted_count += 1
            return False, similarity, False, True

        self.quarantine_buffer.append((clean, vector))
        if len(self.quarantine_buffer) > 20:
            self.quarantine_buffer.pop(0)
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
                return False, similarity, True, True

        self.drift_count += 1
        return True, similarity, False, False

    def check_entry(self, content: str, source_type: str = "transcript") -> tuple[bool, float]:
        is_drift, similarity, _, _ = self.check_entry_with_shift(content, source_type=source_type)
        return is_drift, similarity

    def filter_batch_entries(self, entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []

        batch_key = self._batch_identity(entries)
        if batch_key in self._batch_snapshots:
            # This batch was already filtered once (a retry after a failed
            # synthesis pass). Restore the pre-batch state so the running
            # anchor, the quarantine buffer and the counters are not advanced
            # a second time, which would fabricate a topic shift and duplicate
            # quarantined entries.
            self.restore_state(self._batch_snapshots[batch_key])
        else:
            self._batch_snapshots[batch_key] = self.snapshot_state()
            while len(self._batch_snapshots) > MAX_BATCH_SNAPSHOTS:
                oldest_key = next(iter(self._batch_snapshots))
                del self._batch_snapshots[oldest_key]

        for entry in entries:
            flags = list(entry.get("importance_flags", []) or [])
            is_important = any(f in flags for f in ("exam_relevant", "definition", "action_item"))

            # A single heuristic flag must never delete lecture content. The
            # ad/drift markers come from regex and similarity heuristics that
            # produce false positives on legitimate speech, so an entry is only
            # dropped when no important marker contradicts the flag.
            if not is_important and ("drift" in flags or "ad" in flags):
                # The entry is quarantined instead of being deleted, so a
                # false positive on the ad heuristics does not lose the
                # content. The caller can persist the quarantine.
                self.quarantined_entries.append(dict(entry))
                continue

            content = str(entry.get("content", ""))
            source_type = str(entry.get("source_type", "transcript"))
            if is_refusal_text(content.strip()):
                # A refusal ("please provide the presentation slide image") is
                # a vision-model artefact and never lecture content. It used
                # to be quarantined like a topic drift and could be released
                # back into the notes when an important entry arrived.
                self.drift_count += 1
                continue
            is_drift, score, shift_occurred, confirmed_on_topic = self.check_entry_with_shift(content, source_type=source_type)

            if shift_occurred:
                filtered.extend(self.pending_quarantine_entries)
                self.pending_quarantine_entries.clear()
                filtered.append(entry)
            elif is_important:
                if self.pending_quarantine_entries:
                    filtered.extend(self.pending_quarantine_entries)
                    self.pending_quarantine_entries.clear()
                filtered.append(entry)
            elif is_drift:
                self.pending_quarantine_entries.append(entry)
                if len(self.pending_quarantine_entries) > 20:
                    # The pending buffer is bounded, but the entry that
                    # comes out of it was never lecture material:
                    # deleting it here threw the content away without a
                    # trace. It goes to the quarantine that the
                    # structuring agent writes to disk instead.
                    self.quarantined_entries.append(self.pending_quarantine_entries.pop(0))
            elif confirmed_on_topic:
                self.pending_quarantine_entries.clear()
                filtered.append(entry)
            else:
                filtered.append(entry)

        return filtered
