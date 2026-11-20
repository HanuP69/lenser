import asyncio
import json
import logging
import math
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

from lecturelens.core.fileio import atomic_write_bytes
from lecturelens.core.repeat_guard import Guards
from lecturelens.router.messages import HumanMessage, response_to_text
from lecturelens.schemas.events import GeneralNoteSegment

logger = logging.getLogger(__name__)


@dataclass
class StudentProfile:
    student_id: str = "default_student"
    preferred_language: str = "English"
    detail_level: str = "comprehensive"
    preserved_terms: list[str] = field(default_factory=list)


REWRITE_SYSTEM_PROMPT = """
You are a personal learning assistant customizing lecture notes for a student.
Re-synthesize the provided general lecture note segment according to the student profile:
- Target Language/Style: {preferred_language} (e.g. English, Hindi, Hinglish).
- Detail Level: {detail_level}.

CRITICAL CONSTRAINTS:
1. Preserve all technical terms, formulas ($...$ and $$...$$), and code blocks accurately.
2. Ensure every key point from the source is addressed.
3. Keep the section header with its original timestamp.
4. If writing in Hindi or Hinglish, keep core English technical terminology intact (e.g., 'Binary Search Tree', 'Time Complexity').
5. Output pure markdown notes only.
"""

KEY_POINT_STOPWORDS = frozenset({
    "that", "this", "with", "from", "they", "their", "there", "these", "those",
    "when", "then", "than", "which", "what", "where", "while", "into", "onto",
    "each", "some", "such", "also", "only", "over", "under", "about", "have",
    "will", "would", "could", "should", "because", "very",
})


def highest_contiguous_sequence(applied: Sequence[int]) -> int:
    if not applied:
        return 0
    ordered = sorted(set(applied))
    contiguous = ordered[0]
    for val in ordered[1:]:
        if val == contiguous + 1:
            contiguous = val
        elif val > contiguous + 1:
            break
    return contiguous


class PersonalizationEngine:
    def __init__(self, profile: StudentProfile, llm_router: Any = None) -> None:
        self.profile = profile
        self.llm_router = llm_router

    def build_prompt(self, segment: GeneralNoteSegment) -> str:
        prompt_intro = REWRITE_SYSTEM_PROMPT.format(
            preferred_language=self.profile.preferred_language,
            detail_level=self.profile.detail_level
        )
        preserved = f"\nThese terms must appear verbatim in the rewritten section: {', '.join(self.profile.preserved_terms)}\n" if self.profile.preserved_terms else ""
        return (
            f"{prompt_intro}\n{preserved}\n"
            f"SOURCE SECTION:\nHeading: {segment.heading}\nKey Points: {json.dumps(segment.key_points)}\n"
            f"Concepts: {', '.join(segment.concept_tags)}\n\nSOURCE CONTENT:\n{segment.content}\n"
        )

    def _significant_words(self, text: str) -> list[str]:
        return [w.lower() for w in re.findall(r"\b\w{4,}\b", text) if w.lower() not in KEY_POINT_STOPWORDS]

    @staticmethod
    def _word_matches(word: str, candidates: set[str]) -> bool:
        if word in candidates:
            return True
        if len(word) < 4:
            return False
        stem = word[:3]
        return any(c[:3] == stem and abs(len(c) - len(word)) <= 3 for c in candidates if len(c) >= 3)

    def validate_rewrite(self, segment: GeneralNoteSegment, rewritten_text: str) -> bool:
        if not rewritten_text or len(rewritten_text.strip()) < 20:
            return False
        lower_rewritten = rewritten_text.lower()
        if segment.concept_tags:
            present = sum(1 for tag in segment.concept_tags if tag.replace("_", " ").lower() in lower_rewritten or tag.lower() in lower_rewritten)
            if present < math.ceil(len(segment.concept_tags) * 0.5):
                return False
        if segment.start_time and segment.start_time in segment.heading and segment.start_time not in rewritten_text:
            return False
        for term in self.profile.preserved_terms:
            if term.lower() not in lower_rewritten:
                return False
        if segment.key_points:
            rw_words = set(self._significant_words(rewritten_text))
            src_words = set(self._significant_words(segment.content))
            shared = sum(1 for w in src_words if self._word_matches(w, rw_words))
            if src_words and shared < 2:
                return True
            required = math.ceil(len(segment.key_points) * 0.5)
            covered = 0
            for kp in segment.key_points:
                words = self._significant_words(kp)
                if not words:
                    continue
                needed = 1 if len(words) <= 2 else math.ceil(len(words) * 0.5)
                if sum(1 for w in words if self._word_matches(w, rw_words)) >= needed:
                    covered += 1
            if covered < required:
                return False
        for asset in re.findall(r"!\[.*?\]\((.*?)\)", segment.content):
            if asset not in rewritten_text:
                return False
        return True

    async def rewrite_segment(self, segment: GeneralNoteSegment) -> str:
        if self.llm_router is None:
            return segment.content
        if (
            self.profile.preferred_language.lower() == "english"
            and self.profile.detail_level.lower() == "comprehensive"
            and not self.profile.preserved_terms
        ):
            return segment.content

        placeholders: dict[str, str] = {}
        counter = 0

        def protect(m: re.Match) -> str:
            nonlocal counter
            tok = f"__LL_ASSET_LINK_{counter}__"
            counter += 1
            placeholders[tok] = m.group(0)
            return tok

        masked_content = re.sub(r"!\[.*?\]\(.*?\)|asset://[a-zA-Z0-9_\-\.]+", protect, segment.content)
        masked_seg = segment.model_copy(update={"content": masked_content})

        try:
            resp = await self.llm_router.ainvoke([HumanMessage(content=self.build_prompt(masked_seg))])
            res = response_to_text(resp).strip()
            for tok, link in placeholders.items():
                res = res.replace(tok, link) if tok in res else f"{res}\n\n{link}"
            return res if self.validate_rewrite(segment, res) else segment.content
        except Exception as err:
            logger.error("rewrite failed (%s), fallback to original", err)
            return segment.content


class StudentClient:
    def __init__(
        self,
        server_url: str,
        session_id: str,
        profile: StudentProfile,
        output_dir: Path,
        api_key: Optional[str] = None,
        llm_router: Any = None,
        student_graph: Optional[Any] = None
    ) -> None:
        self.server_url = server_url
        self.session_id = session_id
        self.profile = profile
        self.output_dir = Path(output_dir)
        self.api_key = api_key
        self.student_graph = student_graph
        self.rewriter = PersonalizationEngine(profile, llm_router)
        self.state_file = self.output_dir / f"{session_id}_{profile.student_id}_state.json"
        self.ledger_file = self.output_dir / f"{session_id}_{profile.student_id}_ledger.db"
        self._ledger_was_new = not self.ledger_file.exists()
        self._init_ledger()
        self.notes_file = self.output_dir / f"{session_id}_{profile.student_id}_personalized_notes.md"
        self.legacy_notes_file = self.output_dir / f"{session_id}_personalized_notes.md"
        self._guards = Guards(logger)

        if not self.notes_file.exists() and self.legacy_notes_file.exists():
            try:
                self.legacy_notes_file.replace(self.notes_file)
            except OSError:
                pass

        self.pending_reconciliation = not self.reconcile_notes_from_ledger()
        self.last_sequence_number = self.load_persisted_sequence()
        self.received_segments: list[GeneralNoteSegment] = []
        self.pending_segments: dict[int, GeneralNoteSegment] = {}
        self.is_running = False
        self.auth_error: Optional[str] = None
        self.replay_request_count = 0
        self.max_replay_requests = 10
        self.last_replay_request_time = 0.0
        self.replay_request_interval_seconds = 2.0

    @contextmanager
    def _ledger_conn(self, timeout: float = 10.0) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.ledger_file), timeout=timeout)
        try:
            yield conn
        finally:
            conn.close()

    def _init_ledger(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            with self._ledger_conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS applied_segments (
                        sequence_number INTEGER PRIMARY KEY,
                        session_id TEXT,
                        heading TEXT,
                        content TEXT,
                        created_at REAL
                    );
                """)
                conn.commit()
        except Exception as e:
            logger.warning("failed to init ledger: %s", e)

    def reconcile_notes_from_ledger(self) -> bool:
        if not hasattr(self, "ledger_file") or not self.ledger_file.exists():
            self.pending_reconciliation = False
            return True
        try:
            with self._ledger_conn(timeout=5.0) as conn:
                rows = conn.execute("SELECT sequence_number, content FROM applied_segments ORDER BY sequence_number ASC;").fetchall()

            if not rows:
                if self.state_file.exists() and getattr(self, "_ledger_was_new", False):
                    legacy_seq = 0
                    try:
                        with open(self.state_file, "r", encoding="utf-8") as h:
                            legacy_seq = int(json.load(h).get("last_sequence_number", 0))
                    except Exception as err:
                        self._guards.saw("legacy state parse", err)
                    if legacy_seq > 0:
                        has_usable = self.notes_file.exists() and bool(self.notes_file.read_text(encoding="utf-8").strip())
                        if has_usable:
                            txt = self.notes_file.read_text(encoding="utf-8").strip()
                            with self._ledger_conn() as conn:
                                conn.execute("INSERT OR IGNORE INTO applied_segments VALUES (?, ?, ?, ?, ?)", (legacy_seq, self.session_id, "Legacy Notes", txt, time.time()))
                                conn.commit()
                            self.pending_reconciliation = False
                            return True
                        with open(self.state_file, "w", encoding="utf-8") as h:
                            json.dump({"last_sequence_number": 0}, h)
                        self.pending_reconciliation = True
                        return False
                    self.pending_reconciliation = False
                    return True
                elif self.notes_file.exists():
                    self.notes_file.unlink(missing_ok=True)
                self.pending_reconciliation = False
                return True

            full_md = "".join(f"\n{c.strip()}\n" for _, c in rows if c and c.strip())
            existing = self.notes_file.read_text(encoding="utf-8") if self.notes_file.exists() else None
            if existing != full_md:
                tmp = self.notes_file.with_suffix(f".tmp_{os.getpid()}")
                tmp.write_text(full_md, encoding="utf-8")
                tmp.replace(self.notes_file)
            self.pending_reconciliation = False
            return True
        except Exception as e:
            logger.warning("reconcile ledger failed: %s", e)
            self.pending_reconciliation = True
            return False

    def cache_referenced_assets(self, content: str) -> None:
        links = re.findall(r"!\[.*?\]\((.*?)\)", content)
        if not links:
            return
        assets_dir = self.output_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        base = self.server_url.replace("ws://", "http://").replace("wss://", "https://").rstrip("/")
        import urllib.request
        for link in links:
            fname = Path(link).name
            target = assets_dir / fname
            if not target.exists():
                url = f"{base}/api/assets/{self.session_id}/{fname}" + (f"?key={self.api_key}" if self.api_key else "")
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "LectureLens-StudentClient"})
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        atomic_write_bytes(target, resp.read())
                except Exception:
                    pass

    def load_persisted_sequence(self) -> int:
        persisted = 0
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as h:
                    persisted = int(json.load(h).get("last_sequence_number", 0))
            except Exception:
                pass
        ledger_max, ledger_contiguous = None, None
        if hasattr(self, "ledger_file") and self.ledger_file.exists():
            try:
                with self._ledger_conn(timeout=5.0) as conn:
                    r = conn.execute("SELECT MAX(sequence_number) FROM applied_segments;").fetchone()
                    ledger_max = int(r[0]) if r and r[0] is not None else 0
                    applied = [int(x[0]) for x in conn.execute("SELECT sequence_number FROM applied_segments ORDER BY sequence_number ASC;").fetchall()]
                    ledger_contiguous = highest_contiguous_sequence(applied)
            except Exception:
                self.pending_reconciliation = True
                return 0

        if getattr(self, "pending_reconciliation", False):
            return min(persisted, ledger_contiguous) if ledger_contiguous is not None else 0
        if getattr(self, "_ledger_was_new", False) and (ledger_max is None or ledger_max == 0):
            has_usable = self.notes_file.exists() and bool(self.notes_file.read_text(encoding="utf-8").strip())
            if not has_usable and persisted > 0:
                self.pending_reconciliation = True
                return 0
        return ledger_contiguous if ledger_contiguous is not None else persisted

    def save_persisted_sequence(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.state_file, "w", encoding="utf-8") as h:
                json.dump({"last_sequence_number": self.last_sequence_number}, h)
        except Exception:
            pass

    async def _apply_segment(self, segment: GeneralNoteSegment) -> str:
        content = await self.rewriter.rewrite_segment(segment)
        try:
            with self._ledger_conn() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO applied_segments VALUES (?, ?, ?, ?, ?)",
                    (segment.sequence_number, segment.session_id, segment.heading, content, time.time())
                )
                conn.commit()
        except Exception as err:
            raise RuntimeError(f"saving to student ledger failed: {err}") from err

        if not self.reconcile_notes_from_ledger():
            self.pending_reconciliation = True
            raise RuntimeError("reconciling notes markdown file failed")

        self.pending_reconciliation = False
        self.received_segments.append(segment)
        self.last_sequence_number = max(self.last_sequence_number, segment.sequence_number)
        self.save_persisted_sequence()
        self.cache_referenced_assets(content)

        if self.student_graph:
            for tag in segment.concept_tags:
                try:
                    self.student_graph.record_concept(tag)
                except Exception:
                    pass
        return content

    async def on_note_segment_received(self, segment: GeneralNoteSegment, websocket: Optional[Any] = None) -> str:
        seq = segment.sequence_number
        if seq <= self.last_sequence_number and not getattr(self, "pending_reconciliation", False):
            return ""
        if seq > self.last_sequence_number + 1:
            self.pending_segments[seq] = segment
            if websocket is not None:
                await self.request_replay(websocket)
            return ""

        applied = await self._apply_segment(segment)
        self.replay_request_count = 0
        nxt = self.last_sequence_number + 1
        while nxt in self.pending_segments:
            applied = await self._apply_segment(self.pending_segments.pop(nxt))
            nxt = self.last_sequence_number + 1
        return applied

    async def request_replay(self, websocket: Any) -> None:
        if self.replay_request_count >= self.max_replay_requests:
            if self.replay_request_count == self.max_replay_requests:
                self.replay_request_count += 1
            return
        now = time.monotonic()
        if (now - self.last_replay_request_time) < self.replay_request_interval_seconds:
            return
        self.last_replay_request_time = now
        self.replay_request_count += 1
        try:
            await websocket.send(json.dumps({"type": "request_replay", "last_sequence_number": self.last_sequence_number}))
        except Exception:
            pass

    async def handle_server_frame(self, raw_frame: str, websocket: Optional[Any] = None) -> None:
        try:
            payload = json.loads(raw_frame)
            if payload.get("type") == "note_segment":
                await self.on_note_segment_received(GeneralNoteSegment(**payload.get("payload", {})), websocket)
            elif payload.get("status") == "unauthorized":
                self.auth_error = str(payload.get("error", "unauthorized"))
        except Exception as e:
            logger.warning("frame parse error: %s", e)

    async def run(self, stop_event: Optional[asyncio.Event] = None) -> None:
        try:
            from websockets.asyncio.client import connect
        except ImportError:
            from websockets import connect

        self.is_running = True
        backoff = 1.0
        while self.is_running:
            if stop_event and stop_event.is_set():
                break
            if getattr(self, "pending_reconciliation", False) and self.reconcile_notes_from_ledger():
                self.last_sequence_number = self.load_persisted_sequence()
                self.save_persisted_sequence()

            hello = {
                "type": "hello", "role": "student", "session_id": self.session_id,
                "client_id": self.profile.student_id, "last_sequence_number": self.last_sequence_number,
                "api_key": self.api_key
            }
            try:
                async with connect(self.server_url) as ws:
                    await ws.send(json.dumps(hello))
                    backoff = 1.0
                    while self.is_running:
                        if stop_event and stop_event.is_set():
                            return
                        try:
                            frame = await asyncio.wait_for(ws.recv(), timeout=2.0)
                            await self.handle_server_frame(frame, ws)
                            if self.auth_error:
                                return
                        except asyncio.TimeoutError:
                            if getattr(self, "pending_reconciliation", False) and self.reconcile_notes_from_ledger():
                                self.last_sequence_number = self.load_persisted_sequence()
                                self.save_persisted_sequence()
            except asyncio.CancelledError:
                raise
            except Exception:
                if stop_event and stop_event.is_set():
                    break
                self.replay_request_count = 0
                await asyncio.sleep(backoff)
                backoff = min(10.0, backoff * 1.5)
