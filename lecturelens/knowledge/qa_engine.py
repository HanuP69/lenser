import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx
from langchain_core.messages import HumanMessage

from lecturelens.config.defaults import DEFAULT_EMBEDDING_MODEL, DEFAULT_LOCAL_MODEL
from lecturelens.core.logging_setup import get_logger
from lecturelens.router.messages import response_to_text

logger = get_logger("lecturelens.qa_engine")

EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL
TIMELINE_CHUNK_CHARACTERS = 600
NOTES_CHUNK_CHARACTERS = 900
EMBED_BATCH_SIZE = 32
CHROMA_COLLECTION_NAME = "lecture_chunks"
CHROMA_COLLECTION_METADATA = {"hnsw:space": "cosine"}


def require_chromadb() -> Any:
    try:
        import chromadb
        return chromadb
    except ImportError as error:
        raise ImportError("chromadb is required for lecture indexing. Run: pip install 'lecturelens[full]'") from error


@dataclass
class IndexedChunk:
    chunk_id: str
    text: str
    source: str
    reference: str


def make_chunk_id(source: str, text: str) -> str:
    digest = hashlib.sha1(f"{source}|{text}".encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"{source}_{digest}"


def format_offset(seconds_value: float) -> str:
    tot = int(max(0.0, seconds_value))
    return f"{tot // 60:02d}:{tot % 60:02d}"


class QAEngine:
    def __init__(
        self,
        session_dir: Path,
        ollama_url: Optional[str] = None,
        llm_router: Any = None,
        top_k: int = 5,
        llm_timeout_seconds: float = 180.0,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        fallback_model: str = DEFAULT_LOCAL_MODEL,
        storage: Any = None,
        embed_batch_size: Optional[int] = None
    ) -> None:
        self.embedding_model = embedding_model
        self.fallback_model = fallback_model
        self.session_dir = Path(session_dir)
        self.ollama_url = (ollama_url or os.environ.get("LECTURELENS_OLLAMA_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.llm_router = llm_router
        self.top_k = top_k
        self.llm_timeout_seconds = llm_timeout_seconds
        self.storage = storage
        self.embed_batch_size = embed_batch_size if embed_batch_size is not None else EMBED_BATCH_SIZE
        self.timeline_path = self.session_dir / "timeline.jsonl"
        self.notes_path = self.session_dir / "notes.md"

        chromadb = require_chromadb()
        self.chroma_client = chromadb.PersistentClient(path=str(self.session_dir / "chroma_db"))
        meta_file = self.session_dir / "chroma_db" / "index_meta.json"
        if meta_file.exists():
            try:
                stored = json.loads(meta_file.read_text(encoding="utf-8"))
                if stored.get("embedding_model") != self.embedding_model:
                    try:
                        self.chroma_client.delete_collection(CHROMA_COLLECTION_NAME)
                    except Exception:
                        pass
            except Exception:
                pass
        self.collection = self._open_collection()
        try:
            meta_file.parent.mkdir(parents=True, exist_ok=True)
            meta_file.write_text(json.dumps({"embedding_model": self.embedding_model}), encoding="utf-8")
        except Exception:
            pass

    def _open_collection(self) -> Any:
        try:
            col = self.chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME, metadata=CHROMA_COLLECTION_METADATA)
            if (getattr(col, "metadata", None) or {}).get("hnsw:space") != CHROMA_COLLECTION_METADATA["hnsw:space"]:
                self.chroma_client.delete_collection(CHROMA_COLLECTION_NAME)
                col = self.chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME, metadata=CHROMA_COLLECTION_METADATA)
            return col
        except Exception:
            return self.chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    def _prefix_for(self, text: str, purpose: str) -> str:
        if "nomic" not in self.embedding_model.lower():
            return text
        prefix = "search_query:" if purpose == "query" else "search_document:"
        return text if text.startswith(("search_document:", "search_query:")) else f"{prefix} {text}"

    async def generate_embeddings(self, texts: list[str], purpose: str = "document") -> list[list[float]]:
        embeddings: list[list[float]] = []
        async with httpx.AsyncClient(timeout=120.0) as client:
            for i in range(0, len(texts), self.embed_batch_size):
                batch = texts[i:i + self.embed_batch_size]
                payload = [self._prefix_for(t, purpose) for t in batch]
                res = await client.post(f"{self.ollama_url}/api/embed", json={"model": self.embedding_model, "input": payload, "keep_alive": "10m"})
                res.raise_for_status()
                batch_vecs = res.json().get("embeddings") or []
                if len(batch_vecs) != len(batch):
                    raise ValueError(f"model returned {len(batch_vecs)} vectors for {len(batch)} inputs")
                embeddings.extend(batch_vecs)
        return embeddings

    def collect_timeline_chunks(self) -> list[IndexedChunk]:
        chunks: list[IndexedChunk] = []
        if not self.timeline_path.exists():
            return chunks
        pending_texts: list[str] = []
        pending_chars = 0
        pending_start: Optional[float] = None
        first_epoch: Optional[float] = None

        def flush():
            nonlocal pending_texts, pending_chars, pending_start
            if not pending_texts:
                return
            comb = " ".join(pending_texts)
            offset = format_offset(pending_start - first_epoch) if pending_start is not None and first_epoch is not None else "00:00"
            chunks.append(IndexedChunk(make_chunk_id("timeline", comb), comb, "timeline", f"timeline @ {offset}"))
            pending_texts.clear()
            pending_chars = 0
            pending_start = None

        for line in self.timeline_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            content = str(entry.get("content", "")).strip()
            if not content:
                continue
            ts = float(entry.get("timestamp_epoch", 0.0))
            if first_epoch is None:
                first_epoch = ts
            src = str(entry.get("source_type", "transcript"))
            if src != "transcript":
                flush()
                labeled = f"[{src}] {content}"
                chunks.append(IndexedChunk(make_chunk_id("timeline", labeled), labeled, "timeline", f"{src} @ {format_offset(ts - first_epoch)}"))
                continue
            if pending_start is None:
                pending_start = ts
            pending_texts.append(content)
            pending_chars += len(content)
            if pending_chars >= TIMELINE_CHUNK_CHARACTERS:
                flush()
        flush()
        return chunks

    def collect_notes_chunks(self) -> list[IndexedChunk]:
        if not self.notes_path.exists():
            return []
        chunks: list[IndexedChunk] = []
        curr = ""
        for par in self.notes_path.read_text(encoding="utf-8").split("\n\n"):
            stripped = par.strip()
            if not stripped:
                continue
            if curr and (len(curr) + len(stripped)) > NOTES_CHUNK_CHARACTERS:
                chunks.append(IndexedChunk(make_chunk_id("notes", curr), curr, "notes", "notes.md"))
                curr = ""
            curr = f"{curr}\n\n{stripped}".strip() if curr else stripped
        if curr:
            chunks.append(IndexedChunk(make_chunk_id("notes", curr), curr, "notes", "notes.md"))
        return chunks

    async def build_index(self) -> int:
        wanted = self.collect_timeline_chunks() + self.collect_notes_chunks()
        wanted_ids = {c.chunk_id for c in wanted}
        existing_ids = set(self.collection.get(include=[])["ids"])

        stale = list(existing_ids - wanted_ids)
        if stale:
            self.collection.delete(ids=stale)
            if self.storage and hasattr(self.storage, "delete_embeddings"):
                try:
                    await self.storage.delete_embeddings(self.session_dir.name, stale)
                except Exception:
                    pass

        if self.storage:
            if hasattr(self.storage, "delete_embeddings_not_in"):
                try:
                    await self.storage.delete_embeddings_not_in(self.session_dir.name, list(wanted_ids))
                except Exception:
                    pass
            elif hasattr(self.storage, "get_connection") and hasattr(self.storage, "_lock"):
                try:
                    async with self.storage._lock:
                        conn = await self.storage.get_connection()
                        async with conn.cursor() as cur:
                            await cur.execute("SELECT source_id FROM embeddings WHERE session_id = %s;", (self.session_dir.name,))
                            pg_ids = {str(r["source_id"]) for r in await cur.fetchall() if r and "source_id" in r}
                            stale_pg = list(pg_ids - wanted_ids)
                            if stale_pg:
                                await cur.execute("DELETE FROM embeddings WHERE session_id = %s AND source_id = ANY(%s);", (self.session_dir.name, stale_pg))
                except Exception:
                    pass
            if hasattr(self.storage, "delete_session_embeddings"):
                try:
                    await self.storage.delete_session_embeddings(self.session_dir.name, exclude_model=self.embedding_model)
                except Exception:
                    pass

        new_chunks = [c for c in wanted if c.chunk_id not in existing_ids]
        seen_new: set[str] = set()
        dedup_new = []
        for c in new_chunks:
            if c.chunk_id not in seen_new:
                seen_new.add(c.chunk_id)
                dedup_new.append(c)
        new_chunks = dedup_new

        pg_missing_chunks = []
        if self.storage and hasattr(self.storage, "insert_embedding"):
            already_in_chroma = [c for c in wanted if c.chunk_id in existing_ids]
            if already_in_chroma:
                existing_in_pg: set[str] = set()
                if hasattr(self.storage, "get_connection") and hasattr(self.storage, "_lock"):
                    try:
                        async with self.storage._lock:
                            conn = await self.storage.get_connection()
                            async with conn.cursor() as cur:
                                await cur.execute("SELECT source_id FROM embeddings WHERE session_id = %s;", (self.session_dir.name,))
                                for r in await cur.fetchall():
                                    existing_in_pg.add(str(r["source_id"]))
                    except Exception:
                        pass
                pg_missing_chunks = [c for c in already_in_chroma if c.chunk_id not in existing_in_pg]

        if new_chunks:
            texts = [c.text for c in new_chunks]
            vecs = await self.generate_embeddings(texts, purpose="document")
            if vecs and len(vecs[0]) > 0 and self.storage and hasattr(self.storage, "init_schema"):
                try:
                    await self.storage.init_schema(vector_dimensions=len(vecs[0]))
                except Exception:
                    pass
            if self.storage and hasattr(self.storage, "insert_embedding"):
                for chunk, vec in zip(new_chunks, vecs):
                    try:
                        await self.storage.insert_embedding(
                            session_id=self.session_dir.name, source_type=chunk.source,
                            source_id=chunk.chunk_id, chunk_text=chunk.text, vector=vec,
                            embedding_model=self.embedding_model
                        )
                    except Exception:
                        pass
            self.collection.upsert(
                ids=[c.chunk_id for c in new_chunks], embeddings=vecs, documents=texts,
                metadatas=[{"source": c.source, "reference": c.reference} for c in new_chunks]
            )

        if pg_missing_chunks and self.storage and hasattr(self.storage, "insert_embedding"):
            try:
                chroma_data = self.collection.get(ids=[c.chunk_id for c in pg_missing_chunks], include=["embeddings"])
                c_map = dict(zip(chroma_data.get("ids", []), chroma_data.get("embeddings", [])))
                for chunk in pg_missing_chunks:
                    vec = c_map.get(chunk.chunk_id)
                    if vec is not None:
                        try:
                            await self.storage.insert_embedding(
                                session_id=self.session_dir.name, source_type=chunk.source,
                                source_id=chunk.chunk_id, chunk_text=chunk.text, vector=vec,
                                embedding_model=self.embedding_model
                            )
                        except Exception:
                            pass
            except Exception:
                pass

        return int(self.collection.count())

    async def retrieve(self, question: str) -> list[tuple[str, str]]:
        total = int(self.collection.count())
        has_pg = self.storage is not None and hasattr(self.storage, "search_embeddings")
        if total == 0 and not has_pg:
            return []
        q_vecs = await self.generate_embeddings([question], purpose="query")
        retrieved: list[tuple[str, str]] = []
        seen: set[str] = set()

        if has_pg:
            try:
                for row in await self.storage.search_embeddings(session_id=self.session_dir.name, query_vector=q_vecs[0], top_k=self.top_k, embedding_model=self.embedding_model):
                    txt = str(row.get("chunk_text") or "")
                    if txt and txt not in seen:
                        seen.add(txt)
                        retrieved.append((txt, self._humanize_source_id(row)))
            except Exception:
                pass

        if total > 0 and len(retrieved) < self.top_k:
            res = self.collection.query(query_embeddings=[q_vecs[0]], n_results=min(self.top_k, total))
            for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
                d_str = str(doc)
                if d_str not in seen:
                    seen.add(d_str)
                    retrieved.append((d_str, str(meta.get("reference", "unknown")) if meta else "unknown"))
                    if len(retrieved) >= self.top_k:
                        break
        return retrieved

    @staticmethod
    def _humanize_source_id(row: dict[str, Any]) -> str:
        sid = str(row.get("source_id") or "")
        stype = str(row.get("source_type") or "")
        if not sid:
            return stype or "unknown"
        if "_" in sid:
            prefix = sid.rsplit("_", 1)[0]
            if prefix in ("timeline", "notes"):
                return f"{prefix}.jsonl" if prefix == "timeline" else f"{prefix}.md"
            return prefix
        return sid

    async def call_language_model(self, prompt: str) -> str:
        if self.llm_router is not None:
            resp = await self.llm_router.ainvoke([HumanMessage(content=prompt)])
            return response_to_text(resp)
        async with httpx.AsyncClient(timeout=self.llm_timeout_seconds) as client:
            res = await client.post(f"{self.ollama_url}/api/generate", json={"model": self.fallback_model, "prompt": prompt, "stream": False})
            res.raise_for_status()
            return str(res.json().get("response", "")).strip()

    async def ask_with_sources(self, question: str) -> tuple[str, list[str]]:
        await self.build_index()
        retrieved = await self.retrieve(question)
        if not retrieved:
            return "No indexed lecture content is available for this session yet.", []

        context_blocks = [f"[{i}] ({ref}) {txt}" for i, (txt, ref) in enumerate(retrieved, start=1)]
        refs = list(dict.fromkeys(ref for _, ref in retrieved))
        prompt = f"You answer questions about a lecture using only the excerpts below.\nIf the excerpts do not contain the answer, say so.\nCite excerpt numbers like [1] where you use them.\n\nEXCERPTS:\n{chr(10).join(context_blocks)}\n\nQUESTION: {question}\n"
        answer = await self.call_language_model(prompt)
        return answer, refs

    async def ask(self, question: str) -> str:
        ans, refs = await self.ask_with_sources(question)
        return ans if not refs else f"{ans}\n\nSources:\n" + "\n".join(f"- {r}" for r in refs)
