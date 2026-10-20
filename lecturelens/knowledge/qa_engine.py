import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import chromadb
import httpx
from langchain_core.messages import HumanMessage

from lecturelens.core.logging_setup import get_logger
from lecturelens.router.messages import response_to_text
from lecturelens.config.defaults import DEFAULT_EMBEDDING_MODEL, DEFAULT_LOCAL_MODEL

logger = get_logger("lecturelens.qa_engine")

EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL
TIMELINE_CHUNK_CHARACTERS = 600
NOTES_CHUNK_CHARACTERS = 900
EMBED_BATCH_SIZE = 32


@dataclass
class IndexedChunk:
    chunk_id: str
    text: str
    source: str
    reference: str


def make_chunk_id(source: str, text: str) -> str:
    digest = hashlib.sha1(f"{source}|{text}".encode("utf-8")).hexdigest()
    return f"{source}_{digest}"


def format_offset(seconds_value: float) -> str:
    total_seconds = int(max(0.0, seconds_value))
    minutes = total_seconds // 60
    remaining = total_seconds % 60
    return f"{minutes:02d}:{remaining:02d}"


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
        self.embedding_model: str = embedding_model
        self.fallback_model: str = fallback_model
        self.session_dir: Path = Path(session_dir)
        resolved_url = ollama_url or os.environ.get("LECTURELENS_OLLAMA_URL", "http://localhost:11434")
        self.ollama_url: str = resolved_url.rstrip("/")
        self.llm_router: Any = llm_router
        self.top_k: int = top_k
        self.llm_timeout_seconds: float = llm_timeout_seconds
        self.storage: Any = storage
        self.embed_batch_size: int = embed_batch_size if embed_batch_size is not None else EMBED_BATCH_SIZE
        self.timeline_path: Path = self.session_dir / "timeline.jsonl"
        self.notes_path: Path = self.session_dir / "notes.md"
        self.chroma_client: Any = chromadb.PersistentClient(path=str(self.session_dir / "chroma_db"))
        meta_file = self.session_dir / "chroma_db" / "index_meta.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as mf:
                    stored_meta = json.load(mf)
                if stored_meta.get("embedding_model") != self.embedding_model:
                    try:
                        self.chroma_client.delete_collection("lecture_chunks")
                    except Exception:
                        pass
            except Exception:
                pass
        self.collection: Any = self.chroma_client.get_or_create_collection(name="lecture_chunks")
        try:
            meta_file.parent.mkdir(parents=True, exist_ok=True)
            with open(meta_file, "w", encoding="utf-8") as mf:
                json.dump({"embedding_model": self.embedding_model}, mf)
        except Exception:
            pass

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        async with httpx.AsyncClient(timeout=120.0) as client:
            for start_index in range(0, len(texts), self.embed_batch_size):
                batch_texts = texts[start_index:start_index + self.embed_batch_size]
                response = await client.post(
                    f"{self.ollama_url}/api/embed",
                    json={"model": self.embedding_model, "input": batch_texts, "keep_alive": "10m"}
                )
                response.raise_for_status()
                payload = response.json()
                for vector in payload["embeddings"]:
                    embeddings.append(vector)
        return embeddings

    async def generate_embedding(self, text: str) -> list[float]:
        vectors = await self.generate_embeddings([text])
        return vectors[0]

    def collect_timeline_chunks(self) -> list[IndexedChunk]:
        chunks: list[IndexedChunk] = []
        if not self.timeline_path.exists():
            return chunks

        pending_texts: list[str] = []
        pending_characters = 0
        pending_start: Optional[float] = None
        first_epoch: Optional[float] = None

        def close_pending() -> None:
            nonlocal pending_texts, pending_characters, pending_start
            if len(pending_texts) == 0:
                return
            combined = " ".join(pending_texts)
            offset_label = "00:00"
            if pending_start is not None and first_epoch is not None:
                offset_label = format_offset(pending_start - first_epoch)
            chunks.append(IndexedChunk(
                chunk_id=make_chunk_id("timeline", combined),
                text=combined,
                source="timeline",
                reference=f"timeline @ {offset_label}"
            ))
            pending_texts = []
            pending_characters = 0
            pending_start = None

        with open(self.timeline_path, "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                stripped = line.strip()
                if len(stripped) == 0:
                    continue
                try:
                    entry = json.loads(stripped)
                except ValueError:
                    continue
                content = str(entry.get("content", "")).strip()
                if len(content) == 0:
                    continue
                timestamp = float(entry.get("timestamp_epoch", 0.0))
                if first_epoch is None:
                    first_epoch = timestamp
                source_type = str(entry.get("source_type", "transcript"))
                if source_type != "transcript":
                    close_pending()
                    labeled = f"[{source_type}] {content}"
                    chunks.append(IndexedChunk(
                        chunk_id=make_chunk_id("timeline", labeled),
                        text=labeled,
                        source="timeline",
                        reference=f"{source_type} @ {format_offset(timestamp - first_epoch)}"
                    ))
                    continue
                if pending_start is None:
                    pending_start = timestamp
                pending_texts.append(content)
                pending_characters += len(content)
                if pending_characters >= TIMELINE_CHUNK_CHARACTERS:
                    close_pending()
            close_pending()
        return chunks

    def collect_notes_chunks(self) -> list[IndexedChunk]:
        chunks: list[IndexedChunk] = []
        if not self.notes_path.exists():
            return chunks
        with open(self.notes_path, "r", encoding="utf-8") as file_handle:
            notes_text = file_handle.read()
        paragraphs = notes_text.split("\n\n")
        current = ""
        for paragraph in paragraphs:
            stripped = paragraph.strip()
            if len(stripped) == 0:
                continue
            if len(current) > 0 and (len(current) + len(stripped)) > NOTES_CHUNK_CHARACTERS:
                chunks.append(IndexedChunk(make_chunk_id("notes", current), current, "notes", "notes.md"))
                current = ""
            current = (current + "\n\n" + stripped).strip()
        if len(current) > 0:
            chunks.append(IndexedChunk(make_chunk_id("notes", current), current, "notes", "notes.md"))
        return chunks

    async def build_index(self) -> int:
        wanted_chunks = self.collect_timeline_chunks() + self.collect_notes_chunks()
        wanted_ids: set[str] = set()
        for chunk in wanted_chunks:
            wanted_ids.add(chunk.chunk_id)

        existing = self.collection.get(include=[])
        existing_ids: set[str] = set()
        for existing_id in existing["ids"]:
            existing_ids.add(existing_id)

        stale_ids: list[str] = []
        for existing_id in existing_ids:
            if existing_id not in wanted_ids:
                stale_ids.append(existing_id)
        if len(stale_ids) > 0:
            self.collection.delete(ids=stale_ids)
            if self.storage is not None and hasattr(self.storage, "delete_embeddings"):
                try:
                    await self.storage.delete_embeddings(self.session_dir.name, stale_ids)
                except Exception as del_err:
                    logger.debug("deleting stale postgres embeddings failed: %s", del_err)

        if self.storage is not None and hasattr(self.storage, "delete_session_embeddings"):
            try:
                await self.storage.delete_session_embeddings(self.session_dir.name, exclude_model=self.embedding_model)
            except Exception as clean_err:
                logger.debug("cleaning non-matching postgres embeddings failed: %s", clean_err)

        new_chunks: list[IndexedChunk] = []
        seen_new: set[str] = set()
        for chunk in wanted_chunks:
            if chunk.chunk_id in existing_ids or chunk.chunk_id in seen_new:
                continue
            seen_new.add(chunk.chunk_id)
            new_chunks.append(chunk)

        pg_missing_chunks: list[IndexedChunk] = []
        if self.storage is not None and hasattr(self.storage, "insert_embedding"):
            already_in_chroma = [c for c in wanted_chunks if c.chunk_id in existing_ids]
            if already_in_chroma:
                existing_in_pg: set[str] = set()
                if hasattr(self.storage, "get_connection") and hasattr(self.storage, "_lock"):
                    try:
                        async with self.storage._lock:
                            conn = await self.storage.get_connection()
                            async with conn.cursor() as cur:
                                await cur.execute(
                                    "SELECT source_id FROM embeddings WHERE session_id = %s;",
                                    (self.session_dir.name,)
                                )
                                rows = await cur.fetchall()
                                for r in rows:
                                    existing_in_pg.add(str(r["source_id"]))
                    except Exception as pg_check_err:
                        logger.debug("checking postgres indexed chunks failed: %s", pg_check_err)
                for c in already_in_chroma:
                    if c.chunk_id not in existing_in_pg:
                        pg_missing_chunks.append(c)

        if len(new_chunks) > 0:
            texts: list[str] = []
            for chunk in new_chunks:
                texts.append(chunk.text)
            vectors = await self.generate_embeddings(texts)
            if self.storage is not None and hasattr(self.storage, "insert_embedding"):
                for chunk, vector in zip(new_chunks, vectors):
                    try:
                        await self.storage.insert_embedding(
                            session_id=self.session_dir.name,
                            source_type=chunk.source,
                            source_id=chunk.chunk_id,
                            chunk_text=chunk.text,
                            vector=vector,
                            embedding_model=self.embedding_model
                        )
                    except Exception as err:
                        logger.debug("persisting embedding to postgres failed: %s", err)
            ids: list[str] = []
            metadatas: list[dict[str, str]] = []
            for chunk in new_chunks:
                ids.append(chunk.chunk_id)
                metadatas.append({"source": chunk.source, "reference": chunk.reference})
            self.collection.upsert(
                ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas
            )

        if len(pg_missing_chunks) > 0 and self.storage is not None and hasattr(self.storage, "insert_embedding"):
            try:
                chroma_data = self.collection.get(
                    ids=[c.chunk_id for c in pg_missing_chunks],
                    include=["embeddings"]
                )
                chroma_id_to_vec = {}
                for cid, cvec in zip(chroma_data.get("ids", []), chroma_data.get("embeddings", [])):
                    chroma_id_to_vec[cid] = cvec
                for chunk in pg_missing_chunks:
                    vec = chroma_id_to_vec.get(chunk.chunk_id)
                    if vec is not None:
                        try:
                            await self.storage.insert_embedding(
                                session_id=self.session_dir.name,
                                source_type=chunk.source,
                                source_id=chunk.chunk_id,
                                chunk_text=chunk.text,
                                vector=vec,
                                embedding_model=self.embedding_model
                            )
                        except Exception as retry_err:
                            logger.debug("retrying embedding to postgres failed: %s", retry_err)
            except Exception as fetch_err:
                logger.debug("fetching chroma vectors for postgres retry failed: %s", fetch_err)
        return int(self.collection.count())

    async def retrieve(self, question: str) -> list[tuple[str, str]]:
        total = int(self.collection.count())
        has_pg = self.storage is not None and hasattr(self.storage, "search_embeddings")
        if total == 0 and not has_pg:
            return []
        question_vectors = await self.generate_embeddings([question])
        retrieved: list[tuple[str, str]] = []
        seen_texts: set[str] = set()

        if has_pg:
            try:
                pg_results = await self.storage.search_embeddings(
                    session_id=self.session_dir.name,
                    query_vector=question_vectors[0],
                    top_k=self.top_k,
                    embedding_model=self.embedding_model
                )
                for row in pg_results:
                    txt = str(row.get("chunk_text") or "")
                    if txt and txt not in seen_texts:
                        seen_texts.add(txt)
                        retrieved.append((txt, str(row.get("source_id", "notes.md"))))
            except Exception as err:
                logger.debug("pgvector search fallback to chroma: %s", err)

        if total > 0 and len(retrieved) < self.top_k:
            result = self.collection.query(
                query_embeddings=[question_vectors[0]],
                n_results=min(self.top_k, total)
            )
            documents = result["documents"][0]
            metadatas = result["metadatas"][0]
            for index in range(len(documents)):
                doc = str(documents[index])
                if doc not in seen_texts:
                    seen_texts.add(doc)
                    reference = "unknown"
                    if metadatas[index] is not None:
                        reference = str(metadatas[index].get("reference", "unknown"))
                    retrieved.append((doc, reference))
                    if len(retrieved) >= self.top_k:
                        break
        return retrieved

    async def call_language_model(self, prompt: str) -> str:
        if self.llm_router is not None:
            response = await self.llm_router.ainvoke([HumanMessage(content=prompt)])
            return response_to_text(response)
        async with httpx.AsyncClient(timeout=self.llm_timeout_seconds) as client:
            response = await client.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.fallback_model, "prompt": prompt, "stream": False}
            )
            response.raise_for_status()
            return str(response.json().get("response", "")).strip()

    async def ask_with_sources(self, question: str) -> tuple[str, list[str]]:
        await self.build_index()
        retrieved = await self.retrieve(question)
        if len(retrieved) == 0:
            return "No indexed lecture content is available for this session yet.", []

        context_blocks: list[str] = []
        references: list[str] = []
        for index, (document_text, reference) in enumerate(retrieved, start=1):
            context_blocks.append(f"[{index}] ({reference}) {document_text}")
            if reference not in references:
                references.append(reference)
        context_text = "\n\n".join(context_blocks)

        prompt = f"""
You answer questions about a lecture using only the excerpts below.
If the excerpts do not contain the answer, say so.
Cite excerpt numbers like [1] where you use them.

EXCERPTS:
{context_text}

QUESTION: {question}
"""
        answer = await self.call_language_model(prompt)
        return answer, references

    async def ask(self, question: str) -> str:
        answer, references = await self.ask_with_sources(question)
        if len(references) == 0:
            return answer
        source_line_list: list[str] = []
        for reference in references:
            source_line_list.append(f"- {reference}")
        source_lines = "\n".join(source_line_list)
        return f"{answer}\n\nSources:\n{source_lines}"
