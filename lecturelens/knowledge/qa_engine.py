import json
from pathlib import Path
from typing import Any, Optional
import httpx
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.messages import HumanMessage


class QAEngine:
    def __init__(
        self,
        session_dir: Path,
        ollama_url: str = "http://localhost:11434",
        llm_router: Any = None
    ) -> None:
        self.session_dir: Path = session_dir
        self.ollama_url: str = ollama_url
        self.llm_router: Any = llm_router
        self.chroma_db_dir: Path = session_dir / "chroma_db"
        self.chroma_db_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_client: chromadb.PersistentClient = chromadb.PersistentClient(
            path=str(self.chroma_db_dir),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(name="lecture_content")

    async def generate_embedding(self, text: str) -> list[float]:
        endpoint = f"{self.ollama_url}/api/embeddings"
        payload = {
            "model": "nomic-embed-text",
            "prompt": text,
            "keep_alive": 0
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
            embedding = data.get("embedding", [])
            return embedding

    async def build_index(self) -> int:
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        ids: list[str] = []
        document_counter = 0

        timeline_path = self.session_dir / "timeline.jsonl"
        if timeline_path.exists():
            with open(timeline_path, "r", encoding="utf-8") as file_handle:
                for line in file_handle:
                    line_clean = line.strip()
                    if len(line_clean) == 0:
                        continue
                    try:
                        record = json.loads(line_clean)
                        content = record.get("content", "").strip()
                        if len(content) == 0:
                            continue

                        source_type = record.get("source_type", "unknown")
                        timestamp_epoch = record.get("timestamp_epoch", 0.0)

                        document_counter += 1
                        doc_id = f"timeline_{document_counter}"
                        formatted_text = f"[{source_type.upper()}] {content}"

                        documents.append(formatted_text)
                        metadatas.append({
                            "source": "timeline",
                            "source_type": source_type,
                            "timestamp_epoch": timestamp_epoch
                        })
                        ids.append(doc_id)
                    except Exception:
                        continue

        notes_path = self.session_dir / "notes.md"
        if notes_path.exists():
            with open(notes_path, "r", encoding="utf-8") as file_handle:
                notes_text = file_handle.read()

            paragraphs = notes_text.split("\n\n")
            for paragraph in paragraphs:
                paragraph_clean = paragraph.strip()
                if len(paragraph_clean) == 0:
                    continue

                document_counter += 1
                doc_id = f"notes_{document_counter}"
                documents.append(paragraph_clean)
                metadatas.append({
                    "source": "notes",
                    "source_type": "structured_note",
                    "timestamp_epoch": 0.0
                })
                ids.append(doc_id)

        if len(documents) == 0:
            return 0

        embeddings_list: list[list[float]] = []
        for doc in documents:
            embedding = await self.generate_embedding(doc)
            embeddings_list.append(embedding)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings_list,
            metadatas=metadatas
        )
        return len(documents)

    async def ask(self, question: str, top_k: int = 5) -> str:
        question_embedding = await self.generate_embedding(question)
        query_results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k
        )

        retrieved_documents: list[str] = []
        if query_results and "documents" in query_results:
            first_docs = query_results["documents"]
            if len(first_docs) > 0:
                retrieved_documents = first_docs[0]

        context_text = "\n\n".join(retrieved_documents)
        prompt_content = f"""
You are an expert AI academic tutor and lecture assistant.
Answer the student's question based strictly on the retrieved lecture transcript and notes context below.

CONTEXT:
{context_text}

STUDENT QUESTION:
{question}

Provide a clear, detailed, and accurate answer. If the context does not contain the answer, state that clearly.
"""

        if self.llm_router is not None:
            message = HumanMessage(content=prompt_content)
            response = await self.llm_router.ainvoke([message])
            if hasattr(response, "content"):
                return str(response.content)
            return str(response)

        endpoint = f"{self.ollama_url}/api/generate"
        payload = {
            "model": "gemma4:e4b",
            "prompt": prompt_content,
            "stream": False
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
