from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

from .text_utils import chunk_text, normalize_name, safe_metadata_value, short_hash


class OpenAIEmbeddingFunction:
    """Small Chroma-compatible embedding function using the OpenAI embeddings API."""

    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI()

    def __call__(self, input):
        texts = self._normalize_input(input)
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, input):
        return self.__call__(input)

    def embed_documents(self, input):
        return self.__call__(input)

    def name(self) -> str:
        return "OpenAIEmbeddingFunction"

    def get_config(self) -> dict[str, str]:
        return {"model": self.model}

    @staticmethod
    def build_from_config(config: dict[str, str]) -> "OpenAIEmbeddingFunction":
        return OpenAIEmbeddingFunction(config["model"])

    @staticmethod
    def _normalize_input(input) -> list[str]:
        if isinstance(input, str):
            return [input]
        return [str(x) for x in input]


class MemoryStore:
    def __init__(self, storage_dir: str, embedding_model: str, collection_name: str = "interview_memory"):
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=storage_dir)
        self.embedding_function = self._select_embedding_function(collection_name, embedding_model)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def _select_embedding_function(self, collection_name: str, embedding_model: str):
        existing_dim = self._collection_embedding_dimension(collection_name)
        default_ef = embedding_functions.DefaultEmbeddingFunction()

        if existing_dim is not None:
            default_dim = self._embedding_dimension(default_ef)
            if default_dim == existing_dim:
                return default_ef

            openai_ef = OpenAIEmbeddingFunction(embedding_model)
            openai_dim = self._embedding_dimension(openai_ef)
            if openai_dim == existing_dim:
                return openai_ef

            raise RuntimeError(
                f"Existing Chroma collection '{collection_name}' uses {existing_dim}-dimensional "
                "embeddings, but none of the configured embedding functions match it. "
                "Use a matching embedding model or clear the configured storage_dir and re-ingest."
            )

        try:
            openai_ef = OpenAIEmbeddingFunction(embedding_model)
            self._embedding_dimension(openai_ef)
            return openai_ef
        except Exception:
            return default_ef

    def _collection_embedding_dimension(self, collection_name: str) -> int | None:
        try:
            collection = self.client.get_collection(name=collection_name)
        except Exception as exc:
            if not isinstance(exc, ValueError) and exc.__class__.__name__ != "NotFoundError":
                raise
            return None

        if collection.count() == 0:
            return None

        result = collection.get(limit=1, include=["embeddings"])
        embeddings = result.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return None

        return len(embeddings[0])

    @staticmethod
    def _embedding_dimension(embedding_function) -> int:
        embedding = embedding_function(["test"])[0]
        return len(embedding)

    def upsert_text(
        self,
        *,
        owner_type: str,
        owner_name: str,
        source_type: str,
        source_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: int = 1200,
        overlap: int = 180,
    ) -> int:
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return 0

        base_meta = {
            "owner_type": owner_type,
            "owner_name": normalize_name(owner_name),
            "owner_label": owner_name,
            "source_type": source_type,
            "source_id": source_id,
        }

        if metadata:
            base_meta.update({k: safe_metadata_value(v) for k, v in metadata.items()})

        ids: list[str] = []
        metadatas: list[dict] = []

        for idx, chunk in enumerate(chunks):
            ids.append(
                f"{owner_type}:{normalize_name(owner_name)}:{source_type}:"
                f"{short_hash(source_id)}:{idx}:{short_hash(chunk)}"
            )
            metadatas.append({**base_meta, "chunk_index": idx})

        self.collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
        return len(chunks)

    def exists_source(self, source_id: str) -> bool:
        result = self.collection.get(where={"source_id": source_id}, limit=1)
        return bool(result.get("ids"))

    def has_interviewer_profile(self, interviewer_name: str) -> bool:
        result = self.collection.get(
            where={"owner_name": normalize_name(interviewer_name)},
            limit=20,
        )
        metadatas = result.get("metadatas") or []
        return any(m.get("source_type") == "interviewer_profile" for m in metadatas)

    def query(self, query_text: str, *, n_results: int = 6, where: Optional[Dict[str, Any]] = None) -> str:
        kwargs: dict[str, Any] = {"query_texts": [query_text], "n_results": n_results}
        if where:
            kwargs["where"] = where

        result = self.collection.query(**kwargs)
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]

        lines: list[str] = []
        for doc, meta in zip(docs, metas):
            label = meta.get("owner_label") or meta.get("owner_name") or "unknown"
            source_type = meta.get("source_type", "source")
            source_id = meta.get("source_id", "")
            lines.append(f"[source={source_type}; owner={label}; id={source_id}]\n{doc}")

        return "\n\n".join(lines)

    def get_by_source(self, source_id: str, limit: int = 20) -> str:
        result = self.collection.get(where={"source_id": source_id}, limit=limit)
        docs = result.get("documents") or []
        return "\n\n".join(docs)
