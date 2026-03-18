"""Vector-store and embedding helpers for the legacy MAYA health RAG."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

load_dotenv()


VectorBackend = Literal["pgvector", "in_memory"]
EmbeddingBackend = Literal["huggingface", "deterministic"]


@dataclass(frozen=True)
class RagVectorConfig:
    backend: VectorBackend
    embedding_backend: EmbeddingBackend
    collection_name: str
    embedding_model: str
    connection_string: str | None = None
    embedding_length: int | None = None


class DeterministicEmbeddings(Embeddings):
    """Small deterministic embedding backend used when remote embeddings are unavailable."""

    def __init__(self, size: int = 768):
        self.size = size

    def _embed(self, text: str) -> list[float]:
        values = [0.0] * self.size
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            values[hash(token) % self.size] += 1.0
        norm = sum(value * value for value in values) ** 0.5 or 1.0
        return [value / norm for value in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _normalize_connection_string(value: str | None) -> str | None:
    if not value:
        return None
    if value.startswith("postgresql+psycopg://"):
        return value
    if value.startswith("postgresql://"):
        return value.replace("postgresql://", "postgresql+psycopg://", 1)
    if value.startswith("postgres://"):
        return value.replace("postgres://", "postgresql+psycopg://", 1)
    return value


def load_rag_vector_config() -> RagVectorConfig:
    db_uri = _normalize_connection_string(
        os.getenv("MAYA_DB_URI") or os.getenv("CARDIO_ONCOLOGY_DB_URI") or os.getenv("DATABASE_URL")
    )
    backend_value = os.getenv("MAYA_HEALTH_RAG_BACKEND") or os.getenv("CARDIO_ONCOLOGY_RAG_BACKEND")
    backend = backend_value if backend_value in {"pgvector", "in_memory"} else ("pgvector" if db_uri else "in_memory")

    embedding_value = os.getenv("MAYA_HEALTH_RAG_EMBEDDINGS") or os.getenv("CARDIO_ONCOLOGY_EMBEDDING_BACKEND")
    if embedding_value in {"huggingface", "local"}:
        embedding_backend: EmbeddingBackend = "huggingface"
    elif embedding_value in {"deterministic", "local", "fake"}:
        embedding_backend = "deterministic"
    else:
        embedding_backend = "huggingface" if backend == "pgvector" else "deterministic"

    embedding_model = (
        os.getenv("MAYA_HEALTH_RAG_EMBEDDING_MODEL")
        or os.getenv("CARDIO_ONCOLOGY_EMBEDDING_MODEL")
        or "sentence-transformers/all-mpnet-base-v2"
    )
    embedding_length = (
        os.getenv("MAYA_HEALTH_RAG_EMBEDDING_LENGTH")
        or os.getenv("CARDIO_ONCOLOGY_EMBEDDING_LENGTH")
        or "768"
    )

    return RagVectorConfig(
        backend=backend,
        embedding_backend=embedding_backend,
        collection_name=os.getenv("MAYA_HEALTH_RAG_COLLECTION", "maya_health_rag"),
        embedding_model=embedding_model,
        connection_string=db_uri,
        embedding_length=int(embedding_length) if embedding_length else None,
    )


def build_embeddings(config: RagVectorConfig) -> Embeddings:
    if config.embedding_backend == "deterministic":
        return DeterministicEmbeddings(size=config.embedding_length or 768)
    local_files_only = os.getenv("MAYA_HEALTH_RAG_EMBEDDING_LOCAL_ONLY", "false").lower() in {"1", "true", "yes"}
    return HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        model_kwargs={
            "device": os.getenv("MAYA_HEALTH_RAG_EMBEDDING_DEVICE", "cpu"),
            "local_files_only": local_files_only,
        },
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(config: RagVectorConfig, embeddings: Any, *, pre_delete: bool = False) -> Any:
    if config.backend == "in_memory":
        return InMemoryVectorStore(embedding=embeddings)
    if not config.connection_string:
        raise RuntimeError("MAYA_DB_URI, CARDIO_ONCOLOGY_DB_URI, or DATABASE_URL must be set for pgvector mode.")
    return PGVector(
        embeddings=embeddings,
        connection=config.connection_string,
        collection_name=config.collection_name,
        embedding_length=config.embedding_length,
        pre_delete_collection=pre_delete,
        create_extension=True,
    )
