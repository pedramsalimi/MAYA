"""Postgres-backed LangGraph memory driven by environment variables."""

from __future__ import annotations

import atexit
import os
import re
from contextlib import ExitStack
from typing import Dict, Tuple

from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.base import BaseStore
from langgraph.store.postgres import PostgresStore

_ENV_DB_URI = "MAYA_DB_URI"
_ENV_DB_URI_FALLBACK = "DATABASE_URL"
_ENV_POOL_MIN = "MAYA_DB_POOL_MIN_SIZE"
_ENV_POOL_MAX = "MAYA_DB_POOL_MAX_SIZE"

_STACK = ExitStack()
atexit.register(_STACK.close)

_STORE: BaseStore | None = None
_CHECKPOINTER: BaseCheckpointSaver | None = None


class DeterministicMemoryEmbeddings(Embeddings):
    """Local, deterministic embeddings for semantic memory search."""

    def __init__(self, size: int = 1536):
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


def _connection_string() -> str:
    conn = os.getenv(_ENV_DB_URI) or os.getenv(_ENV_DB_URI_FALLBACK)
    return conn or ""


def _pool_config() -> Dict[str, int] | None:
    cfg: Dict[str, int] = {}
    if os.getenv(_ENV_POOL_MIN):
        cfg["min_size"] = int(os.getenv(_ENV_POOL_MIN))
    if os.getenv(_ENV_POOL_MAX):
        cfg["max_size"] = int(os.getenv(_ENV_POOL_MAX))
    return cfg or None


def get_postgres_memory() -> Tuple[BaseStore, BaseCheckpointSaver]:
    """Return the shared Postgres-backed store and checkpointer."""
    global _STORE, _CHECKPOINTER

    if _STORE is None or _CHECKPOINTER is None:
        conn = _connection_string()
        embeddings = DeterministicMemoryEmbeddings(size=1536)
        if conn:
            pool = _pool_config()
            index_config = {
                "dims": 1536,
                "embed": embeddings,
                "fields": ["question"],
            }

            store_cm = (
                PostgresStore.from_conn_string(conn, pool_config=pool, index=index_config)
                if pool
                else PostgresStore.from_conn_string(conn, index=index_config)
            )

            _STORE = _STACK.enter_context(store_cm)
            _CHECKPOINTER = _STACK.enter_context(PostgresSaver.from_conn_string(conn))
            _STORE.setup()
            _CHECKPOINTER.setup()
        else:
            raise RuntimeError("MAYA_DB_URI or DATABASE_URL must be set for memory persistence.")

    return _STORE, _CHECKPOINTER
