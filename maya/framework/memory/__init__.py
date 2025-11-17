"""Postgres-backed LangGraph memory driven by environment variables."""

from __future__ import annotations

import atexit
import os
from contextlib import ExitStack
from typing import Dict, Tuple
from langchain.embeddings import init_embeddings  # NEW
from langchain_openai import AzureOpenAIEmbeddings
from langgraph.store.base import BaseStore
from langgraph.store.postgres import PostgresStore
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.memory import MemorySaver
from openai import embeddings

_ENV_DB_URI = "MAYA_DB_URI"
_ENV_DB_URI_FALLBACK = "DATABASE_URL"
_ENV_POOL_MIN = "MAYA_DB_POOL_MIN_SIZE"
_ENV_POOL_MAX = "MAYA_DB_POOL_MAX_SIZE"

_STACK = ExitStack()
atexit.register(_STACK.close)

_STORE: BaseStore | None = None
_CHECKPOINTER: BaseCheckpointSaver | None = None


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


# def get_postgres_memory() -> Tuple[BaseStore, BaseCheckpointSaver]:
#     """Return the shared Postgres-backed store and checkpointer."""
#     global _STORE, _CHECKPOINTER

#     if _STORE is None or _CHECKPOINTER is None:
#         conn = _connection_string()
#         if conn:
#             pool = _pool_config()
#             store_cm = (
#                 PostgresStore.from_conn_string(conn, pool_config=pool)
#                 if pool
#                 else PostgresStore.from_conn_string(conn)
#             )
#             _STORE = _STACK.enter_context(store_cm)
#             _CHECKPOINTER = _STACK.enter_context(PostgresSaver.from_conn_string(conn))
#             _STORE.setup()
#             _CHECKPOINTER.setup()
#         else:
#             _STORE = InMemoryStore()
#             _CHECKPOINTER = MemorySaver()

#     return _STORE, _CHECKPOINTER
def get_postgres_memory() -> Tuple[BaseStore, BaseCheckpointSaver]:
    """Return the shared Postgres-backed store and checkpointer."""
    global _STORE, _CHECKPOINTER

    if _STORE is None or _CHECKPOINTER is None:
        conn = _connection_string()

        # We'll use the same embedding model for semantic search in both Postgres and in-memory.
        # text-embedding-3-small → 1536 dims (per OpenAI docs). 
        # embeddings = init_embeddings("openai:text-embedding-3-small")
        embeddings = AzureOpenAIEmbeddings(
            model="text-embedding-3-small",
            api_version="2024-12-01-preview",
            azure_endpoint="https://mayaagent.openai.azure.com/"
        )
        if conn:
            pool = _pool_config()

            index_config = {
                "dims": 1536,
                "embed": embeddings,
                # We will embed the "question" field of values we store (e.g. health_memory Q→A).
                # If you omit "fields", it embeds the whole JSON value.
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
            # In-memory fallback with the same semantic index config
            _STORE = InMemoryStore(
                index={
                    "dims": 1536,
                    "embed": embeddings,
                }
            )
            _CHECKPOINTER = MemorySaver()

    return _STORE, _CHECKPOINTER


# def agent_namespace(agent_id: str) -> Tuple[str, ...]:
#     """Canonical namespace for agent-scoped memory."""
#     return ("agents", agent_id)
