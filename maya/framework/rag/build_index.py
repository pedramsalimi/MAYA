"""Build the persisted health-RAG index from the raw literature folder."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from langchain_core.documents import Document

from maya.framework.rag.corpus import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    build_metadata_path,
    chunk_snapshot_path,
    load_corpus_documents,
    split_documents,
    write_chunk_snapshot,
)
from maya.framework.rag.vectorstore import build_embeddings, build_vector_store, load_rag_vector_config


def _retry_delay_seconds(error: Exception) -> float | None:
    message = str(error)
    match = re.search(r"retry after (\d+)", message, flags=re.I)
    if match:
        return float(match.group(1))
    if "RateLimit" in type(error).__name__ or "rate limit" in message.lower():
        return 15.0
    return None


def _index_chunks(
    vector_store: object,
    chunks: list[Document],
    *,
    batch_size: int = 48,
    pause_seconds: float = 1.0,
    max_retries: int = 6,
) -> None:
    total_batches = max((len(chunks) + batch_size - 1) // batch_size, 1)
    for batch_number, start in enumerate(range(0, len(chunks), batch_size), start=1):
        batch = chunks[start : start + batch_size]
        attempt = 0
        while True:
            try:
                vector_store.add_documents(batch)
                break
            except Exception as error:
                delay = _retry_delay_seconds(error)
                if delay is None or attempt >= max_retries:
                    raise
                wait_seconds = min(delay * (2**attempt), 120.0)
                print(
                    f"Rate limited while indexing batch {batch_number}/{total_batches}. "
                    f"Sleeping {wait_seconds:.1f}s before retry."
                )
                time.sleep(wait_seconds)
                attempt += 1
        if batch_number < total_batches and pause_seconds > 0:
            time.sleep(pause_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the persisted MAYA health-RAG index from data/literature/raw.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--pause-seconds", type=float, default=1.0)
    args = parser.parse_args()

    root = args.root.resolve()
    documents = load_corpus_documents(root)
    if not documents:
        raise RuntimeError(f"No readable literature sources found under {root / 'data' / 'literature' / 'raw'}.")
    source_documents = len({str(doc.metadata.get("source_path", "")) for doc in documents if doc.metadata.get("source_path")})

    chunks = split_documents(
        documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    if not chunks:
        raise RuntimeError("Literature sources were found, but no readable text chunks were produced.")

    vector_config = load_rag_vector_config()
    if vector_config.backend != "pgvector":
        raise RuntimeError("Set MAYA_DB_URI or MAYA_HEALTH_RAG_BACKEND=pgvector before building the persisted index.")

    embeddings = build_embeddings(vector_config)
    vector_store = build_vector_store(vector_config, embeddings, pre_delete=True)
    _index_chunks(
        vector_store,
        chunks,
        batch_size=max(args.batch_size, 1),
        pause_seconds=max(args.pause_seconds, 0.0),
    )

    snapshot = write_chunk_snapshot(chunks, chunk_snapshot_path(root))
    build_meta = {
        "backend": vector_config.backend,
        "collection_name": vector_config.collection_name,
        "embedding_backend": vector_config.embedding_backend,
        "embedding_model": vector_config.embedding_model,
        "source_documents": source_documents,
        "chunks": len(chunks),
        "chunk_snapshot": str(snapshot),
    }
    meta_path = build_metadata_path(root)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(build_meta, indent=2), encoding="utf-8")

    print(f"Built persisted health RAG with {source_documents} source documents and {len(chunks)} chunks.")
    print(f"Backend: {vector_config.backend}")
    print(f"Collection: {vector_config.collection_name}")
    print(f"Chunk snapshot: {snapshot}")
    print(f"Build metadata: {meta_path}")


if __name__ == "__main__":
    main()
