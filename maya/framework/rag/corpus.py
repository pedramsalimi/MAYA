"""Corpus loading and snapshot utilities for the legacy MAYA health RAG."""

from __future__ import annotations

import re
import json
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


SUPPORTED_SUFFIXES = {".pdf", ".txt"}
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 150


def corpus_root(root_dir: Path) -> Path:
    literature_dir = root_dir / "data" / "literature" / "raw"
    if literature_dir.exists():
        return literature_dir
    return root_dir / "data" / "health_papers"


def index_root(root_dir: Path) -> Path:
    return root_dir / "data" / "literature" / "indexes"


def discover_corpus_paths(root_dir: Path) -> list[Path]:
    raw_dir = corpus_root(root_dir)
    if not raw_dir.exists():
        return []
    return sorted(path for path in raw_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)


def _title_from_path(path: Path) -> str:
    return re.sub(r"[_-]+", " ", path.stem).strip().title()


def _normalize_text(text: str) -> str:
    cleaned = text.replace("-\n", "").replace("\n", " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def _load_pdf(path: Path) -> list[Document]:
    try:
        with redirect_stderr(StringIO()):
            reader = PdfReader(str(path))
    except Exception:
        return []

    documents: list[Document] = []
    for page_number, page in enumerate(reader.pages):
        try:
            with redirect_stderr(StringIO()):
                text = _normalize_text(page.extract_text() or "")
        except Exception:
            continue
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "title": _title_from_path(path),
                    "source_path": str(path),
                    "file_name": path.name,
                    "document_id": path.stem,
                    "page_number": page_number,
                },
            )
        )
    return documents


def _load_text(path: Path) -> list[Document]:
    try:
        text = _normalize_text(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not text:
        return []
    return [
        Document(
            page_content=text,
            metadata={
                "title": _title_from_path(path),
                "source_path": str(path),
                "file_name": path.name,
                "document_id": path.stem,
                "page_number": 0,
            },
        )
    ]


def load_corpus_documents(root_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for path in discover_corpus_paths(root_dir):
        if path.suffix.lower() == ".pdf":
            documents.extend(_load_pdf(path))
        elif path.suffix.lower() == ".txt":
            documents.extend(_load_text(path))
    return documents


def split_documents(
    documents: list[Document],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    chunks: list[Document] = []
    for document in documents:
        text = _normalize_text(document.page_content)
        if not text:
            continue
        start = 0
        chunk_index = 0
        while start < len(text):
            stop = min(len(text), start + chunk_size)
            if stop < len(text):
                boundary = text.rfind(" ", max(start + chunk_size // 2, start), stop)
                if boundary > start:
                    stop = boundary
            piece = text[start:stop].strip()
            if piece:
                metadata = dict(document.metadata)
                metadata["chunk_id"] = f"{metadata.get('document_id', 'doc')}:{chunk_index}"
                chunks.append(Document(page_content=piece, metadata=metadata))
                chunk_index += 1
            if stop >= len(text):
                break
            start = max(stop - chunk_overlap, start + 1)
    return chunks


def chunk_snapshot_path(root_dir: Path) -> Path:
    return index_root(root_dir) / "health_rag_chunks.jsonl"


def build_metadata_path(root_dir: Path) -> Path:
    return index_root(root_dir) / "health_rag.build.json"


def write_chunk_snapshot(chunks: list[Document], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for document in chunks:
            payload = {
                "page_content": document.page_content,
                "metadata": dict(document.metadata),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def load_chunk_snapshot(path: Path) -> list[Document]:
    if not path.exists():
        return []

    chunks: list[Document] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            page_content = str(payload.get("page_content") or "").strip()
            metadata = payload.get("metadata") or {}
            if not page_content or not isinstance(metadata, dict):
                continue
            chunks.append(Document(page_content=page_content, metadata=metadata))
    return chunks
