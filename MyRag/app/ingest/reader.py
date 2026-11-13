"""Document reading and chunking."""

from __future__ import annotations

import logging
from pathlib import Path

from llama_index.core.schema import Document

from ..config import get_settings

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "text/plain",
    "application/octet-stream",
}


class UnsupportedFileError(ValueError):
    pass


def read_document(path: Path) -> list[Document]:
    """Load a PDF or TXT file into llama_index documents."""
    from llama_index.core.readers.file.base import SimpleDirectoryReader

    suffix = path.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileError(
            f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    reader = SimpleDirectoryReader(input_files=[str(path)])
    return reader.load_data()


def _build_splitter(chunk_size: int, chunk_overlap: int):
    """Pick a splitter that does not depend on NLTK data files.

    SentenceSplitter loads NLTK stopwords, which fails when the virtualenv is
    built with hardlinked packages (uv/poetry), because NLTK's path check rejects
    multiply-linked files. TokenTextSplitter needs no data files, so it is the
    default; SentenceSplitter is used only when it is actually available.
    """
    try:
        from llama_index.core.node_parser import TokenTextSplitter

        return TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except ImportError:  # pragma: no cover - core always ships this
        pass

    from llama_index.core.node_parser import SentenceSplitter

    return SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def split_documents(
    documents: list[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
    max_chunks: int,
) -> tuple[list, bool]:
    """Split documents into chunks. Returns (chunks, truncated)."""
    splitter = _build_splitter(chunk_size, chunk_overlap)
    chunks = splitter.get_nodes_from_documents(documents)
    truncated = len(chunks) > max_chunks
    if truncated:
        logger.warning("Document produced %d chunks; capping at %d", len(chunks), max_chunks)
        chunks = chunks[:max_chunks]
    return chunks, truncated


ALLOWED_METADATA_KEYS = {
    "filename",
    "file_name",
    "page",
    "page_label",
    "section",
    "sheet_name",
    "source",
    "document_id",
    "machine_id",
}


def sanitize_metadata(raw: dict | None, *, machine_id: int, document_id: int) -> dict:
    """Keep only small, useful metadata; never the text or internal paths."""
    safe: dict = {"machine_id": machine_id, "document_id": document_id}
    if not raw:
        return safe

    for key, value in raw.items():
        if key in {"machine_id", "document_id"}:
            continue
        if key not in ALLOWED_METADATA_KEYS:
            continue
        if isinstance(value, str):
            if len(value) > 300:
                continue
            safe[key] = value
        elif isinstance(value, (int, float, bool)) or value is None:
            safe[key] = value
    return safe


def validate_upload(filename: str, content_type: str, byte_size: int) -> str:
    """Return the lowercase extension, raising on unsupported or oversized uploads."""
    settings = get_settings()
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileError(
            f"Unsupported file type '{suffix or filename}'. Allowed: "
            f"{', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    if content_type and content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise UnsupportedFileError(f"Unsupported content type '{content_type}'.")
    if byte_size <= 0:
        raise ValueError("Uploaded file is empty.")
    if byte_size > settings.max_upload_bytes:
        raise ValueError(
            f"File is {byte_size / 1024 / 1024:.1f} MB; limit is {settings.max_upload_mb} MB."
        )
    return suffix