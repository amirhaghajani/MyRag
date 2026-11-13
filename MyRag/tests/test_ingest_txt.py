"""Ingestion tests: reading, chunking, metadata sanitising, upload validation."""

from pathlib import Path

import pytest

from app.config import Settings
from app.ingest import reader
from app.ingest.pipeline import sanitize_metadata


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch):
    settings = Settings(
        max_upload_mb=1,
        chunk_size=128,
        chunk_overlap=16,
        max_chunks_per_doc=5,
        embed_dim=1536,
        llm_models="a/b=1,2,1000",
        embed_models="e/f=1536,0.02",
        default_llm_model="a/b",
        default_embed_model="e/f",
        api_base="https://x/v1",
    )
    monkeypatch.setattr(reader, "get_settings", lambda: settings)
    monkeypatch.setattr("app.ingest.pipeline.get_settings", lambda: settings, raising=False)
    yield settings


def test_read_txt_document(tmp_path: Path):
    path = tmp_path / "machine.txt"
    path.write_text(
        "Roller Mill RM-200\n\nRated capacity: 4.5 t/h\nMotor: 45 kW\n",
        encoding="utf-8",
    )
    docs = reader.read_document(path)
    assert docs
    assert "Rated capacity" in docs[0].text


def test_read_rejects_unsupported_extension(tmp_path: Path):
    path = tmp_path / "machine.docx"
    path.write_bytes(b"binary")
    with pytest.raises(reader.UnsupportedFileError):
        reader.read_document(path)


def test_validate_upload_rejects_bad_type():
    with pytest.raises(reader.UnsupportedFileError):
        reader.validate_upload("note.md", "text/markdown", 100)


def test_validate_upload_rejects_oversize():
    with pytest.raises(ValueError, match="limit"):
        reader.validate_upload("big.pdf", "application/pdf", 10 * 1024 * 1024)


def test_validate_upload_accepts_pdf_and_txt():
    assert reader.validate_upload("a.pdf", "application/pdf", 1024) == ".pdf"
    assert reader.validate_upload("a.txt", "text/plain", 1024) == ".txt"


def test_split_documents_chunks_and_caps(tmp_path: Path):
    path = tmp_path / "long.txt"
    path.write_text(
        " ".join(f"Sentence number {i} about milling." for i in range(400)),
        encoding="utf-8",
    )

    docs = reader.read_document(path)
    chunks, truncated = reader.split_documents(
        docs, chunk_size=128, chunk_overlap=16, max_chunks=5
    )

    assert len(chunks) == 5
    assert truncated is True


def test_split_documents_not_truncated_when_small():
    from llama_index.core.schema import Document

    docs = [Document(text="short text")]
    chunks, truncated = reader.split_documents(
        docs, chunk_size=512, chunk_overlap=64, max_chunks=100
    )
    assert truncated is False
    assert chunks


def test_sanitize_metadata_drops_internal_paths():
    meta = {
        "filename": "a.pdf",
        "page": 3,
        "text": "x" * 5000,
        "file_path": r"C:\uploads\abc123.pdf",
        "file_name": "abc123.pdf",
        "file_type": "text/plain",
        "file_size": 36,
        "machine_id": 1,
    }
    clean = sanitize_metadata(meta, machine_id=7, document_id=9)

    assert clean["filename"] == "a.pdf"
    assert clean["page"] == 3
    assert "text" not in clean
    # Internal storage paths and derived file metadata are dropped.
    assert "file_path" not in clean
    assert "file_size" not in clean
    assert "file_type" not in clean
    assert clean["machine_id"] == 7
    assert clean["document_id"] == 9


def test_sanitize_metadata_handles_none():
    clean = sanitize_metadata(None, machine_id=1, document_id=2)
    assert clean == {"machine_id": 1, "document_id": 2}