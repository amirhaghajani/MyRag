"""Administrative commands.

Usage:
    python -m app.admin reembed --confirm [--embed-model MODEL]

Changing EMBED_DIM (or switching to an embed model with a different dimension) means the
single chunks.vector column must be rebuilt and every document re-embedded. This command
does that in one step; the web app refuses to start until the column matches EMBED_DIM.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from sqlalchemy import text

from .config import SettingsError, get_settings
from .db import get_engine, set_vector_dim, vector_dim
from .ingest.pipeline import index_document, set_status
from .retrieval.keyword import invalidate as invalidate_bm25

logger = logging.getLogger(__name__)


def _load():
    settings = get_settings()
    settings.validate_models()

    engine = get_engine()
    with engine.connect() as conn:
        current = vector_dim(conn)
        machines = (
            conn.execute(
                text(
                    "SELECT id, title, embed_model_id, embed_dim FROM machines ORDER BY id"
                )
            )
            .mappings()
            .all()
        )
        documents = (
            conn.execute(
                text(
                    "SELECT id, machine_id, filename, stored_path FROM documents ORDER BY id"
                )
            )
            .mappings()
            .all()
        )
        chunks = conn.execute(text("SELECT count(*) FROM chunks")).scalar_one()

    return settings, current, [dict(m) for m in machines], [dict(d) for d in documents], int(chunks)


def reembed(confirm: bool, embed_model: str | None) -> int:
    settings, current, machines, documents, chunk_count = _load()
    target = settings.embed_dim

    print(f"Server EMBED_DIM = {target}")
    print(f"Database chunks.vector = {'(missing)' if current is None else f'vector({current})'}")
    print(f"Machines: {len(machines)}, documents: {len(documents)}, chunks: {chunk_count}")
    for machine in machines:
        print(
            f"  machine {machine['id']} '{machine['title']}': "
            f"{machine['embed_model_id']} (dim {machine['embed_dim']})"
        )

    stale_dims = [m for m in machines if int(m["embed_dim"]) != target]
    if stale_dims and current == target:
        print(
            "Note: these machines report a dimension other than "
            f"{target}: {', '.join(str(m['id']) for m in stale_dims)}. "
            "Fix them from the machine's Embedding settings (same-dim model + re-embed)."
        )

    if embed_model:
        price = settings.embed_prices.get(embed_model)
        if price is None:
            known = ", ".join(sorted(settings.embed_prices))
            print(f"Unknown embed model '{embed_model}'. Configured: {known}")
            return 2
        if price.dim != target:
            print(
                f"Embed model '{embed_model}' has dim {price.dim} but EMBED_DIM is {target}."
            )
            return 2

    if current != target:
        stale = [
            m
            for m in machines
            if settings.embed_prices.get(m["embed_model_id"]) is None
            or settings.embed_prices[m["embed_model_id"]].dim != target
        ]
        if stale and not embed_model:
            print(
                "These machines still use a model whose dimension is not "
                f"{target}: {', '.join(str(m['id']) for m in stale)}. "
                "Pass --embed-model MODEL (dim "
                f"{target}) so every machine can be re-embedded."
            )
            return 2

    if not confirm:
        print()
        print("This deletes all chunks and calls the embedding API for every document.")
        print("Re-run with --confirm to proceed.")
        return 1

    engine = get_engine()

    with engine.begin() as conn:
        if current != target:
            conn.execute(text("DELETE FROM chunks"))
            conn.execute(text("UPDATE documents SET chunk_count = 0"))
            set_vector_dim(conn, target)
            conn.execute(text("UPDATE machines SET embed_dim = :dim"), {"dim": target})

        if embed_model:
            conn.execute(
                text("UPDATE machines SET embed_model_id = :model, embed_dim = :dim"),
                {"model": embed_model, "dim": target},
            )

    if current != target:
        # Recreate the index on the rebuilt column.
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS chunks_vector_hnsw_idx "
                    "ON chunks USING hnsw (vector vector_cosine_ops)"
                )
            )

    failed = 0
    for document in documents:
        stored = document["stored_path"]
        path = Path(stored) if stored else None
        if path is None or not path.exists():
            set_status(document["id"], "failed", "Stored file missing; re-upload.")
            failed += 1
            continue
        print(f"  re-indexing document {document['id']} ({document['filename']})")
        index_document(document["id"], path)

    for machine in machines:
        invalidate_bm25(int(machine["id"]))

    print(f"Done. Re-embedded {len(documents) - failed} documents, {failed} failed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.admin")
    sub = parser.add_subparsers(dest="command", required=True)

    reembed_parser = sub.add_parser("reembed", help="rebuild chunks.vector and re-embed")
    reembed_parser.add_argument("--confirm", action="store_true", help="actually do it")
    reembed_parser.add_argument(
        "--embed-model", default=None, help="switch every machine to this embed model"
    )

    args = parser.parse_args(argv)

    try:
        if args.command == "reembed":
            return reembed(confirm=args.confirm, embed_model=args.embed_model)
    except SettingsError as exc:
        print(f"Configuration error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())