"""Administrative commands.

Usage:
    python -m app.admin reembed [--machine ID] [--embed-model MODEL] [--confirm]

Re-embeds documents for one machine (or every machine) with that machine's embed model.
Vectors are stored in per-dimension columns, so machines can use different models and
dimensions at the same time; no global schema change is involved.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from sqlalchemy import text

from .config import SettingsError, get_settings
from .db import ensure_vector_column, get_engine, session_scope
from .ingest.pipeline import index_document, set_status
from .retrieval.keyword import invalidate as invalidate_bm25

logger = logging.getLogger(__name__)


def _load():
    settings = get_settings()
    settings.validate_models()

    engine = get_engine()
    with engine.connect() as conn:
        machines = (
            conn.execute(
                text("SELECT id, title, embed_model_id, embed_dim FROM machines")
            )
            .mappings()
            .all()
        )

    return settings, [dict(m) for m in machines]


def reembed(machine_id: int | None, embed_model: str | None, confirm: bool) -> int:
    settings, machines = _load()

    if machine_id is not None:
        machines = [m for m in machines if int(m["id"]) == machine_id]
        if not machines:
            print(f"Machine {machine_id} not found.")
            return 2

    if embed_model:
        price = settings.embed_prices.get(embed_model)
        if price is None:
            known = ", ".join(sorted(settings.embed_prices))
            print(f"Unknown embed model '{embed_model}'. Configured: {known}")
            return 2

    for machine in machines:
        model_id = embed_model or machine["embed_model_id"]
        price = settings.embed_prices.get(model_id)
        if price is None:
            known = ", ".join(sorted(settings.embed_prices))
            print(
                f"Machine {machine['id']} uses '{model_id}', which is not configured. "
                f"Configured: {known}. Pass --embed-model MODEL."
            )
            return 2
        machine["_target_model"] = model_id
        machine["_target_dim"] = int(price.dim)

    print("Re-embed plan:")
    for machine in machines:
        print(
            f"  machine {machine['id']} '{machine['title']}': "
            f"{machine['embed_model_id']} (dim {machine['embed_dim']}) -> "
            f"{machine['_target_model']} (dim {machine['_target_dim']})"
        )

    if not confirm:
        print()
        print("This calls the embedding API for every document of the selected machines.")
        print("Re-run with --confirm to proceed.")
        return 1

    engine = get_engine()
    for machine in machines:
        target_model = machine["_target_model"]
        target_dim = machine["_target_dim"]

        with engine.begin() as conn:
            ensure_vector_column(conn, target_dim)
            conn.execute(
                text(
                    "UPDATE machines SET embed_model_id = :model, embed_dim = :dim "
                    "WHERE id = :id"
                ),
                {"model": target_model, "dim": target_dim, "id": machine["id"]},
            )
            conn.execute(
                text("DELETE FROM chunks WHERE machine_id = :id"), {"id": machine["id"]}
            )
            conn.execute(
                text("UPDATE documents SET chunk_count = 0 WHERE machine_id = :id"),
                {"id": machine["id"]},
            )

        with session_scope() as session:
            documents = session.execute(
                text(
                    "SELECT id, stored_path FROM documents "
                    "WHERE machine_id = :id ORDER BY id"
                ),
                {"id": machine["id"]},
            ).mappings().all()

        for document in documents:
            stored = document["stored_path"]
            path = Path(stored) if stored else None
            if path is None or not path.exists():
                set_status(int(document["id"]), "failed", "Stored file missing; re-upload.")
                continue
            print(f"  re-indexing document {document['id']}")
            index_document(int(document["id"]), path)

        invalidate_bm25(int(machine["id"]))

    print(f"Done. Re-embedded {len(machines)} machine(s).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.admin")
    sub = parser.add_subparsers(dest="command", required=True)

    reembed_parser = sub.add_parser("reembed", help="re-embed a machine's documents")
    reembed_parser.add_argument("--machine", type=int, default=None, help="machine id")
    reembed_parser.add_argument(
        "--embed-model", default=None, help="switch the selected machine(s) to this model"
    )
    reembed_parser.add_argument("--confirm", action="store_true", help="actually do it")

    args = parser.parse_args(argv)

    try:
        if args.command == "reembed":
            return reembed(
                machine_id=args.machine,
                embed_model=args.embed_model,
                confirm=args.confirm,
            )
    except SettingsError as exc:
        print(f"Configuration error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())