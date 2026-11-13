# MyRag — per-model embedding dimensions

## Problem

`EMBED_DIM` is a single global integer (`app/config.py:96`) and `chunks.vector` is one fixed-width
column (`app/db.py:184`). That cannot hold `nomic-embed-text` (768), `bge-m3` (1024) and
`text-embedding-3-small` (1536) at the same time: pgvector fixes a column's dimension, and a vector of
the wrong width is rejected with `expected 1536 dimensions, not 8` (observed against the live database
during the previous implementation).

The current code encodes this wrong assumption in four places: `Settings.validate_embed_dims()`
(`config.py:122`) rejects any model whose `dim` differs from the global; `create_machine`
(`machines.py:54`) returns 409 for the same reason; `_insert_chunks` (`pipeline.py:211`) writes
`chunks.vector`; and `EmbeddingRetrievalFlow` (`embedding.py:35`) queries `c.vector`.

## Decisions (confirmed with the user)

1. **Per-dimension vector tables.** `chunks` keeps id / machine_id / document_id / text / metadata /
   embed_model_id / embed_dim and loses the `vector` column. Each distinct dimension gets
   `chunk_vectors_<dim>(chunk_id PK REFERENCES chunks(id) ON DELETE CASCADE, embedding vector(<dim>))`
   with its own HNSW index. Retrieval resolves the dimension from the model, then joins.
2. **Several embed models per machine, chosen per query.** `chunks` already records `embed_model_id`
   per row; retrieval filters on `machine_id` **and** `embed_model_id`. Chat accepts an optional
   `embed_model_id` that defaults to the machine's.

## Design

### Dimension source of truth

`EMBED_MODELS` stays `model_id=dim,price_per_1m` and is now the only place a dimension is declared.
`.env.example` gains `nomic-embed-text=768,...` and `bge-m3=1024,...` entries.

`config.py`:
- delete `embed_dim` field, its validator, and `validate_embed_dims()`.
- add `embed_dims -> frozenset[int]` and `dim_for(model_id) -> int | None`.
- `validate_models()` drops the `validate_embed_dims()` call; keep the existing checks that each
  configured embed model's dim is a positive int (already enforced in `parse_embed_models`).

### Schema (`app/db.py`)

`create_schema(embed_dims: set[int])`:
- keep the extension check and the non-Postgres guard.
- build base metadata: `machines` **without** `embed_dim`; `chunks` **without** `vector`.
- for each dim in `embed_dims`, create `chunk_vectors_<dim>` plus an HNSW index.
- **Index limit:** pgvector caps an index at 2000 dimensions. For `dim > 2000`
  (e.g. `text-embedding-3-large` at 3072) create the table but skip the HNSW index and log a warning —
  exact `<=>` search still works, only slower.
- **Legacy guard:** if `information_schema.columns` shows `chunks.vector` still exists, raise
  `RuntimeError` with the exact reset SQL to run, rather than starting in a half-migrated state.

Table name helper, with a whitelist so a table name is never built from unvalidated input:

```python
def vector_table(dim: int) -> str:          # app/retrieval/vectors.py
    if dim not in get_settings().embed_dims:
        raise ValueError(f"No vector table for dim {dim}")
    return f"chunk_vectors_{dim}"
```

### Ingestion (`app/ingest/pipeline.py`)

- `_machine_row` no longer selects `embed_dim`.
- `_insert_chunks` inserts into `chunks` first, then into `chunk_vectors_<dim>`:
  1. insert chunk rows with `RETURNING id, chunk_index` (executemany) and build
     `chunk_index -> chunk_id`;
  2. executemany insert `(chunk_id, embedding)` into the dimension's table.
- `clear_chunks` is unchanged: deleting `chunks` rows cascades to every `chunk_vectors_*`.
- reject early with a clear error if a returned vector's length does not equal the model's configured
  dimension, instead of letting Postgres fail with `expected N dimensions`.

### Retrieval

New `app/retrieval/vectors.py`:
- `vector_table(dim)`
- `dense_search(*, machine_id, embed_model_id, question, top_k) -> list[RetrievedChunk]`:
  1. `dim = dim_for(embed_model_id)`, raise `UnknownEmbedModel` if missing;
  2. embed the question with that model;
  3. verify `len(embedding) == dim`;
  4. run, against the whitelisted table:

```sql
SELECT c.id AS chunk_id, c.document_id, c.text, c.metadata,
       1 - (v.embedding <=> CAST(:embedding AS vector)) AS score
  FROM chunk_vectors_<dim> v
  JOIN chunks c ON c.id = v.chunk_id
 WHERE c.machine_id = :machine_id
   AND c.embed_model_id = :embed_model_id
 ORDER BY v.embedding <=> CAST(:embedding AS vector)
 LIMIT :top_k
```

`embedding.py` becomes a thin wrapper over `dense_search` (keeps the class, name and description that
the workflow registry and the `workflows` table already use). `hybrid.py` uses `dense_search` for its
dense half; its full-text query is unchanged.

`registry.py` / `workflows/rag_workflow.py` need no change: they already pass `embed_model_id` through.

### Machine and chat API

- `machines.py`: drop the dim check; store only `embed_model_id`; `MachineOut.embed_dim` is computed
  from `dim_for(machine.embed_model_id)` so the existing response shape and UI keep working.
- `MachineSummary` gains `embed_models: list[{model_id, dim, chunks}]` — which models have vectors on
  this machine and how many, so the UI can offer the choice.
- `chat.py`: `ChatRequest` gains optional `embed_model_id`; when absent use the machine's. Return 409
  when the requested model has no chunks for that machine, with the list of models that do.
- `models_api.py`: remove the global `embed_dim` from `ModelsOut` (each `ModelOption` already carries
  its own `dim`).
- `usage.py`: add `by_embed_model` to the machine usage payload, grouped from `ingest_usage`
  (`embed_model_id`, `char_count`, `token_estimate`, `cost_usd`) — this is what makes embedder
  comparison meaningful now that one machine can hold several.

### Tests

- `tests/sqlite_schema.py`: drop `vector` from `chunks`, add `chunk_vectors_8` (the test dim).
- `test_pricing.py`: replace `test_embed_dim_mismatch_is_rejected` with tests that a multi-dim
  `EMBED_MODELS` (768/1024/1536) parses, `embed_dims` contains all three, and `dim_for` returns the
  right value and `None` for an unknown id. Drop `embed_dim` from every `Settings(...)` in the suite
  (`test_chat_flow`, `test_ingest_txt`, `test_api_smoke`, `test_chat_api`).
- `test_api_smoke.py`: `ModelsOut` no longer has `embed_dim`; assert the embeddings list exposes
  per-model `dim`. Add a test that creating a machine with a 768-dim model succeeds and reports
  `embed_dim == 768` (this is the regression the current code fails).
- New `tests/test_vectors.py`: `vector_table(768) == "chunk_vectors_768"`, and a dim not in
  `EMBED_MODELS` raises.
- Ingest test asserts rows land in `chunk_vectors_8` and that `chunks` has no `vector` column.

### Migration

Greenfield dev database, so no data migration — but the live `myragapp` currently has the old schema
with data created during the previous verification. Add `scripts/reset_dev_db.py` that drops
`chunk_vectors_%` (discovered from `information_schema`, not hardcoded) and the base tables, then let
startup recreate everything. The legacy guard in `create_schema` points at this script.

### Docs

`README.md`: remove `EMBED_DIM`, document per-model dimensions, the 2000-dimension index limit, and
the multi-model-per-machine behaviour. `.env.example`: same.

## Verification

1. `pytest -q` and `ruff check app tests` — full suite green.
2. Compile the DDL against the Postgres dialect (as done previously) and confirm `chunk_vectors_768`
   and `chunk_vectors_1024` are emitted with `vector(768)` / `vector(1024)`.
3. Live PostgreSQL + pgvector run with stubbed embedding clients that return the correct width per
   model: create one machine, upload the same document twice under `nomic-embed-text` (768) and
   `bge-m3` (1024), then assert `chunks` holds both sets, `chunk_vectors_768` and `chunk_vectors_1024`
   each hold their own rows, and embedding / hybrid / keyword retrieval return chunks for each model.
4. Confirm `by_embed_model` in the usage payload shows two rows with independent token and cost totals.

## Risks

- **FK vs. reset.** `chunk_vectors_<dim>` references `chunks`, so a naive `DROP TABLE chunks` needs
  `CASCADE`. The reset script drops the vector tables first.
- **Index ceiling.** Dimensions above 2000 cannot be HNSW-indexed; the code degrades to exact search
  with a warning instead of failing at startup.
- **Table proliferation.** One table per dimension is bounded by the number of distinct dimensions in
  `EMBED_MODELS`, which is small by design.
- **Stale `embed_dim` on machines.** Dropping `machines.embed_dim` removes the possibility of it
  disagreeing with `.env`; the API computes it, so the frontend is unaffected.

## Out of scope

- Re-embedding a document when a model's configured dimension changes (that means a new model id).
- Changing a machine's default embed model after documents are indexed (index under the new model
  instead; both sets coexist).
- Quantised or half-precision vector types.