# MyRag — Flour Factory Machine RAG Assistant

FastAPI + React SPA where a user creates a machine, uploads PDF/TXT machine data, and chats with it —
choosing the LLM and the retrieval workflow per query, and seeing what each query cost in tokens and money.

The second purpose is a **RAG evaluation bench**: every retrieval flow is a named class, every query row
records the workflow version that answered it, so flows can be compared on price and quality.

## Location

`C:\l\llamaindex\MyRag` (empty — greenfield).

## Resolved decisions

| Area | Decision |
|---|---|
| Backend | FastAPI (Python 3.11+), async |
| Frontend | React + Vite SPA in `MyRag/web`, JSON + SSE |
| RAG | LlamaIndex core; query flows as `llama_index.core.workflow.Workflow` subclasses |
| LLM | OpenAI-compatible `base_url` + token from `.env`; a list of selectable model ids |
| Embeddings | **Separate endpoint**: own `EMBED_API_BASE` / `EMBED_API_KEY`, own list of model ids with dims |
| Vector store | Hand-written `chunks` table with a `pgvector` `Vector(dim)` column; no `PGVectorStore` |
| Retrieval modes | per machine: `embedding`, `keyword`, `hybrid` — each a registered workflow class |
| DB | PostgreSQL via `DATABASE_URL`, SQLAlchemy Core, `psycopg` |
| Token capture | `BaseEventHandler` on the llama_index dispatcher, correlated by a contextvar |
| Cost | provider-exact cost if the response carries one, else `.env` per-1M price x measured tokens |
| Embed cost | estimated from characters, priced from `.env` (embedding responses carry no token counts) |
| One embed model per machine | vector column dimension is fixed; switching model requires a re-index |

### Verified against source (not assumed)

- `OpenAILikeEmbedding(model_name=..., api_key=..., api_base=...)` — embeddings can point at a different
  host and key than the LLM.
- `EmbeddingEndEvent` carries `chunks: List[str]` and `embeddings: List[List[float]]` — **no token counts**.
- `LLMChatEndEvent.response.raw["usage"]` and `LLMCompletionEndEvent.response.raw["usage"]` — the LLM token source.
- `BM25Retriever(nodes=[...], similarity_top_k=k)` builds an in-memory index from nodes; it can be persisted.
- `Workflow` + `@step` + `StopEvent` is the stable API used by the book's `ch8` examples.

## Layout

```
MyRag/
  .env.example
  requirements.txt
  README.md
  app/
    main.py                # FastAPI app, lifespan: schema init, seed workflows, pgvector check
    config.py              # pydantic-settings; LLM and EMBED endpoints, price maps
    db.py                  # engine, session, schema DDL
    models.py              # tables (below)
    schemas.py             # pydantic request/response models
    llm/
      factory.py           # build_llm(model_id) -> cached OpenAILike (LLM endpoint)
      pricing.py           # price lookup, cost resolve, embed token estimate
      instrumentation.py   # UsageEventHandler + correlation-id contextvar
      usage.py             # query context open/close, persist llm_calls, finalize query
    ingest/
      reader.py            # PDF (pypdf) / TXT -> Document; size and type guards
      chunker.py           # SentenceSplitter with chunk_size/overlap
      embeddings.py        # build embed model for (endpoint, model_id); embed batch -> vectors
      pipeline.py          # upload -> chunks -> embeddings -> chunks rows -> ingest_usage
    retrieval/
      base.py              # RetrievalFlow protocol + shared node->row hydration
      embedding.py         # cosine_distance query on chunks.vector
      keyword.py           # BM25Retriever over the machine's chunk nodes
      hybrid.py            # RRF fusion of dense cosine ranks and Postgres full-text ranks
      registry.py          # WORKFLOWS dict: name -> class
    workflows/
      rag_workflow.py      # RagWorkflow(Workflow): retrieve -> synthesize -> StopEvent
    api/
      machines.py          # POST/GET/DELETE /api/machines
      documents.py         # upload, index, list, delete
      workflows_api.py     # GET /api/workflows
      chat.py              # POST /api/machines/{id}/chat (SSE), GET messages
      usage.py             # usage totals, per-query detail, workflow comparison
  web/
    package.json
    vite.config.js         # dev proxy /api -> localhost:8000
    index.html
    src/
      main.jsx
      api.js
      App.jsx
      components/
        MachineCreate.jsx
        MachineList.jsx
        UploadPanel.jsx
        ChatPanel.jsx
        WorkflowPicker.jsx
        UsagePanel.jsx
  tests/
    test_pricing.py
    test_workflow_registry.py
    test_ingest_txt.py
    test_chat_flow.py
```

## Tables

```python
machines
  id            bigserial pk
  title         text not null
  embed_model_id text not null            # pinned per machine; mixing models is rejected
  embed_dim     int not null
  chunk_size    int not null
  chunk_overlap int not null
  embed_tokens_per_char numeric(9,6)      # calibrated on first index; NULL until then
  created_at    timestamptz default now()

documents
  id            bigserial pk
  machine_id    bigint fk -> machines on delete cascade
  filename      text not null
  content_type  text not null
  byte_size     bigint not null
  status        text not null              # uploaded|indexing|ready|failed
  error         text
  created_at    timestamptz default now()

chunks
  id            bigserial pk
  machine_id    bigint fk -> machines on delete cascade
  document_id   bigint fk -> documents on delete cascade
  chunk_index   int not null
  text          text not null
  metadata      jsonb not null default '{}'   # filename, page, etc. NOT full document text
  embed_model_id text
  embed_dim     int
  vector        vector(EMBED_DIM)            # dimension fixed per .env, see DDL note
  token_estimate int not null default 0
  unique (machine_id, document_id, chunk_index)

workflows                        # seeded; one row per registered flow class
  id            bigserial pk
  name          text unique not null          # 'embedding' | 'keyword' | 'hybrid'
  description   text not null
  class_key     text not null
  enabled       bool not null default true
  created_at    timestamptz default now()

query
  id            bigserial pk
  machine_id    bigint fk -> machines
  workflow_id   bigint fk -> workflows
  workflow_version int not null              # captured at run time (workflows.id)
  llm_model_id  text not null
  question      text not null
  answer        text
  retrieval_ms  int
  total_ms      int
  prompt_tokens int not null default 0
  completion_tokens int not null default 0
  total_tokens  int not null default 0
  cost_usd      numeric(18,8) not null default 0
  cost_source   text not null                 # provider|env|estimated|none
  retrieved_node_ids bigint[] not null default '{}'
  error         text
  created_at    timestamptz default now()

query_llm_call                   # one row per LLM call inside a query
  id            bigserial pk
  query_id      bigint fk -> query on delete cascade
  machine_id    bigint fk -> machines
  model_id      text not null
  call_index    int not null
  purpose       text not null                 # condense|synthesize
  prompt_tokens int not null
  completion_tokens int not null
  total_tokens  int not null
  cost_usd      numeric(18,8) not null
  cost_source   text not null
  usage_raw     jsonb
  created_at    timestamptz default now()

ingest_usage                     # one row per indexing run (embeddings have no token API)
  id            bigserial pk
  machine_id    bigint fk -> machines on delete cascade
  document_id   bigint fk -> documents on delete cascade
  embed_model_id text not null
  char_count    bigint not null
  token_estimate int not null
  cost_usd      numeric(18,8) not null
  cost_source   text not null                 # 'estimated' | 'none'
  created_at    timestamptz default now()
```

Indexes: `chunks(machine_id)`, `chunks(document_id)`, `query(machine_id, created_at)`,
`query(workflow_id)`, `query_llm_call(query_id)`, `ingest_usage(machine_id)`.

### DDL note on the vector column

`pgvector` requires a fixed dimension per column, and `ALTER TABLE ... TYPE vector(n)` cannot change an
existing column's dimension once rows exist. To keep one `chunks` table:

- `.env` declares a single `EMBED_DIM` (default 1536) that **every** entry in `EMBED_MODELS` must match.
  `config.py` validates this at startup and refuses to boot with a message naming the offending model.
- `chunks.vector` is created as `vector(EMBED_DIM)`.
- `machines.embed_dim` must equal `EMBED_DIM`; the create-machine request is rejected with 409 otherwise.
- A machine whose `.env` model later changes dimension is a re-index into a new machine (or a manual
  `ALTER TABLE` migration), not a silent mixed-dimension table.

This is a deliberate constraint: the alternative is a table per dimension, which multiplies the schema
for a benefit v1 does not need.

## .env.example

```
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/myragapp

# LLM endpoint (OpenAI-compatible). May equal the embedding endpoint or not.
API_BASE=https://openrouter.ai/api/v1
API_KEY=sk-...
LLM_MODELS=openai/gpt-4o-mini=0.15,0.60,128000; openai/gpt-4o=2.50,10.00,128000
DEFAULT_LLM_MODEL=openai/gpt-4o-mini

# Embedding endpoint: independently configurable host and key.
# Every EMBED_MODELS entry MUST have dim == EMBED_DIM, or startup fails with a message naming it.
EMBED_API_BASE=https://api.openai.com/v1
EMBED_API_KEY=sk-...
EMBED_MODELS=openai/text-embedding-3-small=1536,0.02; openai/text-embedding-ada-002=1536,0.0001
DEFAULT_EMBED_MODEL=openai/text-embedding-3-small
EMBED_DIM=1536

CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K=4
MAX_UPLOAD_MB=25
MAX_CHUNKS_PER_DOC=2000
```

Formats: `LLM_MODELS` = `model_id=input_per_1m,output_per_1m,context_window` semicolon-separated;
`EMBED_MODELS` = `model_id=dim,input_per_1m`. Embeddings have no output price.

## Retrieval workflows

`app/retrieval/registry.py` maps a name to a class implementing:

```python
class RetrievalFlow(Protocol):
    name: str
    description: str
    def retrieve(self, *, machine_id: int, question: str, top_k: int) -> list[RetrievedChunk]: ...
```

| name | implementation |
|---|---|
| `embedding` | `SELECT ... ORDER BY vector <=> query_vec LIMIT top_k` filtered by `machine_id`, HNSW index on `chunks.vector` with `vector_cosine_ops` |
| `keyword` | `BM25Retriever` built from that machine's chunk `TextNode`s (text + metadata); built per request and cached per `(machine_id, top_k)` for the process lifetime |
| `hybrid` | Postgres full text: generated `tsvector` column + GIN index, `ts_rank` ordering; fused with the dense ranking by reciprocal rank fusion |

`RetrievedChunk` = `(chunk_id, document_id, text, metadata, score)`. The workflow records
`retrieved_node_ids` on the `query` row so quality is comparable across flows.

Adding a v2 of any flow = new class + new registry entry + new `workflows` row. `workflows` rows are
never mutated in place except `description`/`enabled`; `query.workflow_version` keeps history comparable.

**Keyword retrieval stays inside Postgres**, not in a separate `PGVectorStore` table, so all three flows
read the same `chunks` rows.

## Cost accounting

`app/llm/pricing.py`:

```python
def price_for_llm(model_id) -> LlmPrice | None
def resolve_llm_cost(model_id, prompt_tokens, completion_tokens, provider_cost) -> (usd, source)
def estimate_embed_tokens(char_count) -> int          # calibrated ratio, see below
def resolve_embed_cost(model_id, char_count) -> (usd, source)
```

- **LLM tokens**: `UsageEventHandler` captures `LLMChatEndEvent` / `LLMCompletionEndEvent`, reads
  `response.raw["usage"]`, and writes a `query_llm_call` row tagged with the active correlation id.
  Query-level totals are the sum of its calls.
- **Provider cost**: a thin `CostAwareOpenAILike(OpenAILike)` lifts `cost`/`total_cost` out of
  `response.raw` into `_myrag_cost` so the handler sees it without re-parsing. If present it wins;
  else `.env` price x measured tokens; else `cost_source='none'` with `0`.
- **Embed tokens**: embeddings return no token counts (verified: `EmbeddingEndEvent` carries only chunk
  text and vectors). Estimate with `tiktoken` when installed, otherwise `len(text) / 4`, store the
  per-machine ratio on the `machines` row after the first index, and mark every estimate
  `cost_source='estimated'` so it is never mistaken for a measured figure.

## API

```
POST   /api/machines                     {title, embed_model_id?, chunk_size?, chunk_overlap?}
GET    /api/machines                     -> machines with document count, chunk count, total cost
GET    /api/machines/{id}
DELETE /api/machines/{id}

POST   /api/machines/{id}/documents      multipart file (pdf|txt)
POST   /api/documents/{doc_id}/index     -> 202; background task
GET    /api/machines/{id}/documents      -> status, chunk count, ingest cost
DELETE /api/documents/{doc_id}

GET    /api/workflows                    -> seeded rows with enabled state
GET    /api/models                       -> {llms:[...], embeddings:[...]} from .env

POST   /api/machines/{id}/chat           {question, workflow_id, llm_model_id, top_k?, stream?} -> SSE or JSON
GET    /api/machines/{id}/messages       -> recent queries with cost, workflow, model
GET    /api/queries/{query_id}           -> detail incl. llm_calls
GET    /api/machines/{id}/usage          -> totals grouped by model and by workflow
GET    /api/usage/compare?machine_id=    -> per-workflow: runs, avg cost, avg tokens, avg latency
```

SSE events: `token`, `retrieval` (chunk ids + scores), `usage` (per call), `done` (query summary), `error`.

## Chat execution

1. Resolve machine, workflow row, LLM id; insert the `query` row with `answer` null; open the correlation id.
2. `RagWorkflow` steps: `retrieve` (flow.retrieve) -> `synthesize` (LlamaIndex response synthesizer with the
   selected LLM) -> `StopEvent`.
3. `Settings.llm` is set to the selected model for the duration of the run (per-request), not globally at import.
4. Usage handler writes `query_llm_call` rows as calls complete; `close_query_context` sums them onto `query`
   and returns the totals for the `done` event.
5. LLM errors are written to `query.error` and returned as HTTP 502 with the provider message.

## Ingestion

1. Upload: extension + content-type check, size cap, store at `data/uploads/<machine_id>/<uuid><ext>`.
2. `POST /documents/{id}/index`: status `indexing` -> background task:
   - `SimpleDirectoryReader(input_files=[path])` (PDF via `pypdf`, TXT via `llama-index-readers-file`).
   - `SentenceSplitter(chunk_size, chunk_overlap)` -> nodes; strip the full document text from node metadata
     so the DB does not duplicate the PDF (only `filename`, `page`, `machine_id`, `document_id`).
   - Embed in batches via the **embedding endpoint** model; insert `chunks` rows with vectors.
   - Write one `ingest_usage` row.
   - status `ready`, or `failed` with `error`. Re-index deletes the document's existing chunks first.

## Validation

Dependencies (`requirements.txt`): `fastapi`, `uvicorn[standard]`, `python-multipart`, `sse-starlette`,
`sqlalchemy`, `psycopg[binary]`, `pgvector`, `pydantic-settings`, `llama-index-core`,
`llama-index-llms-openai-like`, `llama-index-embeddings-openai-like`, `llama-index-readers-file`,
`pypdf`, `tiktoken`, `pytest`, `pytest-asyncio`, `httpx`.

- `tests/test_pricing.py` — LLM price lookup, missing price -> `none`; provider cost beats `.env`;
  embed estimate monotonic in length; `tiktoken` absent -> falls back to `len/4` without raising.
- `tests/test_workflow_registry.py` — every seeded `workflows` row resolves to a registered class;
  unknown id -> 404.
- `tests/test_ingest_txt.py` — TXT upload -> chunks + vectors, `ingest_usage` row, status `ready`,
  document text not duplicated in `chunks.metadata`.
- `tests/test_chat_flow.py` — with `MockLLM`, chat produces one `query` row with `workflow_version`,
  one `query_llm_call` per LLM call, and cost > 0 when a price is configured.
- Manual smoke: create machine -> upload `ch14/files/nist.ai.100-1.pdf` from the book repo -> index ->
  chat "what is the rated capacity?" -> `/usage` shows tokens and cost -> `/usage/compare` shows the flow.
- Sample TXT: `Building-Data-Driven-Applications-with-LlamaIndex-Second-Edition/ch9/files/sample_document1.txt`.

## Implementation order

1. `requirements.txt`, `.env.example`, `config.py`, `db.py`, `models.py`.
2. `main.py` lifespan + `machines.py` + `workflows_api.py`; minimal React create/list.
3. `llm/factory.py`, `llm/pricing.py`, `llm/instrumentation.py`, `llm/usage.py`; `test_pricing.py` green.
4. `ingest/*` (TXT first, then PDF); `documents.py` + `UploadPanel`.
5. `retrieval/embedding.py` + `workflows/rag_workflow.py` + non-streaming `chat.py`; first real chat.
6. `retrieval/keyword.py`, `retrieval/hybrid.py`; `WorkflowPicker`, SSE streaming, `UsagePanel`.
7. `usage.py` compare endpoint; `README.md`.

## Risks

- **Embedding and LLM endpoints differ** -> two clients, two keys, two `.env` blocks. Handled by
  `EMBED_API_BASE`/`EMBED_API_KEY` being independent of `API_BASE`/`API_KEY`; `GET /api/models` shows both.
- **Embed dimension mismatch** -> rejected at machine create with 409 and a message naming the configured
  `EMBED_DIM`; switching a machine's embed model requires a re-index.
- **Provider returns no `usage`** -> tokens 0, cost 0. Mitigated by an optional character-based estimate
  flagged `estimated`, never silently treated as measured.
- **Provider cost in an unexpected shape** -> logged once per model at debug level; a shape adapter is added
  when observed.
- **Large PDF** -> background task + status polling; `MAX_CHUNKS_PER_DOC` caps the work and the document row
  records the truncation.
- **Embedding cost is always an estimate** -> `cost_source='estimated'` on every `ingest_usage` row, so the
  UI can label it distinctly from measured LLM cost.

## Out of scope (v1)

- Authentication / multi-user. The app binds localhost and has no auth; add before any non-local exposure.
- Token streaming before the synthesis step (non-stream chat is fine for the first working version).
- Alembic migrations (schema init on startup; `alembic.ini` not added in v1).
- Rerankers, evaluation metrics beyond cost + latency + retrieved chunk ids.
- File preview in the UI, batch multi-file upload.
- Per-user API keys or per-machine cost budgets.