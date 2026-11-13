# MyRag — Flour Factory Machine RAG Assistant

FastAPI + LlamaIndex app where you create a machine, upload its PDF/TXT documentation, and chat
with it. Each query can use a different LLM and a different retrieval flow, and every query's token
count and cost is recorded so flows can be compared on price and quality.

## What it does

1. **Create a machine** — give it a title (e.g. "Roller Mill #2") and an embedding model.
2. **Upload machine data** — PDF or TXT. The file is read, split into chunks, embedded, and stored
   in Postgres.
3. **Chat** — pick an LLM and a retrieval flow, ask a question, get an answer grounded in the
   uploaded documents.
4. **Compare** — the usage panel shows per-flow runs, average cost, average tokens, and latency.

## Requirements

- Python 3.11+
- PostgreSQL with the **pgvector** extension (the app refuses to start without it, naming the problem)
- Node.js 18+ (for the React dev server)

## Setup

### 1. Backend

```powershell
cd C:\l\llamaindex\MyRag
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
Copy-Item .env.example .env
```

Then edit `.env`:

- `DATABASE_URL` — must point at a database where `CREATE EXTENSION vector` is allowed.
- `API_BASE` / `API_KEY` — the OpenAI-compatible LLM endpoint (OpenRouter works as-is).
- `EMBED_API_BASE` / `EMBED_API_KEY` — the embedding endpoint. This can be a **different host and key**
  from the LLM endpoint.
- `LLM_MODELS` — `model_id=input_per_1m,output_per_1m,context_window`, semicolon-separated.
- `EMBED_MODELS` — `model_id=dim,input_per_1m`, semicolon-separated. **Every entry must have
  `dim == EMBED_DIM`**; the app refuses to start otherwise, because the `chunks.vector` column has a
  single fixed dimension.

Create the database if it does not exist yet:

```sql
CREATE DATABASE myragapp;
```

### 2. Run the API

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Interactive docs: http://127.0.0.1:8000/docs
Health check: http://127.0.0.1:8000/api/health

The schema, indexes, and the three seeded workflows are created automatically on startup.

### 3. Run the frontend

```powershell
cd web
npm install
npm run dev
```

Open http://localhost:5173. The Vite dev server proxies `/api` to the backend on port 8000.

## Retrieval flows

Each flow is a class in `app/retrieval/`, registered in `registry.py`, and seeded as a row in the
`workflows` table. `query.workflow_version` records which row answered each question, so runs stay
comparable after you add a v2.

| Flow | How it retrieves |
|---|---|
| `embedding` | Dense cosine similarity over `chunks.vector` (pgvector HNSW index), filtered to the machine. |
| `keyword` | BM25 over the machine's chunk text, built from the stored chunks and cached per process. |
| `hybrid` | Postgres full-text (`tsvector` + `ts_rank`) fused with dense results using reciprocal rank fusion. |

To add a version of a flow: write the class, add it to `WORKFLOWS` in `registry.py`, and restart — the
seeder inserts a new `workflows` row. Existing rows are never edited.

## Cost accounting

Prices live in `.env`, per model. For each LLM call the app records prompt/completion/total tokens and
computes cost:

1. If the provider returns an exact cost on the response, use it (`cost_source = provider`).
2. Otherwise use `.env` price × measured tokens (`cost_source = env`).
3. If the model has no configured price, the row is still stored with `cost_source = none` so the gap
   is visible rather than silent.

Token counts come from LlamaIndex's instrumentation events, which capture **every** LLM call in a
query — including ones made internally by the retrieval/synthesis steps, not just your explicit call.

Embedding responses carry no token counts, so embedding cost is estimated from character count (via
`tiktoken` when available) and always marked `cost_source = estimated`.

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/machines` | Create a machine. |
| `GET` | `/api/machines` | List machines with document, chunk, query counts and total cost. |
| `GET` | `/api/machines/{id}` | Machine detail. |
| `DELETE` | `/api/machines/{id}` | Delete a machine and everything under it. |
| `POST` | `/api/machines/{id}/documents` | Upload a PDF/TXT (multipart `file`). |
| `POST` | `/api/documents/{id}/index` | Index the document in the background. |
| `GET` | `/api/machines/{id}/documents` | Document status, chunk count, embedding cost. |
| `DELETE` | `/api/documents/{id}` | Delete a document. |
| `GET` | `/api/workflows` | Seeded retrieval flows. |
| `GET` | `/api/models` | Configured LLMs and embeddings, plus both endpoint URLs. |
| `POST` | `/api/machines/{id}/chat` | Ask a question with a chosen flow and LLM. |
| `GET` | `/api/machines/{id}/messages` | Recent queries with cost and flow. |
| `GET` | `/api/queries/{id}` | One query with its per-call LLM usage. |
| `GET` | `/api/machines/{id}/usage` | Token and cost totals, grouped by model and flow. |
| `GET` | `/api/usage/compare?machine_id=` | Per-flow comparison: runs, avg cost, avg tokens, avg latency. |

## Tests

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff check app tests
```

The suite runs against SQLite (no PostgreSQL needed) and stubs the LLM and embedding clients, so no
network calls are made. Because SQLite has no pgvector, the `keyword` flow is the one exercised
end-to-end there.

The `embedding`, `keyword`, and `hybrid` flows were verified separately against a live PostgreSQL with
pgvector: schema creation, JSONB metadata, the 1536-dimension vector column, the HNSW index, ingestion,
chat, and per-flow cost comparison all returned correct results. That check used stubbed LLM and
embedding clients, so a run with your real API keys and provider is still worth doing once.

## Layout

```
app/
  config.py        .env parsing, price tables, EMBED_DIM validation
  db.py            engine, session, schema DDL
  schemas.py       request/response models
  llm/
    factory.py     cached LLM construction, provider-cost extraction
    pricing.py     price lookup, cost resolution, token estimation
    instrumentation.py  dispatcher wiring
    usage.py       usage handler + correlation id
    query.py       one query: insert -> run workflow -> persist usage
  ingest/
    reader.py      PDF/TXT loading, chunking, upload validation
    embeddings.py  embedding client (separate endpoint) + batch embed
    pipeline.py    document -> chunks -> vectors -> rows
  retrieval/
    base.py        RetrievedChunk, flow protocol, RRF
    embedding.py   dense flow
    keyword.py     BM25 flow
    hybrid.py      full-text + dense fusion
    registry.py    flow registry
  workflows/
    rag_workflow.py  retrieve -> synthesize
  api/             FastAPI routers
web/               React + Vite SPA
tests/             pytest suite
```

## Notes and limits

- **No authentication.** The app binds to localhost and has no auth or multi-user support. Add
  authentication before exposing it beyond localhost.
- **One embedding model per machine.** The `chunks.vector` column has one fixed dimension, so a
  machine is indexed with a single embedding model. Switching models means re-indexing the documents.
- **Embedding cost is an estimate**, not a measured value, because embedding APIs do not return token
  counts. It is labelled `estimated` everywhere it appears.
- **Token streaming is not implemented**; chat returns the full answer in one response.
- **Indexing is a background task.** Poll `GET /api/machines/{id}/documents` to see status move from
  `uploaded` to `indexing` to `ready`/`failed`.
- **NLTK data is not required.** Chunking uses `TokenTextSplitter` rather than `SentenceSplitter`,
  because the installed NLTK copy can reject its own stopword files when the virtualenv is built with
  hardlinked packages (uv/poetry).