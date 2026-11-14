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

## Deploy on Windows

A production run is one process: FastAPI serves the API and the built SPA (`web/dist`) from the same
port. Build the frontend once, then run Uvicorn without `--reload`.

### 1. PostgreSQL with pgvector

Easiest on Windows is Docker Desktop:

```powershell
docker run -d --name myrag-pg `
  -e POSTGRES_PASSWORD=postgres `
  -e POSTGRES_DB=myragapp `
  -p 5432:5432 `
  -v myragpg:/var/lib/postgresql/data `
  pgvector/pgvector:pg16
```

Then allow the extension (once, as a superuser):

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

With a native PostgreSQL install, install pgvector for your server version and put its `dll/`
files on `PATH` before starting the service; the app checks the extension at startup and refuses to
run without it.

### 2. Build the frontend

```powershell
cd C:\l\llamaindex\MyRag\web
npm install
npm run build
```

This produces `web/dist`, which `app/main.py` mounts automatically when present. Rebuild after any
UI change.

### 3. Configure and start

Edit `.env` for the server (not localhost-only values unless it is local):

- `DATABASE_URL` — reachable Postgres URL.
- `AUTH_USERNAME` / `AUTH_PASSWORD` — required unless you are exposing it on a trusted network.
- `API_BASE`, `EMBED_API_BASE` — the LLM/embedding hosts. If Ollama runs on another machine, use its
  LAN address and set `OLLAMA_ORIGINS` there.
- `HOST=0.0.0.0` to accept connections from other machines.

Run it:

```powershell
cd C:\l\llamaindex\MyRag
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://SERVER-IP:8000. Allow the port through Windows Firewall:

```powershell
New-NetFirewallRule -DisplayName "MyRag 8000" -Direction Inbound `
  -Protocol TCP -LocalPort 8000 -Action Allow
```

### 4. Run it as a Windows service (optional)

[NSSM](https://nssm.cc/) wraps any executable as a service:

```powershell
nssm install MyRag "C:\l\llamaindex\MyRag\.venv\Scripts\python.exe" `
  "-m uvicorn app.main:app --host 0.0.0.0 --port 8000"
nssm set MyRag AppDirectory "C:\l\llamaindex\MyRag"
nssm set MyRag Start SERVICE_AUTO_START
nssm start MyRag
```

Uvicorn workers are not supported on Windows, so run a single process. If you need more throughput,
put IIS (with the URL Rewrite + ARR modules) or nginx for Windows in front and forward to several
Uvicorn instances on different ports.

## Put nginx for Windows in front of Uvicorn (recommended)

nginx handles Server-Sent Events correctly (`proxy_buffering off`) and terminates TLS, so Uvicorn
stays on loopback. `deploy/nginx.conf` is a ready-made config for
`https://myrag.myorganization.ir`.

### 1. Install nginx

Download the stable Windows build from nginx.org, unzip it to `C:\nginx` (keep `nginx.exe` at
`C:\nginx\nginx.exe`), and put certificates in `C:\nginx\certs\`.

### 2. DNS and firewall

- Create an `A` record: `myrag.myorganization.ir` → the server's public IP (or the internal IP if it
  is only reachable inside the organization).
- Open inbound TCP 80 and 443:

```powershell
New-NetFirewallRule -DisplayName "MyRag HTTP" -Direction Inbound -Protocol TCP -LocalPort 80 -Action Allow
New-NetFirewallRule -DisplayName "MyRag HTTPS" -Direction Inbound -Protocol TCP -LocalPort 443 -Action Allow
```

Port 80 must be reachable from the internet only if you use Let's Encrypt HTTP-01 validation.

### 3. Certificate

If you already have a PFX (from the organization CA or a commercial CA), convert it to PEM with
OpenSSL (Git for Windows ships one):

```powershell
openssl pkcs12 -in myrag.pfx -clcerts -nokeys -out C:\nginx\certs\myrag.myorganization.ir.crt
openssl pkcs12 -in myrag.pfx -nocerts -nodes  -out C:\nginx\certs\myrag.myorganization.ir.key
```

For Let's Encrypt, use [win-acme](https://www.win-acme.com) and write PEM files straight to nginx:

```powershell
.\wacs.exe --host myrag.myorganization.ir --store pemfiles `
  --pemfilespath C:\nginx\certs --installation none --accepttos --email admin@myorganization.ir
```

Then renew with a reload hook (`--script "C:\nginx\nginx.exe -p C:\nginx -s reload"`).

### 4. Configure and test

```powershell
Copy-Item deploy\nginx.conf C:\nginx\conf\nginx.conf
C:\nginx\nginx.exe -p C:\nginx -t
```

Adjust `server_name` and the certificate paths if your hostname or file names differ.

### 5. Run Uvicorn on loopback

```powershell
cd C:\l\llamaindex\MyRag
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Do **not** open port 8000 in the firewall — nginx is the only public entry point.

### 6. Run both as Windows services

nginx for Windows has no service mode; use [NSSM](https://nssm.cc/) for both:

```powershell
nssm install MyRag "C:\l\llamaindex\MyRag\.venv\Scripts\python.exe" `
  "-m uvicorn app.main:app --host 127.0.0.1 --port 8000"
nssm set MyRag AppDirectory "C:\l\llamaindex\MyRag"
nssm set MyRag Start SERVICE_AUTO_START
nssm start MyRag

nssm install MyRagNginx "C:\nginx\nginx.exe"
nssm set MyRagNginx AppDirectory "C:\nginx"
nssm set MyRagNginx Start SERVICE_AUTO_START
nssm start MyRagNginx
```

After a config change: `C:\nginx\nginx.exe -p C:\nginx -s reload`.

### 7. Verify streaming

```powershell
curl.exe -N -X POST https://myrag.myorganization.ir/api/machines/2/chat/stream `
  -H "Content-Type: application/json" `
  -d '{"question":"What is this document about?","workflow_id":1,"llm_model_id":"deepseek-r1:14b"}'
```

`data:` lines must appear one by one. If they arrive all at once, check that `proxy_buffering off;` is
still present in the `location /` block.

### 8. More throughput

Uvicorn workers do not work on Windows. Run two or three instances on ports 8000-8002 (one NSSM
service each) and list them in the `upstream` block in `deploy/nginx.conf`. All state lives in
Postgres, so any instance can serve any request.

## Put IIS in front of Uvicorn (optional)

Use IIS as the public site and keep Uvicorn on loopback. `deploy/web.config` is a ready-made
reverse-proxy config: it forwards every request to `http://127.0.0.1:8000` and disables dynamic
compression so streamed answers are not buffered.

1. Enable **Internet Information Services** + **IIS Management Console** in Windows Features, then
   install **URL Rewrite 2.1** and **Application Request Routing (ARR) 3.0** (x64) from iis.net.
   ARR requires URL Rewrite.

2. Create the site folder and copy the config:

```powershell
New-Item -ItemType Directory C:\inetpub\myrag
Copy-Item deploy\web.config C:\inetpub\myrag\web.config
```

3. Create the site (no managed code, ARR proxies to Uvicorn):

```powershell
Import-Module WebAdministration
New-WebAppPool -Name MyRag
Set-ItemProperty IIS:\AppPools\MyRag managedRuntimeVersion ""
New-WebSite -Name MyRag -PhysicalPath C:\inetpub\myrag -Port 80 -ApplicationPool MyRag
```

Stop the Default Web Site first if it already owns port 80, or use another port/host header.

4. Bind Uvicorn to loopback only and open just 80/443 in the firewall:

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

5. Raise the ARR timeout: reasoning answers can take minutes. In IIS Manager select the server,
   open **Server Proxy Settings** in the ARR pane, and set **Time-out (seconds)** to `300`.

6. For HTTPS, add an `https` binding with your certificate. IIS terminates TLS and still proxies to
   plain HTTP on 127.0.0.1:8000:

```powershell
New-WebBinding -Name MyRag -Protocol https -Port 443 -HostHeader myrag.local
$cert = Get-ChildItem Cert:\LocalMachine\My | Where-Object Subject -eq "CN=myrag.local"
$cert | New-Item -Path "IIS:\SslBindings!443!myrag.local"
```

7. Verify that tokens actually stream through IIS (they must appear one by one, not in one burst):

```powershell
curl.exe -N -X POST http://localhost/api/machines/2/chat/stream `
  -H "Content-Type: application/json" `
  -d '{"question":"What is this document about?","workflow_id":1,"llm_model_id":"deepseek-r1:14b"}'
```

If the `data:` lines only arrive at the end, something is still buffering: confirm
`doDynamicCompression="false"` in `web.config` and that ARR is installed. nginx for Windows
(`proxy_buffering off;`) is the fallback.

8. To use several cores, run Uvicorn on ports 8000-8002 and point an ARR server farm at them,
   rewriting to `http://myragfarm/{R:0}`. All state is in Postgres, so any instance can serve any
   request.

The app's own login (`AUTH_USERNAME` / `AUTH_PASSWORD`) still applies — IIS only forwards requests.
If you switch to IIS/Windows authentication instead, clear those two `.env` values.

### 5. Data to back up

- The Postgres database (`pg_dump`).
- `data/uploads/` — the original files; re-indexing needs them.

The `.env` file is git-ignored and never part of the repository.

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