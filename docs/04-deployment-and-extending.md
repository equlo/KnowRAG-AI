# 4. Deployment, production hardening, and extending the app

---

## 4.1 How the Docker setup fits together

```
Browser ──► :8080  frontend container (nginx)
                    ├── /            → static React build (index.html, JS, CSS)
                    └── /api/*       → proxy → backend:8000 (FastAPI container)
                                                    │
                                                    ├── /data volume (SQLite DB + embedding model)
                                                    └── HTTPS → Claude API
```

| File | Role |
|---|---|
| `backend/Dockerfile` | Python 3.12 slim image. It copies `requirements.txt` **before** the code, so dependency installs are cached between code edits |
| `frontend/Dockerfile` | A **multi-stage** build: stage 1 (Node) runs `npm ci && npm run build`; stage 2 (nginx) copies only the `dist/` output. The final image contains no Node.js and is only a few MB |
| `frontend/nginx.conf` | Serves the React app and proxies `/api/`. `proxy_buffering off` is **essential** for streaming. Without it nginx collects the whole answer before sending anything |
| `docker-compose.yml` | Wires both containers, loads `backend/.env`, persists `/data` in a named volume, and starts the frontend only after the backend's health check passes |
| `.dockerignore` files | Keep `.env`, `node_modules`, caches and local data out of the images. In particular, **secrets are never baked into an image** |

Common commands:

```bash
docker compose up --build -d      # build and start in the background
docker compose logs -f backend    # follow backend logs
docker compose down               # stop (data volume is kept)
docker compose down -v            # stop AND delete all indexed data
```

---

## 4.2 Before going to production

This project is a complete, working reference app. A public deployment needs a few more layers:

| Concern | Recommendation |
|---|---|
| **Authentication** | Every endpoint is currently open. Add login (OAuth/OIDC, or an API gateway) and scope documents per user or team, for example by adding an `owner_id` column to `documents` and filtering searches by it |
| **Secrets** | Inject `ANTHROPIC_API_KEY` from a secret manager (AWS Secrets Manager, GCP Secret Manager, Kubernetes Secrets), never from a committed file |
| **HTTPS** | Terminate TLS in front of nginx (a load balancer, Caddy, or Traefik) |
| **Rate limiting and cost control** | Limit `/api/chat` per user (e.g. with `slowapi` or at the gateway). Lower `KNOWRAG_CLAUDE_EFFORT` to `low` for cheaper, faster answers. Watch the token counts reported in every `done` event |
| **CORS** | Set `KNOWRAG_CORS_ORIGINS` to your real domain only |
| **Large uploads** | Move ingestion to a background job queue (Celery, RQ, or FastAPI `BackgroundTasks`) and report progress, so big PDFs don't hold an HTTP request open |
| **Multiple server processes** | SQLite plus an in-memory index suits one process. With several workers or replicas, move to PostgreSQL + **pgvector** or a vector DB (Qdrant, Weaviate, Chroma). Only `vector_store.py` changes, because its public methods stay the same |
| **Observability** | Log each request's `_request_id` from the Anthropic SDK, plus retrieval scores and latency, so you can debug bad answers |
| **Evaluation** | Build a small set of question → expected-answer pairs from your documents and re-run it whenever you change chunk size, `top_k`, prompts or models |

---

## 4.3 Ideas for extending KnowRAG

Each idea notes where it plugs in.

### Better retrieval
- **Hybrid search (keyword + semantic).** Combine BM25 keyword scores with vector scores, for example with Reciprocal Rank Fusion. This helps exact terms such as product codes and names. Plugs into `VectorStore.search()` (SQLite's built-in FTS5 can provide BM25).
- **Re-ranking.** Retrieve 20 chunks, then let a cross-encoder model, or a cheap Claude call, re-order them and keep the best 5. Plugs into `RAGService.answer()` between retrieval and prompting.
- **Hosted embeddings.** Add a third provider to `embeddings.py`, for example Voyage AI, which Anthropic recommends. It only needs a class with `name`, `embed_documents` and `embed_query`, plus a branch in `create_embedder()`.
- **Smarter chunking.** Split Markdown on headings, keep each chunk's section title in its text, or use token-based sizes instead of characters. Plugs into `chunker.py`.
- **Metadata filters.** Let users restrict a question to selected documents. Add `document_ids` to `ChatRequest` and a `WHERE document_id IN (...)` filter in the store.

### Better generation
- **Claude's native citations.** Instead of `[n]` markers, pass each chunk as a `document` content block with `citations: {"enabled": true}`. Claude then returns structured citations with exact quoted spans. Plugs into `prompts.py` and `llm.py`.
- **Prompt caching.** If you keep a large, fixed context, such as a long glossary in the system prompt, mark it with `cache_control` so repeated requests read it from cache at a fraction of the cost. Plugs into `llm.py`.
- **Show Claude's reasoning.** Pass `thinking={"type": "adaptive", "display": "summarized"}` and forward `thinking` stream events to the UI as a collapsible "reasoning" section. Plugs into `llm.py` and `Message.jsx`.
- **Agentic RAG.** Give Claude a `search_documents` tool and let it decide when and what to search, possibly several times per question. This suits complex multi-part questions. Plugs into `llm.py` (tool use) and `pipeline.py`.

### Better product
- **Persistent chat history:** store conversations in SQLite and add a sidebar of past chats.
- **More formats:** HTML (BeautifulSoup), PowerPoint (python-pptx), spreadsheets (openpyxl), and scanned PDFs (OCR with Tesseract or `ocrmypdf`). Plugs into `loaders.py`.
- **Feedback buttons:** 👍/👎 on answers, stored alongside the question, retrieved chunks and answer. This is the best raw material for evaluation.

---

## 4.4 Tuning cheat-sheet

| Symptom | Try |
|---|---|
| Answers miss information that is in the documents | Increase `KNOWRAG_TOP_K` (e.g. 8). Use `/api/search` to check whether the right chunk is retrieved at all |
| Retrieved chunks are relevant but cut off mid-thought | Increase `KNOWRAG_CHUNK_SIZE` (e.g. 1500) and `KNOWRAG_CHUNK_OVERLAP` |
| Retrieved chunks are long and mostly irrelevant | Decrease `KNOWRAG_CHUNK_SIZE` (e.g. 600) |
| Exact names or codes aren't found | Add hybrid keyword search (see 4.3) |
| Answers are slow or expensive | `KNOWRAG_CLAUDE_EFFORT=low`; lower `KNOWRAG_TOP_K`; set `KNOWRAG_QUERY_REWRITE=false` |
| Answers need deeper reasoning across many sources | `KNOWRAG_CLAUDE_EFFORT=high` |

Chunk settings apply only to files uploaded afterwards, so delete and re-upload existing documents to re-chunk them. After changing the **embedding model**, delete the whole data folder and re-upload everything, because old and new vectors can't be compared.
