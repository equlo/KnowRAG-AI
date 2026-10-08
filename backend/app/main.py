"""
main.py - Creates the FastAPI application: the entry point of the backend.

Run it with:
    uvicorn app.main:app --reload --port 8000

then open http://localhost:8000/docs for interactive API documentation.
"""

# Configure Python's logging so our log messages appear in the console.
import logging

# `asynccontextmanager` turns a generator into an async "with"-style context manager.
from contextlib import asynccontextmanager

# FastAPI is the web framework; Depends wires dependencies into routes.
from fastapi import Depends, FastAPI

# CORS middleware lets the React app (a different origin in development) call this API.
from fastapi.middleware.cors import CORSMiddleware

# Our routers (groups of endpoints).
from app.api import chat, documents, search

# Dependency that provides the RAG service.
from app.api.deps import get_rag

# Settings loaded from the environment / .env file.
from app.config import Settings

# Embedding provider interface + factory.
from app.rag.embeddings import Embedder, create_embedder

# LLM interface + the real Claude implementation.
from app.rag.llm import LLM, ClaudeLLM

# The service that ties everything together.
from app.rag.pipeline import RAGService

# The on-disk vector store.
from app.rag.vector_store import VectorStore

# The health response shape.
from app.schemas import HealthOut

# Show INFO-level messages and above, with a timestamp and the logger name.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app(settings: Settings | None = None, embedder: Embedder | None = None, llm: LLM | None = None) -> FastAPI:
    """Build the application. Tests pass fakes for `embedder` and `llm`."""
    # Load settings from the environment unless the caller supplied them.
    settings = settings or Settings()

    # "Lifespan": code before `yield` runs at startup, code after it at shutdown.
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Create the embedder (the FastEmbed model loads/downloads here, once).
        active_embedder = embedder or create_embedder(settings)
        # Open the vector store, telling it which embedder its vectors come from.
        store = VectorStore(settings.db_path, active_embedder.name)
        # Build the service and keep it on app.state for the routes to use.
        app.state.rag = RAGService(settings, active_embedder, store, llm or ClaudeLLM(settings))
        # The server now handles requests until it is asked to stop.
        yield
        # Shutdown: close the database cleanly.
        store.close()

    # Create the FastAPI app; title/version appear in the /docs page.
    app = FastAPI(title="KnowRAG-AI", version="1.0.0", lifespan=lifespan)

    # Allow the configured browser origins to call the API from JavaScript.
    app.add_middleware(
        CORSMiddleware,
        # Which websites may call us (e.g. the Vite dev server on port 5173).
        allow_origins=settings.cors_origin_list,
        # Allow every HTTP method (GET, POST, DELETE...).
        allow_methods=["*"],
        # Allow every request header (e.g. Content-Type).
        allow_headers=["*"],
    )

    # Mount each router under the /api prefix, e.g. /api/documents.
    app.include_router(documents.router, prefix="/api")
    app.include_router(search.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")

    # A simple health check used by Docker and by the UI's status badge.
    @app.get("/api/health", response_model=HealthOut, tags=["health"])
    def health(rag: RAGService = Depends(get_rag)) -> HealthOut:
        # Report what the server is running with.
        return HealthOut(
            status="ok",
            model=settings.claude_model,
            embedding=rag.embedder.name,
            documents=rag.store.count_documents(),
        )

    # Return the fully configured application.
    return app


# The module-level `app` object that `uvicorn app.main:app` looks for.
app = create_app()
