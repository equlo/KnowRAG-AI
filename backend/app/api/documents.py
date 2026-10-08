"""
documents.py - HTTP endpoints for managing the knowledge base.

    POST   /api/documents        upload one file  -> it is indexed immediately
    GET    /api/documents        list indexed files
    DELETE /api/documents/{id}   remove a file and its chunks
"""

# `asdict` converts our StoredDocument dataclass into a dict.
from dataclasses import asdict

# FastAPI building blocks:
#   APIRouter     - a group of related routes,
#   Depends       - dependency injection,
#   File          - marks a parameter as an uploaded file,
#   HTTPException - return an HTTP error,
#   UploadFile    - the uploaded file object,
#   status        - named HTTP status codes.
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

# Dependency that provides the RAG service.
from app.api.deps import get_rag

# The error raised for unreadable/unsupported files.
from app.rag.loaders import UnsupportedFileError

# The service type (for the type hints).
from app.rag.pipeline import RAGService

# The response shape.
from app.schemas import DocumentOut

# All routes in this file are grouped under the "documents" tag in /docs.
router = APIRouter(tags=["documents"])


# `def` (not `async def`): parsing and embedding are CPU-heavy and blocking, so
# FastAPI automatically runs this function in a worker thread.
@router.post("/documents", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
def upload_document(file: UploadFile = File(...), rag: RAGService = Depends(get_rag)) -> DocumentOut:
    """Upload and index a single PDF, DOCX, TXT or MD file."""
    # Convert the size limit from megabytes to bytes.
    max_bytes = rag.settings.max_upload_mb * 1024 * 1024
    # Read at most one byte more than allowed - enough to detect oversize files
    # without loading a gigantic upload fully into memory.
    data = file.file.read(max_bytes + 1)
    # Reject files over the limit with "413 Content Too Large" (written as a number
    # because the constant's name differs between Starlette versions).
    if len(data) > max_bytes:
        raise HTTPException(413, f"File exceeds {rag.settings.max_upload_mb} MB limit.")
    # Run the ingestion pipeline (load -> chunk -> embed -> store).
    try:
        document = rag.ingest(file.filename or "untitled.txt", data)
    # Unsupported/empty/corrupt file -> "400 Bad Request" with the reason.
    except UnsupportedFileError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    # Convert the dataclass to the response model.
    return DocumentOut(**asdict(document))


@router.get("/documents", response_model=list[DocumentOut])
def list_documents(rag: RAGService = Depends(get_rag)) -> list[DocumentOut]:
    """List every indexed document, newest first."""
    # Convert each stored document to the response model.
    return [DocumentOut(**asdict(d)) for d in rag.list_documents()]


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, rag: RAGService = Depends(get_rag)) -> None:
    """Delete a document and all of its chunks."""
    # Unknown ID -> "404 Not Found".
    if not rag.delete_document(document_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")
    # 204 means success with an empty body, so nothing is returned.
