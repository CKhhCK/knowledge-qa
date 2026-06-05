"""
Document management API routes.

Endpoints for uploading, listing, searching, and managing
knowledge documents in the RAG pipeline.

POST   /documents/upload     — Upload a document file
POST   /documents/text       — Ingest raw text
GET    /documents            — List all documents
GET    /documents/{id}       — Get document info
DELETE /documents/{id}       — Delete a document
POST   /documents/search     — Semantic search over knowledge base
GET    /documents/stats      — Knowledge base statistics
DELETE /documents/clear      — Clear all documents (danger!)
"""

import os
import tempfile
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from app.core.agent import AdvancedQAAgent
from app.api.dependencies import get_qa_agent
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


class TextIngestRequest(BaseModel):
    """Request to ingest raw text as a knowledge document."""
    document_id: str = Field(..., description="Unique document identifier")
    title: Optional[str] = Field(default=None, description="Document title")
    text: str = Field(..., min_length=1, max_length=100000, description="Text content")
    chunk_strategy: str = Field(default="recursive", description="Chunking strategy: fixed, recursive, markdown")


class SearchRequest(BaseModel):
    """Semantic search request."""
    query: str = Field(..., min_length=1, description="Search query")
    limit: int = Field(default=5, ge=1, le=20, description="Max results")


class IngestResponse(BaseModel):
    """Document ingestion result."""
    success: bool
    message: str
    document_id: str = ""
    file_name: str = ""
    file_size_bytes: int = 0
    ingestion_time_ms: int = 0
    char_count: int = 0


@router.post("/upload", response_model=IngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(default=None),
    chunk_strategy: str = Form(default="recursive"),
    agent: AdvancedQAAgent = Depends(get_qa_agent),
):
    """
    Upload and ingest a document into the knowledge base.

    Supported formats: PDF, DOCX, XLSX, PPTX, TXT, MD, CSV, JSON,
    XML, HTML, images (OCR), audio (transcription), code files.

    The document is:
    1. Saved to a temporary location
    2. Converted to Markdown (via MarkItDown)
    3. Smart-chunked with heading awareness
    4. Embedded and indexed into the vector store
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Check if embedding model is ready
    from app.knowledge.embedding import _model_ready, _model_loading, _embedding_crashed
    if _embedding_crashed:
        raise HTTPException(status_code=503, detail="嵌入模型异常，请重启后端服务")
    if _model_loading and not _model_ready.is_set():
        raise HTTPException(status_code=503,
            detail="嵌入模型正在加载中（约需20秒），请稍后再上传")

    logger.info(f"[UPLOAD] Start | file={file.filename} | strategy={chunk_strategy} | size={file.size or 'unknown'}")

    # Save uploaded file to temp location
    suffix = os.path.splitext(file.filename)[1] or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    file_size = os.path.getsize(tmp_path)
    logger.info(f"[UPLOAD] Saved to temp | path={tmp_path} | size={file_size} bytes")

    try:
        logger.info(f"[UPLOAD] Starting ingestion...")
        result = agent.ingest_document(
            file_path=tmp_path,
            document_id=document_id or file.filename,
            chunk_strategy=chunk_strategy,
        )
        logger.info(f"[UPLOAD] Result: {result}")
        return result
    except Exception as e:
        logger.error(f"[UPLOAD] Failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        try:
            os.unlink(tmp_path)
            logger.info(f"[UPLOAD] Temp file cleaned: {tmp_path}")
        except Exception:
            pass


@router.post("/text", response_model=IngestResponse)
async def ingest_text(
    req: TextIngestRequest,
    agent: AdvancedQAAgent = Depends(get_qa_agent),
):
    """
    Ingest raw text as a knowledge document.

    The text is chunked, embedded, and indexed for semantic retrieval.
    Use this for programmatic knowledge insertion or pasting content.
    """
    try:
        result = agent.ingest_text(
            text=req.text,
            document_id=req.document_id,
            title=req.title,
            chunk_strategy=req.chunk_strategy,
        )
        return result
    except Exception as e:
        logger.error(f"Text ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Text ingestion failed: {str(e)}")


@router.get("")
async def list_documents(
    agent: AdvancedQAAgent = Depends(get_qa_agent),
):
    """List all ingested knowledge documents."""
    docs = agent.list_documents()
    return {
        "documents": docs,
        "count": len(docs),
    }


@router.get("/{document_id}")
async def get_document(
    document_id: str,
    agent: AdvancedQAAgent = Depends(get_qa_agent),
):
    """Get information about a specific document."""
    doc = agent.knowledge.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")
    return doc


@router.get("/{document_id}/chunks")
async def get_document_chunks(
    document_id: str,
    agent: AdvancedQAAgent = Depends(get_qa_agent),
):
    """Get all text chunks for a document."""
    chunks = agent.knowledge.get_chunks(document_id)
    if chunks is None:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")
    return {"document_id": document_id, "chunks": chunks, "count": len(chunks)}


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    agent: AdvancedQAAgent = Depends(get_qa_agent),
):
    """
    Delete a document from the knowledge base.

    Removes the document chunks from the vector index and deletes metadata.
    """
    result = agent.delete_document(document_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("message", "Delete failed"))
    return result


@router.post("/search")
async def search_documents(
    req: SearchRequest,
    agent: AdvancedQAAgent = Depends(get_qa_agent),
):
    """
    Semantic search over the knowledge base.

    Uses vector similarity search with optional HyDE + MQE
    query expansion for improved recall.
    """
    try:
        result = agent.search_knowledge(query=req.query, limit=req.limit)
        return {"query": req.query, "results": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/stats")
async def get_knowledge_stats(
    agent: AdvancedQAAgent = Depends(get_qa_agent),
):
    """Get knowledge base statistics."""
    return agent.knowledge.get_stats()


@router.delete("/clear")
async def clear_all_documents(
    agent: AdvancedQAAgent = Depends(get_qa_agent),
):
    """
    Clear ALL documents from the knowledge base.

    WARNING: This is irreversible. All indexed documents will be removed.
    """
    result = agent.knowledge.clear_all()
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("message", "Clear failed"))
    return result
