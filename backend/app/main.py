"""
FastAPI application entry point for the HelloAgents QA backend.
"""
# Auto-add hello_agents to path (no need to set PYTHONPATH manually)
import sys, os
_hello_agents_path = os.path.join(os.path.dirname(__file__), "..", "..", "..",
    "hello-agents", "Co-creation-projects", "lcyting-StockSage-agent", "HelloAgents Optimized")
_hello_agents_path = os.path.abspath(_hello_agents_path)
if os.path.isdir(_hello_agents_path) and _hello_agents_path not in sys.path:
    sys.path.insert(0, _hello_agents_path)

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings, Settings
from app.core.agent import AdvancedQAAgent
from app.api.routes import chat, health, admin, documents, llm as llm_routes
from app.auth.routes import router as auth_router
from app.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)

from app.utils.middleware import (
    RateLimitMiddleware,
    TimingMiddleware,
    RequestIDMiddleware,
    AccessLogMiddleware,
)
from app.utils.errors import (
    QAAgentError,
    ClassificationError,
    HandlerTimeoutError,
    LLMServiceError,
    RateLimitExceededError,
    SessionNotFoundError,
    ValidationError,
)

logger = get_logger(__name__)


# --- Lifespan Handler ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Startup: Init agent first, then pre-load Qwen3 in background.
    """
    settings = get_settings()
    setup_logging(level=settings.log_level, fmt=settings.log_format)

    logger.info("=" * 50)
    logger.info("HelloAgents QA Backend Starting")
    logger.info(f"Model: {settings.llm_model_id}")
    logger.info("=" * 50)

    app.state.settings = settings

    # Init agent immediately (LLM + tools, no embedding model yet)
    try:
        app.state.qa_agent = AdvancedQAAgent(settings=settings)
        logger.info("QA agent initialized")
    except Exception as e:
        logger.error(f"Failed to initialize QA agent: {e}")
        raise

    # Accept requests immediately
    app.state.ready = True

    # Pre-load Qwen3 in background thread (doesn't block requests)
    import threading as _th
    from app.knowledge.embedding import preload_model
    _th.Thread(target=preload_model, daemon=True, name="embed-warmup").start()

    yield  # --- Application runs here ---

    # Shutdown
    logger.info("Shutting down QA agent...")
    if hasattr(app.state, "qa_agent"):
        await app.state.qa_agent.close()
    logger.info("HelloAgents QA Backend Stopped")


# --- Application Factory ---

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="HelloAgents QA",
        description="""
        Intelligent Q&A Assistant API powered by multi-strategy AI agents.

        ## Features
        - **Automatic question classification** — factual, reasoning, comparison, calculation, mixed
        - **Multi-strategy processing** — ReAct, Plan-and-Solve, RAG, direct LLM
        - **Reflection verification** — automatic answer quality check and refinement
        - **Streaming responses** — real-time SSE streaming of agent reasoning
        - **Complete trace** — full visibility into agent decisions

        ## Agent Strategies
        | Category | Strategy | Description |
        |----------|----------|-------------|
        | factual | Direct + RAG | Knowledge lookup with web search |
        | reasoning | ReAct Loop | Multi-step thought-action-observation |
        | comparison | Plan-and-Solve | Decompose → Execute → Synthesize |
        | calculation | Chain-of-Thought | Extract expression → Compute → Explain |
        | mixed | Decomposition | Split → Route → Synthesize |
        """,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    # --- CORS Middleware ---
    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Request Tracking Middleware ---
    app.add_middleware(AccessLogMiddleware)    # Log every request/response
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(
        RateLimitMiddleware,
        max_requests=settings.rate_limit_per_session,
        window_seconds=60,
    )

    # --- Global Exception Handlers ---

    @app.exception_handler(QAAgentError)
    async def qa_agent_error_handler(request: Request, exc: QAAgentError):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": type(exc).__name__,
                "detail": exc.detail,
                "status_code": exc.status_code,
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.exception_handler(ValidationError)
    async def validation_error_handler(request: Request, exc: ValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": "ValidationError",
                "detail": exc.detail,
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        return JSONResponse(
            status_code=404,
            content={
                "error": "NotFound",
                "detail": f"Path '{request.url.path}' not found",
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled exception",
            extra={"path": str(request.url), "error": str(exc)},
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "detail": "An unexpected error occurred",
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    # --- Register Routes ---
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(chat.router, prefix="/api/v1")
    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(admin.router, prefix="/api/v1")
    app.include_router(llm_routes.router, prefix="/api/v1")

    return app


# --- Main Entry Point ---

app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
