"""
Health check and readiness endpoints.

Provides standard endpoints for monitoring and orchestration:
- /health: Basic liveness check
- /health/ready: Readiness check (verifies LLM connectivity)
"""

from fastapi import APIRouter, Request

from app.core.agent import AdvancedQAAgent
from app.api.dependencies import get_qa_agent

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    """Liveness probe — always returns OK if the server is running."""
    return {
        "status": "healthy",
        "service": "helloagents-qa",
        "version": "1.0.0",
    }


@router.get("/health/ready")
async def readiness_check(request: Request):
    """
    Readiness probe — verifies the QA agent is initialized and ready.

    Returns 503 if not ready (e.g., during startup).
    """
    agent = getattr(request.app.state, "qa_agent", None)
    if agent is None:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "Agent not initialized"},
        )

    return {
        "status": "ready",
        "service": "helloagents-qa",
        "llm_model": agent.settings.llm_model_id,
        "tools_available": agent.tool_registry.list_tools(),
    }
