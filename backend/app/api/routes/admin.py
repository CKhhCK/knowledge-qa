"""
Admin endpoints for system statistics and monitoring.

Provides visibility into agent performance, session metrics,
and available tools.
"""

from fastapi import APIRouter, Request, Depends

from app.core.agent import AdvancedQAAgent
from app.api.dependencies import get_qa_agent

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/stats")
async def get_stats(agent: AdvancedQAAgent = Depends(get_qa_agent)):
    """Get system-wide statistics."""
    session_stats = await agent.get_stats()
    return {
        **session_stats,
        "llm_model": agent.settings.llm_model_id,
        "reflection_enabled": agent.settings.reflection_enabled,
        "max_react_steps": agent.settings.max_react_steps,
    }


@router.get("/tools")
async def list_tools(agent: AdvancedQAAgent = Depends(get_qa_agent)):
    """List all available tools."""
    return {
        "tools": await agent.list_tools(),
        "count": len(await agent.list_tools()),
    }
