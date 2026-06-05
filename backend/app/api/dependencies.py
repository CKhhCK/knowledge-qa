"""
FastAPI dependency injection for the QA agent and user authentication.
"""

from fastapi import Request, Header

from app.core.agent import AdvancedQAAgent
from app.memory.session_store import SessionStore
from app.auth.security import verify_token
from app.utils.errors import ConfigurationError
from app.utils.logging import get_logger

logger = get_logger(__name__)


def get_qa_agent(request: Request) -> AdvancedQAAgent:
    """Dependency that returns the QA agent singleton."""
    agent = getattr(request.app.state, "qa_agent", None)
    if agent is None:
        raise ConfigurationError(detail="QA agent not initialized.")
    return agent


def get_session_store(request: Request) -> SessionStore:
    """Dependency that returns the session store."""
    agent = get_qa_agent(request)
    return agent.session_store


def get_current_user_id(authorization: str = Header(default="")) -> str:
    """
    Extract user_id from JWT Bearer token.

    Returns the user_id if authenticated, "anonymous" otherwise.
    This allows both authenticated and unauthenticated access to the chat.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return "anonymous"

    token = authorization[7:]  # Remove "Bearer " prefix
    payload = verify_token(token)
    if not payload:
        return "anonymous"

    return payload.get("sub", "anonymous")
