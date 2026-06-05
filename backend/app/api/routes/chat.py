"""
Chat API routes — the primary user-facing endpoints.

Endpoints:
- POST /chat: Send a message, get a complete answer with trace
- POST /chat/stream: Send a message, get streaming SSE response
- GET /chat/{session_id}/history: Get conversation history
- GET /chat/sessions: List active sessions
- DELETE /chat/{session_id}: Clear a session
"""

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from app.core.agent import AdvancedQAAgent
from app.api.dependencies import get_qa_agent, get_current_user_id
from app.models.chat import ChatRequest, ChatResponse
from app.utils.errors import (
    QAAgentError,
    SessionNotFoundError,
    RateLimitExceededError,
    LLMServiceError,
    ClassificationError,
    HandlerTimeoutError,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    agent: AdvancedQAAgent = Depends(get_qa_agent),
    user_id: str = Depends(get_current_user_id),
):
    """
    Send a message and get a complete answer.

    Returns the full answer along with a detailed trace of the
    agent's reasoning process (classification, handler, reflection).
    """
    try:
        enable_reflection = (
            request.options.enable_reflection
            if request.options
            else None
        )

        response = await agent.answer(
            session_id=request.session_id,
            question=request.message,
            enable_reflection=enable_reflection,
            user_id=user_id,
        )
        return response

    except (ClassificationError, HandlerTimeoutError) as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except LLMServiceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except QAAgentError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        logger.error("Unhandled error in chat endpoint", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/stream")
async def stream_message(
    request: ChatRequest,
    agent: AdvancedQAAgent = Depends(get_qa_agent),
    user_id: str = Depends(get_current_user_id),
):
    """
    Send a message and get a streaming SSE response.

    Events are emitted as the agent processes the question:
    - status: Pipeline progress updates
    - thought: Agent's reasoning thoughts
    - action: Tool calls
    - observation: Tool results
    - text: Answer text
    - done: Complete answer with metadata
    """
    enable_reflection = (
        request.options.enable_reflection
        if request.options
        else None
    )

    async def event_generator():
        try:
            async for event_json in agent.answer_stream(
                session_id=request.session_id,
                question=request.message,
                enable_reflection=enable_reflection,
                user_id=user_id,
            ):
                yield {
                    "event": "message",
                    "data": event_json,
                }
        except Exception as e:
            logger.error("Stream error", extra={"error": str(e)})
            yield {
                "event": "error",
                "data": json.dumps({
                    "type": "error",
                    "content": f"Stream error: {str(e)}",
                }),
            }

    return EventSourceResponse(event_generator())


@router.get("/{session_id}/history")
async def get_history(
    session_id: str,
    agent: AdvancedQAAgent = Depends(get_qa_agent),
):
    """Get the complete message history for a session."""
    history = await agent.get_history(session_id)
    return {
        "session_id": session_id,
        "messages": history,
        "message_count": len(history) // 2,  # Divide by 2 for Q&A pairs
    }


@router.get("/conversations")
async def list_conversations(
    agent: AdvancedQAAgent = Depends(get_qa_agent),
    user_id: str = Depends(get_current_user_id),
):
    """List all conversations for the current user."""
    convs = agent.list_conversations(user_id)
    return {"conversations": convs, "count": len(convs)}


@router.post("/conversations")
async def create_conversation(
    agent: AdvancedQAAgent = Depends(get_qa_agent),
    user_id: str = Depends(get_current_user_id),
):
    """Create a new conversation."""
    conv = agent.create_conversation(user_id)
    return conv


@router.delete("/conversations/{conversation_id}")
async def delete_conversation_route(
    conversation_id: str,
    agent: AdvancedQAAgent = Depends(get_qa_agent),
):
    """Delete a conversation and all its messages."""
    existed = agent.delete_conversation(conversation_id)
    if not existed:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found")
    return {"status": "ok", "conversation_id": conversation_id}


@router.delete("/{session_id}")
async def clear_session(
    session_id: str,
    agent: AdvancedQAAgent = Depends(get_qa_agent),
):
    """Clear a conversation session."""
    existed = await agent.clear_session(session_id)
    if not existed:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return {"status": "ok", "session_id": session_id, "message": "Session cleared"}
