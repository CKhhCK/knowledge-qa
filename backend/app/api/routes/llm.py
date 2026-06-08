"""
Direct LLM invoke endpoint — bypasses the agent pipeline entirely.

Use this for evaluation, testing, or any scenario where you need
a raw LLM response without classification, RAG, or reflection.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.dependencies import get_qa_agent
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/llm", tags=["LLM"])


class LLMInvokeRequest(BaseModel):
    messages: list[dict] = Field(
        default=...,
        description="List of messages in OpenAI format: [{'role': 'system'|'user'|'assistant', 'content': '...'}]",
    )
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=2048, ge=1, le=32768, description="Max tokens to generate")


class LLMInvokeResponse(BaseModel):
    content: str = Field(description="The LLM's response text")


@router.post("/invoke", response_model=LLMInvokeResponse)
async def llm_invoke(request: LLMInvokeRequest, agent=Depends(get_qa_agent)):
    """
    Call the LLM directly with the given messages.

    No agent pipeline (classification, RAG, reflection) is involved.
    The messages are sent to the LLM as-is.
    """
    content = agent.llm.invoke(request.messages, temperature=request.temperature)
    return LLMInvokeResponse(content=content or "")
