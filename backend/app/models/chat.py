"""
Chat-related Pydantic models.

These define the API contract between frontend and backend.
All models use Pydantic for automatic validation and OpenAPI schema generation.
"""

from datetime import datetime
from typing import Optional
from uuid import uuid4
from pydantic import BaseModel, Field


# --- Request Models ---

class ChatOptions(BaseModel):
    """Per-request options to override agent defaults."""
    enable_reflection: Optional[bool] = Field(
        default=None, description="Override reflection setting (None = use default)"
    )
    max_react_steps: Optional[int] = Field(
        default=None, ge=1, le=10, description="Override max ReAct steps"
    )
    prefer_stream: bool = Field(
        default=False, description="Whether the client prefers streaming response"
    )


class ChatRequest(BaseModel):
    """Incoming chat message request."""
    session_id: str = Field(
        default_factory=lambda: uuid4().hex[:12],
        description="Session identifier (auto-generated if not provided)",
    )
    message: str = Field(
        ..., min_length=1, max_length=4000, description="User's question"
    )
    options: Optional[ChatOptions] = Field(
        default=None, description="Optional per-request overrides"
    )


# --- Response Models ---

class ToolCallRecord(BaseModel):
    """Record of a single tool invocation."""
    tool_name: str
    input_params: str
    output_summary: str
    duration_ms: int = 0
    success: bool = True
    error_message: Optional[str] = None


class StepInfo(BaseModel):
    """A single step in the agent's reasoning trace."""
    step_number: int
    thought: Optional[str] = None
    action: Optional[str] = None
    observation: Optional[str] = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )
    duration_ms: int = 0


class TraceInfo(BaseModel):
    """Complete trace of the agent's reasoning process."""
    category: str = ""
    confidence: float = 0.0
    routing_reasoning: str = ""
    handler_used: str = ""
    draft_answer: Optional[str] = None
    reflection_feedback: Optional[str] = None
    reflection_score: Optional[float] = None
    steps: list[StepInfo] = Field(default_factory=list)
    total_duration_ms: int = 0


class ChatResponse(BaseModel):
    """Complete chat response."""
    session_id: str
    answer: str
    trace: Optional[TraceInfo] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# --- Streaming Models ---

class StreamEventModel(BaseModel):
    """JSON-serializable event emitted during streaming response.

    Event types:
    - "status": Pipeline status updates (classifying, dispatching, verifying)
    - "thought": Agent's reasoning thought
    - "action": Agent's tool call action
    - "observation": Tool result observation
    - "text": Incremental answer text chunk
    - "error": Error event
    - "done": Final event with complete answer and trace
    """
    type: str = Field(
        ..., description="Event type: status, thought, action, observation, text, error, done"
    )
    content: str = Field(default="", description="Event content")
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata (e.g., category, step number)",
    )
