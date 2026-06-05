"""
Session-related Pydantic models for conversation management.
"""

from datetime import datetime
from typing import Optional
from uuid import uuid4
from pydantic import BaseModel, Field
from app.models.chat import TraceInfo


class SessionMessage(BaseModel):
    """A single message exchange in a session."""
    role: str  # "user" or "assistant"
    content: str
    trace: Optional[TraceInfo] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class SessionInfo(BaseModel):
    """Full session data."""
    session_id: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    messages: list[SessionMessage] = Field(default_factory=list)
    message_count: int = 0
    total_duration_ms: int = 0

    def update_timestamp(self) -> None:
        """Update the last-modified timestamp."""
        self.updated_at = datetime.now().isoformat()
        self.message_count = len(self.messages)


class SessionSummary(BaseModel):
    """Lightweight session metadata for listing."""
    session_id: str
    created_at: str
    updated_at: str
    message_count: int
    preview: str = ""  # First 80 chars of the first user message
