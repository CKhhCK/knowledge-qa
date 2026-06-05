from __future__ import annotations
"""
Custom exception classes for the QA agent backend.

Each exception maps to an appropriate HTTP status code,
used by FastAPI's global exception handlers.
"""

from typing import Any


class QAAgentError(Exception):
    """Base exception for all QA agent errors."""
    status_code: int = 500
    detail: str = "Internal server error"

    def __init__(self, detail: str | None = None, **kwargs: Any):
        self.detail = detail or self.detail
        self.extra = kwargs
        super().__init__(self.detail)


class ConfigurationError(QAAgentError):
    """Raised when required configuration is missing or invalid."""
    status_code = 500
    detail = "Server configuration error"


class ClassificationError(QAAgentError):
    """Raised when the question router cannot classify a question."""
    status_code = 422
    detail = "Unable to classify the question"


class HandlerError(QAAgentError):
    """Raised when a handler fails to process a question."""
    status_code = 500
    detail = "Handler execution failed"

    def __init__(self, handler_name: str, detail: str | None = None):
        super().__init__(detail=detail or f"Handler '{handler_name}' execution failed")
        self.handler_name = handler_name


class HandlerTimeoutError(QAAgentError):
    """Raised when a handler exceeds its execution timeout."""
    status_code = 504
    detail = "Handler execution timed out"


class LLMServiceError(QAAgentError):
    """Raised when the LLM service is unavailable or returns an error."""
    status_code = 502
    detail = "LLM service unavailable"


class ToolExecutionError(QAAgentError):
    """Raised when a tool execution fails."""
    status_code = 502
    detail = "Tool execution failed"

    def __init__(self, tool_name: str, detail: str | None = None):
        super().__init__(detail=detail or f"Tool '{tool_name}' execution failed")
        self.tool_name = tool_name


class SessionNotFoundError(QAAgentError):
    """Raised when a requested session does not exist."""
    status_code = 404
    detail = "Session not found"


class RateLimitExceededError(QAAgentError):
    """Raised when a session exceeds its rate limit."""
    status_code = 429
    detail = "Rate limit exceeded"


class ValidationError(QAAgentError):
    """Raised when input validation fails."""
    status_code = 422
    detail = "Invalid input"
