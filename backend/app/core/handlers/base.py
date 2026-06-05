"""
Abstract base class for all question handlers.

Defines the strategy pattern interface that each handler implements.
Every handler receives a question and returns a HandlerResult with
the answer and trace information.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

from hello_agents import HelloAgentsLLM, ToolRegistry

from app.config import Settings
from app.models.chat import StepInfo, ToolCallRecord

if TYPE_CHECKING:
    from app.knowledge.manager import KnowledgeManager


class HandlerResult(BaseModel):
    """Result from a handler's execution."""
    answer: str = Field(description="The handler's draft answer")
    steps: list[StepInfo] = Field(
        default_factory=list, description="Trace steps recorded during execution"
    )
    tool_calls: list[ToolCallRecord] = Field(
        default_factory=list, description="All tool invocations made"
    )
    total_duration_ms: int = Field(
        default=0, description="Total handler execution time in milliseconds"
    )


class HandlerInterface(ABC):
    """
    Abstract base for question handlers (Strategy Pattern).

    Each concrete handler processes a specific category of questions:
    - factual: Knowledge base → web search → LLM
    - reasoning: ReAct loop with tools + knowledge context
    - comparison: Plan-and-Solve decomposition
    - calculation: CoT extraction → CalculatorTool
    - mixed: Decomposition → recursive dispatch
    """

    def __init__(
        self,
        llm: HelloAgentsLLM,
        tool_registry: ToolRegistry,
        config: Settings,
        knowledge: Optional["KnowledgeManager"] = None,
    ):
        self.llm = llm
        self.tool_registry = tool_registry
        self.config = config
        self.knowledge = knowledge

    @property
    @abstractmethod
    def handler_name(self) -> str:
        """Unique name for this handler (used in logging and traces)."""
        ...

    @abstractmethod
    async def handle(self, question: str, context: Optional[dict] = None) -> HandlerResult:
        """
        Process a question and return a draft answer with trace information.

        Args:
            question: The user's question text.
            context: Optional additional context (conversation history, memory).

        Returns:
            HandlerResult containing the draft answer and execution trace.
        """
        ...
