"""Question handlers implementing the strategy pattern.

Each handler implements HandlerInterface and is responsible for
a specific category of questions:

- FactualHandler: Knowledge lookup via RAG/web search
- ReasoningHandler: Multi-step reasoning via ReAct loop
- ComparisonHandler: Comparative analysis via Plan-and-Solve
- CalculationHandler: Mathematical computation via calculator
- MixedHandler: Complex multi-part questions via decomposition
"""
from app.core.handlers.base import HandlerInterface, HandlerResult
from app.core.handlers.factual import FactualHandler
from app.core.handlers.reasoning import ReasoningHandler
from app.core.handlers.comparison import ComparisonHandler
from app.core.handlers.calculation import CalculationHandler
from app.core.handlers.mixed import MixedHandler

__all__ = [
    "HandlerInterface",
    "HandlerResult",
    "FactualHandler",
    "ReasoningHandler",
    "ComparisonHandler",
    "CalculationHandler",
    "MixedHandler",
]
