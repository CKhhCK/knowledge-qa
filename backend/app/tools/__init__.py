"""Custom tools for the QA agent."""
from app.tools.web_search import WebSearchTool
from app.tools.calculator import CalculatorTool
from app.tools.registry import create_tool_registry

__all__ = ["WebSearchTool", "CalculatorTool", "create_tool_registry"]
