"""
Tool registry factory for the QA agent.

Creates and configures the ToolRegistry with core tools:
- WebSearchTool: Internet search (DuckDuckGo / Tavily / SerpAPI)
- CalculatorTool: Safe math expression evaluation

Knowledge management (RAG + Memory) is handled by KnowledgeManager
directly, not through the hello_agents Tool system.
"""

from hello_agents import ToolRegistry

from app.config import Settings
from app.tools.web_search import WebSearchTool
from app.tools.calculator import CalculatorTool
from app.utils.logging import get_logger

logger = get_logger(__name__)


def create_tool_registry(settings: Settings) -> ToolRegistry:
    """
    Create and configure the ToolRegistry with all available tools.

    Args:
        settings: Application settings.

    Returns:
        ToolRegistry with web_search and calculator registered.
    """
    registry = ToolRegistry()

    # WebSearchTool — internet search for real-time information
    web_search = WebSearchTool(
        tavily_api_key=settings.tavily_api_key,
        serpapi_api_key=settings.serpapi_api_key,
    )
    registry.register_tool(web_search)
    logger.info("Tool registered: web_search")

    # CalculatorTool — safe math evaluation
    calculator = CalculatorTool()
    registry.register_tool(calculator)
    logger.info("Tool registered: calculator")

    logger.info(
        "Tool registry initialized",
        extra={"tool_count": len(registry.list_tools()), "tools": registry.list_tools()},
    )

    return registry
