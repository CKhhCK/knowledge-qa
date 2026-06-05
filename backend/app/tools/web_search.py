"""
Web search tool using DuckDuckGo Instant Answer API (free, no key needed)
with optional SerpAPI/Tavily fallback for enhanced results.

Extends the hello_agents Tool base class for seamless integration
with the agent framework (ToolRegistry, ReActAgent, etc.).
"""

import json
import os
import time
import urllib.request
import urllib.parse
import urllib.error
from typing import Any

from hello_agents.tools.base import Tool, ToolParameter

from app.utils.logging import get_logger

logger = get_logger(__name__)


class WebSearchTool(Tool):
    """
    Web search tool for retrieving real-time information from the internet.

    Uses DuckDuckGo Instant Answer API as the primary free backend.
    Optionally enhances results with Tavily or SerpAPI if API keys are configured.

    Usage in ReAct: Action: web_search[query="latest AI news"]
    """

    def __init__(
        self,
        tavily_api_key: str = "",
        serpapi_api_key: str = "",
    ):
        super().__init__(
            name="web_search",
            description=(
                "搜索互联网获取实时信息和最新数据。"
                "适用于查询时事新闻、最新数据、事实核查等需要外部信息的场景。"
                "输入: 搜索查询字符串。返回: 格式化的搜索结果摘要。"
            ),
        )
        self._tavily_key = tavily_api_key
        self._serpapi_key = serpapi_api_key

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="搜索查询词，使用中文或英文关键词",
                required=True,
            )
        ]

    def run(self, parameters: dict[str, Any]) -> str:
        """Execute a web search and return formatted results."""
        query = parameters.get("query") or parameters.get("input", "")
        if not query.strip():
            return "错误：搜索查询不能为空。"

        start_time = time.perf_counter()

        # Try Tavily first if API key is available (better for agent use)
        if self._tavily_key:
            try:
                result = self._search_tavily(query)
                elapsed = (time.perf_counter() - start_time) * 1000
                logger.info(
                    "Web search completed",
                    extra={"provider": "tavily", "query": query, "duration_ms": elapsed},
                )
                return result
            except Exception as e:
                logger.warning("Tavily search failed, falling back", extra={"error": str(e)})

        # Try SerpAPI if available
        if self._serpapi_key:
            try:
                result = self._search_serpapi(query)
                elapsed = (time.perf_counter() - start_time) * 1000
                return result
            except Exception as e:
                logger.warning("SerpAPI search failed, falling back", extra={"error": str(e)})

        # Fallback: DuckDuckGo Instant Answer (free, no key needed)
        try:
            result = self._search_duckduckgo(query)
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.info(
                "Web search completed",
                extra={"provider": "duckduckgo", "query": query, "duration_ms": elapsed},
            )
            return result
        except Exception as e:
            logger.error("All search providers failed", extra={"error": str(e)})
            return f"搜索失败：所有搜索引擎均不可用。错误信息: {str(e)}"

    def _search_tavily(self, query: str) -> str:
        """Search using Tavily API."""
        url = "https://api.tavily.com/search"
        data = json.dumps({
            "api_key": self._tavily_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": True,
            "max_results": 5,
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        if result.get("answer"):
            return f"[Tavily综合回答] {result['answer']}"

        return self._format_results(
            [(r.get("title", ""), r.get("content", ""), r.get("url", ""))
             for r in result.get("results", [])]
        )

    def _search_serpapi(self, query: str) -> str:
        """Search using SerpAPI."""
        params = urllib.parse.urlencode({
            "q": query,
            "api_key": self._serpapi_key,
            "num": 5,
            "output": "json",
        })
        url = f"https://serpapi.com/search?{params}"

        with urllib.request.urlopen(url, timeout=15) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        organic = result.get("organic_results", [])
        return self._format_results(
            [(r.get("title", ""), r.get("snippet", ""), r.get("link", ""))
             for r in organic]
        )

    def _search_duckduckgo(self, query: str) -> str:
        """
        Search using DuckDuckGo Instant Answer API.

        This is a free, no-key-required API. It returns instant answers for
        many types of queries (definitions, calculations, facts, etc.).
        For web links, we also try the HTML version as a fallback.
        """
        # Try Instant Answer API first
        encoded_query = urllib.parse.quote(query)
        ia_url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disambig=1"

        with urllib.request.urlopen(ia_url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        parts = []

        # Abstract (summary answer)
        abstract = data.get("AbstractText", "")
        if abstract:
            source = data.get("AbstractSource", "")
            source_url = data.get("AbstractURL", "")
            parts.append(f"📖 {abstract}")
            if source:
                parts.append(f"   来源: {source} ({source_url})")

        # Answer (direct answer to specific questions)
        answer = data.get("Answer", "")
        if answer:
            parts.append(f"💡 直接回答: {answer}")

        # Definition
        definition = data.get("Definition", "")
        if definition:
            parts.append(f"📚 定义: {definition}")

        # Related topics (as search results)
        related = data.get("RelatedTopics", [])
        if related and not abstract:
            results = []
            for topic in related[:5]:
                if isinstance(topic, dict):
                    text = topic.get("Text", "")
                    url = topic.get("FirstURL", "")
                    # Clean HTML entities
                    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                    results.append((text, url))
            if results:
                parts.append(self._format_related_topics(results))

        if not parts:
            return f"未找到关于'{query}'的搜索结果。请尝试更具体的查询词。"

        return "\n\n".join(parts)

    def _format_results(self, results: list[tuple[str, str, str]]) -> str:
        """Format search results into a readable string."""
        if not results:
            return "未找到相关搜索结果。"

        formatted = []
        for i, (title, snippet, url) in enumerate(results[:5], 1):
            # Clean snippet
            snippet = snippet.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            formatted.append(f"{i}. **{title}**\n   {snippet}\n   🔗 {url}")

        return "\n\n".join(formatted)

    def _format_related_topics(self, topics: list[tuple[str, str]]) -> str:
        """Format DuckDuckGo related topics."""
        lines = ["🔍 相关搜索结果:"]
        for i, (text, url) in enumerate(topics, 1):
            lines.append(f"{i}. {text}\n   🔗 {url}")
        return "\n".join(lines)
