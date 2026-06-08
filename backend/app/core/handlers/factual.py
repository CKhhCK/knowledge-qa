from __future__ import annotations
"""
FactualHandler — handles knowledge-based factual questions.

Uses a three-tier retrieval strategy:
1. Knowledge Base (RAG): Search ingested documents first
2. Cognitive Memory: Recall relevant past interactions
3. Web Search: Fall back to internet search for real-time info

This ensures answers are grounded in the organization's knowledge
while still being able to handle current events and new topics.
"""

import time
from typing import Optional, TYPE_CHECKING

from hello_agents import HelloAgentsLLM, ToolRegistry

from app.config import Settings
from app.core.handlers.base import HandlerInterface, HandlerResult
from app.models.chat import StepInfo, ToolCallRecord
from app.utils.logging import get_logger

if TYPE_CHECKING:
    from app.knowledge.manager import KnowledgeManager

logger = get_logger(__name__)

FACTUAL_SYSTEM_PROMPT = """You are a Q&A assistant. Answer based on the provided document snippets.

## CRITICAL: Source Citation Rules
1. **Must cite**: After every fact, add the source like `【来源: 文档名】`
2. **No source = must say so**: If the retrieved documents don't contain the answer, start with:
   "⚠️ 知识库中未找到相关信息。以下回答基于通用知识，可能不准确："
3. **Don't make things up**: Only state facts that appear in the retrieved snippets

## Guidelines:
1. **Source First**: Always answer from documents first, general knowledge last
2. **Accuracy**: Don't invent facts not in the sources
3. **Clarity**: Use markdown formatting

## Format:
- If documents have the answer: 回答 + `【来源: xxx文档】`
- If not: "⚠️ 未在知识库中找到相关信息。以下为通用知识回答：..."
"""


class FactualHandler(HandlerInterface):
    """
    Handler for factual/knowledge questions.

    Three-tier retrieval strategy:
    1. Knowledge Base (RAG) — primary source, domain knowledge
    2. Cognitive Memory — past interactions and learned concepts
    3. Web Search — real-time internet fallback

    Strategy:
    - If knowledge base has documents: RAG ask first, web search as supplement
    - If no knowledge base: memory recall + web search
    - If real-time info needed: web search is primary
    """

    def __init__(
        self,
        llm: HelloAgentsLLM,
        tool_registry: ToolRegistry,
        config: Settings,
        knowledge: Optional["KnowledgeManager"] = None,
    ):
        super().__init__(llm, tool_registry, config, knowledge)

    @property
    def handler_name(self) -> str:
        return "factual"

    async def handle(self, question: str, context: Optional[dict] = None) -> HandlerResult:
        """Answer a factual question with tiered knowledge retrieval."""
        start_time = time.perf_counter()
        steps: list[StepInfo] = []
        tool_calls: list[ToolCallRecord] = []
        knowledge_context = ""

        has_kb = self.knowledge and self.knowledge.has_knowledge()
        needs_realtime = self._might_need_search(question)

        # --- Tier 1: Knowledge Base (RAG) ---
        if has_kb:
            t0 = time.perf_counter()
            try:
                # Simple search first (fast). Upgrade to HyDE+MQE only if needed.
                kb_answer = await self.knowledge.ask_knowledge(
                    question, limit=5, enable_advanced=False, include_citations=True
                )
                if not kb_answer or len(kb_answer) < 50:
                    logger.info("Simple search insufficient, trying advanced (HyDE+MQE)")
                    kb_answer = await self.knowledge.ask_knowledge(
                        question, limit=5, enable_advanced=True, include_citations=True
                    )
                if kb_answer and len(kb_answer) > 20:
                    knowledge_context = f"[Knowledge Base]:\n{kb_answer}"
                    steps.append(StepInfo(
                        step_number=1,
                        thought="Searching knowledge base for relevant documents",
                        action=f'rag.ask[question="{question[:80]}"]',
                        observation=kb_answer[:500],
                        duration_ms=int((time.perf_counter() - t0) * 1000),
                    ))
                    logger.info("KB answer found", extra={"question": question[:80], "len": len(kb_answer)})
            except Exception as e:
                logger.warning("KB ask failed", extra={"error": str(e)})

        # --- Tier 2: Conversation History Search ---
        # Only triggered when question contains time references
        time_triggers = ["刚才", "之前", "上次", "前面", "刚刚", "那个", "你提到的",
                         "总结一下", "回顾", "上面", "前面说", "你之前说"]
        need_history = any(t in question for t in time_triggers)

        if need_history and context and context.get("history"):
            t_hist = time.perf_counter()
            try:
                history_result = self.knowledge.search_conversation_history(
                    question, context["history"], limit=5
                )
                if history_result and len(history_result) > 20:
                    knowledge_context += f"\n\n[Conversation History]:\n{history_result}"
                    steps.append(StepInfo(
                        step_number=len(steps) + 1,
                        thought="搜索对话历史查找相关讨论",
                        action="history.search",
                        observation=history_result[:400],
                        duration_ms=int((time.perf_counter() - t_hist) * 1000),
                    ))
                    logger.info(f"History search found relevant messages")
            except Exception as e:
                logger.debug(f"History search skipped: {e}")

        # --- Tier 3: Cognitive Memory (use context.memory_context from agent) ---
        if context and context.get("memory_context"):
            mem_context = context["memory_context"]
            if mem_context and len(mem_context) > 10:
                knowledge_context += f"\n\n[Relevant Memory]:\n{mem_context}"

        # --- Tier 3: Web Search (if needed or no KB) ---
        if needs_realtime or not knowledge_context:
            t2 = time.perf_counter()
            try:
                search_tool = self.tool_registry.get_tool("web_search")
                if search_tool:
                    search_result = search_tool.run({"query": question})
                    knowledge_context += f"\n\n[Web Search Results]:\n{search_result}"
                    tool_calls.append(ToolCallRecord(
                        tool_name="web_search",
                        input_params=question,
                        output_summary=search_result[:300],
                        duration_ms=int((time.perf_counter() - t2) * 1000),
                        success=True,
                    ))
                    steps.append(StepInfo(
                        step_number=len(steps) + 1,
                        thought="Searching the web for current information",
                        action=f'web_search[query="{question[:80]}"]',
                        observation=search_result[:500],
                        tool_calls=[tool_calls[-1]],
                        duration_ms=int((time.perf_counter() - t2) * 1000),
                    ))
            except Exception as e:
                logger.warning("Web search failed", extra={"error": str(e)})

        # --- Generate Final Answer ---
        t3 = time.perf_counter()
        logger.info(f"[TIMING] factual_generate_start | so_far={(t3-start_time)*1000:.0f}ms")
        messages = [{"role": "system", "content": FACTUAL_SYSTEM_PROMPT}]

        # Add conversation history (from SessionStore)
        has_history = False
        if context and context.get("history"):
            for msg in context["history"][-10:]:
                if msg.get("role") and msg.get("content"):
                    messages.append({"role": msg["role"], "content": msg["content"]})
                    has_history = True

        # If no session history, try episodic memory (survives restart)
        if not has_history and context and context.get("memory_context"):
            # Inject past conversations as chat messages so LLM can reference them
            memory_text = context["memory_context"]
            messages.append({
                "role": "system",
                "content": (
                    "以下是你之前和用户的对话记录（来自长期记忆）。"
                    "用户可能会问'上次问了什么'之类的问题，请直接引用这些记录回答：\n\n"
                    f"{memory_text}"
                ),
            })

        # Inject retrieved knowledge context with source labels
        if knowledge_context and knowledge_context != context.get("memory_context", ""):
            messages.append({
                "role": "system",
                "content": (
                    f"以下是从知识库检索到的文档片段。请基于这些内容回答，并标注来源。\n"
                    f"如果内容不包含答案，不要编造，明确告知用户。\n\n"
                    f"{knowledge_context}"
                ),
            })

        messages.append({"role": "user", "content": question})

        try:
            answer = self.llm.invoke(messages)
            logger.info(f"[TIMING] factual_llm_done | llm_call={(time.perf_counter()-t3)*1000:.0f}ms")
        except Exception as e:
            logger.error("LLM invoke failed", extra={"error": str(e)})
            answer = knowledge_context if knowledge_context else f"Error: {str(e)}"

        steps.append(StepInfo(
            step_number=len(steps) + 1,
            thought="Synthesizing answer from retrieved knowledge",
            action="Generate final answer",
            observation=answer[:500],
            duration_ms=int((time.perf_counter() - t3) * 1000),
        ))

        total_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info(
            "FactualHandler completed",
            extra={
                "question": question[:80],
                "duration_ms": total_ms,
                "kb_used": has_kb,
                "web_search_used": len(tool_calls) > 0,
            },
        )

        return HandlerResult(
            answer=answer,
            steps=steps,
            tool_calls=tool_calls,
            total_duration_ms=total_ms,
        )

    def _might_need_search(self, question: str) -> bool:
        """Heuristic: check if the question likely needs real-time web search."""
        search_indicators = [
            "latest", "recent", "today", "current", "now", "news",
            "最新", "最近", "今天", "当前", "现在", "今年", "新闻", "事件",
            "2024", "2025", "2026",
        ]
        question_lower = question.lower()
        return any(indicator in question_lower for indicator in search_indicators)
