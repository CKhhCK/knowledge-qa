from __future__ import annotations
"""
MixedHandler — handles complex multi-part questions by decomposition.

Optimized version with:
1. Parallel sub-question dispatch (asyncio.gather) — major speed boost
2. Skip reflection for sub-questions — they're intermediate, not final
3. Lightweight sub-question classification (rule-first, no LLM)
"""

import asyncio
import json
import re
import time
from typing import TYPE_CHECKING

from hello_agents import HelloAgentsLLM, ToolRegistry

from app.config import Settings
from app.core.handlers.base import HandlerInterface, HandlerResult
from app.models.chat import StepInfo, ToolCallRecord
from app.utils.logging import get_logger

if TYPE_CHECKING:
    from app.core.router import QuestionRouter

logger = get_logger(__name__)

DECOMPOSE_PROMPT = """你是一个复杂问题分解专家。将用户的复合问题拆解为独立的子问题。

## 规则:
1. 每个子问题独立、可单独回答
2. 通常拆分为 2-4 个子问题
3. 保留原始语境和约束

## 输出格式:
```python
["子问题1", "子问题2"]
```

请拆解以下问题:"""

SYNTHESIZE_PROMPT = """综合以下子问题分析结果，生成完整连贯的回答。

## 原始问题: {original_question}

## 各子问题回答:
{sub_answers}

## 要求:
1. 整合所有答案，形成连贯整体
2. 使用markdown格式
3. 消除重复，逻辑流畅
4. **必须保留各子问题回答中的来源标注（如 【来源: xxx】），不可删除**

请生成完整答案:"""

# Lightweight prompt for compound factual questions — all sub-questions are
# simple lookups, so we just need one answer from merged chunks.
LIGHTWEIGHT_FACTUAL_PROMPT = """You are a Q&A assistant. Answer a multi-part question based on the provided document snippets.

## CRITICAL: Source Citation Rules
1. **Must cite**: After every fact, add the source like `【来源: 文档名】`
2. **No source = must say so**: If the retrieved documents don't contain the answer, start with:
   "⚠️ 知识库中未找到相关信息。以下回答基于通用知识，可能不准确："
3. **Don't make things up**: Only state facts that appear in the retrieved snippets

## Guidelines:
1. **Source First**: Always answer from documents first
2. **Address all parts**: Answer each sub-question the user asked
3. **Accuracy**: Don't invent facts not in the sources
4. **Clarity**: Use markdown formatting

## Format:
- If documents have the answer: 回答 + `【来源: xxx文档】`
- If not: "⚠️ 未在知识库中找到相关信息。"
- Use headings to separate different parts of the answer"""


class MixedHandler(HandlerInterface):
    """Handler for complex multi-part questions with parallel optimized dispatch."""

    def __init__(self, llm, tool_registry, config, router, knowledge=None):
        super().__init__(llm, tool_registry, config, knowledge)
        self.router = router

    @property
    def handler_name(self) -> str:
        return "mixed"

    async def handle(self, question: str, context: dict | None = None) -> HandlerResult:
        """Process complex question with parallel sub-question dispatch."""
        start_time = time.perf_counter()
        steps: list[StepInfo] = []

        # --- Phase 1: Decompose (1 LLM call) ---
        t0 = time.perf_counter()
        sub_questions = await self._decompose(question)
        logger.info(
            f"[MIXED] Decomposed into {len(sub_questions)} sub-questions | "
            f"{(time.perf_counter()-t0)*1000:.0f}ms"
        )
        steps.append(StepInfo(
            step_number=1,
            thought="检测到复合问题，拆解为独立子问题",
            action=f"拆解为{len(sub_questions)}个子问题",
            observation=f"子问题: {'; '.join(sub_questions)}",
            duration_ms=int((time.perf_counter() - t0) * 1000),
        ))

        if not sub_questions:
            from app.core.handlers.factual import FactualHandler
            return await FactualHandler(self.llm, self.tool_registry, self.config, self.knowledge).handle(question, context)

        # --- Phase 1.5: Check for lightweight path (all factual sub-questions) ---
        # Classify each sub-question by rules only — no LLM cost.
        sub_categories = []
        for sq in sub_questions:
            rule_result = self.router._classify_by_rules(sq)
            sub_categories.append(rule_result.category if rule_result else "unknown")

        all_factual = all(c == "factual" for c in sub_categories)
        has_kb = self.knowledge and self.knowledge.has_knowledge()

        if all_factual and has_kb:
            # Lightweight path: multi-query search → merge chunks → one LLM answer
            logger.info(
                f"[MIXED] All {len(sub_questions)} sub-questions factual, "
                f"using lightweight multi-query search"
            )
            t_light = time.perf_counter()

            # Search each sub-question independently, merge results
            kb_context = await self.knowledge.ask_knowledge_multi(
                sub_questions, limit=8, include_citations=True
            )

            steps.append(StepInfo(
                step_number=2,
                thought="所有子问题均为事实查询，并行检索知识库",
                action=f"多查询检索 ({len(sub_questions)}个子查询)",
                observation=kb_context[:500] if kb_context else "(未找到相关内容)",
                duration_ms=int((time.perf_counter() - t_light) * 1000),
            ))

            # One LLM call to answer
            t_gen = time.perf_counter()
            messages = [
                {"role": "system", "content": LIGHTWEIGHT_FACTUAL_PROMPT},
                {"role": "user", "content": (
                    f"## 用户问题\n{question}\n\n"
                    f"## 知识库检索结果\n{kb_context if kb_context else '未找到相关文档。'}"
                )},
            ]
            answer = await asyncio.to_thread(self.llm.invoke, messages)
            gen_ms = int((time.perf_counter() - t_gen) * 1000)

            steps.append(StepInfo(
                step_number=3,
                thought="基于检索结果综合回答",
                action="生成综合回答",
                observation=answer[:500],
                duration_ms=gen_ms,
            ))

            total_ms = int((time.perf_counter() - start_time) * 1000)
            logger.info(
                f"[MIXED] Lightweight complete | {len(sub_questions)} sub-Qs | "
                f"total={total_ms}ms | kb_chars={len(kb_context)}"
            )

            return HandlerResult(
                answer=answer, steps=steps,
                tool_calls=[], total_duration_ms=total_ms,
            )

        # --- Phase 2: Parallel dispatch (full handler per sub-question) ---
        t1 = time.perf_counter()

        async def process_one(i: int, sub_q: str) -> dict:
            """Process a single sub-question — no reflection for intermediate results."""
            t_start = time.perf_counter()
            # Lightweight classification (rules only, OPTIMIZATION #2+4)
            classification = await self.router.classify(sub_q)
            # Dispatch and get answer (skip reflection for sub-questions)
            answer = await self._dispatch_handler(classification.category, sub_q, context)
            elapsed = (time.perf_counter() - t_start) * 1000
            logger.info(
                f"[MIXED] Sub-Q{i+1} [{classification.category}] {sub_q[:50]}... | {elapsed:.0f}ms"
            )
            return {
                "question": sub_q,
                "category": classification.category,
                "answer": answer,
                "elapsed_ms": int(elapsed),
            }

        # PARALLEL: all sub-questions run concurrently
        tasks = [process_one(i, q) for i, q in enumerate(sub_questions)]
        sub_results = await asyncio.gather(*tasks)

        parallel_ms = int((time.perf_counter() - t1) * 1000)
        # Calculate what sequential would have been
        sequential_ms = sum(r.get("elapsed_ms", 0) for r in sub_results)
        speedup = sequential_ms / max(parallel_ms, 1)
        logger.info(
            f"[MIXED] Parallel dispatch: {parallel_ms}ms "
            f"(sequential would be ~{sequential_ms}ms, {speedup:.1f}x speedup)"
        )

        for i, r in enumerate(sub_results):
            steps.append(StepInfo(
                step_number=i + 2,
                thought=f"子问题 {i+1}/{len(sub_questions)} (类型: {r['category']})",
                action=f"并行分派到{r['category']}处理器",
                observation=f"子问题: {r['question'][:100]}\n回答: {r['answer'][:300]}",
                duration_ms=r.get("elapsed_ms", 0),
            ))

        # --- Phase 3: Synthesize (1 LLM call) ---
        t2 = time.perf_counter()
        final_answer = await self._synthesize(question, sub_results)
        logger.info(f"[MIXED] Synthesized | {(time.perf_counter()-t2)*1000:.0f}ms")

        steps.append(StepInfo(
            step_number=len(sub_questions) + 2,
            thought="整合所有子问题答案",
            action="生成综合回答",
            observation=final_answer[:500],
            duration_ms=int((time.perf_counter() - t2) * 1000),
        ))

        total_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info(
            f"[MIXED] Complete | {len(sub_questions)} sub-Qs | parallel={parallel_ms}ms | "
            f"total={total_ms}ms | speedup={speedup:.1f}x"
        )

        return HandlerResult(
            answer=final_answer, steps=steps,
            tool_calls=[], total_duration_ms=total_ms,
        )

    async def _decompose(self, question: str) -> list[str]:
        """Decompose a complex question into sub-questions (1 LLM call)."""
        try:
            messages = [{"role": "user", "content": f"{DECOMPOSE_PROMPT}\n\n{question}"}]
            response = await asyncio.to_thread(self.llm.invoke, messages, temperature=0.1)

            match = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL)
            list_str = match.group(1).strip() if match else ""
            if not list_str:
                match = re.search(r"\[(.*?)\]", response, re.DOTALL)
                if match:
                    list_str = f"[{match.group(1)}]"
                else:
                    list_str = response.strip()

            sub_qs = json.loads(list_str.replace("'", '"'))
            if isinstance(sub_qs, list) and len(sub_qs) > 0:
                return sub_qs
        except Exception as e:
            logger.warning("Decomposition failed, using fallback", extra={"error": str(e)})

        # Fallback: split by question marks
        parts = re.split(r"[？？?？]\s*", question)
        parts = [p.strip() + "？" for p in parts if p.strip()]
        return parts if len(parts) > 1 else [question]

    async def _dispatch_handler(self, category: str, question: str, context: dict | None) -> str:
        """Dispatch a sub-question to appropriate handler (no reflection for intermediates)."""
        try:
            handler = self._get_handler_for_category(category)
            result = await handler.handle(question, context)
            return result.answer
        except Exception as e:
            logger.error(f"Sub-handler failed [{category}]: {e}")
            return f"处理失败: {str(e)}"

    def _get_handler_for_category(self, category: str) -> HandlerInterface:
        from app.core.handlers.factual import FactualHandler
        from app.core.handlers.reasoning import ReasoningHandler
        from app.core.handlers.comparison import ComparisonHandler
        from app.core.handlers.calculation import CalculationHandler

        handlers = {
            "factual": FactualHandler, "reasoning": ReasoningHandler,
            "comparison": ComparisonHandler, "calculation": CalculationHandler,
        }
        cls = handlers.get(category, FactualHandler)
        return cls(self.llm, self.tool_registry, self.config, self.knowledge)

    async def _synthesize(self, original_question: str, sub_results: list[dict]) -> str:
        """Synthesize sub-results into coherent answer (1 LLM call)."""
        sub_answers_text = "\n\n---\n\n".join([
            f"### 子问题 {i+1} ({r['category']}): {r['question']}\n\n{r['answer']}"
            for i, r in enumerate(sub_results)
        ])

        try:
            prompt = SYNTHESIZE_PROMPT.format(
                original_question=original_question, sub_answers=sub_answers_text,
            )
            messages = [{"role": "user", "content": prompt}]
            return await asyncio.to_thread(self.llm.invoke, messages)
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return f"## 综合回答\n\n{sub_answers_text}"
