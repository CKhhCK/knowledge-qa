from __future__ import annotations
"""
ComparisonHandler — handles comparative analysis questions using Plan-and-Solve.

Implements the Plan-and-Solve pattern from Chapter 4:
1. Planner: Decompose the comparison into structured sub-questions
2. Executor: Answer each sub-question sequentially, gathering information
3. Synthesizer: Merge findings into a coherent comparative analysis

This approach ensures structured, thorough comparisons rather than
superficial one-shot answers.
"""

import asyncio
import json
import re
import time

from hello_agents import HelloAgentsLLM, ToolRegistry

from app.config import Settings
from app.core.handlers.base import HandlerInterface, HandlerResult
from app.models.chat import StepInfo, ToolCallRecord
from app.utils.logging import get_logger

logger = get_logger(__name__)

PLANNER_PROMPT = """你是一个顶级的比较分析规划专家。你的任务是将用户的比较/对比问题分解成一个结构化的分析计划。

## 计划要求:
1. 每个步骤是一个独立的、可执行的子问题
2. 步骤按照逻辑顺序排列
3. 计划应涵盖: 各方独立分析 → 多维度对比 → 综合结论
4. 通常需要 4-6 个步骤

## 输出格式:
请输出一个Python列表，每个元素是一个子问题字符串:
```python
["子问题1", "子问题2", "子问题3", ...]
```

## 示例:
用户问题: "Python和Java在后端开发上有什么区别？"

计划:
```python
[
    "Python在后端开发中的主要特点和优势是什么？",
    "Java在后端开发中的主要特点和优势是什么？",
    "在性能方面，Python和Java后端开发有何差异？",
    "在开发效率和生态系统方面，Python和Java如何对比？",
    "在适用场景和企业采用方面，Python和Java各有什么特点？",
    "综合以上分析，给出选择建议"
]
```

现在请为以下问题制定分析计划，只输出Python列表:"""

EXECUTOR_PROMPT = """你是一个专业的比较分析专家。你正在系统性地回答一个比较分析问题。

## 原始问题:
{original_question}

## 完整分析计划:
{plan_summary}

## 已回答的步骤:
{previous_results}

## 当前需要回答的子问题:
{current_step}

请针对当前子问题给出详细、专业的回答。使用工具(wen_search)搜索最新信息如果需要。
直接给出答案，不需要标注是第几步。"""

SYNTHESIS_PROMPT = """你是一个专业的比较分析专家。请基于以下分步分析结果，生成一个完整的对比分析报告。

## 原始问题:
{original_question}

## 各步骤分析结果:
{all_results}

## 报告要求:
1. 使用markdown格式，结构清晰
2. 包含: 概述 → 各方分析 → 维度对比 → 综合结论
3. 使用表格对比关键差异
4. 给出实用建议或选择指导
5. 语言专业但易懂

请生成完整的对比分析报告:"""


class ComparisonHandler(HandlerInterface):
    """
    Handler for comparison/analysis questions using Plan-and-Solve.

    Pipeline: Plan → Execute each step → Synthesize final report
    """

    @property
    def handler_name(self) -> str:
        return "comparison"

    async def handle(self, question: str, context: dict | None = None) -> HandlerResult:
        """Process a comparison question using Plan-and-Solve."""
        start_time = time.perf_counter()
        steps: list[StepInfo] = []
        tool_calls: list[ToolCallRecord] = []

        # --- Phase 1: Plan ---
        t0 = time.perf_counter()
        plan = await self._generate_plan(question)
        steps.append(StepInfo(
            step_number=1,
            thought=f"制定分析计划，将问题分解为{len(plan)}个子问题",
            action=f"规划了{len(plan)}个分析步骤",
            observation=f"计划: {'; '.join(plan[:3])}{'...' if len(plan) > 3 else ''}",
            duration_ms=int((time.perf_counter() - t0) * 1000),
        ))

        # --- Phase 2: Execute each step ---
        results: list[dict] = []
        for i, sub_question in enumerate(plan):
            t1 = time.perf_counter()

            # Build context from previous results
            previous = "\n".join([
                f"步骤{j+1}: {r['question']}\n回答: {r['answer'][:300]}"
                for j, r in enumerate(results)
            ]) if results else "无"

            plan_summary = "\n".join([f"{j+1}. {s}" for j, s in enumerate(plan)])

            prompt = EXECUTOR_PROMPT.format(
                original_question=question,
                plan_summary=plan_summary,
                previous_results=previous,
                current_step=sub_question,
            )

            try:
                messages = [{"role": "user", "content": prompt}]
                answer = await asyncio.to_thread(self.llm.invoke, messages)
            except Exception as e:
                answer = f"子问题回答失败: {str(e)}"

            results.append({"question": sub_question, "answer": answer})

            steps.append(StepInfo(
                step_number=i + 2,
                thought=f"执行计划步骤 {i+1}/{len(plan)}",
                action=f"回答子问题: {sub_question[:100]}",
                observation=answer[:500],
                duration_ms=int((time.perf_counter() - t1) * 1000),
            ))

        # --- Phase 3: Synthesize ---
        t2 = time.perf_counter()
        all_results_text = "\n\n---\n\n".join([
            f"### 步骤{i+1}: {r['question']}\n{r['answer']}"
            for i, r in enumerate(results)
        ])

        synthesis_prompt = SYNTHESIS_PROMPT.format(
            original_question=question,
            all_results=all_results_text,
        )

        try:
            messages = [{"role": "user", "content": synthesis_prompt}]
            final_answer = await asyncio.to_thread(self.llm.invoke, messages)
        except Exception as e:
            # Fallback: concatenate results
            final_answer = f"## 对比分析报告\n\n{all_results_text}\n\n> 注意: 综合分析生成失败({str(e)[:100]})，以上为分步分析结果。"

        steps.append(StepInfo(
            step_number=len(plan) + 2,
            thought="综合分析所有步骤的结果",
            action="生成综合对比报告",
            observation=final_answer[:500],
            duration_ms=int((time.perf_counter() - t2) * 1000),
        ))

        total_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info(
            "ComparisonHandler completed",
            extra={"question": question[:80], "plan_steps": len(plan), "duration_ms": total_ms},
        )

        return HandlerResult(
            answer=final_answer,
            steps=steps,
            tool_calls=tool_calls,
            total_duration_ms=total_ms,
        )

    async def _generate_plan(self, question: str) -> list[str]:
        """Use LLM to generate a comparison analysis plan."""
        try:
            messages = [
                {"role": "user", "content": f"{PLANNER_PROMPT}\n\n用户问题: {question}"},
            ]
            response = await asyncio.to_thread(self.llm.invoke, messages, temperature=0.3)

            # Extract the Python list
            match = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                plan_str = match.group(1).strip()
            else:
                # Try to find a list directly
                match = re.search(r"\[(.*?)\]", response, re.DOTALL)
                if match:
                    plan_str = f"[{match.group(1)}]"
                else:
                    plan_str = response.strip()

            plan = json.loads(plan_str.replace("'", '"'))
            if isinstance(plan, list) and len(plan) > 0:
                return plan

        except Exception as e:
            logger.warning("Plan generation failed, using default plan", extra={"error": str(e)})

        # Fallback: generate a simple generic plan
        return self._generate_fallback_plan(question)

    def _generate_fallback_plan(self, question: str) -> list[str]:
        """Generate a simple comparison plan when LLM planning fails."""
        # Try to extract the two subjects being compared
        subjects = self._extract_comparison_subjects(question)
        if len(subjects) >= 2:
            return [
                f"详细分析{subject}的特点和优势"
                for subject in subjects
            ] + [
                f"从多个维度对比{'和'.join(subjects)}的差异",
                f"分析{'和'.join(subjects)}各自的最佳适用场景",
                "给出综合结论和选择建议",
            ]
        return [
            "分析第一个被比较对象的特点",
            "分析第二个被比较对象的特点",
            "从多个维度进行对比分析",
            "分析各自的适用场景",
            "给出综合结论和建议",
        ]

    def _extract_comparison_subjects(self, question: str) -> list[str]:
        """Try to extract the subjects being compared from the question."""
        # Common comparison patterns in Chinese
        for pattern in [
            r"(.+?)和(.+?)(?:的区别|哪个|有什么不同|对比|比较|vs)",
            r"(.+?)与(.+?)(?:的区别|哪个|有什么不同|对比|比较)",
            r"(.+?)\s*vs\.?\s*(.+)",
        ]:
            match = re.search(pattern, question)
            if match:
                return [match.group(1).strip(), match.group(2).strip()]
        return []
