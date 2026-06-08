from __future__ import annotations
"""
ReflectionVerifier — answer quality verification using the Reflection pattern.

Implements the Generate → Review → Refine pattern from Chapter 4.
After a handler produces a draft answer, the verifier:
1. Reviews the answer for accuracy, completeness, clarity, and logic
2. Scores each dimension (1-10)
3. If the score is below threshold, generates improvement suggestions
4. Optionally refines the answer based on feedback

Default: 1 iteration (cost-conscious), configurable via Settings.
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass

from pydantic import BaseModel, Field

from hello_agents import HelloAgentsLLM

from app.config import Settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class VerificationResult(BaseModel):
    """Result of answer verification."""
    answer: str = Field(description="The final (possibly refined) answer")
    feedback: str = Field(default="", description="Review feedback")
    score: float = Field(default=0.0, description="Quality score (1-10)")
    was_refined: bool = Field(default=False, description="Whether the answer was refined")
    draft_answer: str = Field(default="", description="Original draft before refinement")


# --- Prompt Templates ---

REVIEW_PROMPT = """你是一个答案质量审核专家。请审核以下针对用户问题的回答。

注意：你**看不到**知识库原文，因此不要判断答案中的事实是否正确。
你只需要判断：答案是否诚实地标注了信息来源。

## 原始问题:
{question}

## 待审核的答案:
{answer}

## 审核维度:
1. **忠实性** (1-10):
   - 答案是否标注了来源引用（如 【来源: xxx】、📎 来自知识库 等）？
   - 如果没有来源标注，是否明确告知用户"知识库中未找到相关信息"（如 ⚠️ 开头）？
   - 评分标准：有明确来源标注 → 8-10；无来源但说明了原因 → 5-7；无来源也无说明 → 1-4
2. **清晰性** (1-10): 表达是否清晰易懂？结构是否合理？格式是否规范？
3. **逻辑连贯性** (1-10): 推理是否连贯？前后是否自相矛盾？
4. **相关性** (1-10): 回答是否切题？有没有答非所问？

## 判定规则:
- 忠实性 < 5 分 → is_acceptable 必须为 false（要求补充来源标注）
- 其他维度不单独决定是否打回，综合评分参考 overall_score
- 不要因为答案"不够详细"或"缺乏通用性"而扣分——只要忠实于来源、表达清晰即可

## 输出格式:
请严格输出以下JSON格式:
```json
{{
    "faithfulness": 8,
    "clarity": 9,
    "logic": 8,
    "relevance": 9,
    "overall_score": 8.5,
    "is_acceptable": true,
    "feedback": "简短评语，如果是负面评价须具体指出问题"
}}
```

请只输出JSON，不要包含其他文字。"""

REFINE_PROMPT = """你是答案优化专家。请根据审核反馈改进以下答案。

## 原始问题:
{question}

## 原始答案:
{original_answer}

## 审核反馈:
{feedback}

## 优化要求:
1. 如果反馈指出缺少来源标注：补充来源引用（注意：只引用答案中已有的信息，不要编造来源）
2. 改善表达清晰度和结构（可调整格式、分段、加粗等）
3. 修正逻辑跳跃或前后矛盾
4. **禁止**：添加答案中原本没有的事实内容
5. **必须**：保留原始答案中已有的所有来源标注（如 【来源: xxx】、📎）

## 严格规则:
- 直接输出优化后的完整答案
- 不要写"好的"、"根据反馈"等开场白
- 不要写"以上是优化后的答案"等结尾语
- 只输出答案本身，使用markdown格式"""



class ReflectionVerifier:
    """
    Answer quality verifier using the Reflection pattern.

    Reviews draft answers and optionally refines them for better quality.
    This is the "quality gate" in the agent pipeline.
    """

    def __init__(self, llm: HelloAgentsLLM, config: Settings):
        """
        Initialize the verifier.

        Args:
            llm: LLM client for review and refinement.
            config: Application settings.
        """
        self.llm = llm
        self.config = config
        self.enabled = config.reflection_enabled
        self.max_iterations = config.max_reflection_iterations

    async def verify(self, question: str, draft_answer: str) -> VerificationResult:
        """
        Verify and optionally refine a draft answer.

        Args:
            question: The original user question.
            draft_answer: The handler's draft answer.

        Returns:
            VerificationResult with the final answer and quality metadata.
        """
        if not self.enabled:
            return VerificationResult(
                answer=draft_answer,
                feedback="Reflection disabled",
                score=0.0,
                was_refined=False,
                draft_answer=draft_answer,
            )

        start_time = time.perf_counter()
        current_answer = draft_answer
        total_feedback = ""
        was_refined = False

        for iteration in range(self.max_iterations):
            # Step 1: Review the current answer
            review_result = await self._review(question, current_answer)

            if review_result is None:
                logger.warning("Review parsing failed, skipping refinement")
                break

            total_feedback = review_result.get("feedback", "")
            score = review_result.get("overall_score", 7.0)
            is_acceptable = review_result.get("is_acceptable", True)

            logger.info(
                "Reflection review",
                extra={
                    "question": question[:80],
                    "iteration": iteration,
                    "score": score,
                    "acceptable": is_acceptable,
                },
            )

            # Stop if the answer is acceptable
            if is_acceptable or score >= 7.0:
                break

            # Step 2: Refine the answer
            if iteration < self.max_iterations:
                try:
                    refined = await self._refine(question, current_answer, total_feedback)
                    if refined:
                        current_answer = refined
                        was_refined = True
                        logger.info("Answer refined", extra={"iteration": iteration})
                except Exception as e:
                    logger.error("Refinement failed", extra={"error": str(e)})
                    break

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info(
            "Verification completed",
            extra={
                "was_refined": was_refined,
                "duration_ms": elapsed_ms,
                "score": review_result.get("overall_score", 0) if review_result else 0,
            },
        )

        return VerificationResult(
            answer=current_answer,
            feedback=total_feedback,
            score=review_result.get("overall_score", 7.0) if review_result else 7.0,
            was_refined=was_refined,
            draft_answer=draft_answer,
        )

    async def _review(self, question: str, answer: str) -> dict | None:
        """Review the answer and return structured feedback."""
        try:
            prompt = REVIEW_PROMPT.format(question=question, answer=answer)
            messages = [{"role": "user", "content": prompt}]
            response = await asyncio.to_thread(self.llm.invoke, messages, temperature=0.2)

            # Parse JSON from response
            return self._parse_review_response(response)

        except Exception as e:
            logger.error("Review LLM call failed", extra={"error": str(e)})
            return None

    async def _refine(self, question: str, original: str, feedback: str) -> str | None:
        """Refine the answer based on feedback."""
        try:
            prompt = REFINE_PROMPT.format(
                question=question,
                original_answer=original,
                feedback=feedback,
            )
            messages = [{"role": "user", "content": prompt}]
            refined = await asyncio.to_thread(self.llm.invoke, messages, temperature=0.4)
            return refined if refined else None
        except Exception as e:
            logger.error("Refine LLM call failed", extra={"error": str(e)})
            return None

    def _parse_review_response(self, response: str) -> dict | None:
        """Parse the JSON review response."""
        try:
            # Try direct JSON parse
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find bare JSON object
        match = re.search(r"\{[^{}]*\"overall_score\"[^{}]*\}", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Heuristic fallback: look for "无需改进" (no improvement needed)
        if "无需改进" in response or "no improvement" in response.lower():
            return {
                "faithfulness": 9,
                "clarity": 9,
                "logic": 9,
                "relevance": 9,
                "overall_score": 9.0,
                "is_acceptable": True,
                "feedback": "答案质量良好，无需改进。",
            }

        logger.warning("Could not parse review response", extra={"response": response[:200]})
        return None
