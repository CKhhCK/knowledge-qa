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

REVIEW_PROMPT = """你是一个严格的答案质量审核专家。请审核以下针对用户问题的回答。

## 原始问题:
{question}

## 待审核的答案:
{answer}

## 审核维度:
1. **事实准确性** (1-10): 答案中的事实是否准确？是否有明显错误或误导信息？
2. **完整性** (1-10): 是否充分回答了问题的所有方面？是否有遗漏？
3. **清晰性** (1-10): 表达是否清晰易懂？结构是否合理？
4. **逻辑连贯性** (1-10): 推理是否连贯？论证是否有跳跃？

## 输出格式:
请严格输出以下JSON格式:
```json
{{
    "accuracy": 8,
    "completeness": 7,
    "clarity": 9,
    "logic": 8,
    "overall_score": 8.0,
    "is_acceptable": true,
    "feedback": "整体质量良好，但在XX方面可以改进..."
}}
```

如果 overall_score >= 7，设置 is_acceptable 为 true。
如果答案有明显问题，在 feedback 中具体指出并提供改进建议。

请只输出JSON，不要包含其他文字。"""

REFINE_PROMPT = """你是一个专业的答案优化专家。请根据审核反馈改进以下答案。

## 原始问题:
{question}

## 原始答案:
{original_answer}

## 审核反馈:
{feedback}

## 优化要求:
1. 修正指出的任何事实错误
2. 补充遗漏的重要信息
3. 改善表达清晰度和结构
4. 确保逻辑连贯、论证完整

请直接输出优化后的完整答案，使用markdown格式。"""


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
                "accuracy": 9,
                "completeness": 9,
                "clarity": 9,
                "logic": 9,
                "overall_score": 9.0,
                "is_acceptable": True,
                "feedback": "答案质量良好，无需改进。",
            }

        logger.warning("Could not parse review response", extra={"response": response[:200]})
        return None
