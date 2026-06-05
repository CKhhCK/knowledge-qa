from __future__ import annotations
"""
CalculationHandler — handles mathematical computation questions.

Uses a Chain-of-Thought approach:
1. Extract the mathematical expression from natural language
2. Execute via CalculatorTool for safe, accurate evaluation
3. Present the result with step-by-step explanation

For simple expressions, the CalculatorTool is invoked directly.
For word problems, the LLM first extracts the expression.
"""

import asyncio
import time

from hello_agents import HelloAgentsLLM, ToolRegistry

from app.config import Settings
from app.core.handlers.base import HandlerInterface, HandlerResult
from app.models.chat import StepInfo, ToolCallRecord
from app.utils.logging import get_logger

logger = get_logger(__name__)

EXTRACTION_PROMPT = """你是一个数学问题分析专家。从以下自然语言问题中提取出数学表达式。

## 要求:
1. 分析问题中的数学关系
2. 提取出可计算的数学表达式（纯数字和运算符形式）
3. 如果问题是纯数学表达式，直接原样输出
4. 输出格式: 只输出数学表达式，不要包含解释文字

## 示例:
问题: "小明有15个苹果，小红给了他3倍，他一共有多少个？"
输出: 15 + 15*3

问题: "计算 144 的平方根"
输出: sqrt(144)

问题: "234 * 567 等于多少"
输出: 234 * 567

问题: "2的10次方是多少"
输出: 2**10

现在请处理以下问题，只输出数学表达式:"""


class CalculationHandler(HandlerInterface):
    """
    Handler for mathematical/calculation questions.

    Uses a two-phase approach:
    1. LLM extracts the mathematical expression from the question
    2. CalculatorTool evaluates the expression safely
    3. Return the result with a step-by-step breakdown
    """

    @property
    def handler_name(self) -> str:
        return "calculation"

    async def handle(self, question: str, context: dict | None = None) -> HandlerResult:
        """Solve a calculation question."""
        start_time = time.perf_counter()
        steps: list[StepInfo] = []
        tool_calls: list[ToolCallRecord] = []

        # Step 1: Check if this is a pure expression
        is_pure = self._is_pure_expression(question)

        if is_pure:
            expression = question.strip()
            steps.append(StepInfo(
                step_number=1,
                thought="识别为纯数学表达式",
                action="直接使用CalculatorTool计算",
            ))
        else:
            # Step 1: Extract expression using LLM
            t0 = time.perf_counter()
            try:
                messages = [
                    {"role": "user", "content": f"{EXTRACTION_PROMPT}\n\n问题: {question}"},
                ]
                expression = await asyncio.to_thread(self.llm.invoke, messages, temperature=0.1)
                expression = expression.strip()
            except Exception as e:
                logger.error("Expression extraction failed", extra={"error": str(e)})
                return HandlerResult(
                    answer=f"无法从问题中提取数学表达式: {str(e)}",
                    steps=steps,
                    tool_calls=tool_calls,
                    total_duration_ms=int((time.perf_counter() - start_time) * 1000),
                )

            steps.append(StepInfo(
                step_number=1,
                thought=f"从问题中提取数学表达式",
                action=f"提取表达式: {expression}",
                observation=f"提取结果: {expression}",
                duration_ms=int((time.perf_counter() - t0) * 1000),
            ))

        # Step 2: Calculate using CalculatorTool
        t1 = time.perf_counter()
        calc_tool = self.tool_registry.get_tool("calculator")

        if calc_tool:
            calc_result = calc_tool.run({"expression": expression})
            success = not calc_result.startswith("错误")
        else:
            calc_result = "错误: Calculator工具未注册"
            success = False

        tool_calls.append(ToolCallRecord(
            tool_name="calculator",
            input_params=expression,
            output_summary=calc_result,
            duration_ms=int((time.perf_counter() - t1) * 1000),
            success=success,
        ))

        steps.append(StepInfo(
            step_number=len(steps) + 1,
            thought=f"计算表达式: {expression}",
            action=f'calculator[expression="{expression}"]',
            observation=calc_result,
            tool_calls=tool_calls.copy(),
            duration_ms=int((time.perf_counter() - t1) * 1000),
        ))

        # Step 3: Format the final answer
        answer = self._format_answer(question, expression, calc_result, is_pure)

        total_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info(
            "CalculationHandler completed",
            extra={"question": question[:80], "expression": expression, "duration_ms": total_ms},
        )

        return HandlerResult(
            answer=answer,
            steps=steps,
            tool_calls=tool_calls,
            total_duration_ms=total_ms,
        )

    def _is_pure_expression(self, text: str) -> bool:
        """Check if the input is already a pure mathematical expression."""
        stripped = text.strip()
        # Must contain at least one digit
        if not any(c.isdigit() for c in stripped):
            return False
        # Check character composition
        math_chars = set("0123456789+-*/().^ sqrtpenilgfabcdxyz%")
        allowed_words = ["sqrt", "sin", "cos", "tan", "log", "pi", "abs",
                         "ceil", "floor", "exp", "pow", "min", "max", "sum"]
        # If very short and mostly operators/numbers
        if len(stripped) < 60:
            non_math = [c for c in stripped.lower() if c not in math_chars and not c.isspace()]
            return len(non_math) <= 5
        return False

    def _format_answer(
        self, question: str, expression: str, result: str, is_pure: bool
    ) -> str:
        """Format the final answer with explanation."""
        if is_pure:
            return f"## 计算结果\n\n**表达式**: `{expression}`\n\n**结果**: {result}"
        else:
            return (
                f"## 计算过程\n\n"
                f"**原始问题**: {question}\n\n"
                f"**提取的表达式**: `{expression}`\n\n"
                f"**计算结果**: {result}\n\n"
                f"---\n*计算由 CalculatorTool 安全执行*"
            )
