from __future__ import annotations
"""
Safe mathematical calculator tool.

Uses ast.literal_eval with a restricted namespace for safe expression evaluation.
Never uses Python's built-in eval() with unrestricted access.

Extends the hello_agents Tool base class for seamless integration
with the agent framework.
"""

import ast
import math
import operator
from typing import Any

from hello_agents.tools.base import Tool, ToolParameter

from app.utils.logging import get_logger

logger = get_logger(__name__)

# Allowed binary operators
_ALLOWED_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,  # Unary minus
    ast.UAdd: operator.pos,  # Unary plus
}

# Allowed math functions and constants
_ALLOWED_MATH: dict[str, Any] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
    "gcd": math.gcd,
}


class CalculatorTool(Tool):
    """
    Safe calculator for evaluating mathematical expressions.

    Uses AST-based evaluation with a restricted namespace — only
    allowed operators and math functions are available. This is
    substantially safer than Python's built-in eval().

    Usage in ReAct: Action: calculator[expression="sqrt(144) + 2**10"]
    """

    def __init__(self):
        super().__init__(
            name="calculator",
            description=(
                "安全地计算数学表达式。支持基本运算(+, -, *, /, **)、"
                "数学函数(sqrt, sin, cos, tan, log, exp, abs, round, pow等)和"
                "常量(pi, e)。输入: 数学表达式字符串。"
                "示例: '2 + 3 * 4', 'sqrt(144)', 'sin(pi/2)'"
            ),
        )

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="expression",
                type="string",
                description="要计算的数学表达式，例如 '2+3*4' 或 'sqrt(144)'",
                required=True,
            )
        ]

    def run(self, parameters: dict[str, Any]) -> str:
        """Evaluate a mathematical expression safely."""
        expression = parameters.get("expression") or parameters.get("input", "")

        if not expression.strip():
            return "错误：计算表达式不能为空。"

        # Sanity check: reject overly long expressions
        if len(expression) > 500:
            return "错误：表达式过长（最大500字符）。"

        try:
            # Parse the expression into an AST
            tree = ast.parse(expression.strip(), mode="eval")

            # Evaluate with restricted namespace
            result = self._eval_node(tree.body)

            # Format result
            if isinstance(result, float):
                # Show up to 10 significant digits for floats
                formatted = f"{result:.10g}"
                return f"计算结果: {formatted}"
            else:
                return f"计算结果: {result}"

        except SyntaxError as e:
            return f"错误：表达式语法无效 - {e}"
        except (ValueError, TypeError, ZeroDivisionError) as e:
            return f"错误：计算失败 - {e}"
        except Exception as e:
            logger.warning("Unexpected calculator error", extra={"expression": expression, "error": str(e)})
            return f"错误：无法计算该表达式 - {e}"

    def _eval_node(self, node: ast.AST) -> int | float:
        """Recursively evaluate an AST node with restricted operations."""
        # Number literals
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, (int, float)):
                return value
            raise TypeError(f"不支持的值类型: {type(value).__name__}")

        # Binary operations: a + b, a * b, etc.
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op_type = type(node.op)
            if op_type in _ALLOWED_OPS:
                return _ALLOWED_OPS[op_type](left, right)
            raise ValueError(f"不支持的运算符: {op_type.__name__}")

        # Unary operations: -a, +a
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op_type = type(node.op)
            if op_type in _ALLOWED_OPS:
                return _ALLOWED_OPS[op_type](operand)
            raise ValueError(f"不支持的一元运算符: {op_type.__name__}")

        # Function calls: sqrt(144), sin(pi/2), etc.
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_MATH:
                func = _ALLOWED_MATH[node.func.id]
                args = [self._eval_node(arg) for arg in node.args]
                return func(*args)
            raise ValueError(f"不支持的函数: {getattr(node.func, 'id', 'unknown')}")

        # Named constants: pi, e, tau
        elif isinstance(node, ast.Name):
            if node.id in _ALLOWED_MATH:
                return _ALLOWED_MATH[node.id]
            raise ValueError(f"未定义的变量: {node.id}")

        raise ValueError(f"不支持的表达式类型: {type(node).__name__}")
