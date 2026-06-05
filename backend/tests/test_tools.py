"""
Tests for custom tools (Calculator, WebSearch).
"""

import pytest
from app.tools.calculator import CalculatorTool
from app.tools.web_search import WebSearchTool


class TestCalculatorTool:
    """Test the CalculatorTool with various expressions."""

    def setup_method(self):
        self.calc = CalculatorTool()

    def test_basic_arithmetic(self):
        """Test basic arithmetic operations."""
        assert "4" in self.calc.run({"expression": "2 + 2"})
        assert "15" in self.calc.run({"expression": "3 * 5"})
        assert "3.0" in self.calc.run({"expression": "9 / 3"})
        assert "10" in self.calc.run({"expression": "20 - 10"})

    def test_power_and_root(self):
        """Test exponentiation and square root."""
        result = self.calc.run({"expression": "2**10"})
        assert "1024" in result

        result = self.calc.run({"expression": "sqrt(144)"})
        assert "12" in result

    def test_math_functions(self):
        """Test math functions."""
        result = self.calc.run({"expression": "sin(pi/2)"})
        assert "1" in result

        result = self.calc.run({"expression": "log(1)"})
        assert "0" in result

    def test_math_constants(self):
        """Test math constants."""
        result = self.calc.run({"expression": "pi"})
        assert "3.14" in result

    def test_error_handling(self):
        """Test error handling for invalid expressions."""
        result = self.calc.run({"expression": ""})
        assert "错误" in result or "不能为空" in result

        result = self.calc.run({"expression": "1/0"})
        assert "错误" in result or "division" in result.lower()

    def test_unsafe_expressions_rejected(self):
        """Test that unsafe expressions are rejected."""
        # __import__ should not work
        result = self.calc.run({"expression": "__import__('os').system('ls')"})
        assert "错误" in result or "不支持" in result

    def test_long_expression_rejected(self):
        """Test that overly long expressions are rejected."""
        long_expr = "1+" * 300 + "1"
        result = self.calc.run({"expression": long_expr})
        assert "过长" in result


class TestWebSearchTool:
    """Test the WebSearchTool (uses DuckDuckGo as free backend)."""

    def setup_method(self):
        self.search = WebSearchTool()

    def test_basic_search(self):
        """Test a basic search query."""
        result = self.search.run({"query": "Python programming language"})
        assert result  # Should return something
        assert len(result) > 10  # Should have content
        print(f"Search result: {result[:200]}...")

    def test_empty_query(self):
        """Test that empty queries return an error."""
        result = self.search.run({"query": ""})
        assert "错误" in result or "不能为空" in result
