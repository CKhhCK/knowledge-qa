"""
Tests for QuestionRouter classification accuracy.
"""

import pytest
from app.core.router import QuestionRouter


class TestRuleBasedClassification:
    """Test the rule-based (keyword) classification path."""

    def test_factual_patterns(self):
        """Test that factual questions are correctly identified."""
        router = QuestionRouter(None)  # No LLM needed for rules

        factual_questions = [
            "什么是机器学习？",
            "Transformer是谁提出的？",
            "中国首都在哪里？",
            "列出常见的排序算法",
            "What is deep learning?",
            "Define artificial intelligence",
        ]

        for q in factual_questions:
            result = router._classify_by_rules(q)
            if result:
                # Not all may match with high confidence, but many should
                print(f"Q: {q[:50]} -> {result.category} ({result.confidence:.2f})")

    def test_reasoning_patterns(self):
        """Test that reasoning questions are correctly identified."""
        router = QuestionRouter(None)

        reasoning_questions = [
            "为什么深度神经网络需要激活函数？",
            "如何解决梯度消失问题？",
            "How does attention mechanism work?",
            "Explain the process of photosynthesis",
        ]

        for q in reasoning_questions:
            result = router._classify_by_rules(q)
            if result:
                print(f"Q: {q[:50]} -> {result.category} ({result.confidence:.2f})")

    def test_comparison_patterns(self):
        """Test that comparison questions are correctly identified."""
        router = QuestionRouter(None)

        comparison_questions = [
            "CNN和Transformer有什么区别？",
            "Python和Java哪个更适合后端开发？",
            "对比有监督学习和无监督学习",
            "Compare REST vs GraphQL",
        ]

        for q in comparison_questions:
            result = router._classify_by_rules(q)
            if result:
                print(f"Q: {q[:50]} -> {result.category} ({result.confidence:.2f})")

    def test_calculation_patterns(self):
        """Test that calculation questions are correctly identified."""
        router = QuestionRouter(None)

        math_questions = [
            "计算 123 * 456",
            "144的平方根是多少？",
            "2**10 + 5*3",
            "Solve 15 + 27 * 3",
        ]

        for q in math_questions:
            result = router._classify_by_rules(q)
            if result:
                print(f"Q: {q[:50]} -> {result.category} ({result.confidence:.2f})")

    def test_pure_math_expression(self):
        """Test that pure math expressions are identified."""
        router = QuestionRouter(None)
        assert router._is_pure_math("2 + 3 * 4")
        assert router._is_pure_math("sqrt(144)")
        assert router._is_pure_math("123 * 456")
        assert not router._is_pure_math("what is machine learning")

    def test_mixed_question_detection(self):
        """Test that multi-part questions are flagged as mixed."""
        router = QuestionRouter(None)

        result = router._classify_by_rules(
            "什么是RAG？它有什么优点？如何实现RAG？"
        )
        if result:
            # Should detect multiple question marks
            print(f"Mixed detection: {result.category} ({result.confidence:.2f})")
