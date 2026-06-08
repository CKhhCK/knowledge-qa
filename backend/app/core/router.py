from __future__ import annotations
"""
QuestionRouter — classifies incoming questions to select the optimal handler.

Uses a two-pass approach:
1. Rule-based pre-filter (fast, no LLM cost) — handles ~60% of cases
2. LLM-based classification (accurate) — fallback for ambiguous questions

Categories:
- factual: Knowledge lookup ("What is", "Who", "When", "Where")
- reasoning: Multi-step reasoning ("Why", "How", "Explain")
- comparison: Comparative analysis ("Compare", "vs", "Difference")
- calculation: Mathematical computation (numbers + operators)
- mixed: Complex multi-part questions
"""

import asyncio
import json
import re
from typing import Optional

from pydantic import BaseModel, Field

from hello_agents import HelloAgentsLLM

from app.config import Settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ClassificationResult(BaseModel):
    """Result of question classification."""
    category: str = Field(
        description="One of: factual, reasoning, comparison, calculation, mixed"
    )
    complexity: str = Field(
        default="simple",
        description="simple (direct lookup) or complex (multi-step reasoning needed)"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Classification confidence"
    )
    reasoning: str = Field(
        default="", description="Brief explanation of the classification"
    )
    sub_questions: list[str] = Field(
        default_factory=list,
        description="Decomposed sub-questions (for mixed type)",
    )


# --- LLM Classification Prompt ---

ROUTER_SYSTEM_PROMPT = """你是一个问题分类专家。你的任务是分析用户的问题，并将其归类为以下五种类型之一。

## 分类标准:

1. **factual** (事实性/知识性)
   - 询问定义、事实、概念
   - 关键词: "什么是"、"谁"、"何时"、"哪里"、"定义"
   - 例: "什么是机器学习？" "Transformer是谁提出的？"

2. **reasoning** (推理性/分析性)
   - 需要多步推理、因果分析、原理解释
   - 关键词: "为什么"、"如何"、"怎么"、"原因"、"原理"
   - 例: "为什么深度神经网络需要激活函数？" "如何处理数据不平衡问题？"

3. **comparison** (对比性/比较性)
   - 比较两个或多个事物的异同、优劣
   - 关键词: "对比"、"区别"、"比较"、"不同"、"vs"、"哪个更好"
   - 例: "CNN和Transformer有什么区别？" "Python和Java哪个更适合后端开发？"

4. **calculation** (计算性/数学性)
   - 涉及数学运算、公式计算，需要实际执行加减乘除
   - 关键词: 包含明确的数字和运算符、"计算"、"求解"、"等于多少"
   - **重要**: 如果问题询问某个产品参数/配置的数值（如"降低多少显存"、"提升多少吞吐量"），
     这属于 factual（查文档），不是 calculation（算数学）
   - 例: "计算 123 * 456" "144的平方根是多少？"
   - 反例: "模型量化INT8可以降低多少显存？" → factual（查文档参数，不是算数学）

5. **mixed** (混合性/复合性)
   - 包含多个并列的子问题，即使用"和"、"以及"、"还有"等连接词将多个独立问题合并
   - 特征: 问题较长，包含多个问号，或用"和/以及/还有/并"连接了多个独立话题
   - 例1: "什么是RAG？它和传统检索有什么区别？如何实现一个基础的RAG系统？"
   - 例2: "公司的技术栈和网络延迟要求是什么？"（两个独立问题用"和"连接）
   - 例3: "公司安全认证有哪些？加密标准是什么？"（多个问句）

## 输出格式:
请严格输出以下JSON格式:
```json
{
  "category": "分类名称",
  "confidence": 0.85,
  "reasoning": "分类理由简述"
}
```

请只输出JSON，不要包含其他文字。"""


class QuestionRouter:
    """
    Routes questions to the appropriate handler based on classification.

    Uses rule-based heuristics first (zero LLM cost, fast) and falls back
    to LLM-based classification for ambiguous questions.
    """

    # --- Rule-based patterns ---
    FACTUAL_PATTERNS = [
        # "是什么" / "什么是" — most common factual pattern
        r"是什么", r"什么是",
        # "是谁" / "何时" / "在哪里"
        r"是谁", r"何时", r"什么时候", r"在哪里", r"在哪",
        # Numeric/standard lookups (these are doc lookups, not calculations)
        r"是多少", r"有多少", r"多少天", r"多少.*标准",
        r"降低多少", r"提升多少", r"减少多少", r"占比多少",
        r"多少$",  # "XX多少" — factual lookup ending with 多少
        # Listing/enumeration
        r"有哪些", r"支持哪些", r"包含什么", r"列出", r"定义",
        # Policy/rule questions
        r"怎么规定", r"怎么处理", r"怎么报销", r"怎么请假",
        r"怎么.*解决", r"标准是", r"有没有", r"是否",
        # Compound indicators (will be overridden by mixed detection)
        r"告诉我关于",
        # English patterns
        r"^what is", r"^who is", r"^where is", r"^when (did|was)",
        r"^define", r"^list", r"^tell me about",
    ]

    REASONING_PATTERNS = [
        # Causal / explanatory
        r"为什么", r"为何", r"原因", r"原理",
        # How-to (with context, not bare "怎么")
        r"如何", r"怎样", r"怎么解决", r"怎么实现", r"怎么做",
        r"怎么处理", r"怎么优化", r"怎么配置", r"怎么部署",
        # Process / steps
        r"机制", r"过程", r"步骤",
        # English patterns
        r"^why", r"^how", r"^explain", r"^describe the process",
    ]

    COMPARISON_PATTERNS = [
        r"对比", r"比较", r"区别", r"不同", r"差异",
        r"哪个更", r"哪个好", r"哪个比较好", r"优缺点", r"优劣",
        r"相比", r"有何不同", r"有什么不同",
        r"vs\.?", r" versus ",
        r"^compare", r"^what.*difference", r"^which.*better",
        r"和.*区别", r"和.*不同", r"与.*对比", r"与.*区别",
    ]

    CALCULATION_PATTERNS = [
        r"计算", r"求解", r"等于多少", r"算一下",
        r"多少\s*[\+\-\*\/]\s*多少", r"的平方根", r"的阶乘",
        r"^\d+[\s\d\+\-\*\/\(\)\^\.]+$",  # Pure math expression
        r"^calculate", r"^compute", r"^solve",
        r"^what is \d+",
    ]

    # Compound question connectors (suggests multiple sub-questions)
    # Match patterns like: "A和B是什么" / "A以及B有哪些" / "A、B和C的X"
    COMPOUND_CONNECTORS = [
        # Connector with question word somewhere after
        r"[和以及、还有并与]\s*\S+\s*(?:是|有|包含|支持|要求|规定|属于|需要).*(?:什么|哪些|多少|怎么|如何|怎样|是什么|是多少)",
        # Two question words connected by "和/以及/还有"
        r"(?:什么|哪些|多少|怎么|如何|是怎样).*[和以及还有并].*(?:什么|哪些|多少|怎么|如何|是怎样)",
        # Simple connector detection for longer questions
        r"^.{10,}[和以及还有并与].{5,}(?:什么|哪些|多少|怎么|如何|是谁|是什么|是多少)",
    ]

    def __init__(self, llm: HelloAgentsLLM):
        """Initialize the router with an LLM client for fallback classification."""
        self.llm = llm
        self._compiled_factual = [re.compile(p, re.IGNORECASE) for p in self.FACTUAL_PATTERNS]
        self._compiled_reasoning = [re.compile(p, re.IGNORECASE) for p in self.REASONING_PATTERNS]
        self._compiled_comparison = [re.compile(p, re.IGNORECASE) for p in self.COMPARISON_PATTERNS]
        self._compiled_calculation = [re.compile(p, re.IGNORECASE) for p in self.CALCULATION_PATTERNS]
        self._compiled_compound = [re.compile(p, re.IGNORECASE) for p in self.COMPOUND_CONNECTORS]
        # Cache: question → ClassificationResult (avoids repeated LLM calls for similar questions)
        self._cache: dict[str, ClassificationResult] = {}
        self._cache_max = 200

    async def classify(self, question: str) -> ClassificationResult:
        # Check cache first (OPTIMIZATION #4)
        cache_key = question.strip().lower()
        if cache_key in self._cache:
            logger.debug("Classification cache hit")
            return self._cache[cache_key]
        """
        Classify a question into a category.

        Uses rule-based matching first; falls back to LLM if confidence is low.
        """
        # Step 1: Rule-based classification
        rule_result = self._classify_by_rules(question)
        if rule_result and rule_result.confidence >= 0.8:
            logger.info(
                "Question classified by rules",
                extra={"question": question[:100], "category": rule_result.category},
            )
            self._cache_result(cache_key, rule_result)
            return rule_result

        # Step 2: LLM-based classification
        logger.info("Falling back to LLM classification", extra={"question": question[:100]})
        llm_result = await self._classify_by_llm(question)
        self._cache_result(cache_key, llm_result)
        return llm_result

    def _cache_result(self, key: str, result: ClassificationResult) -> None:
        """Store classification result in cache, evicting oldest if full."""
        if len(self._cache) >= self._cache_max:
            # Remove oldest entry (first key)
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = result

    def _classify_by_rules(self, question: str) -> ClassificationResult | None:
        """Attempt rule-based classification. Returns None if uncertain."""
        scores = {
            "factual": self._count_matches(question, self._compiled_factual),
            "reasoning": self._count_matches(question, self._compiled_reasoning),
            "comparison": self._count_matches(question, self._compiled_comparison),
            "calculation": self._count_matches(question, self._compiled_calculation),
        }

        # Check for pure math expressions
        if self._is_pure_math(question):
            return ClassificationResult(
                category="calculation", complexity="simple",
                confidence=0.95, reasoning="纯数学表达式",
            )

        # Check for mixed: multiple question marks
        question_marks = question.count("?") + question.count("？")
        if question_marks >= 2 and len(question) > 15:
            return ClassificationResult(
                category="mixed",
                confidence=0.65,
                reasoning=f"检测到{question_marks}个问句，判定为复合问题",
            )

        # Check for mixed: compound connectors suggesting multiple sub-questions
        compound_score = self._count_matches(question, self._compiled_compound)
        if compound_score > 0 and len(question) > 10:
            # Has compound connectors — likely mixed, but let LLM confirm
            return ClassificationResult(
                category="mixed",
                confidence=0.55,
                reasoning="检测到并列结构，可能包含多个子问题",
            )

        # Find the highest scoring category (factual gets tiebreaker over reasoning)
        max_category = max(scores, key=scores.get)
        max_score = scores[max_category]
        # When factual ties with reasoning, prefer factual (company policy questions)
        if (max_category == "reasoning" and scores.get("factual", 0) == max_score):
            max_category = "factual"

        if max_score == 0:
            return None

        total = sum(scores.values())
        confidence = max_score / total if total > 0 else 0.5
        complexity = "simple" if max_category in ("factual", "calculation") else "complex"

        if confidence >= 0.5:
            return ClassificationResult(
                category=max_category, complexity=complexity,
                confidence=min(confidence, 0.70),
                reasoning=f"关键词匹配: {max_category}类模式得分最高",
            )

        return None

    async def _classify_by_llm(self, question: str) -> ClassificationResult:
        """Use LLM to classify the question."""
        try:
            messages = [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": f"请分类以下问题:\n\n{question}"},
            ]

            response = await asyncio.to_thread(self.llm.invoke, messages, temperature=0.1)

            result = self._parse_llm_response(response)

            logger.info(
                "LLM classification result",
                extra={"question": question[:100], "category": result.category},
            )
            return result

        except Exception as e:
            logger.error("LLM classification failed, using fallback", extra={"error": str(e)})
            return ClassificationResult(
                category="factual", complexity="simple",
                confidence=0.3,
                reasoning=f"LLM分类失败: {str(e)[:100]}",
            )

    def _parse_llm_response(self, response: str) -> ClassificationResult:
        """Parse the LLM's JSON response."""
        # Try to extract JSON from the response
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return ClassificationResult(
                category=data.get("category", "factual"),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
            )

        # Fallback: check if any category name appears in the response
        for cat in ["factual", "reasoning", "comparison", "calculation", "mixed"]:
            if cat in response.lower():
                return ClassificationResult(category=cat, confidence=0.6, reasoning="文本匹配")

        return ClassificationResult(category="factual", confidence=0.3, reasoning="无法解析LLM输出")

    def _count_matches(self, text: str, patterns: list[re.Pattern]) -> int:
        """Count how many regex patterns match the text."""
        return sum(1 for p in patterns if p.search(text))

    def _is_pure_math(self, text: str) -> bool:
        """Check if the text is primarily a mathematical expression."""
        stripped = text.strip()
        # If the text is mostly numbers and operators
        math_chars = set("0123456789+-*/^.() sqrtpienl")
        non_math = [c for c in stripped.lower() if c not in math_chars and not c.isspace()]
        return len(non_math) <= 3 and any(c.isdigit() for c in stripped)
