"""
评测脚本 — 检索命中率 + LLM-Judge 回答质量

用法:
  cd backend && python evaluate.py

前提:
  1. 后端 .env 已配置
  2. 已上传企业知识文档到知识库
"""

from __future__ import annotations

import json
import asyncio
import os
import sys
import time
from datetime import datetime

# Auto-add hello_agents to path
_root = os.path.dirname(os.path.abspath(__file__))
_ha_path = os.path.join(_root, "..", "..", "hello-agents", "Co-creation-projects",
                         "lcyting-StockSage-agent", "HelloAgents Optimized")
_ha_path = os.path.abspath(_ha_path)
if os.path.isdir(_ha_path) and _ha_path not in sys.path:
    sys.path.insert(0, _ha_path)

from dotenv import load_dotenv
load_dotenv()

from app.core.agent import AdvancedQAAgent
from app.config import get_settings

# ================================================================
# 测试用例 — 基于企业知识文档
# ================================================================
TEST_CASES = [
    {
        "question": "星辰科技的CEO是谁？",
        "expected_in_chunks": ["张明远"],
        "reference_answer": "张明远，前阿里巴巴达摩院高级算法专家",
        "type": "factual",
    },
    {
        "question": "公司迟到怎么处理？",
        "expected_in_chunks": ["3次以内", "不扣款", "50元"],
        "reference_answer": "每月累计迟到3次以内不扣款，超过3次每次扣50元",
        "type": "factual",
    },
    {
        "question": "差旅住宿标准是多少？",
        "expected_in_chunks": ["400元"],
        "reference_answer": "400元/晚",
        "type": "factual",
    },
    {
        "question": "StarService支持哪些接入渠道？",
        "expected_in_chunks": ["网页", "微信", "钉钉", "飞书"],
        "reference_answer": "支持网页、微信、钉钉、飞书、APP等8个渠道",
        "type": "factual",
    },
    {
        "question": "公司的绩效考核周期是怎样的？A级有什么奖励？",
        "expected_in_chunks": ["季度考核", "年度综合", "1-3个月"],
        "reference_answer": "季度考核+年度综合评定，A级以上可获得1-3个月工资的季度奖金",
        "type": "factual",
    },
    {
        "question": "公司技术栈用了哪些数据库？",
        "expected_in_chunks": ["PostgreSQL", "ClickHouse", "Redis", "Elasticsearch"],
        "reference_answer": "PostgreSQL（业务）、ClickHouse（分析）、Redis（缓存）、Elasticsearch（全文检索）",
        "type": "factual",
    },
    {
        "question": "StarDoc的OCR识别准确率是多少？",
        "expected_in_chunks": ["99.2%", "95.8%"],
        "reference_answer": "中英文99.2%，手写体95.8%",
        "type": "factual",
    },
    {
        "question": "公司有哪些安全认证？",
        "expected_in_chunks": ["ISO 27001", "SOC 2", "等保三级"],
        "reference_answer": "通过ISO 27001、SOC 2 Type II认证，等保三级评测",
        "type": "factual",
    },
    {
        "question": "年假天数是怎么规定的？",
        "expected_in_chunks": ["5天", "10天", "15天", "满1年"],
        "reference_answer": "入职满1年5天，满3年10天，满5年15天",
        "type": "factual",
    },
    {
        "question": "某大型银行智能客服项目的实施效果如何？",
        "expected_in_chunks": ["45%", "82%", "93%", "1200万元"],
        "reference_answer": "减少人工客服工作量45%，客户满意度从82%提升到93%，年节约人力成本约1200万元",
        "type": "factual",
    },
]

# ================================================================
# LLM-as-Judge Prompt
# ================================================================
JUDGE_PROMPT = """你是一个严格的评测专家。评估以下AI回答的质量。

## 问题
{question}

## 参考答案（基于知识文档）
{reference}

## 系统回答
{answer}

## 评分维度（每项 1-5 分）
1. **事实准确性**: 回答中的事实是否与参考答案一致？有无明显错误？
2. **完整性**: 是否包含参考答案中的所有关键信息？
3. **精确性**: 数值是否精确？有无模糊表述？
4. **幻觉检测**: 是否编造了参考答案中不存在的虚假信息？（5=无幻觉，1=严重幻觉）

请输出JSON（不要包含其他文字）:
{{"accuracy": 5, "completeness": 4, "precision": 5, "hallucination": 5, "overall": 4.8, "comments": "简要评语"}}"""


async def evaluate_one(agent: AdvancedQAAgent, tc: dict) -> dict:
    """Evaluate a single test case."""
    question = tc["question"]
    expected = tc["expected_in_chunks"]
    reference = tc["reference_answer"]

    result = {"question": question, "type": tc["type"]}

    # 1. Run the question through the agent
    t0 = time.perf_counter()
    try:
        response = await agent.answer(
            session_id="eval_session",
            question=question,
            enable_reflection=False,  # Skip reflection for faster eval
        )
        result["answer"] = response.answer
        result["category"] = response.trace.category if response.trace else "?"
        result["handler"] = response.trace.handler_used if response.trace else "?"
        result["duration_ms"] = response.trace.total_duration_ms if response.trace else 0
    except Exception as e:
        result["answer"] = f"ERROR: {e}"
        result["category"] = "?"
        result["handler"] = "?"
        result["duration_ms"] = int((time.perf_counter() - t0) * 1000)
        result["retrieval_recall"] = 0.0
        result["judge_score"] = 0.0
        return result

    # 2. Check retrieval recall — did we find the expected text?
    if response.trace and response.trace.steps:
        all_chunks = ""
        for step in response.trace.steps:
            if step.observation:
                all_chunks += step.observation + " "
        hits = sum(1 for kw in expected if kw in all_chunks)
        result["retrieval_recall"] = round(hits / len(expected), 2)
    else:
        result["retrieval_recall"] = 0.0

    # 3. LLM-as-Judge
    try:
        judge_prompt = JUDGE_PROMPT.format(
            question=question,
            reference=reference,
            answer=response.answer[:2000],
        )
        judge_response = agent.llm.invoke(
            [{"role": "user", "content": judge_prompt}],
            temperature=0.1,
        )
        # Parse JSON from judge
        import re as _re
        _match = _re.search(r"\{[^}]+\}", judge_response)
        if _match:
            judge_data = json.loads(_match.group())
            result["judge_score"] = judge_data.get("overall", 0)
            result["judge_detail"] = judge_data
        else:
            result["judge_score"] = 0
            result["judge_detail"] = {}
    except Exception as e:
        result["judge_score"] = 0
        result["judge_detail"] = {"error": str(e)}

    return result


async def main():
    print("=" * 60)
    print("  HelloAgents QA — 评测报告")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    # Init agent
    print("初始化 Agent...")
    t0 = time.perf_counter()
    settings = get_settings()
    settings.reflection_enabled = False  # Disable for eval
    agent = AdvancedQAAgent(settings=settings)
    print(f"Agent 就绪 ({(time.perf_counter()-t0)*1000:.0f}ms)")
    print()

    # Check knowledge base
    docs = agent.list_documents()
    if not docs:
        print("⚠️  知识库为空！请先上传企业知识文档。")
        print("   curl -X POST http://localhost:8566/api/v1/documents/upload -F 'file=@../企业知识文档.txt'")
        return
    print(f"知识库: {len(docs)} 个文档 ({sum(d.get('chunk_count', 0) for d in docs)} 个分块)")
    print()

    # Run evaluations
    print(f"开始评测 {len(TEST_CASES)} 个问题...\n")
    results = []
    total_recall = 0
    total_judge = 0
    total_time = 0

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"[{i}/{len(TEST_CASES)}] {tc['question'][:60]}...", end=" ", flush=True)
        r = await evaluate_one(agent, tc)
        results.append(r)

        recall = r.get("retrieval_recall", 0)
        judge = r.get("judge_score", 0)
        dur = r.get("duration_ms", 0)
        total_recall += recall
        total_judge += judge
        total_time += dur

        status = "✅" if recall >= 0.5 and judge >= 3 else "⚠️" if recall >= 0.3 else "❌"
        print(f"{status} recall={recall:.0%} judge={judge:.1f}/5 {dur}ms")

    # Summary
    n = len(results)
    avg_recall = total_recall / n * 100
    avg_judge = total_judge / n
    avg_time = total_time / n

    print()
    print("=" * 60)
    print("  评测结果")
    print("=" * 60)
    print(f"  总问题数:        {n}")
    print(f"  平均检索召回率:   {avg_recall:.0f}%")
    print(f"  平均 Judge 评分:  {avg_judge:.1f}/5")
    print(f"  平均耗时:         {avg_time:.0f}ms")
    print(f"  总耗时:           {total_time:.0f}ms")
    print()

    # Per-question detail
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r['question'][:60]}")
        print(f"      分类={r['category']}  处理器={r['handler']}  召回={r['retrieval_recall']:.0%}  评分={r['judge_score']:.1f}/5")
        if r.get("judge_detail", {}).get("comments"):
            print(f"      评语: {r['judge_detail']['comments'][:100]}")
        print()

    # Save to file
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": n, "avg_recall_pct": round(avg_recall, 1),
            "avg_judge": round(avg_judge, 2), "avg_time_ms": round(avg_time, 0),
        },
        "results": results,
    }
    report_path = "./data/eval_report.json"
    os.makedirs("./data", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告已保存: {report_path}")

    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
