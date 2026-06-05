"""
评测系统 v2 — 五维打分

维度:
  1. 检索召回 (30%): 关键词是否出现在检索片段中
  2. 回答包含 (30%): 关键词是否出现在最终回答中
  3. 来源标注 (10%): 是否标注了信息来源
  4. Judge 评分 (30%): LLM 对照参考答案打分（1-5）
  5. 耗时: 记录但不计入评分

用法: 后端启动后，python eval_via_api.py
"""
import json, time, requests, os, re
from datetime import datetime

API = "http://localhost:8566/api/v1"

TEST = [
    {
        "q": "星辰科技的CEO是谁？",
        "kw": ["张明远"],
        "ref": "张明远，前阿里巴巴达摩院高级算法专家",
    },
    {
        "q": "公司迟到怎么处理？",
        "kw": ["3次以内", "不扣款", "50元"],
        "ref": "每月累计迟到3次以内不扣款，超过3次每次扣50元",
    },
    {
        "q": "差旅住宿标准是多少？",
        "kw": ["400元"],
        "ref": "400元/晚",
    },
    {
        "q": "StarService支持哪些接入渠道？",
        "kw": ["网页", "微信", "钉钉", "飞书"],
        "ref": "支持网页、微信、钉钉、飞书、APP等8个渠道",
    },
    {
        "q": "绩效考核周期和A级奖励是什么？",
        "kw": ["季度考核", "年度综合", "1-3个月"],
        "ref": "季度考核+年度综合评定，A级以上可获得1-3个月工资的季度奖金",
    },
    {
        "q": "公司用了哪些数据库？",
        "kw": ["PostgreSQL", "ClickHouse", "Redis", "Elasticsearch"],
        "ref": "PostgreSQL、ClickHouse、Redis、Elasticsearch",
    },
    {
        "q": "StarDoc的OCR准确率是多少？",
        "kw": ["99.2%", "95.8%"],
        "ref": "中英文99.2%，手写体95.8%",
    },
    {
        "q": "公司有哪些安全认证？",
        "kw": ["ISO 27001", "SOC 2", "等保三级"],
        "ref": "通过ISO 27001、SOC 2 Type II认证，等保三级评测",
    },
    {
        "q": "年假天数怎么规定？",
        "kw": ["5天", "10天", "15天"],
        "ref": "入职满1年5天，满3年10天，满5年15天",
    },
    {
        "q": "某大型银行智能客服项目的实施效果如何？",
        "kw": ["45%", "82%", "93%", "1200万元"],
        "ref": "减少人工客服工作量45%，客户满意度从82%提升到93%，年节约人力成本约1200万元",
    },
]

# 严格 Judge 提示
JUDGE_PROMPT = """你是严格的评测专家。对照参考答案评估系统回答。

问题: {q}
参考答案（来自知识文档）: {ref}
系统回答: {a}

按以下维度打分（1-5分）:
1. 准确性: 回答中的事实是否与参考答案一致？有明显错误或无关内容扣分
2. 完整性: 是否包含参考答案中的所有关键信息？
3. 简洁性: 是否直接回答问题，不包含无关的冗余内容？
4. 可信度: 是否标注了信息来源或说明了知识来源？（如未标注且答案与参考答案不一致，此项为1分）

输出JSON（不要其他文字）:
{{"accuracy":5,"completeness":5,"conciseness":5,"credibility":5,"overall":5.0,"comment":"一句话评语"}}"""


def check_keywords_in_text(keywords: list[str], text: str) -> tuple[int, list[str]]:
    """检查关键词在文本中的命中情况，返回(命中数, 命中词列表)"""
    hits = [kw for kw in keywords if kw in text]
    return len(hits), hits


def run_judge(q: str, ref: str, answer: str) -> dict:
    """使用 LLM Judge 评分，失败返回默认值"""
    try:
        prompt = JUDGE_PROMPT.replace("{q}", q).replace("{ref}", ref).replace("{a}", answer[:2000])
        resp = requests.post(f"{API}/chat",
            json={"session_id": "judge", "message": prompt},
            headers={"Content-Type": "application/json"}, timeout=60)
        jr = resp.json().get("answer", "")
        m = re.search(r'\{[^}]+\}', jr)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    return {"overall": 0, "comment": "Judge 调用失败"}


def main():
    print("=" * 60)
    print("  Knowledge QA — 评测报告 v2")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 健康检查
    try:
        r = requests.get(f"{API}/health", timeout=5)
        print(f"  Backend: {r.json()['status']}")
    except Exception:
        print("  Backend: 未连接！请先启动后端")
        return

    docs = requests.get(f"{API}/documents").json()
    if docs["count"] == 0:
        print("  Knowledge: 空！请先上传知识文档")
        return
    print(f"  Knowledge: {docs['count']} 个文档\n")

    results = []
    for i, tc in enumerate(TEST, 1):
        q, kw_list, ref = tc["q"], tc["kw"], tc["ref"]
        print(f"[{i}/10] {q[:50]}...", end=" ", flush=True)

        # 发送问题
        t0 = time.perf_counter()
        resp = requests.post(f"{API}/chat",
            json={"session_id": f"eval_{i}", "message": q},
            headers={"Content-Type": "application/json"}, timeout=120)
        elapsed = int((time.perf_counter() - t0) * 1000)
        data = resp.json()
        answer = data.get("answer", "")

        # 1. 检索召回 (30%)
        trace = data.get("trace", {})
        retrieval_text = ""
        for step in trace.get("steps", []):
            if step.get("observation"):
                retrieval_text += step["observation"] + " "
        recall_hits, recall_matched = check_keywords_in_text(kw_list, retrieval_text)
        recall_score = recall_hits / len(kw_list) if kw_list else 0

        # 2. 回答包含 (30%)
        answer_hits, answer_matched = check_keywords_in_text(kw_list, answer)
        answer_score = answer_hits / len(kw_list) if kw_list else 0

        # 3. 来源标注 (10%)
        has_source = bool(re.search(r'【来源|📎|⚠️.*知识库|Source', answer))
        source_score = 1.0 if has_source else 0.0

        # 4. Judge 评分 (30%)
        judge = run_judge(q, ref, answer)
        judge_score = judge.get("overall", 0) / 5.0  # 归一化到 0-1

        # 5. 综合评分
        overall = recall_score * 0.30 + answer_score * 0.30 + source_score * 0.10 + judge_score * 0.30

        grade = "A" if overall >= 0.80 else ("B" if overall >= 0.60 else ("C" if overall >= 0.40 else "D"))
        print(f"{grade} overall={overall:.0%} recall={recall_hits}/{len(kw_list)} answer={answer_hits}/{len(kw_list)} src={'✓' if has_source else '✗'} judge={judge.get('overall',0):.1f}/5 {elapsed}ms")

        results.append({
            "question": q,
            "retrieval": {"score": round(recall_score, 2), "hits": recall_hits, "total": len(kw_list), "matched": recall_matched},
            "answer_content": {"score": round(answer_score, 2), "hits": answer_hits, "total": len(kw_list), "matched": answer_matched},
            "source": {"score": source_score, "has_source": has_source},
            "judge": {"score": round(judge_score, 2), "overall": judge.get("overall", 0), "comment": judge.get("comment", "")},
            "overall": round(overall, 3),
            "grade": grade,
            "duration_ms": elapsed,
            "answer_preview": answer[:300],
        })

    # 汇总
    n = len(results)
    avg_overall = sum(r["overall"] for r in results) / n
    avg_recall = sum(r["retrieval"]["score"] for r in results) / n
    avg_answer = sum(r["answer_content"]["score"] for r in results) / n
    avg_judge = sum(r["judge"]["overall"] for r in results) / n
    source_rate = sum(1 for r in results if r["source"]["has_source"]) / n
    avg_time = sum(r["duration_ms"] for r in results) / n

    print(f"\n{'='*60}")
    print(f"  综合评分:  {avg_overall:.0%}  (A≥80% B≥60% C≥40% D<40%)")
    print(f"  检索召回:  {avg_recall:.0%}  (检索片段是否包含关键词)")
    print(f"  回答包含:  {avg_answer:.0%}  (最终回答是否包含关键词)")
    print(f"  来源标注:  {source_rate:.0%}")
    print(f"  Judge 评分: {avg_judge:.1f}/5")
    print(f"  平均耗时:  {avg_time:.0f}ms")
    print()

    # 逐题详情
    for i, r in enumerate(results, 1):
        kw_info = f"{r['retrieval']['hits']}/{r['retrieval']['total']}"
        print(f"  [{r['grade']}] Q{i}: {r['question'][:50]}")
        print(f"       检索:{kw_info} 回答:{r['answer_content']['hits']}/{r['answer_content']['total']} "
              f"来源={'✓' if r['source']['has_source'] else '✗'} 耗时:{r['duration_ms']}ms")
        if r["judge"]["comment"]:
            print(f"       Judge: {r['judge']['comment'][:100]}")
        if r["retrieval"]["hits"] < r["retrieval"]["total"]:
            missing = set(TEST[i-1]["kw"]) - set(r["retrieval"]["matched"])
            print(f"       ⚠️  检索遗漏: {missing}")
        if r["answer_content"]["hits"] < r["answer_content"]["total"]:
            missing_a = set(TEST[i-1]["kw"]) - set(r["answer_content"]["matched"])
            print(f"       ⚠️  回答遗漏: {missing_a}")
        print()

    # 保存报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "n": n, "overall": round(avg_overall, 3),
            "retrieval": round(avg_recall, 2), "answer": round(avg_answer, 2),
            "source_rate": round(source_rate, 2), "judge": round(avg_judge, 2),
            "avg_ms": round(avg_time, 0),
            "grades": {g: sum(1 for r in results if r["grade"] == g) for g in "ABCD"},
        },
        "results": results,
    }
    os.makedirs("data", exist_ok=True)
    with open("data/eval_report_v2.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告: data/eval_report_v2.json")


if __name__ == "__main__":
    main()
