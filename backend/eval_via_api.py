"""
评测系统 v3 — LLM-as-Judge

用法: 后端启动后，python eval_via_api.py
"""
import json, time, requests, os, re
from datetime import datetime

API = "http://localhost:8566/api/v1"

TEST = [
    {"q": "星辰科技的CEO是谁？", "ref": "张明远，前阿里巴巴达摩院高级算法专家"},
    {"q": "星辰科技什么时候成立的？总部在哪？", "ref": "2019年3月，深圳南山科技园"},
    {"q": "公司获得过哪些投资机构的融资？", "ref": "红杉资本和高瓴资本，B轮融资"},
    {"q": "公司有多少员工？研发占比多少？", "ref": "320人，研发占比65%"},
    {"q": "StarService是什么产品？", "ref": "基于大语言模型的智能客服平台，支持多渠道接入"},
    {"q": "StarDoc的OCR识别准确率是多少？", "ref": "中英文99.2%，手写体95.8%"},
    {"q": "StarInsight支持连接哪些数据源？", "ref": "支持MySQL、PostgreSQL、Oracle、ClickHouse等15种数据源"},
    {"q": "StarService支持哪些接入渠道？", "ref": "网页、微信、钉钉、飞书、APP等8个渠道"},
    {"q": "公司迟到的处理规定是什么？", "ref": "每月累计3次以内不扣款，超过3次每次扣50元"},
    {"q": "差旅住宿的报销标准是多少？", "ref": "400元/晚，交通费实报实销"},
    {"q": "年假天数是怎么规定的？", "ref": "入职满1年5天，满3年10天，满5年15天"},
    {"q": "绩效考核的等级和对应奖励是什么？", "ref": "季度考核+年度综合，A级以上1-3个月工资季度奖金"},
    {"q": "StarService单机标准版的硬件要求是什么？", "ref": "CPU 16核，内存32GB，GPU RTX 4090×1，存储500GB SSD"},
    {"q": "StarService集群旗舰版的GPU配置是什么？", "ref": "A100×6"},
    {"q": "安装StarService后如何验证部署成功？", "ref": "执行 curl http://localhost:8080/api/health 验证"},
    {"q": "对话API的响应里包含哪些字段？", "ref": "reply/intent/confidence/sources/handover/latency_ms"},
    {"q": "知识库搜索接口的请求方式是什么？", "ref": "GET /api/v2/knowledge/search?q=xxx"},
    {"q": "意图分类API返回的内容包含什么？", "ref": "primary_intent/confidence/sub_intents/entities"},
    {"q": "对话响应延迟的告警阈值是多少？", "ref": "P99超过300ms告警，超过500ms严重"},
    {"q": "GPU利用率的严重告警阈值是多少？", "ref": "超过95%"},
    {"q": "日志文件保留多少天？", "ref": "每天轮转，保留30天"},
    {"q": "知识检索返回空怎么解决？", "ref": "重建索引，执行starservice-cli rebuild-index"},
    {"q": "紧急回滚的正确步骤是什么？", "ref": "stop→rollback last-stable→health-check→start"},
    {"q": "使用FP16混合精度推理有什么好处？", "ref": "吞吐量提升40%"},
    {"q": "模型量化INT8可以降低多少显存？", "ref": "显存占用降低50%"},
    {"q": "StarService的SLA保障和告警阈值分别是什么？", "ref": "可用性99.9%，延迟告警P99>300ms"},
    {"q": "公司安全认证有哪些？加密标准是什么？", "ref": "ISO 27001、SOC 2 Type II、等保三级，AES-256加密"},
    {"q": "公司的技术栈和网络延迟要求是什么？", "ref": "Kubernetes，API响应延迟<200ms(P99)"},
    {"q": "报销制度中培训报销上限是多少？", "ref": "与工作相关的培训费可报销80%，年度上限5000元"},
    {"q": "银行客服项目客户满意度提升到多少？分块大小建议是多少？", "ref": "满意度82%→93%，分块建议800-1200字符"},
]

JUDGE_PROMPT = """你是严格的评测专家。对照参考答案评估系统回答的质量。

问题: {q}
参考答案: {ref}
系统回答: {ans}

按以下维度打分(1-5分):
1. accuracy(准确性): 回答事实是否与参考答案一致？数值是否精确？
2. completeness(完整性): 是否包含了参考答案中的所有关键信息？
3. sourced(有据可查): 回答是否标注了来源或说明基于文档？

只输出一行JSON，不要写其他任何文字，不要用markdown代码块包裹：
{"accuracy":5,"completeness":5,"sourced":5,"overall":5.0,"comment":"简短评语"}"""


def judge(q: str, ref: str, answer: str, debug: bool = False) -> dict:
    """LLM-as-Judge scoring — uses direct LLM endpoint (no agent pipeline)."""
    prompt = JUDGE_PROMPT.replace("{q}", q).replace("{ref}", ref).replace("{ans}", answer[:2000])
    try:
        resp = requests.post(f"{API}/llm/invoke",
            json={"messages": [{"role": "user", "content": prompt}], "temperature": 0.1},
            headers={"Content-Type": "application/json"}, timeout=60)
        jr = resp.json().get("content", "")
        # Try to extract JSON: first look for markdown code blocks, then bare JSON
        m = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', jr)
        if not m:
            m = re.search(r'\{[\s\S]*?\}', jr)  # Non-greedy to get first JSON object
        if m:
            raw = m.group(1) if m.lastindex else m.group()
            # Fix: LLM sometimes copies double-braces from the prompt template
            if raw.startswith("{{"):
                raw = raw[1:-1]
            return json.loads(raw)
        # If regex fails, log the raw response for debugging
        if debug:
            print(f"  [DEBUG] No JSON match. Raw: {jr[:200]}")
    except Exception as e:
        if debug:
            print(f"  [DEBUG] Judge error: {e}")
        pass
    return {"accuracy": 0, "completeness": 0, "sourced": 0, "overall": 0, "comment": "Judge失败"}


def main():
    print("=" * 55)
    print("  Knowledge QA 评测 (LLM Judge)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    r = requests.get(f"{API}/health", timeout=5)
    print(f"  Backend: {r.json()['status']}")

    docs = requests.get(f"{API}/documents").json()
    print(f"  Docs: {docs['count']} ({sum(d.get('chunk_count',0) for d in docs['documents'])} chunks)\n")

    results = []
    total_score = total_time = n = 0
    grades = {"A": 0, "B": 0, "C": 0, "D": 0}

    for i, tc in enumerate(TEST, 1):
        print(f"[{i:2d}/30] {tc['q'][:45]}...", end=" ", flush=True)
        t0 = time.perf_counter()

        resp = requests.post(f"{API}/chat",
            json={"session_id": f"judge_{i}", "message": tc["q"]},
            headers={"Content-Type": "application/json"}, timeout=120)
        ms = int((time.perf_counter() - t0) * 1000)
        answer = resp.json().get("answer", "")

        # Check source citation
        has_src = bool(re.search(r'【来源|📎|⚠️.*知识库|Source', answer))

        # LLM Judge
        j = judge(tc["q"], tc["ref"], answer)
        score = j.get("overall", 0)

        total_score += score
        total_time += ms
        n += 1

        g = "A" if score >= 4.0 else ("B" if score >= 3.0 else ("C" if score >= 2.0 else "D"))
        grades[g] += 1

        print(f"{g} {score:.1f}/5 src={'Y' if has_src else 'N'} {ms}ms  {j.get('comment','')[:60]}")

        results.append({
            "q": tc["q"], "answer": answer[:500],
            "score": score, "grade": g, "has_source": has_src,
            "judge": j, "ms": ms,
        })

    avg = total_score / n
    print(f"\n{'='*55}")
    print(f"  平均评分: {avg:.1f}/5")
    print(f"  平均耗时: {total_time/n:.0f}ms")
    print(f"  等级分布: A(≥4)={grades['A']} B(≥3)={grades['B']} C(≥2)={grades['C']} D(<2)={grades['D']}")
    print(f"  来源标注: {sum(1 for r in results if r['has_source'])}/30")

    for i, r in enumerate(results):
        if r["grade"] in ("C", "D"):
            print(f"  [{r['grade']}] Q{i+1}: {r['q'][:45]} ({r['score']:.1f}) {r['judge'].get('comment','')[:80]}")

    os.makedirs("data", exist_ok=True)
    json.dump({"ts": datetime.now().isoformat(), "summary": {
        "n": n, "avg_score": round(avg, 2), "avg_ms": round(total_time/n, 0),
        "grades": grades, "source_rate": f"{sum(1 for r in results if r['has_source'])}/30",
    }, "results": results}, open("data/eval_judge.json", "w", encoding="utf-8"),
        ensure_ascii=False, indent=2)
    print(f"\n报告: data/eval_judge.json")


if __name__ == "__main__":
    main()
