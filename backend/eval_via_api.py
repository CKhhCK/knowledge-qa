"""通过 HTTP API 评测 — 复用已加载的 Qwen3"""
import json, time, requests, os, re

API = "http://localhost:8566/api/v1"
TEST = [
    ("星辰科技的CEO是谁？", ["张明远"], "张明远，前阿里巴巴达摩院高级算法专家"),
    ("公司迟到怎么处理？", ["3次以内","不扣款","50元"], "迟到3次以内不扣款，超过3次每次50元"),
    ("差旅住宿标准是多少？", ["400元"], "400元/晚"),
    ("StarService支持哪些接入渠道？", ["网页","微信","钉钉","飞书"], "网页微信钉钉飞书APP等8渠道"),
    ("绩效考核周期和A级奖励？", ["季度考核","年度综合","1-3个月"], "季度考核+年度综合，A级1-3个月工资奖金"),
    ("公司用了哪些数据库？", ["PostgreSQL","ClickHouse","Redis","Elasticsearch"], "PostgreSQL/ClickHouse/Redis/ES"),
    ("StarDoc的OCR准确率？", ["99.2%","95.8%"], "中英文99.2%，手写体95.8%"),
    ("公司有哪些安全认证？", ["ISO 27001","SOC 2","等保三级"], "ISO27001 SOC2 等保三级"),
    ("年假天数怎么规定？", ["5天","10天","15天","满1年"], "满1年5天，满3年10天，满5年15天"),
    ("银行智能客服项目效果？", ["45%","82%","93%","1200万元"], "减少人工45%，满意度82%→93%，节约1200万"),
]

print(f"{'='*50}")
print(f"  HelloAgents QA 评测 (API 模式)")
print(f"{'='*50}")

# Check backend
r = requests.get(f"{API}/health", timeout=5)
print(f"Backend: {r.json()['status']}")

docs = requests.get(f"{API}/documents").json()
print(f"Knowledge: {docs['count']} docs\n")

total_recall = 0
total_time = 0
results = []

for i, (q, kw, ref) in enumerate(TEST, 1):
    print(f"[{i}/10] {q[:40]}...", end=" ", flush=True)

    # Send question
    t0 = time.perf_counter()
    resp = requests.post(f"{API}/chat",
        json={"session_id": f"eval_{i}", "message": q},
        headers={"Content-Type": "application/json"},
        timeout=120)
    elapsed = (time.perf_counter() - t0) * 1000
    data = resp.json()

    # Check retrieval recall
    trace = data.get("trace", {})
    all_text = ""
    for step in trace.get("steps", []):
        if step.get("observation"):
            all_text += step["observation"] + " "
    recall = sum(1 for k in kw if k in all_text) / len(kw)
    
    # Check source citation
    answer = data.get("answer", "")
    has_source = bool(re.search(r'【来源|📎|⚠️.*知识库|Source', answer))

    total_recall += recall
    total_time += elapsed

    status = "OK" if recall >= 0.5 else ("??" if recall >= 0.3 else "FAIL")
    print(f"{status} recall={recall:.0%} src={'✓' if has_source else '✗'} {elapsed:.0f}ms")

    results.append({
        "question": q, "recall": round(recall,2),
        "has_source": has_source, "duration_ms": int(elapsed),
        "answer_preview": answer[:200],
    })

n = len(TEST)
print(f"\n{'='*50}")
print(f"  检索召回率:  {total_recall/n*100:.0f}%")
print(f"  平均耗时:    {total_time/n:.0f}ms")
print(f"  总耗时:      {total_time:.0f}ms")
for i, r in enumerate(results):
    print(f"  [{i}] recall={r['recall']:.0%} src={'✓' if r['has_source'] else '✗'} {r['duration_ms']}ms")

os.makedirs("data", exist_ok=True)
json.dump({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "summary": {
    "n": n, "avg_recall": round(total_recall/n*100,1),
    "avg_ms": round(total_time/n, 0), "total_ms": round(total_time, 0),
}, "results": results}, open("data/eval_api.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\n报告: data/eval_api.json")
