import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

from agent import ask, get_vector_store

HERE = Path(__file__).resolve().parent


def run_evaluation(configs=None):
    questions = json.loads((HERE / "eval_questions.json").read_text(encoding="utf-8"))
    configs = configs or [(300, 3), (500, 3), (500, 5), (800, 5)]
    results = []
    for chunk_size, top_k in configs:
        index_started = __import__("time").perf_counter()
        get_vector_store(chunk_size)
        index_load_ms = (__import__("time").perf_counter() - index_started) * 1000
        route_hits = tool_trace_hits = retrieval_hits = retrieval_total = answer_check_hits = 0
        latencies = []
        details = []
        for item in questions:
            response = ask(item["question"], top_k=top_k, chunk_size=chunk_size)
            route_ok = response["route"] == item["route"]
            expected_tools = {
                "retrieve": ["retrieve_eval_docs"],
                "analyze": ["analyze_results"],
                "both": ["retrieve_eval_docs", "analyze_results"],
            }[item["route"]]
            tool_trace_ok = response.get("tool_trace") == expected_tools
            answer_ok = all(token.lower() in response["answer"].lower() for token in item["answer_contains"])
            if "source_contains" in item:
                retrieval_total += 1
                source_ok = any(item["source_contains"].lower() in str(source).lower()
                                for source in response.get("sources", []))
                retrieval_hits += int(source_ok)
            else:
                source_ok = None
            route_hits += int(route_ok)
            tool_trace_hits += int(tool_trace_ok)
            answer_check_hits += int(answer_ok)
            latencies.append(response["latency_ms"])
            details.append({"question": item["question"], "route_ok": route_ok,
                            "tool_trace": response.get("tool_trace", []),
                            "tool_trace_ok": tool_trace_ok,
                            "answer_ok": answer_ok, "retrieval_ok": source_ok,
                            "latency_ms": response["latency_ms"]})
        results.append({
            "chunk_size": chunk_size, "top_k": top_k, "questions": len(questions),
            "route_accuracy": route_hits / len(questions),
            "tool_trace_accuracy": tool_trace_hits / len(questions),
            "answer_check_rate": answer_check_hits / len(questions),
            "retrieval_cases": retrieval_total,
            "retrieval_hit_rate": retrieval_hits / retrieval_total,
            "index_load_ms": index_load_ms,
            "avg_latency_ms": statistics.mean(latencies),
            "median_latency_ms": statistics.median(latencies),
            "p95_latency_ms": sorted(latencies)[max(0, int(len(latencies) * .95) - 1)],
            "details": details,
        })
    payload = {"run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
               "dataset": "eval_questions.json", "results": results}
    (HERE / "evaluation_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Evaluation Results", "", "20 个问题全部来自现有评测规则和结构化结果。", "",
             "检索命中率按期望来源元数据判断；答案字段校验通过率按预先定义的关键字段包含关系判断，不是人工语义评分。", "",
             "Tool Trace Accuracy 检查 LangGraph 实际调用的注册工具及顺序是否符合预期。", "",
             "| chunk_size | top_k | Route Accuracy | Tool Trace Accuracy | Retrieval Hit Rate | Answer Check Pass Rate | Median Latency | P95 Latency |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['chunk_size']} | {row['top_k']} | {row['route_accuracy']:.1%} | "
                     f"{row['tool_trace_accuracy']:.1%} | "
                     f"{row['retrieval_hit_rate']:.1%} | {row['answer_check_rate']:.1%} | "
                     f"{row['median_latency_ms']:.1f} ms | {row['p95_latency_ms']:.1f} ms |")
    (HERE / "EVALUATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Only run chunk_size=500, top_k=3")
    args = parser.parse_args()
    data = run_evaluation([(500, 3)] if args.quick else None)
    for row in data["results"]:
        print({key: value for key, value in row.items() if key != "details"})
