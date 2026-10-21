import argparse
import csv
import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path

from agent import get_vector_store, ask

PROJECT_DIR = Path(__file__).resolve().parent
K = 5
CONFIGS = [("dense", False), ("dense", True), ("hybrid", False), ("hybrid", True)]


def matches(metadata, expected):
    return all(metadata.get(key) == value for key, value in expected.items())


def relevant_total(store_metadata, expected_groups):
    ids = set()
    for expected in expected_groups:
        for index, metadata in enumerate(store_metadata):
            if matches(metadata, expected):
                ids.add(index)
    return max(1, len(ids))


def run_benchmark():
    benchmark = json.loads((PROJECT_DIR / "benchmark_questions.json").read_text(encoding="utf-8"))
    metadata = get_vector_store(500).get(include=["metadatas"])["metadatas"]
    summaries, detailed = [], []
    for retrieval_mode, use_reranker in CONFIGS:
        config = f"{retrieval_mode}{'+reranker' if use_reranker else ''}"
        route_hits = tool_hits = answer_hits = task_hits = 0
        retrieval_cases = retrieval_hit = 0
        relevant_found = relevant_total_count = 0
        precision_values, reciprocal_ranks, ndcg_values = [], [], []
        latencies = []
        for case in benchmark["cases"]:
            result = ask(case["question"], top_k=K, chunk_size=500,
                         retrieval_mode=retrieval_mode, use_reranker=use_reranker)
            route_ok = result["route"] == case["expected_route"]
            tool_ok = result.get("tool_trace", []) == case["expected_tools"]
            answer_ok = all(token.lower() in result["answer"].lower()
                            for token in case["answer_contains"])
            if case["insufficient_evidence"]:
                answer_ok = answer_ok and result["answer"].lower().startswith("insufficient evidence:")
            route_hits += int(route_ok)
            tool_hits += int(tool_ok)
            answer_hits += int(answer_ok)
            latencies.append(result["latency_ms"])
            retrieved = result.get("retrieved", [])
            if case["relevant_metadata"]:
                retrieval_cases += 1
                total_relevant = relevant_total(metadata, case["relevant_metadata"])
                relevance = [any(matches(item.get("metadata", {}), expected)
                                 for expected in case["relevant_metadata"]) for item in retrieved[:K]]
                hits = sum(relevance)
                relevant_found += hits
                relevant_total_count += total_relevant
                retrieval_hit += int(hits > 0)
                precision_values.append(hits / K)
                ranks = [position for position, is_relevant in enumerate(relevance, start=1) if is_relevant]
                reciprocal_ranks.append(1 / ranks[0] if ranks else 0.0)
                dcg = sum(1 / math.log2(position + 1) for position, is_relevant in enumerate(relevance, start=1)
                          if is_relevant)
                ideal_hits = min(total_relevant, K)
                idcg = sum(1 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
                ndcg_values.append(dcg / idcg if idcg else 0.0)
                retrieval_ok = hits > 0
            else:
                retrieval_ok = True
            task_hits += int(route_ok and tool_ok and answer_ok and retrieval_ok)
            detailed.append({"config": config, "id": case["id"], "dimension": case["dimension"],
                             "question": case["question"], "route_ok": route_ok, "tool_trace_ok": tool_ok,
                             "answer_fields_ok": answer_ok, "retrieval_ok": retrieval_ok,
                             "tool_trace": result.get("tool_trace", []), "cache_hit": result.get("cache_hit"),
                             "latency_ms": result["latency_ms"], "timings_ms": result.get("timings_ms", {})})
        count = len(benchmark["cases"])
        summaries.append({"configuration": config, "retrieval_cases": retrieval_cases,
                          "recall_at_5": relevant_found / relevant_total_count if relevant_total_count else 0,
                          "precision_at_5": statistics.mean(precision_values) if precision_values else 0,
                          "mrr": statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0,
                          "ndcg_at_5": statistics.mean(ndcg_values) if ndcg_values else 0,
                          "hit_rate": retrieval_hit / retrieval_cases if retrieval_cases else 0,
                          "route_accuracy": route_hits / count, "tool_trace_accuracy": tool_hits / count,
                          "field_accuracy": answer_hits / count, "task_success_rate": task_hits / count,
                          "median_latency_ms": statistics.median(latencies),
                          "p95_latency_ms": sorted(latencies)[max(0, math.ceil(.95 * len(latencies)) - 1)],
                          "first_request_ms": latencies[0],
                          "steady_p95_latency_ms": sorted(latencies[1:])[max(0, math.ceil(.95 * len(latencies[1:])) - 1)],
                          "evidence_coverage": None, "citation_correctness": None})
        print(json.dumps(summaries[-1], ensure_ascii=False))

    run = {"run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
           "benchmark": "benchmark_questions.json", "cases": len(benchmark["cases"]), "results": summaries,
           "details": detailed}
    (PROJECT_DIR / "benchmark_results.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = list(summaries[0])
    with (PROJECT_DIR / "ablation_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    lines = ["# Retrieval Ablation", "", "80 题来自当前工作簿和 CSV，测试 Dense / Hybrid 与 BGE reranker 组合。",
             "检索指标只在具有明确 metadata relevance 的 retrieval/hybrid/multi-hop 样本上计算；答案字段指标为预设字段检查。",
             "Evidence coverage 与 citation correctness 暂不自动打分；当前运行未配置 LLM 时记为 N/A。", "",
             "总请求 P95 包含每组首次模型/索引加载；steady P95 排除该组首个请求，单列首次请求耗时。", "",
             "| Configuration | Recall@5 | Precision@5 | MRR | nDCG@5 | Route Acc. | Tool Trace Acc. | Field Acc. | Task Success | Median ms | Steady P95 ms | First Request ms |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['configuration']} | {row['recall_at_5']:.3f} | {row['precision_at_5']:.3f} | "
                     f"{row['mrr']:.3f} | {row['ndcg_at_5']:.3f} | {row['route_accuracy']:.1%} | "
                     f"{row['tool_trace_accuracy']:.1%} | {row['field_accuracy']:.1%} | "
                     f"{row['task_success_rate']:.1%} | {row['median_latency_ms']:.2f} | "
                     f"{row['steady_p95_latency_ms']:.2f} | {row['first_request_ms']:.2f} |")
    (PROJECT_DIR / "BENCHMARK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-reranker", action="store_true", help="Only run dense and hybrid baselines")
    args = parser.parse_args()
    if args.no_reranker:
        CONFIGS = [("dense", False), ("hybrid", False)]
    run_benchmark()
