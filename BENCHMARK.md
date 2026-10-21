# Retrieval Ablation

80 题来自当前工作簿和 CSV，测试 Dense / Hybrid 与 BGE reranker 组合。
检索指标只在具有明确 metadata relevance 的 retrieval/hybrid/multi-hop 样本上计算；答案字段指标为预设字段检查。
Evidence coverage 与 citation correctness 暂不自动打分；当前运行未配置 LLM 时记为 N/A。

总请求 P95 包含每组首次模型/索引加载；steady P95 排除该组首个请求，单列首次请求耗时。

| Configuration | Recall@5 | Precision@5 | MRR | nDCG@5 | Route Acc. | Tool Trace Acc. | Field Acc. | Task Success | Median ms | Steady P95 ms | First Request ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 1.000 | 0.200 | 0.957 | 0.968 | 100.0% | 100.0% | 100.0% | 100.0% | 11.38 | 75.63 | 88.53 |
| dense+reranker | 1.000 | 0.200 | 0.987 | 0.990 | 100.0% | 100.0% | 100.0% | 100.0% | 12.84 | 1016.07 | 2559.08 |
| hybrid | 1.000 | 0.200 | 0.943 | 0.958 | 100.0% | 100.0% | 100.0% | 100.0% | 12.84 | 79.78 | 66.90 |
| hybrid+reranker | 1.000 | 0.200 | 0.987 | 0.990 | 100.0% | 100.0% | 100.0% | 100.0% | 11.09 | 1004.22 | 170.57 |
