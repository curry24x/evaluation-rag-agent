# Evaluation Results

20 个问题全部来自现有评测规则和结构化结果。

检索命中率按期望来源元数据判断；答案字段校验通过率按预先定义的关键字段包含关系判断，不是人工语义评分。

Tool Trace Accuracy 检查 LangGraph 实际调用的注册工具及顺序是否符合预期。

| chunk_size | top_k | Route Accuracy | Tool Trace Accuracy | Retrieval Hit Rate | Answer Check Pass Rate | Median Latency | P95 Latency |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 34.8 ms | 72.6 ms |
| 500 | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 29.6 ms | 70.9 ms |
| 500 | 5 | 100.0% | 100.0% | 100.0% | 100.0% | 29.9 ms | 67.7 ms |
| 800 | 5 | 100.0% | 100.0% | 100.0% | 100.0% | 28.6 ms | 62.4 ms |
