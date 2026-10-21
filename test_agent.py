from agent import (analyze_results, ask, classify_route, get_vector_store, load_eval_documents,
                   retrieve_eval_docs, plan_query, _bm25_scores, _rrf_rank, _query_metadata_filter,
                   _index_version, _runtime_index_version)
from llm_provider import synthesize_grounded


def test_real_documents_loaded():
    docs = load_eval_documents()
    assert len(docs) >= 205
    assert any("Tier 3 评测规则" in doc.page_content for doc in docs)
    assert any("马格纳" in doc.page_content for doc in docs)


def test_real_counts():
    answer = analyze_results("四个模型各有多少错误图？")
    assert "Qwen 错误图：98" in answer
    assert "Doubao 错误图：100" in answer
    assert "Gemini 错误图：108" in answer
    assert "ChatGPT 错误图：64" in answer


def test_routes_and_graph():
    assert classify_route("Tier 2 是什么意思？") == "retrieve"
    assert classify_route("Doubao Tier 3 有多少错误？") == "analyze"
    assert classify_route("Tier 3 是什么，Qwen 在 Tier 3 有多少错误？") == "both"
    result = ask("Doubao Tier 3 有多少错误？")
    assert result["route"] == "analyze"
    assert result["tool_trace"] == ["analyze_results"]
    assert "22" in result["answer"]


def test_mixed_route_invokes_both_registered_tools():
    result = ask("Tier 3 是什么，Qwen 在 Tier 3 有多少错误？")
    assert result["tool_trace"] == ["retrieve_eval_docs", "analyze_results"]


def test_retrieval_answer_exposes_tier_metadata():
    result = ask("哪一层级重点检查元素串位、颜色污染和左右镜像？", top_k=3, chunk_size=300)
    assert "Tier 2" in result["answer"]


def test_bm25_and_reciprocal_rank_fusion():
    corpus = [{"id": "a", "content": "Tier 2 检查元素串位和左右镜像", "metadata": {}},
              {"id": "b", "content": "Tier 3 关注特殊画风和角色特征", "metadata": {}}]
    lexical = _bm25_scores("元素串位", corpus)
    assert lexical[0][0] == 0
    fused = _rrf_rank([dict(corpus[1], dense_score=0.7), dict(corpus[0], dense_score=0.5)],
                      [dict(corpus[0], bm25_score=1.2)], 2)
    assert fused[0]["id"] == "a"
    assert fused[0]["dense_rank"] == 2 and fused[0]["bm25_rank"] == 1


def test_metadata_filter_uses_observed_values():
    assert _query_metadata_filter("Qwen Tier 2 主要错误") == {
        "$and": [{"model": {"$eq": "Qwen"}}, {"tier": {"$eq": "Tier 2"}}]
    }
    assert _query_metadata_filter("不存在的模型") is None


def test_query_plan_decomposes_hybrid_multi_clause_request():
    plan = plan_query("Doubao 和 Qwen 哪个模型错误更多，主要错误类型分别是什么，并解释 Tier 2 的判定规则？")
    assert plan.route == "both"
    assert len(plan.analysis_tasks) == 2
    assert len(plan.retrieval_queries) == 1


def test_llm_provider_requires_valid_citations_and_falls_back(monkeypatch):
    import llm_provider

    class FakeProvider:
        def __init__(self, content):
            self.content = content

        def complete(self, messages):
            return self.content

    evidence = [{"source": "facts.csv", "content": "Qwen error count: 98"}]
    monkeypatch.setattr(llm_provider, "configured_provider", lambda: FakeProvider("Qwen 有 98 个错误 [1]。"))
    answer, _, mode = synthesize_grounded("Qwen 错误数？", evidence, ["analyze_results"])
    assert mode == "llm" and "[1]" in answer
    monkeypatch.setattr(llm_provider, "configured_provider", lambda: FakeProvider("Qwen 有 99 个错误。"))
    answer, _, mode = synthesize_grounded("Qwen 错误数？", evidence, ["analyze_results"])
    assert answer is None and mode == "fallback:invalid_citations"


def test_llm_timeout_falls_back_without_key_leak(monkeypatch):
    import llm_provider

    class TimeoutProvider:
        def complete(self, messages):
            raise TimeoutError("test timeout")

    monkeypatch.setattr(llm_provider, "configured_provider", lambda: TimeoutProvider())
    answer, _, mode = synthesize_grounded("Qwen 错误数", [{"source": "real.csv", "content": "98"}], [])
    assert answer is None and mode == "fallback:TimeoutError"


def test_llm_missing_key_uses_offline_mode(monkeypatch):
    import llm_provider

    for name in ("LLM_PROVIDER", "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    answer, latency, mode = llm_provider.synthesize_grounded("问题", [], [])
    assert answer is None and latency == 0 and mode == "deterministic_no_provider"


def test_unknown_model_and_tier_return_insufficient_evidence():
    assert "Insufficient evidence" in ask("Mistral模型有多少错误？")["answer"]
    assert "Insufficient evidence" in ask("Tier 9 判定规则是什么？")["answer"]


def test_retrieval_cache_reuses_query_result():
    result = ask("Tier 2 元素串位", retrieval_mode="hybrid")
    assert result["timings_ms"]["total"] >= 0
    repeated = ask("Tier 2 元素串位", retrieval_mode="hybrid")
    assert repeated["cache_hit"] is True


def test_fastapi_health_and_ask_fallback():
    from fastapi.testclient import TestClient
    from api import app

    client = TestClient(app)
    assert client.get("/health").status_code == 200
    response = client.post("/ask", json={"query": "ChatGPT 总体准确率是多少？"})
    assert response.status_code == 200
    payload = response.json()
    assert "67.3%" in payload["answer"]
    assert payload["tools_used"] == ["analyze_results"]
    assert payload["evidence"]
    assert payload["index_version"]
    assert client.get("/stats").json()["chunk_count"] > 0
    assert client.post("/ask", json={"query": "   "}).status_code == 422


def test_corrupted_csv_returns_service_error_without_editing_source(monkeypatch):
    import analysis_tools
    from fastapi.testclient import TestClient
    from api import app

    def broken_read(_name):
        raise ValueError("simulated malformed csv")

    analysis_tools._facts.cache_clear()
    monkeypatch.setattr(analysis_tools, "read_csv", broken_read)
    response = TestClient(app).post("/ask", json={"query": "Qwen 有多少错误？"})
    assert response.status_code == 503
    analysis_tools._facts.cache_clear()


def test_index_version_tracks_content_chunk_and_retrieval_config():
    from langchain_core.documents import Document

    first = Document("tier rule", metadata={"tier": "Tier 2"})
    changed = Document("tier rule updated", metadata={"tier": "Tier 2"})
    assert _index_version([first], 500) != _index_version([changed], 500)
    assert _index_version([first], 500) != _index_version([first], 300)
    assert _runtime_index_version(500, "dense", False) != _runtime_index_version(500, "hybrid", False)
    assert _runtime_index_version(500, "dense", False) != _runtime_index_version(500, "dense", True)
    assert _runtime_index_version(500, "dense", False, 3) != _runtime_index_version(500, "dense", False, 5)


def test_dense_and_hybrid_retrieval_modes():
    dense = retrieve_eval_docs("Tier 2 元素串位", top_k=2, retrieval_mode="dense")
    hybrid = retrieve_eval_docs("Tier 2 元素串位", top_k=2, retrieval_mode="hybrid")
    assert len(dense) == len(hybrid) == 2
    assert all("metadata" in item and "dense_score" in item for item in dense)
    assert all("metadata" in item and "bm25_score" in item for item in hybrid)


def test_chroma_reopens_persisted_collection():
    first = get_vector_store(500)
    count = len(first.get(include=[])["ids"])
    get_vector_store.cache_clear()
    reopened = get_vector_store(500)
    assert len(reopened.get(include=[])["ids"]) == count > 0
