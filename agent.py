from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Literal, TypedDict

from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph

from analysis_tools import _facts, _models_in, analyze_results
from data_store import (DATA_DIR, DB_DIR, EMBEDDING_MODEL, MODEL_DIR, PROJECT_DIR,
                        _REQUEST_METRICS, add_timing,
                        get_vector_store, index_version, load_eval_documents)
from llm_provider import synthesize_grounded
from retrieval import (_bm25_scores, _metadata_matches, _query_metadata_filter, _rrf_rank,
                       retrieve_eval_docs)
from schemas import QueryPlan

RERANKER_MODEL = "BAAI/bge-reranker-base"
RRF_K = 60


def _index_version(documents, chunk_size: int) -> str:
    return index_version(documents, chunk_size)


def _runtime_index_version(chunk_size: int, retrieval_mode: str, use_reranker: bool,
                           top_k: int = 3) -> str:
    base = _index_version(load_eval_documents(), chunk_size)
    value = f"{base}|{EMBEDDING_MODEL}|{chunk_size}|{retrieval_mode}|{RERANKER_MODEL if use_reranker else 'none'}|{top_k}|rrf={RRF_K}"
    import hashlib
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


RETRIEVE_TOOL = StructuredTool.from_function(retrieve_eval_docs)
ANALYZE_TOOL = StructuredTool.from_function(analyze_results)
TOOLS = [RETRIEVE_TOOL, ANALYZE_TOOL]


class AgentState(TypedDict, total=False):
    question: str
    route: Literal["retrieve", "analyze", "both"]
    retrieved: list[dict[str, Any]]
    analysis: str
    answer: str
    top_k: int
    chunk_size: int
    retrieval_mode: str
    use_reranker: bool
    tool_trace: list[str]
    query_plan: dict[str, Any]
    evidence: list[dict[str, Any]]
    llm_time_ms: float
    synthesis_mode: str


def classify_route(question: str) -> Literal["retrieve", "analyze", "both"]:
    q = question.lower()
    doc_terms = ["定义", "规则", "红线", "如何判", "怎么判", "原子特征", "核对特征", "扫描顺序",
                 "测试目标", "核心问题", "判定标准", "判错标准"]
    data_terms = ["多少", "错误数", "错误数量", "错误图", "错误案例", "准确率", "占比", "分布", "最多", "最少",
                  "对比", "比较", "排名", "top", "主要错误", "错误类型", "错误类别", "更多", "更高"]
    has_data = any(term in q for term in data_terms) or (
        "模型" in q and any(term in q for term in ("数量", "总量", "分别", "几个", "更高", "更多"))
    )
    has_doc = any(term in q for term in doc_terms) or bool(
        re.search(r"tier\s*[1-4]\s*(?:的)?(?:是什么|什么是|什么意思|含义)", q, re.I)
    )
    if has_doc and has_data:
        return "both"
    if any(term in q for term in ("主要错误类型", "主要错误原因", "错误类别", "error pattern")) and not has_doc:
        return "analyze"
    if has_data or _models_in(q):
        return "analyze"
    return "retrieve"


def plan_query(question: str) -> QueryPlan:
    """Split multi-clause questions into retrieval and deterministic analysis tasks."""
    clauses = [part.strip(" ，,；;。？? ") for part in
               re.split(r"并且|同时|另外|以及|并|[，,；;。]", question) if part.strip(" ，,；;。？? ")]
    retrieval_queries = [part for part in clauses if classify_route(part) == "retrieve" and any(
        term in part.lower() for term in ("tier", "task", "规则", "红线", "判", "特征", "角色", "工作簿", "测试目标")
    )]
    analysis_tasks = [part for part in clauses if classify_route(part) == "analyze"]
    if len(clauses) > 1:
        selected = [part for part in analysis_tasks if _models_in(part) or
                    re.search(r"tier\s*[1-4]|task\s*[1-4]", part, re.I) or
                    any(term in part for term in ("比较", "汇总", "统计", "计算", "准确率", "多少")) or
                    (any(term in part for term in ("主要错误", "错误类型", "错误类别")) and
                     not re.match(r"^从.{1,20}看", part))]
        analysis_tasks = []
        for part in selected:
            position = clauses.index(part)
            if position and classify_route(clauses[position - 1]) == "analyze" and not _models_in(clauses[position - 1]):
                part = f"{clauses[position - 1]} {part}"
            analysis_tasks.append(part)
    if not retrieval_queries and not analysis_tasks:
        if classify_route(question) == "retrieve":
            retrieval_queries.append(question)
        else:
            analysis_tasks.append(question)
    if not retrieval_queries and classify_route(question) == "both":
        retrieval_queries.append(question)
    if not analysis_tasks and classify_route(question) == "both":
        analysis_tasks.append(question)
    route: Literal["retrieve", "analyze", "both"] = (
        "both" if retrieval_queries and analysis_tasks else "retrieve" if retrieval_queries else "analyze"
    )
    tier_values = list(dict.fromkeys(re.findall(r"tier\s*([1-4])", question, re.I)))
    if route == "both" and len(tier_values) == 1:
        analysis_tasks = [task if re.search(r"tier\s*[1-4]", task, re.I) else
                          f"Tier {tier_values[0]} {task}" for task in analysis_tasks]
    if retrieval_queries:
        retrieval_queries = ["；".join(dict.fromkeys(retrieval_queries))]
    return QueryPlan(route=route, retrieval_queries=list(dict.fromkeys(retrieval_queries)),
                     analysis_tasks=list(dict.fromkeys(analysis_tasks)))


def _route_node(state: AgentState) -> dict[str, Any]:
    plan = plan_query(state["question"])
    return {"route": plan.route, "query_plan": plan.model_dump()}


def _retrieve_node(state: AgentState) -> dict[str, Any]:
    plan = state.get("query_plan", {})
    queries = plan.get("retrieval_queries") or [state["question"]]
    results, trace = [], list(state.get("tool_trace", []))
    for query in queries:
        started = time.perf_counter()
        results.extend(RETRIEVE_TOOL.invoke({"query": query, "top_k": state.get("top_k", 3),
            "chunk_size": state.get("chunk_size", 500), "retrieval_mode": state.get("retrieval_mode", "dense"),
            "use_reranker": state.get("use_reranker", False)}))
        add_timing("tool_ms", (time.perf_counter() - started) * 1000)
        trace.append(RETRIEVE_TOOL.name)
    deduped = list({(item["metadata"].get("source"), item["content"]): item for item in results}.values())
    return {"retrieved": deduped, "tool_trace": trace}


def _analyze_node(state: AgentState) -> dict[str, Any]:
    plan = state.get("query_plan", {})
    tasks = plan.get("analysis_tasks") or [state["question"]]
    results, trace = [], list(state.get("tool_trace", []))
    for task in tasks:
        started = time.perf_counter()
        results.append(ANALYZE_TOOL.invoke({"query": task}))
        add_timing("tool_ms", (time.perf_counter() - started) * 1000)
        trace.append(ANALYZE_TOOL.name)
    return {"analysis": "\n".join(results), "tool_trace": trace}


def _synthesize_node(state: AgentState) -> dict[str, Any]:
    parts, evidence = [], []
    if state.get("analysis"):
        parts.append(state["analysis"])
        if not state["analysis"].lower().startswith("insufficient evidence:"):
            evidence.append({"source": "structured_analysis", "content": state["analysis"],
                             "score": None, "metadata": {"tool": ANALYZE_TOOL.name}})
    if state.get("retrieved"):
        excerpts = []
        for item in state["retrieved"]:
            content, metadata = item["content"].strip(), item.get("metadata", {})
            label = " / ".join(str(metadata[key]) for key in ("tier", "section", "source") if metadata.get(key))
            excerpt = content if len(content) <= 650 else content[:650] + "…"
            excerpts.append(f"[{label}]\n{excerpt}" if label else excerpt)
            evidence.append({"source": metadata.get("source", "unknown"), "content": content,
                             "score": item.get("reranker_score", item.get("score")), "metadata": metadata})
        parts.append("检索依据：\n" + "\n---\n".join(excerpts))
    deterministic_answer = "\n\n".join(parts)
    if not evidence:
        deterministic_answer = next((part for part in parts if part.lower().startswith("insufficient evidence:")),
            "Insufficient evidence: 当前项目数据中没有足够依据回答该问题。")
    llm_answer, llm_time, synthesis_mode = synthesize_grounded(
        state["question"], evidence, state.get("tool_trace", []))
    add_timing("llm_ms", llm_time)
    return {"answer": llm_answer or deterministic_answer, "evidence": evidence,
            "llm_time_ms": round(llm_time, 2), "synthesis_mode": synthesis_mode}


def build_agent():
    builder = StateGraph(AgentState)
    builder.add_node("route", _route_node)
    builder.add_node("retrieve", _retrieve_node)
    builder.add_node("analyze", _analyze_node)
    builder.add_node("synthesize", _synthesize_node)
    builder.add_edge(START, "route")
    builder.add_conditional_edges("route", lambda state: state["route"], {
        "retrieve": "retrieve", "analyze": "analyze", "both": "retrieve"})
    builder.add_conditional_edges("retrieve", lambda state: "analyze" if state["route"] == "both" else "synthesize",
        {"analyze": "analyze", "synthesize": "synthesize"})
    builder.add_edge("analyze", "synthesize")
    builder.add_edge("synthesize", END)
    return builder.compile()


AGENT = build_agent()


def ask(question: str, top_k: int = 3, chunk_size: int = 500, retrieval_mode: str = "dense",
        use_reranker: bool = False) -> dict[str, Any]:
    if not question.strip():
        raise ValueError("query cannot be blank")
    metrics: dict[str, float] = {}
    token = _REQUEST_METRICS.set(metrics)
    started = time.perf_counter()
    try:
        state = AGENT.invoke({"question": question, "top_k": top_k, "chunk_size": chunk_size,
                              "retrieval_mode": retrieval_mode, "use_reranker": use_reranker})
        state["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        state["sources"] = [item["metadata"] for item in state.get("retrieved", [])]
        state["tools_used"] = list(dict.fromkeys(state.get("tool_trace", [])))
        state["index_version"] = _runtime_index_version(chunk_size, retrieval_mode, use_reranker, top_k)
        state["timings_ms"] = {key: round(metrics.get(key, 0.0), 2) for key in
            ("embedding_ms", "retrieval_ms", "bm25_ms", "rerank_ms", "tool_ms", "llm_ms")}
        state["timings_ms"]["total"] = state["latency_ms"]
        state["cache_hit"] = bool(metrics.get("cache_hits", 0))
        _write_request_log({"query": question, "route": state["route"], "tools_used": state["tools_used"],
            "top_k": top_k, "chunk_size": chunk_size, "retrieval_mode": retrieval_mode,
            "use_reranker": use_reranker, "index_version": state["index_version"],
            "cache_hit": state["cache_hit"], "timings_ms": state["timings_ms"], "latency_ms": state["latency_ms"]})
        return state
    except Exception as exc:
        _write_request_log({"query": question, "status": "error", "error_type": type(exc).__name__,
            "retrieval_mode": retrieval_mode, "use_reranker": use_reranker,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2)})
        raise
    finally:
        _REQUEST_METRICS.reset(token)


def _write_request_log(record: dict[str, Any]) -> None:
    log_path = PROJECT_DIR / "logs" / "requests.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("evaluation_rag_agent.requests")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    record["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    logger.info(json.dumps(record, ensure_ascii=False))


def main() -> None:
    print("Multimodal Evaluation RAG Agent（输入 exit 退出）")
    print("首次提问会建立本地 Chroma 索引。")
    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit", "q", "退出"}:
            break
        if not question:
            continue
        result = ask(question)
        print(f"\nAgent [{result['route']}]：\n{result['answer']}")
        print(f"\n耗时：{result['latency_ms']} ms")


if __name__ == "__main__":
    main()
