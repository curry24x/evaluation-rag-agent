from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException

from agent import (DATA_DIR, DB_DIR, EMBEDDING_MODEL, MODEL_DIR, ask, get_vector_store,
                   load_eval_documents, _runtime_index_version)
from schemas import AskRequest, AskResponse, EvidenceItem

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("evaluation_rag_agent.api")
app = FastAPI(title="Evaluation RAG Agent", version="0.2.0")


@app.get("/health")
def health() -> dict[str, Any]:
    required = ["200角色看图打分工作簿.xlsx", "error_analysis_per_image.csv",
                "error_pattern_summary.csv", "qwen_doubao_task_accuracy.csv",
                "task_model_accuracy_4x4.csv", "error_analysis_summary.md"]
    missing = [name for name in required if not (DATA_DIR / name).is_file()]
    sqlite_files = list(DB_DIR.rglob("chroma.sqlite3")) if DB_DIR.exists() else []
    model_cached = any(MODEL_DIR.glob("models--*bge-small-zh-v1.5*"))
    return {"service": "ok", "chroma": "available" if sqlite_files else "not_initialized",
            "embedding_model": {"name": EMBEDDING_MODEL, "status": "cached" if model_cached else "not_cached"},
            "dataset": "ready" if not missing else "incomplete", "missing_inputs": missing}


@app.get("/stats")
def stats() -> dict[str, Any]:
    try:
        store = get_vector_store(500)
        stored = store.get(include=["metadatas"])
        versions = sorted({item.get("index_version", "") for item in stored["metadatas"] if item.get("index_version")})
        return {"document_count": len(load_eval_documents()), "chunk_count": len(stored["ids"]),
                "index_version": _runtime_index_version(500, "dense", False, 3),
                "persisted_vector_version": versions[-1] if versions else None,
                "embedding_model": EMBEDDING_MODEL, "chunk_size": 500}
    except Exception as exc:
        logger.exception("stats failed")
        raise HTTPException(status_code=503, detail="index unavailable") from exc


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest) -> AskResponse:
    try:
        result = ask(request.query, top_k=request.top_k, chunk_size=request.chunk_size,
                     retrieval_mode=request.retrieval_mode, use_reranker=request.use_reranker)
        evidence = [EvidenceItem(source=str(item.get("source", "unknown")),
                                 content=str(item.get("content", "")), score=item.get("score"),
                                 metadata=item.get("metadata", {})) for item in result.get("evidence", [])]
        return AskResponse(answer=result["answer"], route=result["route"],
                           tools_used=result.get("tools_used", []), evidence=evidence,
                           latency_ms=result["latency_ms"], timings_ms=result.get("timings_ms", {}),
                           retrieval_mode=request.retrieval_mode,
                           synthesis_mode=result.get("synthesis_mode", "deterministic"),
                           index_version=result["index_version"])
    except Exception as exc:
        logger.exception("ask failed")
        raise HTTPException(status_code=503, detail="query processing unavailable") from exc
