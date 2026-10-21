from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from collections import Counter, OrderedDict
from functools import lru_cache
from typing import Any

from fastembed.rerank.cross_encoder import TextCrossEncoder

from data_store import (MODEL_DIR, _REQUEST_METRICS, add_timing, get_vector_store,
                        index_version, load_eval_documents)

RERANKER_MODEL = "BAAI/bge-reranker-base"
RRF_K = 60
_RETRIEVAL_CACHE: OrderedDict[tuple[Any, ...], list[dict[str, Any]]] = OrderedDict()
_RETRIEVAL_CACHE_LIMIT = 128


def clear_retrieval_cache() -> None:
    _RETRIEVAL_CACHE.clear()


def _query_metadata_filter(query: str) -> dict[str, Any] | None:
    clauses: list[dict[str, Any]] = []
    lowered = query.lower()
    model_aliases = {"qwen": ("qwen", "通义"), "doubao": ("doubao", "豆包"),
                     "gemini": ("gemini",), "chatgpt": ("chatgpt", "gpt")}
    for model, aliases in model_aliases.items():
        if any(alias in lowered for alias in aliases):
            clauses.append({"model": {"$eq": model.title() if model != "chatgpt" else "ChatGPT"}})
            break
    tier_match = re.search(r"tier\s*([1-4])|task\s*([1-4])", query, re.I)
    if tier_match:
        tier_num = next(value for value in tier_match.groups() if value)
        clauses.append({"tier": {"$eq": f"Tier {tier_num}"}})
    if any(term in lowered for term in ("原子特征", "核对特征", "核对顺序")):
        clauses.append({"document_type": {"$eq": "checklist"}})
    elif tier_match and any(term in lowered for term in ("规则", "判错", "核心问题", "红线", "测试目标",
                                                          "是什么", "什么是", "含义", "解释", "为什么")):
        clauses.append({"document_type": {"$eq": "tier_rule"}})
    elif not tier_match and "层级" in lowered and any(term in lowered for term in ("重点", "检查", "判定", "红线")):
        clauses.append({"document_type": {"$eq": "tier_rule"}})
    pattern_names = dict.fromkeys(str(doc.metadata.get("error_type")) for doc in load_eval_documents()
                                  if doc.metadata.get("error_type"))
    for pattern in pattern_names:
        if pattern.lower() in lowered:
            clauses.append({"error_type": {"$eq": pattern}})
            break
    character_matches = []
    character_seen = set()
    for document in load_eval_documents():
        character = document.metadata.get("character")
        if not character:
            continue
        character = str(character)
        if character in character_seen:
            continue
        character_seen.add(character)
        unprefixed = re.sub(r"^原创设定[：:]", "", character)
        primary_name = re.split(r"\s*[（(]", unprefixed, maxsplit=1)[0].strip()
        matched_length = 0
        if character.lower() in lowered:
            matched_length = len(character)
        elif primary_name and primary_name not in {"原创设定", "角色", "未知"}:
            if re.search(r"[\u4e00-\u9fff]", primary_name):
                if primary_name.lower() in lowered:
                    matched_length = len(primary_name)
            elif len(primary_name) >= 2 and re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(primary_name)}(?![A-Za-z0-9_])", query, re.I
            ):
                matched_length = len(primary_name)
        if matched_length:
            character_matches.append((matched_length, character))
    if character_matches:
        _, character = max(character_matches, key=lambda item: item[0])
        clauses.append({"character": {"$eq": character}})
    if not any("character" in clause for clause in clauses) and any(
        term in lowered for term in ("原子特征", "核对特征", "角色特征")
    ):
        candidate = re.search(r"([\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9_-]{1,})(?:的)?(?:原子特征|核对特征|角色特征)", query)
        if candidate:
            name = re.sub(r"^(请问|请查|请给我|角色|目标角色)", "", candidate.group(1)).strip()
            if name:
                clauses.append({"character": {"$eq": name}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _lexical_tokens(text: str) -> list[str]:
    tokens = [value.lower() for value in re.findall(r"[A-Za-z0-9_]+", text)]
    for segment in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.extend(segment[i:i + 2] for i in range(len(segment) - 1))
        if len(segment) == 1:
            tokens.append(segment)
    return tokens


def _bm25_scores(query: str, documents: list[dict[str, Any]]) -> list[tuple[int, float]]:
    query_tokens = set(_lexical_tokens(query))
    tokenized = [_lexical_tokens(item["content"]) for item in documents]
    if not tokenized or not query_tokens:
        return []
    lengths = [len(tokens) for tokens in tokenized]
    avg_len = sum(lengths) / len(lengths) or 1.0
    frequencies = [Counter(tokens) for tokens in tokenized]
    df = Counter(token for counts in frequencies for token in counts)
    scored = []
    for index, counts in enumerate(frequencies):
        score = 0.0
        for token in query_tokens:
            tf = counts.get(token, 0)
            if tf:
                idf = math.log(1 + (len(tokenized) - df[token] + 0.5) / (df[token] + 0.5))
                score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * lengths[index] / avg_len))
        scored.append((index, score))
    return sorted((pair for pair in scored if pair[1] > 0), key=lambda pair: pair[1], reverse=True)


def _rrf_rank(dense: list[dict[str, Any]], lexical: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for ranking, field in ((dense, "dense_rank"), (lexical, "bm25_rank")):
        for rank, item in enumerate(ranking, start=1):
            key = item["id"]
            merged.setdefault(key, item.copy())
            merged[key][field] = rank
            merged[key]["rrf_score"] = merged[key].get("rrf_score", 0.0) + 1 / (RRF_K + rank)
    return sorted(merged.values(), key=lambda item: item["rrf_score"], reverse=True)[:limit]


def _document_key(content: str, metadata: dict[str, Any]) -> str:
    packed = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(f"{content}|{packed}".encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _reranker() -> TextCrossEncoder:
    return TextCrossEncoder(model_name=RERANKER_MODEL, cache_dir=str(MODEL_DIR))


def retrieve_eval_docs(query: str, top_k: int = 3, chunk_size: int = 500,
                       retrieval_mode: str = "dense", use_reranker: bool = False) -> list[dict[str, Any]]:
    """Query evaluation rules, atomic features, redlines, and experiment notes."""
    cache_key = (index_version(load_eval_documents(), chunk_size), query, top_k, chunk_size,
                 retrieval_mode, use_reranker, RERANKER_MODEL if use_reranker else None)
    if cache_key in _RETRIEVAL_CACHE:
        _RETRIEVAL_CACHE.move_to_end(cache_key)
        add_timing("cache_hits", 1)
        return copy.deepcopy(_RETRIEVAL_CACHE[cache_key])
    retrieval_started = time.perf_counter()
    filter_value = _query_metadata_filter(query)
    if re.search(r"tier\s*(?:[5-9]|\d{2,})", query, re.I):
        return []
    tier_match = re.search(r"tier\s*([1-4])", query, re.I)
    if filter_value is None and tier_match and any(term in query for term in ("什么", "规则", "判错", "重点", "为什么", "含义")):
        filter_value = {"section": f"Tier {tier_match.group(1)}"}
    elif filter_value is None and any(term in query for term in ("扫描顺序", "拒绝生成", "拒绝如何", "如何计分")):
        filter_value = {"section": "操作规则"}
    if retrieval_mode not in {"dense", "hybrid"}:
        raise ValueError("retrieval_mode must be 'dense' or 'hybrid'")
    fetch_k = max(top_k * 5, 20)
    store = get_vector_store(chunk_size)
    dense_started = time.perf_counter()
    stored = store.get(include=["documents", "metadatas"])
    ids_by_content = {_document_key(content, metadata): doc_id
                      for doc_id, content, metadata in zip(stored["ids"], stored["documents"], stored["metadatas"])}
    dense_pairs = store.similarity_search_with_relevance_scores(query, k=fetch_k, filter=filter_value)
    add_timing("retrieval_ms", (time.perf_counter() - dense_started) * 1000)
    dense = [{"id": ids_by_content.get(_document_key(doc.page_content, doc.metadata), ""),
              "content": doc.page_content, "metadata": doc.metadata, "dense_score": float(score)}
             for doc, score in dense_pairs]
    if retrieval_mode == "dense":
        results = dense[:max(top_k * 5, top_k)]
    else:
        corpus = []
        for index, (doc, metadata) in enumerate(zip(stored["documents"], stored["metadatas"])):
            if filter_value and not _metadata_matches(metadata, filter_value):
                continue
            corpus.append({"id": stored["ids"][index], "content": doc, "metadata": metadata})
        bm25_started = time.perf_counter()
        lexical_rank = _bm25_scores(query, corpus)[:fetch_k]
        add_timing("bm25_ms", (time.perf_counter() - bm25_started) * 1000)
        lexical = [dict(corpus[index], bm25_score=score) for index, score in lexical_rank]
        dense_ids = {pair["id"]: pair for pair in dense}
        dense = [dict(dense_ids.get(item["id"], item), **item) for item in dense]
        results = _rrf_rank(dense, lexical, max(top_k * 5, top_k))
    if use_reranker and results:
        rerank_started = time.perf_counter()
        scores = list(_reranker().rerank(query, [item["content"] for item in results]))
        add_timing("rerank_ms", (time.perf_counter() - rerank_started) * 1000)
        for item, score in zip(results, scores):
            item["reranker_score"] = float(score)
        results.sort(key=lambda item: item["reranker_score"], reverse=True)
    output = [{"content": item["content"], "metadata": item["metadata"],
               "score": round(float(item.get("rrf_score", item.get("dense_score", 0))), 6),
               "dense_score": round(float(item.get("dense_score", 0)), 6),
               "bm25_score": round(float(item.get("bm25_score", 0)), 6),
               **({"reranker_score": round(float(item["reranker_score"]), 6)} if "reranker_score" in item else {})}
              for item in results[:top_k]]
    add_timing("retrieval_pipeline_ms", (time.perf_counter() - retrieval_started) * 1000)
    _RETRIEVAL_CACHE[cache_key] = copy.deepcopy(output)
    _RETRIEVAL_CACHE.move_to_end(cache_key)
    while len(_RETRIEVAL_CACHE) > _RETRIEVAL_CACHE_LIMIT:
        _RETRIEVAL_CACHE.popitem(last=False)
    return output


def _metadata_matches(metadata: dict[str, Any], filter_value: dict[str, Any]) -> bool:
    if "$and" in filter_value:
        return all(_metadata_matches(metadata, clause) for clause in filter_value["$and"])
    key, condition = next(iter(filter_value.items()))
    return metadata.get(key) == condition["$eq"]
