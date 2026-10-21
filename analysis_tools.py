from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from typing import Any

from data_store import data_signature, read_csv


@lru_cache(maxsize=2)
def _facts_for_signature(signature: tuple[Any, ...]) -> dict[str, Any]:
    errors = read_csv("error_analysis_per_image.csv")
    patterns = read_csv("error_pattern_summary.csv")
    accuracy: dict[tuple[str, str], dict[str, float | int]] = {}
    for row in read_csv("qwen_doubao_task_accuracy.csv"):
        accuracy[(row["Model"], row["Tier"])] = {
            "correct": int(row["Correct"]), "wrong": int(row["Wrong"]),
            "valid": int(row["Valid"]), "accuracy": float(row["Accuracy"].rstrip("%")),
        }
    for row in read_csv("task_model_accuracy_4x4.csv"):
        model = row["Model"].split()[-1]
        if model not in {"Gemini", "ChatGPT"}:
            continue
        accuracy[(model, row["Tier"])] = {
            "correct": int(row["Correct_Count"]), "wrong": int(row["Wrong_Count"]),
            "valid": int(row["Valid_Denominator"]), "accuracy": float(row["Accuracy"].rstrip("%")),
        }
    return {"errors": errors, "patterns": patterns, "accuracy": accuracy}


def _facts() -> dict[str, Any]:
    return _facts_for_signature(data_signature())


_facts.cache_clear = _facts_for_signature.cache_clear


def _models_in(query: str) -> list[str]:
    aliases = {"qwen": "Qwen", "通义": "Qwen", "doubao": "Doubao", "豆包": "Doubao",
               "gemini": "Gemini", "chatgpt": "ChatGPT", "gpt": "ChatGPT"}
    lowered = query.lower()
    found = []
    for alias, model in aliases.items():
        if alias in lowered and model not in found:
            found.append(model)
    return found


def analyze_results(query: str) -> str:
    """Analyze source CSVs for error counts, Tier breakdown, accuracy, and patterns."""
    facts = _facts()
    errors, patterns, accuracy = facts["errors"], facts["patterns"], facts["accuracy"]
    models = _models_in(query)
    model_reference = re.search(r"\b([A-Za-z][A-Za-z0-9_-]{1,})\s*(?:模型|model)", query, re.I)
    generic_model_terms = {"四个", "全部", "所有", "哪个", "各个", "模型"}
    aggregate_model_request = any(term in query for term in ("四个模型", "四模型", "所有模型", "全部模型", "各模型"))
    if model_reference and not models and model_reference.group(1).lower() not in generic_model_terms and not aggregate_model_request:
        return f"Insufficient evidence: 数据集中没有模型 {model_reference.group(1)} 的记录。"
    tier_match = re.search(r"tier\s*([1-4])|task\s*([1-4])", query, re.I)
    tier = f"Tier {next(x for x in tier_match.groups() if x)}" if tier_match else None
    error_counts = Counter(row["Model"] for row in errors)
    tier_counts = Counter((row["Model"], row["Tier"]) for row in errors)
    if "准确率" in query or "accuracy" in query.lower() or "正确" in query or "最高" in query and "模型" in query:
        rows = []
        targets = models or ["Qwen", "Doubao", "Gemini", "ChatGPT"]
        if tier:
            for model in targets:
                item = accuracy[(model, tier)]
                rows.append(f"{model} {tier}：{item['accuracy']:.1f}%（{item['correct']}/{item['valid']}）")
        else:
            totals = {}
            for model in targets:
                parts = [accuracy[(model, value)] for value in ("Tier 1", "Tier 2", "Tier 3", "Tier 4")]
                correct, valid = sum(int(x["correct"]) for x in parts), sum(int(x["valid"]) for x in parts)
                totals[model] = (correct, valid, correct / valid * 100)
            for model, (correct, valid, rate) in totals.items():
                rows.append(f"{model} 总体准确率：{rate:.1f}%（{correct}/{valid}）")
            if "最高" in query or "最好" in query:
                best = max(totals, key=lambda model: totals[model][2])
                rows.insert(0, f"总体准确率最高的是 {best}。")
        return "\n".join(rows)
    if "pattern" in query.lower() or "类型" in query or "类别" in query or "主要错误" in query or "错误原因" in query:
        targets = models or ["Qwen", "Doubao", "Gemini", "ChatGPT"]
        output = []
        for model in targets:
            candidates = [row for row in patterns if row["Model"] == model and row["Tier"] == (tier or "All")]
            candidates.sort(key=lambda row: int(row["Count"]), reverse=True)
            top = candidates[0]
            denominator = tier_counts[(model, tier)] if tier else error_counts[model]
            output.append(f"{model} {tier or '总体'}主要错误类型：{top['Error_Pattern']}，{top['Count']}/{denominator}。")
        return "\n".join(output)
    targets = models or ["Qwen", "Doubao", "Gemini", "ChatGPT"]
    if tier:
        output = [f"{model} {tier} 错误数：{tier_counts[(model, tier)]}" for model in targets]
    else:
        output = [f"{model} 错误图：{error_counts[model]}" for model in targets]
        if "最多" in query:
            top = max(error_counts, key=error_counts.get)
            output.insert(0, f"错误图最多的是 {top}（{error_counts[top]} 张）。")
    return "\n".join(output)
