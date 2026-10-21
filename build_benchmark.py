"""Build a frozen 80-item benchmark from the current project sources."""

import csv
import json
from collections import Counter
from pathlib import Path

from agent import DATA_DIR, load_eval_documents

PROJECT_DIR = Path(__file__).resolve().parent
MODELS = ["Qwen", "Doubao", "Gemini", "ChatGPT"]


def read_csv(name):
    with (DATA_DIR / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def make_item(item_id, dimension, query, expected_route, answer_contains=(), relevant_metadata=(),
              insufficient=False, expected_tools=None):
    expected_tools = expected_tools or {"retrieve": ["retrieve_eval_docs"], "analyze": ["analyze_results"],
                                        "both": ["retrieve_eval_docs", "analyze_results"]}[expected_route]
    return {"id": item_id, "dimension": dimension, "question": query,
            "expected_route": expected_route, "expected_tools": expected_tools,
            "answer_contains": list(answer_contains), "relevant_metadata": list(relevant_metadata),
            "insufficient_evidence": insufficient}


def main():
    errors = read_csv("error_analysis_per_image.csv")
    patterns = read_csv("error_pattern_summary.csv")
    accuracy_rows = read_csv("qwen_doubao_task_accuracy.csv") + read_csv("task_model_accuracy_4x4.csv")
    error_totals = Counter(row["Model"] for row in errors)
    tier_totals = Counter((row["Model"], row["Tier"]) for row in errors)
    accuracy = {}
    for row in accuracy_rows:
        correct = row.get("Correct") or row.get("Correct_Count") or ""
        valid = row.get("Valid") or row.get("Valid_Denominator") or ""
        if not correct or not valid:
            continue
        accuracy[(row["Model"].split()[-1], row["Tier"])] = (
            int(correct), int(valid), float(str(row.get("Accuracy", "0")).rstrip("%")))
    cases = []

    # Retrieval: four rule records and sixteen real workbook character records.
    rule_phrasings = ["请按规则解释 {tier} 要核对的核心问题与判错红线。",
                      "工作簿里 {tier} 的测试目标和判错边界是什么？",
                      "查询 {tier} 规则：主要检查什么，什么情况算失败？",
                      "{tier} 对图像生成结果的判定标准如何定义？"]
    for number, tier in enumerate(("Tier 1", "Tier 2", "Tier 3", "Tier 4"), start=1):
        query = rule_phrasings[number - 1].format(tier=tier)
        cases.append(make_item(f"R{number:02d}", "retrieval", query, "retrieve",
                               relevant_metadata=[{"document_type": "tier_rule", "tier": tier}]))
    checklist = [doc for doc in load_eval_documents() if doc.metadata.get("document_type") == "checklist"]
    selected = [checklist[round(i * (len(checklist) - 1) / 15)] for i in range(16)]
    for number, document in enumerate(selected, start=5):
        character = document.metadata["character"]
        index = document.metadata["index"]
        phrasings = ("请依据清单说明“{character}”的原子特征和判错红线。",
                     "查找角色“{character}”在工作簿中的核对特征。",
                     "角色“{character}”需要保留哪些关键视觉特征？",
                     "“{character}”的图像审核应检查哪些属性？")
        query = phrasings[number % len(phrasings)].format(character=character)
        cases.append(make_item(f"R{number:02d}", "retrieval", query, "retrieve",
                               relevant_metadata=[{"document_type": "checklist", "index": index}]))

    # Structured analysis: counts, per-tier counts, overall accuracy, and dominant patterns.
    for index, model in enumerate(MODELS, start=1):
        cases.append(make_item(f"S{index:02d}", "structured", f"汇总 {model} 的错误案例数。", "analyze",
                               [f"{model} 错误图：{error_totals[model]}"]))
    for index, model in enumerate(MODELS, start=5):
        tier = f"Tier {index - 4}"
        cases.append(make_item(f"S{index:02d}", "structured", f"统计 {model} {tier} 错误样本数量。", "analyze",
                               [f"{model} {tier} 错误数：{tier_totals[(model, tier)]}"]))
    for index, model in enumerate(MODELS, start=9):
        correct = sum(accuracy[(model, f"Tier {tier}")][0] for tier in range(1, 5))
        valid = sum(accuracy[(model, f"Tier {tier}")][1] for tier in range(1, 5))
        rate = correct / valid * 100
        cases.append(make_item(f"S{index:02d}", "structured", f"计算 {model} 跨 Tier 的整体准确率。", "analyze",
                               [f"{model} 总体准确率：{rate:.1f}%", f"{correct}/{valid}"]))
    serial = 13
    for model in MODELS:
        for tier in ("All", f"Tier {(serial % 4) + 1}"):
            row = max((item for item in patterns if item["Model"] == model and item["Tier"] == tier),
                      key=lambda item: int(item["Count"]))
            denominator = error_totals[model] if tier == "All" else tier_totals[(model, tier)]
            expected = f"{model} {('总体' if tier == 'All' else tier)}主要错误类型：{row['Error_Pattern']}，{row['Count']}/{denominator}"
            query = (f"从主要错误类别看，{model} 最常出现什么问题？" if tier == "All" else
                     f"{model} 在 {tier} 占比最高的错误类别和数量是什么？")
            cases.append(make_item(f"S{serial:02d}", "structured", query, "analyze", [expected]))
            serial += 1

    # Hybrid: one analytical question plus one rule retrieval per case.
    for index in range(20):
        model = MODELS[index % len(MODELS)]
        tier = f"Tier {(index % 4) + 1}"
        count = tier_totals[(model, tier)]
        query = f"请给出 {model} {tier} 错误数量，并概述 {tier} 的测试目标和判错规则。"
        cases.append(make_item(f"H{index + 1:02d}", "hybrid", query, "both",
                               [f"{model} {tier} 错误数：{count}"],
                               [{"document_type": "tier_rule", "tier": tier}]))

    # Multi-hop: compare two models, retrieve a tier criterion, and cite both sources.
    for index in range(10):
        model_a, model_b = MODELS[index % 4], MODELS[(index + 1) % 4]
        tier = f"Tier {(index % 4) + 1}"
        count_a, count_b = tier_totals[(model_a, tier)], tier_totals[(model_b, tier)]
        query = (f"比较 {model_a} 和 {model_b} 在 {tier} 的错误数量，列出 {model_a} 与 {model_b} "
                 f"在 {tier} 的主要错误类型，并解释 {tier} 的测试目标。")
        expected = [f"{model_a} {tier} 错误数：{count_a}", f"{model_b} {tier} 错误数：{count_b}"]
        for model in (model_a, model_b):
            row = max((item for item in patterns if item["Model"] == model and item["Tier"] == tier),
                      key=lambda item: int(item["Count"]))
            expected.append(f"{model} {tier}主要错误类型：{row['Error_Pattern']}")
        cases.append(make_item(f"M{index + 1:02d}", "multi_hop", query, "both", expected,
                               [{"document_type": "tier_rule", "tier": tier}],
                               expected_tools=["retrieve_eval_docs", "analyze_results", "analyze_results"]))

    # Robustness: aliases, mixed-language phrasing, and deliberately out-of-scope requests.
    robust = [
        ("通义在 Tier 1 错误样本有多少？", "analyze", [f"Qwen Tier 1 错误数：{tier_totals[('Qwen', 'Tier 1')]}"]),
        ("豆包在 Tier 4 的错误数量是多少？", "analyze", [f"Doubao Tier 4 错误数：{tier_totals[('Doubao', 'Tier 4')]}"]),
        ("How many error images are recorded for Qwen?", "analyze", [f"Qwen 错误图：{error_totals['Qwen']}"]),
        ("ChatGPT Task 2 准确率是多少？", "analyze", [f"ChatGPT Tier 2：{accuracy[('ChatGPT', 'Tier 2')][2]:.1f}%"]),
        ("Gemini Tier 4 的准确率有多少？", "analyze", [f"Gemini Tier 4：{accuracy[('Gemini', 'Tier 4')][2]:.1f}%"]),
        ("请汇总四个模型各自的错误图数量。", "analyze", [f"{model} 错误图：{error_totals[model]}" for model in MODELS]),
        ("Mistral模型有多少错误案例？", "analyze", ["Insufficient evidence"], True),
        ("Tier 9 的图像审核规则是什么？", "retrieve", ["Insufficient evidence"], True),
        ("火星人甲的核对特征是什么？", "retrieve", ["Insufficient evidence"], True),
        ("Qwen overall accuracy in Tier 1 是多少？", "analyze", [f"Qwen Tier 1：{accuracy[('Qwen', 'Tier 1')][2]:.1f}%"]),
    ]
    for index, item in enumerate(robust, start=1):
        query, route, expected = item[:3]
        insufficient = item[3] if len(item) > 3 else False
        cases.append(make_item(f"B{index:02d}", "robustness", query, route, expected,
                               insufficient=insufficient))

    if len(cases) != 80:
        raise RuntimeError(f"expected 80 benchmark cases, got {len(cases)}")
    counts = Counter(item["dimension"] for item in cases)
    output = PROJECT_DIR / "benchmark_questions.json"
    output.write_text(json.dumps({"source_files": list(_source_names()), "dimensions": dict(counts),
                                  "cases": cases}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"written": str(output), "cases": len(cases), "dimensions": counts}, ensure_ascii=False))


def _source_names():
    return ["200角色看图打分工作簿.xlsx", "error_analysis_per_image.csv",
            "error_pattern_summary.csv", "qwen_doubao_task_accuracy.csv", "task_model_accuracy_4x4.csv"]


if __name__ == "__main__":
    main()
