"""Optional OpenAI-compatible synthesis; deterministic application logic stays local."""

from __future__ import annotations

import json
import os
import re
import time
from urllib.request import Request, urlopen
from typing import Any


class BaseLLMProvider:
    def complete(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError


class OpenAICompatibleProvider(BaseLLMProvider):
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 25.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def complete(self, messages: list[dict[str, str]]) -> str:
        endpoint = self.base_url if self.base_url.endswith("/chat/completions") else self.base_url + "/chat/completions"
        payload = json.dumps({"model": self.model, "messages": messages, "temperature": 0},
                             ensure_ascii=False).encode("utf-8")
        request = Request(endpoint, data=payload, headers={
            "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()


def configured_provider() -> BaseLLMProvider | None:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()
    if not any((provider, base_url, api_key, model)):
        return None
    if provider not in {"openai", "openai_compatible"} or not all((base_url, api_key, model)):
        return None
    return OpenAICompatibleProvider(base_url, api_key, model)


def synthesize_grounded(query: str, evidence: list[dict[str, Any]], tools_used: list[str]) -> tuple[str | None, float, str]:
    provider = configured_provider()
    if provider is None:
        return None, 0.0, "deterministic_no_provider"
    if not evidence:
        return "Insufficient evidence: 当前项目数据中没有足够依据回答该问题。", 0.0, "insufficient_evidence"
    context = []
    for index, item in enumerate(evidence, start=1):
        context.append(f"[{index}] 来源={item.get('source', 'unknown')}\n{item.get('content', '')}")
    messages = [
        {"role": "system", "content": (
            "你是项目评测数据助手。只能依据给定证据回答，不得添加证据外的项目事实、数字或推断。"
            "统计结果必须原样保留。重要结论用 [n] 引用对应证据编号。证据不足时只回答："
            "Insufficient evidence: 当前项目数据中没有足够依据回答该问题。"
        )},
        {"role": "user", "content": f"问题：{query}\n已执行工具：{', '.join(tools_used) or 'none'}\n证据：\n" + "\n\n".join(context)},
    ]
    started = time.perf_counter()
    try:
        answer = provider.complete(messages)
        citations = {int(value) for value in re.findall(r"\[(\d+)\]", answer)}
        insufficient = answer.lower().startswith("insufficient evidence:")
        if not insufficient and (not citations or any(value < 1 or value > len(evidence) for value in citations)):
            return None, (time.perf_counter() - started) * 1000, "fallback:invalid_citations"
        return answer, (time.perf_counter() - started) * 1000, "llm"
    except Exception as exc:
        return None, (time.perf_counter() - started) * 1000, f"fallback:{type(exc).__name__}"
