"""
llm.py — Qwen3 wrapper for ChatOllama.

Phase 4 additions:
- tool-friendly system prompt
- bind_tools helper for native LangChain tool calling
"""

from __future__ import annotations

import os
from typing import Any, Sequence

from langchain_ollama import ChatOllama
from langchain_core.tools import BaseTool

_THINKING_PARAMS = {
    "temperature": float(os.getenv("TEMPERATURE_THINKING", "0.35")),
    "top_p": float(os.getenv("TOP_P_THINKING", "0.90")),
    "top_k": int(os.getenv("TOP_K_THINKING", "40")),
}

_FAST_PARAMS = {
    "temperature": float(os.getenv("TEMPERATURE_FAST", "0.50")),
    "top_p": float(os.getenv("TOP_P_FAST", "0.85")),
    "top_k": int(os.getenv("TOP_K_FAST", "30")),
}

SYSTEM_PROMPT = """\
You are a helpful, precise personal AI assistant.

Follow these rules:
- Be concise but complete.
- Use the provided memory and context carefully.
- If the user asks for code, return runnable code with minimal explanation unless more detail is requested.
- If unsure, say so rather than guessing.
- Prefer correctness over verbosity.

Tool-use rules:
- Use tools when they clearly improve the answer.
- Do not invent file contents, web results, time values, or code execution results.
- When a tool result is available, trust it over guesswork.
- If a task can be completed with tools, finish it instead of only describing how to do it.
- TOOL USAGE RULES

If a tool can answer the question:

DO NOT describe the tool.
DO NOT discuss using the tool.
DO NOT plan tool usage.

Call the tool immediately.

Bad:
"I should use web_search..."

Good:
<tool call>

After tool results arrive:
answer the user.
"""


def build_system_prompt(extra_context: str = "") -> str:
    base = SYSTEM_PROMPT.strip()
    if extra_context.strip():
        return f"{base}\n\n{extra_context.strip()}"
    return base


def _default_num_thread() -> int:
    cpu_count = os.cpu_count() or 8
    return max(1, cpu_count // 2)


def create_llm(
    thinking: bool = True,
    **overrides: Any,
) -> ChatOllama:
    """
    Create a ChatOllama instance tuned for Qwen3:8B on modest hardware.
    """
    profile = _THINKING_PARAMS if thinking else _FAST_PARAMS

    params: dict[str, Any] = {
        "model": os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "num_ctx": int(os.getenv("NUM_CTX", "8192")),
        "num_predict": int(os.getenv("NUM_PREDICT", "512")),
        "repeat_penalty": float(os.getenv("REPEAT_PENALTY", "1.05")),
        "num_thread": int(os.getenv("NUM_THREAD", str(_default_num_thread()))),
        "temperature": profile["temperature"],
        "top_p": profile["top_p"],
        "top_k": profile["top_k"],
        "reasoning": thinking,
        "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "10m"),
    }

    num_gpu = os.getenv("NUM_GPU")
    if num_gpu is not None and num_gpu.strip():
        params["num_gpu"] = int(num_gpu)

    params.update(overrides)
    return ChatOllama(**params)


def bind_tools(llm: ChatOllama, tools: Sequence[BaseTool]) -> Any:
    """
    Bind LangChain tools to the model.
    Kept as a thin helper so the rest of the app does not depend on the exact
    model wrapper type.
    """
    return llm.bind_tools(list(tools))
