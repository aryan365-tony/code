from __future__ import annotations

import asyncio
import json
from typing import Any, Iterable

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from tools.base import get_tool, get_tool_entry
import time

from ui import (
    print_tool_call,
    print_tool_result,
)

def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _normalize_tool_call(raw: Any) -> dict[str, Any]:
    """
    Accepts tool-call shapes produced by LangChain streaming / invoke results.
    Returns: {id, name, args}
    """
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported tool call type: {type(raw)!r}")

    call_id = raw.get("id") or raw.get("tool_call_id") or raw.get("call_id") or ""
    name = raw.get("name")

    args = raw.get("args")
    if args is None:
        function = raw.get("function") or {}
        name = name or function.get("name")
        args = function.get("arguments")

    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"_raw": args}
    elif args is None:
        args = {}

    if not isinstance(args, dict):
        args = {"_value": args}

    return {
        "id": str(call_id) if call_id is not None else "",
        "name": str(name) if name is not None else "",
        "args": args,
    }


def extract_tool_calls(message: Any) -> list[dict[str, Any]]:
    """
    Read tool calls from an AIMessage / AIMessageChunk in a version-tolerant way.
    """
    collected: list[dict[str, Any]] = []

    direct = getattr(message, "tool_calls", None)
    if direct:
        for call in direct:
            collected.append(_normalize_tool_call(call))

    additional = getattr(message, "additional_kwargs", None) or {}
    raw_calls = additional.get("tool_calls") or []
    for call in raw_calls:
        collected.append(_normalize_tool_call(call))

    # Deduplicate by (id, name, args)
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for call in collected:
        fingerprint = (
            call.get("id", ""),
            call.get("name", ""),
            json.dumps(call.get("args", {}), sort_keys=True, ensure_ascii=False),
        )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(call)

    return unique


async def _run_single_tool(call: dict[str, Any]) -> ToolMessage:
    name = call["name"]
    tool = get_tool(name)
    tool_id = call.get("id", "")

    if tool is None:
        return ToolMessage(
            content=f"Error: tool '{name}' is not registered.",
            tool_call_id=tool_id or name,
            name=name,
        )

    entry = get_tool_entry(name)
    timeout_s = entry.metadata.timeout_s if entry else 30.0
    args = call.get("args", {})
    print_tool_call(
        name,
        args,
    )

    started = time.perf_counter()

    async def _invoke() -> Any:
        # Use a worker thread so slow file or web operations do not block the event loop.
        return await asyncio.to_thread(tool.invoke, args)

    try:
        result = await asyncio.wait_for(_invoke(), timeout=timeout_s)
        elapsed = (
            time.perf_counter()
            - started
        )

        result_text = _safe_text(result)

        print_tool_result(
            name,
            result_text,
            elapsed,
        )

        return ToolMessage(
            content=result_text,
            tool_call_id=tool_id or name,
            name=name,
        )
        return ToolMessage(
            content=_safe_text(result),
            tool_call_id=tool_id or name,
            name=name,
        )
    except asyncio.TimeoutError:

        msg = (
            f"Error: tool '{name}' "
            f"timed out after {timeout_s} seconds."
        )

        print_tool_result(
            name,
            msg,
        )

        return ToolMessage(
            content=msg,
            tool_call_id=tool_id or name,
            name=name,
        )
        return ToolMessage(
            content=f"Error: tool '{name}' timed out after {timeout_s} seconds.",
            tool_call_id=tool_id or name,
            name=name,
        )
    except Exception as exc:

        msg = (
            f"Error while running tool "
            f"'{name}': {exc}"
        )

        print_tool_result(
            name,
            msg,
        )

        return ToolMessage(
            content=msg,
            tool_call_id=tool_id or name,
            name=name,
        )

async def execute_tool_calls(calls: Iterable[dict[str, Any]]) -> list[ToolMessage]:
    results: list[ToolMessage] = []
    for call in calls:
        results.append(await _run_single_tool(call))
    return results


async def run_tool_loop(
    llm: Any,
    messages: list[BaseMessage],
    *,
    max_iterations: int = 12,
    stream_fn,
) -> tuple[str, str]:
    """
    Drive the model/tool loop until the model returns no tool calls.
    `stream_fn` must stream one model step and return:
        (reasoning_text, answer_text, tool_calls)
    """
    working_messages: list[BaseMessage] = list(messages)
    final_reasoning = ""
    final_answer = ""

    for _ in range(max_iterations):
        reasoning_text, answer_text, tool_calls = await stream_fn(llm, working_messages)

        if tool_calls:
            assistant_message = AIMessage(
                content=answer_text or "",
                additional_kwargs={"tool_calls": tool_calls},
            )
            working_messages.append(assistant_message)

            tool_messages = await execute_tool_calls(tool_calls)
            working_messages.extend(tool_messages)
            continue

        final_reasoning = reasoning_text
        final_answer = answer_text
        return final_reasoning, final_answer

    raise RuntimeError(f"Tool loop exceeded {max_iterations} iterations.")
