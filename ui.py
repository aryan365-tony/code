"""
ui.py — Rich terminal UI for Gemma 4 (served via vLLM).

Phase 4 fixes:
- stream reasoning from chunk.additional_kwargs["reasoning_content"]
- keep fallback support for <think>...</think>
- extract tool calls from streamed chunks
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from rich.syntax import Syntax
from rich.panel import Panel
from rich import box
from rich.console import Console
from rich.rule import Rule
from rich.style import Style
from rich.text import Text

console = Console(highlight=False)
SHOW_TOOL_OUTPUT = True
STYLE_THINKING = Style(color="bright_black", italic=True)
STYLE_RESPONSE = Style(color="white")

STYLE_USER_LABEL = Style(color="cyan", bold=True)
STYLE_AI_LABEL = Style(color="green", bold=True)

STYLE_COMMAND = Style(color="yellow")
STYLE_ERROR = Style(color="red", bold=True)
STYLE_DIM = Style(color="bright_black")

STYLE_BANNER = Style(color="bright_cyan", bold=True)

STYLE_MODE_ON = Style(color="magenta", bold=True)
STYLE_MODE_OFF = Style(color="yellow", bold=True)

COMMANDS = {
    "/help": "List commands",
    "/think": "Enable Gemma 4 reasoning",
    "/no_think": "Disable Gemma 4 reasoning",
    "/clear": "Clear history",
    "/undo": "Remove last exchange",
    "/compact": "Summarise history",
    "/tools on": "Show tool execution",
    "/tools off": "Hide tool execution",
    "/export": "Export conversation",
    "/session": "Show session stats",
    "/tools": "List available tools",
    "/exit": "Quit",
}


def print_banner(model: str, thinking: bool = True) -> None:
    banner = Text()
    banner.append("  Personal AI Assistant  ", style=STYLE_BANNER)
    banner.append(f"  model: {model}  ", style=STYLE_DIM)

    console.print(
        Panel(
            banner,
            box=box.DOUBLE_EDGE,
            padding=(0, 2),
        )
    )

    _print_mode_line(thinking)

    console.print(
        Text("Type ", style=STYLE_DIM)
        + Text("/help", style=STYLE_COMMAND)
        + Text(" for commands, ", style=STYLE_DIM)
        + Text("/exit", style=STYLE_COMMAND)
        + Text(" to quit.\n", style=STYLE_DIM)
    )


def _print_mode_line(thinking: bool) -> None:
    if thinking:
        console.print(
            Text("  Mode: ", style=STYLE_DIM) + Text("🧠 thinking ON", style=STYLE_MODE_ON)
        )
    else:
        console.print(
            Text("  Mode: ", style=STYLE_DIM) + Text("⚡ thinking OFF", style=STYLE_MODE_OFF)
        )
    console.print()


def print_mode_change(thinking: bool) -> None:
    if thinking:
        console.print("\n[magenta bold]🧠 Thinking mode enabled[/magenta bold]\n")
    else:
        console.print("\n[yellow bold]⚡ Thinking mode disabled[/yellow bold]\n")


def print_help() -> None:
    console.print(Rule("Commands", style=STYLE_DIM))
    for cmd, desc in COMMANDS.items():
        console.print(f"[yellow]{cmd:<14}[/yellow] [bright_black]{desc}[/bright_black]")
    console.print()


def print_error(msg: str) -> None:
    console.print(f"\n[red bold]✗ {msg}[/red bold]\n")

def set_tool_visibility(enabled: bool) -> None:
    global SHOW_TOOL_OUTPUT
    SHOW_TOOL_OUTPUT = enabled


def print_tool_call(
    tool_name: str,
    args: dict,
) -> None:

    if not SHOW_TOOL_OUTPUT:
        return

    console.print()

    console.print(
        Panel(
            f"[bold cyan]{tool_name}[/bold cyan]",
            title="🔧 Tool Call",
            expand=False,
        )
    )

    formatted = json.dumps(
        args,
        indent=2,
        ensure_ascii=False,
        default=str,
    )

    console.print(
        Syntax(
            formatted,
            "json",
            word_wrap=True,
        )
    )


def print_tool_result(
    tool_name: str,
    result: str,
    elapsed: float | None = None,
) -> None:

    if not SHOW_TOOL_OUTPUT:
        return

    title = f"📄 Result • {tool_name}"

    if elapsed is not None:
        title += f" ({elapsed:.2f}s)"

    console.print()

    console.print(
        Panel(
            result[:5000],
            title=title,
            expand=False,
        )
    )


def print_info(msg: str) -> None:
    console.print(f"[bright_black]{msg}[/bright_black]")


def _print_thinking_header() -> None:
    console.print(
        "\n[bright_black]"
        "╭─ 💭 Thinking "
        "─────────────────────────────────────────"
        "[/bright_black]"
    )


def _print_thinking_footer(chars: int) -> None:
    console.print(
        f"[bright_black]"
        f"╰──────────────────────── {chars:,} chars ─╯"
        f"[/bright_black]\n"
    )


_OPEN = "<think>"
_CLOSE = "</think>"


class ThinkParser:
    """
    Fallback parser for models that emit <think>...</think> in plain content.
    Gemma 4 reasoning should normally arrive through reasoning_content instead.
    """

    def __init__(self) -> None:
        self.state = "BEFORE"
        self.buffer = ""

    def feed(self, chunk: str) -> list[tuple[str, str]]:
        result: list[tuple[str, str]] = []
        self.buffer += chunk

        while self.buffer:
            if self.state == "BEFORE":
                idx = self.buffer.find(_OPEN)

                if idx == -1:
                    safe = len(self.buffer) - len(_OPEN)
                    if safe > 0:
                        result.append(("response", self.buffer[:safe]))
                        self.buffer = self.buffer[safe:]
                    break

                if idx > 0:
                    result.append(("response", self.buffer[:idx]))

                self.buffer = self.buffer[idx + len(_OPEN):]
                self.state = "THINK"

            elif self.state == "THINK":
                idx = self.buffer.find(_CLOSE)

                if idx == -1:
                    safe = len(self.buffer) - len(_CLOSE)
                    if safe > 0:
                        result.append(("think", self.buffer[:safe]))
                        self.buffer = self.buffer[safe:]
                    break

                if idx > 0:
                    result.append(("think", self.buffer[:idx]))

                self.buffer = self.buffer[idx + len(_CLOSE):]
                self.state = "AFTER"

            else:
                result.append(("response", self.buffer))
                self.buffer = ""
                break

        return result

    def flush(self) -> list[tuple[str, str]]:
        if not self.buffer:
            return []

        kind = "think" if self.state == "THINK" else "response"
        remaining = self.buffer
        self.buffer = ""
        return [(kind, remaining)]


def _extract_tool_calls(chunk: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    direct = getattr(chunk, "tool_calls", None)
    if direct:
        for call in direct:
            if isinstance(call, dict):
                calls.append(call)

    additional = getattr(chunk, "additional_kwargs", None) or {}
    raw = additional.get("tool_calls") or []
    for call in raw:
        if isinstance(call, dict):
            calls.append(call)

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for call in calls:
        marker = json.dumps(call, sort_keys=True, default=str, ensure_ascii=False)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(call)

    return unique


def _extract_reasoning(chunk: Any) -> str:
    """
    Gemma 4 streams reasoning content here when thinking mode is active.
    With vLLM, this requires the server to be launched with
    --enable-reasoning --reasoning-parser <parser-for-gemma4>; langchain_openai
    surfaces the field verbatim in additional_kwargs. If your vLLM build/parser
    doesn't expose this field, reasoning will still show up via the
    <think>...</think> fallback in `content` handled by ThinkParser below.
    """
    additional = getattr(chunk, "additional_kwargs", None) or {}
    reasoning = additional.get("reasoning_content", "") or ""
    if isinstance(reasoning, str) and reasoning:
        return reasoning
        
    # Check directly in the message dict/kwargs for Ollama compatibility
    if hasattr(chunk, "__dict__"):
        reasoning = chunk.__dict__.get("kwargs", {}).get("reasoning_content", "")
        if isinstance(reasoning, str) and reasoning:
            return reasoning

    return ""


async def stream_response(stream: AsyncIterator) -> tuple[str, str, list[dict[str, Any]]]:
    """
    Stream one model response, returning:
        (thinking_text, answer_text, tool_calls)

    This function handles both:
    - Gemma 4 reasoning_content streaming
    - fallback <think>...</think> content streaming
    - tool call extraction from streamed chunks
    """
    parser = ThinkParser()
    thinking: list[str] = []
    response: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    thinking_started = False
    thinking_closed = False
    response_started = False

    async for chunk in stream:
        reasoning = _extract_reasoning(chunk)
        if reasoning:
            if not thinking_started:
                thinking_started = True
                _print_thinking_header()

            thinking.append(reasoning)
            console.print(reasoning, end="", style=STYLE_THINKING)

        content = getattr(chunk, "content", "") or ""
        if content:
            for kind, text in parser.feed(str(content)):
                if not text:
                    continue

                if kind == "think":
                    if not thinking_started:
                        thinking_started = True
                        _print_thinking_header()

                    thinking.append(text)
                    console.print(text, end="", style=STYLE_THINKING)

                else:
                    if thinking_started and not thinking_closed:
                        thinking_closed = True
                        console.print()
                        _print_thinking_footer(sum(len(x) for x in thinking))

                    if not response_started:
                        response_started = True
                        console.print(Text("Assistant ", style=STYLE_AI_LABEL))

                    response.append(text)
                    console.print(text, end="", style=STYLE_RESPONSE)

        tool_calls.extend(_extract_tool_calls(chunk))

    for kind, text in parser.flush():
        if not text:
            continue

        if kind == "think":
            thinking.append(text)
            # No need to re-print thinking — header already shown if thinking_started.
        else:
            # ── FIX: print the lookahead tail that ThinkParser held back ──────
            # ThinkParser keeps up to len("<think>") or len("</think>") bytes in
            # its buffer while scanning for tag boundaries.  flush() returns that
            # tail, which was correctly appended to response[] but was NEVER
            # printed — causing the last few characters to vanish from the
            # terminal even though the executor received the full text.
            if thinking_started and not thinking_closed:
                thinking_closed = True
                console.print()
                _print_thinking_footer(sum(len(x) for x in thinking))

            if not response_started:
                response_started = True
                console.print(Text("Assistant ", style=STYLE_AI_LABEL))

            response.append(text)
            console.print(text, end="", style=STYLE_RESPONSE)

    if thinking_started and not thinking_closed:
        thinking_closed = True
        console.print()
        _print_thinking_footer(sum(len(x) for x in thinking))

    if response_started:
        console.print()
    elif response:
        console.print()

    return "".join(thinking).strip(), "".join(response).strip(), tool_calls


def print_user_label() -> None:
    console.print()
    console.print(Text("You", style=STYLE_USER_LABEL), end="  ")