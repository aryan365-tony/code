"""
ui.py — Rich terminal UI for Qwen3.

Features
--------
• Live token streaming
• Qwen3 <think>...</think> parser
• Thinking panel rendering
• /think and /no_think support
• Session banners and commands
"""

from __future__ import annotations

from typing import AsyncIterator

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.text import Text
from rich import box

# ──────────────────────────────────────────────────────────────────────────────
# Shared Console
# ──────────────────────────────────────────────────────────────────────────────

console = Console(highlight=False)

# ──────────────────────────────────────────────────────────────────────────────
# Styles
# ──────────────────────────────────────────────────────────────────────────────

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

# ──────────────────────────────────────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────────────────────────────────────

COMMANDS = {
    "/help": "List commands",
    "/think": "Enable Qwen3 reasoning",
    "/no_think": "Disable Qwen3 reasoning",
    "/clear": "Clear history",
    "/undo": "Remove last exchange",
    "/compact": "Summarise history",
    "/export": "Export conversation",
    "/session": "Show session stats",
    "/exit": "Quit",
}

# ──────────────────────────────────────────────────────────────────────────────
# Banner
# ──────────────────────────────────────────────────────────────────────────────

def print_banner(model: str, thinking: bool = True) -> None:
    banner = Text()
    banner.append(
        "  Personal AI Assistant  ",
        style=STYLE_BANNER,
    )
    banner.append(
        f"  model: {model}  ",
        style=STYLE_DIM,
    )

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
            Text("  Mode: ", style=STYLE_DIM)
            + Text(
                "🧠 thinking ON",
                style=STYLE_MODE_ON,
            )
        )
    else:
        console.print(
            Text("  Mode: ", style=STYLE_DIM)
            + Text(
                "⚡ thinking OFF",
                style=STYLE_MODE_OFF,
            )
        )

    console.print()


def print_mode_change(thinking: bool) -> None:
    if thinking:
        console.print(
            "\n[magenta bold]🧠 Thinking mode enabled[/magenta bold]\n"
        )
    else:
        console.print(
            "\n[yellow bold]⚡ Thinking mode disabled[/yellow bold]\n"
        )

# ──────────────────────────────────────────────────────────────────────────────
# Help
# ──────────────────────────────────────────────────────────────────────────────

def print_help() -> None:
    console.print(
        Rule(
            "Commands",
            style=STYLE_DIM,
        )
    )

    for cmd, desc in COMMANDS.items():
        console.print(
            f"[yellow]{cmd:<14}[/yellow] "
            f"[bright_black]{desc}[/bright_black]"
        )

    console.print()

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def print_error(msg: str) -> None:
    console.print(
        f"\n[red bold]✗ {msg}[/red bold]\n"
    )


def print_info(msg: str) -> None:
    console.print(
        f"[bright_black]{msg}[/bright_black]"
    )

# ──────────────────────────────────────────────────────────────────────────────
# Thinking Block
# ──────────────────────────────────────────────────────────────────────────────

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

# ──────────────────────────────────────────────────────────────────────────────
# Qwen3 Think Parser
# ──────────────────────────────────────────────────────────────────────────────

_OPEN = "<think>"
_CLOSE = "</think>"


class ThinkParser:
    def __init__(self):
        self.state = "BEFORE"
        self.buffer = ""

    def feed(
        self,
        chunk: str,
    ) -> list[tuple[str, str]]:

        result = []
        self.buffer += chunk

        while self.buffer:

            if self.state == "BEFORE":

                idx = self.buffer.find(_OPEN)

                if idx == -1:
                    safe = len(self.buffer) - len(_OPEN)

                    if safe > 0:
                        result.append(
                            (
                                "response",
                                self.buffer[:safe],
                            )
                        )
                        self.buffer = self.buffer[safe:]

                    break

                if idx > 0:
                    result.append(
                        (
                            "response",
                            self.buffer[:idx],
                        )
                    )

                self.buffer = self.buffer[
                    idx + len(_OPEN):
                ]

                self.state = "THINK"

            elif self.state == "THINK":

                idx = self.buffer.find(_CLOSE)

                if idx == -1:
                    safe = len(self.buffer) - len(_CLOSE)

                    if safe > 0:
                        result.append(
                            (
                                "think",
                                self.buffer[:safe],
                            )
                        )
                        self.buffer = self.buffer[safe:]

                    break

                if idx > 0:
                    result.append(
                        (
                            "think",
                            self.buffer[:idx],
                        )
                    )

                self.buffer = self.buffer[
                    idx + len(_CLOSE):
                ]

                self.state = "AFTER"

            else:
                result.append(
                    (
                        "response",
                        self.buffer,
                    )
                )

                self.buffer = ""
                break

        return result

    def flush(self):
        if not self.buffer:
            return []

        kind = (
            "think"
            if self.state == "THINK"
            else "response"
        )

        remaining = self.buffer
        self.buffer = ""

        return [(kind, remaining)]

# ──────────────────────────────────────────────────────────────────────────────
# Streaming Renderer
# ──────────────────────────────────────────────────────────────────────────────

async def stream_response(
    stream: AsyncIterator,
) -> tuple[str, str]:

    parser = ThinkParser()

    thinking = []
    response = []

    thinking_started = False
    thinking_closed = False
    response_started = False

    async for chunk in stream:

        content = chunk.content or ""

        if not content:
            continue

        for kind, text in parser.feed(content):

            if not text:
                continue

            if kind == "think":

                if not thinking_started:
                    thinking_started = True
                    _print_thinking_header()

                thinking.append(text)

                console.print(
                    text,
                    end="",
                    style=STYLE_THINKING,
                )

            else:

                if (
                    thinking_started
                    and not thinking_closed
                ):
                    thinking_closed = True

                    console.print()

                    _print_thinking_footer(
                        sum(
                            len(x)
                            for x in thinking
                        )
                    )

                if not response_started:
                    response_started = True

                    console.print(
                        Text(
                            "Assistant ",
                            style=STYLE_AI_LABEL,
                        )
                    )

                response.append(text)

                console.print(
                    text,
                    end="",
                    style=STYLE_RESPONSE,
                )

    for kind, text in parser.flush():

        if kind == "think":
            thinking.append(text)
        else:
            response.append(text)

    if response_started:
        console.print()

    return (
        "".join(thinking),
        "".join(response),
    )

# ──────────────────────────────────────────────────────────────────────────────
# Prompt Label
# ──────────────────────────────────────────────────────────────────────────────

def print_user_label() -> None:
    console.print()

    console.print(
        Text(
            "You",
            style=STYLE_USER_LABEL,
        ),
        end="  ",
    )