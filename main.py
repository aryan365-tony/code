from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime
from pathlib import Path
from ui import set_tool_visibility
# ── Load .env FIRST ───────────────────────────────────────────────────────────

_env_path = Path(__file__).parent / ".env"

if _env_path.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(_env_path, override=True)
    except ImportError:
        pass

# ── LangChain ────────────────────────────────────────────────────────────────

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ── Local modules ────────────────────────────────────────────────────────────

import tools  # noqa: F401  # side-effect import: registers tools
from executor import run_tool_loop
from llm import build_system_prompt, create_llm, bind_tools
from memory import MemoryManager
from ui import (
    console,
    print_banner,
    print_error,
    print_help,
    print_info,
    print_mode_change,
    print_user_label,
    stream_response,
)
from tools.base import list_tools_text

# ─────────────────────────────────────────────────────────────────────────────
# Session / Config
# ─────────────────────────────────────────────────────────────────────────────

AUTO_COMPACT = os.getenv("AUTO_COMPACT", "1").strip() != "0"
COMPACT_AT_TOKENS = int(os.getenv("COMPACT_AT_TOKENS", "6000"))
SUMMARY_NUM_PREDICT = int(os.getenv("SUMMARY_NUM_PREDICT", "256"))
MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", "12"))


class Session:
    def __init__(self) -> None:
        self.show_tools = True
        self.start_time = time.monotonic()
        self.turn_count = 0
        self.cancelled = False


# ─────────────────────────────────────────────────────────────────────────────
# Compact Helper
# ─────────────────────────────────────────────────────────────────────────────

async def _compact(memory: MemoryManager, summarizer_llm) -> str:
    """
    Summarise conversation history and replace it with one compact summary.
    """
    prompt = (
        "Summarise the following conversation into 3-5 concise bullet points.\n"
        "Keep names, preferences, constraints, and important decisions.\n"
        "Do not add anything that is not present in the conversation.\n\n"
    )

    async def summarise(raw: str) -> str:
        result = await summarizer_llm.ainvoke(
            prompt + raw,
            reasoning=False,
        )
        return (result.content or "").strip()

    return await memory.compact_history(summarise)


# ─────────────────────────────────────────────────────────────────────────────
# Export Helper
# ─────────────────────────────────────────────────────────────────────────────

def _export(memory: MemoryManager) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(f"conversation_{ts}.md")

    lines = [
        "# Conversation Export",
        f"Generated: {datetime.now()}",
        "",
        "---",
        "",
    ]

    for msg in memory.get_history():
        if isinstance(msg, HumanMessage):
            role = "You"
        elif isinstance(msg, AIMessage):
            role = "Assistant"
        elif isinstance(msg, SystemMessage):
            role = "System"
        else:
            role = type(msg).__name__

        lines.append(f"## {role}")
        lines.append(str(msg.content))
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Tool loop wrapper
# ─────────────────────────────────────────────────────────────────────────────

async def _run_agent_step(llm, messages):
    """
    Run one full tool-calling turn until the model returns a final answer.
    """
    bound_llm = bind_tools(llm, tools.get_lc_tools())

    async def _stream_fn(model, model_messages):
        return await stream_response(model.astream(model_messages))

    return await run_tool_loop(
        bound_llm,
        messages,
        max_iterations=MAX_TOOL_ITERATIONS,
        stream_fn=_stream_fn,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    memory = await MemoryManager.create()

    thinking_mode = True
    llm = create_llm(thinking=thinking_mode)

    summarizer_llm = create_llm(
        thinking=False,
        num_predict=SUMMARY_NUM_PREDICT,
        temperature=0.2,
        top_p=0.9,
        top_k=20,
    )

    session = Session()
    model_name = os.getenv("OLLAMA_MODEL", "qwen3:8b")

    print_banner(
        model_name,
        thinking=thinking_mode,
    )

    while not session.cancelled:
        print_user_label()

        try:
            user_input = await asyncio.to_thread(input, "")
        except (EOFError, KeyboardInterrupt):
            console.print()
            print_info("Goodbye.")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("/exit", "/quit", "exit", "quit"):
            print_info("Goodbye.")
            break

        if cmd == "/help":
            print_help()
            continue

        if cmd == "/tools on":
            session.show_tools = True
            set_tool_visibility(True)
            print_info("Tool output enabled.")
            continue

        if cmd == "/tools off":
            session.show_tools = False
            set_tool_visibility(False)
            print_info("Tool output disabled.")
            continue

        if cmd == "/tools":
            console.print()
            print_info("Registered tools:")
            console.print(list_tools_text())
            console.print()
            continue

        if cmd == "/think":
            thinking_mode = True
            llm = create_llm(thinking=True)
            print_mode_change(True)
            continue

        if cmd == "/no_think":
            thinking_mode = False
            llm = create_llm(thinking=False)
            print_mode_change(False)
            continue

        if cmd == "/clear":
            memory.clear_history()
            print_info("History cleared.")
            continue

        if cmd == "/undo":
            if memory.undo_last_exchange():
                print_info("Last exchange removed.")
            else:
                print_info("Nothing to undo.")
            continue

        if cmd == "/compact":
            print_info("Summarising history...")
            summary = await _compact(memory, summarizer_llm)
            if summary:
                print_info("History compacted.")
            else:
                print_info("Nothing to compact.")
            continue

        if cmd == "/export":
            path = _export(memory)
            print_info(f"Saved to {path}")
            continue

        if cmd == "/session":
            elapsed = int(time.monotonic() - session.start_time)
            mins, secs = divmod(elapsed, 60)
            tok_est = memory.history_token_estimate()

            print_info(
                f"Turns: {session.turn_count} | "
                f"Uptime: {mins}m {secs}s | "
                f"Tokens: ~{tok_est:,}"
            )
            continue

        if user_input.startswith("/"):
            print_error(f"Unknown command: {user_input}")
            continue

        try:
            ctx = await memory.get_context(query=user_input)
            system_prompt = build_system_prompt(extra_context=ctx)

            messages = [SystemMessage(content=system_prompt)]
            messages.extend(memory.get_history())
            messages.append(HumanMessage(content=user_input))

            reasoning_text, response_text = await _run_agent_step(llm, messages)

            if not response_text.strip() and not tool_calls:
                print_error("Model generated reasoning but neither a tool call nor answer.")
                continue

            memory.add_message(HumanMessage(content=user_input))
            memory.add_message(AIMessage(content=response_text))

            await memory.save_interaction(
                user_input,
                response_text,
            )

            session.turn_count += 1

            if AUTO_COMPACT and memory.history_token_estimate() >= COMPACT_AT_TOKENS:
                print_info("History is getting large. Compacting...")
                summary = await _compact(memory, summarizer_llm)
                if summary:
                    print_info("History compacted.")

        except KeyboardInterrupt:
            console.print()
            print_info("Interrupted.")
            continue
        except Exception as exc:
            print_error(f"LLM Error: {exc}")
            continue

    memory.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print()
