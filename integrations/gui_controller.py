import asyncio
import time
import json
from pathlib import Path
from PySide6.QtCore import QObject, Signal

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from kernel import RuntimeKernel
from memory import MemoryManager
from llm import create_llm, build_system_prompt, bind_tools
import tools
from executor import run_tool_loop
from ui import ThinkParser, _extract_tool_calls

from main import _compact, _export, AUTO_COMPACT, COMPACT_AT_TOKENS

class GuiController(QObject):
    state_changed = Signal(str)  # "IDLE", "THINKING", "RESPONDING", "TOOL_RUNNING", "ERROR"

    def __init__(self, kernel: RuntimeKernel, memory: MemoryManager, session, terminal_view):
        super().__init__()
        self.kernel = kernel
        self.memory = memory
        self.session = session
        self.terminal_view = terminal_view
        
        self.thinking_mode = True
        self.show_tools = True
        
        self.llm_thinking = create_llm(thinking=True)
        self.llm_fast = create_llm(thinking=False)
        self.summarizer_llm = create_llm(thinking=False, max_tokens=256, temperature=0.4)

    async def _stream_fn(self, model, model_messages) -> tuple[str, str, list]:
        self.state_changed.emit("RESPONDING")
        
        parser = ThinkParser()
        full_content = []
        full_thinking = []
        full_chunk = None
        
        thinking_started = False
        thinking_closed = False
        response_started = False
        
        async for chunk in model.astream(model_messages):
            if full_chunk is None:
                full_chunk = chunk
            else:
                full_chunk += chunk
                
            # Gemma 4 reasoning payload
            reasoning = self._extract_reasoning(chunk)
            if reasoning:
                if not thinking_started:
                    thinking_started = True
                    self.state_changed.emit("THINKING")
                    self.terminal_view.print_thinking_header()
                self.terminal_view.print_thinking_token(reasoning)
                full_thinking.append(reasoning)
                
            # Content parsing (might contain <think> fallback)
            content = chunk.content or ""
            if content:
                for type_str, text in parser.feed(content):
                    if not text: continue
                    
                    if type_str == "think":
                        if not thinking_started:
                            thinking_started = True
                            self.state_changed.emit("THINKING")
                            self.terminal_view.print_thinking_header()
                        self.terminal_view.print_thinking_token(text)
                        full_thinking.append(text)
                    else:
                        if thinking_started and not thinking_closed:
                            thinking_closed = True
                            self.terminal_view.print_thinking_footer(sum(len(x) for x in full_thinking))
                            
                        if not response_started:
                            response_started = True
                            self.state_changed.emit("RESPONDING")
                            self.terminal_view.print_assistant_label()
                            
                        self.terminal_view.print_assistant_token(text)
                        full_content.append(text)
                        
        # Flush the parser buffer
        for type_str, text in parser.flush():
            if not text: continue
            if type_str == "think":
                self.terminal_view.print_thinking_token(text)
                full_thinking.append(text)
            else:
                if thinking_started and not thinking_closed:
                    thinking_closed = True
                    self.terminal_view.print_thinking_footer(sum(len(x) for x in full_thinking))
                if not response_started:
                    response_started = True
                    self.state_changed.emit("RESPONDING")
                    self.terminal_view.print_assistant_label()
                self.terminal_view.print_assistant_token(text)
                full_content.append(text)
                
        # Close thinking if it was never closed
        if thinking_started and not thinking_closed:
            self.terminal_view.print_thinking_footer(sum(len(x) for x in full_thinking))
                
        # Extract tool calls from the complete accumulated chunk
        tool_calls = _extract_tool_calls(full_chunk) if full_chunk else []
        
        return "".join(full_thinking), "".join(full_content), tool_calls

    def _extract_reasoning(self, chunk) -> str:
        additional = getattr(chunk, "additional_kwargs", None) or {}
        reasoning = additional.get("reasoning_content", "") or ""
        if isinstance(reasoning, str) and reasoning:
            return reasoning
        if hasattr(chunk, "__dict__"):
            reasoning = chunk.__dict__.get("kwargs", {}).get("reasoning_content", "")
            if isinstance(reasoning, str) and reasoning:
                return reasoning
        return ""

    def monkeypatch_ui_for_tools(self):
        """Hijack ui.py prints so executor loops write to our terminal_view instead"""
        import ui
        
        def _intercept_print_tool_call(tool_name: str, args: dict):
            if self.show_tools:
                args_json = json.dumps(args, indent=2, ensure_ascii=False, default=str)
                self.terminal_view.print_tool_call(tool_name, args_json)
            self.state_changed.emit("TOOL_RUNNING")

        def _intercept_print_tool_result(tool_name: str, result: str, elapsed: float | None = None):
            if self.show_tools:
                self.terminal_view.print_tool_result(tool_name, result, elapsed)

        ui.print_tool_call = _intercept_print_tool_call
        ui.print_tool_result = _intercept_print_tool_result
        ui.SHOW_TOOL_OUTPUT = self.show_tools

    async def process_input(self, user_text: str):
        user_text = user_text.strip()
        if not user_text:
            return
            
        cmd = user_text.lower()
        
        # Replicate main.py slash commands natively in GUI
        if cmd == "/help":
            self.terminal_view.print_system("Commands: /clear, /undo, /compact, /tools on, /tools off, /export, /session, /think, /no_think")
            return
        if cmd == "/tools on":
            self.show_tools = True
            self.terminal_view.print_system("Tool output enabled.")
            self.monkeypatch_ui_for_tools()
            return
        if cmd == "/tools off":
            self.show_tools = False
            self.terminal_view.print_system("Tool output disabled.")
            self.monkeypatch_ui_for_tools()
            return
        if cmd == "/think":
            self.thinking_mode = True
            self.terminal_view.print_system("🧠 Thinking mode enabled", error=False)
            return
        if cmd == "/no_think":
            self.thinking_mode = False
            self.terminal_view.print_system("⚡ Thinking mode disabled", error=False)
            return
        if cmd == "/clear":
            self.memory.clear_history()
            self.session.history.clear()
            self.kernel.sessions.save(self.session)
            self.terminal_view.clear()
            self.terminal_view.print_system("History cleared.")
            return
        if cmd == "/undo":
            if self.memory.undo_last_exchange():
                if len(self.session.history) >= 2:
                    self.session.history = self.session.history[:-2]
                self.terminal_view.print_system("Last exchange removed from memory (text remains in view).")
            else:
                self.terminal_view.print_system("Nothing to undo.")
            return
        if cmd == "/compact":
            self.terminal_view.print_system("Summarising history...")
            summary = await _compact(self.memory, self.summarizer_llm)
            if summary:
                self.terminal_view.print_system("History compacted.")
            else:
                self.terminal_view.print_system("Nothing to compact.")
            return
        if cmd == "/export":
            path = _export(self.memory)
            self.terminal_view.print_system(f"Saved to {path}")
            return
        if cmd == "/session":
            elapsed = int(time.monotonic() - self.session.start_time)
            mins, secs = divmod(elapsed, 60)
            tok_est = self.memory.history_token_estimate()
            self.terminal_view.print_system(
                f"ID: {self.session.session_id} | Turns: {self.session.turn_count} | Uptime: {mins}m {secs}s | Tokens: ~{tok_est:,}"
            )
            return
        if cmd.startswith("/"):
            self.terminal_view.print_system(f"Unknown command: {user_text}", error=True)
            return

        # Normal generation
        self.terminal_view.print_user(user_text)
        self.state_changed.emit("THINKING")
        self.monkeypatch_ui_for_tools()
        
        try:
            ctx = await self.memory.get_context(query=user_text)
            system_prompt = build_system_prompt(extra_context=ctx, thinking=self.thinking_mode)
            
            messages = [SystemMessage(content=system_prompt)]
            messages.extend(self.memory.get_history())
            messages.append(HumanMessage(content=user_text))
            
            llm = self.llm_thinking if self.thinking_mode else self.llm_fast
            bound_llm = bind_tools(llm, tools.get_lc_tools())
            
            # Run the executor loop
            reasoning_text, response_text = await run_tool_loop(
                bound_llm, 
                messages, 
                max_iterations=12, 
                stream_fn=self._stream_fn, 
                metrics=self.kernel.metrics
            )
            
            if response_text.strip():
                self.memory.add_message(HumanMessage(content=user_text))
                self.memory.add_message(AIMessage(content=response_text))
                await self.memory.save_interaction(user_text, response_text)
                
            self.session.history = list(self.memory.get_history())
            self.session.turn_count += 1
            self.kernel.sessions.save(self.session)
            
            if AUTO_COMPACT and self.memory.history_token_estimate() >= COMPACT_AT_TOKENS:
                self.terminal_view.print_system("History large — compacting...")
                summary = await _compact(self.memory, self.summarizer_llm)
                if summary:
                    self.terminal_view.print_system("History compacted.")
                    
        except asyncio.CancelledError:
            self.terminal_view.print_system("\n[Generation Cancelled]", error=True)
            raise
        except Exception as e:
            self.terminal_view.print_system(f"\nError: {e}", error=True)
        finally:
            self.state_changed.emit("IDLE")
