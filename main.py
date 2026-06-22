from __future__ import annotations
import argparse
import asyncio
import os
import time
from datetime import datetime
from pathlib import Path
_env_path=Path(__file__).parent/".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path,override=True)
    except ImportError:
        pass
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
import tools
from executor import run_tool_loop
from kernel import RuntimeKernel
from llm import build_system_prompt,create_llm,bind_tools,get_active_model_name,summarizer_ainvoke_kwargs
from memory import MemoryManager
from ui import (console,print_banner,print_error,print_help,print_info,print_mode_change,print_user_label,set_tool_visibility,stream_response)
from tools.base import list_tools_text
AUTO_COMPACT=os.getenv("AUTO_COMPACT","1").strip()!="0"
COMPACT_AT_TOKENS=int(os.getenv("COMPACT_AT_TOKENS","6000"))
SUMMARY_NUM_PREDICT=int(os.getenv("SUMMARY_NUM_PREDICT","256"))
MAX_TOOL_ITERATIONS=int(os.getenv("MAX_TOOL_ITERATIONS","12"))
async def _compact(memory:MemoryManager,summarizer_llm)->str:
    prompt=("Summarise the following conversation into 3-5 concise bullet points.\n"
            "Keep names, preferences, constraints, and important decisions.\n"
            "Do not add anything that is not present in the conversation.\n\n")
    extra=summarizer_ainvoke_kwargs()
    async def summarise(raw:str)->str:
        result=await summarizer_llm.ainvoke(prompt+raw,**extra)
        return (result.content or "").strip()
    return await memory.compact_history(summarise)
def _export(memory:MemoryManager)->Path:
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    out=Path(f"conversation_{ts}.md")
    lines=["# Conversation Export",f"Generated: {datetime.now()}","","---",""]
    for msg in memory.get_history():
        if isinstance(msg,HumanMessage):
            role="You"
        elif isinstance(msg,AIMessage):
            role="Assistant"
        elif isinstance(msg,SystemMessage):
            role="System"
        else:
            role=type(msg).__name__
        lines+=[f"## {role}",str(msg.content),""]
    out.write_text("\n".join(lines),encoding="utf-8")
    return out
async def _run_agent_step(llm,messages,metrics=None):
    bound_llm=bind_tools(llm,tools.get_lc_tools())
    async def _stream_fn(model,model_messages):
        return await stream_response(model.astream(model_messages))
    return await run_tool_loop(bound_llm,messages,max_iterations=MAX_TOOL_ITERATIONS,stream_fn=_stream_fn,metrics=metrics)
async def main()->None:
    parser=argparse.ArgumentParser(description="Personal AI Assistant")
    parser.add_argument("--resume",action="store_true",help="Resume last session")
    args=parser.parse_args()
    memory=await MemoryManager.create()
    kernel=await RuntimeKernel.create(memory)
    if args.resume:
        session=kernel.sessions.load()
        if session:
            for msg in session.history:
                memory.add_message(msg)
            print_info(f"Resumed session {session.session_id} ({session.turn_count} turns, {len(session.history)} messages)")
        else:
            print_info("No saved session found. Starting fresh.")
            session=kernel.sessions.create()
    else:
        session=kernel.sessions.create()
    thinking_mode=True
    llm=create_llm(thinking=thinking_mode)
    summarizer_llm=create_llm(thinking=False,max_tokens=SUMMARY_NUM_PREDICT,temperature=0.4)
    model_name=get_active_model_name()
    print_banner(model_name,thinking=thinking_mode)
    try:
        while not session.cancelled:
            print_user_label()
            try:
                user_input=await asyncio.to_thread(input,"")
            except (EOFError,KeyboardInterrupt):
                console.print()
                print_info("Goodbye.")
                break
            user_input=user_input.strip()
            if not user_input:
                continue
            cmd=user_input.lower()
            if cmd in ("/exit","/quit","exit","quit"):
                print_info("Goodbye.")
                break
            if cmd=="/help":
                print_help()
                continue
            if cmd=="/tools on":
                session.show_tools=True
                set_tool_visibility(True)
                print_info("Tool output enabled.")
                continue
            if cmd=="/tools off":
                session.show_tools=False
                set_tool_visibility(False)
                print_info("Tool output disabled.")
                continue
            if cmd=="/tools":
                console.print()
                print_info("Registered tools:")
                console.print(list_tools_text())
                console.print()
                continue
            if cmd=="/think":
                thinking_mode=True
                llm=create_llm(thinking=True)
                print_mode_change(True)
                continue
            if cmd=="/no_think":
                thinking_mode=False
                llm=create_llm(thinking=False)
                print_mode_change(False)
                continue
            if cmd=="/clear":
                memory.clear_history()
                session.history.clear()
                print_info("History cleared.")
                continue
            if cmd=="/undo":
                if memory.undo_last_exchange():
                    if len(session.history)>=2:
                        session.history=session.history[:-2]
                    print_info("Last exchange removed.")
                else:
                    print_info("Nothing to undo.")
                continue
            if cmd=="/compact":
                print_info("Summarising history...")
                summary=await _compact(memory,summarizer_llm)
                print_info("History compacted." if summary else "Nothing to compact.")
                continue
            if cmd=="/export":
                path=_export(memory)
                print_info(f"Saved to {path}")
                continue
            if cmd=="/session":
                elapsed=int(time.monotonic()-session.start_time)
                mins,secs=divmod(elapsed,60)
                tok_est=memory.history_token_estimate()
                print_info(f"ID: {session.session_id} | Turns: {session.turn_count} | Tools used: {session.tool_calls_total} | Uptime: {mins}m {secs}s | Tokens: ~{tok_est:,}")
                continue
            if cmd=="/metrics":
                console.print()
                console.print(kernel.metrics.format_table())
                console.print()
                continue
            if user_input.startswith("/"):
                print_error(f"Unknown command: {user_input}")
                continue
            try:
                ctx=await memory.get_context(query=user_input)
                system_prompt=build_system_prompt(extra_context=ctx,thinking=thinking_mode)
                messages=[SystemMessage(content=system_prompt)]
                messages.extend(memory.get_history())
                messages.append(HumanMessage(content=user_input))
                reasoning_text,response_text=await _run_agent_step(llm,messages,metrics=kernel.metrics)
                if not response_text.strip():
                    print_error("Model returned an empty response.")
                    continue
                memory.add_message(HumanMessage(content=user_input))
                memory.add_message(AIMessage(content=response_text))
                session.history=list(memory.get_history())
                await memory.save_interaction(user_input,response_text)
                session.turn_count+=1
                kernel.sessions.save(session)
                if AUTO_COMPACT and memory.history_token_estimate()>=COMPACT_AT_TOKENS:
                    print_info("History large — compacting...")
                    summary=await _compact(memory,summarizer_llm)
                    if summary:
                        print_info("History compacted.")
            except KeyboardInterrupt:
                console.print()
                print_info("Interrupted.")
                continue
            except Exception as exc:
                print_error(f"Error: {exc}")
                continue
    finally:
        await kernel.shutdown(session)
if __name__=="__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print()
