from __future__ import annotations
import asyncio
import json
import time
from typing import Any,Iterable
from langchain_core.messages import AIMessage,BaseMessage,ToolMessage
from tools.base import get_tool,get_tool_entry
from ui import print_tool_call,print_tool_result

def _safe_text(value:Any)->str:
    if value is None:
        return ""
    if isinstance(value,str):
        return value
    return json.dumps(value,ensure_ascii=False,indent=2,default=str)

def _normalize_tool_call(raw:Any)->dict[str,Any]:
    if not isinstance(raw,dict):
        raise TypeError(f"Unsupported tool call type: {type(raw)!r}")
    call_id=raw.get("id") or raw.get("tool_call_id") or raw.get("call_id") or ""
    name=raw.get("name")
    args=raw.get("args")
    if args is None:
        function=raw.get("function") or {}
        name=name or function.get("name")
        args=function.get("arguments")
    if isinstance(args,str):
        try:
            args=json.loads(args)
        except json.JSONDecodeError:
            args={"_raw":args}
    elif args is None:
        args={}
    if not isinstance(args,dict):
        args={"_value":args}
    return {"id":str(call_id) if call_id is not None else "","name":str(name) if name is not None else "","args":args}

def extract_tool_calls(message:Any)->list[dict[str,Any]]:
    collected:list[dict[str,Any]]=[]
    direct=getattr(message,"tool_calls",None)
    if direct:
        for call in direct:
            collected.append(_normalize_tool_call(call))
    additional=getattr(message,"additional_kwargs",None) or {}
    for call in additional.get("tool_calls") or []:
        collected.append(_normalize_tool_call(call))
    seen:set[tuple[str,str,str]]=set()
    unique:list[dict[str,Any]]=[]
    for call in collected:
        fp=(call.get("id",""),call.get("name",""),json.dumps(call.get("args",{}),sort_keys=True,ensure_ascii=False))
        if fp in seen:
            continue
        seen.add(fp)
        unique.append(call)
    return unique

async def _run_single_tool(call:dict[str,Any],metrics=None)->ToolMessage:
    name=call["name"]
    tool=get_tool(name)
    tool_id=call.get("id","")
    if tool is None:
        return ToolMessage(content=f"Error: tool '{name}' is not registered.",tool_call_id=tool_id or name,name=name)
    entry=get_tool_entry(name)
    meta=entry.metadata if entry else None
    timeout_s=meta.timeout_s if meta else 30.0
    retry_limit=meta.retry_limit if meta else 0
    backoff_base=meta.retry_backoff_base if meta else 1.5
    args=call.get("args",{})
    print_tool_call(name,args)
    started=time.perf_counter()
    retries=0
    last_exc:Exception|None=None
    for attempt in range(retry_limit+1):
        if attempt>0:
            retries+=1
            await asyncio.sleep(backoff_base**(attempt-1))
        try:
            result=await asyncio.wait_for(asyncio.to_thread(tool.invoke,args),timeout=timeout_s)
            elapsed_ms=(time.perf_counter()-started)*1000
            result_text=_safe_text(result)
            print_tool_result(name,result_text,(time.perf_counter()-started))
            if metrics:
                metrics.record(name,elapsed_ms,True,retries)
            return ToolMessage(content=result_text,tool_call_id=tool_id or name,name=name)
        except asyncio.TimeoutError:
            elapsed_ms=(time.perf_counter()-started)*1000
            msg=f"Error: tool '{name}' timed out after {timeout_s}s."
            print_tool_result(name,msg)
            if metrics:
                metrics.record(name,elapsed_ms,False,retries)
            return ToolMessage(content=msg,tool_call_id=tool_id or name,name=name)
        except Exception as exc:
            last_exc=exc
    elapsed_ms=(time.perf_counter()-started)*1000
    msg=f"Error: tool '{name}' failed after {retries} retries: {last_exc}"
    print_tool_result(name,msg)
    if metrics:
        metrics.record(name,elapsed_ms,False,retries)
    return ToolMessage(content=msg,tool_call_id=tool_id or name,name=name)

MAX_PARALLEL=3

async def execute_tool_calls(calls:Iterable[dict[str,Any]],metrics=None)->tuple[list[ToolMessage],int]:
    call_list=list(calls)
    parallel:list[tuple[int,dict]]=[]
    sequential:list[tuple[int,dict]]=[]
    for i,call in enumerate(call_list):
        entry=get_tool_entry(call["name"])
        meta=entry.metadata if entry else None
        if meta and (meta.risk_level=="destructive" or not meta.supports_parallel):
            sequential.append((i,call))
        else:
            parallel.append((i,call))
    parallel=parallel[:MAX_PARALLEL]
    results:list[ToolMessage|None]=[None]*len(call_list)
    if parallel:
        parallel_results=await asyncio.gather(*[_run_single_tool(c,metrics) for _,c in parallel])
        for (i,_),r in zip(parallel,parallel_results):
            results[i]=r
    for i,call in sequential:
        results[i]=await _run_single_tool(call,metrics)
    completed=[r for r in results if r is not None]
    return completed,len(parallel)

# ── Nudge messages ─────────────────────────────────────────────────────────────
# Gemma4 needs an explicit hard-stop instruction after every tool result.
# The old code only nudged after *multi-tool* calls (parallel_count > 1),
# leaving single-tool calls with no synthesis cue — causing the infinite loop.
_NUDGE_SINGLE=(
    "STOP. Do NOT call any tools. "
    "The tool result is shown above. "
    "Write your final answer to the user now using only that result."
)
_NUDGE_MULTI=(
    "STOP. Do NOT call any more tools. "
    "All tool results are shown above. "
    "Summarise what was done and write your final answer now."
)
_NUDGE_STUCK="Stop deliberating. Answer the user's message directly and concisely right now."
# Injected when the model tries to call the exact same tool+args a second time.
_NUDGE_LOOP=(
    "STOP IMMEDIATELY. You already called this tool and received its result above. "
    "Do NOT call it again. Write your final answer now using the result already provided."
)

_REASONING_LOOP_THRESHOLD=1800

def _call_fingerprint(call:dict[str,Any])->str:
    """Stable key for a tool call ignoring its ephemeral id."""
    return f"{call.get('name','')}:{json.dumps(call.get('args',{}),sort_keys=True,ensure_ascii=False)}"

async def run_tool_loop(
    llm:Any,
    messages:list[BaseMessage],
    *,
    max_iterations:int=12,
    stream_fn,
    metrics=None,
)->tuple[str,str]:
    from langchain_core.messages import HumanMessage
    working_messages=list(messages)
    ran_tools=False
    last_parallel_count=0
    # Maps call fingerprint → number of times it has been *dispatched*.
    # If the model tries to fire the same call twice we catch it before
    # wasting another round-trip to the tool.
    dispatched_counts:dict[str,int]={}

    for _ in range(max_iterations):
        reasoning_text,answer_text,tool_calls=await stream_fn(llm,working_messages)

        if tool_calls:
            # ── Repeated-call detection ───────────────────────────────────────
            # Check every requested call against what has already been dispatched.
            # If ANY call is a repeat, skip execution entirely, inject a hard-stop
            # nudge, and let the model try once more to write a real answer.
            repeated_calls=[c for c in tool_calls if dispatched_counts.get(_call_fingerprint(c),0)>0]
            if repeated_calls:
                working_messages.append(HumanMessage(content=_NUDGE_LOOP))
                reasoning_text,answer_text,_=await stream_fn(llm,working_messages)
                return reasoning_text,answer_text

            # Record dispatch counts before running so a retry inside
            # execute_tool_calls doesn't bypass the check.
            for call in tool_calls:
                fp=_call_fingerprint(call)
                dispatched_counts[fp]=dispatched_counts.get(fp,0)+1

            tool_messages,parallel_count=await execute_tool_calls(tool_calls,metrics)
            normalized_calls = [_normalize_tool_call(c) for c in tool_calls]
            working_messages.append(AIMessage(content=answer_text or "", tool_calls=normalized_calls))
            working_messages.extend(tool_messages)

            # ── Always nudge after tool results (fixes the single-tool loop) ──
            # Previously only _NUDGE_MULTI was appended and only when
            # parallel_count > 1.  Gemma4 needs an explicit synthesis cue even
            # after a single sequential tool call; without it the model just
            # re-invokes the tool instead of answering.
            nudge=_NUDGE_MULTI if parallel_count>1 else _NUDGE_SINGLE
            working_messages.append(HumanMessage(content=nudge))

            ran_tools=True
            last_parallel_count=parallel_count
            continue

        if not answer_text.strip():
            if ran_tools:
                nudge=_NUDGE_MULTI if last_parallel_count>1 else _NUDGE_SINGLE
            elif reasoning_text and len(reasoning_text)>=_REASONING_LOOP_THRESHOLD:
                nudge=_NUDGE_STUCK
            else:
                return reasoning_text,answer_text
            working_messages.append(HumanMessage(content=nudge))
            reasoning_text,answer_text,_=await stream_fn(llm,working_messages)

        return reasoning_text,answer_text

    raise RuntimeError(f"Tool loop exceeded {max_iterations} iterations.")