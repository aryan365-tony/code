from __future__ import annotations
import os
from typing import Any,Sequence
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.tools import BaseTool
_THINKING_PARAMS={"temperature":float(os.getenv("TEMPERATURE_THINKING","1.0")),"top_p":float(os.getenv("TOP_P_THINKING","0.95")),"top_k":int(os.getenv("TOP_K_THINKING","64"))}
_FAST_PARAMS={"temperature":float(os.getenv("TEMPERATURE_FAST","1.0")),"top_p":float(os.getenv("TOP_P_FAST","0.95")),"top_k":int(os.getenv("TOP_K_FAST","64"))}
_GEMMA4_THINK_TOKEN="<|think|>"
SYSTEM_PROMPT="""\
You are an expert personal AI assistant — capable, direct, and proactive.

## Core behaviour
- Act immediately. When a tool can answer the question, call it — do not describe, discuss, or plan.
- Complete tasks end-to-end. Never stop at "here is a plan" when execution is possible.
- Be concise but thorough. Omit filler. Lead with the answer, follow with detail.
- Admit uncertainty instead of guessing. Say what you don't know.
- Proactively surface the next useful step after finishing a task.

## Tool discipline
- Call the right tool immediately. No preamble like "I'll now use web_search…".
- When multiple independent tools are needed, call them in the same turn — they run in parallel.
- After tool results arrive, synthesize and answer. Do not re-describe what the tool returned.
- If a tool errors, report exactly what failed and why, then attempt an alternative if one exists.
- Never fabricate file contents, search results, timestamps, or code outputs.

## Coding workflow (always in this order)
1. Read all relevant files first (in parallel when possible).
2. Plan the change — identify every affected location.
3. Write or patch files.
4. Run linter and tests.
5. Report git diff / status.

## Memory and context
- Trust the memory context provided at the top of the prompt.
- Refer to the user by name when known.
- Carry forward preferences and constraints from earlier turns without being asked.

## Reasoning discipline
- If you are unsure whether context was provided, ASK in one sentence — do not reason about it for more than a few tokens.
- Never loop over the same uncertainty twice in reasoning. Decide and act.
- If the user's request is ambiguous, state the assumption you are making and proceed.
- Short questions get short answers. Do not over-engineer simple responses.

## Output quality
- Use markdown only when it improves readability (headers, code fences, tables).
- Keep reasoning internal. Never leak thinking blocks or raw tool JSON into the final answer.
- Prefer concrete examples over abstract explanations.
"""
def build_system_prompt(extra_context:str="",thinking:bool=True)->str:
    base=SYSTEM_PROMPT.strip()
    if extra_context.strip():
        base=f"{base}\n\n{extra_context.strip()}"
    if thinking:
        return f"{_GEMMA4_THINK_TOKEN}\n{base}"
    return base
def get_active_model_name()->str:
    backend=os.getenv("INFERENCE_BACKEND","vllm").lower()
    if backend=="ollama":return os.getenv("OLLAMA_MODEL","gemma4:e4b")
    if backend=="llamacpp":return os.getenv("LLAMACPP_MODEL","gemma4:e4b")
    return os.getenv("VLLM_MODEL","gemma4-e4b")
def _default_num_thread()->int:
    return max(1,(os.cpu_count() or 8)//2)
def create_llm(thinking:bool=True,**overrides:Any)->Any:
    backend=os.getenv("INFERENCE_BACKEND","vllm").lower()
    profile=_THINKING_PARAMS if thinking else _FAST_PARAMS
    if backend=="ollama":
        if "max_tokens" in overrides:
            overrides["num_predict"]=overrides.pop("max_tokens")
        params:dict[str,Any]={
            "model":os.getenv("OLLAMA_MODEL","gemma4:e4b"),
            "base_url":os.getenv("OLLAMA_BASE_URL","http://localhost:11434"),
            "num_ctx":int(os.getenv("NUM_CTX","32768")),
            "num_predict":int(os.getenv("NUM_PREDICT","2048")),
            "repeat_penalty":float(os.getenv("REPEAT_PENALTY","1.0")),
            "num_thread":int(os.getenv("NUM_THREAD",str(_default_num_thread()))),
            "temperature":profile["temperature"],
            "top_p":profile["top_p"],
            "top_k":profile["top_k"],
            "keep_alive":os.getenv("OLLAMA_KEEP_ALIVE","10m"),"reasoning":thinking,
        }
        num_gpu=os.getenv("NUM_GPU")
        if num_gpu and num_gpu.strip():
            params["num_gpu"]=int(num_gpu)
        params.update(overrides)
        return ChatOllama(**params)
    if backend=="llamacpp":
        extra_body:dict[str,Any]={
            "top_k":profile["top_k"],
            "repeat_penalty":float(os.getenv("REPEAT_PENALTY","1.0")),
            "parallel_tool_calls":True,
        }
        params={
            "model":os.getenv("LLAMACPP_MODEL","gemma4:e4b"),
            "base_url":os.getenv("LLAMACPP_BASE_URL","http://localhost:8080/v1"),
            "api_key":os.getenv("LLAMACPP_API_KEY","none"),
            "max_tokens":int(os.getenv("NUM_PREDICT","2048")),
            "temperature":profile["temperature"],
            "top_p":profile["top_p"],
            "streaming":True,
            "extra_body":extra_body,
        }
        params.update(overrides)
        return ChatOpenAI(**params)
    extra_body={
        "top_k":profile["top_k"],
        "repetition_penalty":float(os.getenv("REPEAT_PENALTY","1.0")),
    }
    params={
        "model":os.getenv("VLLM_MODEL","gemma4-e4b"),
        "base_url":os.getenv("VLLM_BASE_URL","http://localhost:8000/v1"),
        "api_key":os.getenv("VLLM_API_KEY","EMPTY"),
        "max_tokens":int(os.getenv("NUM_PREDICT","2048")),
        "temperature":profile["temperature"],
        "top_p":profile["top_p"],
        "streaming":True,
        "extra_body":extra_body,
    }
    params.update(overrides)
    return ChatOpenAI(**params)
def bind_tools(llm:Any,tools:Sequence[BaseTool])->Any:
    return llm.bind_tools(list(tools))
def summarizer_ainvoke_kwargs()->dict:
    backend=os.getenv("INFERENCE_BACKEND","vllm").lower()
    if backend=="ollama":
        return {"reasoning":False}
    return {}
