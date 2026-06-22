# Ollama → vLLM migration notes

## What changed in code
Ollama coupling was isolated to `llm.py` plus two small spots in `main.py`. Nothing in
`executor.py`, `memory.py`, `kernel/*`, `tools/*`, or the tool-loop logic touches the
model client directly — they work against LangChain's generic `BaseMessage` /
`tool_calls` interfaces, so they needed **no changes**.

- `llm.py` — `ChatOllama` → `ChatOpenAI` (from `langchain_openai`), pointed at vLLM's
  OpenAI-compatible `/v1` endpoint. `top_k` and `repetition_penalty` aren't part of the
  OpenAI chat-completions spec, so they're now sent via `extra_body` (vLLM accepts them
  there). `num_predict` → `max_tokens`.
- `main.py` — `OLLAMA_MODEL` env var → `VLLM_MODEL`; removed the `reasoning=False` kwarg
  on the summarizer's `.ainvoke()` call (an Ollama-only override with no OpenAI-API
  equivalent — harmless to drop since the summarizer prompt never contains the
  `<|think|>` token anyway).
- `requirements.txt` — `langchain-ollama` → `langchain-openai`.
- `ui.py` — unchanged logic. It already has a robust fallback: it reads
  `additional_kwargs["reasoning_content"]` *and* parses `<think>...</think>` tags in
  plain content. Whichever way your vLLM server surfaces reasoning, one of the two
  paths will catch it.

## What you must change outside the code: the vLLM server launch flags

These have **no Python-side equivalent** — they were previously implicit in how Ollama
auto-handled things, and now need to be explicit `vllm serve` flags:

```bash
vllm serve <path-or-hf-id-of-gemma4> \
  --served-model-name gemma4-e4b \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser <parser-matching-gemma4> \
  --enable-reasoning \
  --reasoning-parser <parser-matching-gemma4> \
  --gpu-memory-utilization 0.9
```

Two of these are **not optional** for this app to keep working, not just nice-to-haves:

1. **`--enable-auto-tool-choice` + `--tool-call-parser`** — without this, `bind_tools()`
   in `llm.py` and the whole tool-calling loop in `executor.py` will silently stop
   working (the model will never emit `tool_calls`). The correct parser name depends on
   which Gemma 4 chat template vLLM ships for tool calling — check vLLM's docs/release
   notes for the current Gemma-compatible parser name, since this is the part most
   likely to have changed between vLLM versions.
2. **`--enable-reasoning` + `--reasoning-parser`** — without this, the `/think` panel in
   `ui.py` will just stop showing anything (it'll silently fall through — no crash,
   just no thinking text streamed). If your vLLM build doesn't support a Gemma reasoning
   parser yet, the `<think>...</think>` tag fallback in `ui.py` still has a chance of
   catching it if the model emits the tags in plain content instead.

I haven't verified the exact tool-call-parser / reasoning-parser names for Gemma 4
against the vLLM version you're running — those change across vLLM releases, so check
your installed vLLM's `--help` output or release notes before launch.

## Quick sanity test after switching

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4-e4b","messages":[{"role":"user","content":"<|think|>\nsay hi"}],"stream":false}'
```
Confirms the server is up, the model name matches `VLLM_MODEL`, and the `<|think|>`
token still triggers reasoning before you debug it through the full app.
