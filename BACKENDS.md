# Inference Backend Guide

Set `INFERENCE_BACKEND` in `.env` to one of: `vllm`, `ollama`, `llamacpp`

## vLLM

```bash
vllm serve <hf-model-id-or-path> \
  --served-model-name gemma4-e4b \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser <gemma4-parser> \
  --enable-reasoning \
  --reasoning-parser <gemma4-parser> \
  --gpu-memory-utilization 0.9
```

Tool calling and `reasoning_content` streaming both require the server flags above.
Without `--enable-reasoning`, thinking still works via the `<think>...</think>` tag
fallback in `ui.py`.

## Ollama

```bash
ollama serve
ollama pull gemma4:e4b
```

Set `NUM_GPU` to the number of GPU layers to offload (leave blank for CPU-only).
`NUM_THREAD` controls CPU thread count (defaults to half of logical cores).

## llama.cpp (llama-server)

```bash
llama-server \
  -m ./models/gemma4.gguf \
  --port 8080 \
  --ctx-size 32768 \
  --n-predict 2048 \
  -ngl 99 \
  --jinja \
  --reasoning-format deepseek
```

`--jinja` enables the Jinja2 chat template which is required for tool calling.
`--reasoning-format deepseek` makes llama-server emit `<think>...</think>` tags
in content; `ui.py`'s `ThinkParser` handles these automatically.
Thinking content does NOT come back as `reasoning_content` with llama-server —
only the tag-based fallback path in `ui.py` is used.
