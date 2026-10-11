# nvllm-serve

Ollama-style resident inference service backed by vLLM — part of the **NvLLM**
model-backend inference engine.

Built to serve **NVFP4-quantized** models at full speed on **DGX Spark (GB10)**
unified memory. It discovers models from your local HuggingFace hub cache,
serves them on a single fixed port (default **8004**), and exposes both the
**Ollama** (`/api/*`) and **OpenAI** (`/v1/*`) APIs. Each model runs in its own
`vllm serve` subprocess so crashes are isolated and the supervisor can reap and
respawn.

## How it works

- **Discovery** — scans `HF_HUB_CACHE` (or `~/.cache/huggingface/hub`) and builds a
  catalog of models, detecting capabilities (completion / embedding / tools) and
  flagging anything vLLM can't serve.
- **Workers** — one `vllm serve` process per model. The manager starts it on a
  free port, polls `/health` until ready, and reaps crashed workers. Up to
  `--max-models` non-judge models (default `1`) may be resident at once; when a
  new model would exceed that, the least-valued resident is evicted (expired
  workers first, then least-recently-used). Idle workers past their `--keep-alive`
  are unloaded automatically. The configured Router judge is never evicted.
  Loading is **serialized** (one model loads at a time — safe on shared/unified
  memory), while already-loaded models serve **concurrently**.
- **Proxy** — each request is resolved against the catalog, a worker is ensured,
  then the call is forwarded. Ollama bodies are translated to OpenAI and back,
  including streaming (OpenAI SSE in, Ollama NDJSON out).

## Routing (auto model selection)

Set `--router-judge` to a small, fast model (it stays resident and is never
evicted). Then any inference request can use `model: "auto"`: the judge reads the
request plus the list of servable models, picks the best one, and the service
forwards to it. If the judge can't be consulted or returns something
unrecognizable, the router falls back to the largest model rather than failing.

```bash
nvllm-serve --router-judge nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4

curl -s localhost:8004/api/router | jq          # routing status + candidates
curl -s localhost:8004/api/chat -d '{
  "model": "auto",
  "messages": [{"role": "user", "content": "hello"}]
}'
```

## Quickstart

```bash
# default bind is 0.0.0.0:8004
# from the service/ directory
.venv/bin/python -m nvllm_service
# or, once installed (pip install -e .):
nvllm-serve
```

Then:

```bash
curl -s localhost:8004/api/tags | jq
curl -s localhost:8004/v1/models | jq

# Ollama chat
curl -s localhost:8004/api/chat -d '{
  "model": "org/Model",
  "messages": [{"role": "user", "content": "hello"}],
  "stream": false
}' | jq

# OpenAI chat
curl -s localhost:8004/v1/chat/completions -d '{
  "model": "org/Model",
  "messages": [{"role": "user", "content": "hello"}]
}' | jq
```

## Configuration

Precedence: **CLI flag > environment variable > default**.

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--host` | `NVLLM_HOST` | `0.0.0.0` | bind host |
| `--port` | `NVLLM_PORT` | `8004` | bind port |
| `--hf-hub-root` | `HF_HUB_CACHE` / `HF_HOME` | `~/.cache/huggingface/hub` | HF cache to scan |
| `--keep-alive` | `NVLLM_KEEP_ALIVE` | `300` | idle seconds before unload (`-1` keep forever, `0` unload immediately) |
| `--max-models` | `NVLLM_MAX_MODELS` | `1` | max non-judge models resident at once (judge excluded) |
| `--max-model-len` | `NVLLM_MAX_MODEL_LEN` | (vLLM default) | passed to workers |
| `--gpu-memory-utilization` | `NVLLM_GPU_MEMORY_UTILIZATION` | `0.5` | passed to workers |
| `--router-judge` | `NVLLM_ROUTER_JUDGE` | — | model name to act as the Router judge |

Also honored by workers: `NVLLM_MAX_NUM_SEQS`, `NVLLM_TRUST_REMOTE_CODE`,
`NVLLM_WORKER_MAX_JOBS`, and `NVLLM_WORKER_EXTRA_ARGS` (raw args appended to every
`vllm serve`, e.g. `"--enable-auto-tool-choice --tool-call-parser hermes"`).

> **DGX Spark (GB10):** the OS/desktop consumes a large share of the 128G
> unified memory, so vLLM's default `gpu-memory-utilization` of 0.92 fails at
> startup. `0.5` is a safe default there; raise it on a dedicated machine.

> **NVFP4 first load (DGX Spark):** the first request JIT-compiles the
> FlashInfer CUTLASS NVFP4 GEMM kernels into `~/.cache/flashinfer`. That is a
> large, memory-hungry one-time build — at the default parallelism it
> OOM-kills `nvcc` (SIGKILL / exit 137) on the Spark's shared 128G. Set
> `NVLLM_WORKER_MAX_JOBS=2` (or 3–4). The kernels are cached, so subsequent
> loads are fast.

## Multi-model concurrency

By default only **one** non-judge model is resident at a time (`--max-models 1`)
— safe on the Spark's shared 128G. Raise it to keep several models warm:

```bash
# keep two 8B-class models resident at once
NVLLM_MAX_MODELS=2 NVLLM_GPU_MEMORY_UTILIZATION=0.35 .venv/bin/python -m nvllm_service
```

- **Residency is concurrent, loading is serial.** Already-loaded models serve
  requests in parallel, but only one model loads at a time (two simultaneous
  `vllm serve` loads — each JIT-compiling kernels — can OOM the shared memory).
  A request for a not-yet-loaded model waits its turn.
- **Eviction is LRU, expired-first.** When a new model would exceed
  `--max-models`, the manager evicts expired residents first, then the
  least-recently-used. The judge is never evicted.
- **`keep-alive` now really unloads.** Idle workers past their keep-alive are
  reaped by the monitor; a worker mid-request is never killed.
- **Memory math (DGX Spark, 128G):** `gpu-memory-utilization` is per-worker, as
  a fraction of *total* device memory. So `N` workers × `U` must fit: at the
  default `U=0.5`, two workers already claim the whole 128G. For `N` models use
  roughly `U ≈ headroom / N` (e.g. `N=2 → U≈0.35`).

## Endpoints

| Method | Path | API |
|---|---|---|
| GET | `/healthz`, `/api/version` | meta |
| GET | `/api/tags` | Ollama — list models |
| GET | `/api/ps` | Ollama — loaded models |
| POST | `/api/show` | Ollama — model info |
| POST | `/api/chat`, `/api/generate` | Ollama — inference (stream + non-stream) |
| POST | `/api/embed`, `/api/embeddings` | Ollama — embeddings |
| GET | `/api/router` | routing status (judge + candidates) |
| GET | `/v1/models` | OpenAI — list models |
| POST | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` | OpenAI — inference |

## Tests

```bash
.venv/bin/python -m pytest -q
```

The suite uses a fake HF hub cache and a stub OpenAI-compatible worker, so it
runs without a GPU or a real vLLM install.

## Milestones

- **M1** — catalog discovery + metadata API + worker process management. ✅
- **M2** — inference proxy (Ollama + OpenAI, streaming). ✅
- **M3** — Router / auto-routing (`--router-judge` + `model: "auto"`). ✅
- **M4** — end-to-end NVFP4 smoke test on DGX Spark (GB10): `Llama-3.1-8B-Instruct-NVFP4`
  served over both `/api/chat` (Ollama) and `/v1/chat/completions` (OpenAI). ✅
- **M5** — multi-model concurrency: `--max-models` (co-located non-judge models,
  LRU/expired-first eviction), `keep-alive` expiry reaping, and in-flight protection
  so a request is never cut off. ✅
- **Next** — throughput/latency tuning for NVFP4, and a Router smoke test with a
  real judge model.
