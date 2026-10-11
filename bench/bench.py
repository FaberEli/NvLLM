#!/usr/bin/env python
"""nvllm-serve benchmark: TTFT, TPOT, and throughput through the proxy API.

Targets the service's OpenAI-compatible `/v1/chat/completions` (streaming), so we
measure what a real client sees end-to-end (discovery -> worker -> vLLM -> back).

Metrics per request:
  - TTFT  : time to first content token (s)
  - TPOT  : (total - TTFT) / (out_tokens - 1), avg time per generated token (s)
  - out   : output tokens (counted as non-empty content deltas)

Aggregate:
  - agg tok/s : total output tokens / total wall-clock across the batch

Usage:
  .venv/bin/python bench/bench.py --model nvidia/Llama-3.1-8B-Instruct-NVFP4 \
      --concurrency 1 4 8 --requests 6 --prompt-words 256 --max-tokens 128
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass

import httpx

BASE = "http://127.0.0.1"  # host is fixed by the service (worker binds 127.0.0.1)


def build_prompt(words: int) -> str:
    """A deterministic filler passage of roughly `words` words (stable TTFT/TPOT)."""
    base = (
        "The quick brown fox jumps over the lazy dog while the river runs steady "
        "and the old lighthouse keeps its beam on the dark water at night. "
    )
    reps = max(1, words // len(base.split())) + 1
    text = (base * reps)
    return " ".join(text.split()[:words])


@dataclass
class Result:
    ttft: float
    total: float
    out_tokens: int


async def run_one(client: httpx.AsyncClient, port: int, model: str, prompt: str, max_tokens: int) -> Result:
    t0 = time.perf_counter()
    ttft = None
    out = 0
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    async with client.stream("POST", f"{BASE}:{port}/v1/chat/completions", json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except (ValueError, json.JSONDecodeError):
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            if delta.get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                out += 1
    total = time.perf_counter() - t0
    if ttft is None:
        ttft = total
    return Result(ttft=ttft, total=total, out_tokens=out)


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


async def run_batch(port: int, model: str, prompt: str, max_tokens: int, concurrency: int, requests: int) -> dict:
    timeout = httpx.Timeout(600.0, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        sem = asyncio.Semaphore(concurrency)

        async def task() -> Result:
            async with sem:
                return await run_one(client, port, model, prompt, max_tokens)

        wall = time.perf_counter()
        results = await asyncio.gather(*[task() for _ in range(requests)])
        wall = time.perf_counter() - wall

    ttfts = [r.ttft for r in results]
    totals = [r.total for r in results]
    outs = [r.out_tokens for r in results]
    tpots = [(r.total - r.ttft) / (r.out_tokens - 1) for r in results if r.out_tokens > 1]
    total_out = sum(outs)
    return {
        "concurrency": concurrency,
        "requests": requests,
        "avg_out_tokens": round(statistics.mean(outs), 1) if outs else 0,
        "ttft_ms_mean": round(statistics.mean(ttfts) * 1000, 1) if ttfts else None,
        "ttft_ms_p95": round(_pct(ttfts, 0.95) * 1000, 1) if ttfts else None,
        "tpot_ms_mean": round(statistics.mean(tpots) * 1000, 2) if tpots else None,
        "per_stream_tok_s": round(statistics.mean([1.0 / t for t in tpots]) if tpots else None, 2),
        "agg_tok_s": round(total_out / wall, 1) if wall else None,
        "wall_s": round(wall, 1),
    }


async def sweep(args) -> None:
    prompt = build_prompt(args.prompt_words)
    print(
        f"# bench: model={args.model} prompt_words={args.prompt_words} max_tokens={args.max_tokens} "
        f"requests/level={args.requests}"
    )
    header = (
        f"{'conc':>5} {'reqs':>5} {'out':>6} {'ttft_ms':>9} {'ttft_p95':>10} "
        f"{'tpot_ms':>9} {'str_tok/s':>10} {'agg_tok/s':>10} {'wall_s':>8}"
    )
    print(header)
    print("-" * len(header))
    for conc in args.concurrency:
        row = await run_batch(args.port, args.model, prompt, args.max_tokens, conc, args.requests)
        print(
            f"{row['concurrency']:>5} {row['requests']:>5} {row['avg_out_tokens']:>6} "
            f"{row['ttft_ms_mean']:>9} {row['ttft_ms_p95']:>10} "
            f"{row['tpot_ms_mean']:>9} {row['per_stream_tok_s']:>10} "
            f"{row['agg_tok_s']:>10} {row['wall_s']:>8}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="nvllm-serve throughput/latency benchmark")
    p.add_argument("--port", type=int, default=8004)
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1])
    p.add_argument("--requests", type=int, default=6, help="requests per concurrency level")
    p.add_argument("--prompt-words", type=int, default=256)
    p.add_argument("--max-tokens", type=int, default=128)
    args = p.parse_args()
    asyncio.run(sweep(args))


if __name__ == "__main__":
    main()
