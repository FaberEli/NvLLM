"""Proxying to vLLM workers: request translation and streaming forwarding.

Workers are `vllm serve` processes and speak OpenAI natively, so:

- /v1/* endpoints -> OpenAI body passthrough (only the model name is rewritten)
- /api/chat, /api/generate -> Ollama body in, OpenAI out; OpenAI SSE stream
  translated back to Ollama's NDJSON stream on the way home

Streaming helpers are async so that connect failures / worker 4xx surface as
proper HTTP errors *before* the response stream has started.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from .workers import Worker

log = logging.getLogger(__name__)

# streaming: generation can legitimately take a long time, so no read timeout;
# connect/write are bounded
_STREAM_TIMEOUT = httpx.Timeout(connect=15.0, read=None, write=60.0, pool=15.0)

_KEEP_ALIVE_RE = re.compile(r"^(\d+)(ms|s|m|h)?$")


def _http(status_code: int, detail: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail=detail)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _usage_counts(usage: Any) -> tuple[int, int]:
    if not isinstance(usage, dict):
        return 0, 0
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


# ---- keep_alive -----------------------------------------------------------

def parse_keep_alive(value: Any) -> int | None:
    """Parse Ollama keep_alive: seconds (int) or a duration string like '30s'/'5m'."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    m = _KEEP_ALIVE_RE.match(str(value).strip())
    if not m:
        return None
    n = int(m.group(1))
    mult = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}.get(m.group(2) or "s", 1)
    return int(n * mult)


# ---- Ollama request -> OpenAI request --------------------------------------

def _options_to_openai(options: Any) -> dict[str, Any]:
    """Map Ollama `options` onto the OpenAI parameters vLLM understands.

    top_k has no OpenAI counterpart and is dropped; num_predict -> max_tokens.
    """
    if not isinstance(options, dict):
        return {}
    out: dict[str, Any] = {}
    for key in ("temperature", "top_p", "stop", "seed", "frequency_penalty", "presence_penalty"):
        if options.get(key) is not None:
            out[key] = options[key]
    if options.get("num_predict") is not None:
        out["max_tokens"] = int(options["num_predict"])
    return out


def _ollama_message_to_openai(msg: dict) -> dict[str, Any]:
    out: dict[str, Any] = {"role": msg.get("role", "user")}

    tcs = msg.get("tool_calls") or []
    if tcs:
        out["tool_calls"] = []
        for i, tc in enumerate(tcs):
            fn = (tc or {}).get("function") or {}
            args = fn.get("arguments", {})
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            out["tool_calls"].append(
                {
                    "id": tc.get("id") or f"call_{i}",
                    "type": "function",
                    "function": {"name": fn.get("name", ""), "arguments": args},
                }
            )

    text = msg.get("content")
    images = msg.get("images") or []
    if text or images:
        if images:
            parts: list[dict] = []
            if text:
                parts.append({"type": "text", "text": text})
            for img in images:
                url = str(img)
                if not url.startswith("data:"):
                    url = f"data:image/png;base64,{url}"
                parts.append({"type": "image_url", "image_url": {"url": url}})
            out["content"] = parts
        else:
            out["content"] = text if text is not None else ""
    return out


def openai_chat_body(model_name: str, body: dict) -> dict[str, Any]:
    oai: dict[str, Any] = {
        "model": model_name,
        "stream": bool(body.get("stream", False)),
        "messages": [_ollama_message_to_openai(m) for m in (body.get("messages") or []) if isinstance(m, dict)],
    }
    oai.update(_options_to_openai(body.get("options")))
    if body.get("tools"):
        # Ollama and OpenAI tools share the same shape
        oai["tools"] = body["tools"]
        if body.get("tool_choice"):
            oai["tool_choice"] = body["tool_choice"] if isinstance(body["tool_choice"], str) else "auto"
    return oai


def openai_completions_body(model_name: str, body: dict) -> dict[str, Any]:
    oai: dict[str, Any] = {
        "model": model_name,
        "prompt": body.get("prompt", ""),
        "stream": bool(body.get("stream", False)),
    }
    oai.update(_options_to_openai(body.get("options")))
    return oai


# ---- worker error handling -------------------------------------------------

async def _worker_error(resp: httpx.Response) -> HTTPException:
    detail = ""
    try:
        # streaming responses must be materialized before .json() (ResponseNotRead);
        # note httpx's Response.json() is synchronous (only aread() is a coroutine)
        if not resp.is_stream_consumed and not resp.is_closed:
            await resp.aread()
        body = resp.json()
    except (json.JSONDecodeError, ValueError, httpx.HTTPError):
        body = None
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            detail = str(err.get("message") or err)
        elif err is not None:
            detail = str(err)
        elif "message" in body:
            detail = str(body["message"])
    if not detail:
        try:
            detail = (await resp.text())[:300]
        except (httpx.HTTPError, ValueError):
            pass
        detail = detail or f"worker returned HTTP {resp.status_code}"
    return _http(resp.status_code, detail)


# ---- forwarding primitives -------------------------------------------------

async def forward_json(worker: Worker, path: str, body: dict) -> Any:
    """POST to the worker, wait for the full response, return parsed JSON."""
    payload = dict(body)
    payload["model"] = worker.name
    try:
        async with httpx.AsyncClient(timeout=_STREAM_TIMEOUT) as client:
            resp = await client.post(f"{worker.base_url}{path}", json=payload)
    except httpx.HTTPError as e:
        raise _http(502, f"cannot reach worker: {e}")
    if resp.status_code >= 400:
        raise await _worker_error(resp)
    try:
        return resp.json()
    except (json.JSONDecodeError, ValueError):
        raise _http(502, "worker returned a non-JSON response")


def _rewrite_sse_line(line: str, as_model: str) -> str:
    """Rewrite one SSE line's `model` field (virtual-model name echo).

    Non-data lines (and `data: [DONE]`) pass through untouched; a data payload
    that does not parse as a JSON object is left as-is rather than mangled.
    """
    line = line.strip()
    if line.startswith("data:"):
        payload = line[len("data:") :].strip()
        if payload and payload != "[DONE]":
            try:
                evt = json.loads(payload)
            except json.JSONDecodeError:
                return line
            if isinstance(evt, dict):
                evt["model"] = as_model
                return f"data: {json.dumps(evt, ensure_ascii=False)}"
    return line


async def openai_passthrough(worker: Worker, path: str, body: dict, as_model: str | None = None) -> StreamingResponse:
    """Forward an OpenAI request, streaming the worker's bytes back verbatim.

    Holds the worker's in-flight reference for the whole stream (released in the
    generator's ``finally``). The response body is delivered *after* the endpoint
    returns — outside the endpoint's own ``track()`` window — so without this a
    ``keep-alive=0`` worker could be reaped by the monitor mid-stream.

    ``as_model`` (optional): echo this name in each SSE event's ``model`` field
    instead of the worker's — used so a created (virtual) model answers under
    the name the client requested.
    """
    payload = dict(body)
    payload["model"] = worker.name
    client = httpx.AsyncClient(timeout=_STREAM_TIMEOUT)
    try:
        resp = await client.send(client.build_request("POST", f"{worker.base_url}{path}", json=payload), stream=True)
    except httpx.HTTPError as e:
        await client.aclose()
        raise _http(502, f"cannot reach worker: {e}")
    if resp.status_code >= 400:
        exc = await _worker_error(resp)
        await resp.aclose()
        await client.aclose()
        raise exc
    media_type = resp.headers.get("content-type", "application/json")
    worker.inflight += 1

    async def gen() -> AsyncIterator[bytes]:
        try:
            if as_model is None:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            else:
                # line-buffer the byte stream so a `model` value split across
                # chunks is still rewritten correctly
                buf = ""
                async for chunk in resp.aiter_bytes():
                    buf += chunk.decode("utf-8", errors="replace")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        yield (_rewrite_sse_line(line, as_model) + "\n").encode("utf-8")
                if buf.strip():
                    yield _rewrite_sse_line(buf, as_model).encode("utf-8")
        finally:
            await resp.aclose()
            await client.aclose()
            worker.inflight = max(0, worker.inflight - 1)

    return StreamingResponse(gen(), media_type=media_type)


async def _openai_stream(worker: Worker, path: str, body: dict) -> tuple[httpx.Response, httpx.AsyncClient]:
    """Start a streaming OpenAI request; returns (response, client) — caller owns both."""
    payload = dict(body)
    payload["model"] = worker.name
    payload["stream"] = True
    payload.setdefault("stream_options", {"include_usage": True})
    client = httpx.AsyncClient(timeout=_STREAM_TIMEOUT)
    try:
        resp = await client.send(client.build_request("POST", f"{worker.base_url}{path}", json=payload), stream=True)
    except httpx.HTTPError as e:
        await client.aclose()
        raise _http(502, f"cannot reach worker: {e}")
    if resp.status_code >= 400:
        exc = await _worker_error(resp)
        await resp.aclose()
        await client.aclose()
        raise exc
    return resp, client


async def _iter_sse_events(resp: httpx.Response) -> AsyncIterator[dict]:
    async for line in resp.aiter_lines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            log.debug("skipping unparseable SSE payload: %.200s", payload)


def _ndjson(obj: dict) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n"


def _finish_reason(fr: Any) -> str:
    # Ollama only uses "stop" / "length"
    return "length" if fr == "length" else "stop"


def _openai_tool_call_to_ollama(tc: dict) -> dict:
    fn = tc.get("function") or {}
    args = fn.get("arguments", "")
    if isinstance(args, str):
        try:
            args = json.loads(args) if args else {}
        except json.JSONDecodeError:
            args = {}
    return {"function": {"name": fn.get("name", ""), "arguments": args}}


# ---- Ollama /api/chat ------------------------------------------------------

def ollama_chat_response(model_name: str, oai_resp: dict, t0: float) -> dict:
    """One-shot (non-streaming) chat: OpenAI response -> Ollama response."""
    choice = (oai_resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    out_msg: dict[str, Any] = {"role": msg.get("role", "assistant")}
    if msg.get("content") is not None:
        out_msg["content"] = msg["content"]
    if msg.get("tool_calls"):
        out_msg["tool_calls"] = [_openai_tool_call_to_ollama(tc) for tc in msg["tool_calls"]]
    prompt_tokens, completion_tokens = _usage_counts(oai_resp.get("usage"))
    return {
        "model": model_name,
        "created_at": _now_iso(),
        "message": out_msg,
        "done": True,
        "done_reason": _finish_reason(choice.get("finish_reason")),
        "total_duration": int((time.time() - t0) * 1e9),
        "prompt_eval_count": prompt_tokens,
        "eval_count": completion_tokens,
    }


class _ChatStreamAcc:
    """Accumulates tool-call fragments across SSE chunks (keyed by index)."""

    def __init__(self) -> None:
        self.tcs: dict[int, dict] = {}
        self.finish: str | None = None
        self.usage: dict | None = None

    def merged_tool_calls(self) -> list[dict]:
        merged = []
        for idx in sorted(self.tcs):
            slot = self.tcs[idx]
            try:
                args = json.loads(slot["arguments"]) if slot["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            merged.append({"function": {"name": slot["name"], "arguments": args}})
        return merged

    def apply(self, evt: dict) -> list[dict]:
        """Consume one OpenAI SSE chunk; return Ollama message payloads to emit."""
        out: list[dict] = []
        for choice in evt.get("choices") or []:
            if choice.get("finish_reason"):
                self.finish = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if delta.get("tool_calls"):
                for tc in delta["tool_calls"]:
                    idx = int(tc.get("index", 0))
                    slot = self.tcs.setdefault(idx, {"name": "", "arguments": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
                msg = {"role": "assistant"}
                calls = self.merged_tool_calls()
                if calls:
                    msg["tool_calls"] = calls
                out.append(msg)
            text = delta.get("content")
            if text:
                out.append({"role": "assistant", "content": text})
        if isinstance(evt.get("usage"), dict):
            self.usage = evt["usage"]
        return out


async def ollama_chat_stream(worker: Worker, model_name: str, oai_body: dict, t0: float) -> StreamingResponse:
    """Streaming chat: OpenAI SSE in, Ollama NDJSON out.

    Holds the worker's in-flight reference for the whole stream (released in the
    generator's ``finally``) so the monitor cannot reap a ``keep-alive=0`` worker
    while the response is still being delivered.
    """
    resp, client = await _openai_stream(worker, "/v1/chat/completions", oai_body)
    acc = _ChatStreamAcc()
    created = _now_iso()
    worker.inflight += 1

    async def gen() -> AsyncIterator[bytes]:
        try:
            yield _ndjson({"model": model_name, "created_at": created, "message": {"role": "assistant", "content": ""}, "done": False})
            try:
                async for evt in _iter_sse_events(resp):
                    for msg in acc.apply(evt):
                        yield _ndjson({"model": model_name, "message": msg, "done": False})
                prompt_tokens, completion_tokens = _usage_counts(acc.usage)
                yield _ndjson(
                    {
                        "model": model_name,
                        "message": {"role": "assistant"},
                        "done": True,
                        "done_reason": _finish_reason(acc.finish),
                        "total_duration": int((time.time() - t0) * 1e9),
                        "prompt_eval_count": prompt_tokens,
                        "eval_count": completion_tokens,
                    }
                )
            except httpx.HTTPError as e:
                log.error("chat stream from worker %s interrupted: %s", worker.name, e)
                yield _ndjson({"model": model_name, "error": f"stream interrupted: {e}", "done": False})
        finally:
            await resp.aclose()
            await client.aclose()
            worker.inflight = max(0, worker.inflight - 1)

    return StreamingResponse(gen(), media_type="application/x-ndjson")


# ---- Ollama /api/generate --------------------------------------------------

def ollama_generate_response(model_name: str, oai_resp: dict, t0: float) -> dict:
    choice = (oai_resp.get("choices") or [{}])[0]
    prompt_tokens, completion_tokens = _usage_counts(oai_resp.get("usage"))
    return {
        "model": model_name,
        "created_at": _now_iso(),
        "response": choice.get("text") or "",
        "done": True,
        "done_reason": _finish_reason(choice.get("finish_reason")),
        "total_duration": int((time.time() - t0) * 1e9),
        "prompt_eval_count": prompt_tokens,
        "eval_count": completion_tokens,
    }


async def ollama_generate_stream(worker: Worker, model_name: str, oai_body: dict, t0: float) -> StreamingResponse:
    """Streaming completions: OpenAI SSE in, Ollama NDJSON out.

    Holds the worker's in-flight reference for the whole stream (released in the
    generator's ``finally``) so the monitor cannot reap a ``keep-alive=0`` worker
    while the response is still being delivered.
    """
    resp, client = await _openai_stream(worker, "/v1/completions", oai_body)
    created = _now_iso()
    worker.inflight += 1

    async def gen() -> AsyncIterator[bytes]:
        try:
            yield _ndjson({"model": model_name, "created_at": created, "response": "", "done": False})
            try:
                finish: str | None = None
                usage: dict | None = None
                async for evt in _iter_sse_events(resp):
                    for choice in evt.get("choices") or []:
                        if choice.get("finish_reason"):
                            finish = choice["finish_reason"]
                        text = (choice.get("delta") or {}).get("text") or choice.get("text")
                        if text:
                            yield _ndjson({"model": model_name, "response": text, "done": False})
                    if isinstance(evt.get("usage"), dict):
                        usage = evt["usage"]
                prompt_tokens, completion_tokens = _usage_counts(usage)
                yield _ndjson(
                    {
                        "model": model_name,
                        "response": "",
                        "done": True,
                        "done_reason": _finish_reason(finish),
                        "total_duration": int((time.time() - t0) * 1e9),
                        "prompt_eval_count": prompt_tokens,
                        "eval_count": completion_tokens,
                    }
                )
            except httpx.HTTPError as e:
                log.error("generate stream from worker %s interrupted: %s", worker.name, e)
                yield _ndjson({"model": model_name, "error": f"stream interrupted: {e}", "done": False})
        finally:
            await resp.aclose()
            await client.aclose()
            worker.inflight = max(0, worker.inflight - 1)

    return StreamingResponse(gen(), media_type="application/x-ndjson")
