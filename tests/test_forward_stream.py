"""Streaming forwards must hold the worker's in-flight reference for the whole
stream, and release it when the stream ends.

Regression: `main.py` releases its `track()` reference the moment the endpoint
returns, but a streaming body is delivered *after* that (uvicorn iterates the
response generator later). Without the forward holding the reference itself, a
`keep-alive=0` worker would be reaped by the monitor while it was still sending
the stream, so the client got a partial response + error.
"""
from __future__ import annotations

import asyncio

from nvllm_service import forward
from nvllm_service.catalog import ModelInfo
from nvllm_service.workers import STATE_READY, Worker


def _worker(port: int) -> tuple[Worker, str]:
    info = ModelInfo(repo="acme/TinyLM", tag="main", commit="c" * 40, snapshot_dir="/x")
    return Worker(info=info, port=port, state=STATE_READY), info.name


def test_ollama_chat_stream_holds_inflight(stub_server):
    w, name = _worker(stub_server)
    assert w.inflight == 0

    async def scenario() -> bytes:
        resp = await forward.ollama_chat_stream(
            w, name, {"messages": [{"role": "user", "content": "hi"}]}, 0.0
        )
        assert w.inflight == 1, "streaming forward must hold the worker while the stream is live"
        body = b"".join([c async for c in resp.body_iterator])
        assert w.inflight == 0, "streaming forward must release the worker after the stream ends"
        return body

    body = asyncio.run(scenario())
    assert b'"done": true' in body


def test_ollama_generate_stream_holds_inflight(stub_server):
    w, name = _worker(stub_server)
    assert w.inflight == 0

    async def scenario() -> bytes:
        resp = await forward.ollama_generate_stream(
            w, name, {"prompt": "hi"}, 0.0
        )
        assert w.inflight == 1, "streaming forward must hold the worker while the stream is live"
        body = b"".join([c async for c in resp.body_iterator])
        assert w.inflight == 0, "streaming forward must release the worker after the stream ends"
        return body

    body = asyncio.run(scenario())
    assert b'"done": true' in body


def test_openai_passthrough_holds_inflight(stub_server):
    w, name = _worker(stub_server)
    assert w.inflight == 0

    async def scenario() -> bytes:
        resp = await forward.openai_passthrough(
            w, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "stream": True}
        )
        assert w.inflight == 1, "streaming forward must hold the worker while the stream is live"
        body = b"".join([c async for c in resp.body_iterator])
        assert w.inflight == 0, "streaming forward must release the worker after the stream ends"
        return body

    body = asyncio.run(scenario())
    assert b"chat.completion.chunk" in body


def test_streaming_inflight_released_on_client_abort(stub_server):
    """If the client stops reading, the reference is still released (no leak)."""
    w, name = _worker(stub_server)
    assert w.inflight == 0

    async def scenario() -> None:
        resp = await forward.ollama_chat_stream(
            w, name, {"messages": [{"role": "user", "content": "hi"}]}, 0.0
        )
        gen = resp.body_iterator
        first = await anext(gen)
        assert b'"done": false' in first
        # client aborts: close the generator without draining it
        await gen.aclose()
        assert w.inflight == 0, "aborted stream must still release the worker reference"

    asyncio.run(scenario())
