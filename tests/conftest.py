"""Shared fixtures: a fake HF hub cache and a stub OpenAI-compatible worker server."""
from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.routing import Route

_SHA = "a" * 40


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_stub_vllm(captured: list) -> Starlette:
    """A minimal stand-in for `vllm serve`: OpenAI-compatible endpoints.

    `captured` records the OpenAI request bodies the stub receives, so tests can
    assert on the exact parameters the proxy forwarded to the worker.
    """

    async def health(_request):
        return PlainTextResponse("OK")

    async def chat(request):
        body = await request.json()
        captured.append(body)
        model = body.get("model")
        if body.get("stream"):
            async def gen():
                for piece in ("Hello", " world"):
                    chunk = {
                        "id": "c1",
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield f"data: {json.dumps({'id': 'c1', 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                yield f"data: {json.dumps({'id': 'c1', 'model': model, 'choices': [], 'usage': {'prompt_tokens': 7, 'completion_tokens': 2, 'total_tokens': 9}})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")
        return JSONResponse(
            {
                "id": "c1",
                "object": "chat.completion",
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello world"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
            }
        )

    async def completions(request):
        body = await request.json()
        model = body.get("model")
        if body.get("stream"):
            async def gen():
                yield f"data: {json.dumps({'id': 'k1', 'model': model, 'choices': [{'index': 0, 'text': 'stub-'}]})}\n\n"
                yield f"data: {json.dumps({'id': 'k1', 'model': model, 'choices': [{'index': 0, 'text': 'completion'}]})}\n\n"
                yield f"data: {json.dumps({'id': 'k1', 'model': model, 'choices': [{'index': 0, 'text': '', 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 4, 'completion_tokens': 2, 'total_tokens': 6}})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")
        return JSONResponse(
            {
                "id": "k1",
                "object": "text_completion",
                "model": model,
                "choices": [{"index": 0, "text": "stub-completion", "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            }
        )

    async def embeddings(request):
        body = await request.json()
        inp = body["input"]
        if isinstance(inp, str):
            inp = [inp]
        return JSONResponse(
            {
                "object": "list",
                "model": body.get("model"),
                "data": [
                    {"object": "embedding", "index": i, "embedding": [round(0.1 * (i + 1), 1), 0.2, 0.3]}
                    for i in range(len(inp))
                ],
                "usage": {"prompt_tokens": 3, "total_tokens": 3},
            }
        )

    return Starlette(
        routes=[
            Route("/health", health),
            Route("/v1/chat/completions", chat, methods=["POST"]),
            Route("/v1/completions", completions, methods=["POST"]),
            Route("/v1/embeddings", embeddings, methods=["POST"]),
        ]
    )


@pytest.fixture()
def stub_captured() -> list:
    """OpenAI request bodies received by the stub worker (see make_stub_vllm)."""
    return []


@pytest.fixture()
def stub_server(stub_captured):
    """Run the stub vLLM server in a thread; yields its port."""
    port = _free_port()
    config = uvicorn.Config(make_stub_vllm(stub_captured), host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0).status_code == 200:
                return port
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    raise RuntimeError("stub worker server did not start in time")


@pytest.fixture()
def fake_hub(tmp_path: Path) -> Path:
    """A fake HF hub cache with three models: text, embedding, unsupported (video)."""
    root = tmp_path / "hub"

    def make(org: str, name: str, cfg: dict, tokenizer_cfg: dict | None = None) -> None:
        mdir = root / f"models--{org}--{name}"
        (mdir / "refs").mkdir(parents=True)
        (mdir / "refs" / "main").write_text(_SHA)
        snap = mdir / "snapshots" / _SHA
        snap.mkdir(parents=True)
        (snap / "config.json").write_text(json.dumps(cfg))
        if tokenizer_cfg is not None:
            (snap / "tokenizer_config.json").write_text(json.dumps(tokenizer_cfg))
        (snap / "model.safetensors").write_bytes(b"\0" * 1024)

    make(
        "acme",
        "TinyLM",
        {"architectures": ["TinyLMForCausalLM"], "model_type": "tinylm", "hidden_size": 128, "num_hidden_layers": 2, "vocab_size": 512},
        {"chat_template": "{% for m in messages %}{{m.content}}{% endfor %}"},
    )
    make("acme", "Embedder", {"architectures": ["EmbedderModel"], "model_type": "embedder", "hidden_size": 64, "pooling": "mean"})
    make("acme", "WanVideo", {"architectures": ["WanForTextToVideo"], "model_type": "wan"})
    return root


@pytest.fixture()
def settings(fake_hub: Path):
    from nvllm_service.config import Settings

    return Settings(hf_hub_root=fake_hub)


@pytest.fixture()
def client(settings, stub_server, monkeypatch):
    """TestClient for the service with WorkerManager.ensure pointed at the stub server."""
    from fastapi.testclient import TestClient

    from nvllm_service.main import create_app
    from nvllm_service.workers import STATE_READY, Worker

    app = create_app(settings)
    manager = app.state.manager

    async def fake_ensure(info, keep_alive=None, is_judge=False):
        w = manager.workers.get(info.name)
        if w is None or w.state != STATE_READY:
            w = Worker(info=info, port=stub_server, state=STATE_READY, pid=9999)
            manager.workers[info.name] = w
        return w

    monkeypatch.setattr(manager, "ensure", fake_ensure)
    with TestClient(app) as c:
        yield c
