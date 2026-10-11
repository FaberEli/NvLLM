"""Router (M3): model auto-selection via a judge model.

Unit tests exercise the pure Router logic (prompt, parsing, fallback). The
integration test runs the real app with a judge configured and a stub worker
that, when it sees the router's system prompt, answers with a chosen model name;
the service must then forward the original request to that model.
"""
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
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from nvllm_service.catalog import Capability, ModelInfo
from nvllm_service.router import ROUTE_ALIAS, Router

# ---- pure Router unit tests --------------------------------------------------


def _mi(repo: str, caps, size: int = 0, params: int = 0) -> ModelInfo:
    return ModelInfo(
        repo=repo,
        tag="main",
        commit="c",
        snapshot_dir=f"/snap/{repo}",
        size=size,
        parameter_count=params,
        capabilities=set(caps),
        supported=True,
    )


def test_excludes_judge_from_candidates():
    judge = _mi("acme/Judge", {Capability.COMPLETION})
    a = _mi("acme/A", {Capability.COMPLETION})
    b = _mi("acme/B", {Capability.EMBEDDING})
    r = Router(judge=judge, candidates=[judge, a, b])
    assert judge not in r.candidates
    assert a in r.candidates and b in r.candidates


def test_requires_at_least_one_candidate():
    judge = _mi("acme/Judge", {Capability.COMPLETION})
    with pytest.raises(ValueError):
        Router(judge=judge, candidates=[judge])


def test_build_route_prompt_lists_candidates_and_request():
    judge = _mi("acme/Judge", {Capability.COMPLETION})
    a = _mi("acme/A", {Capability.COMPLETION}, params=70_000_000_000)
    r = Router(judge=judge, candidates=[judge, a])
    msgs = r.build_route_prompt("summarize this")
    assert msgs[0]["role"] == "system"
    assert "model router" in msgs[0]["content"].lower()
    assert "acme/A:main" in msgs[1]["content"]
    assert "summarize this" in msgs[1]["content"]


def test_parse_choice_variants():
    judge = _mi("acme/Judge", {Capability.COMPLETION})
    a = _mi("acme/A", {Capability.COMPLETION})
    b = _mi("acme/B", {Capability.COMPLETION})
    r = Router(judge=judge, candidates=[judge, a, b])
    assert r.parse_choice("acme/A:main") is a
    assert r.parse_choice('I choose "acme/B:main" because it fits.') is b
    assert r.parse_choice("acme/A") is a
    assert r.parse_choice("A") is a
    assert r.parse_choice("totally/unknown") is None


def test_parse_choice_short_base_name():
    """A judge that uses a model's short/base name (no quant tag) still resolves."""
    judge = _mi("acme/Judge", {Capability.COMPLETION})
    a = _mi("acme/Llama-3.1-8B-Instruct-NVFP4", {Capability.COMPLETION})
    b = _mi("acme/Qwen3-14B-NVFP4", {Capability.COMPLETION})
    r = Router(judge=judge, candidates=[judge, a, b])
    # short name (base name) of the only model mentioned -> resolves to it
    assert r.parse_choice("I think Llama-3.1-8B-Instruct is the best fit") is a
    # full bare name still works
    assert r.parse_choice("acme/Qwen3-14B-NVFP4") is b


def test_parse_choice_ambiguous_is_none():
    """A reply naming several models must not mis-route (returns None -> fallback)."""
    judge = _mi("acme/Judge", {Capability.COMPLETION})
    a = _mi("acme/Llama-3.1-8B-Instruct-NVFP4", {Capability.COMPLETION})
    b = _mi("acme/Qwen3-14B-NVFP4", {Capability.COMPLETION})
    r = Router(judge=judge, candidates=[judge, a, b])
    # both base names appear -> ambiguous -> None (safe fallback)
    assert r.parse_choice("weighing Llama-3.1-8B-Instruct vs Qwen3-14B, I lean toward one") is None


def test_fallback_is_largest_candidate():
    judge = _mi("acme/Judge", {Capability.COMPLETION})
    small = _mi("acme/Small", {Capability.COMPLETION}, size=1)
    big = _mi("acme/Big", {Capability.COMPLETION}, size=999)
    r = Router(judge=judge, candidates=[judge, small, big])
    assert r.fallback() is big


# ---- integration: model="auto" routes to the judge's choice -----------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_router_hub(root: Path) -> Path:
    """Judge (small completion) + BigLM (large completion) + Embedder."""
    sha = "b" * 40
    hub = root / "hub"

    def make(org: str, name: str, cfg: dict, weight: int) -> None:
        mdir = hub / f"models--{org}--{name}"
        (mdir / "refs").mkdir(parents=True)
        (mdir / "refs" / "main").write_text(sha)
        snap = mdir / "snapshots" / sha
        snap.mkdir(parents=True)
        (snap / "config.json").write_text(json.dumps(cfg))
        (snap / "model.safetensors").write_bytes(b"\0" * weight)

    make("acme", "JudgeLM", {"architectures": ["JudgeForCausalLM"], "model_type": "judge",
                             "hidden_size": 64, "num_hidden_layers": 1, "vocab_size": 256}, 1024)
    make("acme", "BigLM", {"architectures": ["BigForCausalLM"], "model_type": "big",
                           "hidden_size": 512, "num_hidden_layers": 8, "vocab_size": 4096}, 8_000_000)
    make("acme", "Embedder", {"architectures": ["EmbedderModel"], "model_type": "embedder",
                              "hidden_size": 64, "pooling": "mean"}, 1024)
    return hub


def make_router_stub(target_name: str) -> Starlette:
    """Answers the judge's routing call with `target_name`; other chat with 'routed-ok'."""

    async def health(_r):
        return PlainTextResponse("OK")

    async def chat(request):
        body = await request.json()
        msgs = body.get("messages", [])
        system = next((m.get("content", "") for m in msgs if m.get("role") == "system"), "")
        content = target_name if "model router" in system.lower() else "routed-ok"
        return JSONResponse(
            {
                "id": "r",
                "object": "chat.completion",
                "model": body.get("model", "stub"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    return Starlette(routes=[Route("/health", health), Route("/v1/chat/completions", chat, methods=["POST"])])


@pytest.fixture()
def router_client(tmp_path: Path, monkeypatch):
    from fastapi.testclient import TestClient

    from nvllm_service.config import Settings
    from nvllm_service.main import create_app
    from nvllm_service.workers import STATE_READY, Worker

    settings = Settings(hf_hub_root=_make_router_hub(tmp_path), router_judge="acme/JudgeLM")
    app = create_app(settings)
    assert app.state.router is not None

    port = _free_port()
    config = uvicorn.Config(make_router_stub("acme/BigLM"), host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    else:
        raise RuntimeError("router stub did not start")

    manager = app.state.manager

    async def fake_ensure(info, keep_alive=None, is_judge=False):
        w = manager.workers.get(info.name)
        if w is None or w.state != STATE_READY:
            w = Worker(info=info, port=port, state=STATE_READY, pid=9999, is_judge=is_judge)
            manager.workers[info.name] = w
        else:
            w.is_judge = is_judge
        return w

    monkeypatch.setattr(manager, "ensure", fake_ensure)
    with TestClient(app) as c:
        yield c


def test_auto_routes_to_judges_choice(router_client):
    r = router_client.post(
        "/api/chat",
        json={"model": "auto", "messages": [{"role": "user", "content": "hard math, need a big model"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "acme/BigLM:main"  # routed to the judge's pick, not "auto"
    assert body["message"]["content"] == "routed-ok"


def test_judge_loaded_as_judge(router_client):
    """Regression: the judge must load as the judge, not an evictable worker.

    It used to be matched by "repo:tag" name against the "repo" router_judge
    string, so it loaded as a normal (non-judge) worker and got evicted the moment
    the routed candidate loaded.
    """
    r = router_client.post(
        "/api/chat",
        json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    manager = router_client.app.state.manager
    judge_w = manager.workers.get("acme/JudgeLM:main")
    assert judge_w is not None
    assert judge_w.is_judge is True  # loaded as the judge, never evicted


def test_router_endpoint_reports_config(router_client):
    r = router_client.get("/api/router")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert body["alias"] == ROUTE_ALIAS
    assert body["judge"] == "acme/JudgeLM:main"
    assert "acme/BigLM:main" in body["candidates"]
    assert "acme/JudgeLM:main" not in body["candidates"]  # judge is not a target


def test_auto_400_when_routing_not_configured(client):
    # `client` fixture has no router_judge -> router disabled
    r = client.post("/api/chat", json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400
    assert "routing not configured" in r.json()["error"]


def test_router_candidates_whitelist(tmp_path: Path):
    """router_candidates restricts the Router to the named repos only."""
    from nvllm_service.config import Settings
    from nvllm_service.main import create_app

    settings = Settings(
        hf_hub_root=_make_router_hub(tmp_path),
        router_judge="acme/JudgeLM",
        router_candidates=["acme/BigLM"],
    )
    app = create_app(settings)
    r = app.state.router
    assert r is not None
    names = {m.name for m in r.candidates}
    assert "acme/BigLM:main" in names
    assert "acme/Embedder:main" not in names  # excluded by the whitelist
    assert "acme/JudgeLM:main" not in names  # judge is never a target
