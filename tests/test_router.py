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
from nvllm_service.router import ROUTE_ALIAS, Router, option_symbol, softmax_topk

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


# ---- M13: option-symbol scoring (decision-model native readout) -------------


def test_option_symbol_sequence():
    assert [option_symbol(i) for i in range(3)] == ["A", "B", "C"]
    assert option_symbol(25) == "Z"
    assert option_symbol(26) == "a"
    assert option_symbol(51) == "z"
    assert option_symbol(52) == "0"


def test_softmax_topk_picks_argmax_and_normalizes():
    # the shared scoring core: softmax over a subset of log-scale values
    best, probs = softmax_topk([(0, -2.0), (1, -1.0), (2, -5.0)])
    assert best == 1  # least-negative (highest) value wins
    assert len(probs) == 3
    assert abs(sum(probs) - 1.0) < 1e-9
    assert probs[1] > probs[0] > probs[2]


def test_softmax_topk_is_shift_invariant():
    # adding a constant to every value must not change the distribution
    b1, p1 = softmax_topk([(0, -100.0), (1, -99.0), (2, -200.0)])
    b2, p2 = softmax_topk([(0, 0.0), (1, 1.0), (2, -100.0)])
    assert b1 == b2 == 1
    assert all(abs(x - y) < 1e-9 for x, y in zip(p1, p2))


def test_softmax_topk_single_option_is_one():
    best, probs = softmax_topk([(5, -3.0)])
    assert best == 5
    assert probs == [1.0]


def test_score_logprobs_picks_argmax_and_normalizes():
    judge = _mi("acme/Judge", {Capability.COMPLETION})
    a = _mi("acme/A", {Capability.COMPLETION})
    b = _mi("acme/B", {Capability.COMPLETION})
    c = _mi("acme/C", {Capability.COMPLETION})
    r = Router(judge=judge, candidates=[judge, a, b, c])
    # b has the highest logprob -> chosen; distribution renormalizes over the options
    top = [
        {"token": "B", "logprob": -0.1},
        {"token": "A", "logprob": -1.5},
        {"token": "C", "logprob": -2.5},
    ]
    chosen, dist = r.score_logprobs(top)
    assert chosen is b
    assert set(dist) == {a.name, b.name, c.name}
    assert abs(sum(dist.values()) - 1.0) < 1e-6
    assert dist[b.name] > dist[a.name] > dist[c.name]


def test_score_logprobs_ignores_non_option_tokens():
    judge = _mi("acme/Judge", {Capability.COMPLETION})
    a = _mi("acme/A", {Capability.COMPLETION})
    b = _mi("acme/B", {Capability.COMPLETION})
    r = Router(judge=judge, candidates=[judge, a, b])
    # "the", "best", "X" (not a candidate option) must be ignored
    top = [
        {"token": "the", "logprob": -0.01},
        {"token": "B", "logprob": -0.4},
        {"token": "A", "logprob": -3.0},
        {"token": "X", "logprob": -0.2},
    ]
    chosen, dist = r.score_logprobs(top)
    assert chosen is b
    assert set(dist) == {a.name, b.name}


def test_score_logprobs_no_option_letter_returns_none():
    judge = _mi("acme/Judge", {Capability.COMPLETION})
    a = _mi("acme/A", {Capability.COMPLETION})
    r = Router(judge=judge, candidates=[judge, a])
    chosen, dist = r.score_logprobs([{"token": "sure", "logprob": -0.1}, {"token": "thing", "logprob": -0.5}])
    assert chosen is None
    assert dist == {}


def test_parse_letter_variants():
    judge = _mi("acme/Judge", {Capability.COMPLETION})
    a = _mi("acme/A", {Capability.COMPLETION})
    b = _mi("acme/B", {Capability.COMPLETION})
    r = Router(judge=judge, candidates=[judge, a, b])
    assert r.parse_choice_text("A") is a
    assert r.parse_choice_text("(B)") is b
    assert r.parse_choice_text("option A") is a
    assert r.parse_choice_text("A.") is a
    # a model name is NOT a letter -> falls through to name parsing
    assert r.parse_choice_text("acme/B") is b
    assert r.parse_choice_text("totally unknown") is None


def test_choice_prompt_uses_option_letters_and_request():
    judge = _mi("acme/Judge", {Capability.COMPLETION})
    a = _mi("acme/A", {Capability.COMPLETION})
    b = _mi("acme/B", {Capability.COMPLETION})
    r = Router(judge=judge, candidates=[judge, a, b])
    msgs = r.build_choice_prompt("summarize this")
    assert "model router" in msgs[0]["content"].lower()
    user = msgs[1]["content"]
    assert "A. acme/A" in user
    assert "B. acme/B" in user
    assert "summarize this" in user


# ---- M13: integration — the router follows the judge's logprob distribution --


def _make_logprob_hub(root: Path) -> Path:
    """Judge + SmallLM + BigLM, all completion-capable, distinct sizes.

    (Unlike _make_router_hub's Embedder, both candidates are servable via /api/chat,
    so routing to the smallest one returns 200 rather than a capability 400.)
    """
    sha = "d" * 40
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
                             "hidden_size": 64, "num_hidden_layers": 1, "vocab_size": 256}, 512)
    make("acme", "SmallLM", {"architectures": ["SmallForCausalLM"], "model_type": "small",
                             "hidden_size": 64, "num_hidden_layers": 1, "vocab_size": 256}, 1024)
    make("acme", "BigLM", {"architectures": ["BigForCausalLM"], "model_type": "big",
                           "hidden_size": 512, "num_hidden_layers": 8, "vocab_size": 4096}, 8_000_000)
    return hub


def make_logprobs_stub(prefer_letter: str) -> Starlette:
    """A judge whose /v1/chat/completions returns top_logprobs favoring `prefer_letter`.

    Distinguishes logprob-scoring from the fallback (which would pick the largest
    candidate regardless of the distribution).
    """

    async def health(_r):
        return PlainTextResponse("OK")

    async def chat(request):
        body = await request.json()
        msgs = body.get("messages", [])
        system = next((m.get("content", "") for m in msgs if m.get("role") == "system"), "")
        if "model router" in system.lower():
            logp = {L: (-0.3 if L == prefer_letter else -2.0) for L in "ABCDEF"}
            top = sorted(({"token": L, "logprob": v} for L, v in logp.items()), key=lambda e: e["logprob"], reverse=True)
            return JSONResponse(
                {
                    "id": "r",
                    "object": "chat.completion",
                    "model": body.get("model", "stub"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": prefer_letter},
                            "finish_reason": "length",
                            "logprobs": {"content": [{"token": prefer_letter, "logprob": logp[prefer_letter], "top_logprobs": top}]},
                        }
                    ],
                }
            )
        return JSONResponse(
            {
                "id": "r",
                "object": "chat.completion",
                "model": body.get("model", "stub"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "routed-ok"}, "finish_reason": "stop"}],
            }
        )

    return Starlette(routes=[Route("/health", health), Route("/v1/chat/completions", chat, methods=["POST"])])


@pytest.fixture()
def logprob_router_client(tmp_path: Path, monkeypatch):
    """App with a judge configured and a logprobs-returning stub worker.

    The stub is parameterised per-test via ``logprob_router_client._set_preferred``.
    """
    from fastapi.testclient import TestClient

    from nvllm_service.config import Settings
    from nvllm_service.main import create_app
    from nvllm_service.workers import STATE_READY, Worker

    settings = Settings(hf_hub_root=_make_logprob_hub(tmp_path), router_judge="acme/JudgeLM")
    app = create_app(settings)

    state = {"prefer": "B"}  # default: favor the last (smaller) candidate, not the fallback-largest
    port = _free_port()

    # rebuild the stub on each request using the current preference
    class _Dynamic(Starlette):
        def __init__(self):
            super().__init__(routes=[
                Route("/health", lambda _r: PlainTextResponse("OK")),
                Route("/v1/chat/completions", self._chat, methods=["POST"]),
            ])

        async def _chat(self, request):
            prefer = state["prefer"]
            logp = {L: (-0.3 if L == prefer else -2.0) for L in "ABCDEF"}
            top = sorted(({"token": L, "logprob": v} for L, v in logp.items()), key=lambda e: e["logprob"], reverse=True)
            body = await request.json()
            return JSONResponse(
                {
                    "id": "r",
                    "object": "chat.completion",
                    "model": body.get("model", "stub"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": prefer},
                            "finish_reason": "length",
                            "logprobs": {"content": [{"token": prefer, "logprob": logp[prefer], "top_logprobs": top}]},
                        }
                    ],
                }
            )

    config = uvicorn.Config(_Dynamic(), host="127.0.0.1", port=port, log_level="error")
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
        raise RuntimeError("logprob stub did not start")

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
    c = TestClient(app)
    c._set_preferred = lambda letter: state.update(prefer=letter)
    with c:
        yield c


def test_auto_routes_by_logprob_not_fallback(logprob_router_client):
    """The router must follow the judge's distribution, not fall back to largest.

    Favoring the SMALLEST candidate (which is not the largest-model fallback) and
    asserting it is chosen proves the choice came from the logprob distribution.
    """
    r = logprob_router_client.app.state.router
    smallest = min(r.candidates, key=lambda m: m.size or 0)
    assert smallest is not r.fallback()  # sanity: they differ, so this test is meaningful
    letter = option_symbol(r.candidates.index(smallest))
    logprob_router_client._set_preferred(letter)
    resp = logprob_router_client.post("/api/chat", json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    assert resp.json()["model"] == smallest.name


def test_api_route_returns_distribution(logprob_router_client):
    r = logprob_router_client.app.state.router
    logprob_router_client._set_preferred(option_symbol(0))
    resp = logprob_router_client.post("/api/route", json={"text": "hello"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["alias"] == ROUTE_ALIAS
    assert body["backend"] in ("vllm-logprob", "vllm-text")
    assert body["judge"] == "acme/JudgeLM:main"
    assert body["chosen"] in [m.name for m in r.candidates]
    # a real distribution over the candidates, summing to ~1
    assert body["distribution"]
    assert abs(sum(body["distribution"].values()) - 1.0) < 1e-6


def test_api_route_400_when_not_configured(client):
    r = client.post("/api/route", json={"text": "hello"})
    assert r.status_code == 400
    assert "routing not configured" in r.json()["error"]


def test_router_endpoint_reports_backend(router_client):
    body = router_client.get("/api/router").json()
    assert body["enabled"] is True
    assert body["backend"] == "vllm"  # default backend
