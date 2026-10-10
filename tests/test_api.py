"""API contract tests against the app with a stub vLLM worker (see conftest)."""
import json

import pytest

# ---- metadata ---------------------------------------------------------------


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_api_version(client):
    r = client.get("/api/version")
    assert r.status_code == 200
    assert "version" in r.json()


def test_api_tags_lists_models(client):
    r = client.get("/api/tags")
    assert r.status_code == 200
    names = [m["name"] for m in r.json()["models"]]
    assert "acme/TinyLM:main" in names
    assert "acme/WanVideo:main" in names


def test_api_show(client):
    r = client.post("/api/show", json={"model": "acme/TinyLM"})
    assert r.status_code == 200
    body = r.json()
    assert body["model_info"]["general.architecture"] == "TinyLMForCausalLM"
    assert "completion" in body["capabilities"]


def test_v1_models_only_supported(client):
    r = client.get("/v1/models")
    ids = [m["id"] for m in r.json()["data"]]
    assert "acme/TinyLM:main" in ids
    assert "acme/WanVideo:main" not in ids


def test_unknown_model_404(client):
    r = client.post("/api/chat", json={"model": "nope/Nope", "messages": []})
    assert r.status_code == 404
    assert "error" in r.json()


def test_unsupported_model_400(client):
    r = client.post("/api/chat", json={"model": "acme/WanVideo", "messages": []})
    assert r.status_code == 400
    assert "not servable" in r.json()["error"]


def test_capability_mismatch_400(client):
    r = client.post("/api/embed", json={"model": "acme/TinyLM", "input": "x"})
    assert r.status_code == 400
    assert "embedding" in r.json()["error"]


# ---- OpenAI-compatible endpoints ---------------------------------------------


def test_v1_chat_completions_non_stream(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "acme/TinyLM", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "Hello world"
    # the model name is rewritten to the served catalog name
    assert body["model"] == "acme/TinyLM:main"


def test_v1_chat_completions_stream(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "acme/TinyLM", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    data_lines = [ln[5:].strip() for ln in r.text.splitlines() if ln.startswith("data:")]
    assert data_lines[-1] == "[DONE]"
    content = ""
    for d in data_lines[:-1]:
        for ch in json.loads(d).get("choices") or []:
            content += (ch.get("delta") or {}).get("content") or ""
    assert content == "Hello world"


def test_v1_completions(client):
    r = client.post("/v1/completions", json={"model": "acme/TinyLM", "prompt": "hi"})
    assert r.status_code == 200
    assert r.json()["choices"][0]["text"] == "stub-completion"


def test_v1_embeddings(client):
    r = client.post("/v1/embeddings", json={"model": "acme/Embedder", "input": "hello"})
    assert r.status_code == 200
    assert r.json()["data"][0]["embedding"] == [0.1, 0.2, 0.3]


# ---- Ollama-compatible inference endpoints ------------------------------------


def test_api_chat_non_stream(client):
    r = client.post("/api/chat", json={"model": "acme/TinyLM", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "acme/TinyLM:main"
    assert body["done"] is True
    assert body["done_reason"] == "stop"
    assert body["message"]["content"] == "Hello world"
    assert body["prompt_eval_count"] == 7
    assert body["eval_count"] == 2


def test_api_chat_stream(client):
    r = client.post(
        "/api/chat",
        json={"model": "acme/TinyLM", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert r.status_code == 200
    lines = [json.loads(ln) for ln in r.text.splitlines() if ln.strip()]
    assert all(l["model"] == "acme/TinyLM:main" for l in lines)
    assert lines[0]["done"] is False
    assert lines[-1]["done"] is True
    assert lines[-1]["done_reason"] == "stop"
    content = "".join(l.get("message", {}).get("content", "") for l in lines)
    assert content == "Hello world"
    assert lines[-1]["prompt_eval_count"] == 7


def test_api_chat_options_mapping(client, stub_captured):
    # Ollama `options` must be translated onto the worker's OpenAI body
    r = client.post(
        "/api/chat",
        json={
            "model": "acme/TinyLM",
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"temperature": 0.5, "num_predict": 12, "top_k": 40},
        },
    )
    assert r.status_code == 200
    assert r.json()["message"]["content"] == "Hello world"
    sent = stub_captured[-1]
    assert sent["max_tokens"] == 12          # num_predict -> max_tokens
    assert sent["temperature"] == 0.5
    assert "top_k" not in sent               # dropped: no OpenAI counterpart
    assert sent["model"] == "acme/TinyLM:main"


def test_api_generate(client):
    r = client.post("/api/generate", json={"model": "acme/TinyLM", "prompt": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["done"] is True
    assert body["response"] == "stub-completion"
    assert body["prompt_eval_count"] == 4


def test_api_generate_stream(client):
    r = client.post("/api/generate", json={"model": "acme/TinyLM", "prompt": "hi", "stream": True})
    assert r.status_code == 200
    lines = [json.loads(ln) for ln in r.text.splitlines() if ln.strip()]
    assert lines[-1]["done"] is True
    assert "".join(l.get("response", "") for l in lines) == "stub-completion"


def test_api_embed(client):
    r = client.post("/api/embed", json={"model": "acme/Embedder", "input": "hello"})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "acme/Embedder:main"
    assert body["embeddings"] == [[0.1, 0.2, 0.3]]


def test_api_embed_list_input(client):
    r = client.post("/api/embed", json={"model": "acme/Embedder", "input": ["a", "b"]})
    assert r.status_code == 200
    assert len(r.json()["embeddings"]) == 2


def test_api_embeddings_legacy(client):
    r = client.post("/api/embeddings", json={"model": "acme/Embedder", "prompt": "hello"})
    assert r.status_code == 200
    assert r.json()["embedding"] == [0.1, 0.2, 0.3]


def test_api_ps_empty(client):
    r = client.get("/api/ps")
    assert r.status_code == 200
    assert r.json()["models"] == []
