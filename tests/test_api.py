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


# ---- unload (`nvllm stop`) --------------------------------------------------


def test_api_unload_loaded_model(client):
    # load a model first (ensure is stubbed; the chat call triggers it)
    r = client.post("/api/chat", json={"model": "acme/TinyLM", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    names = [m["name"] for m in client.get("/api/ps").json()["models"]]
    assert "acme/TinyLM:main" in names

    r = client.post("/api/unload", json={"model": "acme/TinyLM"})  # bare name must resolve
    assert r.status_code == 200
    assert r.json()["model"] == "acme/TinyLM:main"
    assert client.get("/api/ps").json()["models"] == []


def test_api_unload_not_loaded(client):
    r = client.post("/api/unload", json={"model": "acme/Embedder"})
    assert r.status_code == 404
    assert "not loaded" in r.json()["error"]


def test_api_unload_virtual_name_unloads_base(client):
    """`nvllm stop <created>` resolves to the base worker (virtuals share it)."""
    assert client.post("/api/create", json={"name": "assistant", "model": "acme/TinyLM"}).status_code == 200
    r = client.post("/api/chat", json={"model": "assistant", "messages": [{"role": "user", "content": "hi"}], "stream": False})
    assert r.status_code == 200
    names = [m["name"] for m in client.get("/api/ps").json()["models"]]
    assert "acme/TinyLM:main" in names
    r = client.post("/api/unload", json={"model": "assistant"})
    assert r.status_code == 200
    assert r.json()["model"] == "acme/TinyLM:main"
    assert client.get("/api/ps").json()["models"] == []


def test_api_unload_unknown_model(client):
    r = client.post("/api/unload", json={"model": "nope/Nope"})
    assert r.status_code == 404
    assert "not found" in r.json()["error"]


def test_api_unload_requires_model(client):
    r = client.post("/api/unload", json={})
    assert r.status_code == 400


# ---- Anthropic /v1/messages -------------------------------------------------


def test_v1_messages_non_stream(client, stub_captured):
    r = client.post(
        "/v1/messages",
        json={
            "model": "acme/TinyLM",
            "max_tokens": 50,
            "system": "be brief",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "message"
    assert body["content"] == [{"type": "text", "text": "Hello world"}]
    assert body["stop_reason"] == "end_turn"
    assert body["usage"]["input_tokens"] == 7
    # the worker received a translated OpenAI body
    oai = stub_captured[-1]
    assert oai["messages"][0] == {"role": "system", "content": "be brief"}
    assert oai["max_tokens"] == 50


def test_v1_messages_stream(client):
    with client.stream(
        "POST",
        "/v1/messages",
        json={"model": "acme/TinyLM", "max_tokens": 50, "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    ) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        text = "".join(r.iter_text())
    for event in ("message_start", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"):
        assert f"event: {event}" in text, event
    assert "Hello" in text and " world" in text
    assert "end_turn" in text


def test_v1_messages_unknown_model_anthropic_error_shape(client):
    r = client.post("/v1/messages", json={"model": "nope/Nope", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 404
    body = r.json()
    assert body["type"] == "error"
    assert "not found" in body["error"]["message"]


def test_v1_messages_embedding_model_rejected(client):
    """Same capability guard as /api/chat: an embedding model cannot serve chat."""
    r = client.post(
        "/v1/messages",
        json={"model": "acme/Embedder", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 400
    body = r.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert "completion" in body["error"]["message"]


# ---- model aliases (Claude Code 客户端模型名校验的解法) ----------------------


def test_v1_aliases_register_and_resolve(client, stub_captured):
    # 注册：别名 → 目录中真实模型（解析成完整名 org/name:tag）
    r = client.post("/v1/aliases", json={"name": "claude-sonnet-4-5", "model": "acme/TinyLM"})
    assert r.status_code == 200
    assert r.json() == {"name": "claude-sonnet-4-5", "model": "acme/TinyLM:main"}

    # 目标模型不存在 → 404；缺字段 → 400
    assert client.post("/v1/aliases", json={"name": "x", "model": "nope/Nope"}).status_code == 404
    assert client.post("/v1/aliases", json={"name": "x"}).status_code == 400

    # 别名可查询
    r = client.get("/v1/aliases")
    assert r.status_code == 200
    assert r.json()["aliases"] == {"claude-sonnet-4-5": "acme/TinyLM:main"}

    # 闭环：/v1/messages 用别名 → 解析到真实模型，worker 收到真实模型名
    r = client.post(
        "/v1/messages",
        json={"model": "claude-sonnet-4-5", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "acme/TinyLM:main"
    assert body["content"] == [{"type": "text", "text": "Hello world"}]
    assert stub_captured[-1]["model"] == "acme/TinyLM:main"


# ---- vision + tools (server-side translation, already supported) --------------

_B64 = "aW1hZ2U="  # any base64 payload; the stub doesn't decode it


def test_api_chat_images_become_image_url_parts(client, stub_captured):
    r = client.post(
        "/api/chat",
        json={
            "model": "acme/TinyLM",
            "messages": [
                {"role": "user", "content": "what is this?", "images": [f"data:image/png;base64,{_B64}"]},
            ],
            "stream": False,
        },
    )
    assert r.status_code == 200
    content = stub_captured[-1]["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "what is this?"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_api_chat_raw_base64_images_default_png_mime(client, stub_captured):
    client.post(
        "/api/chat",
        json={
            "model": "acme/TinyLM",
            "messages": [{"role": "user", "content": "hi", "images": [_B64]}],
            "stream": False,
        },
    )
    content = stub_captured[-1]["messages"][0]["content"]
    assert content[1]["image_url"]["url"] == f"data:image/png;base64,{_B64}"


def test_api_chat_tools_passthrough(client, stub_captured):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    r = client.post(
        "/api/chat",
        json={
            "model": "acme/TinyLM",
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "tools": tools,
            "stream": False,
        },
    )
    assert r.status_code == 200
    assert stub_captured[-1]["tools"] == tools
