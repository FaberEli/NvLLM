"""CLI client-command tests.

The CLI talks to a *real* server over HTTP (like `ollama` → `ollama serve`),
so we stand up the real app on a free port (same pattern as the `stub_server`
fixture) with `ensure()` pointed at the stub vLLM, and drive `cli.main()` with
explicit argv. No GPU, no vLLM, CI-safe.
"""
from __future__ import annotations

import httpx
import pytest

from nvllm_service import client as c
from nvllm_service import cli

# fixtures: live_server / nvllm_env live in conftest.py (shared with test_launch)


def _load_model(url: str, model: str = "acme/TinyLM") -> None:
    r = httpx.post(
        f"{url}/api/chat",
        json={"model": model, "messages": [{"role": "user", "content": "hi"}], "stream": False},
    )
    assert r.status_code == 200


def _loaded_names(url: str) -> list[str]:
    return [m["name"] for m in httpx.get(f"{url}/api/ps").json()["models"]]


# ---- list --------------------------------------------------------------------


def test_list(nvllm_env, capsys):
    rc = cli.main(["list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "NAME" in out and "MODIFIED" in out
    assert "acme/TinyLM:main" in out
    # unsupported model is listed, flagged
    assert "acme/WanVideo:main  (unservable)" in out
    assert "a" * 12 in out  # 12-hex ID


def test_list_alias_ls(nvllm_env, capsys):
    rc = cli.main(["ls"])
    assert rc == 0
    assert "acme/TinyLM:main" in capsys.readouterr().out


def test_list_prefix_filter(nvllm_env, capsys):
    rc = cli.main(["list", "acme/Embedder"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "acme/Embedder:main" in out
    assert "TinyLM" not in out


def test_list_no_match(nvllm_env, capsys):
    rc = cli.main(["list", "zzz/"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no models matching" in out


# ---- ps ----------------------------------------------------------------------


def test_ps_empty(nvllm_env, capsys):
    rc = cli.main(["ps"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no models loaded" in out


def test_ps_shows_loaded_model(nvllm_env, capsys):
    _load_model(nvllm_env)
    rc = cli.main(["ps"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "NAME" in out and "PROCESSOR" in out and "UNTIL" in out
    assert "acme/TinyLM:main" in out
    assert "100% GPU" in out


# ---- show --------------------------------------------------------------------


def test_show(nvllm_env, capsys):
    rc = cli.main(["show", "acme/TinyLM"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "TinyLMForCausalLM" in out
    assert "tinylm" in out
    assert "completion" in out
    assert "nvllm run acme/TinyLM" in out


def test_show_unknown(nvllm_env, capsys):
    rc = cli.main(["show", "nope/Nope"])
    assert rc == 1
    assert "not found" in capsys.readouterr().err


# ---- stop --------------------------------------------------------------------


def test_stop(nvllm_env, capsys):
    _load_model(nvllm_env)
    assert "acme/TinyLM:main" in _loaded_names(nvllm_env)
    rc = cli.main(["stop", "acme/TinyLM"])  # bare name must resolve server-side
    out = capsys.readouterr().out
    assert rc == 0
    assert "unloaded 'acme/TinyLM:main'" in out
    assert _loaded_names(nvllm_env) == []


def test_stop_not_loaded(nvllm_env, capsys):
    rc = cli.main(["stop", "acme/Embedder"])
    assert rc == 1
    assert "not loaded" in capsys.readouterr().err


# ---- run ---------------------------------------------------------------------


def test_run_oneshot(nvllm_env, capsys):
    rc = cli.main(["run", "acme/TinyLM", "hello"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Hello world" in out  # stub streams "Hello" + " world"


def test_run_oneshot_multiword_prompt(nvllm_env, capsys):
    rc = cli.main(["run", "acme/TinyLM", "tell", "me", "about", "cats"])
    assert rc == 0
    assert "Hello world" in capsys.readouterr().out


def test_run_unknown_model(nvllm_env, capsys):
    rc = cli.main(["run", "nope/Nope", "hello"])
    assert rc == 1
    assert "not found" in capsys.readouterr().err


def test_repl_scripted_session(nvllm_env, stub_captured):
    out: list[str] = []
    lines = iter(["/set temperature 0.1", "hello", "/bye"])

    def sink(*a, **k) -> None:
        out.append("".join(str(x) for x in a))

    rc = c.repl(
        nvllm_env,
        "acme/TinyLM",
        in_fn=lambda prompt: next(lines),
        out_fn=sink,
    )
    assert rc == 0
    joined = "".join(out)
    assert "Hello world" in joined
    assert "temperature = 0.1" in joined
    body = stub_captured[-1]
    assert body["temperature"] == 0.1  # /set value reached the worker request
    assert [m for m in body["messages"] if m["role"] == "user"][-1]["content"] == "hello"


def test_repl_clear_resets_history(nvllm_env, stub_captured):
    out: list[str] = []
    lines = iter(["first", "/clear", "second", "/bye"])
    c.repl(
        nvllm_env,
        "acme/TinyLM",
        in_fn=lambda prompt: next(lines),
        out_fn=lambda *a, **k: out.append("".join(str(x) for x in a)),
    )
    assert "conversation cleared" in "".join(out)
    last = stub_captured[-1]
    contents = [m["content"] for m in last["messages"]]
    assert "first" not in contents
    assert "second" in contents


def test_repl_help_and_eof(nvllm_env):
    out: list[str] = []
    lines = ["/help"]

    def in_fn(prompt: str) -> str:
        if lines:
            return lines.pop(0)
        raise EOFError  # simulate Ctrl-D

    c.repl(
        nvllm_env,
        "acme/TinyLM",
        in_fn=in_fn,
        out_fn=lambda *a, **k: out.append("".join(str(x) for x in a)),
    )
    joined = "".join(out)
    assert "/set temperature" in joined
    assert "/bye" in joined


def test_repl_unknown_command(nvllm_env):
    out: list[str] = []
    lines = iter(["/frob", "/bye"])
    c.repl(
        nvllm_env,
        "acme/TinyLM",
        in_fn=lambda prompt: next(lines),
        out_fn=lambda *a, **k: out.append("".join(str(x) for x in a)),
    )
    assert "unknown command '/frob'" in "".join(out)


# ---- refusals / help / version -------------------------------------------------


def test_pull_refused(capsys):
    rc = cli.main(["pull", "llama3.2"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "does not download" in out
    assert "hf download" in out


# NOTE: `rm` of a REAL model (guidance + rc 1) is covered by
# test_cli_create_list_rm — `rm` is now a real subcommand (virtual models),
# and `create` is no longer a refusal (see test_cli_create_*).


@pytest.mark.parametrize("cmd", ["push", "login", "cp"])
def test_other_refusals(cmd, capsys):
    assert cli.main([cmd]) == 1
    assert capsys.readouterr().out.strip()


def test_help(capsys):
    rc = cli.main(["help"])
    out = capsys.readouterr().out
    assert rc == 0
    for token in ("nvllm serve", "nvllm list", "nvllm run", "nvllm launch", "nvllm stop", "NVLLM_SERVER"):
        assert token in out


def test_no_args_prints_help(capsys):
    rc = cli.main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "nvllm serve" in out


def test_launch_list_via_cli(capsys):
    rc = cli.main(["launch", "--list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "claude" in out and "codex" in out
    assert "anthropic" in out and "openai" in out


def test_version(nvllm_env, capsys):
    import nvllm_service

    rc = cli.main(["version"])
    out = capsys.readouterr().out
    assert rc == 0
    # assert against the real version so this does not break on every bump
    assert f"nvllm {nvllm_service.__version__}" in out
    assert f"server  nvllm-serve {nvllm_service.__version__}" in out


# ---- create / rm (virtual models) -----------------------------------------------


def test_cli_create_list_rm(nvllm_env, capsys):
    assert cli.main(["create", "assistant", "--from", "acme/TinyLM", "--system", "be brief"]) == 0
    assert "created 'assistant'" in capsys.readouterr().out
    capsys.readouterr()
    cli.main(["list"])
    assert "assistant" in capsys.readouterr().out
    assert cli.main(["rm", "assistant"]) == 0
    assert "deleted 'assistant'" in capsys.readouterr().out
    # a real cache model: never deleted — guidance + non-zero exit
    assert cli.main(["rm", "acme/TinyLM"]) == 1
    assert "huggingface" in capsys.readouterr().out.lower()


def test_cli_create_validation_error(nvllm_env, capsys):
    assert cli.main(["create", "assistant", "--from", "nope/Nope"]) == 1
    assert "not found" in capsys.readouterr().err


def test_show_created_model_shows_wrapper(nvllm_env, capsys):
    """`nvllm show <virtual>` displays the injected system prompt + options."""
    assert cli.main(["create", "assistant", "--from", "acme/TinyLM", "--system", "be brief", "--temperature", "0.3"]) == 0
    capsys.readouterr()
    assert cli.main(["show", "assistant"]) == 0
    out = capsys.readouterr().out
    assert "system prompt:" in out
    assert "be brief" in out
    assert "temperature = 0.3" in out
    capsys.readouterr()
    cli.main(["rm", "assistant"])


# ---- vision + tools (run + REPL) --------------------------------------------------


def _png(tmp_path):
    p = tmp_path / "x.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 8)
    return p


def test_cli_run_image_and_tools(nvllm_env, tmp_path, stub_captured, capsys):
    import json as _json

    img = _png(tmp_path)
    tf = tmp_path / "tools.json"
    tf.write_text(_json.dumps([{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]))
    rc = cli.main(["run", "acme/TinyLM", "what is this?", "--image", str(img), "--tools", str(tf)])
    assert rc == 0
    body = stub_captured[-1]
    assert body["tools"][0]["function"]["name"] == "get_weather"
    parts = body["messages"][0]["content"]
    assert any(p.get("type") == "text" and p["text"] == "what is this?" for p in parts)
    assert any(p.get("type") == "image_url" for p in parts)


def test_repl_image_turn_and_tools(nvllm_env, tmp_path, stub_captured):
    img = _png(tmp_path)
    tf = tmp_path / "tools.json"
    tf.write_text('[{"type":"function","function":{"name":"t","parameters":{"type":"object"}}}]')
    lines = iter([str(img), f"/tools {tf}", "hello", "/bye"])
    c.repl(nvllm_env, "acme/TinyLM", in_fn=lambda prompt: next(lines), out_fn=lambda *a, **k: None)
    bodies = list(stub_captured)
    # image turn: the user message carries an image_url part
    parts = bodies[0]["messages"][0]["content"]
    assert any(p.get("type") == "image_url" for p in parts)
    # after /tools, the next turn carries the tool definitions
    assert bodies[1]["tools"][0]["function"]["name"] == "t"


def test_repl_bad_tools_file_message(nvllm_env, capsys):
    lines = iter(["/tools /no/such/file.json", "/bye"])
    outs = []
    c.repl(
        nvllm_env, "acme/TinyLM",
        in_fn=lambda prompt: next(lines),
        out_fn=lambda *a, **k: outs.append(" ".join(str(x) for x in a)),
    )
    assert any("could not load tools" in s for s in outs)


# ---- error paths ----------------------------------------------------------------


def test_server_down_clear_error(monkeypatch, capsys):
    monkeypatch.setenv("NVLLM_SERVER", "http://127.0.0.1:1")  # nothing listens on :1
    rc = cli.main(["list"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "nvllm serve" in err
    assert "NVLLM_SERVER" in err


def test_server_flag_beats_env(nvllm_env, monkeypatch, capsys):
    # --server pointing at a dead port must fail even though NVLLM_SERVER is live
    rc = cli.main(["list", "--server", "http://127.0.0.1:1"])
    assert rc == 1
    assert "nvllm serve" in capsys.readouterr().err
