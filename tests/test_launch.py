"""Tests for `nvllm launch` (agent registry, model default, exec wiring).

Exec is tested deterministically with a fake agent binary on PATH that prints
its environment/args — no real agent, no network beyond the local stub server.
"""
from __future__ import annotations

import os
import stat

import pytest

from nvllm_service import launch as L


def test_registry_env_openai_agent(monkeypatch):
    monkeypatch.setattr(L, "find_binary", lambda name: f"/usr/bin/{name}")
    _a, binary, env, args = L.launch_config("codex", "http://127.0.0.1:8004", "org/Mod:main")
    assert binary == "/usr/bin/codex"
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:8004/v1"
    assert env["OPENAI_API_KEY"] == "nvllm"
    assert args == ["--model", "org/Mod:main"]


def test_registry_env_anthropic_agent(monkeypatch, tmp_path):
    monkeypatch.setattr(L, "find_binary", lambda name: f"/usr/bin/{name}")
    monkeypatch.setenv("NVLLM_STATE_DIR", str(tmp_path))
    _a, _binary, env, args = L.launch_config("claude", "http://127.0.0.1:8004", "org/Mod:main")
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8004"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "nvllm"
    # 模型名走 settings.json（CLAUDE_CONFIG_DIR）——原生 Claude Code 构建对
    # ANTHROPIC_MODEL 做白名单门禁，而 settings.json 的 `model` 字段接受任意名字
    assert "ANTHROPIC_MODEL" not in env
    cfg_dir = env["CLAUDE_CONFIG_DIR"]
    import json
    from pathlib import Path

    doc = json.loads(Path(cfg_dir, "settings.json").read_text(encoding="utf-8"))
    assert doc["model"] == "org/Mod:main"
    assert doc["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8004"
    assert args == []  # claude takes the model via its settings.json


def test_unknown_agent():
    with pytest.raises(L.CLIError, match="unknown agent"):
        L.pick_agent("doesnotexist")


def test_missing_binary(monkeypatch):
    monkeypatch.setattr(L, "find_binary", lambda name: None)
    with pytest.raises(L.CLIError, match="not found on PATH"):
        L.launch_config("claude", "http://x", "m")


def test_pick_agent_auto_detect_prefers_order(monkeypatch):
    def fake_which(name):
        return f"/bin/{name}" if name in ("gemini", "codex") else None

    monkeypatch.setattr(L, "find_binary", fake_which)
    # claude (preferred) is absent -> codex before gemini
    assert L.pick_agent(None).name == "codex"


def test_pick_agent_none_available(monkeypatch):
    monkeypatch.setattr(L, "find_binary", lambda name: None)
    with pytest.raises(L.CLIError, match="no known agent CLI is installed"):
        L.pick_agent(None)


def test_pick_default_model_largest_completion():
    catalog = [
        {"name": "org/Small", "capabilities": ["completion"], "size": 100},
        {"name": "org/Big", "capabilities": ["completion"], "size": 900},
        {"name": "org/Embed", "capabilities": ["embedding"], "size": 5000},  # not completion
    ]
    assert L.pick_default_model(catalog) == "org/Big"


def test_pick_default_model_empty():
    assert L.pick_default_model([]) is None


def _fake_claude(tmp_path):
    # 打印它收到的关键配置（模拟真实 claude 读取 settings.json）
    fake = tmp_path / "claude"
    fake.write_text(
        '#!/bin/sh\necho "CFGDIR=$CLAUDE_CONFIG_DIR BASE=$ANTHROPIC_BASE_URL ARGS:$*"\n'
        'cat "$CLAUDE_CONFIG_DIR/settings.json"\n'
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return fake


def test_launch_exec_passes_env_and_args(nvllm_env, tmp_path, monkeypatch, capfd):
    monkeypatch.setenv("NVLLM_STATE_DIR", str(tmp_path / "state"))
    fake = _fake_claude(tmp_path)
    monkeypatch.setattr(L, "find_binary", lambda name: str(fake) if name == "claude" else None)
    rc = L.cmd_launch(nvllm_env, "claude", "acme/TinyLM", extra=["-p", "hi"])
    out = capfd.readouterr().out
    assert rc == 0
    assert f"BASE={nvllm_env}" in out
    assert "ARGS:-p hi" in out
    # claude 的配置目录里带着真实模型名（按请求名，服务端解析全名）与 base URL
    assert '"model": "acme/TinyLM"' in out
    assert f'"ANTHROPIC_BASE_URL": "{nvllm_env}"' in out


def test_launch_default_model_from_catalog(nvllm_env, tmp_path, monkeypatch, capfd):
    monkeypatch.setenv("NVLLM_STATE_DIR", str(tmp_path / "state"))
    fake = _fake_claude(tmp_path)
    monkeypatch.setattr(L, "find_binary", lambda name: str(fake) if name == "claude" else None)
    rc = L.cmd_launch(nvllm_env, "claude", None)  # no model -> catalog default
    out = capfd.readouterr().out
    assert rc == 0
    assert '"model": "acme/TinyLM:main"' in out  # 目录中最大的 completion 模型


def test_launch_env_override(monkeypatch, nvllm_env, tmp_path, capfd):
    monkeypatch.setenv("NVLLM_STATE_DIR", str(tmp_path / "state"))
    fake = _fake_claude(tmp_path)
    monkeypatch.setattr(L, "find_binary", lambda name: str(fake) if name == "claude" else None)
    monkeypatch.setenv("NVLLM_LAUNCH_MODEL", "acme/TinyLM:main")
    rc = L.cmd_launch(nvllm_env, None, None)  # no agent either -> auto-detect fake
    out = capfd.readouterr().out
    assert rc == 0
    assert '"model": "acme/TinyLM:main"' in out


def test_launch_list(capsys):
    rc = L.cmd_launch("http://127.0.0.1:1", list_only=True)  # --list needs no server
    out = capsys.readouterr().out
    assert rc == 0
    assert "AGENT" in out and "PROTOCOL" in out
    for name in ("claude", "codex", "qwen", "gemini", "opencode", "aider"):
        assert name in out
