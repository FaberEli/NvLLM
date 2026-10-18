"""`nvllm launch` — start a local Agent CLI wired to the nvllm model service.

The Ollama `ollama launch` counterpart: pick an agent (Claude Code, Codex, …),
and it is exec'd in the foreground with its model backend pointed at the
running nvllm server — no cloud, no keys.

- Anthropic-protocol agents (claude) point ANTHROPIC_BASE_URL at the server,
  which serves the Anthropic Messages API at /v1/messages.
- OpenAI-protocol agents (codex, qwen, …) point OPENAI_BASE_URL at /v1.

Claude Code gates the model name on the ANTHROPIC_MODEL env var and the
--model flag (only known claude-* names pass), but the `model` key in a
settings.json — a "custom model option" — accepts ANY name. Verified against
the native 2026 build: ANTHROPIC_MODEL=nvidia/... fails locally with zero
requests sent, while settings.json "model" = "nvidia/..." reaches the server.
So for claude, `launch` writes a small config dir (settings.json with the base
URL, token, and the real model) and execs claude with CLAUDE_CONFIG_DIR.

Agent wiring is pure data in AGENTS, so adding one is a few lines; the exec
path, server check, and model default are shared.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import client
from .client import CLIError


@dataclass(frozen=True)
class Agent:
    name: str                      # CLI name, e.g. "claude"
    display: str                   # human name for --list / help
    binary: str                    # executable searched on PATH
    protocol: str                  # "anthropic" or "openai"
    env: dict[str, str] = field(default_factory=dict)  # {placeholder: value}; <base> / <model> substituted
    args: list[str] = field(default_factory=list)       # base args; <model> substituted
    install_hint: str = ""

    def env_for(self, base_url: str, model: str) -> dict[str, str]:
        return {k: v.replace("<base>", base_url).replace("<model>", model) for k, v in self.env.items()}

    def args_for(self, model: str) -> list[str]:
        return [a.replace("<model>", model) for a in self.args]


# Order = auto-detect preference (first available wins).
AGENTS: dict[str, Agent] = {
    # claude 的模型名不走 env/args（ANTHROPIC_MODEL 被客户端白名单门禁），
    # 而是经 _claude_config_dir() 写进 settings.json 的 `model` 字段
    # （"custom model option"，接受任意名字——见模块 docstring 的验证结论）。
    "claude": Agent(
        name="claude",
        display="Claude Code",
        binary="claude",
        protocol="anthropic",
        env={
            "ANTHROPIC_BASE_URL": "<base>",
            "ANTHROPIC_AUTH_TOKEN": "nvllm",
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_ERROR_REPORTING": "1",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_AUTOUPDATER": "1",
        },
        install_hint="npm install -g @anthropic-ai/claude-code",
    ),
    "codex": Agent(
        name="codex",
        display="Codex CLI",
        binary="codex",
        protocol="openai",
        env={"OPENAI_BASE_URL": "<base>/v1", "OPENAI_API_KEY": "nvllm"},
        args=["--model", "<model>"],
        install_hint="npm install -g @openai/codex",
    ),
    "qwen": Agent(
        name="qwen",
        display="Qwen Code",
        binary="qwen",
        protocol="openai",
        env={"OPENAI_BASE_URL": "<base>/v1", "OPENAI_API_KEY": "nvllm"},
        args=["--model", "<model>"],
        install_hint="npm install -g @qwen-code/qwen-code",
    ),
    "gemini": Agent(
        name="gemini",
        display="Gemini CLI",
        binary="gemini",
        protocol="openai",
        env={"OPENAI_BASE_URL": "<base>/v1", "OPENAI_API_KEY": "nvllm"},
        args=["--model", "<model>"],
        install_hint="npm install -g @google/gemini-cli",
    ),
    "opencode": Agent(
        name="opencode",
        display="opencode",
        binary="opencode",
        protocol="openai",
        env={"OPENAI_BASE_URL": "<base>/v1", "OPENAI_API_KEY": "nvllm"},
        args=["--model", "<model>"],
        install_hint="curl -fsSL https://opencode.ai/install | bash",
    ),
    "aider": Agent(
        name="aider",
        display="aider",
        binary="aider",
        protocol="openai",
        env={"OPENAI_API_KEY": "nvllm"},
        args=["--model", "openai/<model>", "--openai-api-base", "<base>/v1"],
        install_hint="pip install aider-chat",
    ),
}


def find_binary(name: str) -> str | None:
    return shutil.which(name)


def available_agents() -> list[Agent]:
    return [a for a in AGENTS.values() if find_binary(a.binary)]


def pick_agent(name: str | None) -> Agent:
    if name:
        if name not in AGENTS:
            known = ", ".join(AGENTS)
            raise CLIError(f"unknown agent '{name}' — known agents: {known} (see `nvllm launch --list`)")
        return AGENTS[name]
    avail = available_agents()
    if not avail:
        lines = [
            "no known agent CLI is installed on PATH — install one, e.g.:",
            "    claude:   " + AGENTS["claude"].install_hint,
            "    codex:    " + AGENTS["codex"].install_hint,
            "then retry:  nvllm launch",
        ]
        raise CLIError("\n".join(lines))
    return avail[0]


def pick_default_model(catalog: list[dict]) -> str | None:
    """Default launch model: the largest completion-capable model (same
    philosophy as the Router's fallback: prefer capability, not novelty)."""
    best = None
    for m in catalog:
        if "completion" not in (m.get("capabilities") or []):
            continue
        if best is None or int(m.get("size") or 0) > int(best.get("size") or 0):
            best = m
    return best["name"] if best else None


def default_model(catalog: list[dict] | None) -> str | None:
    """$NVLLM_LAUNCH_MODEL wins, else the catalog default."""
    m = os.environ.get("NVLLM_LAUNCH_MODEL")
    if m:
        return m
    return pick_default_model(catalog) if catalog else None


def claude_config_dir(base_url: str, model: str) -> Path:
    """Write the config dir Claude Code will use (see module docstring).

    settings.json: base URL + token in `env`, and the REAL model name in the
    top-level `model` key — the one channel the native Claude Code build
    accepts for non-claude model names.
    """
    state = Path(os.environ.get("NVLLM_STATE_DIR") or (Path.home() / ".local" / "share" / "nvllm-serve"))
    d = state / "claude-launch"
    d.mkdir(parents=True, exist_ok=True)
    (d / "settings.json").write_text(
        json.dumps(
            {
                "env": {
                    "ANTHROPIC_BASE_URL": base_url,
                    "ANTHROPIC_AUTH_TOKEN": "nvllm",
                    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                },
                "model": model,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return d


def launch_config(agent_name: str | None, base_url: str, model: str) -> tuple[Agent, str, dict[str, str], list[str]]:
    """Resolve the agent + exact exec config (testable without exec'ing)."""
    agent = pick_agent(agent_name)
    binary = find_binary(agent.binary)
    if binary is None:
        raise CLIError(
            f"agent '{agent.name}' not found on PATH (looked for '{agent.binary}')\n"
            f"  install:  {agent.install_hint}"
        )
    env = {**os.environ, **agent.env_for(base_url, model)}
    if agent.name == "claude":
        # model name goes via settings.json (CLAUDE_CONFIG_DIR), not ANTHROPIC_MODEL
        env["CLAUDE_CONFIG_DIR"] = str(claude_config_dir(base_url, model))
    return agent, binary, env, agent.args_for(model)


def cmd_launch(
    url: str,
    agent_name: str | None = None,
    model: str | None = None,
    extra: list[str] | None = None,
    list_only: bool = False,
) -> int:
    """`nvllm launch [AGENT] [EXTRA...]` — exec the agent in the foreground."""
    if list_only:
        print(f"{'AGENT':<10} {'PROTOCOL':<11} STATUS")
        for a in AGENTS.values():
            status = find_binary(a.binary) or f"not installed — {a.install_hint}"
            print(f"{a.name:<10} {a.protocol:<11} {status}")
        return 0

    client.require_server(url)
    if not model:
        data = client._get_json(url, "/api/tags")
        model = default_model(data.get("models") or [])
        if not model:
            raise CLIError("no completion-capable model in the catalog — pass --model")
    else:
        # validate the name resolves server-side before exec'ing anything
        client._post_json(url, "/api/show", {"model": model})

    agent, binary, env, args = launch_config(agent_name, url, model)
    if agent.name == "claude":
        print(f"nvllm: claude config: {env.get('CLAUDE_CONFIG_DIR')} (model '{model}')", file=sys.stderr)
    cmd = [binary, *args, *(extra or [])]
    print(f"nvllm: launching {agent.display} → model '{model}' @ {url}", file=sys.stderr)
    try:
        rc = subprocess.call(cmd, env=env)
    except FileNotFoundError:
        raise CLIError(f"failed to exec '{binary}'") from None
    return int(rc)
