"""Ollama-style client commands for the `nvllm` CLI.

Mirrors how `ollama` talks to `ollama serve`: every command (list / ps /
show / run / stop) is a thin HTTP client against a running nvllm server.
The server is the single source of truth for the catalog (scanned from the
local HuggingFace cache) and worker state, so these commands never touch the
cache themselves and never need vLLM installed locally.

Server URL precedence: ``--server`` flag > ``$NVLLM_SERVER`` > http://127.0.0.1:8004
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import httpx

from . import __version__
from .render import human_bytes, human_time, is_past, render_table

DEFAULT_URL = "http://127.0.0.1:8004"
_CONNECT = httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=5.0)
_STREAM = httpx.Timeout(connect=15.0, read=None, write=60.0, pool=15.0)


class CLIError(Exception):
    """Fatal, user-facing error; the message is already worded for display."""


# ---- plumbing ---------------------------------------------------------------

def server_url(flag: str | None = None) -> str:
    return (flag or os.environ.get("NVLLM_SERVER") or DEFAULT_URL).rstrip("/")


def _connect_hint(url: str) -> CLIError:
    return CLIError(
        f"could not connect to the nvllm server at {url}\n"
        f"  start it first:       nvllm serve\n"
        f"  or point elsewhere:   NVLLM_SERVER=http://host:port nvllm <cmd>"
    )


def require_server(url: str) -> None:
    try:
        with httpx.Client(timeout=5.0) as c:
            r = c.get(f"{url}/healthz")
            r.raise_for_status()
    except httpx.HTTPError as e:
        raise _connect_hint(url) from e


def _error_detail(r: httpx.Response) -> str:
    try:
        detail = r.json().get("error")
        if isinstance(detail, str) and detail:
            return detail
    except (ValueError, httpx.HTTPError):
        pass
    return f"server returned HTTP {r.status_code}"


def _get_json(url: str, path: str) -> dict:
    try:
        with httpx.Client(timeout=_CONNECT) as c:
            r = c.get(f"{url}{path}")
    except httpx.HTTPError as e:
        raise _connect_hint(url) from e
    return _ok(r)


def _post_json(url: str, path: str, body: dict) -> dict:
    try:
        with httpx.Client(timeout=_CONNECT) as c:
            r = c.post(f"{url}{path}", json=body)
    except httpx.HTTPError as e:
        raise _connect_hint(url) from e
    return _ok(r)


def _ok(r: httpx.Response) -> dict:
    if r.status_code >= 400:
        raise CLIError(_error_detail(r))
    if not r.content:
        return {}
    try:
        return r.json()
    except ValueError:
        return {}


# ---- list / ps / show / stop / version ---------------------------------------

def cmd_list(url: str, prefix: str | None = None) -> int:
    require_server(url)
    data = _get_json(url, "/api/tags")
    rows: list[list[str]] = []
    for m in data.get("models") or []:
        name = str(m.get("name") or m.get("model") or "")
        if (m.get("details") or {}).get("note"):
            name = f"{name}  (unservable)"
        if prefix and not name.lower().startswith(prefix.lower()):
            continue
        rows.append(
            [name, str(m.get("digest") or "")[:12], human_bytes(int(m.get("size") or 0)), human_time(m.get("modified_at"))]
        )
    if not rows:
        print(f"no models matching '{prefix}'" if prefix else "no models found in the HuggingFace cache")
        return 0
    print(render_table(["NAME", "ID", "SIZE", "MODIFIED"], rows))
    return 0


def cmd_ps(url: str) -> int:
    require_server(url)
    data = _get_json(url, "/api/ps")
    rows: list[list[str]] = []
    for m in data.get("models") or []:
        until = m.get("expires_at")
        if until == "persistent":
            until_s = "persistent"
        else:
            until_s = "Stopping..." if is_past(until) else human_time(until)
        rows.append(
            [str(m.get("name")), str(m.get("digest") or "")[:12], human_bytes(int(m.get("size") or 0)), "100% GPU", until_s]
        )
    if not rows:
        print("no models loaded")
        return 0
    print(render_table(["NAME", "ID", "SIZE", "PROCESSOR", "UNTIL"], rows))
    return 0


def cmd_show(url: str, model: str) -> int:
    require_server(url)
    body = _post_json(url, "/api/show", {"model": model})
    details = body.get("details") or {}

    def kv(label: str, value: Any) -> None:
        if value not in (None, "", [], 0):
            print(f"  {label:<17}{value}")

    caps = body.get("capabilities") or []
    print(f"{model}")
    print(f"  {'family':<17}{details.get('family') or '—'}")
    kv("parameter_size", details.get("parameter_size"))
    kv("quantization", details.get("quantization_level"))
    kv("context_length", details.get("context_length"))
    kv("embedding_length", details.get("embedding_length"))
    if caps:
        print(f"  {'capabilities':<17}{', '.join(caps)}")
    model_info = body.get("model_info") or {}
    if model_info:
        print()
        width = max(len(k) for k in model_info) + 1
        for k, v in model_info.items():
            if v not in (None, "", 0):
                print(f"  {k:<{width}}{v}")
    # created (virtual) models: show the wrapper the server injects per request
    if body.get("system"):
        print("\n  system prompt:")
        for ln in str(body["system"]).splitlines():
            print(f"    {ln}")
    if body.get("options"):
        print("\n  options:")
        for k, v in body["options"].items():
            print(f"    {k} = {v}")
    print(f"\n  run: nvllm run {model}")
    return 0


def cmd_stop(url: str, model: str) -> int:
    require_server(url)
    body = _post_json(url, "/api/unload", {"model": model})
    print(f"unloaded '{body.get('model') or model}'")
    return 0


def cmd_create(url: str, name: str, base: str, system: str = "", options: dict | None = None) -> int:
    """`nvllm create NAME --from BASE [--system ...]` — virtual model over a catalog model."""
    require_server(url)
    body = _post_json(
        url, "/api/create", {"name": name, "model": base, "system": system, "options": options or {}}
    )
    print(f"created '{body.get('name') or name}' (base: {body.get('base') or base})")
    return 0


def cmd_rm(url: str, name: str) -> int:
    """Delete a *created* (virtual) model. Real HF-cache models are never deleted
    by the server — for those we print the NvLLM-trait guidance instead."""
    require_server(url)
    try:
        body = _post_json(url, "/api/rm", {"name": name})
    except CLIError as e:
        if "not a created model" in str(e):
            print(
                f"'{name}' is a real model from your HuggingFace cache, not a created one.\n"
                "nvllm does not delete cache models — delete the cache directory yourself:\n"
                "\n"
                "    rm -rf ~/.cache/huggingface/hub/models--<org>--<name>\n"
            )
            return 1
        raise
    print(f"deleted '{body.get('name') or name}'")
    return 0


def cmd_route(url: str, text: str) -> int:
    """`nvllm route "text"` — preview which model the router would pick (dry-run).

    Runs only the judge (decision model), not the routed backend model, so it is
    cheap and safe to use while exploring routing.
    """
    require_server(url)
    body = _post_json(url, "/api/route", {"model": "auto", "text": text})
    print(f"route -> {body.get('chosen')}    (backend: {body.get('backend')}, judge: {body.get('judge')})")
    dist = body.get("distribution") or {}
    if dist:
        print("  distribution:")
        for name, p in sorted(dist.items(), key=lambda kv: kv[1], reverse=True):
            bar = "#" * int(round(float(p) * 30))
            print(f"    {float(p):7.4f}  {bar:<30}  {name}")
    raw = body.get("raw")
    if raw:
        print(f"  raw answer: {raw!r}")
    return 0


def _router_on_body(judge: str, candidates: list[str] | None, backend: str | None, top_logprobs: int | None) -> dict:
    body = {"action": "on", "judge": judge}
    if candidates:
        body["candidates"] = candidates
    if backend:
        body["backend"] = backend
    if top_logprobs:
        body["top_logprobs"] = top_logprobs
    return body


def _pick_recommender(url: str) -> tuple[str, bool]:
    """(recommender name, was_loaded). Prefer a *currently-loaded* completion model
    that is not a decision model; else the largest completion model in the cache.
    The recommender must be a real chat model (the judge itself is a decision
    model and is excluded here)."""
    tags = _get_json(url, "/api/tags").get("models") or []
    completion = [
        m for m in tags
        if isinstance(m, dict)
        and "completion" in (m.get("capabilities") or [])
        and not (m.get("details") or {}).get("decision_model")
    ]
    if not completion:
        return "", False
    try:
        loaded = {m.get("name") or m.get("model") for m in _get_json(url, "/api/ps").get("models") or []}
    except CLIError:
        loaded = set()
    for m in sorted(completion, key=lambda m: m.get("size") or 0, reverse=True):
        if m.get("name") in loaded:
            return str(m["name"]), True
    biggest = max(completion, key=lambda m: m.get("size") or 0)
    return str(biggest.get("name")), False


def cmd_router(
    url: str,
    action: str,
    *,
    judge: str | None = None,
    candidates: list[str] | None = None,
    backend: str | None = None,
    top_logprobs: int | None = None,
    recommend: bool = True,
    in_fn=None,
    out_fn=None,
) -> int:
    """``nvllm router on|off|status`` — activate / deactivate / show auto-routing.

    ``on`` scans the cache for decision models, lets you pick the judge (and, if
    you like, asks a running local model to recommend one), and persists the choice;
    it takes effect on the next ``nvllm serve`` (the judge becomes resident and an
    ``AutoRouter`` option appears). ``in_fn``/``out_fn`` are injectable for tests.
    """
    in_fn = in_fn or input
    out_fn = out_fn or print
    require_server(url)

    if action == "status":
        info = _get_json(url, "/api/router")
        if info.get("enabled"):
            out_fn(f"router: ACTIVE     (judge: {info.get('judge')}, backend: {info.get('backend')})")
        elif info.get("configured"):
            out_fn(f"router: CONFIGURED (judge: {info.get('judge')}) — active on the next `nvllm serve`")
        else:
            out_fn("router: not activated (run `nvllm router on`)")
        if info.get("enabled") or info.get("configured"):
            cands = info.get("candidates") or []
            out_fn(f"  candidates: {', '.join(cands) if cands else 'all servable non-decision models'}")
        dms = info.get("decision_models") or []
        if dms:
            out_fn(f"decision models in cache ({len(dms)}):")
            for i, d in enumerate(dms, 1):
                size = f" [{d['parameter_size']}]" if d.get("parameter_size") else ""
                desc = f" — {d['description']}" if d.get("description") else ""
                out_fn(f"  {i}. {d['name']}{size}{desc}")
        else:
            out_fn("decision models in cache: none (download one with `hf download <org>/<decision-model>`)")
        return 0

    if action == "off":
        _post_json(url, "/api/router", {"action": "off"})
        out_fn("router deactivated — takes effect on the next `nvllm serve`")
        return 0

    if action != "on":
        raise CLIError(f"unknown router action '{action}' (expected on/off/status)")

    # --- on ----------------------------------------------------------------
    info = _get_json(url, "/api/router")
    dms = [d for d in (info.get("decision_models") or []) if d.get("name")]

    # non-interactive: --judge given directly
    if judge:
        body = _post_json(url, "/api/router", _router_on_body(judge, candidates, backend, top_logprobs))
        out_fn(f"router activated (judge: {body.get('judge')})")
        out_fn(f"  {body.get('note') or 'the judge becomes resident; AutoRouter appears on the next `nvllm serve`'}")
        return 0

    if not dms:
        out_fn("could not activate the router: no System One decision models found in the cache.")
        out_fn("  NvLLM detects them by a decision-marker file (e.g. `decision_config.json` /")
        out_fn("  `model_router.yaml`) or a 'decision' name; or name one directly with `nvllm router on --judge <name>`.")
        out_fn("  Download one, then retry `nvllm router on`:")
        out_fn("    pip install -U 'huggingface_hub[cli]'")
        out_fn("    hf download <org>/<decision-model>      # e.g. a StartLux / SII / Intern decision model")
        return 1

    def _list() -> None:
        out_fn("Choose the decision model to act as the router judge:")
        for i, d in enumerate(dms, 1):
            size = f" [{d['parameter_size']}]" if d.get("parameter_size") else ""
            desc = f" — {d['description']}" if d.get("description") else ""
            out_fn(f"  {i}. {d['name']}{size}{desc}")

    recommender, was_loaded = ("", False)
    if recommend:
        try:
            recommender, was_loaded = _pick_recommender(url)
        except CLIError:
            recommender = ""

    _list()
    if recommender:
        tag = "currently loaded" if was_loaded else "largest local model"
        out_fn(f"  R. let {recommender} ({tag}) recommend one")

    while True:
        try:
            choice = in_fn("select: ").strip()
        except (EOFError, StopIteration):
            raise CLIError("aborted — no selection made")
        if recommender and choice.lower() in ("r", "recommend"):
            body = _post_json(url, "/api/router/recommend", {"recommender": recommender, "decision_models": dms})
            rec_text = (body.get("recommendation") or "").strip()
            out_fn(f"\n{recommender} says:")
            for line in (rec_text.splitlines() or ["  (no recommendation text)"]):
                out_fn(f"    {line}")
            if body.get("suggested"):
                out_fn(f"  → it suggests: {body['suggested']}")
            _list()
            out_fn(f"  R. ask {recommender} again")
            continue
        if choice.isdigit():
            n = int(choice)
            if 1 <= n <= len(dms):
                chosen = dms[n - 1]["name"]
                body = _post_json(url, "/api/router", _router_on_body(chosen, candidates, backend, top_logprobs))
                out_fn(f"router activated (judge: {body.get('judge')})")
                out_fn(f"  {body.get('note') or 'the judge becomes resident; AutoRouter appears on the next `nvllm serve`'}")
                return 0
        out_fn(f"  enter a number 1-{len(dms)}" + (" or R for a recommendation" if recommender else ""))


def cmd_version(url: str) -> int:
    print(f"nvllm {__version__}")
    try:
        data = _get_json(url, "/api/version")
        print(f"server  nvllm-serve {data.get('version')} @ {url}")
    except CLIError:
        print(f"server  not running at {url} (start it with: nvllm serve)")
    return 0


# ---- run: one-shot prompt + interactive REPL ---------------------------------

def _looks_loaded(url: str, model: str) -> bool:
    """Best-effort check (for the 'loading…' hint): is `model` already resident?"""
    m = model.strip()
    last = m.split("/")[-1]
    try:
        data = _get_json(url, "/api/ps")
    except CLIError:
        return True  # unknown; don't spam the hint
    for e in data.get("models") or []:
        n = str(e.get("name") or "")
        repo = n.split(":", 1)[0]
        bare = repo.split("/")[-1]
        if n == m or repo == m or bare == last:
            return True
    return False


def _chat_stream(url: str, body: dict, out_fn: Callable[..., None]) -> tuple[str, dict, list]:
    """POST /api/chat (streaming NDJSON), printing content as it arrives.

    Returns (accumulated_text, stats, tool_calls) — stats has eval_count /
    total_duration; tool_calls is the assistant's tool calls (empty if none).
    ``out_fn`` must accept ``print``-style args (tests inject a sink).
    """
    try:
        with httpx.Client(timeout=_STREAM) as c:
            with c.stream("POST", f"{url}/api/chat", json=body) as r:
                if r.status_code >= 400:
                    r.read()  # materialize the body (streaming responses need this for .json())
                    raise CLIError(_error_detail(r))
                parts: list[str] = []
                stats: dict = {}
                tcs: list = []
                for line in r.iter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("error"):
                        raise CLIError(str(obj["error"]))
                    msg = obj.get("message") or {}
                    if msg.get("tool_calls"):
                        tcs = msg["tool_calls"]
                    content = msg.get("content")
                    if content:
                        parts.append(content)
                        out_fn(content, end="")
                        sys.stdout.flush()
                    if obj.get("done"):
                        stats = {
                            "eval_count": int(obj.get("eval_count") or 0),
                            "total_duration": int(obj.get("total_duration") or 0),
                        }
                        # tool calls: the server re-emits the merged list on each
                        # tool chunk; print the final list once, after the text
                        for tc in tcs:
                            fn = (tc or {}).get("function") or {}
                            args = fn.get("arguments") or {}
                            try:
                                args_s = json.dumps(args, ensure_ascii=False)
                            except (TypeError, ValueError):
                                args_s = str(args)
                            out_fn(f"\n⚙ tool: {fn.get('name', '')}({args_s})")
                        break
    except httpx.HTTPError as e:
        raise CLIError(f"connection to {url} dropped: {e}") from e
    return "".join(parts), stats, tcs


_IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}
_IMAGE_EXTS = set(_IMAGE_MIME)


def _image_data_uri(path: str) -> str:
    """Read a local image file as a data: URI (the server turns it into an
    OpenAI image_url part; sending a full data: URI preserves the right MIME)."""
    p = Path(path)
    if not p.is_file():
        raise CLIError(f"image not found: {path}")
    mime = _IMAGE_MIME.get(p.suffix.lower(), "image/png")
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def is_image_path(line: str) -> bool:
    p = Path(line)
    return bool(p.suffix) and p.suffix.lower() in _IMAGE_EXTS and p.is_file()


def load_tools_file(path: str) -> list[dict]:
    """Load an OpenAI-style tools JSON array (validated: list of objects)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise CLIError(f"could not read tools file {path}: {e}")
    if not isinstance(data, list) or not all(isinstance(t, dict) for t in data):
        raise CLIError(f"tools file {path} must contain a JSON array of tool objects")
    return data


def run_prompt(
    url: str,
    model: str,
    prompt: str,
    options: dict | None = None,
    verbose: bool = False,
    images: list[str] | None = None,
    tools: list[dict] | None = None,
) -> int:
    """One-shot: `nvllm run MODEL "prompt" [--image P ...] [--tools FILE]`."""
    require_server(url)
    if not _looks_loaded(url, model):
        print("loading model (first load can take a while)…", file=sys.stderr)
    user_msg: dict[str, Any] = {"role": "user", "content": prompt}
    if images:
        user_msg["images"] = [_image_data_uri(p) for p in images]
    body: dict[str, Any] = {
        "model": model,
        "messages": [user_msg],
        "stream": True,
    }
    if options:
        body["options"] = options
    if tools:
        body["tools"] = tools
    _text, stats, _tcs = _chat_stream(url, body, out_fn=print)
    print()
    if verbose:
        n, dur_ns = int(stats.get("eval_count") or 0), int(stats.get("total_duration") or 0)
        if n and dur_ns > 0:
            print(f"{n / (dur_ns / 1e9):.1f} tok/s — {n} tokens in {dur_ns / 1e9:.1f}s", file=sys.stderr)
    return 0


_REPL_HELP = """\
commands:
  /help                      show this help
  /set temperature <0-2>     set sampling temperature
  /set top_p <0-1>           set nucleus sampling
  /set num_predict <n>       max tokens per reply
  /tools <file>              load OpenAI-style tools (JSON array) for the session
  /show                      show the current model's details
  /clear                     clear the conversation history
  /bye                       exit (also: /exit, /quit)

Type a message to chat; end a line with a backslash (\\) to continue it.
Type a local image path (png/jpg/jpeg/webp/gif/bmp) to send it as a turn.
"""


def repl(
    url: str,
    model: str,
    options: dict | None = None,
    tools: list[dict] | None = None,
    *,
    in_fn: Callable[[str], str] = input,
    out_fn: Callable[..., None] = print,
) -> int:
    """Interactive chat REPL (like `ollama run`).

    ``in_fn``/``out_fn`` are injected so tests can script a whole session.
    """
    require_server(url)
    options = dict(options or {})
    session_tools: list[dict] = list(tools or [])
    history: list[dict] = []
    out_fn(f"nvllm run {model} — /help for commands, /bye to exit")
    while True:
        try:
            line = in_fn(f"{model} >>> ")
        except EOFError:
            return 0
        # multi-line: trailing backslash continues on the next line (like Ollama)
        while line.endswith("\\"):
            try:
                cont = in_fn("        … ")
            except EOFError:
                break
            line = line[:-1] + " " + cont
        line = line.strip()
        if not line:
            continue
        if line in ("/bye", "/exit", "/quit"):
            return 0
        if line in ("/help", "/?"):
            out_fn(_REPL_HELP)
            continue
        if line == "/clear":
            history.clear()
            out_fn("conversation cleared")
            continue
        if line == "/show":
            try:
                cmd_show(url, model)
            except CLIError as e:
                out_fn(f"error: {e}")
            continue
        if line.startswith("/set "):
            parts = line.split()
            if len(parts) != 3:
                out_fn("usage: /set <temperature|top_p|num_predict> <value>")
                continue
            key, raw = parts[1], parts[2]
            if key not in ("temperature", "top_p", "num_predict"):
                out_fn(f"unknown option '{key}' — try: temperature, top_p, num_predict")
                continue
            try:
                options[key] = int(raw) if key == "num_predict" else float(raw)
            except ValueError:
                out_fn(f"invalid value '{raw}' for {key}")
                continue
            out_fn(f"{key} = {options[key]}")
            continue
        if line == "/tools" or line.startswith("/tools "):
            arg = line.split(None, 1)[1].strip() if line != "/tools" else None
            if arg is None:
                if session_tools:
                    names = ", ".join(((t.get("function") or {}).get("name")) or "?" for t in session_tools)
                    out_fn(f"tools ({len(session_tools)}): {names}")
                else:
                    out_fn("no tools loaded — usage: /tools /path/to/tools.json")
                continue
            try:
                session_tools = load_tools_file(arg)
                out_fn(f"loaded {len(session_tools)} tool(s) from {arg}")
            except CLIError as e:
                out_fn(f"could not load tools from {arg}: {e}")
            continue
        # image turn FIRST: an absolute image path starts with "/" and must not
        # be mistaken for a slash command
        if is_image_path(line):
            user_msg: dict[str, Any] = {"role": "user", "content": "", "images": [_image_data_uri(line)]}
        elif line.startswith("/"):
            out_fn(f"unknown command '{line.split()[0]}' — /help for a list")
            continue
        else:
            user_msg = {"role": "user", "content": line}
        history.append(user_msg)
        body: dict[str, Any] = {"model": model, "messages": list(history), "stream": True}
        if options:
            body["options"] = options
        if session_tools:
            body["tools"] = session_tools
        try:
            text, _stats, tcs = _chat_stream(url, body, out_fn)
        except CLIError as e:
            history.pop()  # drop the failed turn so the history stays clean
            out_fn(f"error: {e}")
            continue
        out_fn()
        asst: dict[str, Any] = {"role": "assistant"}
        if text:
            asst["content"] = text
        if tcs:
            asst["tool_calls"] = tcs
        if len(asst) > 1:
            history.append(asst)
