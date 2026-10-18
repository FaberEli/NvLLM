"""Anthropic Messages API <-> OpenAI chat translation.

Lets Anthropic-protocol clients (notably Claude Code via `nvllm launch claude`)
talk to nvllm workers, which speak OpenAI. The worker path is unchanged: this
module only translates request bodies in and responses out — the same Worker
machinery (ensure / track / keep-alive) serves the endpoint.

Supported: text + image content blocks, system (string or blocks), tools /
tool_use / tool_result, stop_sequences, and streaming (SSE event translation).
"""
from __future__ import annotations

import json
import re
from typing import Any

from . import __version__

_MSG_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _msg_id() -> str:
    import secrets

    return f"msg_{secrets.token_hex(12)}"


def _data_url(block: dict) -> str:
    src = block.get("source") or {}
    media = str(src.get("media_type") or "image/png")
    data = str(src.get("data") or "")
    return f"data:{media};base64,{data}"


def _openai_tool(name: str, description: str, input_schema: Any) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or "",
            "parameters": input_schema or {"type": "object", "properties": {}},
        },
    }


def _tool_choice_openai(tc: Any) -> Any:
    if tc is None:
        return None
    if isinstance(tc, str):
        return tc
    if isinstance(tc, dict):
        t = tc.get("type")
        if t == "auto":
            return "auto"
        if t in ("any", "required"):
            return "required"
        if t == "tool" and tc.get("name"):
            return {"type": "function", "function": {"name": tc["name"]}}
    return None


def _block_to_text(block: dict) -> str:
    if block.get("type") == "text":
        return str(block.get("text") or "")
    if block.get("type") == "thinking":
        return str(block.get("thinking") or "")
    return ""


def anthropic_chat_body(model: str, body: dict) -> dict:
    """Anthropic Messages request -> OpenAI /v1/chat/completions body."""
    messages: list[dict] = []

    system = body.get("system")
    if isinstance(system, str) and system:
        messages.append({"role": "system", "content": system})
    elif isinstance(system, list):
        text = "".join(_block_to_text(b) for b in system if isinstance(b, dict)).strip()
        if text:
            messages.append({"role": "system", "content": text})

    for m in body.get("messages") or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        tool_calls: list[dict] = []

        if role == "tool" or (isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        )):
            # tool results: one OpenAI 'tool' message per block
            blocks = content if isinstance(content, list) else [{"type": "tool_result", "content": content}]
            for b in blocks:
                if not isinstance(b, dict) or b.get("type") != "tool_result":
                    continue
                inner = b.get("content")
                if isinstance(inner, list):
                    inner = "".join(_block_to_text(x) for x in inner if isinstance(x, dict))
                messages.append(
                    {"role": "tool", "tool_call_id": str(b.get("tool_use_id") or "call_0"), "content": str(inner or "")}
                )
            continue

        if isinstance(content, str) or content is None:
            messages.append({"role": role if role in ("user", "assistant") else "user", "content": content or ""})
            continue

        # content is a list of blocks — preserve order (text / image / tool_use)
        parts: list[dict] = []
        for b in content:
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "text":
                t = str(b.get("text") or "")
                if t:
                    parts.append({"type": "text", "text": t})
            elif bt == "image":
                parts.append({"type": "image_url", "image_url": {"url": _data_url(b)}})
            elif bt == "tool_use":
                args = b.get("input") or {}
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                tool_calls.append(
                    {
                        "id": str(b.get("id") or "call_0"),
                        "type": "function",
                        "function": {"name": str(b.get("name") or ""), "arguments": args},
                    }
                )
            # thinking / redacted_thinking: not forwarded (OpenAI has no counterpart)
        msg: dict[str, Any] = {"role": role if role in ("user", "assistant") else "user"}
        if parts:
            msg["content"] = parts
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if "content" not in msg and "tool_calls" not in msg:
            msg["content"] = ""
        messages.append(msg)

    oai: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": int(body.get("max_tokens") or 4096),
        "stream": bool(body.get("stream", False)),
    }
    for key in ("temperature", "top_p"):
        if body.get(key) is not None:
            oai[key] = body[key]
    stops = body.get("stop_sequences")
    if stops:
        oai["stop"] = stops if isinstance(stops, list) else [stops]
    tools = body.get("tools")
    if tools:
        oai["tools"] = [
            _openai_tool(t.get("name", ""), t.get("description", ""), t.get("input_schema"))
            for t in tools
            if isinstance(t, dict) and t.get("name")
        ]
    tc = _tool_choice_openai(body.get("tool_choice"))
    if tc is not None:
        oai["tool_choice"] = tc
    return oai


# ---- responses ---------------------------------------------------------------

def _ollama_stop_reason(finish_reason: Any) -> str:
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "tool_calls":
        return "tool_use"
    return "end_turn"


def _openai_tool_calls_to_blocks(calls: list) -> list[dict]:
    blocks = []
    for i, tc in enumerate(calls or []):
        fn = tc.get("function") or {}
        args = fn.get("arguments", "")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args else {}
            except json.JSONDecodeError:
                args = {}
        blocks.append(
            {
                "type": "tool_use",
                "id": str(tc.get("id") or f"toolu_{i}"),
                "name": str(fn.get("name") or ""),
                "input": args,
            }
        )
    return blocks


def anthropic_message(model: str, oai_resp: dict) -> dict:
    """OpenAI chat completion -> Anthropic message (non-streaming)."""
    choice = (oai_resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content: list[dict] = []
    text = msg.get("content")
    if text:
        content.append({"type": "text", "text": text})
    content.extend(_openai_tool_calls_to_blocks(msg.get("tool_calls") or []))

    usage = oai_resp.get("usage") or {}
    return {
        "id": _msg_id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": _ollama_stop_reason(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    }


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


class AnthropicStream:
    """State machine translating OpenAI chat chunks (streaming) into Anthropic SSE.

    Usage from the endpoint::

        acc = AnthropicStream(model)
        for ev in acc.start():      yield ev
        for evt in openai_chunks:   # async iterator of parsed OpenAI chunks
            for ev in acc.apply(evt): yield ev
        for ev in acc.close():      yield ev

    Text deltas stream live (index 0); tool-call fragments are buffered and
    emitted as trailing ``tool_use`` blocks before ``message_delta``.
    Note: ``finish`` (the OpenAI finish_reason) is an *attribute*; the tail-
    event method is ``close()`` — do not rename one onto the other (an instance
    attribute would shadow the method).
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self.id = _msg_id()
        self.tool_acc: dict[int, dict] = {}
        self.finish: str | None = None
        self.usage: dict | None = None

    def start(self) -> list[bytes]:
        return [
            _sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": self.id,
                        "type": "message",
                        "role": "assistant",
                        "model": self.model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                },
            ),
            _sse("ping", {"type": "ping"}),
            _sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ]

    def apply(self, evt: dict) -> list[bytes]:
        out: list[bytes] = []
        for choice in evt.get("choices") or []:
            if choice.get("finish_reason"):
                self.finish = choice["finish_reason"]
            delta = choice.get("delta") or {}
            for tc in delta.get("tool_calls") or []:
                idx = int(tc.get("index", 0))
                slot = self.tool_acc.setdefault(idx, {"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
            text = delta.get("content")
            if text:
                out.append(
                    _sse("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}})
                )
        if isinstance(evt.get("usage"), dict):
            self.usage = evt["usage"]
        return out

    def close(self) -> list[bytes]:
        out = [_sse("content_block_stop", {"type": "content_block_stop", "index": 0})]
        idx = 1
        for key in sorted(self.tool_acc):
            slot = self.tool_acc[key]
            try:
                args = json.loads(slot["arguments"]) if slot["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            block = {"type": "tool_use", "id": f"toolu_{key}", "name": slot["name"], "input": args}
            out.append(_sse("content_block_start", {"type": "content_block_start", "index": idx, "content_block": block}))
            out.append(_sse("content_block_stop", {"type": "content_block_stop", "index": idx}))
            idx += 1
        usage = self.usage or {}
        out.append(
            _sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": _ollama_stop_reason(self.finish), "stop_sequence": None},
                    "usage": {"output_tokens": int(usage.get("completion_tokens") or 0)},
                },
            )
        )
        out.append(_sse("message_stop", {"type": "message_stop"}))
        return out


# Anthropic's documented error types per HTTP status (clients such as Claude
# Code branch on `error.type`, so map the common codes precisely).
_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    408: "timeout_error",
    413: "request_too_large_error",
    429: "rate_limit_error",
}


def error_response(status_code: int, message: str) -> tuple[int, dict]:
    return status_code, {
        "type": "error",
        "error": {"type": _ERROR_TYPES.get(status_code, "api_error"), "message": message},
    }
