"""Unit tests for the Anthropic <-> OpenAI translation (no server needed)."""
import json

from nvllm_service import anthropic as A


def test_request_body_basic():
    oai = A.anthropic_chat_body(
        "real/model:main",
        {
            "model": "claude-anything",
            "max_tokens": 100,
            "system": "be brief",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.3,
            "top_p": 0.9,
            "stop_sequences": ["END"],
        },
    )
    assert oai["model"] == "real/model:main"  # server always overrides the model
    assert oai["messages"][0] == {"role": "system", "content": "be brief"}
    assert oai["messages"][1] == {"role": "user", "content": "hi"}
    assert oai["max_tokens"] == 100
    assert oai["temperature"] == 0.3
    assert oai["top_p"] == 0.9
    assert oai["stop"] == ["END"]
    assert oai["stream"] is False


def test_request_body_system_blocks():
    oai = A.anthropic_chat_body(
        "m",
        {"model": "x", "max_tokens": 10, "system": [{"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}],
         "messages": [{"role": "user", "content": "hi"}]},
    )
    assert oai["messages"][0] == {"role": "system", "content": "part one part two"}


def test_request_body_content_blocks_image_and_tool_result():
    oai = A.anthropic_chat_body(
        "m",
        {
            "model": "x",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}},
                    ],
                },
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "result!"}]},
            ],
        },
    )
    msgs = oai["messages"]
    assert msgs[0]["content"][0] == {"type": "text", "text": "look"}
    assert msgs[0]["content"][1]["image_url"]["url"] == "data:image/png;base64,AAA"
    assert msgs[2] == {"role": "tool", "tool_call_id": "t1", "content": "result!"}


def test_request_body_tools_and_tool_use():
    oai = A.anthropic_chat_body(
        "m",
        {
            "model": "x",
            "max_tokens": 10,
            "tools": [
                {"name": "get_weather", "description": "d", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}
            ],
            "tool_choice": {"type": "auto"},
            "messages": [
                {"role": "user", "content": "weather in Paris"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "tu1", "name": "get_weather", "input": {"city": "Paris"}}]},
            ],
        },
    )
    assert oai["tools"][0]["type"] == "function"
    assert oai["tools"][0]["function"]["name"] == "get_weather"
    assert oai["tool_choice"] == "auto"
    tc = oai["messages"][1]["tool_calls"][0]
    assert tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"city": "Paris"}


def test_response_message():
    oai = {
        "choices": [{"message": {"role": "assistant", "content": "hello!"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }
    msg = A.anthropic_message("m1", oai)
    assert msg["type"] == "message"
    assert msg["role"] == "assistant"
    assert msg["content"] == [{"type": "text", "text": "hello!"}]
    assert msg["stop_reason"] == "end_turn"
    assert msg["usage"] == {"input_tokens": 5, "output_tokens": 3}
    assert msg["id"].startswith("msg_")


def test_response_stop_reasons_and_tool_use_blocks():
    m = A.anthropic_message("m1", {"choices": [{"message": {}, "finish_reason": "length"}]})
    assert m["stop_reason"] == "max_tokens"

    m2 = A.anthropic_message(
        "m1",
        {"choices": [{"message": {"tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": '{"a": 1}'}}]}, "finish_reason": "tool_calls"}]},
    )
    assert m2["stop_reason"] == "tool_use"
    assert m2["content"][0]["type"] == "tool_use"
    assert m2["content"][0]["name"] == "f"
    assert m2["content"][0]["input"] == {"a": 1}


def test_stream_state_machine():
    acc = A.AnthropicStream("m1")
    evs = acc.start() + acc.apply({"choices": [{"delta": {"content": "Hel"}}]}) + acc.apply({"choices": [{"delta": {"content": "lo"}}]}) + acc.close()
    text = "".join(e.decode() for e in evs)
    for event in ("message_start", "ping", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"):
        assert f"event: {event}" in text, event
    assert '"text": "Hel"' in text and '"text": "lo"' in text
    assert "end_turn" in text


def test_stream_buffers_tool_calls_into_blocks():
    acc = A.AnthropicStream("m1")
    evs = (
        acc.start()
        + acc.apply({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "get_"}}]}}]})
        + acc.apply({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"x": 2}'}}]}}]})
        + acc.apply({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
        + acc.close()
    )
    text = "".join(e.decode() for e in evs)
    assert '"name": "get_"' in text
    assert "toolu_0" in text
    assert '"input": {"x": 2}' in text
    assert '"stop_reason": "tool_use"' in text
