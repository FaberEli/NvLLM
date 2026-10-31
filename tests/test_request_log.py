"""Request-log middleware (0.7.0): one line per request, model context, level by status."""
from __future__ import annotations

import logging


def _request_records(caplog):
    return [rec for rec in caplog.records if rec.getMessage().startswith("request ")]


def test_request_line_on_success(client, caplog):
    with caplog.at_level(logging.INFO, logger="nvllm_service.main"):
        r = client.post(
            "/api/chat",
            json={"model": "acme/TinyLM:main", "messages": [{"role": "user", "content": "hi"}], "stream": False},
        )
    assert r.status_code == 200
    recs = _request_records(caplog)
    assert recs, "expected a request-log line"
    msg = recs[0].getMessage()
    assert "POST /api/chat" in msg
    assert "-> 200" in msg
    assert "model=acme/TinyLM:main" in msg


def test_request_line_404_logs_warning(client, caplog):
    with caplog.at_level(logging.WARNING, logger="nvllm_service.main"):
        r = client.post(
            "/api/chat",
            json={"model": "acme/DoesNotExist", "messages": [{"role": "user", "content": "hi"}], "stream": False},
        )
    assert r.status_code == 404
    recs = _request_records(caplog)
    assert recs
    assert recs[0].levelno >= logging.WARNING


def test_request_line_keeps_virtual_name(client, caplog):
    # a created (virtual) model must show up as requested=<virtual> model=<base>
    cr = client.post(
        "/api/create",
        json={"name": "assistant", "model": "acme/TinyLM:main", "system": "be brief"},
    )
    assert cr.status_code == 200
    with caplog.at_level(logging.INFO, logger="nvllm_service.main"):
        r = client.post(
            "/api/chat",
            json={"model": "assistant", "messages": [{"role": "user", "content": "hi"}], "stream": False},
        )
    assert r.status_code == 200
    recs = _request_records(caplog)
    assert recs
    msg = recs[0].getMessage()
    assert "requested=assistant" in msg
    assert "model=acme/TinyLM:main" in msg


def test_metadata_endpoints_are_logged(client, caplog):
    with caplog.at_level(logging.INFO, logger="nvllm_service.main"):
        r = client.get("/api/tags")
    assert r.status_code == 200
    recs = _request_records(caplog)
    assert any("GET /api/tags" in rec.getMessage() for rec in recs)
