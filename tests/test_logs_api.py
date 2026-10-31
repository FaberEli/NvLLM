"""`GET /api/logs` — list + tail + path-traversal guard (unauthenticated endpoint: strict)."""
from __future__ import annotations


def test_list_empty(client, settings):
    settings.logs_dir().mkdir(parents=True, exist_ok=True)
    r = client.get("/api/logs")
    assert r.status_code == 200
    body = r.json()
    assert body["content"] is None
    assert body["logs"] == []
    assert body["dir"] == str(settings.logs_dir())


def test_tail_returns_last_n_lines(client, settings):
    d = settings.logs_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / "service.log").write_text("\n".join(f"line {i}" for i in range(10)) + "\n", encoding="utf-8")

    r = client.get("/api/logs?name=service.log&lines=3")
    assert r.status_code == 200
    body = r.json()
    assert body["content"] == ["line 7", "line 8", "line 9"]
    assert [lf["name"] for lf in body["logs"]] == ["service.log"]
    assert body["logs"][0]["bytes"] > 0


def test_list_shows_all_log_files(client, settings):
    d = settings.logs_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / "service.log").write_text("s\n", encoding="utf-8")
    (d / "acme_TinyLM-main.log").write_text("w\n", encoding="utf-8")
    (d / "not-a-log.txt").write_text("x\n", encoding="utf-8")  # only *.log are listed

    r = client.get("/api/logs")
    names = [lf["name"] for lf in r.json()["logs"]]
    assert names == ["acme_TinyLM-main.log", "service.log"]


def test_traversal_rejected(client, settings):
    settings.logs_dir().mkdir(parents=True, exist_ok=True)
    assert client.get("/api/logs?name=../etc/passwd").status_code == 400
    assert client.get("/api/logs?name=..%2F..%2Fsecret.log").status_code in (400, 404)
    assert client.get("/api/logs?name=%2e%2e%2fpasswd.log").status_code in (400, 404)
    assert client.get("/api/logs?name=sub%2Finner.log").status_code == 400
    assert client.get("/api/logs?name=.hidden.log").status_code == 400
    assert client.get("/api/logs?name=README.md").status_code == 400


def test_missing_log_404(client, settings):
    settings.logs_dir().mkdir(parents=True, exist_ok=True)
    r = client.get("/api/logs?name=nope.log")
    assert r.status_code == 404
    assert "not found" in r.json()["error"]


def test_lines_capped_at_1000(client, settings):
    d = settings.logs_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / "big.log").write_text("\n".join(f"x{i}" for i in range(1500)) + "\n", encoding="utf-8")
    r = client.get("/api/logs?name=big.log&lines=999999")
    assert r.status_code == 200
    assert len(r.json()["content"]) == 1000
    assert r.json()["content"][-1] == "x1499"


def test_tail_large_file_is_accurate(client, settings):
    # >64 KiB so the tail scanner takes multiple backward blocks
    d = settings.logs_dir()
    d.mkdir(parents=True, exist_ok=True)
    lines = [f"pad {i:05d} " + "z" * 60 for i in range(3000)]
    (d / "big2.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    r = client.get("/api/logs?name=big2.log&lines=5")
    content = r.json()["content"]
    assert len(content) == 5
    assert content[0] == lines[-5]
    assert content[-1] == lines[-1]
