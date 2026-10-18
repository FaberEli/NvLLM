"""CORS: browser clients (the index.html demo panel, web apps) may call the API cross-origin."""


def test_response_allows_any_origin(client):
    r = client.get("/api/tags", headers={"Origin": "http://demo.example.com"})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "*"


def test_preflight_allows_post_chat(client):
    r = client.options(
        "/api/chat",
        headers={
            "Origin": "http://demo.example.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "*"
    assert "POST" in r.headers["access-control-allow-methods"]
