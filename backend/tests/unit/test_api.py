from __future__ import annotations

from fastapi.testclient import TestClient

from clipforge.api.app import create_app
from tests.conftest import make_settings, register


def test_health_and_security_headers(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["x-request-id"]


def test_public_config(client):
    cfg = client.get("/api/v1/config").json()
    assert cfg["aspect_ratios"] == ["9:16", "16:9", "1:1"]
    assert {p["id"] for p in cfg["caption_presets"]} >= {"clean", "bold", "creator", "minimal", "high_contrast"}


def test_worker_health_offline(client):
    r = client.get("/api/v1/health/worker")
    assert r.status_code == 503 and r.json()["error"]["code"] == "WORKER_OFFLINE"


def test_auth_flow(client):
    assert client.get("/api/v1/me").status_code == 401
    body = register(client)
    assert body["user"]["email"] == "user@clipforge.dev"
    assert client.get("/api/v1/me").status_code == 200
    # duplicate registration
    r = client.post("/api/v1/auth/register", json={"email": "USER@clipforge.dev", "password": "another-pass-1"})
    assert r.status_code == 409
    assert client.post("/api/v1/auth/logout").status_code == 204
    assert client.get("/api/v1/me").status_code == 401
    r = client.post("/api/v1/auth/login", json={"email": "user@clipforge.dev", "password": "wrong-password"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "UNAUTHORIZED"
    r = client.post("/api/v1/auth/login", json={"email": "user@clipforge.dev", "password": "correct-horse-1"})
    assert r.status_code == 200


def test_csrf_required_for_cookie_mutations(client):
    register(client)
    del client.headers["X-CSRF-Token"]
    r = client.patch("/api/v1/me/settings", json={"default_num_clips": 3})
    assert r.status_code == 403
    token = client.cookies.get("cf_csrf")
    r = client.patch("/api/v1/me/settings", json={"default_num_clips": 3}, headers={"X-CSRF-Token": token})
    assert r.status_code == 200 and r.json()["default_num_clips"] == 3


def test_bearer_tokens_skip_csrf(app):
    with TestClient(app) as c:
        token = register(c)["token"]
    with TestClient(app) as c2:
        r = c2.patch("/api/v1/me/settings", json={"default_num_clips": 4}, headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200


def test_user_scoping(app):
    with TestClient(app) as a, TestClient(app) as b:
        register(a, "a@clipforge.dev")
        register(b, "b@clipforge.dev")
        init = a.post("/api/v1/uploads/init", json={"filename": "x.mp4", "size_bytes": 1000}).json()
        assert a.get(f"/api/v1/uploads/{init['id']}").status_code == 200
        assert b.get(f"/api/v1/uploads/{init['id']}").status_code == 404
        assert b.post("/api/v1/jobs", json={"upload_id": init["id"]}).status_code == 404


def test_admin_routes_require_admin(app, settings):
    with TestClient(app) as owner, TestClient(app) as c:
        assert register(owner, "owner@clipforge.dev")["user"]["role"] == "admin"  # first local user owns the install
        assert register(c)["user"]["role"] == "user"
        assert c.get("/api/v1/admin/system").status_code == 403


def test_admin_system(tmp_path):
    s = make_settings(tmp_path / "d", ADMIN_EMAILS=["boss@clipforge.dev"])
    with TestClient(create_app(s)) as c:
        register(c, "boss@clipforge.dev")
        r = c.get("/api/v1/admin/system")
        assert r.status_code == 200
        assert "metrics" in r.json() and "disk" in r.json()
        assert c.get("/api/v1/admin/users").json()["total"] == 1


def test_upload_validation(client):
    register(client)
    r = client.post("/api/v1/uploads/init", json={"filename": "malware.exe", "size_bytes": 10})
    assert r.status_code == 415 and r.json()["error"]["code"] == "UNSUPPORTED_FORMAT"
    r = client.post("/api/v1/uploads/init", json={"filename": "a.mp4", "size_bytes": 10**13})
    assert r.status_code == 413 and r.json()["error"]["code"] == "MEDIA_TOO_LARGE"


def test_validation_error_shape(client):
    register(client)
    r = client.post("/api/v1/jobs", json={"upload_id": "x", "settings": {"aspect_ratio": "4:3"}})
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "VALIDATION_ERROR" and err["details"]["fields"]


def test_local_mode_without_auth(tmp_path):
    s = make_settings(tmp_path / "d", AUTH_ENABLED=False)
    with TestClient(create_app(s)) as c:
        r = c.get("/api/v1/me")
        assert r.status_code == 200 and r.json()["user"]["role"] == "admin"


def test_quota_exceeded(tmp_path):
    from clipforge.db.models import Asset, User
    from clipforge.db.session import session_scope

    s = make_settings(tmp_path / "d", DEPLOYMENT_MODE="cloud", FREE_DAILY_MINUTES=1)
    app = create_app(s)
    with TestClient(app) as c:
        register(c)
        init = c.post("/api/v1/uploads/init", json={"filename": "x.mp4", "size_bytes": 1000}).json()
        with session_scope(app.state.session_factory) as db:
            a = db.get(Asset, init["id"])
            a.status, a.duration_seconds = "READY", 600.0
            app.state.storage.put_bytes(a.storage_key, b"x")
            assert db.get(User, a.user_id).plan == "free"
        r = c.post("/api/v1/jobs", json={"upload_id": init["id"]})
        assert r.status_code == 402 and r.json()["error"]["code"] == "QUOTA_EXCEEDED"


def test_rate_limit_on_login(tmp_path):
    s = make_settings(tmp_path / "d", RATE_LIMIT_ENABLED=True, RATE_LIMIT_AUTH_PER_MINUTE=3)
    with TestClient(create_app(s)) as c:
        codes = [c.post("/api/v1/auth/login", json={"email": "n@clipforge.dev", "password": "x"}).status_code
                 for _ in range(5)]
        assert codes[:3] == [401, 401, 401] and codes[-1] == 429


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200 and "clipforge_" in r.text
