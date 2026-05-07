"""FastAPI 라우트 단위 테스트 (TestClient 사용 — 서버 불필요)."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.fixtures.hwpx_factory import make_hwpx

client = TestClient(app)


# ── /health ───────────────────────────────────

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── /api/sessions ─────────────────────────────

def test_create_session():
    r = client.post("/api/sessions")
    assert r.status_code == 201
    data = r.json()
    assert "session_id" in data
    assert len(data["session_id"]) == 36  # UUID


def test_delete_session():
    sid = client.post("/api/sessions").json()["session_id"]
    r = client.delete(f"/api/sessions/{sid}")
    assert r.status_code == 204


def test_delete_nonexistent_session():
    r = client.delete("/api/sessions/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 204  # 없어도 204 (멱등)


# ── /api/upload ───────────────────────────────

@pytest.fixture
def sid():
    return client.post("/api/sessions").json()["session_id"]


def test_upload_form(sid):
    hwpx = make_hwpx(paragraphs=["연구 목표"])
    r = client.post(
        "/api/upload",
        params={"session_id": sid, "file_type": "form"},
        files={"file": ("form.hwpx", hwpx, "application/octet-stream")},
    )
    assert r.status_code == 200
    assert r.json()["file_type"] == "form"


def test_upload_material(sid):
    r = client.post(
        "/api/upload",
        params={"session_id": sid, "file_type": "material"},
        files={"file": ("cv.txt", b"my CV content", "text/plain")},
    )
    assert r.status_code == 200
    assert r.json()["file_type"] == "material"


def test_upload_invalid_file_type(sid):
    r = client.post(
        "/api/upload",
        params={"session_id": sid, "file_type": "unknown"},
        files={"file": ("file.txt", b"data", "text/plain")},
    )
    assert r.status_code == 400


def test_upload_empty_file(sid):
    r = client.post(
        "/api/upload",
        params={"session_id": sid, "file_type": "form"},
        files={"file": ("empty.hwpx", b"", "application/octet-stream")},
    )
    assert r.status_code == 400


def test_upload_invalid_session():
    r = client.post(
        "/api/upload",
        params={"session_id": "bad-id", "file_type": "form"},
        files={"file": ("f.hwpx", b"data", "application/octet-stream")},
    )
    assert r.status_code == 404


# ── /api/chat ─────────────────────────────────

def test_chat_general_qa(sid):
    r = client.post("/api/chat", json={"session_id": sid, "message": "안녕하세요"})
    assert r.status_code == 200
    data = r.json()
    assert "reply" in data
    assert "intent" in data


def test_chat_start_fill_intent(sid):
    r = client.post("/api/chat", json={"session_id": sid, "message": "채우기 시작"})
    assert r.status_code == 200
    assert r.json()["intent"] == "start_fill"


def test_chat_rewrite_intent(sid):
    r = client.post("/api/chat", json={"session_id": sid, "message": "3번 다시 써줘"})
    assert r.status_code == 200
    assert r.json()["intent"] == "rewrite_item"


def test_chat_invalid_session():
    r = client.post("/api/chat", json={"session_id": "bad-id", "message": "hi"})
    assert r.status_code == 404


# ── /api/download ─────────────────────────────

def test_download_returns_bytes(sid):
    r = client.get(f"/api/download/{sid}")
    assert r.status_code == 200
    assert len(r.content) > 0
    assert "attachment" in r.headers.get("content-disposition", "")


def test_download_invalid_session():
    r = client.get("/api/download/bad-session-id")
    assert r.status_code == 404
