"""SessionStore 단위 테스트."""

import time

import pytest

from app.session_store import SessionStore


@pytest.fixture
def s() -> SessionStore:
    return SessionStore()


def test_create_and_get(s):
    session = s.create()
    assert s.get(session.session_id) is not None


def test_get_missing(s):
    assert s.get("nonexistent-id") is None


def test_delete(s):
    session = s.create()
    s.delete(session.session_id)
    assert s.get(session.session_id) is None


def test_delete_clears_data(s):
    session = s.create()
    session.add_message("user", "안녕")
    sid = session.session_id
    s.delete(sid)
    assert session.history == []


def test_expired_session_returns_none(monkeypatch, s):
    session = s.create()
    # last_accessed를 TTL보다 오래 전으로 조작
    session.last_accessed = time.time() - 9999
    assert s.get(session.session_id) is None


def test_purge_expired(monkeypatch, s):
    s1 = s.create()
    s2 = s.create()
    s1.last_accessed = time.time() - 9999  # 만료
    removed = s.purge_expired()
    assert removed == 1
    assert s.get(s2.session_id) is not None


def test_history_sliding_window(s):
    from app.config import settings
    session = s.create()
    window = settings.history_window_size  # 기본 10
    for i in range(window + 3):
        session.add_message("user", f"msg {i}")
    assert len(session.history) == window
    # 가장 오래된 메시지가 제거됐는지 확인
    assert session.history[0]["content"] == f"msg {3}"
