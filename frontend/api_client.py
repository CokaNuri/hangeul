"""FastAPI 서버 HTTP 클라이언트 유틸 — Step 9.

Step 18에서 SSE 스트리밍으로 교체 예정.
"""

from __future__ import annotations

import httpx

API_BASE = "http://localhost:8000"
TIMEOUT = 60.0


def create_session() -> str:
    """새 세션을 생성하고 session_id를 반환한다."""
    resp = httpx.post(f"{API_BASE}/api/sessions", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()["session_id"]


def upload_file(session_id: str, file_bytes: bytes, filename: str, file_type: str) -> dict:
    """파일을 서버에 업로드하고 응답 딕셔너리를 반환한다."""
    resp = httpx.post(
        f"{API_BASE}/api/upload",
        params={"session_id": session_id, "file_type": file_type},
        files={"file": (filename, file_bytes, "application/octet-stream")},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def send_chat(session_id: str, message: str) -> dict:
    """채팅 메시지를 보내고 응답 딕셔너리를 반환한다."""
    resp = httpx.post(
        f"{API_BASE}/api/chat",
        json={"session_id": session_id, "message": message},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def download_hwpx(session_id: str) -> bytes:
    """완성된 .hwpx 파일 bytes를 반환한다."""
    resp = httpx.get(f"{API_BASE}/api/download/{session_id}", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.content


def health_check() -> bool:
    """서버가 응답하는지 확인한다."""
    try:
        resp = httpx.get(f"{API_BASE}/health", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False
