"""In-memory 세션 스토어.

보안 원칙:
- 모든 데이터는 메모리에만 존재 (디스크 영속 X)
- TTL 만료 시 즉시 GC
- 세션 종료 / 프로세스 종료 시 자동 삭제
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.models import FormDoc, MaterialBundle, ItemPlan, DraftItem


@dataclass
class Session:
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    # 양식 / 자료 (세션 동안 전체본 유지)
    form_doc: FormDoc | None = None
    material_bundle: MaterialBundle | None = None

    # 계획 · 초안 (누적)
    item_plans: list[ItemPlan] = field(default_factory=list)
    drafts: dict[str, DraftItem] = field(default_factory=dict)  # item_id → DraftItem

    # 대화 히스토리 (슬라이딩 윈도우, 최대 HISTORY_WINDOW_SIZE 턴)
    history: list[dict[str, str]] = field(default_factory=list)

    # LangGraph 체크포인트 (그래프 재개용)
    graph_state: dict[str, Any] = field(default_factory=dict)

    # interrupt 상태 (Step 16)
    is_interrupted: bool = False        # True이면 다음 메시지를 Command(resume=...) 로 처리
    pending_question: str = ""          # interrupt가 전달한 질문 텍스트

    # 업로드된 원본 파일명 목록 (참고용)
    uploaded_files: list[str] = field(default_factory=list)

    def touch(self) -> None:
        self.last_accessed = time.time()

    def is_expired(self) -> bool:
        return (time.time() - self.last_accessed) > settings.session_ttl_seconds

    def add_message(self, role: str, content: str) -> None:
        """대화 히스토리에 메시지 추가. 슬라이딩 윈도우 적용."""
        self.history.append({"role": role, "content": content})
        window = settings.history_window_size
        if len(self.history) > window:
            self.history = self.history[-window:]

    def clear(self) -> None:
        """세션 데이터 즉시 삭제 (GC 보조)."""
        self.form_doc = None
        self.material_bundle = None
        self.item_plans.clear()
        self.drafts.clear()
        self.history.clear()
        self.graph_state.clear()
        self.uploaded_files.clear()
        self.is_interrupted = False
        self.pending_question = ""


class SessionStore:
    def __init__(self) -> None:
        self._store: dict[str, Session] = {}

    def create(self) -> Session:
        session_id = str(uuid.uuid4())
        session = Session(session_id=session_id)
        self._store[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        session = self._store.get(session_id)
        if session is None:
            return None
        if session.is_expired():
            self.delete(session_id)
            return None
        session.touch()
        return session

    def delete(self, session_id: str) -> None:
        session = self._store.pop(session_id, None)
        if session is not None:
            session.clear()

    def purge_expired(self) -> int:
        """만료된 세션을 모두 삭제하고 삭제 수를 반환."""
        expired = [sid for sid, s in self._store.items() if s.is_expired()]
        for sid in expired:
            self.delete(sid)
        return len(expired)

    def __len__(self) -> int:
        return len(self._store)


# 프로세스 전역 싱글턴
store = SessionStore()
