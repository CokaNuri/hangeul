"""FastAPI 라우터 — 스켈레톤 엔드포인트 (Step 2).

각 엔드포인트는 Step 8~18에서 실제 LangGraph 로직으로 교체된다.
지금은 세션 관리와 요청/응답 스키마만 확정한다.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from fastapi.responses import Response
from pydantic import BaseModel

from app.graph.graph import graph
from app.graph.state import initial_state
from app.models import Intent
from app.session_store import store, Session

router = APIRouter()


# ──────────────────────────────────────────────
# 의존성
# ──────────────────────────────────────────────

def get_session(session_id: str) -> Session:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="세션을 찾을 수 없습니다.")
    return session


# ──────────────────────────────────────────────
# 스키마
# ──────────────────────────────────────────────

class SessionResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    intent: str = "unknown"


class UploadResponse(BaseModel):
    session_id: str
    file_name: str
    file_type: str          # "form" | "material"
    message: str


# ──────────────────────────────────────────────
# 엔드포인트
# ──────────────────────────────────────────────

@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "active_sessions": len(store)}


@router.post("/api/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session() -> SessionResponse:
    """새 세션을 생성하고 session_id를 반환한다."""
    session = store.create()
    return SessionResponse(session_id=session.session_id)


@router.delete("/api/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str) -> None:
    """세션을 명시적으로 종료하고 데이터를 즉시 삭제한다."""
    store.delete(session_id)


@router.post("/api/upload", response_model=UploadResponse)
async def upload_file(
    session_id: str,
    file_type: str,          # "form" | "material"
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> UploadResponse:
    """양식(.hwpx) 또는 자료 파일을 업로드한다.

    Step 3, 6에서 실제 파서 연동으로 교체 예정.
    현재는 파일 수신만 확인하고 파일명을 세션에 기록한다.
    """
    if file_type not in ("form", "material"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="file_type은 'form' 또는 'material'이어야 합니다.")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="빈 파일은 업로드할 수 없습니다.")

    session.uploaded_files.append(file.filename or "unknown")

    # TODO(Step 3): form → HwpxParser 호출
    # TODO(Step 6): material → MaterialIngestor + PII Masker 호출

    return UploadResponse(
        session_id=session_id,
        file_name=file.filename or "unknown",
        file_type=file_type,
        message=f"[스텁] '{file.filename}' 수신 완료 ({len(content):,} bytes). 파서 연동은 Step 3/6에서 구현 예정.",
    )


@router.post("/api/chat", response_model=ChatResponse)
async def chat(body: ChatRequest) -> ChatResponse:
    """사용자 메시지를 받아 LangGraph 그래프를 실행하고 응답을 반환한다.

    TODO(Step 18): SSE 스트리밍으로 교체 예정.
    """
    session = get_session(body.session_id)
    session.add_message("user", body.message)

    # ── 인텐트 파싱 (스텁: "채우기 시작" 키워드만 인식, Step 13에서 LLM 분류로 교체)
    intent = _parse_intent_stub(body.message)

    # ── LangGraph 그래프 실행
    graph_state = dict(session.graph_state) if session.graph_state else initial_state()
    graph_state["current_intent"] = intent
    graph_state["form_doc"] = session.form_doc
    graph_state["material_bundle"] = session.material_bundle
    graph_state["conversation_history"] = list(session.history)

    config = {"configurable": {"thread_id": body.session_id}}
    result = graph.invoke(graph_state, config=config)

    # 세션에 그래프 상태 저장
    session.graph_state = dict(result)
    session.item_plans = result.get("item_plans", [])
    session.drafts = result.get("drafts", {})

    approved = result.get("approved_items", [])
    reply = _build_reply(intent, approved, result)
    session.add_message("assistant", reply)

    return ChatResponse(
        session_id=body.session_id,
        reply=reply,
        intent=intent,
    )


def _parse_intent_stub(message: str) -> str:
    """키워드 기반 인텐트 스텁 (Step 13에서 LLM 분류로 교체)."""
    msg = message.strip()
    if any(k in msg for k in ("채우기 시작", "작성 시작", "시작")):
        return Intent.START_FILL.value
    if any(k in msg for k in ("자료 추가", "파일 추가")):
        return Intent.ADD_MATERIAL.value
    if any(k in msg for k in ("다시", "재작성", "수정")):
        return Intent.REWRITE_ITEM.value
    return Intent.GENERAL_QA.value


def _build_reply(intent: str, approved: list, result: dict) -> str:
    """그래프 실행 결과를 사용자 친화적 메시지로 변환한다 (스텁)."""
    if intent == Intent.START_FILL.value:
        total = len(result.get("item_plans", []))
        return (
            f"양식 채우기를 완료했습니다. "
            f"총 {total}개 항목 중 {len(approved)}개를 작성했습니다.\n\n"
            f"[스텁] Step 10~17 구현 후 실제 초안이 표시됩니다."
        )
    return f"[스텁] '{intent}' 인텐트로 처리했습니다."


@router.get("/api/download/{session_id}")
async def download(session_id: str) -> Response:
    """완성된 .hwpx 파일을 반환한다.

    Step 17에서 실제 렌더러 연동으로 교체 예정.
    현재는 더미 bytes를 반환한다.
    """
    session = get_session(session_id)

    # TODO(Step 17): Renderer가 생성한 hwpx bytes 반환
    dummy_bytes = b"HWPX_STUB"
    filename = "output_stub.hwpx"

    return Response(
        content=dummy_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
