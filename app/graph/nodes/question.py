"""Question 노드 — Step 16 실제 구현.

LangGraph interrupt()를 사용해 사용자 입력을 기다린다.
  - graph.invoke() 첫 호출 시 interrupt(question) → 실행 일시정지, result에 __interrupt__ 포함
  - API가 Command(resume=answer)로 재개하면 interrupt()가 answer를 반환 → 이후 로직 진행
"""
from __future__ import annotations

import logging

from langgraph.types import interrupt

from app.cesp import emit, CATEGORY_INPUT_REQUIRED
from app.graph.state import GraphState

logger = logging.getLogger(__name__)


def question_node(state: GraphState) -> GraphState:
    """현재 항목에 대한 질문을 interrupt로 전송하고 사용자 답변을 기다린다."""
    plans = state.get("item_plans", [])
    idx = state.get("current_item_index", 0)
    if idx >= len(plans):
        logger.warning("[Question] current_item_index=%d >= len(plans)=%d", idx, len(plans))
        return {}

    plan = plans[idx]

    # label 조회 (form_doc에서)
    item_label = plan.item_id
    form_doc = state.get("form_doc")
    if form_doc:
        meta = next((it for it in form_doc.items if it.item_id == plan.item_id), None)
        if meta:
            item_label = meta.label

    question_text = (
        plan.question_text
        or f"'{item_label}' 항목을 작성하려면 추가 정보가 필요합니다. 관련 내용을 알려주세요."
    )

    logger.info("[Question] item_id=%s 질문 전송, interrupt 대기", plan.item_id)
    emit(CATEGORY_INPUT_REQUIRED)

    # ── 여기서 실행이 일시정지된다 ──────────────────
    user_answer = interrupt(question_text)
    # ── Command(resume=answer) 호출 시 여기서 재개 ──

    logger.info("[Question] item_id=%s 답변 수신 (%d자)", plan.item_id, len(user_answer or ""))
    return {
        "user_answer": str(user_answer).strip(),
        "pending_question": "",
    }
