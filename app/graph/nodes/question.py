"""Question 노드 스텁 — Step 8. 실제 human-in-the-loop interrupt는 Step 16에서 구현."""
from __future__ import annotations
import logging
from app.graph.state import GraphState

logger = logging.getLogger(__name__)

def question_node(state: GraphState) -> GraphState:
    """정보 부족 항목에 대해 사용자에게 질문한다 (스텁 — 빈 답변으로 통과)."""
    plans = state.get("item_plans", [])
    idx = state.get("current_item_index", 0)
    if idx < len(plans):
        q = plans[idx].question_text or "추가 정보를 입력해주세요."
        logger.debug("[Question] item_id=%s 질문: %s (stub, interrupt 없이 통과)", plans[idx].item_id, q)
    return {"user_answer": "[스텁 답변]", "pending_question": ""}
