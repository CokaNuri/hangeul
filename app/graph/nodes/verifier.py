"""Verifier 노드 스텁 — Step 8. 실제 LLM 검증은 Step 15에서 구현."""
from __future__ import annotations
import logging
from app.graph.state import GraphState

logger = logging.getLogger(__name__)

VERIFIER_RESULT_KEY = "__verifier_verdict__"

def verifier_node(state: GraphState) -> GraphState:
    """초안의 품질을 검증한다 (스텁 — 항상 ok 반환)."""
    plans = state.get("item_plans", [])
    idx = state.get("current_item_index", 0)
    item_id = plans[idx].item_id if idx < len(plans) else "unknown"
    logger.debug("[Verifier] item_id=%s → ok (stub)", item_id)
    # 라우팅에 사용할 verdict를 errors 필드에 임시 저장
    # Step 15에서 전용 필드로 교체 예정
    return {VERIFIER_RESULT_KEY: "ok"}
