"""Planner 노드 스텁 — Step 8. 실제 LLM 계획은 Step 14에서 구현."""
from __future__ import annotations
import logging
from app.graph.state import GraphState
from app.models import ItemPlan

logger = logging.getLogger(__name__)

def planner_node(state: GraphState) -> GraphState:
    """FormDoc 항목별 ItemPlan을 생성한다 (스텁 — 더미 플랜 반환)."""
    form_doc = state.get("form_doc")
    if form_doc is None:
        logger.debug("[Planner] form_doc 없음, 빈 계획 반환 (stub)")
        return {"item_plans": [], "current_item_index": 0}

    # 스텁: 각 FormItem을 단순 ItemPlan으로 변환
    plans = [
        ItemPlan(
            item_id=item.item_id,
            confidence=0.0,
            needs_question=False,
        )
        for item in form_doc.items
    ]
    logger.debug("[Planner] %d개 항목 계획 생성 (stub)", len(plans))
    return {"item_plans": plans, "current_item_index": 0}
