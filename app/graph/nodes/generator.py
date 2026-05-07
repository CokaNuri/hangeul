"""Generator 노드 스텁 — Step 8. 실제 LLM 초안 생성은 Step 15에서 구현."""
from __future__ import annotations
import logging
from app.graph.state import GraphState
from app.models import DraftItem

logger = logging.getLogger(__name__)

def generator_node(state: GraphState) -> GraphState:
    """현재 항목의 초안을 생성한다 (스텁 — 더미 초안 반환)."""
    plans = state.get("item_plans", [])
    idx = state.get("current_item_index", 0)
    if idx >= len(plans):
        return {}

    plan = plans[idx]
    drafts = dict(state.get("drafts", {}))
    drafts[plan.item_id] = DraftItem(
        item_id=plan.item_id,
        text=f"[스텁 초안] {plan.item_id}",
        citations=[],
    )
    logger.debug("[Generator] item_id=%s 스텁 초안 생성", plan.item_id)
    return {"drafts": drafts}
