"""Renderer 노드 스텁 — Step 8. 실제 XML 삽입은 Step 17에서 구현."""
from __future__ import annotations
import logging
from app.graph.state import GraphState

logger = logging.getLogger(__name__)

def renderer_node(state: GraphState) -> GraphState:
    """승인된 초안을 FormDoc XML에 삽입한다 (스텁 — approved_items만 갱신)."""
    plans = state.get("item_plans", [])
    idx = state.get("current_item_index", 0)
    if idx >= len(plans):
        return {}

    item_id = plans[idx].item_id
    approved = list(state.get("approved_items", []))
    if item_id not in approved:
        approved.append(item_id)

    next_idx = idx + 1
    logger.debug("[Renderer] item_id=%s 승인 (stub). 다음 인덱스=%d", item_id, next_idx)
    return {"approved_items": approved, "current_item_index": next_idx}
