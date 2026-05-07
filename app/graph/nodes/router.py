"""Router 노드 스텁 — Step 8. 실제 LLM 분류는 Step 13에서 구현."""
from __future__ import annotations
import logging
from app.graph.state import GraphState
from app.models import Intent

logger = logging.getLogger(__name__)

def router_node(state: GraphState) -> GraphState:
    """사용자 메시지의 인텐트를 분류한다 (현재는 스텁 — start_fill 반환)."""
    logger.debug("[Router] intent=%s (stub)", state.get("current_intent"))
    return {"current_intent": state.get("current_intent", Intent.GENERAL_QA.value)}
