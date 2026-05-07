"""FormParser 노드 스텁 — Step 8. 실제 파싱 로직은 Step 10에서 구현."""
from __future__ import annotations
import logging
from app.graph.state import GraphState

logger = logging.getLogger(__name__)

def form_parser_node(state: GraphState) -> GraphState:
    """세션에 저장된 form_doc을 확인한다 (스텁 — 현재 상태 그대로 통과)."""
    logger.debug("[FormParser] form_doc=%s (stub)", "있음" if state.get("form_doc") else "없음")
    return {}
