"""MaterialIngestor 노드 스텁 — Step 8. 실제 파싱은 Step 11에서 구현."""
from __future__ import annotations
import logging
from app.graph.state import GraphState

logger = logging.getLogger(__name__)

def material_ingestor_node(state: GraphState) -> GraphState:
    """세션의 material_bundle을 확인한다 (스텁 — 현재 상태 그대로 통과)."""
    logger.debug("[MaterialIngestor] bundle=%s (stub)", "있음" if state.get("material_bundle") else "없음")
    return {}
