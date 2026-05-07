"""FormParser 노드 — Step 10 실제 구현.

담당 역할:
  - state["form_doc"]의 raw bytes로 HWPX를 재파싱해 실제 빈칸·PII 항목을 식별한다.
  - 식별된 FormDoc을 state에 다시 저장한다.
  - form_doc이 없으면 에러 메시지를 추가하고 종료한다.
"""
from __future__ import annotations

import logging

from app.graph.state import GraphState
from app.models import ItemType
from app.parsers.hwpx_parser import parse_hwpx

logger = logging.getLogger(__name__)


def form_parser_node(state: GraphState) -> GraphState:
    """HWPX를 파싱해 빈칸·PII 항목을 실제로 식별한다."""
    form_doc = state.get("form_doc")

    if form_doc is None:
        logger.warning("[FormParser] form_doc이 없습니다. 양식 파일을 먼저 업로드하세요.")
        errors = list(state.get("errors") or [])
        errors.append("양식 파일이 없습니다. .hwpx 파일을 먼저 업로드해 주세요.")
        return {"errors": errors}

    hwpx_bytes: bytes | None = form_doc.metadata.get("hwpx_bytes")
    if not hwpx_bytes:
        logger.warning("[FormParser] form_doc에 hwpx_bytes가 없습니다. 파서 재실행을 건너뜁니다.")
        return {}

    try:
        parsed = parse_hwpx(hwpx_bytes)
    except ValueError as exc:
        logger.error("[FormParser] HWPX 파싱 실패: %s", exc)
        errors = list(state.get("errors") or [])
        errors.append(f"양식 파싱 실패: {exc}")
        return {"errors": errors}

    total = len(parsed.items)
    pii_count = sum(1 for it in parsed.items if it.item_type == ItemType.PII)
    text_count = total - pii_count

    logger.info(
        "[FormParser] 빈칸 %d개 식별 (일반 %d개, PII %d개), 표 %d개",
        total, text_count, pii_count, len(parsed.tables),
    )

    return {"form_doc": parsed}
