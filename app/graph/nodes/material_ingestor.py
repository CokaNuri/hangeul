"""MaterialIngestor 노드 — Step 11 실제 구현.

담당 역할:
  - state["material_bundle"]의 각 MaterialDoc에 LLM 요약을 추가한다.
  - 긴 문서는 청킹 후 요약을 취합한다.
  - material_bundle이 없거나 비어 있으면 경고만 남기고 통과한다.

PII 안전망:
  - MaterialDoc.masked_text는 파서 단계에서 이미 PII 마스킹이 완료된 상태다.
  - 이 노드는 masked_text만 LLM에 전달한다.
"""
from __future__ import annotations

import logging

from app.graph.state import GraphState
from app.llm.solar_client import client, load_prompt, SolarAPIError
from app.models import MaterialBundle, MaterialDoc
from app.config import settings

logger = logging.getLogger(__name__)

# 청킹 설정 — Solar Mini 컨텍스트 여유분을 남기고 안전하게 자름
_MAX_CHUNK_CHARS = 6_000
_CHUNK_OVERLAP = 200


def material_ingestor_node(state: GraphState) -> GraphState:
    """material_bundle의 각 문서에 Solar Mini 요약을 채운다."""
    bundle: MaterialBundle | None = state.get("material_bundle")

    if bundle is None or not bundle.docs:
        logger.warning("[MaterialIngestor] material_bundle 없음 또는 비어있음. 건너뜁니다.")
        return {}

    enriched: list[MaterialDoc] = []
    for doc in bundle.docs:
        if doc.summary:
            # 이미 요약이 있으면 그대로 유지
            enriched.append(doc)
            continue

        if not doc.masked_text.strip():
            logger.info("[MaterialIngestor] '%s' 텍스트 없음. 요약 생략.", doc.source_name)
            enriched.append(doc)
            continue

        try:
            summary = _summarize(doc.masked_text, doc.source_name)
            logger.info("[MaterialIngestor] '%s' 요약 완료 (%d자)", doc.source_name, len(summary))
        except SolarAPIError as exc:
            logger.error("[MaterialIngestor] '%s' 요약 실패: %s", doc.source_name, exc)
            summary = ""

        enriched.append(MaterialDoc(
            source_name=doc.source_name,
            masked_text=doc.masked_text,
            summary=summary,
            doc_type=doc.doc_type,
        ))

    return {"material_bundle": MaterialBundle(docs=enriched)}


# ──────────────────────────────────────────────
# 내부 함수
# ──────────────────────────────────────────────

def _summarize(masked_text: str, source_name: str) -> str:
    """청킹 + Solar Mini로 문서를 요약한다."""
    chunks = _chunk_text(masked_text)

    if len(chunks) == 1:
        return _call_summarizer(chunks[0], source_name)

    # 청크가 여러 개일 때: 각 청크를 요약한 뒤 취합 요약
    partial_summaries: list[str] = []
    for i, chunk in enumerate(chunks):
        chunk_name = f"{source_name} (파트 {i + 1}/{len(chunks)})"
        try:
            partial = _call_summarizer(chunk, chunk_name)
            if partial:
                partial_summaries.append(partial)
        except SolarAPIError:
            logger.warning("[MaterialIngestor] 청크 %d 요약 실패, 건너뜀", i + 1)

    if not partial_summaries:
        return ""

    if len(partial_summaries) == 1:
        return partial_summaries[0]

    # 부분 요약을 합쳐 최종 요약 생성
    combined = "\n".join(f"[파트 {i + 1}] {s}" for i, s in enumerate(partial_summaries))
    return _call_summarizer(combined, f"{source_name} (종합)")


def _call_summarizer(text: str, source_name: str) -> str:
    """summarizer 프롬프트로 Solar Mini를 호출한다."""
    prompt = load_prompt("summarizer", source_name=source_name, text=text)
    return client.call(
        messages=[{"role": "user", "content": prompt}],
        model=settings.solar_mini_model,
        temperature=0.2,
        max_tokens=512,
    )


def _chunk_text(text: str) -> list[str]:
    """텍스트를 _MAX_CHUNK_CHARS 단위로 오버랩 청킹한다."""
    if len(text) <= _MAX_CHUNK_CHARS:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + _MAX_CHUNK_CHARS
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - _CHUNK_OVERLAP  # 오버랩: 앞 청크 끝부분을 다음 청크에 포함

    return chunks
