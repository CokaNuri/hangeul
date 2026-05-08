"""Generator 노드 — Step 15 실제 구현.

담당 역할:
  - 현재 항목(item_plans[current_item_index])의 초안을 Solar Pro로 생성한다.
  - 근거(source_evidence) + 사용자 답변(user_answer)을 프롬프트에 전달한다.
  - JSON 파싱 실패 시 LLM 응답 전체를 draft text로 사용(소프트 폴백).
"""
from __future__ import annotations

import json
import logging
import re

from app.config import settings
from app.graph.state import GraphState
from app.llm.solar_client import client, load_prompt, SolarAPIError
from app.models import DraftItem

logger = logging.getLogger(__name__)

_MAX_EVIDENCE_CHARS = 3000  # 프롬프트 내 근거 텍스트 최대 길이


# ──────────────────────────────────────────────
# 노드 진입점
# ──────────────────────────────────────────────

def generator_node(state: GraphState) -> GraphState:
    """현재 항목의 초안을 Solar Pro로 생성한다."""
    plans = state.get("item_plans", [])
    idx = state.get("current_item_index", 0)
    if idx >= len(plans):
        logger.warning("[Generator] current_item_index=%d >= len(plans)=%d", idx, len(plans))
        return {}

    plan = plans[idx]
    item_id = plan.item_id

    # FormDoc에서 항목 메타 조회
    form_doc = state.get("form_doc")
    item_meta = None
    if form_doc:
        item_meta = next((it for it in form_doc.items if it.item_id == item_id), None)

    item_label = item_meta.label if item_meta else item_id
    item_type = item_meta.item_type.value if item_meta else "text"
    char_hint = item_meta.char_hint if item_meta else 0
    tone_guide = state.get("target_tone") or "공식적"
    user_answer = (state.get("user_answer") or "").strip()

    evidence_text = _build_evidence_text(plan.source_evidence, user_answer)

    try:
        text, citations = _generate(
            item_id=item_id,
            item_label=item_label,
            item_type=item_type,
            char_hint=char_hint,
            tone_guide=tone_guide,
            evidence_text=evidence_text,
            user_answer=user_answer,
        )
    except SolarAPIError as exc:
        logger.error("[Generator] LLM 실패 item_id=%s: %s", item_id, exc)
        text = f"[생성 실패] {item_id} 항목을 생성하지 못했습니다."
        citations = []

    drafts = dict(state.get("drafts") or {})
    prev = drafts.get(item_id)
    drafts[item_id] = DraftItem(
        item_id=item_id,
        text=text,
        citations=citations,
        retry_count=prev.retry_count if prev else 0,
    )

    logger.info("[Generator] item_id=%s 초안 생성 완료 (%d자)", item_id, len(text))
    return {"drafts": drafts, "user_answer": ""}  # user_answer 소비


# ──────────────────────────────────────────────
# LLM 호출
# ──────────────────────────────────────────────

def _generate(
    item_id: str,
    item_label: str,
    item_type: str,
    char_hint: int,
    tone_guide: str,
    evidence_text: str,
    user_answer: str,
) -> tuple[str, list[str]]:
    """Solar Pro에 초안 생성을 요청하고 (text, citations)를 반환한다."""
    prompt = load_prompt(
        "generator",
        item_id=item_id,
        item_label=item_label,
        item_type=item_type,
        char_hint=char_hint,
        tone_guide=tone_guide,
        evidence_text=evidence_text or "(근거 자료 없음)",
        user_answer=user_answer or "",
    )
    raw = client.call(
        messages=[{"role": "user", "content": prompt}],
        model=settings.solar_pro_model,
        temperature=0.4,
        max_tokens=1024,
    )
    return _parse_response(raw)


def _parse_response(raw: str) -> tuple[str, list[str]]:
    """LLM 응답에서 {"text": ..., "citations": [...]} 를 파싱한다."""
    stripped = raw.strip()

    # 1. 직접 JSON 파싱
    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            return _extract_fields(data)
    except json.JSONDecodeError:
        pass

    # 2. 마크다운 코드 블록
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return _extract_fields(data)
        except json.JSONDecodeError:
            pass

    # 3. 첫 번째 {...} 블록
    start = stripped.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(stripped[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(stripped[start:i + 1])
                        if isinstance(data, dict):
                            return _extract_fields(data)
                    except json.JSONDecodeError:
                        break

    # 4. 소프트 폴백: 전체 텍스트를 초안으로 사용
    logger.warning("[Generator] JSON 파싱 실패, 전체 텍스트 사용. raw=%r", raw[:80])
    return stripped, []


def _extract_fields(data: dict) -> tuple[str, list[str]]:
    text = str(data.get("text") or "").strip()
    citations_raw = data.get("citations") or []
    citations = [str(c) for c in citations_raw if c] if isinstance(citations_raw, list) else []
    return text, citations


# ──────────────────────────────────────────────
# 근거 텍스트 빌더
# ──────────────────────────────────────────────

def _build_evidence_text(source_evidence: list[str], user_answer: str) -> str:
    """근거 스니펫 목록과 사용자 답변을 하나의 문자열로 합친다."""
    parts = [s.strip() for s in (source_evidence or []) if s and s.strip()]
    combined = "\n\n".join(parts)
    if len(combined) > _MAX_EVIDENCE_CHARS:
        combined = combined[:_MAX_EVIDENCE_CHARS] + "\n...(이하 생략)"
    if user_answer:
        prefix = f"{combined}\n\n" if combined else ""
        combined = f"{prefix}[사용자 직접 제공]\n{user_answer}"
    return combined
