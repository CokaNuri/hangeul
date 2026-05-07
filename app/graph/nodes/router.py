"""Router 노드 — Step 13 실제 구현.

담당 역할:
  - 대화 히스토리에서 최신 사용자 메시지를 추출한다.
  - Solar Mini로 인텐트를 분류하고 confidence를 평가한다.
  - confidence < 0.6이면 DISAMBIGUATION으로 분기한다.
  - rewrite_item / change_tone 인텐트에서 target_item을 추출해 current_item_id에 저장한다.
  - LLM 실패 또는 JSON 파싱 실패 시 GENERAL_QA로 소프트 실패한다.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from app.config import settings
from app.graph.state import GraphState
from app.llm.solar_client import client, load_prompt, SolarAPIError
from app.models import Intent

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD = 0.6
_VALID_INTENTS = frozenset(i.value for i in Intent)

_DISAMBIGUATION_MESSAGE = (
    "죄송합니다, 요청을 정확히 이해하지 못했습니다. 아래 중 원하시는 작업을 선택해 주세요:\n\n"
    "1. 양식 채우기 시작\n"
    "2. 특정 항목 재작성\n"
    "3. 문체·톤 변경\n"
    "4. 자료 추가\n"
    "5. 기타 질문"
)


# ──────────────────────────────────────────────
# 노드 진입점
# ──────────────────────────────────────────────

def router_node(state: GraphState) -> GraphState:
    """사용자 메시지의 인텐트를 Solar Mini로 분류한다."""
    history = state.get("conversation_history") or []
    user_msgs = [m for m in history if m.get("role") == "user"]

    if not user_msgs:
        logger.warning("[Router] 대화 히스토리에 사용자 메시지 없음. 현재 인텐트 유지.")
        return {}

    user_message = user_msgs[-1]["content"]

    try:
        result = _classify(user_message)
    except SolarAPIError as exc:
        logger.error("[Router] LLM 분류 실패, general_qa로 fallback: %s", exc)
        return {"current_intent": Intent.GENERAL_QA.value}

    logger.info(
        "[Router] intent=%s confidence=%.2f target=%s",
        result.intent, result.confidence, result.target_item,
    )

    updates: dict = {"current_intent": result.intent}

    if result.target_item:
        updates["current_item_id"] = result.target_item

    if result.intent == Intent.DISAMBIGUATION.value:
        updates["pending_question"] = _DISAMBIGUATION_MESSAGE

    return updates


# ──────────────────────────────────────────────
# 분류 로직
# ──────────────────────────────────────────────

@dataclass
class _RouterResult:
    intent: str
    confidence: float
    target_item: str | None


def _classify(user_message: str) -> _RouterResult:
    """Solar Mini로 인텐트를 분류하고 _RouterResult를 반환한다."""
    prompt = load_prompt("router", user_message=user_message)
    raw = client.call(
        messages=[{"role": "user", "content": prompt}],
        model=settings.solar_mini_model,
        temperature=0.0,
        max_tokens=80,
    )
    return _parse_response(raw)


def _parse_response(raw: str) -> _RouterResult:
    """LLM 응답을 파싱해 _RouterResult를 반환한다.

    JSON 파싱 실패 시 키워드 기반 fallback으로 소프트 실패한다.
    """
    data = _extract_json(raw)
    if data is not None:
        return _build_result(data)

    logger.warning("[Router] JSON 파싱 실패, 키워드 fallback 사용. raw=%r", raw[:100])
    return _keyword_fallback(raw)


def _build_result(data: dict) -> _RouterResult:
    """파싱된 dict에서 _RouterResult를 생성한다."""
    raw_intent = str(data.get("intent", "")).strip()
    intent = raw_intent if raw_intent in _VALID_INTENTS else Intent.GENERAL_QA.value

    try:
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    # "null" 문자열도 None으로 처리
    raw_target = data.get("target_item")
    target_item: str | None = None
    if raw_target is not None and str(raw_target).lower() not in ("null", "none", ""):
        target_item = str(raw_target).strip()

    # confidence 임계값 미달 → disambiguation
    if confidence < _CONFIDENCE_THRESHOLD:
        intent = Intent.DISAMBIGUATION.value

    return _RouterResult(intent=intent, confidence=confidence, target_item=target_item)


def _extract_json(text: str) -> dict | None:
    """텍스트에서 JSON 객체를 추출한다.

    우선순위:
      1. 전체 텍스트가 JSON인 경우
      2. 마크다운 코드 블록 안의 JSON
      3. 텍스트 내 첫 번째 {...} 블록
    """
    stripped = text.strip()

    # 1. 전체가 JSON
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 2. 마크다운 코드 블록
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. 첫 번째 {...} 블록
    match = re.search(r"\{[^{}]*\}", stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def _keyword_fallback(text: str) -> _RouterResult:
    """JSON 파싱이 완전히 실패했을 때 키워드로 인텐트를 추정한다."""
    lower = text.lower()
    if any(k in lower for k in ("start_fill", "채우기", "작성 시작")):
        return _RouterResult(Intent.START_FILL.value, 0.6, None)
    if any(k in lower for k in ("rewrite_item", "다시 써", "수정")):
        return _RouterResult(Intent.REWRITE_ITEM.value, 0.6, None)
    if any(k in lower for k in ("add_material", "자료 추가")):
        return _RouterResult(Intent.ADD_MATERIAL.value, 0.6, None)
    if any(k in lower for k in ("change_tone", "문체", "톤")):
        return _RouterResult(Intent.CHANGE_TONE.value, 0.6, None)
    if any(k in lower for k in ("upload_form", "양식 업로드")):
        return _RouterResult(Intent.UPLOAD_FORM.value, 0.6, None)
    if any(k in lower for k in ("upload_material", "자료 업로드")):
        return _RouterResult(Intent.UPLOAD_MATERIAL.value, 0.6, None)
    return _RouterResult(Intent.GENERAL_QA.value, 0.5, None)
