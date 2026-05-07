"""Planner 노드 — Step 14 실제 구현.

담당 역할:
  - FormDoc 항목 목록과 MaterialBundle 요약을 Solar Pro에 전달한다.
  - LLM 응답(JSON 배열)을 파싱해 ItemPlan 리스트를 반환한다.
  - PII 항목은 LLM 응답을 무시하고 항상 needs_question=False, source_evidence=[] 를 강제한다.
  - confidence < 0.5인 항목은 needs_question=True 를 강제한다.
  - LLM 실패 또는 JSON 파싱 실패 시 모든 항목에 소프트 실패 플랜을 반환한다.
"""
from __future__ import annotations

import json
import logging
import re

from app.config import settings
from app.graph.state import GraphState
from app.llm.solar_client import client, load_prompt, SolarAPIError
from app.models import ItemPlan, ItemType, MaterialBundle

logger = logging.getLogger(__name__)

_CONFIDENCE_QUESTION_THRESHOLD = 0.5
# 항목이 너무 많으면 LLM 컨텍스트가 초과될 수 있어 배치로 나눈다
_BATCH_SIZE = 20


# ──────────────────────────────────────────────
# 노드 진입점
# ──────────────────────────────────────────────

def planner_node(state: GraphState) -> GraphState:
    """FormDoc 항목 × MaterialBundle 요약 → ItemPlan 리스트 생성."""
    form_doc = state.get("form_doc")
    if form_doc is None:
        logger.warning("[Planner] form_doc 없음, 빈 계획 반환.")
        return {"item_plans": [], "current_item_index": 0}

    # PII 항목은 LLM 없이 즉시 처리
    pii_items = [item for item in form_doc.items if item.item_type == ItemType.PII]
    non_pii_items = [item for item in form_doc.items if item.item_type != ItemType.PII]

    pii_plans = [
        ItemPlan(item_id=item.item_id, source_evidence=[], confidence=1.0, needs_question=False)
        for item in pii_items
    ]

    if not non_pii_items:
        logger.info("[Planner] non-PII 항목 없음, PII 플랜만 반환.")
        return {"item_plans": pii_plans, "current_item_index": 0}

    materials_summary = _build_materials_summary(state.get("material_bundle"))

    # 배치 처리 (항목 수 ≤ _BATCH_SIZE면 단일 호출)
    llm_plans: list[ItemPlan] = []
    batches = [non_pii_items[i:i + _BATCH_SIZE] for i in range(0, len(non_pii_items), _BATCH_SIZE)]

    for batch in batches:
        try:
            batch_plans = _plan_batch(batch, materials_summary)
        except SolarAPIError as exc:
            logger.error("[Planner] LLM 실패, 소프트 실패 플랜 사용: %s", exc)
            batch_plans = _soft_fail_plans(batch)
        llm_plans.extend(batch_plans)

    all_plans = pii_plans + llm_plans

    logger.info(
        "[Planner] %d개 ItemPlan 생성 (PII=%d, LLM=%d)",
        len(all_plans), len(pii_plans), len(llm_plans),
    )
    return {"item_plans": all_plans, "current_item_index": 0}


# ──────────────────────────────────────────────
# LLM 호출
# ──────────────────────────────────────────────

def _plan_batch(items, materials_summary: str) -> list[ItemPlan]:
    """항목 배치에 대해 Solar Pro 호출 후 ItemPlan 리스트를 반환한다."""
    form_items_json = json.dumps(
        [
            {
                "item_id": item.item_id,
                "label": item.label,
                "type": item.item_type.value,
                "context": item.context[:200] if item.context else "",
            }
            for item in items
        ],
        ensure_ascii=False,
        indent=2,
    )

    prompt = load_prompt(
        "planner",
        form_items_json=form_items_json,
        materials_summary=materials_summary,
    )
    raw = client.call(
        messages=[{"role": "user", "content": prompt}],
        model=settings.solar_pro_model,
        temperature=0.0,
        max_tokens=2048,
    )

    return _parse_response(raw, items)


# ──────────────────────────────────────────────
# 응답 파싱
# ──────────────────────────────────────────────

def _parse_response(raw: str, items) -> list[ItemPlan]:
    """LLM JSON 배열 응답을 ItemPlan 리스트로 변환한다."""
    data = _extract_json_array(raw)
    if data is None:
        logger.warning("[Planner] JSON 배열 파싱 실패, 소프트 실패 플랜 사용. raw=%r", raw[:120])
        return _soft_fail_plans(items)

    # item_id → LLM 결과 맵
    plan_map: dict[str, dict] = {}
    for entry in data:
        if isinstance(entry, dict) and "item_id" in entry:
            plan_map[str(entry["item_id"])] = entry

    plans: list[ItemPlan] = []
    for item in items:
        entry = plan_map.get(item.item_id)
        if entry is None:
            logger.warning("[Planner] '%s' 항목 LLM 응답 없음, 소프트 실패.", item.item_id)
            plans.append(_make_soft_fail_plan(item.item_id))
            continue
        plans.append(_build_plan(entry, item))

    return plans


def _build_plan(entry: dict, item) -> ItemPlan:
    """LLM 응답 dict에서 ItemPlan을 생성하고 후처리 규칙을 적용한다."""
    try:
        confidence = float(entry.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.0

    source_evidence = []
    raw_evidence = entry.get("source_evidence") or []
    if isinstance(raw_evidence, list):
        source_evidence = [str(e) for e in raw_evidence if e]

    needs_question = bool(entry.get("needs_question", False))
    question_text = str(entry.get("question_text") or "").strip()

    # 규칙 강제
    if item.item_type == ItemType.PII:
        source_evidence = []
        needs_question = False
        question_text = ""
    elif confidence < _CONFIDENCE_QUESTION_THRESHOLD:
        needs_question = True
        if not question_text:
            question_text = f"'{item.label}' 항목을 작성하려면 추가 정보가 필요합니다. 관련 내용을 알려주세요."

    return ItemPlan(
        item_id=item.item_id,
        source_evidence=source_evidence,
        confidence=confidence,
        needs_question=needs_question,
        question_text=question_text,
    )


# ──────────────────────────────────────────────
# JSON 추출
# ──────────────────────────────────────────────

def _extract_json_array(text: str) -> list | None:
    """텍스트에서 JSON 배열을 추출한다.

    우선순위:
      1. 전체가 JSON 배열
      2. 마크다운 코드 블록 안의 JSON 배열
      3. 텍스트 내 첫 번째 [...] 블록
    """
    stripped = text.strip()

    # 1. 전체가 JSON 배열
    try:
        result = json.loads(stripped)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # 2. 마크다운 코드 블록
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", stripped, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # 3. 첫 번째 [...] 블록 (중첩 허용)
    start = stripped.find("[")
    if start != -1:
        depth = 0
        for i, ch in enumerate(stripped[start:], start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        result = json.loads(stripped[start:i + 1])
                        if isinstance(result, list):
                            return result
                    except json.JSONDecodeError:
                        break

    return None


# ──────────────────────────────────────────────
# 자료 요약 빌더
# ──────────────────────────────────────────────

def _build_materials_summary(bundle: MaterialBundle | None) -> str:
    """MaterialBundle의 요약을 하나의 문자열로 합친다."""
    if bundle is None or not bundle.docs:
        return "제공된 자료 없음."

    parts: list[str] = []
    for doc in bundle.docs:
        summary = doc.summary.strip() if doc.summary else ""
        if summary:
            parts.append(f"[{doc.source_name}]\n{summary}")
        else:
            # 요약이 없으면 앞 500자를 사용
            preview = doc.masked_text[:500].strip()
            if preview:
                parts.append(f"[{doc.source_name}] (요약 없음, 발췌)\n{preview}")

    return "\n\n".join(parts) if parts else "제공된 자료 없음."


# ──────────────────────────────────────────────
# 소프트 실패
# ──────────────────────────────────────────────

def _soft_fail_plans(items) -> list[ItemPlan]:
    return [_make_soft_fail_plan(item.item_id) for item in items]


def _make_soft_fail_plan(item_id: str) -> ItemPlan:
    return ItemPlan(
        item_id=item_id,
        source_evidence=[],
        confidence=0.0,
        needs_question=True,
        question_text=f"'{item_id}' 항목 계획 생성에 실패했습니다. 관련 내용을 직접 알려주세요.",
        status="soft_fail",
    )
