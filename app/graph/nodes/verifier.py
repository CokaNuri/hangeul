"""Verifier 노드 — Step 15 실제 구현.

담당 역할:
  - 현재 항목의 초안을 Solar Mini로 품질 검증한다.
  - verdict "retry" 시 retry_counts를 증가시킨다.
  - 검증 결과를 verifier_verdict 상태 키에 저장 (_route_verifier가 읽음).
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

# graph.py의 _route_verifier가 이 키를 읽는다
VERIFIER_RESULT_KEY = "verifier_verdict"

_VERDICT_OK = "ok"
_VERDICT_RETRY = "retry"


# ──────────────────────────────────────────────
# 노드 진입점
# ──────────────────────────────────────────────

def verifier_node(state: GraphState) -> GraphState:
    """현재 항목 초안을 Solar Mini로 검증하고 verdict를 상태에 기록한다."""
    plans = state.get("item_plans", [])
    idx = state.get("current_item_index", 0)
    if idx >= len(plans):
        logger.warning("[Verifier] current_item_index=%d >= len(plans)=%d", idx, len(plans))
        return {VERIFIER_RESULT_KEY: _VERDICT_OK}

    plan = plans[idx]
    item_id = plan.item_id
    drafts = state.get("drafts") or {}
    draft = drafts.get(item_id)

    if draft is None:
        logger.warning("[Verifier] item_id=%s 초안 없음 → ok로 처리", item_id)
        return {VERIFIER_RESULT_KEY: _VERDICT_OK}

    # FormDoc에서 항목 메타 조회
    form_doc = state.get("form_doc")
    item_meta = None
    if form_doc:
        item_meta = next((it for it in form_doc.items if it.item_id == item_id), None)
    item_label = item_meta.label if item_meta else item_id

    evidence_text = "\n\n".join(e.strip() for e in (plan.source_evidence or []) if e.strip())

    try:
        verdict, reason = _verify(
            item_label=item_label,
            evidence_text=evidence_text,
            draft_text=draft.text,
        )
    except SolarAPIError as exc:
        logger.error("[Verifier] LLM 실패 item_id=%s: %s → ok로 처리", item_id, exc)
        verdict, reason = _VERDICT_OK, ""

    updates: dict = {VERIFIER_RESULT_KEY: verdict}

    if verdict == _VERDICT_RETRY:
        retry_counts = dict(state.get("retry_counts") or {})
        retry_counts[item_id] = retry_counts.get(item_id, 0) + 1
        new_drafts = dict(drafts)
        new_drafts[item_id] = DraftItem(
            item_id=item_id,
            text=draft.text,
            citations=draft.citations,
            retry_count=retry_counts[item_id],
        )
        updates["retry_counts"] = retry_counts
        updates["drafts"] = new_drafts
        logger.info(
            "[Verifier] item_id=%s → retry #%d, 이유: %s",
            item_id, retry_counts[item_id], reason,
        )
    else:
        logger.info("[Verifier] item_id=%s → ok", item_id)

    return updates


# ──────────────────────────────────────────────
# LLM 호출
# ──────────────────────────────────────────────

def _verify(item_label: str, evidence_text: str, draft_text: str) -> tuple[str, str]:
    """Solar Mini로 초안을 검증하고 (verdict, reason)을 반환한다."""
    prompt = load_prompt(
        "verifier",
        item_label=item_label,
        evidence_text=evidence_text or "(근거 자료 없음)",
        draft_text=draft_text,
    )
    raw = client.call(
        messages=[{"role": "user", "content": prompt}],
        model=settings.solar_mini_model,
        temperature=0.0,
        max_tokens=100,
    )
    return _parse_response(raw)


def _parse_response(raw: str) -> tuple[str, str]:
    """{"verdict": "ok"/"retry", "reason": "..."} 파싱."""
    stripped = raw.strip()

    for candidate in _iter_json_candidates(stripped):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict) and "verdict" in data:
                verdict = str(data["verdict"]).lower().strip()
                if verdict not in (_VERDICT_OK, _VERDICT_RETRY):
                    verdict = _VERDICT_OK
                reason = str(data.get("reason") or "").strip()
                return verdict, reason
        except json.JSONDecodeError:
            continue

    # 키워드 폴백
    if "retry" in stripped.lower():
        return _VERDICT_RETRY, ""
    return _VERDICT_OK, ""


def _iter_json_candidates(text: str):
    """텍스트에서 JSON 후보 문자열을 순서대로 산출한다."""
    yield text

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        yield match.group(1)

    start = text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[start:i + 1]
                    break
