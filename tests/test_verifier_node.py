"""Verifier 노드 테스트 — Step 15."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from app.graph.nodes.verifier import verifier_node, VERIFIER_RESULT_KEY, _parse_response
from app.models import FormDoc, FormItem, ItemPlan, ItemType, DraftItem, MaterialBundle


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────

def _make_plan(item_id: str, evidence: list[str] | None = None) -> ItemPlan:
    return ItemPlan(item_id=item_id, source_evidence=evidence or [], confidence=0.8)


def _make_draft(item_id: str, text: str = "초안 텍스트", retry: int = 0) -> DraftItem:
    return DraftItem(item_id=item_id, text=text, citations=[], retry_count=retry)


def _make_form_doc(item_id: str, label: str = "연구 목표") -> FormDoc:
    item = FormItem(item_id=item_id, label=label, item_type=ItemType.TEXT, xml_path="//p[1]")
    return FormDoc(raw_xml=b"<root/>", items=[item])


def _mock_llm(response: str):
    return patch("app.graph.nodes.verifier.client.call", return_value=response)


def _ok_response() -> str:
    return json.dumps({"verdict": "ok"})


def _retry_response(reason: str = "환각 내용 포함") -> str:
    return json.dumps({"verdict": "retry", "reason": reason})


# ──────────────────────────────────────────────
# 경계 조건
# ──────────────────────────────────────────────

def test_no_plans_returns_ok():
    updates = verifier_node({"item_plans": [], "current_item_index": 0})
    assert updates[VERIFIER_RESULT_KEY] == "ok"


def test_no_draft_returns_ok():
    state = {
        "item_plans": [_make_plan("q1")],
        "current_item_index": 0,
        "drafts": {},
    }
    updates = verifier_node(state)
    assert updates[VERIFIER_RESULT_KEY] == "ok"


# ──────────────────────────────────────────────
# 정상 검증
# ──────────────────────────────────────────────

def test_verdict_ok():
    state = {
        "item_plans": [_make_plan("q1", ["딥러닝 연구"])],
        "current_item_index": 0,
        "drafts": {"q1": _make_draft("q1", "딥러닝 기반 연구를 수행하였습니다.")},
        "form_doc": _make_form_doc("q1", "연구 목표"),
    }
    with _mock_llm(_ok_response()):
        updates = verifier_node(state)
    assert updates[VERIFIER_RESULT_KEY] == "ok"
    # ok 시 retry_counts 변화 없음
    assert "retry_counts" not in updates


def test_verdict_retry_increments_count():
    state = {
        "item_plans": [_make_plan("q1")],
        "current_item_index": 0,
        "drafts": {"q1": _make_draft("q1")},
        "retry_counts": {},
    }
    with _mock_llm(_retry_response()):
        updates = verifier_node(state)
    assert updates[VERIFIER_RESULT_KEY] == "retry"
    assert updates["retry_counts"]["q1"] == 1


def test_verdict_retry_increments_existing_count():
    state = {
        "item_plans": [_make_plan("q1")],
        "current_item_index": 0,
        "drafts": {"q1": _make_draft("q1", retry=1)},
        "retry_counts": {"q1": 1},
    }
    with _mock_llm(_retry_response()):
        updates = verifier_node(state)
    assert updates["retry_counts"]["q1"] == 2
    assert updates["drafts"]["q1"].retry_count == 2


def test_verdict_retry_updates_draft_retry_count():
    """retry 시 DraftItem.retry_count가 retry_counts와 동기화되어야 한다."""
    state = {
        "item_plans": [_make_plan("q1")],
        "current_item_index": 0,
        "drafts": {"q1": _make_draft("q1")},
        "retry_counts": {"q1": 0},
    }
    with _mock_llm(_retry_response()):
        updates = verifier_node(state)
    assert updates["drafts"]["q1"].retry_count == 1


def test_other_items_retry_counts_preserved():
    """다른 항목의 retry_counts는 영향받지 않아야 한다."""
    state = {
        "item_plans": [_make_plan("q1")],
        "current_item_index": 0,
        "drafts": {"q1": _make_draft("q1")},
        "retry_counts": {"q2": 3},
    }
    with _mock_llm(_retry_response()):
        updates = verifier_node(state)
    assert updates["retry_counts"]["q2"] == 3


# ──────────────────────────────────────────────
# 소프트 실패
# ──────────────────────────────────────────────

def test_solar_api_error_defaults_to_ok():
    from app.llm.solar_client import SolarAPIError
    state = {
        "item_plans": [_make_plan("q1")],
        "current_item_index": 0,
        "drafts": {"q1": _make_draft("q1")},
    }
    with patch("app.graph.nodes.verifier.client.call", side_effect=SolarAPIError("timeout")):
        updates = verifier_node(state)
    assert updates[VERIFIER_RESULT_KEY] == "ok"


# ──────────────────────────────────────────────
# _parse_response 단위 테스트
# ──────────────────────────────────────────────

def test_parse_ok_verdict():
    verdict, reason = _parse_response('{"verdict": "ok"}')
    assert verdict == "ok"
    assert reason == ""


def test_parse_retry_verdict_with_reason():
    verdict, reason = _parse_response('{"verdict": "retry", "reason": "환각 내용"}')
    assert verdict == "retry"
    assert reason == "환각 내용"


def test_parse_unknown_verdict_defaults_to_ok():
    verdict, _ = _parse_response('{"verdict": "pass"}')
    assert verdict == "ok"


def test_parse_markdown_block():
    raw = '```json\n{"verdict": "ok"}\n```'
    verdict, _ = _parse_response(raw)
    assert verdict == "ok"


def test_parse_embedded_json():
    raw = '검증 결과: {"verdict": "retry", "reason": "문체 불량"} 입니다.'
    verdict, reason = _parse_response(raw)
    assert verdict == "retry"
    assert reason == "문체 불량"


def test_parse_keyword_fallback_retry():
    """JSON 파싱 실패해도 'retry' 키워드 있으면 retry 반환."""
    verdict, _ = _parse_response("이 초안은 retry가 필요합니다.")
    assert verdict == "retry"


def test_parse_keyword_fallback_ok():
    """키워드 없으면 ok 기본값."""
    verdict, _ = _parse_response("알 수 없는 응답입니다.")
    assert verdict == "ok"


def test_parse_uppercase_verdict():
    """대소문자 무관하게 파싱되어야 한다."""
    verdict, _ = _parse_response('{"verdict": "OK"}')
    assert verdict == "ok"
