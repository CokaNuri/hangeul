"""Router 노드 테스트 — Step 13.

Solar API 호출을 mock해 router_node의 인텐트 분류 로직을 검증한다.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from app.graph.nodes.router import router_node, _parse_response, _RouterResult
from app.models import Intent


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────

def _make_state(user_message: str | None = "채우기 시작해줘") -> dict:
    history = []
    if user_message is not None:
        history.append({"role": "user", "content": user_message})
    return {"conversation_history": history}


def _mock_llm(response_text: str):
    """client.call 을 response_text를 반환하도록 패치한다."""
    return patch(
        "app.graph.nodes.router.client.call",
        return_value=response_text,
    )


# ──────────────────────────────────────────────
# 정상 분류
# ──────────────────────────────────────────────

def test_start_fill_intent():
    resp = json.dumps({"intent": "start_fill", "confidence": 0.97, "target_item": None})
    with _mock_llm(resp):
        updates = router_node(_make_state("채우기 시작해줘"))
    assert updates["current_intent"] == Intent.START_FILL.value


def test_rewrite_item_with_target():
    resp = json.dumps({"intent": "rewrite_item", "confidence": 0.92, "target_item": "3"})
    with _mock_llm(resp):
        updates = router_node(_make_state("3번 항목 다시 써줘"))
    assert updates["current_intent"] == Intent.REWRITE_ITEM.value
    assert updates["current_item_id"] == "3"


def test_change_tone_with_target():
    resp = json.dumps({"intent": "change_tone", "confidence": 0.88, "target_item": "연구 목표"})
    with _mock_llm(resp):
        updates = router_node(_make_state("연구 목표 부분 좀 더 공식적으로 바꿔줘"))
    assert updates["current_intent"] == Intent.CHANGE_TONE.value
    assert updates["current_item_id"] == "연구 목표"


def test_add_material_no_target():
    resp = json.dumps({"intent": "add_material", "confidence": 0.95, "target_item": None})
    with _mock_llm(resp):
        updates = router_node(_make_state("CV 파일 추가할게요"))
    assert updates["current_intent"] == Intent.ADD_MATERIAL.value
    assert "current_item_id" not in updates


def test_general_qa_intent():
    resp = json.dumps({"intent": "general_qa", "confidence": 0.80, "target_item": None})
    with _mock_llm(resp):
        updates = router_node(_make_state("이 양식 마감일이 언제예요?"))
    assert updates["current_intent"] == Intent.GENERAL_QA.value


def test_upload_form_intent():
    resp = json.dumps({"intent": "upload_form", "confidence": 0.93, "target_item": None})
    with _mock_llm(resp):
        updates = router_node(_make_state("양식 파일 올릴게요"))
    assert updates["current_intent"] == Intent.UPLOAD_FORM.value


def test_upload_material_intent():
    resp = json.dumps({"intent": "upload_material", "confidence": 0.91, "target_item": None})
    with _mock_llm(resp):
        updates = router_node(_make_state("CV 업로드합니다"))
    assert updates["current_intent"] == Intent.UPLOAD_MATERIAL.value


# ──────────────────────────────────────────────
# disambiguation 분기
# ──────────────────────────────────────────────

def test_low_confidence_sets_disambiguation():
    resp = json.dumps({"intent": "disambiguation", "confidence": 0.3, "target_item": None})
    with _mock_llm(resp):
        updates = router_node(_make_state("음..."))
    assert updates["current_intent"] == Intent.DISAMBIGUATION.value
    assert "pending_question" in updates
    assert "1." in updates["pending_question"]


def test_confidence_below_threshold_overrides_intent():
    """confidence 0.6 미만이면 어떤 intent든 disambiguation으로 바뀐다."""
    resp = json.dumps({"intent": "start_fill", "confidence": 0.4, "target_item": None})
    with _mock_llm(resp):
        updates = router_node(_make_state("뭔가 해줘"))
    assert updates["current_intent"] == Intent.DISAMBIGUATION.value
    assert "pending_question" in updates


def test_confidence_at_threshold_not_disambiguation():
    """confidence == 0.6 이면 disambiguation이 아니어야 한다."""
    resp = json.dumps({"intent": "start_fill", "confidence": 0.6, "target_item": None})
    with _mock_llm(resp):
        updates = router_node(_make_state("채우기"))
    assert updates["current_intent"] == Intent.START_FILL.value


# ──────────────────────────────────────────────
# 소프트 실패
# ──────────────────────────────────────────────

def test_solar_api_error_fallback():
    from app.llm.solar_client import SolarAPIError
    with patch("app.graph.nodes.router.client.call", side_effect=SolarAPIError("timeout")):
        updates = router_node(_make_state("시작해줘"))
    assert updates["current_intent"] == Intent.GENERAL_QA.value


def test_no_user_message_returns_empty():
    state = {"conversation_history": [{"role": "assistant", "content": "안녕하세요"}]}
    updates = router_node(state)
    assert updates == {}


def test_empty_history_returns_empty():
    updates = router_node({"conversation_history": []})
    assert updates == {}


def test_missing_history_key_returns_empty():
    updates = router_node({})
    assert updates == {}


# ──────────────────────────────────────────────
# JSON 파싱 폴백
# ──────────────────────────────────────────────

def test_markdown_json_block_parsed():
    """마크다운 코드 블록 안 JSON도 파싱돼야 한다."""
    resp = '```json\n{"intent": "general_qa", "confidence": 0.8, "target_item": null}\n```'
    with _mock_llm(resp):
        updates = router_node(_make_state("질문 있어요"))
    assert updates["current_intent"] == Intent.GENERAL_QA.value


def test_bare_json_in_text_parsed():
    """텍스트 내 첫 번째 {...} 블록에서 JSON을 추출해야 한다."""
    resp = '분석 결과: {"intent": "start_fill", "confidence": 0.9, "target_item": null} 이상입니다.'
    with _mock_llm(resp):
        updates = router_node(_make_state("시작해줘"))
    assert updates["current_intent"] == Intent.START_FILL.value


def test_keyword_fallback_on_json_failure():
    """JSON 파싱이 완전히 실패하면 키워드 폴백이 동작해야 한다."""
    with _mock_llm("채우기 시작하면 start_fill 인텐트입니다"):
        updates = router_node(_make_state("채우기 시작해"))
    assert updates["current_intent"] == Intent.START_FILL.value


def test_keyword_fallback_rewrite():
    with _mock_llm("rewrite_item 인텐트로 다시 써야 합니다"):
        updates = router_node(_make_state("다시 써줘"))
    assert updates["current_intent"] == Intent.REWRITE_ITEM.value


def test_keyword_fallback_general_qa_default():
    """키워드가 하나도 없으면 general_qa로 기본 처리."""
    with _mock_llm("이것은 완전히 인식 불가한 응답입니다..."):
        updates = router_node(_make_state("알 수 없는 요청"))
    assert updates["current_intent"] == Intent.GENERAL_QA.value


# ──────────────────────────────────────────────
# _parse_response 단위 테스트
# ──────────────────────────────────────────────

def test_parse_null_string_target_item():
    """target_item이 "null" 문자열이면 None으로 처리해야 한다."""
    raw = '{"intent": "start_fill", "confidence": 0.9, "target_item": "null"}'
    result = _parse_response(raw)
    assert result.target_item is None


def test_parse_invalid_intent_falls_back_to_general_qa():
    raw = '{"intent": "unknown_intent", "confidence": 0.9, "target_item": null}'
    result = _parse_response(raw)
    assert result.intent == Intent.GENERAL_QA.value


def test_parse_confidence_clamped():
    raw = '{"intent": "start_fill", "confidence": 1.5, "target_item": null}'
    result = _parse_response(raw)
    assert result.confidence == 1.0


def test_parse_confidence_invalid_type():
    """confidence가 파싱 불가능한 타입이면 0.5 기본값 사용."""
    raw = '{"intent": "start_fill", "confidence": "high", "target_item": null}'
    result = _parse_response(raw)
    assert result.confidence == 0.5
