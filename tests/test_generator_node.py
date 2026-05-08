"""Generator 노드 테스트 — Step 15."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from app.graph.nodes.generator import generator_node, _parse_response, _build_evidence_text
from app.models import FormDoc, FormItem, ItemPlan, ItemType, DraftItem


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────

def _make_plan(item_id: str, evidence: list[str] | None = None) -> ItemPlan:
    return ItemPlan(item_id=item_id, source_evidence=evidence or [], confidence=0.8)


def _make_form_item(item_id: str, label: str = "연구 목표", char_hint: int = 0) -> FormItem:
    return FormItem(item_id=item_id, label=label, item_type=ItemType.TEXT,
                    xml_path="//p[1]", char_hint=char_hint)


def _make_form_doc(*items: FormItem) -> FormDoc:
    return FormDoc(raw_xml=b"<root/>", items=list(items))


def _mock_llm(response: str):
    return patch("app.graph.nodes.generator.client.call", return_value=response)


def _llm_ok(text: str = "연구 목표 초안입니다.", citations: list | None = None) -> str:
    return json.dumps({"text": text, "citations": citations or ["cv.pdf"]}, ensure_ascii=False)


# ──────────────────────────────────────────────
# 경계 조건
# ──────────────────────────────────────────────

def test_no_plans_returns_empty():
    updates = generator_node({"item_plans": [], "current_item_index": 0})
    assert updates == {}


def test_index_out_of_range_returns_empty():
    updates = generator_node({"item_plans": [_make_plan("q1")], "current_item_index": 5})
    assert updates == {}


# ──────────────────────────────────────────────
# 정상 생성
# ──────────────────────────────────────────────

def test_draft_stored_in_drafts():
    state = {
        "item_plans": [_make_plan("q1", evidence=["딥러닝 연구"])],
        "current_item_index": 0,
        "form_doc": _make_form_doc(_make_form_item("q1", "연구 목표")),
        "target_tone": "공식적",
    }
    with _mock_llm(_llm_ok("연구 목표 초안입니다.")):
        updates = generator_node(state)
    assert "q1" in updates["drafts"]
    assert updates["drafts"]["q1"].text == "연구 목표 초안입니다."


def test_citations_stored():
    state = {
        "item_plans": [_make_plan("q1")],
        "current_item_index": 0,
    }
    with _mock_llm(_llm_ok(citations=["paper.pdf", "cv.pdf"])):
        updates = generator_node(state)
    assert updates["drafts"]["q1"].citations == ["paper.pdf", "cv.pdf"]


def test_user_answer_included_in_prompt_and_consumed():
    """user_answer가 프롬프트에 전달되고 소비(빈 문자열)되어야 한다."""
    state = {
        "item_plans": [_make_plan("q1")],
        "current_item_index": 0,
        "user_answer": "연구비 예산은 5000만원입니다.",
    }
    with _mock_llm(_llm_ok()) as mock_call:
        updates = generator_node(state)
    # 프롬프트에 user_answer 포함 여부 확인
    call_args = mock_call.call_args
    prompt_content = call_args[1].get("messages") or call_args[0][0]
    assert any("5000만원" in str(m) for m in prompt_content)
    # user_answer 소비
    assert updates["user_answer"] == ""


def test_existing_draft_retry_count_preserved():
    """재시도 시 이전 DraftItem의 retry_count가 유지되어야 한다."""
    existing_draft = DraftItem(item_id="q1", text="이전 초안", citations=[], retry_count=1)
    state = {
        "item_plans": [_make_plan("q1")],
        "current_item_index": 0,
        "drafts": {"q1": existing_draft},
    }
    with _mock_llm(_llm_ok()):
        updates = generator_node(state)
    assert updates["drafts"]["q1"].retry_count == 1


def test_tone_guide_used():
    """target_tone이 프롬프트에 전달되어야 한다."""
    state = {
        "item_plans": [_make_plan("q1")],
        "current_item_index": 0,
        "target_tone": "친근한",
    }
    with _mock_llm(_llm_ok()) as mock_call:
        generator_node(state)
    prompt_content = str(mock_call.call_args)
    assert "친근한" in prompt_content


# ──────────────────────────────────────────────
# LLM 실패
# ──────────────────────────────────────────────

def test_solar_api_error_soft_fallback():
    from app.llm.solar_client import SolarAPIError
    state = {
        "item_plans": [_make_plan("q1")],
        "current_item_index": 0,
    }
    with patch("app.graph.nodes.generator.client.call", side_effect=SolarAPIError("timeout")):
        updates = generator_node(state)
    assert "[생성 실패]" in updates["drafts"]["q1"].text


# ──────────────────────────────────────────────
# _parse_response 단위 테스트
# ──────────────────────────────────────────────

def test_parse_direct_json():
    text, cits = _parse_response('{"text": "연구 목표", "citations": ["cv.pdf"]}')
    assert text == "연구 목표"
    assert cits == ["cv.pdf"]


def test_parse_markdown_block():
    raw = '```json\n{"text": "초안", "citations": []}\n```'
    text, cits = _parse_response(raw)
    assert text == "초안"


def test_parse_embedded_json():
    raw = '결과: {"text": "초안 내용", "citations": []} 이상입니다.'
    text, _ = _parse_response(raw)
    assert text == "초안 내용"


def test_parse_invalid_json_soft_fallback():
    raw = "이것은 JSON이 아닌 텍스트 초안입니다."
    text, cits = _parse_response(raw)
    assert text == raw
    assert cits == []


def test_parse_empty_citations():
    text, cits = _parse_response('{"text": "초안", "citations": null}')
    assert cits == []


# ──────────────────────────────────────────────
# _build_evidence_text 단위 테스트
# ──────────────────────────────────────────────

def test_build_evidence_combines_snippets():
    result = _build_evidence_text(["근거1", "근거2"], "")
    assert "근거1" in result
    assert "근거2" in result


def test_build_evidence_appends_user_answer():
    result = _build_evidence_text(["근거1"], "사용자 답변")
    assert "사용자 직접 제공" in result
    assert "사용자 답변" in result


def test_build_evidence_no_snippets_user_answer_only():
    result = _build_evidence_text([], "사용자 답변")
    assert "사용자 답변" in result


def test_build_evidence_truncated():
    long_snippet = "A" * 5000
    result = _build_evidence_text([long_snippet], "")
    assert len(result) < 5000
    assert "이하 생략" in result
