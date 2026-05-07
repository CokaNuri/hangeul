"""Planner 노드 테스트 — Step 14.

Solar Pro 호출을 mock해 planner_node의 계획 생성 로직을 검증한다.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from app.graph.nodes.planner import planner_node, _extract_json_array, _build_materials_summary
from app.models import FormDoc, FormItem, ItemPlan, ItemType, MaterialBundle, MaterialDoc


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────

def _make_form_item(item_id: str, label: str, item_type: ItemType = ItemType.TEXT) -> FormItem:
    return FormItem(item_id=item_id, label=label, item_type=item_type, xml_path="//p[1]")


def _make_form_doc(*items: FormItem) -> FormDoc:
    return FormDoc(raw_xml=b"<root/>", items=list(items))


def _make_material_bundle(summary: str = "딥러닝 기반 NLP 연구 경력 3년") -> MaterialBundle:
    return MaterialBundle(docs=[MaterialDoc(source_name="cv.pdf", masked_text="...", summary=summary)])


def _mock_llm(response_text: str):
    return patch("app.graph.nodes.planner.client.call", return_value=response_text)


def _llm_response(*plans: dict) -> str:
    return json.dumps(plans, ensure_ascii=False)


# ──────────────────────────────────────────────
# form_doc 없음
# ──────────────────────────────────────────────

def test_no_form_doc_returns_empty():
    updates = planner_node({})
    assert updates["item_plans"] == []
    assert updates["current_item_index"] == 0


def test_none_form_doc_returns_empty():
    updates = planner_node({"form_doc": None})
    assert updates["item_plans"] == []


# ──────────────────────────────────────────────
# PII 항목 처리
# ──────────────────────────────────────────────

def test_pii_item_never_calls_llm():
    form_doc = _make_form_doc(_make_form_item("p1", "주민번호", ItemType.PII))
    with patch("app.graph.nodes.planner.client.call") as mock_call:
        updates = planner_node({"form_doc": form_doc})
        mock_call.assert_not_called()
    plan = updates["item_plans"][0]
    assert plan.item_id == "p1"
    assert plan.needs_question is False
    assert plan.source_evidence == []
    assert plan.confidence == 1.0


def test_only_pii_items_no_llm_call():
    form_doc = _make_form_doc(
        _make_form_item("p1", "성명", ItemType.PII),
        _make_form_item("p2", "생년월일", ItemType.PII),
    )
    with patch("app.graph.nodes.planner.client.call") as mock_call:
        updates = planner_node({"form_doc": form_doc})
        mock_call.assert_not_called()
    assert len(updates["item_plans"]) == 2


# ──────────────────────────────────────────────
# 정상 LLM 응답
# ──────────────────────────────────────────────

def test_high_confidence_plan():
    form_doc = _make_form_doc(_make_form_item("q1", "연구 목표"))
    bundle = _make_material_bundle()
    resp = _llm_response(
        {"item_id": "q1", "source_evidence": ["딥러닝 연구"], "confidence": 0.9,
         "needs_question": False, "question_text": ""}
    )
    with _mock_llm(resp):
        updates = planner_node({"form_doc": form_doc, "material_bundle": bundle})
    plan = updates["item_plans"][0]
    assert plan.item_id == "q1"
    assert plan.confidence == 0.9
    assert plan.needs_question is False
    assert "딥러닝 연구" in plan.source_evidence


def test_low_confidence_forces_needs_question():
    """confidence < 0.5이면 needs_question=True 강제."""
    form_doc = _make_form_doc(_make_form_item("q1", "연구 성과"))
    resp = _llm_response(
        {"item_id": "q1", "source_evidence": [], "confidence": 0.3,
         "needs_question": False, "question_text": ""}
    )
    with _mock_llm(resp):
        updates = planner_node({"form_doc": form_doc})
    plan = updates["item_plans"][0]
    assert plan.needs_question is True
    assert plan.question_text  # 자동 생성된 질문 있어야 함


def test_low_confidence_preserves_existing_question():
    """LLM이 이미 question_text를 제공한 경우 그대로 사용."""
    form_doc = _make_form_doc(_make_form_item("q1", "연구 성과"))
    resp = _llm_response(
        {"item_id": "q1", "source_evidence": [], "confidence": 0.2,
         "needs_question": True, "question_text": "어떤 성과가 있나요?"}
    )
    with _mock_llm(resp):
        updates = planner_node({"form_doc": form_doc})
    assert updates["item_plans"][0].question_text == "어떤 성과가 있나요?"


def test_multiple_items_all_planned():
    form_doc = _make_form_doc(
        _make_form_item("q1", "연구 목표"),
        _make_form_item("q2", "연구 방법"),
    )
    resp = _llm_response(
        {"item_id": "q1", "source_evidence": ["목표1"], "confidence": 0.8,
         "needs_question": False, "question_text": ""},
        {"item_id": "q2", "source_evidence": ["방법1"], "confidence": 0.7,
         "needs_question": False, "question_text": ""},
    )
    with _mock_llm(resp):
        updates = planner_node({"form_doc": form_doc})
    assert len(updates["item_plans"]) == 2
    assert updates["current_item_index"] == 0


def test_mixed_pii_and_text_items():
    form_doc = _make_form_doc(
        _make_form_item("p1", "주민번호", ItemType.PII),
        _make_form_item("q1", "연구 목표", ItemType.TEXT),
    )
    resp = _llm_response(
        {"item_id": "q1", "source_evidence": ["목표"], "confidence": 0.85,
         "needs_question": False, "question_text": ""},
    )
    with _mock_llm(resp):
        updates = planner_node({"form_doc": form_doc})
    assert len(updates["item_plans"]) == 2
    pii_plan = next(p for p in updates["item_plans"] if p.item_id == "p1")
    assert pii_plan.needs_question is False


# ──────────────────────────────────────────────
# 소프트 실패
# ──────────────────────────────────────────────

def test_solar_api_error_soft_fail():
    from app.llm.solar_client import SolarAPIError
    form_doc = _make_form_doc(_make_form_item("q1", "연구 목표"))
    with patch("app.graph.nodes.planner.client.call", side_effect=SolarAPIError("timeout")):
        updates = planner_node({"form_doc": form_doc})
    plan = updates["item_plans"][0]
    assert plan.confidence == 0.0
    assert plan.needs_question is True
    assert plan.status == "soft_fail"


def test_json_parse_failure_soft_fail():
    form_doc = _make_form_doc(_make_form_item("q1", "연구 목표"))
    with _mock_llm("이것은 JSON이 아닙니다"):
        updates = planner_node({"form_doc": form_doc})
    plan = updates["item_plans"][0]
    assert plan.status == "soft_fail"


def test_missing_item_in_llm_response_soft_fail():
    """LLM 응답에 일부 항목이 빠져있으면 해당 항목만 소프트 실패."""
    form_doc = _make_form_doc(
        _make_form_item("q1", "연구 목표"),
        _make_form_item("q2", "연구 방법"),
    )
    # q2가 응답에 없음
    resp = _llm_response(
        {"item_id": "q1", "source_evidence": ["목표"], "confidence": 0.8,
         "needs_question": False, "question_text": ""},
    )
    with _mock_llm(resp):
        updates = planner_node({"form_doc": form_doc})
    plans = {p.item_id: p for p in updates["item_plans"]}
    assert plans["q1"].confidence == 0.8
    assert plans["q2"].status == "soft_fail"


# ──────────────────────────────────────────────
# JSON 배열 추출 단위 테스트
# ──────────────────────────────────────────────

def test_extract_json_array_direct():
    raw = '[{"a": 1}]'
    assert _extract_json_array(raw) == [{"a": 1}]


def test_extract_json_array_markdown_block():
    raw = '```json\n[{"a": 1}]\n```'
    assert _extract_json_array(raw) == [{"a": 1}]


def test_extract_json_array_embedded():
    raw = '결과는 다음과 같습니다: [{"item_id": "q1"}] 이상입니다.'
    result = _extract_json_array(raw)
    assert result == [{"item_id": "q1"}]


def test_extract_json_array_invalid_returns_none():
    assert _extract_json_array("not json") is None


def test_extract_json_array_object_returns_none():
    """배열이 아닌 객체는 None 반환."""
    assert _extract_json_array('{"a": 1}') is None


# ──────────────────────────────────────────────
# _build_materials_summary 단위 테스트
# ──────────────────────────────────────────────

def test_build_materials_summary_with_summary():
    bundle = _make_material_bundle("연구 경력 3년")
    result = _build_materials_summary(bundle)
    assert "cv.pdf" in result
    assert "연구 경력 3년" in result


def test_build_materials_summary_no_summary_uses_preview():
    bundle = MaterialBundle(docs=[
        MaterialDoc(source_name="paper.pdf", masked_text="딥러닝 연구 내용", summary="")
    ])
    result = _build_materials_summary(bundle)
    assert "paper.pdf" in result
    assert "딥러닝 연구 내용" in result


def test_build_materials_summary_none_bundle():
    assert _build_materials_summary(None) == "제공된 자료 없음."


def test_build_materials_summary_empty_bundle():
    assert _build_materials_summary(MaterialBundle(docs=[])) == "제공된 자료 없음."


# ──────────────────────────────────────────────
# confidence 경계값
# ──────────────────────────────────────────────

def test_confidence_exactly_at_threshold_no_question():
    """confidence == 0.5 이면 needs_question을 강제하지 않는다."""
    form_doc = _make_form_doc(_make_form_item("q1", "연구 목표"))
    resp = _llm_response(
        {"item_id": "q1", "source_evidence": ["근거"], "confidence": 0.5,
         "needs_question": False, "question_text": ""},
    )
    with _mock_llm(resp):
        updates = planner_node({"form_doc": form_doc})
    assert updates["item_plans"][0].needs_question is False


def test_confidence_clamped_above_1():
    form_doc = _make_form_doc(_make_form_item("q1", "연구 목표"))
    resp = _llm_response(
        {"item_id": "q1", "source_evidence": [], "confidence": 1.5,
         "needs_question": False, "question_text": ""},
    )
    with _mock_llm(resp):
        updates = planner_node({"form_doc": form_doc})
    assert updates["item_plans"][0].confidence == 1.0
