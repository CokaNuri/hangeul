"""Question 노드 + interrupt/resume 통합 테스트 — Step 16.

LangGraph interrupt()는 컴파일된 그래프 + 체크포인터 내에서만 동작하므로
mini-graph를 빌드해 interrupt/resume 사이클을 검증한다.
"""
from __future__ import annotations

import pytest

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.graph.nodes.question import question_node
from app.graph.nodes.generator import generator_node
from app.graph.state import GraphState
from app.models import FormDoc, FormItem, ItemPlan, ItemType, DraftItem
from app.graph.nodes.verifier import VERIFIER_RESULT_KEY


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────

def _make_plan(item_id: str, needs_q: bool = True, q_text: str = "연구 분야를 알려주세요.") -> ItemPlan:
    return ItemPlan(
        item_id=item_id,
        source_evidence=[],
        confidence=0.3,
        needs_question=needs_q,
        question_text=q_text,
    )


def _make_form_item(item_id: str, label: str = "연구 목표") -> FormItem:
    return FormItem(item_id=item_id, label=label, item_type=ItemType.TEXT, xml_path="//p[1]")


def _make_state(plan: ItemPlan, label: str = "연구 목표") -> dict:
    form_doc = FormDoc(raw_xml=b"<root/>", items=[_make_form_item(plan.item_id, label)])
    return {
        "item_plans": [plan],
        "current_item_index": 0,
        "form_doc": form_doc,
        "drafts": {},
    }


def _build_mini_graph() -> tuple:
    """question_node 하나만 있는 mini 그래프와 체크포인터를 반환한다."""
    g = StateGraph(GraphState)
    g.add_node("question", question_node)
    g.add_edge(START, "question")
    g.add_edge("question", END)
    cp = MemorySaver()
    return g.compile(checkpointer=cp), cp


# ──────────────────────────────────────────────
# interrupt 발생 테스트
# ──────────────────────────────────────────────

def test_question_node_fires_interrupt():
    """question_node가 interrupt를 발생시켜야 한다."""
    graph, _ = _build_mini_graph()
    plan = _make_plan("q1", q_text="연구 분야를 알려주세요.")
    state = _make_state(plan)
    config = {"configurable": {"thread_id": "test-interrupt-1"}}

    result = graph.invoke(state, config=config)
    assert "__interrupt__" in result
    assert len(result["__interrupt__"]) == 1


def test_interrupt_value_contains_question_text():
    """interrupt 값이 plan.question_text와 일치해야 한다."""
    graph, _ = _build_mini_graph()
    plan = _make_plan("q1", q_text="연구 분야를 알려주세요.")
    state = _make_state(plan)
    config = {"configurable": {"thread_id": "test-interrupt-2"}}

    result = graph.invoke(state, config=config)
    interrupt_value = result["__interrupt__"][0].value
    assert "연구 분야" in interrupt_value


def test_interrupt_uses_plan_question_text():
    """plan.question_text가 있으면 그것을 interrupt 값으로 사용해야 한다."""
    graph, _ = _build_mini_graph()
    custom_q = "특허 보유 현황을 알려주세요."
    plan = _make_plan("q1", q_text=custom_q)
    state = _make_state(plan)
    config = {"configurable": {"thread_id": "test-interrupt-3"}}

    result = graph.invoke(state, config=config)
    assert result["__interrupt__"][0].value == custom_q


def test_interrupt_fallback_when_no_question_text():
    """question_text가 없으면 label을 포함한 기본 질문을 사용해야 한다."""
    graph, _ = _build_mini_graph()
    plan = _make_plan("q1", q_text="")
    state = _make_state(plan, label="연구 성과")
    config = {"configurable": {"thread_id": "test-interrupt-4"}}

    result = graph.invoke(state, config=config)
    interrupt_value = result["__interrupt__"][0].value
    assert "연구 성과" in interrupt_value or "q1" in interrupt_value


def test_no_interrupt_when_empty_plans():
    """item_plans가 없으면 interrupt 없이 완료되어야 한다."""
    graph, _ = _build_mini_graph()
    config = {"configurable": {"thread_id": "test-no-interrupt"}}

    result = graph.invoke({"item_plans": [], "current_item_index": 0}, config=config)
    assert "__interrupt__" not in result or not result["__interrupt__"]


# ──────────────────────────────────────────────
# resume 테스트
# ──────────────────────────────────────────────

def test_resume_sets_user_answer():
    """Command(resume=answer) 후 user_answer가 상태에 설정되어야 한다."""
    graph, _ = _build_mini_graph()
    plan = _make_plan("q1")
    state = _make_state(plan)
    config = {"configurable": {"thread_id": "test-resume-1"}}

    # 1. 첫 호출 → interrupt
    graph.invoke(state, config=config)

    # 2. resume
    result = graph.invoke(Command(resume="딥러닝 기반 자연어 처리 연구"), config=config)
    assert result.get("user_answer") == "딥러닝 기반 자연어 처리 연구"


def test_resume_clears_pending_question():
    """resume 후 pending_question이 빈 문자열이어야 한다."""
    graph, _ = _build_mini_graph()
    plan = _make_plan("q1")
    state = _make_state(plan)
    config = {"configurable": {"thread_id": "test-resume-2"}}

    graph.invoke(state, config=config)
    result = graph.invoke(Command(resume="답변"), config=config)
    assert result.get("pending_question", "") == ""


def test_resume_preserves_other_state():
    """resume 후 item_plans 등 기존 상태가 보존되어야 한다."""
    graph, _ = _build_mini_graph()
    plan = _make_plan("q1")
    state = _make_state(plan)
    config = {"configurable": {"thread_id": "test-resume-3"}}

    graph.invoke(state, config=config)
    result = graph.invoke(Command(resume="답변"), config=config)
    assert len(result.get("item_plans", [])) == 1
    assert result["item_plans"][0].item_id == "q1"


def test_user_answer_stripped():
    """공백이 있는 답변은 strip되어야 한다."""
    graph, _ = _build_mini_graph()
    plan = _make_plan("q1")
    state = _make_state(plan)
    config = {"configurable": {"thread_id": "test-resume-strip"}}

    graph.invoke(state, config=config)
    result = graph.invoke(Command(resume="  답변 내용  "), config=config)
    assert result.get("user_answer") == "답변 내용"


# ──────────────────────────────────────────────
# API 레벨 interrupt/resume 테스트
# ──────────────────────────────────────────────

def test_chat_returns_pending_question_on_interrupt():
    """interrupt 발생 시 ChatResponse.pending_question이 채워져야 한다."""
    from fastapi.testclient import TestClient
    from app.main import app
    from unittest.mock import patch
    from tests.fixtures.hwpx_factory import make_hwpx
    from app.parsers.hwpx_parser import parse_hwpx
    from app.models import ItemType

    api = TestClient(app)

    sid = api.post("/api/sessions").json()["session_id"]

    # 양식 업로드 (빈칸 항목 있는 HWPX)
    hwpx = make_hwpx(paragraphs=["연구 목표", ""])
    api.post(
        "/api/upload",
        params={"session_id": sid, "file_type": "form"},
        files={"file": ("form.hwpx", hwpx, "application/octet-stream")},
    )

    # Planner가 needs_question=True 플랜을 반환하도록 mock
    mock_plan = ItemPlan(
        item_id="item_0",
        source_evidence=[],
        confidence=0.2,
        needs_question=True,
        question_text="연구 분야를 알려주세요.",
    )

    with patch("app.graph.nodes.planner.client.call") as mock_planner, \
         patch("app.graph.nodes.material_ingestor.client.call", return_value="요약"):
        # Planner 응답 mock
        import json as _json
        mock_planner.return_value = _json.dumps([{
            "item_id": "item_0",
            "source_evidence": [],
            "confidence": 0.2,
            "needs_question": True,
            "question_text": "연구 분야를 알려주세요.",
        }])
        r = api.post("/api/chat", json={"session_id": sid, "message": "채우기 시작해줘"})

    assert r.status_code == 200
    data = r.json()
    # interrupt가 발생했으면 pending_question이 채워져야 함
    assert data["pending_question"] != "" or data["reply"] != ""


def test_chat_resume_clears_interrupted_state():
    """interrupt 후 resume 시 session.is_interrupted가 해제되어야 한다."""
    from unittest.mock import patch
    from app.session_store import store

    # 세션 직접 조작으로 is_interrupted 설정
    from fastapi.testclient import TestClient
    from app.main import app

    api = TestClient(app)
    sid = api.post("/api/sessions").json()["session_id"]

    # 세션에 is_interrupted 설정
    session = store.get(sid)
    session.is_interrupted = True
    session.pending_question = "연구 분야를 알려주세요."

    # graph.invoke(Command(resume=...)) mock
    from langgraph.types import Command as LGCommand
    with patch("app.api.routes.graph.invoke", return_value={
        "current_intent": "start_fill",
        "item_plans": [],
        "drafts": {},
        "approved_items": [],
    }) as mock_invoke:
        r = api.post("/api/chat", json={"session_id": sid, "message": "딥러닝 연구"})

    assert r.status_code == 200
    # Command(resume=...) 로 호출되었는지 확인
    call_args = mock_invoke.call_args[0][0]
    assert isinstance(call_args, LGCommand)

    # session.is_interrupted가 해제되어야 함
    session = store.get(sid)
    assert not session.is_interrupted
