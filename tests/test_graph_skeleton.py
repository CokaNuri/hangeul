"""LangGraph 스켈레톤 그래프 단위 테스트 — Step 8."""

import pytest

from app.graph.graph import build_graph
from app.graph.state import initial_state
from app.models import Intent
from app.parsers.hwpx_parser import parse_hwpx
from tests.fixtures.hwpx_factory import make_hwpx


@pytest.fixture
def g():
    """테스트마다 독립적인 그래프 인스턴스를 생성한다."""
    return build_graph()


def _cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


# ── 인텐트별 라우팅 ────────────────────────────

class TestRouting:
    def test_general_qa_reaches_end(self, g):
        state = initial_state()
        state["current_intent"] = Intent.GENERAL_QA.value
        result = g.invoke(state, config=_cfg("t-qa"))
        assert result["errors"] == []

    def test_disambiguation_reaches_end(self, g):
        state = initial_state()
        state["current_intent"] = Intent.DISAMBIGUATION.value
        result = g.invoke(state, config=_cfg("t-dis"))
        assert result["errors"] == []

    def test_upload_form_reaches_end(self, g):
        state = initial_state()
        state["current_intent"] = Intent.UPLOAD_FORM.value
        result = g.invoke(state, config=_cfg("t-uf"))
        assert result["errors"] == []

    def test_upload_material_reaches_end(self, g):
        state = initial_state()
        state["current_intent"] = Intent.UPLOAD_MATERIAL.value
        result = g.invoke(state, config=_cfg("t-um"))
        assert result["errors"] == []


# ── start_fill 전체 흐름 ──────────────────────

class TestStartFill:
    @pytest.fixture
    def form_doc_2items(self):
        hwpx = make_hwpx(paragraphs=["연구 목표", "연구 내용"], table_rows=None)
        return parse_hwpx(hwpx)

    def test_all_items_approved(self, g, form_doc_2items):
        state = initial_state()
        state["current_intent"] = Intent.START_FILL.value
        state["form_doc"] = form_doc_2items
        result = g.invoke(state, config=_cfg("t-fill"))
        assert len(result["approved_items"]) == len(form_doc_2items.items)

    def test_drafts_created_for_all_items(self, g, form_doc_2items):
        state = initial_state()
        state["current_intent"] = Intent.START_FILL.value
        state["form_doc"] = form_doc_2items
        result = g.invoke(state, config=_cfg("t-drafts"))
        assert len(result["drafts"]) == len(form_doc_2items.items)

    def test_no_form_doc_produces_empty_plan(self, g):
        state = initial_state()
        state["current_intent"] = Intent.START_FILL.value
        # form_doc=None
        result = g.invoke(state, config=_cfg("t-noform"))
        assert result["item_plans"] == []
        assert result["approved_items"] == []

    def test_errors_empty_on_happy_path(self, g, form_doc_2items):
        state = initial_state()
        state["current_intent"] = Intent.START_FILL.value
        state["form_doc"] = form_doc_2items
        result = g.invoke(state, config=_cfg("t-err"))
        assert result["errors"] == []


# ── rewrite_item ──────────────────────────────

class TestRewriteItem:
    def test_rewrite_creates_draft(self, g):
        hwpx = make_hwpx(paragraphs=["연구 목표"])
        form_doc = parse_hwpx(hwpx)
        target_id = form_doc.items[0].item_id

        state = initial_state()
        state["current_intent"] = Intent.REWRITE_ITEM.value
        state["current_item_id"] = target_id
        # 기존 item_plans 없이 generator로 직행 (스텁은 idx=0 처리)
        from app.models import ItemPlan
        state["item_plans"] = [ItemPlan(item_id=target_id)]
        result = g.invoke(state, config=_cfg("t-rewrite"))
        assert target_id in result["drafts"]


# ── GraphState 초기값 ─────────────────────────

class TestInitialState:
    def test_initial_state_has_required_keys(self):
        s = initial_state()
        for key in ("form_doc", "material_bundle", "item_plans", "drafts",
                    "retry_counts", "approved_items", "errors",
                    "current_intent", "conversation_history"):
            assert key in s, f"키 없음: {key}"

    def test_initial_lists_empty(self):
        s = initial_state()
        assert s["item_plans"] == []
        assert s["approved_items"] == []
        assert s["errors"] == []
