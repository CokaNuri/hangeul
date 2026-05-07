"""HwpAgent LangGraph 그래프 — Step 8 스켈레톤.

노드는 모두 스텁 구현이다. Step 10~17에서 단계별로 실제 로직으로 교체한다.

그래프 흐름:
  START → router
  router --start_fill / rewrite / change_tone--> form_parser → material_ingestor → planner → item_loop
  router --upload_form--> form_parser → END
  router --upload_material--> material_ingestor → END
  router --add_material--> material_ingestor → planner → END
  router --general_qa / disambiguation--> END

  item_loop (conditional):
    all_done          → END
    needs_question    → question → generator → verifier
    else              → generator → verifier

  verifier (conditional):
    ok                → renderer → item_loop (loop)
    retry (< max)     → generator (재시도)
    soft_fail         → renderer → item_loop

  renderer → item_loop (loop back)
"""

from __future__ import annotations

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.graph.nodes.form_parser import form_parser_node
from app.graph.nodes.generator import generator_node
from app.graph.nodes.material_ingestor import material_ingestor_node
from app.graph.nodes.planner import planner_node
from app.graph.nodes.question import question_node
from app.graph.nodes.renderer import renderer_node
from app.graph.nodes.router import router_node
from app.graph.nodes.verifier import verifier_node, VERIFIER_RESULT_KEY
from app.graph.state import GraphState
from app.models import Intent

logger = logging.getLogger(__name__)

_MAX_RETRIES = settings.verifier_max_retries


# ──────────────────────────────────────────────
# 조건부 엣지 함수
# ──────────────────────────────────────────────

def _route_intent(state: GraphState) -> str:
    """Router 노드 이후 인텐트에 따라 다음 노드를 결정한다."""
    intent = state.get("current_intent", Intent.GENERAL_QA.value)
    routes = {
        Intent.UPLOAD_FORM.value:      "form_parser",
        Intent.UPLOAD_MATERIAL.value:  "material_ingestor",
        Intent.START_FILL.value:       "form_parser",
        Intent.REWRITE_ITEM.value:     "generator",
        Intent.CHANGE_TONE.value:      "generator",
        Intent.ADD_MATERIAL.value:     "material_ingestor",
        Intent.GENERAL_QA.value:       END,
        Intent.DISAMBIGUATION.value:   END,
    }
    return routes.get(intent, END)


def _route_after_form_parser(state: GraphState) -> str:
    """FormParser 이후 — start_fill이면 material_ingestor, upload_form이면 END."""
    intent = state.get("current_intent", "")
    if intent == Intent.START_FILL.value:
        return "material_ingestor"
    return END


def _route_after_material_ingestor(state: GraphState) -> str:
    """MaterialIngestor 이후 — start_fill / add_material이면 planner, 아니면 END."""
    intent = state.get("current_intent", "")
    if intent in (Intent.START_FILL.value, Intent.ADD_MATERIAL.value):
        return "planner"
    return END


def _route_item_loop(state: GraphState) -> str:
    """Planner / Renderer 이후 다음 항목 처리 여부를 결정한다."""
    plans = state.get("item_plans", [])
    idx = state.get("current_item_index", 0)

    if idx >= len(plans):
        logger.debug("[ItemLoop] 모든 항목 처리 완료 → END")
        return END

    plan = plans[idx]
    if plan.needs_question and not state.get("user_answer"):
        logger.debug("[ItemLoop] item_id=%s → question", plan.item_id)
        return "question"

    logger.debug("[ItemLoop] item_id=%s → generator", plan.item_id)
    return "generator"


def _route_verifier(state: GraphState) -> str:
    """Verifier 결과에 따라 renderer(ok/soft_fail) 또는 generator(retry)로 분기한다."""
    verdict = state.get(VERIFIER_RESULT_KEY, "ok")

    if verdict == "retry":
        plans = state.get("item_plans", [])
        idx = state.get("current_item_index", 0)
        item_id = plans[idx].item_id if idx < len(plans) else ""
        retries = state.get("retry_counts", {}).get(item_id, 0)
        if retries < _MAX_RETRIES:
            logger.debug("[Verifier] retry %d/%d → generator", retries + 1, _MAX_RETRIES)
            return "generator"
        logger.debug("[Verifier] soft_fail (retry 한계 초과) → renderer")

    return "renderer"


# ──────────────────────────────────────────────
# 그래프 조립
# ──────────────────────────────────────────────

def build_graph(checkpointer=None):
    """컴파일된 LangGraph 그래프를 반환한다.

    Args:
        checkpointer: LangGraph 체크포인터. None이면 MemorySaver 사용.
    """
    g = StateGraph(GraphState)

    # ── 노드 등록 ─────────────────────────────
    g.add_node("router",             router_node)
    g.add_node("form_parser",        form_parser_node)
    g.add_node("material_ingestor",  material_ingestor_node)
    g.add_node("planner",            planner_node)
    g.add_node("generator",          generator_node)
    g.add_node("verifier",           verifier_node)
    g.add_node("question",           question_node)
    g.add_node("renderer",           renderer_node)

    # ── 진입점 ───────────────────────────────
    g.add_edge(START, "router")

    # ── Router → 다음 노드 (인텐트 기반) ──────
    g.add_conditional_edges(
        "router",
        _route_intent,
        {
            "form_parser":        "form_parser",
            "material_ingestor":  "material_ingestor",
            "generator":          "generator",
            END:                  END,
        },
    )

    # ── FormParser 이후 ───────────────────────
    g.add_conditional_edges(
        "form_parser",
        _route_after_form_parser,
        {"material_ingestor": "material_ingestor", END: END},
    )

    # ── MaterialIngestor 이후 ─────────────────
    g.add_conditional_edges(
        "material_ingestor",
        _route_after_material_ingestor,
        {"planner": "planner", END: END},
    )

    # ── Planner → item_loop (첫 항목 진입) ────
    g.add_conditional_edges(
        "planner",
        _route_item_loop,
        {"question": "question", "generator": "generator", END: END},
    )

    # ── Question → Generator ──────────────────
    g.add_edge("question", "generator")

    # ── Generator → Verifier ─────────────────
    g.add_edge("generator", "verifier")

    # ── Verifier → Generator(retry) or Renderer
    g.add_conditional_edges(
        "verifier",
        _route_verifier,
        {"generator": "generator", "renderer": "renderer"},
    )

    # ── Renderer → item_loop (다음 항목 or END)
    g.add_conditional_edges(
        "renderer",
        _route_item_loop,
        {"question": "question", "generator": "generator", END: END},
    )

    cp = checkpointer or MemorySaver()
    return g.compile(checkpointer=cp)


# 전역 싱글턴 (FastAPI에서 임포트해서 사용)
graph = build_graph()


# ──────────────────────────────────────────────
# CLI 검증용 __main__
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from app.graph.state import initial_state
    from app.models import Intent

    print("=== 그래프 스켈레톤 실행 테스트 ===")

    config = {"configurable": {"thread_id": "test-run-1"}}

    # 시나리오 1: general_qa — END 직행
    state = initial_state()
    state["current_intent"] = Intent.GENERAL_QA.value
    result = graph.invoke(state, config={"configurable": {"thread_id": "test-qa"}})
    print(f"[general_qa] approved={result.get('approved_items')} errors={result.get('errors')}")

    # 시나리오 2: start_fill with dummy FormDoc
    from tests.fixtures.hwpx_factory import make_hwpx
    from app.parsers.hwpx_parser import parse_hwpx

    hwpx = make_hwpx(paragraphs=["연구 목표", "연구 내용"], table_rows=None)
    form_doc = parse_hwpx(hwpx)

    state2 = initial_state()
    state2["current_intent"] = Intent.START_FILL.value
    state2["form_doc"] = form_doc
    result2 = graph.invoke(state2, config={"configurable": {"thread_id": "test-fill"}})
    approved = result2.get("approved_items", [])
    print(f"[start_fill] approved={len(approved)}개, errors={result2.get('errors')}")
    print("OK: 그래프 E2E 통과")
