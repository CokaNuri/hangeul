"""LangGraph GraphState 정의 — Step 8.

TypedDict로 그래프 전체에서 공유하는 상태를 정의한다.
각 노드는 변경된 키만 포함하는 딕셔너리를 반환하면 LangGraph가 병합한다.

리스트 필드(item_plans, errors 등)는 reducer 없이 사용 — 노드가 전체 리스트를 반환한다.
"""

from __future__ import annotations

from typing import TypedDict

from app.models import DraftItem, FormDoc, Intent, ItemPlan, MaterialBundle


class GraphState(TypedDict, total=False):
    # ── 양식 / 자료 ──────────────────────────────
    form_doc: FormDoc | None
    material_bundle: MaterialBundle | None

    # ── 계획 / 초안 ──────────────────────────────
    item_plans: list[ItemPlan]
    drafts: dict[str, DraftItem]       # item_id → DraftItem
    retry_counts: dict[str, int]       # item_id → 재시도 횟수
    approved_items: list[str]          # 승인된 item_id 목록

    # ── 현재 처리 중인 항목 (item 순회용) ──────────
    current_item_index: int            # item_plans 인덱스

    # ── 라우팅 ──────────────────────────────────
    current_intent: str                # Intent enum value
    current_item_id: str | None        # rewrite_item / change_tone 대상
    target_tone: str                   # change_tone 시 사용

    # ── 대화 ────────────────────────────────────
    conversation_history: list[dict[str, str]]
    pending_question: str              # Question 노드가 설정
    user_answer: str                   # interrupt 재개 시 채워짐

    # ── Verifier 결과 ────────────────────────────
    verifier_verdict: str              # "ok" | "retry" — _route_verifier가 읽음

    # ── 에러 ────────────────────────────────────
    errors: list[str]


def initial_state() -> GraphState:
    """빈 GraphState 초기값을 반환한다."""
    return GraphState(
        form_doc=None,
        material_bundle=None,
        item_plans=[],
        drafts={},
        retry_counts={},
        approved_items=[],
        current_item_index=0,
        current_intent=Intent.GENERAL_QA.value,
        current_item_id=None,
        target_tone="공식적",
        conversation_history=[],
        pending_question="",
        user_answer="",
        verifier_verdict="ok",
        errors=[],
    )
