"""FormParser 노드 단위 테스트 — Step 10."""

import pytest

from app.graph.nodes.form_parser import form_parser_node
from app.graph.state import initial_state
from app.models import FormDoc, ItemType
from app.parsers.hwpx_parser import parse_hwpx, _is_blank_text
from tests.fixtures.hwpx_factory import make_hwpx


# ── 픽스처 ────────────────────────────────────

@pytest.fixture
def blank_para_hwpx() -> bytes:
    """빈 단락이 포함된 양식 — 연구 목표 레이블 + 빈칸."""
    return make_hwpx(
        paragraphs=["연구 목표", ""],
        table_rows=None,
    )


@pytest.fixture
def pii_hwpx() -> bytes:
    """PII 레이블 + 빈칸 조합."""
    return make_hwpx(
        paragraphs=["성명", "", "주민등록번호", "", "연락처", ""],
        table_rows=None,
    )


@pytest.fixture
def bracket_hwpx() -> bytes:
    """괄호 빈칸 패턴이 포함된 양식."""
    return make_hwpx(
        paragraphs=["연구 기간", "[   ]", "연구비 총액", "(   )"],
        table_rows=None,
    )


@pytest.fixture
def table_hwpx() -> bytes:
    """빈 표 셀이 포함된 양식."""
    return make_hwpx(
        paragraphs=[],
        table_rows=[
            ["항목", "내용", "비고"],
            ["논문명", "", ""],
            ["게재연도", "", ""],
        ],
    )


@pytest.fixture
def mixed_hwpx() -> bytes:
    """빈 단락 + 표 + PII 혼합."""
    return make_hwpx(
        paragraphs=["연구 목표", "", "성명", ""],
        table_rows=[
            ["구분", "내용"],
            ["논문명", ""],
        ],
    )


# ── _is_blank_text 단위 테스트 ───────────────────

def test_empty_string_is_blank():
    assert _is_blank_text("") is True


def test_whitespace_is_blank():
    assert _is_blank_text("   ") is True
    assert _is_blank_text("\t\n") is True


def test_bracket_pattern_is_blank():
    assert _is_blank_text("[]") is True
    assert _is_blank_text("[   ]") is True
    assert _is_blank_text("(   )") is True
    assert _is_blank_text("□") is True
    assert _is_blank_text("___") is True


def test_text_is_not_blank():
    assert _is_blank_text("연구 목표") is False
    assert _is_blank_text("성명") is False
    assert _is_blank_text("[내용 있음]") is False


# ── 빈 단락 감지 ─────────────────────────────────

def test_blank_para_detected(blank_para_hwpx):
    """빈 단락이 FormItem으로 식별돼야 한다."""
    doc = parse_hwpx(blank_para_hwpx)
    assert len(doc.items) >= 1


def test_blank_para_label_from_prev(blank_para_hwpx):
    """빈칸의 레이블은 직전 단락 텍스트여야 한다."""
    doc = parse_hwpx(blank_para_hwpx)
    assert any("연구 목표" in it.label for it in doc.items)


def test_non_blank_para_not_in_items(blank_para_hwpx):
    """비어있지 않은 단락은 FormItem에 포함되지 않아야 한다."""
    doc = parse_hwpx(blank_para_hwpx)
    # "연구 목표" 텍스트가 label이 아닌 item_id에 있으면 안 됨
    assert not any(it.label == "연구 목표" and it.context == "" for it in doc.items)


# ── 괄호 빈칸 패턴 감지 ──────────────────────────

def test_bracket_pattern_detected(bracket_hwpx):
    """[ ] 패턴 단락이 빈칸으로 식별돼야 한다."""
    doc = parse_hwpx(bracket_hwpx)
    assert len(doc.items) >= 1


def test_bracket_label_from_prev(bracket_hwpx):
    """[ ] 빈칸의 레이블은 직전 단락이어야 한다."""
    doc = parse_hwpx(bracket_hwpx)
    labels = {it.label for it in doc.items}
    assert "연구 기간" in labels or "연구비 총액" in labels


# ── PII 분류 ─────────────────────────────────────

def test_pii_items_classified(pii_hwpx):
    """PII 키워드 레이블 뒤 빈칸은 ItemType.PII로 분류돼야 한다."""
    doc = parse_hwpx(pii_hwpx)
    pii_items = [it for it in doc.items if it.item_type == ItemType.PII]
    assert len(pii_items) >= 1


def test_pii_label_keywords(pii_hwpx):
    """성명·주민등록번호·연락처 레이블이 PII 아이템에 있어야 한다."""
    doc = parse_hwpx(pii_hwpx)
    pii_labels = {it.label for it in doc.items if it.item_type == ItemType.PII}
    assert any(kw in label for label in pii_labels for kw in ("성명", "주민등록번호", "연락처"))


def test_non_pii_not_classified_as_pii(blank_para_hwpx):
    """일반 레이블("연구 목표") 뒤 빈칸은 PII가 아니어야 한다."""
    doc = parse_hwpx(blank_para_hwpx)
    for it in doc.items:
        if "연구 목표" in it.label:
            assert it.item_type == ItemType.TEXT


# ── 표 빈 셀 감지 ────────────────────────────────

def test_blank_table_cells_detected(table_hwpx):
    """표의 빈 셀이 FormItem으로 식별돼야 한다."""
    doc = parse_hwpx(table_hwpx)
    assert len(doc.items) >= 1


def test_blank_cell_label_includes_headers(table_hwpx):
    """빈 셀 FormItem 레이블에 행 헤더 또는 열 헤더가 포함돼야 한다."""
    doc = parse_hwpx(table_hwpx)
    assert any("논문명" in it.label or "게재연도" in it.label for it in doc.items)


def test_header_cells_not_in_items(table_hwpx):
    """헤더 행 셀은 FormItem에 포함되지 않아야 한다."""
    doc = parse_hwpx(table_hwpx)
    # 헤더 행 텍스트("항목", "내용", "비고")가 단독 label로 나타나면 안 됨
    for it in doc.items:
        assert it.label not in ("항목", "내용", "비고")


# ── FormParser 노드 테스트 ────────────────────────

def test_node_updates_form_doc(mixed_hwpx):
    """form_parser_node가 form_doc을 재파싱해 state를 업데이트해야 한다."""
    doc = parse_hwpx(mixed_hwpx)
    state = initial_state()
    state["form_doc"] = doc

    result = form_parser_node(state)

    assert "form_doc" in result
    assert result["form_doc"] is not None


def test_node_blank_items_present(mixed_hwpx):
    """노드 실행 후 form_doc에 빈칸 항목이 ≥ 1개 있어야 한다."""
    doc = parse_hwpx(mixed_hwpx)
    state = initial_state()
    state["form_doc"] = doc

    result = form_parser_node(state)
    assert len(result["form_doc"].items) >= 1


def test_node_pii_classified(mixed_hwpx):
    """노드 실행 후 PII 항목이 type=PII로 분류돼야 한다."""
    doc = parse_hwpx(mixed_hwpx)
    state = initial_state()
    state["form_doc"] = doc

    result = form_parser_node(state)
    pii_items = [it for it in result["form_doc"].items if it.item_type == ItemType.PII]
    assert len(pii_items) >= 1


def test_node_no_form_doc_adds_error():
    """form_doc이 None이면 errors에 메시지가 추가돼야 한다."""
    state = initial_state()
    state["form_doc"] = None

    result = form_parser_node(state)

    assert "errors" in result
    assert len(result["errors"]) >= 1


def test_node_idempotent(mixed_hwpx):
    """노드를 두 번 실행해도 결과가 동일해야 한다 (idempotent)."""
    doc = parse_hwpx(mixed_hwpx)
    state = initial_state()
    state["form_doc"] = doc

    result1 = form_parser_node(state)
    state["form_doc"] = result1["form_doc"]
    result2 = form_parser_node(state)

    assert len(result1["form_doc"].items) == len(result2["form_doc"].items)
