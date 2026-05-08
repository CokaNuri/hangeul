"""Renderer 노드 테스트 — Step 17.

XML 삽입 로직과 노드 동작을 검증한다.
"""
from __future__ import annotations

import io
import zipfile

import pytest
from lxml import etree

from app.graph.nodes.renderer import renderer_node, _insert_text, _find_para, _find_cell
from app.models import FormDoc, FormItem, ItemPlan, ItemType, DraftItem, MaterialBundle
from tests.fixtures.hwpx_factory import make_hwpx, make_section_xml


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────

HP_NS = "http://www.hancom.co.kr/hwpml/2011/paragraph"


def _make_form_item(item_id: str, xml_path: str, label: str = "연구 목표") -> FormItem:
    return FormItem(item_id=item_id, label=label, item_type=ItemType.TEXT, xml_path=xml_path)


def _make_form_doc(hwpx: bytes, *items: FormItem) -> FormDoc:
    return FormDoc(
        raw_xml=b"",
        items=list(items),
        metadata={"hwpx_bytes": hwpx, "section_paths": ["Contents/section0.xml"]},
    )


def _make_plan(item_id: str) -> ItemPlan:
    return ItemPlan(item_id=item_id, confidence=0.9)


def _make_draft(item_id: str, text: str = "초안 텍스트") -> DraftItem:
    return DraftItem(item_id=item_id, text=text, citations=[])


def _read_section_xml(hwpx: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(hwpx), "r") as zf:
        return zf.read("Contents/section0.xml")


def _para_texts(hwpx: bytes) -> list[str]:
    """HWPX에서 모든 최상위 단락 텍스트를 추출한다."""
    sec_xml = _read_section_xml(hwpx)
    root = etree.fromstring(sec_xml)
    texts = []
    for p in root.iter(f"{{{HP_NS}}}p"):
        parent = p.getparent()
        if parent is not None and parent.tag == f"{{{HP_NS}}}tc":
            continue
        parts = [t.text or "" for t in p.iter(f"{{{HP_NS}}}t") if t.text]
        texts.append(" ".join(parts).strip())
    return texts


def _cell_text(hwpx: bytes, tbl_idx: int, row_i: int, col_i: int) -> str:
    sec_xml = _read_section_xml(hwpx)
    root = etree.fromstring(sec_xml)
    tbls = list(root.iter(f"{{{HP_NS}}}tbl"))
    tr = tbls[tbl_idx].findall(f"{{{HP_NS}}}tr")[row_i]
    tc = tr.findall(f"{{{HP_NS}}}tc")[col_i]
    parts = [t.text or "" for t in tc.iter(f"{{{HP_NS}}}t") if t.text]
    return " ".join(parts).strip()


# ──────────────────────────────────────────────
# _insert_text 단위 테스트
# ──────────────────────────────────────────────

def test_insert_text_paragraph():
    """단락 빈칸에 텍스트를 삽입해야 한다."""
    hwpx = make_hwpx(paragraphs=["레이블", ""])  # p0: 레이블, p1: 빈칸
    path = "Contents/section0.xml::p1"
    result = _insert_text(hwpx, path, "삽입된 초안")
    texts = _para_texts(result)
    assert texts[1] == "삽입된 초안"


def test_insert_text_preserves_other_paragraphs():
    hwpx = make_hwpx(paragraphs=["레이블", "", "다른 단락"])
    path = "Contents/section0.xml::p1"
    result = _insert_text(hwpx, path, "새 텍스트")
    texts = _para_texts(result)
    assert texts[0] == "레이블"
    assert texts[1] == "새 텍스트"
    assert texts[2] == "다른 단락"


def test_insert_text_replaces_existing_text():
    """이미 텍스트가 있는 단락도 교체할 수 있어야 한다."""
    hwpx = make_hwpx(paragraphs=["기존 텍스트"])
    path = "Contents/section0.xml::p0"
    result = _insert_text(hwpx, path, "교체된 텍스트")
    texts = _para_texts(result)
    assert texts[0] == "교체된 텍스트"


def test_insert_text_table_cell():
    """표 셀에 텍스트를 삽입해야 한다."""
    hwpx = make_hwpx(
        paragraphs=[],
        table_rows=[["헤더A", "헤더B"], ["행1", ""]],
    )
    path = "Contents/section0.xml::tbl0::r1c1"
    result = _insert_text(hwpx, path, "셀 초안")
    assert _cell_text(result, 0, 1, 1) == "셀 초안"


def test_insert_text_preserves_zip_structure():
    """삽입 후 HWPX ZIP 구조가 유지되어야 한다."""
    hwpx = make_hwpx(paragraphs=["레이블", ""])
    result = _insert_text(hwpx, "Contents/section0.xml::p1", "텍스트")
    with zipfile.ZipFile(io.BytesIO(result), "r") as zf:
        names = set(zf.namelist())
    assert "Contents/section0.xml" in names
    assert "mimetype" in names


def test_insert_text_invalid_path_raises():
    hwpx = make_hwpx(paragraphs=[""])
    with pytest.raises(ValueError):
        _insert_text(hwpx, "Contents/section0.xml", "텍스트")  # :: 없음


def test_insert_text_invalid_section_raises():
    hwpx = make_hwpx(paragraphs=[""])
    with pytest.raises(ValueError):
        _insert_text(hwpx, "Contents/nofile.xml::p0", "텍스트")


def test_insert_text_out_of_range_para_raises():
    hwpx = make_hwpx(paragraphs=["단락"])
    with pytest.raises(ValueError):
        _insert_text(hwpx, "Contents/section0.xml::p99", "텍스트")


# ──────────────────────────────────────────────
# renderer_node 노드 테스트
# ──────────────────────────────────────────────

def test_renderer_advances_index():
    hwpx = make_hwpx(paragraphs=["레이블", ""])
    form_doc = _make_form_doc(hwpx, _make_form_item("q1", "Contents/section0.xml::p1"))
    state = {
        "item_plans": [_make_plan("q1")],
        "current_item_index": 0,
        "form_doc": form_doc,
        "drafts": {"q1": _make_draft("q1", "초안 텍스트")},
        "approved_items": [],
    }
    updates = renderer_node(state)
    assert updates["current_item_index"] == 1


def test_renderer_adds_to_approved():
    hwpx = make_hwpx(paragraphs=["레이블", ""])
    form_doc = _make_form_doc(hwpx, _make_form_item("q1", "Contents/section0.xml::p1"))
    state = {
        "item_plans": [_make_plan("q1")],
        "current_item_index": 0,
        "form_doc": form_doc,
        "drafts": {"q1": _make_draft("q1", "초안")},
        "approved_items": [],
    }
    updates = renderer_node(state)
    assert "q1" in updates["approved_items"]


def test_renderer_inserts_text_into_hwpx():
    hwpx = make_hwpx(paragraphs=["레이블", ""])
    form_doc = _make_form_doc(hwpx, _make_form_item("q1", "Contents/section0.xml::p1"))
    state = {
        "item_plans": [_make_plan("q1")],
        "current_item_index": 0,
        "form_doc": form_doc,
        "drafts": {"q1": _make_draft("q1", "삽입된 초안 텍스트")},
        "approved_items": [],
    }
    updates = renderer_node(state)
    assert "form_doc" in updates
    new_hwpx = updates["form_doc"].metadata["hwpx_bytes"]
    texts = _para_texts(new_hwpx)
    assert texts[1] == "삽입된 초안 텍스트"


def test_renderer_no_draft_still_advances():
    hwpx = make_hwpx(paragraphs=["레이블", ""])
    form_doc = _make_form_doc(hwpx, _make_form_item("q1", "Contents/section0.xml::p1"))
    state = {
        "item_plans": [_make_plan("q1")],
        "current_item_index": 0,
        "form_doc": form_doc,
        "drafts": {},  # 초안 없음
        "approved_items": [],
    }
    updates = renderer_node(state)
    assert updates["current_item_index"] == 1
    assert "q1" in updates["approved_items"]
    assert "form_doc" not in updates  # XML 삽입은 안 됨


def test_renderer_no_form_doc_still_advances():
    state = {
        "item_plans": [_make_plan("q1")],
        "current_item_index": 0,
        "form_doc": None,
        "drafts": {"q1": _make_draft("q1", "초안")},
        "approved_items": [],
    }
    updates = renderer_node(state)
    assert updates["current_item_index"] == 1


def test_renderer_index_out_of_range_returns_empty():
    state = {
        "item_plans": [_make_plan("q1")],
        "current_item_index": 5,
    }
    updates = renderer_node(state)
    assert updates == {}


def test_renderer_multiple_items_sequential():
    """두 항목을 순서대로 렌더링하면 둘 다 HWPX에 삽입되어야 한다."""
    hwpx = make_hwpx(paragraphs=["레이블1", "", "레이블2", ""])
    form_doc = _make_form_doc(
        hwpx,
        _make_form_item("q1", "Contents/section0.xml::p1", "레이블1"),
        _make_form_item("q2", "Contents/section0.xml::p3", "레이블2"),
    )
    state1 = {
        "item_plans": [_make_plan("q1"), _make_plan("q2")],
        "current_item_index": 0,
        "form_doc": form_doc,
        "drafts": {"q1": _make_draft("q1", "첫 번째 초안"), "q2": _make_draft("q2", "두 번째 초안")},
        "approved_items": [],
    }
    u1 = renderer_node(state1)

    # 두 번째 항목 렌더링 (업데이트된 form_doc, index=1)
    state2 = {**state1, "form_doc": u1["form_doc"], "current_item_index": 1, "approved_items": u1["approved_items"]}
    u2 = renderer_node(state2)

    final_hwpx = u2["form_doc"].metadata["hwpx_bytes"]
    texts = _para_texts(final_hwpx)
    assert texts[1] == "첫 번째 초안"
    assert texts[3] == "두 번째 초안"


def test_renderer_preserves_existing_approved_items():
    hwpx = make_hwpx(paragraphs=["레이블", ""])
    form_doc = _make_form_doc(hwpx, _make_form_item("q2", "Contents/section0.xml::p1"))
    state = {
        "item_plans": [_make_plan("q2")],
        "current_item_index": 0,
        "form_doc": form_doc,
        "drafts": {"q2": _make_draft("q2", "초안")},
        "approved_items": ["q1"],  # 이미 승인된 항목
    }
    updates = renderer_node(state)
    assert "q1" in updates["approved_items"]
    assert "q2" in updates["approved_items"]
