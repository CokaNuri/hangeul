"""HWPX 파서 단위 테스트."""

import pytest

from app.models import ItemType
from app.parsers.hwpx_parser import parse_hwpx
from tests.fixtures.hwpx_factory import make_hwpx


@pytest.fixture
def simple_hwpx() -> bytes:
    return make_hwpx(
        paragraphs=["연구 목표", "연구 내용을 서술하세요.", "성명", ""],
        table_rows=[
            ["항목", "내용"],
            ["논문명", ""],
            ["게재연도", ""],
        ],
    )


@pytest.fixture
def empty_hwpx() -> bytes:
    return make_hwpx(paragraphs=[], table_rows=None)


# ── parse_hwpx 기본 동작 ──────────────────────

def test_returns_form_doc(simple_hwpx):
    from app.models import FormDoc
    doc = parse_hwpx(simple_hwpx)
    assert isinstance(doc, FormDoc)


def test_items_extracted(simple_hwpx):
    doc = parse_hwpx(simple_hwpx)
    assert len(doc.items) >= 1


def test_tables_extracted(simple_hwpx):
    doc = parse_hwpx(simple_hwpx)
    assert len(doc.tables) == 1
    tbl = doc.tables[0]
    assert tbl.header_row == ["항목", "내용"]
    assert len(tbl.data_rows) == 2


def test_table_data_rows_content(simple_hwpx):
    doc = parse_hwpx(simple_hwpx)
    tbl = doc.tables[0]
    assert tbl.data_rows[0] == ["논문명", ""]
    assert tbl.data_rows[1] == ["게재연도", ""]


def test_pii_item_classified(simple_hwpx):
    """'성명' 단락은 PII 타입으로 분류돼야 한다."""
    doc = parse_hwpx(simple_hwpx)
    pii_items = [i for i in doc.items if i.item_type == ItemType.PII]
    assert len(pii_items) >= 1
    assert any("성명" in i.label for i in pii_items)


def test_metadata_has_hwpx_bytes(simple_hwpx):
    doc = parse_hwpx(simple_hwpx)
    assert "hwpx_bytes" in doc.metadata
    assert doc.metadata["hwpx_bytes"] == simple_hwpx


def test_metadata_has_section_paths(simple_hwpx):
    doc = parse_hwpx(simple_hwpx)
    assert "section_paths" in doc.metadata
    assert len(doc.metadata["section_paths"]) >= 1


def test_raw_xml_is_bytes(simple_hwpx):
    doc = parse_hwpx(simple_hwpx)
    assert isinstance(doc.raw_xml, bytes)
    assert len(doc.raw_xml) > 0


def test_empty_hwpx_no_items(empty_hwpx):
    doc = parse_hwpx(empty_hwpx)
    assert doc.items == []
    assert doc.tables == []


# ── 에러 처리 ─────────────────────────────────

def test_bad_zip_raises_value_error():
    with pytest.raises(ValueError, match="ZIP 손상"):
        parse_hwpx(b"not a zip file at all")


def test_no_sections_raises_value_error():
    import io, zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("mimetype", b"application/hwp+zip")
    with pytest.raises(ValueError, match="섹션"):
        parse_hwpx(buf.getvalue())
