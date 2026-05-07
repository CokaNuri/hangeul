"""HWPX 렌더러 단위 테스트."""

import io
import zipfile

import pytest

from app.parsers.hwpx_parser import parse_hwpx
from app.parsers.hwpx_renderer import inject_text, repack_hwpx, render_preview_markdown, PII_PLACEHOLDER
from tests.fixtures.hwpx_factory import make_hwpx


@pytest.fixture
def form_doc():
    hwpx = make_hwpx(
        paragraphs=["연구 목표", "여기에 내용을 입력하세요.", ""],
        table_rows=[["항목", "내용"], ["논문명", ""]],
    )
    return parse_hwpx(hwpx)


# ── inject_text ────────────────────────────────

def test_inject_sets_modified_sections(form_doc):
    item_id = form_doc.items[0].item_id
    inject_text(form_doc, item_id, "딥러닝 기반 NLP 연구")
    assert "modified_sections" in form_doc.metadata
    sec_path = item_id.split("::")[0]
    assert sec_path in form_doc.metadata["modified_sections"]


def test_inject_idempotent_key(form_doc):
    """같은 섹션에 두 번 inject해도 버퍼가 하나만 유지되어야 한다."""
    item0 = form_doc.items[0].item_id
    item1 = form_doc.items[1].item_id
    inject_text(form_doc, item0, "첫 번째 내용")
    inject_text(form_doc, item1, "두 번째 내용")
    sec_path = item0.split("::")[0]
    assert len(form_doc.metadata["modified_sections"]) == 1  # 같은 섹션


def test_inject_invalid_item_id_is_noop(form_doc):
    """잘못된 item_id는 조용히 무시한다."""
    inject_text(form_doc, "invalid-id-no-double-colon", "텍스트")
    assert form_doc.metadata.get("modified_sections", {}) == {}


# ── repack_hwpx ───────────────────────────────

def test_repack_returns_valid_zip(form_doc):
    item_id = form_doc.items[0].item_id
    inject_text(form_doc, item_id, "딥러닝 기반 NLP 연구")
    result = repack_hwpx(form_doc)
    assert isinstance(result, bytes)
    assert zipfile.is_zipfile(io.BytesIO(result))


def test_repack_contains_modified_text(form_doc):
    """재패키징된 ZIP의 섹션 XML에 삽입한 텍스트가 있어야 한다."""
    item_id = form_doc.items[0].item_id
    sec_path = item_id.split("::")[0]
    inject_text(form_doc, item_id, "딥러닝 기반 NLP 연구")
    result = repack_hwpx(form_doc)

    with zipfile.ZipFile(io.BytesIO(result)) as zf:
        section_xml = zf.read(sec_path).decode("utf-8")
    assert "딥러닝 기반 NLP 연구" in section_xml


def test_repack_preserves_other_files(form_doc):
    """수정하지 않은 파일(mimetype 등)은 원본 그대로여야 한다."""
    inject_text(form_doc, form_doc.items[0].item_id, "테스트")
    result = repack_hwpx(form_doc)

    with zipfile.ZipFile(io.BytesIO(result)) as zf:
        names = zf.namelist()
    assert "mimetype" in names
    assert "META-INF/container.xml" in names


def test_repack_without_inject_returns_valid_zip(form_doc):
    """inject 없이 repack해도 유효한 ZIP이어야 한다."""
    result = repack_hwpx(form_doc)
    assert zipfile.is_zipfile(io.BytesIO(result))


def test_repack_no_hwpx_bytes_raises():
    from app.models import FormDoc
    doc = FormDoc(raw_xml=b"", metadata={})
    with pytest.raises(ValueError, match="hwpx_bytes"):
        repack_hwpx(doc)


# ── render_preview_markdown ───────────────────

def test_preview_contains_item_id():
    md = render_preview_markdown("section0::p1", "연구 내용입니다.", [])
    assert "section0::p1" in md


def test_preview_contains_text():
    md = render_preview_markdown("section0::p1", "연구 내용입니다.", [])
    assert "연구 내용입니다." in md


def test_preview_contains_citations():
    md = render_preview_markdown("section0::p1", "텍스트", ["CV.pdf", "논문.docx"])
    assert "CV.pdf" in md
    assert "논문.docx" in md


def test_preview_no_citations():
    md = render_preview_markdown("section0::p1", "텍스트", [])
    assert "참고 자료" not in md
