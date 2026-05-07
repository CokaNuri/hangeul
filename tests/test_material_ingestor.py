"""MaterialIngestor 단위 테스트."""

import pytest

from app.models import MaterialDoc, MaterialBundle
from app.parsers.material_ingestor import ingest_file, build_material_bundle
from tests.fixtures.docx_factory import make_docx, make_minimal_pdf_bytes


# ── TXT 파싱 ──────────────────────────────────

class TestTxt:
    def test_basic_text(self):
        text = "본 연구는 딥러닝 기반 자연어 처리 연구입니다."
        doc = ingest_file(text.encode("utf-8"), "cv.txt")
        assert isinstance(doc, MaterialDoc)
        assert "딥러닝" in doc.masked_text

    def test_pii_masked_in_txt(self):
        text = "이메일: hong@example.com, 전화: 010-1234-5678"
        doc = ingest_file(text.encode("utf-8"), "contact.txt")
        assert "hong@example.com" not in doc.masked_text
        assert "010-1234-5678" not in doc.masked_text

    def test_source_name_preserved(self):
        doc = ingest_file(b"hello", "my_cv.txt")
        assert doc.source_name == "my_cv.txt"

    def test_doc_type_is_txt(self):
        doc = ingest_file(b"hello", "note.txt")
        assert doc.doc_type == "txt"

    def test_cp949_encoding(self):
        text = "연구 목표: 자연어 처리"
        doc = ingest_file(text.encode("cp949"), "korean.txt")
        assert "연구 목표" in doc.masked_text

    def test_summary_empty_after_ingest(self):
        """파서는 summary를 생성하지 않는다 — LLM 요약은 노드에서 수행."""
        doc = ingest_file(b"some content", "file.txt")
        assert doc.summary == ""

    def test_empty_file_summary_empty(self):
        doc = ingest_file(b"   ", "empty.txt")
        assert doc.summary == ""


# ── DOCX 파싱 ─────────────────────────────────

class TestDocx:
    @pytest.fixture
    def cv_docx(self):
        return make_docx(
            paragraphs=[
                "홍길동 (연구원)",
                "연구 분야: 자연어 처리, 딥러닝",
                "이메일: hong@lab.ac.kr",
                "전화: 010-9999-8888",
            ],
            table_rows=[
                ["논문명", "저자", "년도"],
                ["딥러닝 기반 NLP", "홍길동", "2023"],
            ],
        )

    def test_text_extracted(self, cv_docx):
        doc = ingest_file(cv_docx, "cv.docx")
        assert "연구 분야" in doc.masked_text

    def test_table_extracted(self, cv_docx):
        doc = ingest_file(cv_docx, "cv.docx")
        assert "논문명" in doc.masked_text

    def test_pii_masked(self, cv_docx):
        doc = ingest_file(cv_docx, "cv.docx")
        assert "hong@lab.ac.kr" not in doc.masked_text
        assert "010-9999-8888" not in doc.masked_text

    def test_doc_type(self, cv_docx):
        doc = ingest_file(cv_docx, "cv.docx")
        assert doc.doc_type == "docx"


# ── PDF 파싱 ──────────────────────────────────

class TestPdf:
    def test_minimal_pdf_no_crash(self):
        """최소 PDF는 크래시 없이 빈 텍스트를 반환해야 한다."""
        pdf_bytes = make_minimal_pdf_bytes()
        doc = ingest_file(pdf_bytes, "empty.pdf")
        assert isinstance(doc, MaterialDoc)
        assert doc.doc_type == "pdf"

    def test_bad_pdf_returns_empty_text(self):
        """손상된 PDF는 소프트 실패 — 빈 텍스트 반환."""
        doc = ingest_file(b"not a pdf", "bad.pdf")
        assert doc.masked_text == ""


# ── 지원하지 않는 확장자 ──────────────────────

class TestUnsupported:
    def test_unsupported_ext_raises(self):
        with pytest.raises(ValueError, match="지원하지 않는"):
            ingest_file(b"data", "file.xlsx")


# ── build_material_bundle ─────────────────────

class TestBundle:
    def test_multiple_files(self):
        files = [
            ("연구 목표입니다.".encode("utf-8"), "goal.txt"),
            (make_docx(["연구 내용"]), "content.docx"),
        ]
        bundle = build_material_bundle(files)
        assert isinstance(bundle, MaterialBundle)
        assert len(bundle.docs) == 2

    def test_source_names_preserved(self):
        files = [
            (b"text1", "a.txt"),
            (b"text2", "b.txt"),
        ]
        bundle = build_material_bundle(files)
        names = [d.source_name for d in bundle.docs]
        assert "a.txt" in names
        assert "b.txt" in names

    def test_one_failure_does_not_stop_others(self):
        """한 파일 파싱 실패 시 나머지는 계속 처리돼야 한다."""
        files = [
            (b"good text", "good.txt"),
            (b"data", "bad.xlsx"),      # 지원 안 함 → 소프트 실패
            (b"more good", "good2.txt"),
        ]
        bundle = build_material_bundle(files)
        assert len(bundle.docs) == 2  # xlsx 제외

    def test_all_masked_texts_no_pii(self):
        pii_text = "연락처: hong@test.com 전화: 010-0000-1111"
        files = [(pii_text.encode("utf-8"), "pii.txt")]
        bundle = build_material_bundle(files)
        for doc in bundle.docs:
            assert "hong@test.com" not in doc.masked_text
            assert "010-0000-1111" not in doc.masked_text

    def test_empty_file_list(self):
        bundle = build_material_bundle([])
        assert bundle.docs == []
