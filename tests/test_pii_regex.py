"""PII 정규식 마스킹 단위 테스트."""

import pytest

from app.pii.regex_masker import mask_text, scan_for_pii, MaskResult, _Tag


# ── 헬퍼 ──────────────────────────────────────

def _masked(text: str) -> str:
    return mask_text(text).masked_text

def _has_pii(text: str) -> bool:
    return mask_text(text).has_pii


# ── 주민등록번호 ──────────────────────────────

class TestRRN:
    def test_with_hyphen(self):
        result = mask_text("주민번호는 900101-1234567입니다.")
        assert "900101-1234567" not in result.masked_text
        assert f"[{_Tag.RRN}_0]" in result.masked_text

    def test_without_hyphen(self):
        result = mask_text("주민번호: 9001011234567")
        assert "9001011234567" not in result.masked_text

    def test_in_sentence(self):
        text = "홍길동(900101-1234567)은 서울 거주"
        assert "900101-1234567" not in _masked(text)

    def test_mask_map_stores_original(self):
        result = mask_text("주민번호 900101-1234567")
        assert "900101-1234567" in result.mask_map.values()

    def test_foreigner_rrn_5_prefix(self):
        # 외국인: 뒤 첫자리 5~8
        result = mask_text("외국인번호 900101-5234567")
        assert "900101-5234567" not in result.masked_text


# ── 전화번호 ──────────────────────────────────

class TestPhone:
    def test_mobile_with_hyphen(self):
        assert "010-1234-5678" not in _masked("연락처: 010-1234-5678")

    def test_mobile_without_hyphen(self):
        assert "01012345678" not in _masked("tel: 01012345678")

    def test_landline(self):
        assert "02-1234-5678" not in _masked("전화: 02-1234-5678")

    def test_area_code_031(self):
        assert "031-123-4567" not in _masked("031-123-4567 로 연락주세요")


# ── 이메일 ────────────────────────────────────

class TestEmail:
    def test_basic_email(self):
        assert "hong@example.com" not in _masked("이메일: hong@example.com")

    def test_email_with_dots(self):
        assert "hong.gil.dong@university.ac.kr" not in _masked(
            "hong.gil.dong@university.ac.kr 으로 보내세요"
        )

    def test_mask_map_key(self):
        result = mask_text("메일: test@test.com")
        assert f"[{_Tag.EMAIL}_0]" in result.mask_map


# ── 계좌번호 ──────────────────────────────────

class TestAccount:
    def test_standard_account(self):
        assert "110-123-456789" not in _masked("계좌: 110-123-456789")

    def test_long_account(self):
        assert "123-456789-01-234" not in _masked("계좌번호 123-456789-01-234")


# ── 카드번호 ──────────────────────────────────

class TestCard:
    def test_card_with_hyphen(self):
        assert "1234-5678-9012-3456" not in _masked("카드: 1234-5678-9012-3456")

    def test_card_with_space(self):
        assert "1234 5678 9012 3456" not in _masked("카드 1234 5678 9012 3456")


# ── 여권번호 ──────────────────────────────────

class TestPassport:
    def test_passport_single_letter(self):
        assert "M12345678" not in _masked("여권번호 M12345678")

    def test_passport_two_letters(self):
        assert "AB1234567" not in _masked("여권 AB1234567")


# ── 13자리 연속 숫자 ──────────────────────────

class TestNum13:
    def test_13digit_number(self):
        assert "1234567890123" not in _masked("번호: 1234567890123")

    def test_12digit_not_masked(self):
        # 12자리는 NUM13 패턴에 안 걸려야 함
        result = mask_text("번호: 123456789012")
        assert "[NUM13" not in result.masked_text


# ── 복합 텍스트 ───────────────────────────────

class TestCombined:
    def test_multiple_pii_types(self, sample_text_with_pii):
        """conftest.py의 복합 PII 텍스트: 주민번호+전화+이메일+계좌."""
        result = mask_text(sample_text_with_pii)
        assert "900101-1234567" not in result.masked_text
        assert "010-1234-5678" not in result.masked_text
        assert "hong@example.com" not in result.masked_text
        assert "123-456789-01-234" not in result.masked_text
        assert result.has_pii is True
        assert len(result.mask_map) >= 4

    def test_clean_text_unchanged(self, sample_clean_text):
        """PII 없는 학술 텍스트는 변경되지 않아야 한다."""
        result = mask_text(sample_clean_text)
        assert result.masked_text == sample_clean_text
        assert result.has_pii is False

    def test_multiple_same_type(self):
        text = "전화 010-1111-2222 또는 010-3333-4444"
        result = mask_text(text)
        assert "010-1111-2222" not in result.masked_text
        assert "010-3333-4444" not in result.masked_text
        # 토큰 인덱스가 각각 달라야 함
        assert f"[{_Tag.PHONE}_0]" in result.masked_text
        assert f"[{_Tag.PHONE}_1]" in result.masked_text


# ── scan_for_pii (3차 안전망용) ───────────────

class TestScanForPii:
    def test_detects_email(self):
        found = scan_for_pii("연락: foo@bar.com")
        assert _Tag.EMAIL in found

    def test_detects_phone(self):
        found = scan_for_pii("전화 010-1234-5678")
        assert _Tag.PHONE in found

    def test_clean_text_empty(self, sample_clean_text):
        assert scan_for_pii(sample_clean_text) == []

    def test_already_masked_token_not_detected(self):
        """마스킹 토큰 자체는 PII로 재탐지되지 않아야 한다."""
        masked = "[EMAIL_0] 으로 연락주세요"
        found = scan_for_pii(masked)
        assert _Tag.EMAIL not in found
