"""통합 PII 마스킹 테스트 — Step 12.

masker.mask()가 regex + Presidio 두 단계를 순서대로 적용하는지 검증한다.
기존 regex 테스트(test_pii_regex.py)는 별도로 실행돼야 계속 통과해야 한다.
"""
from __future__ import annotations

import pytest

from app.pii.masker import mask, MaskResult
from app.pii.presidio_masker import mask_presidio


# ── 통합 마스킹 기본 동작 ─────────────────────

def test_returns_mask_result():
    result = mask("연구 목표: 자연어 처리")
    assert isinstance(result, MaskResult)


def test_clean_text_unchanged():
    text = "본 연구는 딥러닝 기반 자연어 처리 연구입니다."
    result = mask(text)
    assert result.masked_text == text
    assert not result.has_pii


# ── 1차: 정규식 PII가 통합 마스킹에서도 제거되는지 ─

def test_email_masked_via_combined(sample_text_with_pii):
    result = mask(sample_text_with_pii)
    assert "hong@example.com" not in result.masked_text


def test_phone_masked_via_combined(sample_text_with_pii):
    result = mask(sample_text_with_pii)
    assert "010-1234-5678" not in result.masked_text


def test_rrn_masked_via_combined(sample_text_with_pii):
    result = mask(sample_text_with_pii)
    assert "900101-1234567" not in result.masked_text


def test_account_masked_via_combined(sample_text_with_pii):
    result = mask(sample_text_with_pii)
    assert "123-456789-01-234" not in result.masked_text


def test_combined_mask_map_contains_regex_tokens(sample_text_with_pii):
    result = mask(sample_text_with_pii)
    assert result.has_pii


# ── 2차: Presidio — 한국 이름 마스킹 ─────────

def test_korean_name_masked():
    """필드 레이블(신청자:) 뒤 성명이 마스킹돼야 한다."""
    result = mask_presidio("신청자: 홍길동 입니다.")
    assert "홍길동" not in result.masked_text
    assert result.has_pii


def test_korean_name_2char_masked():
    """2자 성명도 레이블 뒤에서 마스킹돼야 한다."""
    result = mask_presidio("저자: 김민")
    assert "김민" not in result.masked_text


def test_korean_name_in_combined():
    """통합 마스킹에서도 레이블 뒤 이름이 제거돼야 한다."""
    result = mask("책임자: 박철수 교수")
    assert "박철수" not in result.masked_text


# ── 2차: Presidio — 학번 마스킹 ──────────────

def test_student_id_masked():
    """학번 형식 숫자가 마스킹돼야 한다."""
    result = mask_presidio("학번: 2021123456")
    assert "2021123456" not in result.masked_text
    assert result.has_pii


def test_student_id_with_hyphen_masked():
    """하이픈 포함 학번도 마스킹돼야 한다."""
    result = mask_presidio("학번 2021-12345")
    assert "2021-12345" not in result.masked_text


def test_student_id_in_combined():
    result = mask("학번: 1998123456 학생의 논문")
    assert "1998123456" not in result.masked_text


# ── 2차: Presidio — 주소 마스킹 ──────────────

def test_korean_address_masked():
    """시도·구 형식 주소가 마스킹돼야 한다."""
    result = mask_presidio("주소: 서울특별시 강남구 테헤란로")
    assert "서울특별시 강남구" not in result.masked_text
    assert result.has_pii


def test_provincial_address_masked():
    result = mask_presidio("경기도 수원시 영통구에 거주")
    assert "경기도 수원시" not in result.masked_text


def test_address_in_combined():
    result = mask("거주지: 부산광역시 해운대구 우동")
    assert "부산광역시 해운대구" not in result.masked_text


# ── 2차: Presidio — 소속기관 마스킹 ──────────

def test_org_masked():
    """소속 레이블 뒤 기관명이 마스킹돼야 한다."""
    result = mask_presidio("소속: 한국과학기술연구원")
    assert "한국과학기술연구원" not in result.masked_text
    assert result.has_pii


def test_university_masked():
    """재직 레이블 뒤 대학교명이 마스킹돼야 한다."""
    result = mask_presidio("재직: 서울대학교")
    assert "서울대학교" not in result.masked_text


# ── mask_map 병합 ─────────────────────────────

def test_combined_mask_map_has_both_regex_and_presidio():
    """regex와 Presidio 토큰이 모두 mask_map에 있어야 한다."""
    text = "성명: 홍길동  연락처: hong@example.com"
    result = mask(text)
    tokens = set(result.mask_map.keys())
    # EMAIL은 regex가 잡음
    email_tokens = [t for t in tokens if "EMAIL" in t]
    assert email_tokens, "이메일 토큰 없음"
    # 이름은 Presidio가 잡음 (성명: 레이블 필요)
    name_tokens = [t for t in tokens if "KR_NAME" in t]
    assert name_tokens, "이름 토큰 없음"


# ── 마스킹 토큰 자체가 재마스킹되지 않아야 함 ──

def test_already_masked_token_not_remasked():
    """regex 마스킹 토큰([PHONE_0] 등)이 Presidio에서 재마스킹되지 않아야 한다."""
    text_with_token = "연락처: [PHONE_0]"
    result = mask_presidio(text_with_token)
    # 토큰이 그대로 남아 있어야 함 (재마스킹 X)
    assert "[PHONE_0]" in result.masked_text


# ── Presidio 소프트 실패 ──────────────────────

def test_presidio_returns_mask_result_on_empty():
    result = mask_presidio("")
    assert isinstance(result, MaskResult)
    assert result.masked_text == ""


def test_presidio_does_not_crash_on_pure_number():
    result = mask_presidio("12345")
    assert isinstance(result, MaskResult)
