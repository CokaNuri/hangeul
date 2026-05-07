"""pytest 공통 픽스처."""

import pytest


@pytest.fixture
def sample_text_with_pii() -> str:
    return (
        "홍길동(주민번호: 900101-1234567)은 서울특별시 강남구에 거주하며 "
        "연락처는 010-1234-5678, 이메일은 hong@example.com 입니다. "
        "계좌번호는 123-456789-01-234 입니다."
    )


@pytest.fixture
def sample_clean_text() -> str:
    return "본 연구는 딥러닝 기반 자연어 처리 기법을 활용하여 한국어 문서 분류 성능을 향상시키는 것을 목표로 합니다."
