"""통합 PII 마스킹 진입점 — Step 12.

regex (1차) → Presidio (2차) 순서로 마스킹을 적용한다.
이 모듈의 mask() 함수가 프로젝트 전체의 단일 PII 마스킹 진입점이다.

사용법:
    from app.pii.masker import mask

    result = mask(text)
    # result.masked_text → LLM에 전달할 안전한 텍스트
    # result.mask_map    → 토큰→원본 매핑 (서버 메모리에만 보관)
"""
from __future__ import annotations

from app.pii.regex_masker import mask_text, MaskResult, scan_for_pii


def mask(text: str) -> MaskResult:
    """regex + Presidio 순차 마스킹 — 단일 진입점.

    Steps:
        1. 정규식 마스킹 (주민번호, 전화, 이메일, 계좌, 카드 등)
        2. Presidio 마스킹 (이름, 학번, 주소, 소속기관)

    Args:
        text: 원본 텍스트 (사용자 업로드 파일에서 추출한 텍스트)

    Returns:
        MaskResult — masked_text에 두 단계 마스킹이 모두 적용됨
    """
    # 1차: 정규식 마스킹
    step1 = mask_text(text)

    # 2차: Presidio 마스킹 (1차 결과에 적용해 잔류 PII를 추가 차단)
    from app.pii.presidio_masker import mask_presidio
    step2 = mask_presidio(step1.masked_text)

    return MaskResult(
        masked_text=step2.masked_text,
        mask_map={**step1.mask_map, **step2.mask_map},
    )


# scan_for_pii는 regex 기반 — re-export해 기존 호출 코드가 변경 없이 동작하도록
__all__ = ["mask", "scan_for_pii", "MaskResult"]
