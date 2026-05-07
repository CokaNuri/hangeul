"""PII 정규식 마스킹 — Step 5 (1차 안전망).

LLM에 텍스트를 보내기 전에 반드시 이 모듈을 통과해야 한다.

커버리지:
  - 주민등록번호 / 외국인등록번호 (하이픈 있/없)
  - 여권번호
  - 계좌번호
  - 카드번호 (16자리)
  - 전화번호 (휴대폰·유선)
  - 이메일 주소
  - 13자리 연속 숫자 (주민번호 형식 임의 숫자열 포함)

보안 원칙:
  - mask_text()가 반환하는 masked_text에는 PII가 없어야 한다.
  - mask_map은 서버 메모리에만 존재하며 LLM에 전달되지 않는다.
  - 마스킹 토큰은 [PII_TYPE_N] 형식 (예: [PHONE_0], [EMAIL_1]).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ──────────────────────────────────────────────
# 마스킹 토큰 접두사
# ──────────────────────────────────────────────

class _Tag:
    RRN      = "RRN"       # 주민등록번호 (Resident Registration Number)
    FRN      = "FRN"       # 외국인등록번호
    PASSPORT = "PASSPORT"  # 여권번호
    ACCOUNT  = "ACCOUNT"   # 계좌번호
    CARD     = "CARD"      # 카드번호
    PHONE    = "PHONE"     # 전화번호
    EMAIL    = "EMAIL"     # 이메일
    NUM13    = "NUM13"     # 13자리 연속 숫자


# ──────────────────────────────────────────────
# 정규식 패턴 정의 (순서 중요: 더 구체적인 것이 먼저)
# ──────────────────────────────────────────────

_PATTERNS: list[tuple[str, re.Pattern]] = [
    # 1. 주민등록번호: 900101-1234567 (앞 6자리-뒤 7자리, 뒤 첫자리 1~4)
    #    ※ \b 대신 (?<!\d)/(?!\d) 사용: Python 3에서 \b는 한글도 \w로 취급
    (_Tag.RRN, re.compile(
        r"(?<!\d)\d{6}"         # 생년월일 6자리
        r"[-\s]?"               # 하이픈 또는 공백 (선택)
        r"[1-4]\d{6}(?!\d)",    # 성별+일련번호 7자리 (내국인)
    )),
    # 2. 외국인등록번호: 뒤 첫자리 5~8
    (_Tag.FRN, re.compile(
        r"(?<!\d)\d{6}"
        r"[-\s]?"
        r"[5-8]\d{6}(?!\d)",
    )),
    # 3. 여권번호: 영문 1~2자 + 숫자 7~8자 (예: M12345678, AB1234567)
    (_Tag.PASSPORT, re.compile(
        r"(?<![A-Za-z])[A-Za-z]{1,2}\d{7,8}(?!\d)"
    )),
    # 4. 카드번호: 4자리씩 4묶음 (하이픈·공백 구분자 허용)
    (_Tag.CARD, re.compile(
        r"(?<!\d)\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}(?!\d)"
    )),
    # 5. 전화번호 (계좌번호보다 먼저: 010- 으로 시작하는 패턴 우선 처리)
    #    휴대폰: 010-XXXX-XXXX, 010XXXXXXXX
    #    유선:   02-XXXX-XXXX, 031-XXX-XXXX 등
    (_Tag.PHONE, re.compile(
        r"(?<!\d)0\d{1,2}"      # 0으로 시작하는 지역번호/통신사 코드
        r"[-\s]?"
        r"\d{3,4}"
        r"[-\s]?"
        r"\d{4}(?!\d)"
    )),
    # 6. 계좌번호: 은행별로 3~4자리-N자리-N자리 구조
    #    예) 110-123-456789 / 123-456789-01-234 / 1002-123-456789
    #    ※ 전화번호보다 뒤에 위치해 010- 패턴이 전화번호로 먼저 처리됨
    (_Tag.ACCOUNT, re.compile(
        r"(?<!\d)\d{3,4}"       # 은행 코드
        r"-\d{2,6}"             # 중간
        r"(?:-\d{2,6})+"        # 나머지 (1회 이상 반복)
        r"(?!\d)"
    )),
    # 7. 이메일
    (_Tag.EMAIL, re.compile(
        r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
    )),
    # 8. 13자리 연속 숫자 (주민번호 형식 의심 숫자열, 위 패턴에서 안 걸린 것)
    (_Tag.NUM13, re.compile(
        r"(?<!\d)\d{13}(?!\d)"
    )),
]


# ──────────────────────────────────────────────
# 공개 API
# ──────────────────────────────────────────────

@dataclass
class MaskResult:
    masked_text: str
    mask_map: dict[str, str] = field(default_factory=dict)  # 토큰 → 원본값

    @property
    def has_pii(self) -> bool:
        return bool(self.mask_map)


def mask_text(text: str) -> MaskResult:
    """텍스트에서 PII를 마스킹하고 MaskResult를 반환한다.

    LLM에는 반드시 result.masked_text만 전달할 것.
    mask_map은 서버 메모리에만 보관하며 외부로 노출하지 않는다.

    Args:
        text: 원본 텍스트

    Returns:
        MaskResult(masked_text, mask_map)
        masked_text: PII가 [TAG_N] 토큰으로 교체된 텍스트
        mask_map: {토큰: 원본값} — 로깅/감사 용도
    """
    result_text = text
    mask_map: dict[str, str] = {}
    counters: dict[str, int] = {}

    for tag, pattern in _PATTERNS:
        count = counters.get(tag, 0)

        def _replace(m: re.Match, _tag: str = tag, _count_ref: list = [count]) -> str:
            token = f"[{_tag}_{_count_ref[0]}]"
            _count_ref[0] += 1
            mask_map[token] = m.group(0)
            return token

        result_text, n = _replace_all(pattern, result_text, _replace)
        counters[tag] = count + n

    return MaskResult(masked_text=result_text, mask_map=mask_map)


def scan_for_pii(text: str) -> list[str]:
    """텍스트에서 감지된 PII 유형 목록을 반환한다 (마스킹 없이 탐지만).

    Generator 출력 사후 검사(3차 안전망)에서 사용한다.
    """
    found: list[str] = []
    for tag, pattern in _PATTERNS:
        if pattern.search(text):
            found.append(tag)
    return found


# ──────────────────────────────────────────────
# 내부 유틸
# ──────────────────────────────────────────────

def _replace_all(
    pattern: re.Pattern,
    text: str,
    repl_fn,
) -> tuple[str, int]:
    """패턴을 모두 교체하고 (교체된 텍스트, 교체 횟수)를 반환한다.

    re.sub의 callable repl은 count를 추적하기 어려우므로
    finditer로 수동 교체한다.
    """
    result_parts: list[str] = []
    prev_end = 0
    count = 0

    for m in pattern.finditer(text):
        result_parts.append(text[prev_end:m.start()])
        result_parts.append(repl_fn(m))
        prev_end = m.end()
        count += 1

    result_parts.append(text[prev_end:])
    return "".join(result_parts), count
