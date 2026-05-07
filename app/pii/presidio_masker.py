"""Presidio 한국어 PII 마스킹 — Step 12 (2차 안전망).

정규식이 잡지 못하는 이름·주소·소속·학번을 커스텀 PatternRecognizer로 마스킹한다.

spaCy 학습 모델이 없어도 동작한다 (영어 blank 모델을 NLP 엔진으로 사용).
PatternRecognizer는 NLP 엔진의 NER을 사용하지 않고 정규식만으로 동작하기 때문이다.

커버리지:
  - KR_NAME:       한국 성씨 + 이름 (홍길동, 김철수 등)
  - KR_STUDENT_ID: 학번 (19/20xx + 4-8자리 숫자)
  - KR_ADDRESS:    한국 주소 (특별시/광역시/도 + 구/동/로/길)
  - KR_ORG:        소속기관 (대학교/연구원/연구소/기관 등)

초기화 실패 시 소프트 실패 — 원본 텍스트를 그대로 반환한다 (1차 안전망은 이미 적용됨).
"""
from __future__ import annotations

import logging

from app.pii.regex_masker import MaskResult

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 주요 한국 성씨 (통계청 기준 상위 100여 개)
# ──────────────────────────────────────────────

_KR_SURNAMES = (
    "김|이|박|최|정|강|조|윤|장|임|한|오|서|신|권|황|안|송|류|전|홍"
    "|고|문|양|손|배|백|허|유|남|심|노|하|곽|성|차|주|우|구|민|나"
    "|진|지|엄|채|원|천|방|공|현|함|변|염|여|추|도|소|석|선|설|마"
    "|길|연|위|표|명|기|반|라|왕|금|옥|육|인|맹|제|모|탁|국|편|복|예"
)

# ──────────────────────────────────────────────
# Presidio 지연 초기화
# ──────────────────────────────────────────────

_analyzer = None


def _build_analyzer():
    """AnalyzerEngine을 spaCy blank 영어 모델로 구성한다.

    Notes:
        - RecognizerRegistry(recognizers=[])로 기본 영어 recognizer를 제외한다.
        - AnalyzerEngine에 supported_languages를 명시하지 않는다
          (명시하면 registry.supported_languages와 비교 검증해 오류 발생).
        - 'ko' 언어로 analyze()를 호출하면 PatternRecognizer의
          supported_language='ko' 설정에 따라 올바르게 적용된다.
    """
    import spacy
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    from presidio_analyzer.nlp_engine import SpacyNlpEngine

    blank_nlp = spacy.blank("en")

    class _BlankKoreanNlpEngine(SpacyNlpEngine):
        def __init__(self) -> None:
            super().__init__()
            # 영어 blank 모델을 'ko' 키로도 등록해 language='ko' 요청을 처리
            self.nlp = {"ko": blank_nlp, "en": blank_nlp}

    registry = RecognizerRegistry(recognizers=[])
    for recognizer in _make_recognizers():
        registry.add_recognizer(recognizer)

    return AnalyzerEngine(
        nlp_engine=_BlankKoreanNlpEngine(),
        registry=registry,
        # supported_languages 미지정 → NLP 엔진에서 자동 추론
    )


def _get_analyzer():
    global _analyzer
    if _analyzer is not None:
        return _analyzer
    try:
        _analyzer = _build_analyzer()
        logger.info("[Presidio] 한국어 PII 분석 엔진 초기화 완료.")
        return _analyzer
    except Exception as exc:
        logger.warning("[Presidio] 초기화 실패, 2차 안전망 비활성화: %s", exc)
        return None


# ──────────────────────────────────────────────
# 공개 API
# ──────────────────────────────────────────────

def mask_presidio(text: str) -> MaskResult:
    """Presidio로 텍스트의 한국어 PII를 마스킹한다.

    Args:
        text: 1차(정규식) 마스킹이 완료된 텍스트

    Returns:
        MaskResult — 마스킹된 텍스트와 토큰→원본 매핑
    """
    if not text:
        return MaskResult(masked_text=text, mask_map={})

    analyzer = _get_analyzer()
    if analyzer is None:
        return MaskResult(masked_text=text, mask_map={})

    try:
        results = analyzer.analyze(text=text, language="ko")
    except Exception as exc:
        logger.error("[Presidio] 분석 실패: %s", exc)
        return MaskResult(masked_text=text, mask_map={})

    if not results:
        return MaskResult(masked_text=text, mask_map={})

    return _replace_entities(text, results)


# ──────────────────────────────────────────────
# Recognizer 팩토리
# ──────────────────────────────────────────────

def _make_recognizers() -> list:
    """커스텀 한국어 PII Recognizer 목록을 생성한다.

    이름·소속 recognizer는 false positive를 줄이기 위해
    필드 레이블(성명:, 신청자: 등)을 패턴에 포함한다.
    학번·주소 recognizer는 구조가 충분히 특이해 레이블 없이도 동작한다.
    """
    from presidio_analyzer import PatternRecognizer, Pattern

    # ── KR_NAME: 필드 레이블 + 성씨 + 이름 ────────
    # 한국어에서 성씨 음절은 일반 어휘에도 많이 등장하므로
    # 레이블(성명:, 신청자: 등)이 앞에 있는 경우에만 이름으로 판정한다.
    _name_label = (
        r"(?:성명|이름|신청인|신청자|연구자|저자|책임자"
        r"|담당자|대표자|참여연구원|공동연구원|수행자)"
        r"\s*[:：\s]\s*"
    )
    name_labeled_pattern = Pattern(
        name="korean_name_labeled",
        regex=_name_label + r"(?:" + _KR_SURNAMES + r")[가-힣]{1,3}",
        score=0.85,
    )

    # ── KR_STUDENT_ID: 년도 + 순번 ────────────────
    student_id_pattern = Pattern(
        name="korean_student_id",
        regex=r"(?<!\d)(?:19|20)\d{2}[-\s]?\d{4,8}(?!\d)",
        score=0.6,
    )

    # ── KR_ADDRESS: 시·도 + 구·동·로 ─────────────
    address_pattern = Pattern(
        name="korean_address",
        regex=(
            r"[가-힣]{2,6}"
            r"(?:특별시|광역시|특별자치시|특별자치도|도|시)"
            r"[\s,·]+"
            r"[가-힣]{2,6}"
            r"(?:시|군|구|읍|면|동|로|대로|길)"
        ),
        score=0.7,
    )

    # ── KR_ORG: 필드 레이블 + 기관명 ─────────────
    _org_label = (
        r"(?:소속|재직|근무|출신|기관|학교)\s*[:：\s]\s*"
    )
    org_labeled_pattern = Pattern(
        name="korean_org_labeled",
        regex=(
            _org_label
            + r"[가-힣]{2,10}"
            + r"(?:대학교|대학원|대학|연구원|연구소|기관|센터"
            + r"|병원|재단|협회|학회|공단|공사|주식회사|유한회사)"
        ),
        score=0.8,
    )
    # 레이블 없이도 기관명이 명확한 경우 (낮은 점수)
    org_bare_pattern = Pattern(
        name="korean_org_bare",
        regex=(
            r"[가-힣]{2,10}"
            r"(?:대학교|대학원|연구원|연구소|재단|공단|공사)"
        ),
        score=0.55,
    )

    return [
        PatternRecognizer(
            supported_entity="KR_NAME",
            patterns=[name_labeled_pattern],
            supported_language="ko",
        ),
        PatternRecognizer(
            supported_entity="KR_STUDENT_ID",
            patterns=[student_id_pattern],
            context=["학번", "학생번호", "학적번호", "입학", "졸업"],
            supported_language="ko",
        ),
        PatternRecognizer(
            supported_entity="KR_ADDRESS",
            patterns=[address_pattern],
            supported_language="ko",
        ),
        PatternRecognizer(
            supported_entity="KR_ORG",
            patterns=[org_labeled_pattern, org_bare_pattern],
            supported_language="ko",
        ),
    ]


# ──────────────────────────────────────────────
# 내부 유틸
# ──────────────────────────────────────────────

def _replace_entities(text: str, results) -> MaskResult:
    """분석 결과를 토큰으로 교체하고 MaskResult를 반환한다.

    끝 위치부터 역순으로 교체해 위치 오프셋 이동을 방지한다.
    """
    deduplicated = _remove_overlaps(results)
    sorted_results = sorted(deduplicated, key=lambda r: r.start, reverse=True)

    masked = text
    mask_map: dict[str, str] = {}
    counters: dict[str, int] = {}

    for result in sorted_results:
        entity = result.entity_type
        count = counters.get(entity, 0)
        token = f"[{entity}_{count}]"
        counters[entity] = count + 1

        original = masked[result.start:result.end]
        mask_map[token] = original
        masked = masked[:result.start] + token + masked[result.end:]

    return MaskResult(masked_text=masked, mask_map=mask_map)


def _remove_overlaps(results) -> list:
    """겹치는 분석 결과 중 score가 높은 것을 남긴다."""
    if not results:
        return []
    sorted_by_start = sorted(results, key=lambda r: (r.start, -r.score))
    kept = [sorted_by_start[0]]
    for result in sorted_by_start[1:]:
        last = kept[-1]
        if result.start < last.end:
            if result.score > last.score:
                kept[-1] = result
        else:
            kept.append(result)
    return kept
