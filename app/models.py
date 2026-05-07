"""프로젝트 전체에서 공유하는 데이터 모델 (데이터클래스)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ──────────────────────────────────────────────
# 양식 관련
# ──────────────────────────────────────────────

class ItemType(str, Enum):
    TEXT = "text"        # 일반 텍스트 빈칸
    TABLE = "table"      # 표 셀
    PII = "pii"          # 개인정보 필드 → AI가 채우지 않음


@dataclass
class FormItem:
    item_id: str
    label: str                   # 양식에 표시된 항목명 ("연구 목표" 등)
    item_type: ItemType
    xml_path: str                # lxml에서 해당 노드를 찾기 위한 xpath
    context: str = ""            # 주변 텍스트 (AI가 문맥 파악용)
    char_hint: int = 0           # 예상 글자 수 (0 = 제한 없음)


@dataclass
class FormTable:
    table_id: str
    header_row: list[str]        # 헤더 셀 텍스트
    data_rows: list[list[str]]   # 데이터 셀 (빈 셀 = "")
    xml_path: str


@dataclass
class FormDoc:
    raw_xml: bytes               # 원본 contents.xml bytes
    items: list[FormItem] = field(default_factory=list)
    tables: list[FormTable] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────
# 자료 관련
# ──────────────────────────────────────────────

@dataclass
class MaterialDoc:
    source_name: str             # 파일명
    masked_text: str             # PII 마스킹 완료된 텍스트
    summary: str = ""            # LLM 요약 (Step 11에서 채워짐)
    doc_type: str = "unknown"    # "pdf" | "docx" | "txt"


@dataclass
class MaterialBundle:
    docs: list[MaterialDoc] = field(default_factory=list)


# ──────────────────────────────────────────────
# 계획·초안 관련
# ──────────────────────────────────────────────

@dataclass
class ItemPlan:
    item_id: str
    source_evidence: list[str] = field(default_factory=list)  # 근거 텍스트 조각
    confidence: float = 0.0      # 0~1, 매핑 신뢰도
    needs_question: bool = False
    question_text: str = ""      # needs_question=True 일 때 AI가 물어볼 내용
    status: str = "pending"      # pending | drafted | approved | soft_fail


@dataclass
class DraftItem:
    item_id: str
    text: str
    citations: list[str] = field(default_factory=list)  # 참조한 source_name 목록
    retry_count: int = 0


# ──────────────────────────────────────────────
# 라우터
# ──────────────────────────────────────────────

class Intent(str, Enum):
    UPLOAD_FORM = "upload_form"
    UPLOAD_MATERIAL = "upload_material"
    START_FILL = "start_fill"
    REWRITE_ITEM = "rewrite_item"
    CHANGE_TONE = "change_tone"
    ADD_MATERIAL = "add_material"
    GENERAL_QA = "general_qa"
    DISAMBIGUATION = "disambiguation"  # 저신뢰도 → 사용자에게 재질문
