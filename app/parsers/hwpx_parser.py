"""HWPX 파서 — Step 10 실제 구현.

HWPX 포맷: ZIP 컨테이너 + XML 섹션 파일
- mimetype
- META-INF/container.xml
- Contents/content.hpf  (매니페스트)
- Contents/section0.xml (본문 섹션)
- BinData/              (이미지 등 바이너리)

네임스페이스:
  hp  → http://www.hancom.co.kr/hwpml/2011/paragraph
  hs  → http://www.hancom.co.kr/hwpml/2011/section

주요 태그:
  <hp:p>   단락
  <hp:r>   런 (run)
  <hp:t>   텍스트
  <hp:tbl> 표
  <hp:tr>  행
  <hp:tc>  셀

빈칸 식별 기준 (Step 10):
  1. 빈 단락: <hp:p> 텍스트가 공백만이거나 완전히 없음
  2. 괄호 패턴: [ ], ( ), □, ___ 등 관용적 빈칸 표기
  3. 빈 표 셀: <hp:tc> 텍스트가 공백만이거나 없음 (헤더 셀 제외)
  4. PII 분류: 빈칸의 앞 단락(레이블) 텍스트에 PII 키워드 포함 여부로 판단
"""

from __future__ import annotations

import io
import re
import zipfile
from typing import Any

from lxml import etree

from app.models import FormDoc, FormItem, FormTable, ItemType

# ──────────────────────────────────────────────
# 네임스페이스
# ──────────────────────────────────────────────

HP_NS = "http://www.hancom.co.kr/hwpml/2011/paragraph"
HS_NS = "http://www.hancom.co.kr/hwpml/2011/section"

NS = {"hp": HP_NS, "hs": HS_NS}

# PII 필드 키워드
_PII_KEYWORDS = frozenset([
    "성명", "이름", "주민등록번호", "주민번호", "외국인등록번호",
    "연락처", "전화번호", "휴대폰", "이메일", "e-mail",
    "주소", "거주지", "계좌번호", "계좌", "카드번호",
    "학번", "사번", "소속기관", "가족관계",
])

# 괄호 빈칸 패턴: [ ], ( ), □, ___ 등
_BLANK_BRACKET_RE = re.compile(
    r"(\[[\s　]*\])"      # [ ] — 대괄호 빈칸
    r"|(\([\s　]*\))"     # ( ) — 소괄호 빈칸
    r"|(□+)"              # □ — 체크박스 문자
    r"|(_{3,})"           # ___ — 밑줄 빈칸 (3자 이상)
)


# ──────────────────────────────────────────────
# 공개 API
# ──────────────────────────────────────────────

def parse_hwpx(file_bytes: bytes) -> FormDoc:
    """HWPX bytes를 파싱해 FormDoc을 반환한다.

    Args:
        file_bytes: .hwpx 파일의 원본 bytes

    Returns:
        FormDoc — items(단락 항목), tables(표 항목) 포함

    Raises:
        ValueError: ZIP 파일이 손상됐거나 섹션을 찾을 수 없을 때
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(file_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"HWPX 파일을 열 수 없습니다 (ZIP 손상): {exc}") from exc

    with zf:
        section_paths = _find_section_paths(zf)
        if not section_paths:
            raise ValueError("HWPX 파일에서 섹션 XML을 찾을 수 없습니다.")

        all_items: list[FormItem] = []
        all_tables: list[FormTable] = []

        for sec_path in section_paths:
            raw = zf.read(sec_path)
            items, tables = _parse_section_xml(raw, sec_path)
            all_items.extend(items)
            all_tables.extend(tables)

        # 렌더러가 전체 ZIP을 수정할 수 있도록 원본 bytes 보관
        primary_raw = zf.read(section_paths[0])

    return FormDoc(
        raw_xml=primary_raw,
        items=all_items,
        tables=all_tables,
        metadata={
            "hwpx_bytes": file_bytes,       # 렌더러가 재패키징에 사용
            "section_paths": section_paths,  # 섹션 파일 경로 목록
        },
    )


# ──────────────────────────────────────────────
# 내부 함수
# ──────────────────────────────────────────────

def _find_section_paths(zf: zipfile.ZipFile) -> list[str]:
    """ZIP 내 섹션 XML 파일 경로를 정렬해 반환한다."""
    names = zf.namelist()

    # content.hpf 기반 우선 탐색
    hpf_path = next((n for n in names if n.endswith("content.hpf")), None)
    if hpf_path:
        try:
            paths = _sections_from_hpf(zf.read(hpf_path), hpf_path)
            if paths:
                return paths
        except Exception:
            pass  # fallback으로 계속

    # fallback: Contents/ 아래 section*.xml 패턴
    prefix = hpf_path.rsplit("/", 1)[0] + "/" if hpf_path else "Contents/"
    sections = sorted(
        n for n in names
        if n.startswith(prefix) and "section" in n.lower() and n.endswith(".xml")
    )
    return sections


def _sections_from_hpf(hpf_bytes: bytes, hpf_path: str) -> list[str]:
    """content.hpf XML을 파싱해 섹션 파일 경로 목록을 반환한다."""
    root = etree.fromstring(hpf_bytes)
    base_dir = hpf_path.rsplit("/", 1)[0] + "/"

    # hpf는 네임스페이스가 다양하므로 localname으로 매칭
    refs = []
    for elem in root.iter():
        local = etree.QName(elem.tag).localname if elem.tag and "{" in elem.tag else elem.tag
        if local in ("item", "manifest-item", "file"):
            href = elem.get("href") or elem.get("src") or ""
            if "section" in href.lower() and href.endswith(".xml"):
                path = href if "/" in href else base_dir + href
                refs.append(path)

    return sorted(set(refs))


def _parse_section_xml(
    raw_xml: bytes, sec_path: str
) -> tuple[list[FormItem], list[FormTable]]:
    """섹션 XML에서 빈칸 항목과 표를 추출한다.

    빈칸 감지 기준:
      - 단락 텍스트가 비어있음 (공백 포함)
      - 단락 텍스트가 괄호 빈칸 패턴과 일치함 ([ ], □ 등)
      - 표 셀 텍스트가 비어있음 (헤더 행 제외)

    PII 분류: 빈칸 직전 단락(레이블)의 텍스트에 PII 키워드 포함 여부로 판단.
    """
    try:
        root = etree.fromstring(raw_xml)
    except etree.XMLSyntaxError:
        return [], []  # 파싱 실패 시 빈 결과 (soft fail)

    items: list[FormItem] = []
    tables: list[FormTable] = []

    # ── 최상위 단락 수집 (표 안 단락 제외) ────────────────
    para_index = 0
    prev_label = ""  # 직전 비어있지 않은 단락 텍스트 (빈칸의 레이블 후보)

    for para in root.iter(f"{{{HP_NS}}}p"):
        # 표 셀 안 단락은 표 처리 루프에서 별도로 다룸
        parent = para.getparent()
        if parent is not None and parent.tag == f"{{{HP_NS}}}tc":
            continue

        text = _para_text(para)
        item_id = f"{sec_path}::p{para_index}"

        if _is_blank_text(text):
            # 빈칸 단락 → FormItem 생성
            label = prev_label or f"항목 {para_index}"
            item_type = _classify_item_type(label)
            items.append(FormItem(
                item_id=item_id,
                label=label[:40],
                item_type=item_type,
                xml_path=item_id,
                context=prev_label,
            ))
        else:
            # 비어있지 않은 단락 → 다음 빈칸의 레이블 후보로 기억
            prev_label = text

        para_index += 1

    # ── 표 수집 + 빈 셀 FormItem 생성 ─────────────────────
    tbl_index = 0
    for tbl in root.iter(f"{{{HP_NS}}}tbl"):
        table_id = f"{sec_path}::tbl{tbl_index}"
        header_row: list[str] = []
        data_rows: list[list[str]] = []

        all_rows = tbl.findall(f"{{{HP_NS}}}tr")
        for row_i, tr in enumerate(all_rows):
            cells = [_cell_text(tc) for tc in tr.findall(f"{{{HP_NS}}}tc")]
            if row_i == 0:
                header_row = cells
                continue  # 헤더 행은 빈칸 감지 대상에서 제외
            data_rows.append(cells)

            # 헤더 첫 열(행 레이블)을 제외하고 빈 셀을 FormItem으로 추가
            row_header = cells[0] if cells else ""
            for col_i, cell_text in enumerate(cells):
                if col_i == 0:
                    continue  # 첫 번째 셀은 행 레이블 → 건너뜀
                if _is_blank_text(cell_text):
                    col_header = header_row[col_i] if col_i < len(header_row) else f"열{col_i}"
                    label = f"{row_header} / {col_header}" if row_header else col_header
                    cell_id = f"{table_id}::r{row_i}c{col_i}"
                    items.append(FormItem(
                        item_id=cell_id,
                        label=label[:40],
                        item_type=_classify_item_type(label),
                        xml_path=cell_id,
                        context=str(header_row),
                    ))

        tables.append(FormTable(
            table_id=table_id,
            header_row=header_row,
            data_rows=data_rows,
            xml_path=table_id,
        ))
        tbl_index += 1

    return items, tables


def _para_text(para_elem) -> str:
    """단락 요소에서 텍스트를 추출한다 (<hp:t> 텍스트 연결)."""
    parts = []
    for t in para_elem.iter(f"{{{HP_NS}}}t"):
        if t.text:
            parts.append(t.text.strip())
    return " ".join(p for p in parts if p)


def _cell_text(tc_elem) -> str:
    """셀 요소에서 텍스트를 추출한다."""
    parts = []
    for t in tc_elem.iter(f"{{{HP_NS}}}t"):
        if t.text:
            parts.append(t.text.strip())
    return " ".join(p for p in parts if p)


def _is_blank_text(text: str) -> bool:
    """텍스트가 빈칸(공백 전용 또는 괄호 빈칸 패턴)인지 판별한다."""
    stripped = text.strip()
    if not stripped:
        return True
    return bool(_BLANK_BRACKET_RE.fullmatch(stripped) or _BLANK_BRACKET_RE.search(stripped) and stripped == _BLANK_BRACKET_RE.search(stripped).group())


def _classify_item_type(label: str) -> ItemType:
    """레이블 텍스트로 PII 필드 여부를 판별한다.

    빈칸 자체가 아닌 레이블(앞 단락 텍스트 또는 열·행 헤더)을 기준으로 분류.
    """
    lower = label.lower()
    for keyword in _PII_KEYWORDS:
        if keyword in lower:
            return ItemType.PII
    return ItemType.TEXT
