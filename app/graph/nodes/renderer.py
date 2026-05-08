"""Renderer 노드 — Step 17 실제 구현.

담당 역할:
  - 현재 항목의 승인된 초안 텍스트를 HWPX XML에 삽입한다.
  - 삽입 후 갱신된 HWPX bytes를 FormDoc.metadata["hwpx_bytes"]에 저장한다.
  - approved_items에 item_id를 추가하고 current_item_index를 +1 진행한다.

xml_path 형식 (hwpx_parser.py와 동일):
  - 단락: "{sec_path}::p{para_index}"
    예) "Contents/section0.xml::p2"
  - 표 셀: "{sec_path}::tbl{tbl_index}::r{row_i}c{col_i}"
    예) "Contents/section0.xml::tbl0::r1c2"

삽입 전략:
  - 대상 <hp:p> 안의 기존 <hp:r> 런을 모두 제거한다.
  - 새 <hp:r><hp:t>text</hp:t></hp:r>를 추가한다.
  - XML을 재직렬화해 ZIP에 교체한다.
"""
from __future__ import annotations

import io
import logging
import re
import zipfile
from copy import deepcopy
from dataclasses import replace

from lxml import etree

from app.graph.state import GraphState
from app.models import FormDoc

logger = logging.getLogger(__name__)

HP_NS = "http://www.hancom.co.kr/hwpml/2011/paragraph"
_HP = f"{{{HP_NS}}}"


# ──────────────────────────────────────────────
# 노드 진입점
# ──────────────────────────────────────────────

def renderer_node(state: GraphState) -> GraphState:
    """현재 항목의 초안을 HWPX XML에 삽입하고 인덱스를 진행한다."""
    plans = state.get("item_plans", [])
    idx = state.get("current_item_index", 0)
    if idx >= len(plans):
        return {}

    plan = plans[idx]
    item_id = plan.item_id
    drafts = state.get("drafts") or {}
    draft = drafts.get(item_id)

    # approved_items 갱신 + 인덱스 진행 (항상 수행)
    approved = list(state.get("approved_items") or [])
    if item_id not in approved:
        approved.append(item_id)
    updates: dict = {"approved_items": approved, "current_item_index": idx + 1}

    if draft is None or not draft.text:
        logger.warning("[Renderer] item_id=%s 초안 없음, XML 삽입 건너뜀.", item_id)
        return updates

    form_doc = state.get("form_doc")
    if form_doc is None:
        return updates

    hwpx_bytes = form_doc.metadata.get("hwpx_bytes")
    if not hwpx_bytes:
        logger.warning("[Renderer] hwpx_bytes 없음, XML 삽입 건너뜀.")
        return updates

    # FormItem에서 xml_path 조회
    item_meta = next((it for it in form_doc.items if it.item_id == item_id), None)
    if item_meta is None:
        logger.warning("[Renderer] FormItem '%s' 없음, XML 삽입 건너뜀.", item_id)
        return updates

    try:
        new_hwpx = _insert_text(hwpx_bytes, item_meta.xml_path, draft.text)
        new_form_doc = FormDoc(
            raw_xml=form_doc.raw_xml,
            items=form_doc.items,
            tables=form_doc.tables,
            metadata={**form_doc.metadata, "hwpx_bytes": new_hwpx},
        )
        updates["form_doc"] = new_form_doc
        logger.info("[Renderer] item_id=%s XML 삽입 완료 (%d자)", item_id, len(draft.text))
    except Exception as exc:
        logger.error("[Renderer] item_id=%s XML 삽입 실패 (소프트 실패): %s", item_id, exc)
        # 삽입 실패해도 approved에는 추가 (초안은 메모리에 있음)

    return updates


# ──────────────────────────────────────────────
# XML 삽입 핵심 로직
# ──────────────────────────────────────────────

def _insert_text(hwpx_bytes: bytes, xml_path: str, text: str) -> bytes:
    """xml_path가 가리키는 XML 요소에 text를 삽입하고 갱신된 HWPX bytes를 반환한다."""
    parts = xml_path.split("::")
    if len(parts) < 2:
        raise ValueError(f"잘못된 xml_path: {xml_path!r}")

    sec_path = parts[0]
    spec = parts[1:]  # e.g. ["p2"] or ["tbl0", "r1c2"]

    # ── ZIP 읽기 ─────────────────────────────
    orig_data: dict[str, bytes] = {}
    orig_infos: list[zipfile.ZipInfo] = []
    with zipfile.ZipFile(io.BytesIO(hwpx_bytes), "r") as zf:
        orig_infos = zf.infolist()
        for info in orig_infos:
            orig_data[info.filename] = zf.read(info.filename)

    if sec_path not in orig_data:
        raise ValueError(f"섹션 파일 없음: {sec_path!r}")

    # ── XML 파싱 + 요소 탐색 ─────────────────
    raw_xml = orig_data[sec_path]
    root = etree.fromstring(raw_xml)

    if spec[0].startswith("p"):
        para_idx = int(spec[0][1:])
        elem = _find_para(root, para_idx)
        _set_run_text(elem, text)
    elif spec[0].startswith("tbl"):
        tbl_idx = int(spec[0][3:])
        m = re.fullmatch(r"r(\d+)c(\d+)", spec[1]) if len(spec) > 1 else None
        if not m:
            raise ValueError(f"셀 경로 형식 오류: {spec!r}")
        row_i, col_i = int(m.group(1)), int(m.group(2))
        tc = _find_cell(root, tbl_idx, row_i, col_i)
        # 셀 안 첫 번째 <hp:p> 찾기 (없으면 생성)
        para = tc.find(f".//{_HP}p")
        if para is None:
            para = etree.SubElement(tc, f"{_HP}p")
        _set_run_text(para, text)
    else:
        raise ValueError(f"알 수 없는 요소 유형: {spec[0]!r}")

    # ── XML 재직렬화 ─────────────────────────
    new_xml = etree.tostring(root, encoding="utf-8", xml_declaration=True)
    orig_data[sec_path] = new_xml

    # ── ZIP 재조립 ───────────────────────────
    out_buf = io.BytesIO()
    with zipfile.ZipFile(out_buf, "w", compression=zipfile.ZIP_DEFLATED) as new_zf:
        for info in orig_infos:
            new_zf.writestr(info, orig_data[info.filename])

    return out_buf.getvalue()


# ──────────────────────────────────────────────
# 요소 탐색 헬퍼
# ──────────────────────────────────────────────

def _find_para(root, para_idx: int):
    """최상위 단락 중 para_idx 번째를 반환한다 (표 셀 안 단락 제외)."""
    count = 0
    for para in root.iter(f"{_HP}p"):
        parent = para.getparent()
        if parent is not None and parent.tag == f"{_HP}tc":
            continue  # 표 셀 안 단락은 건너뜀
        if count == para_idx:
            return para
        count += 1
    raise ValueError(f"단락 인덱스 {para_idx} 없음 (총 {count}개)")


def _find_cell(root, tbl_idx: int, row_i: int, col_i: int):
    """tbl_idx번째 표의 row_i행 col_i열 셀(<hp:tc>)을 반환한다."""
    tbls = list(root.iter(f"{_HP}tbl"))
    if tbl_idx >= len(tbls):
        raise ValueError(f"표 인덱스 {tbl_idx} 없음 (총 {len(tbls)}개)")
    tbl = tbls[tbl_idx]
    rows = tbl.findall(f"{_HP}tr")
    if row_i >= len(rows):
        raise ValueError(f"행 인덱스 {row_i} 없음 (총 {len(rows)}개)")
    cells = rows[row_i].findall(f"{_HP}tc")
    if col_i >= len(cells):
        raise ValueError(f"열 인덱스 {col_i} 없음 (총 {len(cells)}개)")
    return cells[col_i]


# ──────────────────────────────────────────────
# 텍스트 삽입
# ──────────────────────────────────────────────

def _set_run_text(para_elem, text: str) -> None:
    """<hp:p> 요소 안의 기존 <hp:r>런을 모두 제거하고 새 런을 추가한다."""
    # 기존 런 제거
    for run in para_elem.findall(f"{_HP}r"):
        para_elem.remove(run)

    # 새 런 삽입 (텍스트가 있을 때만)
    if text:
        run = etree.SubElement(para_elem, f"{_HP}r")
        t_elem = etree.SubElement(run, f"{_HP}t")
        t_elem.text = text
