"""HWPX 렌더러 — Step 4 스텁 구현.

역할: 승인된 DraftItem 텍스트를 FormDoc의 XML 노드에 삽입하고
     유효한 .hwpx bytes(ZIP)로 재패키징한다.

실제 XPath 기반 노드 탐색 및 표 셀 삽입 정교화는 Step 17에서 수행한다.
현재는 item_id를 키로 단락 텍스트를 교체하는 수준에 집중한다.

item_id 형식 (Step 3 파서 기준):
  "Contents/section0.xml::p{index}"   → 단락
  "Contents/section0.xml::tbl{index}" → 표 (Step 17에서 처리)
"""

from __future__ import annotations

import io
import zipfile

from lxml import etree

from app.models import FormDoc

HP_NS = "http://www.hancom.co.kr/hwpml/2011/paragraph"
HS_NS = "http://www.hancom.co.kr/hwpml/2011/section"

# PII 항목에 삽입할 플레이스홀더
PII_PLACEHOLDER = "[본인 직접 입력]"


# ──────────────────────────────────────────────
# 공개 API
# ──────────────────────────────────────────────

def inject_text(form_doc: FormDoc, item_id: str, text: str) -> None:
    """FormDoc의 특정 항목에 텍스트를 삽입한다 (in-place).

    단락 항목(p{n})의 경우 해당 섹션 XML을 파싱해 n번째 <hp:p>의
    첫 번째 <hp:t> 텍스트를 교체한다.

    Args:
        form_doc: 파서가 반환한 FormDoc (metadata["hwpx_bytes"] 필요)
        item_id:  FormItem.item_id (예: "Contents/section0.xml::p2")
        text:     삽입할 텍스트
    """
    if "::" not in item_id:
        return

    sec_path, node_ref = item_id.split("::", 1)

    # metadata의 섹션별 수정 버퍼에 기록
    # (repack_hwpx 호출 시 버퍼를 모아 ZIP을 재조립)
    _ensure_section_buffer(form_doc, sec_path)
    buf: dict[str, bytes] = form_doc.metadata.setdefault("modified_sections", {})

    raw = buf.get(sec_path) or _read_section(form_doc, sec_path)
    if raw is None:
        return

    modified = _apply_text_to_section(raw, node_ref, text)
    buf[sec_path] = modified


def repack_hwpx(form_doc: FormDoc) -> bytes:
    """수정된 섹션 XML을 원본 ZIP에 합쳐 새 .hwpx bytes를 반환한다.

    원본 ZIP의 모든 파일을 복사하되,
    inject_text로 수정된 섹션은 교체된 버전을 사용한다.

    Returns:
        완성된 .hwpx 파일의 bytes
    """
    original_bytes: bytes = form_doc.metadata.get("hwpx_bytes", b"")
    modified_sections: dict[str, bytes] = form_doc.metadata.get("modified_sections", {})

    if not original_bytes:
        raise ValueError("FormDoc에 원본 hwpx_bytes가 없습니다.")

    out_buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(original_bytes)) as src_zip, \
         zipfile.ZipFile(out_buf, "w", zipfile.ZIP_DEFLATED) as dst_zip:

        for item in src_zip.infolist():
            if item.filename in modified_sections:
                dst_zip.writestr(item.filename, modified_sections[item.filename])
            else:
                dst_zip.writestr(item, src_zip.read(item.filename))

    return out_buf.getvalue()


def render_preview_markdown(item_id: str, text: str, citations: list[str]) -> str:
    """항목 미리보기용 마크다운 문자열을 반환한다.

    Streamlit 채팅 메시지에서 `[✓ 적용] [✏ 수정] [🔁 다시]` 버튼 위에
    표시할 텍스트 블록을 생성한다.
    """
    cite_str = ""
    if citations:
        cite_str = "\n> 참고 자료: " + ", ".join(f"`{c}`" for c in citations)

    return (
        f"**[{item_id}]** 초안\n\n"
        f"```\n{text}\n```"
        f"{cite_str}"
    )


# ──────────────────────────────────────────────
# 내부 함수
# ──────────────────────────────────────────────

def _ensure_section_buffer(form_doc: FormDoc, sec_path: str) -> None:
    """metadata에 modified_sections 딕셔너리를 초기화한다."""
    form_doc.metadata.setdefault("modified_sections", {})


def _read_section(form_doc: FormDoc, sec_path: str) -> bytes | None:
    """원본 hwpx_bytes ZIP에서 섹션 XML을 읽는다."""
    original_bytes: bytes = form_doc.metadata.get("hwpx_bytes", b"")
    if not original_bytes:
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(original_bytes)) as zf:
            return zf.read(sec_path)
    except (KeyError, zipfile.BadZipFile):
        return None


def _apply_text_to_section(raw_xml: bytes, node_ref: str, text: str) -> bytes:
    """섹션 XML bytes에서 node_ref가 가리키는 노드의 텍스트를 교체한다.

    node_ref 형식:
      "p{index}"    → index번째 최상위 <hp:p>의 첫 <hp:t> 교체
      "tbl{index}"  → Step 17에서 구현 (현재는 no-op)
    """
    try:
        root = etree.fromstring(raw_xml)
    except etree.XMLSyntaxError:
        return raw_xml  # 파싱 실패 시 원본 반환

    if node_ref.startswith("p"):
        _replace_para_text(root, node_ref, text)
    # tbl 처리는 Step 17에서 추가

    return etree.tostring(root, encoding="utf-8", xml_declaration=True)


def _replace_para_text(root, node_ref: str, text: str) -> None:
    """n번째 최상위 <hp:p>의 첫 <hp:t> 텍스트를 교체한다."""
    try:
        index = int(node_ref[1:])  # "p3" → 3
    except ValueError:
        return

    # 표 셀 안 단락은 제외하고 최상위 단락만 수집
    top_paras = [
        elem for elem in root.iter(f"{{{HP_NS}}}p")
        if elem.getparent() is not None
        and elem.getparent().tag != f"{{{HP_NS}}}tc"
    ]

    if index >= len(top_paras):
        return

    para = top_paras[index]
    t_elems = list(para.iter(f"{{{HP_NS}}}t"))

    if t_elems:
        # 첫 번째 <hp:t>에 텍스트 설정, 나머지 비움
        t_elems[0].text = text
        for t in t_elems[1:]:
            t.text = ""
    else:
        # <hp:t>가 없으면 <hp:r><hp:t> 구조를 새로 생성
        run = etree.SubElement(para, f"{{{HP_NS}}}r")
        t = etree.SubElement(run, f"{{{HP_NS}}}t")
        t.text = text
