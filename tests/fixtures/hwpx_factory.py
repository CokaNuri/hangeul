"""테스트용 최소 HWPX 파일 생성 헬퍼.

실제 한글 프로그램 없이도 ZIP+XML 구조의 유효한 .hwpx bytes를 생성한다.
"""

from __future__ import annotations

import io
import zipfile


HP_NS = "http://www.hancom.co.kr/hwpml/2011/paragraph"
HS_NS = "http://www.hancom.co.kr/hwpml/2011/section"

_MIMETYPE = b"application/hwp+zip"

_CONTAINER_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="Contents/content.hpf" media-type="application/xml"/>
  </rootfiles>
</container>"""

_CONTENT_HPF = b"""<?xml version="1.0" encoding="UTF-8"?>
<hpf:HWPDocumentFile xmlns:hpf="http://www.hancom.co.kr/hwpml/2012/core">
  <hpf:manifest>
    <hpf:item id="section0" href="section0.xml" media-type="application/xml"/>
  </hpf:manifest>
</hpf:HWPDocumentFile>"""


def make_section_xml(paragraphs: list[str], table_rows: list[list[str]] | None = None) -> bytes:
    """단락 목록과 표 행 목록으로 섹션 XML bytes를 생성한다."""
    para_blocks = ""
    for text in paragraphs:
        escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        para_blocks += f"""
  <hp:p xmlns:hp="{HP_NS}">
    <hp:r><hp:t>{escaped}</hp:t></hp:r>
  </hp:p>"""

    table_block = ""
    if table_rows:
        rows_xml = ""
        for row in table_rows:
            cells_xml = ""
            for cell in row:
                escaped = cell.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                cells_xml += f"""
      <hp:tc xmlns:hp="{HP_NS}">
        <hp:p xmlns:hp="{HP_NS}"><hp:r><hp:t>{escaped}</hp:t></hp:r></hp:p>
      </hp:tc>"""
            rows_xml += f"""
    <hp:tr xmlns:hp="{HP_NS}">{cells_xml}
    </hp:tr>"""
        table_block = f"""
  <hp:tbl xmlns:hp="{HP_NS}">{rows_xml}
  </hp:tbl>"""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<hs:sec xmlns:hs="{HS_NS}" xmlns:hp="{HP_NS}">{para_blocks}{table_block}
</hs:sec>""".encode("utf-8")


def make_hwpx(
    paragraphs: list[str] | None = None,
    table_rows: list[list[str]] | None = None,
) -> bytes:
    """최소 유효 .hwpx bytes를 생성한다."""
    if paragraphs is None:
        paragraphs = ["연구 목표", "연구 내용을 입력하세요.", "성명", ""]

    section_xml = make_section_xml(paragraphs, table_rows)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", _MIMETYPE)
        zf.writestr("META-INF/container.xml", _CONTAINER_XML)
        zf.writestr("Contents/content.hpf", _CONTENT_HPF)
        zf.writestr("Contents/section0.xml", section_xml)
    return buf.getvalue()
