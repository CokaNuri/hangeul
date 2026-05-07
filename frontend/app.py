"""HwpAgent Streamlit UI — Step 9 스켈레톤.

E2E 흐름: 양식 업로드 → 자료 업로드 → 채우기 시작 → 항목 미리보기 → 다운로드
Step 20에서 액션 버튼·양식 구조 트리·표 시각화 등 고도화 예정.
"""

from __future__ import annotations

import streamlit as st

import frontend.api_client as api

# ──────────────────────────────────────────────
# 페이지 설정
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="HwpAgent",
    page_icon=None,
    layout="wide",
)

# ──────────────────────────────────────────────
# 세션 초기화
# ──────────────────────────────────────────────

def _init_session() -> None:
    """Streamlit 세션 상태를 초기화하고 FastAPI 세션을 생성한다."""
    if "session_id" not in st.session_state:
        if not api.health_check():
            st.error("서버에 연결할 수 없습니다. `python -m uvicorn app.main:app --port 8000` 을 먼저 실행해주세요.")
            st.stop()
        st.session_state.session_id = api.create_session()
        st.session_state.messages = []          # 채팅 표시용
        st.session_state.form_uploaded = False
        st.session_state.form_filename = ""
        st.session_state.material_filenames = []
        st.session_state.form_ready = False     # 다운로드 가능 여부


_init_session()

# ──────────────────────────────────────────────
# 사이드바: 파일 업로드
# ──────────────────────────────────────────────

with st.sidebar:
    st.title("HwpAgent")
    st.caption(f"세션: `{st.session_state.session_id[:8]}…`")
    st.divider()

    # ── 양식 업로드 ──────────────────────────
    st.subheader("1. 양식 파일")
    form_file = st.file_uploader(
        "국가지원사업 양식 (.hwpx)",
        type=["hwpx"],
        key="form_uploader",
    )
    if form_file and not st.session_state.form_uploaded:
        with st.spinner(f"'{form_file.name}' 업로드 중..."):
            try:
                result = api.upload_file(
                    st.session_state.session_id,
                    form_file.read(),
                    form_file.name,
                    "form",
                )
                st.session_state.form_uploaded = True
                st.session_state.form_filename = form_file.name
                st.success(f"양식 업로드 완료: {form_file.name}")
                _msg = f"양식 파일 '{form_file.name}'이 업로드됐습니다."
                st.session_state.messages.append({"role": "assistant", "content": _msg})
            except Exception as e:
                st.error(f"업로드 실패: {e}")

    if st.session_state.form_uploaded:
        st.caption(f"양식: {st.session_state.form_filename}")

    st.divider()

    # ── 자료 업로드 ──────────────────────────
    st.subheader("2. 자료 파일")
    material_files = st.file_uploader(
        "CV, 논문, 이전 제안서 등 (PDF·DOCX·TXT)",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        key="material_uploader",
    )
    if material_files:
        new_files = [
            f for f in material_files
            if f.name not in st.session_state.material_filenames
        ]
        for mf in new_files:
            with st.spinner(f"'{mf.name}' 업로드 중..."):
                try:
                    api.upload_file(
                        st.session_state.session_id,
                        mf.read(),
                        mf.name,
                        "material",
                    )
                    st.session_state.material_filenames.append(mf.name)
                    st.success(f"자료 업로드: {mf.name}")
                except Exception as e:
                    st.error(f"'{mf.name}' 업로드 실패: {e}")

    if st.session_state.material_filenames:
        st.caption("업로드된 자료:")
        for name in st.session_state.material_filenames:
            st.caption(f"  · {name}")

    st.divider()

    # ── 채우기 시작 버튼 ─────────────────────
    fill_disabled = not st.session_state.form_uploaded
    if st.button(
        "양식 채우기 시작",
        disabled=fill_disabled,
        use_container_width=True,
        type="primary",
    ):
        st.session_state.messages.append({"role": "user", "content": "채우기 시작"})
        with st.spinner("양식 채우는 중..."):
            try:
                resp = api.send_chat(st.session_state.session_id, "채우기 시작")
                st.session_state.messages.append(
                    {"role": "assistant", "content": resp["reply"]}
                )
                st.session_state.form_ready = True
            except Exception as e:
                st.error(f"오류: {e}")
        st.rerun()

    if fill_disabled:
        st.caption("양식 파일을 먼저 업로드하세요.")

    st.divider()

    # ── 다운로드 버튼 ────────────────────────
    st.subheader("3. 완성 파일 다운로드")
    if st.session_state.form_ready:
        try:
            hwpx_bytes = api.download_hwpx(st.session_state.session_id)
            st.download_button(
                label="완성된 양식 다운로드 (.hwpx)",
                data=hwpx_bytes,
                file_name="output.hwpx",
                mime="application/octet-stream",
                use_container_width=True,
            )
            st.caption("PII 항목([본인 직접 입력])은 직접 작성해주세요.")
        except Exception as e:
            st.error(f"다운로드 실패: {e}")
    else:
        st.caption("채우기 완료 후 활성화됩니다.")


# ──────────────────────────────────────────────
# 메인 영역: 채팅 스레드
# ──────────────────────────────────────────────

st.title("HwpAgent — 한글 양식 자동 작성")
st.caption("국가지원사업 양식을 내 자료 기반으로 자동으로 채워드립니다.")

# 시작 안내 메시지
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "안녕하세요! 왼쪽 사이드바에서 **양식 파일(.hwpx)**과 **자료 파일**을 업로드하면 시작할 수 있습니다.\n\n"
            "- 양식: 국가지원사업 신청 양식 (.hwpx)\n"
            "- 자료: CV, 논문 목록, 이전 제안서 등 (PDF·DOCX·TXT)"
        )

# 대화 히스토리 렌더링
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── 자유 채팅 입력 ───────────────────────────
if prompt := st.chat_input("메시지를 입력하세요 (예: '3번 항목 더 공식적으로')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("처리 중..."):
            try:
                resp = api.send_chat(st.session_state.session_id, prompt)
                reply = resp["reply"]
            except Exception as e:
                reply = f"오류가 발생했습니다: {e}"
        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
