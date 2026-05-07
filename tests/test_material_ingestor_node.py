"""MaterialIngestor 노드 단위 테스트 — Step 11.

Solar API는 mock으로 대체해 네트워크 없이 실행한다.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.graph.nodes.material_ingestor import (
    material_ingestor_node,
    _chunk_text,
    _MAX_CHUNK_CHARS,
    _CHUNK_OVERLAP,
)
from app.graph.state import initial_state
from app.models import MaterialBundle, MaterialDoc

_MOCK_SUMMARY = "연구 분야는 자연어 처리이며, 주요 성과는 논문 3편 게재입니다."


# ── 픽스처 ────────────────────────────────────

@pytest.fixture
def txt_bundle() -> MaterialBundle:
    return MaterialBundle(docs=[
        MaterialDoc(
            source_name="cv.txt",
            masked_text="연구 분야: NLP. 논문 3편. [EMAIL_0] 마스킹 완료.",
            summary="",
            doc_type="txt",
        )
    ])


@pytest.fixture
def multi_bundle() -> MaterialBundle:
    return MaterialBundle(docs=[
        MaterialDoc(source_name="cv.txt",    masked_text="CV 내용입니다.", summary="", doc_type="txt"),
        MaterialDoc(source_name="paper.txt", masked_text="논문 내용입니다.", summary="", doc_type="txt"),
    ])


@pytest.fixture
def presummarized_bundle() -> MaterialBundle:
    """이미 summary가 있는 문서는 재요약하지 않아야 한다."""
    return MaterialBundle(docs=[
        MaterialDoc(
            source_name="done.txt",
            masked_text="텍스트",
            summary="기존 요약입니다.",
            doc_type="txt",
        )
    ])


@pytest.fixture
def empty_text_bundle() -> MaterialBundle:
    return MaterialBundle(docs=[
        MaterialDoc(source_name="empty.txt", masked_text="", summary="", doc_type="txt")
    ])


# ── 기본 동작 ─────────────────────────────────

@patch("app.graph.nodes.material_ingestor.client.call", return_value=_MOCK_SUMMARY)
def test_summary_generated(mock_call, txt_bundle):
    """요약이 빈 문서는 LLM 호출 후 summary가 채워져야 한다."""
    state = initial_state()
    state["material_bundle"] = txt_bundle

    result = material_ingestor_node(state)

    assert "material_bundle" in result
    doc = result["material_bundle"].docs[0]
    assert doc.summary == _MOCK_SUMMARY
    assert mock_call.called


@patch("app.graph.nodes.material_ingestor.client.call", return_value=_MOCK_SUMMARY)
def test_all_docs_summarized(mock_call, multi_bundle):
    """bundle 내 모든 문서에 요약이 생성돼야 한다."""
    state = initial_state()
    state["material_bundle"] = multi_bundle

    result = material_ingestor_node(state)

    docs = result["material_bundle"].docs
    assert len(docs) == 2
    assert all(d.summary for d in docs)


@patch("app.graph.nodes.material_ingestor.client.call", return_value=_MOCK_SUMMARY)
def test_existing_summary_not_overwritten(mock_call, presummarized_bundle):
    """이미 summary가 있는 문서는 LLM을 호출하지 않아야 한다."""
    state = initial_state()
    state["material_bundle"] = presummarized_bundle

    result = material_ingestor_node(state)

    mock_call.assert_not_called()
    doc = result["material_bundle"].docs[0]
    assert doc.summary == "기존 요약입니다."


@patch("app.graph.nodes.material_ingestor.client.call", return_value=_MOCK_SUMMARY)
def test_empty_text_skips_llm(mock_call, empty_text_bundle):
    """텍스트가 없는 문서는 LLM을 호출하지 않아야 한다."""
    state = initial_state()
    state["material_bundle"] = empty_text_bundle

    result = material_ingestor_node(state)

    mock_call.assert_not_called()


# ── bundle 없음 처리 ──────────────────────────

def test_no_bundle_returns_empty_dict():
    """material_bundle이 없으면 빈 dict를 반환해야 한다."""
    state = initial_state()
    state["material_bundle"] = None

    result = material_ingestor_node(state)
    assert result == {}


def test_empty_bundle_returns_empty_dict():
    """docs가 없는 bundle도 빈 dict를 반환해야 한다."""
    state = initial_state()
    state["material_bundle"] = MaterialBundle(docs=[])

    result = material_ingestor_node(state)
    assert result == {}


# ── PII 안전망 ────────────────────────────────

@patch("app.graph.nodes.material_ingestor.client.call", return_value=_MOCK_SUMMARY)
def test_pii_masked_text_sent_to_llm(mock_call, txt_bundle):
    """LLM에는 masked_text가 전달돼야 한다 (원본 PII 아님)."""
    state = initial_state()
    state["material_bundle"] = txt_bundle

    material_ingestor_node(state)

    call_args = mock_call.call_args
    prompt_content = call_args[1]["messages"][0]["content"]
    # [EMAIL_0] 마스킹 토큰이 프롬프트에 있고, 실제 이메일이 없어야 함
    assert "[EMAIL_0]" in prompt_content


# ── Solar Mini 모델 사용 확인 ─────────────────

@patch("app.graph.nodes.material_ingestor.client.call", return_value=_MOCK_SUMMARY)
def test_uses_solar_mini_model(mock_call, txt_bundle):
    """Solar Mini 모델을 사용해야 한다."""
    from app.config import settings

    state = initial_state()
    state["material_bundle"] = txt_bundle

    material_ingestor_node(state)

    call_args = mock_call.call_args
    assert call_args[1]["model"] == settings.solar_mini_model


# ── Solar API 실패 처리 ───────────────────────

@patch("app.graph.nodes.material_ingestor.client.call",
       side_effect=__import__("app.llm.solar_client", fromlist=["SolarAPIError"]).SolarAPIError("timeout"))
def test_api_error_leaves_summary_empty(mock_call, txt_bundle):
    """Solar API 실패 시 summary는 빈 문자열이고 에러가 전파되지 않아야 한다."""
    state = initial_state()
    state["material_bundle"] = txt_bundle

    result = material_ingestor_node(state)

    doc = result["material_bundle"].docs[0]
    assert doc.summary == ""


# ── 청킹 단위 테스트 ──────────────────────────

def test_short_text_single_chunk():
    """짧은 텍스트는 청크 1개여야 한다."""
    text = "짧은 텍스트"
    chunks = _chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_long_text_multiple_chunks():
    """_MAX_CHUNK_CHARS를 초과하면 여러 청크로 분할해야 한다."""
    text = "가" * (_MAX_CHUNK_CHARS + 1000)
    chunks = _chunk_text(text)
    assert len(chunks) >= 2


def test_chunks_cover_full_text():
    """모든 청크를 합치면 원본 텍스트의 내용이 모두 포함돼야 한다."""
    text = "A" * (_MAX_CHUNK_CHARS * 2 + 500)
    chunks = _chunk_text(text)
    # 각 청크가 원본의 일부를 포함
    assert chunks[0].startswith("A")
    assert chunks[-1].endswith("A")


def test_chunk_overlap():
    """인접한 청크 사이에 오버랩이 있어야 한다."""
    text = "X" * (_MAX_CHUNK_CHARS + _CHUNK_OVERLAP + 100)
    chunks = _chunk_text(text)
    assert len(chunks) >= 2
    # 첫 청크 끝 부분이 두 번째 청크 시작에 포함돼야 함
    overlap_region = chunks[0][-_CHUNK_OVERLAP:]
    assert chunks[1].startswith(overlap_region)


@patch("app.graph.nodes.material_ingestor.client.call", return_value=_MOCK_SUMMARY)
def test_long_doc_multiple_llm_calls(mock_call):
    """긴 문서는 청크마다 LLM을 호출하고 취합 요약을 생성해야 한다."""
    long_text = "연구 내용. " * 2000   # ~15,000자
    bundle = MaterialBundle(docs=[
        MaterialDoc(source_name="long.txt", masked_text=long_text, summary="", doc_type="txt")
    ])
    state = initial_state()
    state["material_bundle"] = bundle

    result = material_ingestor_node(state)

    # 청크 요약 + 취합 요약으로 2회 이상 호출
    assert mock_call.call_count >= 2
    doc = result["material_bundle"].docs[0]
    assert doc.summary != ""
