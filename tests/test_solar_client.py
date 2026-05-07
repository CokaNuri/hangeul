"""Solar 클라이언트 단위 테스트.

API 키 없이도 동작하는 mock 테스트와
UPSTAGE_API_KEY가 있을 때만 실행하는 실제 호출 테스트를 분리한다.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from app.llm.solar_client import SolarClient, SolarAPIError, load_prompt


# ── 픽스처 ────────────────────────────────────

@pytest.fixture
def solar():
    return SolarClient()


def _make_response(content: str) -> MagicMock:
    """openai ChatCompletion 응답 mock 생성."""
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ── 동기 call ─────────────────────────────────

class TestSyncCall:
    def test_returns_text(self, solar):
        with patch.object(solar._sync.chat.completions, "create", return_value=_make_response("안녕하세요")):
            result = solar.call([{"role": "user", "content": "안녕"}])
        assert result == "안녕하세요"

    def test_uses_pro_model_by_default(self, solar):
        from app.config import settings
        with patch.object(solar._sync.chat.completions, "create", return_value=_make_response("ok")) as mock_create:
            solar.call([{"role": "user", "content": "test"}])
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == settings.solar_pro_model

    def test_custom_model(self, solar):
        from app.config import settings
        with patch.object(solar._sync.chat.completions, "create", return_value=_make_response("ok")) as mock_create:
            solar.call([{"role": "user", "content": "test"}], model=settings.solar_mini_model)
        assert mock_create.call_args.kwargs["model"] == settings.solar_mini_model

    def test_empty_content_returns_empty_string(self, solar):
        with patch.object(solar._sync.chat.completions, "create", return_value=_make_response(None)):
            result = solar.call([{"role": "user", "content": "test"}])
        assert result == ""

    def test_server_error_raises_solar_api_error(self, solar):
        err = openai.APIStatusError(
            "Internal Server Error",
            response=MagicMock(status_code=500),
            body={},
        )
        with patch.object(solar._sync.chat.completions, "create", side_effect=err):
            with pytest.raises((SolarAPIError, openai.APIStatusError)):
                solar.call([{"role": "user", "content": "test"}])

    def test_non_retryable_status_not_retried(self, solar):
        """400 에러는 재시도하지 않아야 한다."""
        err = openai.APIStatusError(
            "Bad Request",
            response=MagicMock(status_code=400),
            body={},
        )
        with patch.object(solar._sync.chat.completions, "create", side_effect=err) as mock_create:
            with pytest.raises(openai.APIStatusError):
                solar.call([{"role": "user", "content": "test"}])
        # 재시도 없이 1번만 호출
        assert mock_create.call_count == 1


# ── 비동기 acall ──────────────────────────────

class TestAsyncCall:
    @pytest.mark.asyncio
    async def test_returns_text(self, solar):
        with patch.object(
            solar._async.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=_make_response("비동기 응답"),
        ):
            result = await solar.acall([{"role": "user", "content": "테스트"}])
        assert result == "비동기 응답"

    @pytest.mark.asyncio
    async def test_retries_on_500(self, solar):
        """500 에러 시 재시도 후 성공하는 시나리오."""
        err = openai.APIStatusError(
            "Internal Server Error",
            response=MagicMock(status_code=500),
            body={},
        )
        call_count = 0

        async def _flaky_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise err
            return _make_response("재시도 성공")

        with patch.object(solar._async.chat.completions, "create", side_effect=_flaky_create):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await solar.acall([{"role": "user", "content": "test"}])
        assert result == "재시도 성공"
        assert call_count == 2


# ── load_prompt ───────────────────────────────

class TestLoadPrompt:
    def test_router_prompt_loads(self):
        prompt = load_prompt("router", user_message="양식 올렸어요")
        assert "양식 올렸어요" in prompt
        assert "intent" in prompt

    def test_planner_prompt_loads(self):
        prompt = load_prompt("planner", form_items_json="[]", materials_summary="없음")
        assert "form_items_json" not in prompt  # 치환 완료
        assert "[]" in prompt

    def test_generator_prompt_loads(self):
        prompt = load_prompt(
            "generator",
            item_id="p0",
            item_label="연구 목표",
            item_type="text",
            char_hint="500",
            tone_guide="공식적",
            evidence_text="딥러닝 연구",
            user_answer="",
        )
        assert "연구 목표" in prompt

    def test_verifier_prompt_loads(self):
        prompt = load_prompt(
            "verifier",
            item_label="연구 목표",
            evidence_text="딥러닝 연구",
            draft_text="딥러닝 기반 NLP 연구를 수행합니다.",
        )
        assert "딥러닝 기반 NLP 연구" in prompt

    def test_missing_prompt_raises(self):
        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent_prompt_xyz")

    def test_prompt_without_kwargs(self):
        """kwargs 없이 호출해도 템플릿 원문이 반환돼야 한다."""
        raw = load_prompt("router")
        assert "{user_message}" in raw  # 미치환 상태


# ── 실제 API 호출 (UPSTAGE_API_KEY 있을 때만) ──

@pytest.mark.skipif(
    not __import__("os").getenv("UPSTAGE_API_KEY"),
    reason="UPSTAGE_API_KEY 환경변수 없음",
)
class TestLiveCall:
    def test_live_solar_mini(self):
        from app.config import settings
        live_client = SolarClient()
        result = live_client.call(
            messages=[{"role": "user", "content": "한 단어로만 답하세요: 하늘은 무슨 색?"}],
            model=settings.solar_mini_model,
            max_tokens=20,
        )
        assert isinstance(result, str)
        assert len(result) > 0
