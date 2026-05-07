"""Upstage Solar LLM 클라이언트 래퍼 — Step 7.

Solar API는 OpenAI 호환 인터페이스를 제공한다.
openai 패키지에 base_url만 교체해서 사용한다.

기능:
  - 동기(call) / 비동기(acall) 모두 지원
  - 5xx / timeout / connection error 시 지수 백오프 재시도 (최대 3회)
  - 재시도 소진 후 SolarAPIError 발생 → 그래프 에러 핸들러가 처리
  - 프롬프트 템플릿 로드 유틸 (load_prompt)

사용 예:
    from app.llm.solar_client import client, load_prompt

    prompt = load_prompt("router", user_message="양식 올렸어요")
    reply = client.call([{"role": "user", "content": prompt}])
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import openai
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    RetryError,
)

from app.config import settings

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


# ──────────────────────────────────────────────
# 예외
# ──────────────────────────────────────────────

class SolarAPIError(RuntimeError):
    """재시도를 모두 소진하고도 실패한 경우."""


# ──────────────────────────────────────────────
# 재시도 조건
# ──────────────────────────────────────────────

def _is_retryable(exc: BaseException) -> bool:
    """재시도할 예외인지 판별한다."""
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code >= 500
    return isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError))


# ──────────────────────────────────────────────
# 클라이언트
# ──────────────────────────────────────────────

class SolarClient:
    """Upstage Solar API 클라이언트."""

    def __init__(self) -> None:
        _kwargs: dict[str, Any] = {
            "api_key": settings.upstage_api_key or "dummy",
            "base_url": settings.upstage_api_base,
            "timeout": 60.0,
        }
        self._sync = openai.OpenAI(**_kwargs)
        self._async = openai.AsyncOpenAI(**_kwargs)

    # ── 동기 호출 ────────────────────────────────

    def call(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """Solar API를 동기로 호출하고 응답 텍스트를 반환한다.

        Args:
            messages:    [{"role": "system"|"user"|"assistant", "content": str}]
            model:       None이면 settings.solar_pro_model 사용
            temperature: 0~1, 낮을수록 결정적
            max_tokens:  최대 출력 토큰 수

        Returns:
            응답 텍스트 (content)

        Raises:
            SolarAPIError: 재시도 소진 후 실패
        """
        try:
            return self._call_with_retry(messages, model, temperature, max_tokens)
        except RetryError as exc:
            raise SolarAPIError(f"Solar API 재시도 소진: {exc.last_attempt.exception()}") from exc

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(settings.llm_max_retries),
        wait=wait_exponential(
            min=settings.llm_retry_wait_min,
            max=settings.llm_retry_wait_max,
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=False,
    )
    def _call_with_retry(
        self,
        messages: list[dict[str, str]],
        model: str | None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        try:
            response = self._sync.chat.completions.create(
                model=model or settings.solar_pro_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except (openai.APIStatusError, openai.APITimeoutError, openai.APIConnectionError):
            raise
        except Exception as exc:
            raise SolarAPIError(f"Solar API 예기치 않은 오류: {exc}") from exc

    # ── 비동기 호출 ──────────────────────────────

    async def acall(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """Solar API를 비동기로 호출한다 (LangGraph async 노드용)."""
        return await self._acall_with_retry(messages, model, temperature, max_tokens)

    async def _acall_with_retry(
        self,
        messages: list[dict[str, str]],
        model: str | None,
        temperature: float,
        max_tokens: int,
        _attempt: int = 0,
    ) -> str:
        """tenacity가 async를 지원하지만 데코레이터 중첩이 복잡해 수동 구현."""
        max_attempts = settings.llm_max_retries
        wait_min = settings.llm_retry_wait_min
        wait_max = settings.llm_retry_wait_max

        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                response = await self._async.chat.completions.create(
                    model=model or settings.solar_pro_model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""
            except (openai.APIStatusError, openai.APITimeoutError, openai.APIConnectionError) as exc:
                if not _is_retryable(exc):
                    raise SolarAPIError(str(exc)) from exc
                last_exc = exc
                wait = min(wait_min * (2 ** attempt), wait_max)
                logger.warning("Solar API 재시도 %d/%d (%.1fs 후): %s", attempt + 1, max_attempts, wait, exc)
                import asyncio
                await asyncio.sleep(wait)
            except Exception as exc:
                raise SolarAPIError(f"Solar API 예기치 않은 오류: {exc}") from exc

        raise SolarAPIError(f"Solar API 재시도 {max_attempts}회 소진: {last_exc}") from last_exc


# ──────────────────────────────────────────────
# 프롬프트 템플릿 유틸
# ──────────────────────────────────────────────

def load_prompt(name: str, **kwargs: str) -> str:
    """prompts/{name}.txt 를 읽어 kwargs로 포맷팅한다.

    Args:
        name:   프롬프트 파일명 (확장자 제외, 예: "router")
        kwargs: 템플릿 변수 (예: user_message="...")

    Returns:
        포맷팅된 프롬프트 문자열

    Raises:
        FileNotFoundError: 프롬프트 파일 없음
    """
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"프롬프트 파일 없음: {path}")
    template = path.read_text(encoding="utf-8")
    return template.format(**kwargs) if kwargs else template


# ──────────────────────────────────────────────
# 전역 싱글턴
# ──────────────────────────────────────────────

client = SolarClient()


# ──────────────────────────────────────────────
# CLI 검증용 __main__
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if not settings.upstage_api_key:
        print("❌ UPSTAGE_API_KEY가 .env에 설정되지 않았습니다.")
        sys.exit(1)

    print("Solar API 연결 테스트 중...")
    try:
        reply = client.call(
            messages=[{"role": "user", "content": "안녕하세요. 한 문장으로 답해주세요."}],
            model=settings.solar_mini_model,
            max_tokens=50,
        )
        print(f"✅ 응답: {reply}")
    except SolarAPIError as e:
        print(f"❌ 실패: {e}")
        sys.exit(1)
