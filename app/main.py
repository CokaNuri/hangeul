"""HwpAgent FastAPI 애플리케이션 진입점."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings
from app.session_store import store

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 만료 세션 주기적 GC (60초마다)
# ──────────────────────────────────────────────

async def _gc_loop() -> None:
    while True:
        await asyncio.sleep(60)
        removed = store.purge_expired()
        if removed:
            logger.info("GC: %d 만료 세션 삭제", removed)


@asynccontextmanager
async def lifespan(app: FastAPI):
    gc_task = asyncio.create_task(_gc_loop())
    logger.info("HwpAgent 서버 시작 (세션 GC 활성화)")
    yield
    gc_task.cancel()
    logger.info("HwpAgent 서버 종료")


# ──────────────────────────────────────────────
# 앱 생성
# ──────────────────────────────────────────────

app = FastAPI(
    title="HwpAgent API",
    description="국가지원사업 .hwpx 양식 자동 작성 AI 어시스턴트",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit 기본 포트
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
