from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Upstage Solar API
    upstage_api_key: str = ""
    upstage_api_base: str = "https://api.upstage.ai/v1/solar"
    solar_pro_model: str = "solar-pro"
    solar_mini_model: str = "solar-mini"

    # FastAPI
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # 세션
    session_ttl_seconds: int = 3600
    history_window_size: int = 10

    # Verifier 재시도
    verifier_max_retries: int = 2

    # LLM 재시도
    llm_max_retries: int = 3
    llm_retry_wait_min: int = 1
    llm_retry_wait_max: int = 8

    # 로그
    log_level: str = "INFO"


settings = Settings()
