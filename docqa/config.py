from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="DOCQA_", extra="ignore")

    database_url: str = "postgresql://docqa:docqa@localhost:5432/docqa"

    # Chunking: words per chunk and overlap between consecutive chunks of a page.
    chunk_words: int = 180
    chunk_overlap: int = 40

    # Retrieval
    top_k: int = 5
    rrf_k: int = 60            # Reciprocal Rank Fusion constant
    candidates: int = 30       # results fetched from each retriever before fusion

    # LLM. Without an API key the service runs in "extractive" mode and returns
    # the best passages instead of a generated answer.
    llm_api_key: str | None = None
    llm_model: str = "claude-sonnet-5-5"
    llm_max_tokens: int = 700

    max_upload_mb: int = 25


@lru_cache
def get_settings() -> Settings:
    return Settings()
