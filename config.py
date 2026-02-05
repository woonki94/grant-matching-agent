from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Final, Literal, Optional

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

from client.llm_client import LLMChatClient, LLMConfig  # adjust path if different
from client.embedding_client import EmbeddingClient, EmbeddingConfig

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    # =========================
    # Postgres Settings
    # =========================
    pguser: str
    pgpassword: str
    pghost: str
    pgport: int = 5432
    pgdatabase: str

    @computed_field
    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.pguser}:{self.pgpassword}"
            f"@{self.pghost}:{self.pgport}/{self.pgdatabase}"
        )

    # =========================
    # Grant.gov
    # =========================
    grant_api_key: str
    simpler_search_url: str
    simpler_detail_base_url: str

    # =========================
    # OpenAlex: fetching publications
    # =========================
    openalex_base_url: str = "https://api.openalex.org"

    # =========================
    # LLM+EMBEDDING Provider
    # =========================
    llm_provider: Literal["openai", "bedrock"]
    embedding_provider: Literal["openrouter", "bedrock"]

    # =========================
    # OpenAI
    # =========================
    openai_model: str
    openai_api_key: str

    # =========================
    # Qwen (OpenRouter)
    # =========================
    qwen_embed_model: str
    openrouter_base_url: str
    openrouter_api_key: str

    # =========================
    # AWS
    # =========================
    aws_region: str = "us-east-2"
    aws_profile: Optional[str] = None

    # =========================
    # Claude (Bedrock)
    # =========================
    bedrock_model_id: Optional[str] = None

    # =========================
    # Embeddings (Bedrock)
    # =========================
    bedrock_embed_model_id: Optional[str] = None

    # =========================
    # LLM config
    # =========================
    llm_temperature: float = 0.0

    @computed_field
    @property
    def embed_dim(self) -> int:
        return 4096 if self.embedding_provider == "openrouter" else 1024

    # =========================
    # Extracted Content Storage
    # =========================
    # Env you provided:
    #   EXTRACTED_CONTENT_BACKEND=s3
    #   EXTRACTED_CONTENT_BUCKET=grant-matcher
    #   EXTRACTED_CONTENT_PREFIX=extracted-context-opportunities
    #   AWS_REGION=us-east-2
    extracted_content_backend: Literal["local", "s3"] = "local"

    # Local backend (only used if backend=local)
    extracted_content_path: Optional[Path] = None

    # S3 backend (required if backend=s3)
    extracted_content_bucket: Optional[str] = None
    extracted_content_prefix: str = ""

    @computed_field
    @property
    def extracted_content_enabled(self) -> bool:
        """
        Convenience flag: True when extraction storage is configured enough to write.
        """
        if self.extracted_content_backend == "local":
            return self.extracted_content_path is not None
        if self.extracted_content_backend == "s3":
            return bool(self.extracted_content_bucket)
        return False

    @computed_field
    @property
    def extracted_content_s3_uri(self) -> Optional[str]:
        """
        Convenience URI for logging/debugging.
        Returns None unless backend=s3 and bucket is set.
        """
        if self.extracted_content_backend != "s3" or not self.extracted_content_bucket:
            return None
        prefix = (self.extracted_content_prefix or "").strip("/")
        return (
            f"s3://{self.extracted_content_bucket}/{prefix}"
            if prefix
            else f"s3://{self.extracted_content_bucket}"
        )

    # NOTE:
    # We intentionally DO NOT mkdir anything here.
    # Any code that previously did:
    #   CONTENT_BASE_DIR = settings.extracted_content_path
    #   CONTENT_BASE_DIR.mkdir(...)
    # must be updated to:
    #   if settings.extracted_content_backend == "local": mkdir(...)
    #   if "s3": do NOT mkdir.

    # =========================
    # Pipeline local scratch paths (safe defaults)
    # =========================
    # If your pipeline temporarily downloads files before uploading to S3,
    # these defaults avoid Settings validation errors.
    opportunity_attachment_path: Path = BASE_DIR / "data" / "opportunity_attachments"
    opportunity_additional_link_path: Path = (
        BASE_DIR / "data" / "opportunity_additional_links"
    )
    faculty_additional_link_path: Path = BASE_DIR / "data" / "faculty_additional_links"

    # =========================
    # OSU Faculty Scraper
    # =========================
    osu_eng_base_url: str = "https://engineering.oregonstate.edu"
    osu_eng_list_path: str = "/people"
    scraper_timeout_secs: int = 20
    scraper_user_agent: str = "Mozilla/5.0 (+faculty-link-scraper; OSU project)"

    # =========================
    # University Name (needed for publication extraction)
    # =========================
    university_name: str = "Oregon State University"

    # =========================
    # LOG Config
    # =========================
    log_level: str = "INFO"
    log_dir: Path = BASE_DIR / "logs"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()

# =========================
# Frequently used aliases: Constant
# =========================
OPENAI_API_KEY: Final[str] = settings.openai_api_key
OPENAI_MODEL: Final[str] = settings.openai_model

Grant_API_KEY: Final[str] = settings.grant_api_key
OPENROUTER_API_KEY: Final[str] = settings.openrouter_api_key
QWEN_MODEL: Final[str] = settings.qwen_embed_model


@lru_cache(maxsize=1)
def get_llm_client() -> LLMChatClient:
    return LLMChatClient(
        LLMConfig(
            provider=settings.llm_provider,
            temperature=settings.llm_temperature,
            openai_model=settings.openai_model,
            openai_api_key=settings.openai_api_key,
            aws_region=settings.aws_region,
            bedrock_model_id=settings.bedrock_model_id,
            aws_profile=settings.aws_profile,
        )
    )


@lru_cache(maxsize=1)
def get_embedding_client() -> EmbeddingClient:
    return EmbeddingClient(
        EmbeddingConfig(
            provider=settings.embedding_provider,
            # Bedrock
            aws_region=settings.aws_region,
            aws_profile=settings.aws_profile,
            bedrock_embed_model_id=settings.bedrock_embed_model_id,
            # OpenRouter(Qwen)
            openrouter_api_key=settings.openrouter_api_key,
            openrouter_base_url=settings.openrouter_base_url,
            openrouter_embed_model=settings.qwen_embed_model,
        )
    )
