from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Final, Literal, Optional

from pydantic import computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from client.llm_client import LLMChatClient, LLMConfig
from client.embedding_client import EmbeddingClient, EmbeddingConfig

BASE_DIR = Path(__file__).resolve().parent

LLMProvider = Literal["bedrock"]
EmbeddingProvider = Literal["bedrock"]


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
    # Providers (Bedrock only)
    # =========================
    llm_provider: LLMProvider = "bedrock"
    embedding_provider: EmbeddingProvider = "bedrock"

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _normalize_llm_provider(cls, v):
        if v is None:
            return "bedrock"
        return str(v).strip().lower()

    @field_validator("embedding_provider", mode="before")
    @classmethod
    def _normalize_embedding_provider(cls, v):
        if v is None:
            return "bedrock"
        return str(v).strip().lower()

    # =========================
    # AWS / Bedrock
    # =========================
    aws_region: str = "us-east-2"
    aws_profile: Optional[str] = None

    # Bedrock LLM models
    bedrock_claude_haiku: str
    bedrock_claude_sonnet: Optional[str] = None
    bedrock_claude_opus: Optional[str] = None
    bedrock_embed_model_id: str

    # =========================
    # LLM config
    # =========================
    llm_temperature: float = 0.0

    # Keep if your code expects it; tune per embedding model if needed.
    @computed_field
    @property
    def embed_dim(self) -> int:
        return 1024

    @computed_field
    @property
    def haiku(self) -> str:
        return self.bedrock_claude_haiku

    @computed_field
    @property
    def sonnet(self) -> Optional[str]:
        return self.bedrock_claude_sonnet

    @computed_field
    @property
    def opus(self) -> Optional[str]:
        return self.bedrock_claude_opus

    # =========================
    # Extracted Content Storage (S3 ONLY)
    # =========================
    extracted_content_bucket: str  # REQUIRED
    extracted_content_prefix_opportunity: str = ""
    extracted_content_prefix_faculty: str = ""

    @computed_field
    @property
    def extracted_content_s3_uri_opportunity(self) -> str:
        prefix = (self.extracted_content_prefix_opportunity or "").strip("/")
        return (
            f"s3://{self.extracted_content_bucket}/{prefix}"
            if prefix
            else f"s3://{self.extracted_content_bucket}"
        )

    @computed_field
    @property
    def extracted_content_s3_uri_faculty(self) -> str:
        prefix = (self.extracted_content_prefix_faculty or "").strip("/")
        return (
            f"s3://{self.extracted_content_bucket}/{prefix}"
            if prefix
            else f"s3://{self.extracted_content_bucket}"
        )

    # =========================
    # Pipeline local scratch paths (safe defaults)
    # (These are NOT extracted-content storage; they’re local temp paths.)
    # =========================
    opportunity_attachment_path: Path = BASE_DIR / "data" / "opportunity_attachments"
    opportunity_additional_link_path: Path = BASE_DIR / "data" / "opportunity_additional_links"
    faculty_additional_link_path: Path = BASE_DIR / "data" / "faculty_additional_links"

    # =========================
    # OSU Faculty Scraper
    # =========================
    osu_eng_base_url: str = "https://engineering.oregonstate.edu"
    osu_eng_list_path: str = "/people"
    scraper_timeout_secs: int = 20
    scraper_user_agent: str = "Mozilla/5.0 (+faculty-link-scraper; OSU project)"

    # =========================
    # University Name
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
Grant_API_KEY: Final[str] = settings.grant_api_key


@lru_cache(maxsize=8)
def get_llm_client(model_id: Optional[str] = None) -> LLMChatClient:
    use_model_id = (model_id or settings.haiku or "").strip()
    if not use_model_id:
        raise ValueError("No Bedrock model id configured. Set BEDROCK_CLAUDE_HAIKU.")

    return LLMChatClient(
        LLMConfig(
            temperature=settings.llm_temperature,
            aws_region=settings.aws_region,
            aws_profile=settings.aws_profile,
            bedrock_model_id=use_model_id,
        )
    )


@lru_cache(maxsize=1)
def get_embedding_client() -> EmbeddingClient:
    return EmbeddingClient(
        EmbeddingConfig(
            aws_region=settings.aws_region,
            aws_profile=settings.aws_profile,
            bedrock_embed_model_id=settings.bedrock_embed_model_id,
        )
    )
