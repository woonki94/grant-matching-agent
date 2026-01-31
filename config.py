from typing import Final,Literal
from functools import lru_cache

from pydantic import computed_field, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

from client.llm_client import LLMChatClient, LLMConfig  # adjust path if different
from client.embedding_client import EmbeddingClient,EmbeddingConfig


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
    # Qwen
    # =========================
    qwen_embed_model: str
    openrouter_base_url: str
    openrouter_api_key: str

    #AWS
    aws_region: str = "us-east-2"
    aws_profile: str | None = None
    # =========================
    # Claude (Bedrock)
    # =========================
    bedrock_model_id: str | None = None
    # =========================
    # Embeddings (Bedrock)
    # =========================
    bedrock_embed_model_id: str | None = None

    # =========================
    # LLM config
    # =========================
    llm_temperature: float = 0.0

    @computed_field
    @property
    def embed_dim(self) -> int:
        return 4096 if self.embedding_provider == "openrouter" else 1024

    # =========================
    # Opportunity Extracted Content saved path
    # =========================
    extracted_content_path: Path
    opportunity_attachment_path: Path
    opportunity_additional_link_path: Path

    # =========================
    # Faculty Extracted Content saved path
    # =========================
    faculty_additional_link_path: Path


    # =========================
    # OSU Faculty Scraper
    # =========================
    osu_eng_base_url: str = "https://engineering.oregonstate.edu"
    osu_eng_list_path: str = "/people"
    scraper_timeout_secs: int = 20
    scraper_user_agent: str = (
        "Mozilla/5.0 (+faculty-link-scraper; OSU project)"
    )


    # =========================
    # University Name(needed for publication extraction)
    # =========================
    university_name: str = "Oregon State University"

    #======================
    #LOG Config
    #=====================
    log_level: str = "INFO"
    log_dir: Path = Path(BASE_DIR) / "logs"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR/".env",
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
OPENROUTER_API_KEY : Final[str] = settings.openrouter_api_key
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