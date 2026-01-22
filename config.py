from typing import Final,Literal
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

from llm.client import LLMChatClient, LLMConfig  # adjust path if different


class Settings(BaseSettings):
    # =========================
    # OpenAI
    # =========================
    openai_model: str
    openai_api_key: str

    # =========================
    # Claude (Bedrock)
    # =========================
    aws_region: str = "us-east-1"
    bedrock_model_id: str | None = None
    aws_profile: str | None = None


    # =========================
    # LLM Provider
    # =========================
    llm_provider: Literal["openai", "bedrock"] = "bedrock"
    llm_temperature: float = 0.0

    # =========================
    # Qwen
    # =========================
    qwen_embed_model: str


    openrouter_base_url: str =  "https://openrouter.ai/api/v1"
    openrouter_api_key: str

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
    # OSU Faculty Scraper
    # =========================
    osu_eng_base_url: str = "https://engineering.oregonstate.edu"
    osu_eng_list_path: str = "/people"
    scraper_timeout_secs: int = 20
    scraper_user_agent: str = (
        "Mozilla/5.0 (+faculty-link-scraper; OSU project)"
    )


    # =========================
    # Opportunity Extracted Content saved path
    # =========================
    extracted_content_path: Path = Path("/home/ec2-user/grant-matching-agent/data/extracted_text")
    opportunity_attachment_path: Path = Path("opportunities/attachments")
    opportunity_additional_link_path: Path = Path("opportunities/additional_links")

    # =========================
    # Faculty Extracted Content saved path
    # =========================
    faculty_additional_link_path: Path = Path("faculty/additional_links")

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
        case_sensitive=False
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
