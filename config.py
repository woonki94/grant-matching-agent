from typing import Final

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    # =========================
    # OpenAI
    # =========================
    openai_model: str
    openai_api_key: str


    # =========================
    # Qwen
    # =========================
    qwen_embed_model: str

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