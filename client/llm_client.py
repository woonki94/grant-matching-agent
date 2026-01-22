# llm_client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import boto3
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrock


@dataclass(frozen=True)
class LLMConfig:
    provider: str  # "openai" | "bedrock"
    temperature: float = 0.0

    # OpenAI
    openai_model: Optional[str] = None
    openai_api_key: Optional[str] = None

    # Bedrock
    aws_region: Optional[str] = None
    bedrock_model_id: Optional[str] = None
    aws_profile: Optional[str] = None


class LLMChatClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    def build(self):
        provider = (self.config.provider or "").lower().strip()

        if provider == "openai":
            return ChatOpenAI(
                model=self.config.openai_model,
                api_key=self.config.openai_api_key,
                temperature=self.config.temperature,
            )

        if provider == "bedrock":
            session = (
                boto3.Session(profile_name=self.config.aws_profile, region_name=self.config.aws_region)
                if self.config.aws_profile
                else boto3.Session(region_name=self.config.aws_region)
            )

            bedrock_runtime = session.client("bedrock-runtime")

            return ChatBedrock(
                model_id=self.config.bedrock_model_id,
                client=bedrock_runtime,
                model_kwargs={"temperature": self.config.temperature},
            )

        raise ValueError(f"Unknown provider: {self.config.provider!r}")