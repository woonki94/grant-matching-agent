# client/llm_client.py  (Bedrock-only)
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import boto3
from langchain_aws import ChatBedrock


@dataclass(frozen=True)
class LLMConfig:
    # Bedrock only
    temperature: float = 0.0
    aws_region: Optional[str] = None
    aws_profile: Optional[str] = None
    bedrock_model_id: str = ""  # required


class LLMChatClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    def build(self) -> ChatBedrock:
        if not self.config.bedrock_model_id:
            raise ValueError("bedrock_model_id is required for Bedrock LLM")

        session = (
            boto3.Session(
                profile_name=self.config.aws_profile,
                region_name=self.config.aws_region,
            )
            if self.config.aws_profile
            else boto3.Session(region_name=self.config.aws_region)
        )

        bedrock_runtime = session.client("bedrock-runtime")

        return ChatBedrock(
            model_id=self.config.bedrock_model_id,
            client=bedrock_runtime,
            model_kwargs={"temperature": self.config.temperature},
        )
