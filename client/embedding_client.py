# embedding_client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import boto3
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_aws import BedrockEmbeddings


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: Literal["bedrock", "openrouter"]

    # Bedrock
    aws_region: Optional[str] = None
    aws_profile: Optional[str] = None
    bedrock_embed_model_id: Optional[str] = None  # e.g. "amazon.titan-embed-text-v2:0"

    # OpenRouter (OpenAI-compatible)
    openrouter_api_key: Optional[str] = None
    openrouter_base_url: Optional[str] = None
    openrouter_embed_model: Optional[str] = None  # e.g. "qwen/qwen3-embedding-0.6b"


class EmbeddingClient:
    def __init__(self, config: EmbeddingConfig):
        self.config = config

    def build(self) -> Embeddings:
        provider = (self.config.provider or "").lower().strip()


        if provider == "bedrock":
            if not self.config.aws_region:
                raise ValueError("aws_region is required for Bedrock embeddings")
            if not self.config.bedrock_embed_model_id:
                raise ValueError("bedrock_embed_model_id is required for Bedrock embeddings")

            session = (
                boto3.Session(profile_name=self.config.aws_profile, region_name=self.config.aws_region)
                if self.config.aws_profile
                else boto3.Session(region_name=self.config.aws_region)
            )
            bedrock_runtime = session.client("bedrock-runtime")

            return BedrockEmbeddings(
                model_id=self.config.bedrock_embed_model_id,
                client=bedrock_runtime,
            )

        if provider == "openrouter":
            if not self.config.openrouter_api_key:
                raise ValueError("openrouter_api_key is required for OpenRouter embeddings")
            if not self.config.openrouter_base_url:
                raise ValueError("openrouter_base_url is required for OpenRouter embeddings")
            if not self.config.openrouter_embed_model:
                raise ValueError("openrouter_embed_model is required for OpenRouter embeddings")

            return OpenAIEmbeddings(
                model=self.config.openrouter_embed_model,
                api_key=self.config.openrouter_api_key,
                base_url=self.config.openrouter_base_url.rstrip("/"),
            )

        raise ValueError(f"Unknown embeddings provider: {self.config.provider!r}")