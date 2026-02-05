# client/embedding_client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import boto3
from langchain_core.embeddings import Embeddings
from langchain_aws import BedrockEmbeddings


@dataclass(frozen=True)
class EmbeddingConfig:
    # Bedrock only
    aws_region: Optional[str] = None
    aws_profile: Optional[str] = None
    bedrock_embed_model_id: str = ""  # required


class EmbeddingClient:
    def __init__(self, config: EmbeddingConfig):
        self.config = config

    def build(self) -> Embeddings:
        if not self.config.aws_region:
            raise ValueError("aws_region is required for Bedrock embeddings")
        if not self.config.bedrock_embed_model_id:
            raise ValueError("bedrock_embed_model_id is required for Bedrock embeddings")

        session = (
            boto3.Session(
                profile_name=self.config.aws_profile,
                region_name=self.config.aws_region,
            )
            if self.config.aws_profile
            else boto3.Session(region_name=self.config.aws_region)
        )

        bedrock_runtime = session.client("bedrock-runtime")

        return BedrockEmbeddings(
            model_id=self.config.bedrock_embed_model_id,
            client=bedrock_runtime,
        )
