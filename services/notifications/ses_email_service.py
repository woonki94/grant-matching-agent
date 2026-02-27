from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import boto3

from config import settings


@dataclass(frozen=True)
class SesEmailConfig:
    aws_region: str
    aws_profile: Optional[str]
    from_email: str
    reply_to_email: Optional[str] = None
    configuration_set: Optional[str] = None


class SesEmailService:
    def __init__(self, config: Optional[SesEmailConfig] = None):
        if config is None:
            from_email = str(settings.ses_from_email or "").strip()
            if not from_email:
                raise RuntimeError("SES sender is not configured. Set SES_FROM_EMAIL in .env.")
            config = SesEmailConfig(
                aws_region=str(settings.aws_region or "us-east-2"),
                aws_profile=(str(settings.aws_profile).strip() if settings.aws_profile else None),
                from_email=from_email,
                reply_to_email=(str(settings.ses_reply_to_email).strip() if settings.ses_reply_to_email else None),
                configuration_set=(str(settings.ses_configuration_set).strip() if settings.ses_configuration_set else None),
            )

        self.config = config
        session = (
            boto3.Session(
                profile_name=self.config.aws_profile,
                region_name=self.config.aws_region,
            )
            if self.config.aws_profile
            else boto3.Session(region_name=self.config.aws_region)
        )
        self.client = session.client("sesv2")

    def send_email(
        self,
        *,
        to_addresses: List[str],
        subject: str,
        text_body: str,
        html_body: Optional[str] = None,
    ) -> Dict[str, str]:
        recipients = [str(x).strip() for x in list(to_addresses or []) if str(x).strip()]
        if not recipients:
            raise ValueError("to_addresses is required")
        if not subject.strip():
            raise ValueError("subject is required")
        if not text_body.strip():
            raise ValueError("text_body is required")

        body: Dict[str, Dict[str, str]] = {"Text": {"Data": text_body, "Charset": "UTF-8"}}
        if html_body and html_body.strip():
            body["Html"] = {"Data": html_body, "Charset": "UTF-8"}

        payload = {
            "FromEmailAddress": self.config.from_email,
            "Destination": {"ToAddresses": recipients},
            "Content": {
                "Simple": {
                    "Subject": {"Data": subject, "Charset": "UTF-8"},
                    "Body": body,
                }
            },
        }
        if self.config.reply_to_email:
            payload["ReplyToAddresses"] = [self.config.reply_to_email]
        if self.config.configuration_set:
            payload["ConfigurationSetName"] = self.config.configuration_set

        out = self.client.send_email(**payload)
        return {"message_id": str(out.get("MessageId") or "")}
