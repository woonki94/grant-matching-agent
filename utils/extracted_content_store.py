import os
import boto3
from botocore.exceptions import ClientError
from typing import Optional

_BUCKET = os.getenv("EXTRACTED_CONTENT_BUCKET", "").strip()
_PREFIX = os.getenv("EXTRACTED_CONTENT_PREFIX", "").strip().strip("/")
_REGION = os.getenv("AWS_REGION", "us-west-2")

_s3 = boto3.client("s3", region_name=_REGION)


def build_s3_key(filename: str) -> str:
    # filename like "abc123.txt" or "opportunities/abc123.txt"
    filename = filename.lstrip("/")
    if _PREFIX:
        return f"{_PREFIX}/{filename}"
    return filename


def put_text(filename: str, text: str) -> str:
    """
    Upload text to S3 and return the S3 key.
    """
    if not _BUCKET:
        raise RuntimeError("EXTRACTED_CONTENT_BUCKET is not set")

    key = build_s3_key(filename)
    _s3.put_object(
        Bucket=_BUCKET,
        Key=key,
        Body=text.encode("utf-8"),
        ContentType="text/plain; charset=utf-8",
    )
    return key


def get_text(key_or_filename: str) -> Optional[str]:
    """
    Read text from S3 by full key (preferred) or filename (will be prefixed).
    Returns None if missing.
    """
    if not _BUCKET:
        raise RuntimeError("EXTRACTED_CONTENT_BUCKET is not set")

    key = key_or_filename
    # If caller passes just a filename, prefix it:
    if _PREFIX and not key_or_filename.startswith(f"{_PREFIX}/"):
        key = build_s3_key(key_or_filename)

    try:
        resp = _s3.get_object(Bucket=_BUCKET, Key=key)
        return resp["Body"].read().decode("utf-8")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("NoSuchKey", "404"):
            return None
        raise
