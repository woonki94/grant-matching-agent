import os
from pathlib import Path

import openai
from dotenv import load_dotenv
from openai import OpenAI


if __name__ == '__main__':

    env_path = Path(__file__).resolve().parents[1] / "api.env"
    loaded = load_dotenv(dotenv_path=env_path, override=True)
    openai_key = os.getenv("OPENAI_API_KEY")

    client = OpenAI(api_key=openai_key)

    resp = client.responses.create(
        model="gpt-5",
        input="Hello!"
    )

    print(resp.output_text)

