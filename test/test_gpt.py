import os
from pathlib import Path

from dotenv import load_dotenv
import openai


if __name__ == '__main__':

    env_path = Path(__file__).resolve().parents[1] / "api.env"
    loaded = load_dotenv(dotenv_path=env_path, override=True)
    openai_key = os.getenv("OPENAI_API_KEY")

    client = openai.OpenAI(api_key=openai_key)

    print(openai_key)
    print(client)



