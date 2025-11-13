import os

from dotenv import load_dotenv
from google import genai

load_dotenv("../api.env")  # or just .env
key = os.getenv("GEMINI_API_KEY")
#print(key)
client = genai.Client(api_key=key)



response = client.models.generate_content(
    model="gemini-2.5-flash", contents="Explain how AI works in a few words"
)
print(response.text)