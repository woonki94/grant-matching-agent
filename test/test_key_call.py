from pathlib import Path
from dotenv import load_dotenv
import os

# Absolute or relative path to your .env
env_path = Path(__file__).resolve().parents[0] / "api.env"

print(f"Looking for: {env_path}")

# Load and verify
loaded = load_dotenv(dotenv_path=env_path, override=True)
print(f"load_dotenv returned: {loaded}")

# Debug info
print(f"Working dir: {os.getcwd()}")
print(f" Interpreter: {os.sys.executable}")

# Print key safely (no full value)
api_key = os.getenv("API_KEY")
print(f" SIMPLER_API_KEY present? {bool(api_key)}")
if api_key:
    print(f"   Starts with: {api_key[:5]}...")
else:
    print("Missing key!")