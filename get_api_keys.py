from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="secrets.env")

def get_secret(key: str, default: str = None) -> str:
    return os.getenv(key, default)
