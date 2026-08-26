"""Load this folder's own .env so the service is self-contained.

Import this at the top of any script in this folder:

    from env_setup import load_env
    load_env()

Then read the key the normal way, e.g. ``OpenAI()`` picks up OPENAI_API_KEY on
its own. Run this file directly to confirm the key is being found:

    python env_setup.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# This folder's own .env. Copy .env.example to .env and fill it in; the file is
# gitignored and .dockerignored, so it reaches neither the repo nor the image --
# Render supplies the same names as real environment variables instead.
ENV_PATH = Path(__file__).resolve().parent / ".env"


def load_env() -> None:
    """Load this folder's .env into this process's environment variables."""
    load_dotenv(ENV_PATH)


def check() -> bool:
    """Report whether OPENAI_API_KEY is present. Never prints the value itself."""
    load_env()
    key = os.getenv("OPENAI_API_KEY") or ""
    print(f"reading: {ENV_PATH}")
    print(f".env exists: {ENV_PATH.is_file()}")
    print(f"OPENAI_API_KEY found: {bool(key)} (length {len(key)})")
    return bool(key)


if __name__ == "__main__":
    assert check(), f"OPENAI_API_KEY missing or empty in {ENV_PATH}"
    print("OK")
