"""Load the repository's existing .env instead of keeping a second copy of the key.

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

# The repo root is one level up from this folder; that is where .env lives.
REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"


def load_env() -> None:
    """Load the repo-root .env into this process's environment variables."""
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
