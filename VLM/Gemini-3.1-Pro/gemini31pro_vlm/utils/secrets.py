from __future__ import annotations

import json
import os
from pathlib import Path

from gemini31pro_vlm.config.common import API_KEY_ENV, LOCAL_API_KEY_FILE
from gemini31pro_vlm.utils.io import ensure


def resolve_api_key() -> str:
    local_key = _load_local_api_key(LOCAL_API_KEY_FILE)
    env_key = os.environ.get(API_KEY_ENV, "").strip()
    api_key = local_key or env_key
    ensure(api_key, f"{API_KEY_ENV} is required. Set the environment variable or create {LOCAL_API_KEY_FILE}.")
    return api_key


def _load_local_api_key(path: Path) -> str:
    if not path.is_file():
        return ""
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    ensure(isinstance(payload, dict), f"{path} must contain a JSON object.")
    raw = payload.get("api_key", payload.get(API_KEY_ENV))
    ensure(raw is None or isinstance(raw, str), f"{path} must store the API key as a string.")
    return "" if raw is None else raw.strip()
