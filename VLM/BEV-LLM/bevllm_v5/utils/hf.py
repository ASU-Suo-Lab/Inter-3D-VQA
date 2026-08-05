from __future__ import annotations

import os
from pathlib import Path

from packaging.version import Version

from bevllm_v5.utils.io import ensure


MIN_TRANSFORMERS_VERSION = "4.45.0"


def resolve_access_token(explicit_token: str | None) -> str | None:
    token = explicit_token or os.environ.get("HF_TOKEN")
    return token.strip() if isinstance(token, str) and token.strip() else None


def ensure_cache_dir(cache_dir: str | Path) -> Path:
    path = Path(cache_dir).expanduser().resolve()
    if path.exists():
        ensure(path.is_dir(), f"Model cache path is not a directory: {path}")
    else:
        path.mkdir(parents=True, exist_ok=True)
    ensure(os.access(path, os.W_OK), f"Model cache directory is not writable: {path}")
    return path


def ensure_transformers_version(min_version: str = MIN_TRANSFORMERS_VERSION) -> None:
    import transformers

    current = Version(transformers.__version__)
    required = Version(min_version)
    ensure(
        current >= required,
        (
            f"transformers>={min_version} is required for meta-llama/Llama-3.2-1B-Instruct "
            f"(current: {transformers.__version__}). The older version cannot parse Llama 3.2 rope_scaling."
        ),
    )


def _is_cached_file(model_id: str, filename: str, cache_dir: Path) -> bool:
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(model_id, filename, cache_dir=str(cache_dir))
    return isinstance(cached, str) and Path(cached).is_file()


def has_local_model_cache(model_id: str, cache_dir: str | Path) -> bool:
    root = Path(cache_dir).expanduser().resolve()
    has_config = _is_cached_file(model_id, "config.json", root)
    has_tokenizer = any(
        _is_cached_file(model_id, filename, root)
        for filename in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")
    )
    return has_config and has_tokenizer


def ensure_model_access(model_id: str, explicit_token: str | None, cache_dir: str | Path) -> None:
    cache_root = ensure_cache_dir(cache_dir)
    if has_local_model_cache(model_id, cache_root):
        return

    token = resolve_access_token(explicit_token)
    ensure(
        token is not None,
        (
            f"Access token is required to download {model_id}. "
            "Set HF_TOKEN or pass --access-token after accepting the gated model terms on Hugging Face."
        ),
    )

    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    api = HfApi(token=token)
    try:
        api.model_info(model_id)
    except GatedRepoError as exc:
        raise ValueError(
            f"Token does not have access to gated model {model_id}. Accept access at https://huggingface.co/{model_id}."
        ) from exc
    except RepositoryNotFoundError as exc:
        raise ValueError(f"Model repository not found or inaccessible: {model_id}") from exc
    except HfHubHTTPError as exc:
        raise ValueError(f"Unable to verify access to {model_id}: {exc}") from exc


def ensure_hf_runtime_ready(model_id: str, explicit_token: str | None, cache_dir: str | Path) -> Path:
    ensure_transformers_version()
    cache_root = ensure_cache_dir(cache_dir)
    ensure_model_access(model_id, explicit_token, cache_root)
    return cache_root
