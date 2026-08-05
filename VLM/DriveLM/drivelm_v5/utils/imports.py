from __future__ import annotations

import sys
from pathlib import Path

from drivelm_v5.config.common import REPO_ROOT


LLAMA_ADAPTER_ROOT = (REPO_ROOT / "challenge" / "llama_adapter_v2_multimodal7b").resolve()


def add_llama_adapter_to_path() -> Path:
    root = str(LLAMA_ADAPTER_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return LLAMA_ADAPTER_ROOT

