from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Iterable, Mapping

from nuscenesqa_v5.config.common import LLM_ROOT


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> Any:
    ensure(path.is_file(), f"File not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    ensure(path.is_file(), f"File not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def load_pickle(path: Path) -> Any:
    ensure(path.is_file(), f"File not found: {path}")
    with path.open("rb") as file:
        return pickle.load(file)


def normalize_data_path(raw_path: str) -> str:
    ensure(bool(raw_path), "Encountered empty data path.")
    path = Path(raw_path)
    if path.is_absolute():
        return str(path.resolve())
    return str((LLM_ROOT / raw_path).resolve())

