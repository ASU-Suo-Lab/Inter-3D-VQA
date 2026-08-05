from __future__ import annotations

import json
import pathlib
import pickle
from pathlib import Path
from typing import Any, Iterable, Mapping

from bevllm_v5.config.common import LLM_ROOT


class CrossPlatformPathUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "pathlib" and name == "PosixPath":
            return pathlib.PurePosixPath
        if module == "pathlib" and name == "WindowsPath":
            return pathlib.PureWindowsPath
        return super().find_class(module, name)


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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
    with path.open("rb") as file:
        return CrossPlatformPathUnpickler(file).load()


def dump_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(payload, file)


def normalize_data_path(raw_path: str) -> str:
    ensure(bool(raw_path), "Encountered empty path while normalizing a data path.")
    normalized = str(raw_path).replace("\\", "/")
    if normalized.startswith("/data/"):
        return str((LLM_ROOT / normalized.lstrip("/")).resolve())
    if normalized.startswith("data/"):
        return str((LLM_ROOT / normalized).resolve())
    if Path(normalized).is_absolute():
        return str(Path(normalized).resolve())
    return str((LLM_ROOT / normalized).resolve())
