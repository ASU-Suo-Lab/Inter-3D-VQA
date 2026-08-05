from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: str | Path) -> Any:
    resolved = Path(path).resolve()
    ensure(resolved.is_file(), f"Missing JSON file: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def dump_json(path: str | Path, payload: Any) -> None:
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path).resolve()
    ensure(resolved.is_file(), f"Missing JSONL file: {resolved}")
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            ensure(isinstance(payload, dict), f"Expected JSON object on line {line_number} of {resolved}")
            rows.append(payload)
    return rows


def dump_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
