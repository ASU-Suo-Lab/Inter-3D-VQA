from __future__ import annotations

import argparse
from pathlib import Path

from geovlm_intersection.config.common import DEFAULT_DATASET_VERSION, resolve_dataset_version_paths
from geovlm_intersection.data import SUPPORTED_STRUCTURED_SUBTEMPLATES
from geovlm_intersection.utils import dump_json, dump_jsonl, ensure, load_json, load_jsonl


SPLITS = ("train", "val", "val_eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the GeoVLM v5 structured-subset dataset.")
    parser.add_argument("--dataset-version", choices=["v5"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--qa-json", default=None)
    parser.add_argument("--source-prepared-dir", default=None)
    parser.add_argument("--prepared-dir", default=None)
    return parser.parse_args()


def _filter_records(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    filtered: list[dict[str, object]] = []
    for row in rows:
        ensure(isinstance(row, dict), f"Prepared row must be a JSON object, got: {type(row)!r}")
        subtemplate = row.get("subtemplate")
        ensure(isinstance(subtemplate, str) and subtemplate, f"Prepared row missing subtemplate: {row}")
        if subtemplate not in SUPPORTED_STRUCTURED_SUBTEMPLATES:
            continue
        ensure("structured_targets" in row, f"Prepared row missing structured_targets for question_id={row.get('question_id')}")
        filtered.append(row)
    return filtered


def main() -> None:
    args = parse_args()
    resolved = resolve_dataset_version_paths(
        args.dataset_version,
        qa_json=args.qa_json,
        source_prepared_dir=args.source_prepared_dir,
        prepared_dir=args.prepared_dir,
    )
    qa_json = Path(resolved["qa_json"]).resolve()
    source_prepared_dir = Path(resolved["source_prepared_dir"]).resolve()
    prepared_dir = Path(resolved["prepared_dir"]).resolve()
    ensure(qa_json.is_file(), f"Missing QA JSON: {qa_json}")
    ensure(source_prepared_dir.is_dir(), f"Missing source prepared dir: {source_prepared_dir}")
    prepared_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = load_json(source_prepared_dir / "split_manifest.json")
    ensure(isinstance(source_manifest, dict), f"Source split_manifest must be an object: {source_prepared_dir / 'split_manifest.json'}")
    ensure(
        source_manifest.get("dataset_version") == resolved["dataset_version"],
        f"Source prepared dataset_version={source_manifest.get('dataset_version')} does not match requested {resolved['dataset_version']}.",
    )

    filtered_counts: dict[str, int] = {}
    filtered_question_ids: dict[str, set[str]] = {}
    for split in SPLITS:
        source_path = source_prepared_dir / f"{split}.json"
        rows = load_json(source_path)
        ensure(isinstance(rows, list), f"Prepared split must be a JSON list: {source_path}")
        filtered = _filter_records(rows)
        dump_json(prepared_dir / f"{split}.json", filtered)
        filtered_counts[split] = len(filtered)
        filtered_question_ids[split] = {str(row["question_id"]) for row in filtered}

    source_sidecar = source_prepared_dir / "sidecar_val.jsonl"
    sidecar_rows = load_jsonl(source_sidecar)
    val_question_ids = filtered_question_ids["val"] | filtered_question_ids["val_eval"]
    filtered_sidecar = [row for row in sidecar_rows if str(row.get("question_id")) in val_question_ids]
    dump_jsonl(prepared_dir / "sidecar_val.jsonl", filtered_sidecar)

    manifest = {
        "dataset_version": resolved["dataset_version"],
        "qa_json": str(qa_json),
        "source_prepared_dir": str(source_prepared_dir),
        "prepared_dir": str(prepared_dir),
        "supported_structured_subtemplates": sorted(SUPPORTED_STRUCTURED_SUBTEMPLATES),
        "source_counts": {
            split: int(source_manifest.get("counts", {}).get(f"{split}_qas", 0))
            if split in {"train", "val"}
            else None
            for split in SPLITS
        },
        "filtered_counts": filtered_counts,
        "sidecar_val_count": len(filtered_sidecar),
    }
    dump_json(prepared_dir / "split_manifest.json", manifest)


if __name__ == "__main__":
    main()
