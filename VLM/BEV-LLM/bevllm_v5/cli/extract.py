from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from bevllm_v5.config.common import (
    DEFAULT_BEV_FEATURE_KEY,
    DEFAULT_BEVFUSION_CKPT,
    DEFAULT_BEVFUSION_CONFIG,
    DEFAULT_DATASET_VERSION,
    DEFAULT_INFO_PKL,
    DEFAULT_OPENPCDET_ROOT,
    DEFAULT_PREPARED_DIR,
    DEFAULT_WORK_DIR,
    resolve_dataset_version_paths,
)
from bevllm_v5.utils.dist import cleanup_distributed, init_distributed, resolve_local_rank_device, shard_sequence
from bevllm_v5.utils.io import dump_json, ensure, load_json, load_pickle
from bevllm_v5.utils.openpcdet import (
    apply_prepared_frame_paths,
    build_extract_only_bevfusion_model,
    create_openpcdet_logger,
    ensure_openpcdet_on_path,
    import_openpcdet_module,
    load_data_to_gpu,
    load_openpcdet_cfg,
    load_openpcdet_checkpoint,
    load_prepared_frame_lookup,
    normalize_openpcdet_info,
    probe_openpcdet_extract_stack,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract BEV features for BEV-LLM Intersection V5.")
    parser.add_argument("--dataset-version", choices=["v5", "v6"], default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--info-pkl", default=str(DEFAULT_INFO_PKL))
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--openpcdet-root", default=str(DEFAULT_OPENPCDET_ROOT))
    parser.add_argument("--bevfusion-config", default=str(DEFAULT_BEVFUSION_CONFIG))
    parser.add_argument("--bevfusion-ckpt", default=str(DEFAULT_BEVFUSION_CKPT))
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-key", default=DEFAULT_BEV_FEATURE_KEY)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_feature_key(payload: Any, feature_key: str) -> Any:
    current = payload
    for part in feature_key.split("."):
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, (list, tuple)):
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


class OpenPCDetBEVExtractor:
    def __init__(
        self,
        *,
        openpcdet_root: Path,
        config_path: Path,
        checkpoint_path: Path,
        info_pkl: Path,
        prepared_dir: Path,
        device: str,
        feature_key: str,
        rank: int,
    ):
        ensure_openpcdet_on_path(openpcdet_root)
        cfg = load_openpcdet_cfg(config_path, openpcdet_root)
        probe_openpcdet_extract_stack(openpcdet_root, cfg)
        CarlaDataset = import_openpcdet_module(
            "pcdet.datasets.carla.carla_dataset",
            openpcdet_root,
        ).CarlaDataset

        self.device = device
        self.feature_key = feature_key
        self.load_data_to_gpu = load_data_to_gpu
        self.logger = create_openpcdet_logger(rank=rank)
        self.cfg = cfg
        self.openpcdet_root = openpcdet_root.resolve()
        self.tools_dir = self.openpcdet_root / "tools"
        ensure(self.tools_dir.is_dir(), f"OpenPCDet tools directory not found: {self.tools_dir}")

        # Feature extraction is always single-sample and should be deterministic.
        if self.cfg.MODEL.get("VTRANSFORM", None) is not None:
            self.cfg.MODEL.VTRANSFORM.BATCH_SIZE = 1
        for processor_cfg in self.cfg.DATA_CONFIG.get("DATA_PROCESSOR", []):
            if processor_cfg.NAME == "shuffle_points":
                processor_cfg.SHUFFLE_ENABLED["test"] = False

        self.dataset = CarlaDataset(
            dataset_cfg=self.cfg.DATA_CONFIG,
            class_names=self.cfg.CLASS_NAMES,
            training=False,
            logger=self.logger,
        )
        raw_infos = load_pickle(info_pkl)
        ensure(isinstance(raw_infos, list) and raw_infos, f"OpenPCDet info PKL must contain a non-empty list: {info_pkl}")
        self.prepared_frames = load_prepared_frame_lookup(prepared_dir)
        raw_info_by_token = {
            str(info["token"]): info
            for info in raw_infos
            if isinstance(info, dict) and "token" in info
        }
        missing_tokens = sorted(set(self.prepared_frames) - set(raw_info_by_token))
        ensure(
            not missing_tokens,
            f"Prepared frames are missing from OpenPCDet info PKL {info_pkl}: first_missing={missing_tokens[:10]} total_missing={len(missing_tokens)}",
        )

        prepared_infos = []
        for frame_token in sorted(self.prepared_frames):
            frame_record = self.prepared_frames[frame_token]
            prepared_info = apply_prepared_frame_paths(raw_info_by_token[frame_token], frame_record)
            prepared_infos.append(normalize_openpcdet_info(prepared_info, tools_dir=self.tools_dir))

        self.dataset.infos = prepared_infos
        self.token_to_index = {str(info["token"]): index for index, info in enumerate(self.dataset.infos)}
        ensure(self.token_to_index, f"No token mapping could be built from prepared frames under {prepared_dir}")
        if rank == 0:
            example_frame = next(iter(self.prepared_frames.values()))
            self.logger.info(
                "Prepared frames matched to OpenPCDet infos: prepared=%s matched=%s raw_info_pool=%s",
                len(self.prepared_frames),
                len(self.dataset.infos),
                len(raw_info_by_token),
            )
            self.logger.info(
                "Resolved example frame %s lidar=%s image0=%s",
                example_frame["frame_token"],
                example_frame["point_cloud_path"],
                example_frame["images"][0],
            )

        self.model = build_extract_only_bevfusion_model(
            cfg=self.cfg,
            dataset=self.dataset,
            openpcdet_root=self.openpcdet_root,
        )
        to_cpu = not device.startswith("cuda")
        checkpoint_meta = load_openpcdet_checkpoint(self.model, checkpoint_path=checkpoint_path, to_cpu=to_cpu)
        self.logger.info(
            "Loaded BEVFusion extractor weights %s (%s/%s tensors matched).",
            checkpoint_path,
            checkpoint_meta["loaded"],
            checkpoint_meta["total"],
        )
        self.model.to(device)
        self.model.eval()

    def _prepare_batch(self, frame_token: str) -> dict[str, Any]:
        ensure(frame_token in self.token_to_index, f"Frame token {frame_token} not found in OpenPCDet info PKL.")
        sample = self.dataset[self.token_to_index[frame_token]]
        batch = self.dataset.collate_batch([sample])
        self.load_data_to_gpu(batch)
        return batch

    @torch.no_grad()
    def extract(self, frame_token: str) -> torch.Tensor:
        batch_dict = self._prepare_batch(frame_token)
        for module_name, cur_module in zip(self.model.module_topology, self.model.module_list):
            batch_dict = cur_module(batch_dict)
            if module_name == "backbone_2d":
                break

        tensor = resolve_feature_key(batch_dict, self.feature_key)
        ensure(isinstance(tensor, torch.Tensor), f"Resolved feature '{self.feature_key}' is not a tensor.")
        ensure(tensor.ndim == 4, f"Resolved feature '{self.feature_key}' must be 4D, got shape {tuple(tensor.shape)}")
        ensure(tensor.shape[0] == 1, f"Expected batch size 1 from extractor, got shape {tuple(tensor.shape)}")
        return tensor.squeeze(0).detach().cpu()


def extract_split(
    *,
    args: argparse.Namespace,
    split: str,
    rank: int,
    world_size: int,
    local_rank: int,
    device_name: str,
    extractor: OpenPCDetBEVExtractor,
) -> None:
    prepared_dir = Path(args.prepared_dir).resolve()
    frame_path = prepared_dir / f"frames_{split}.json"
    ensure(frame_path.is_file(), f"Frame manifest not found: {frame_path}")
    frames = load_json(frame_path)
    ensure(isinstance(frames, list) and frames, f"{frame_path} does not contain a non-empty list.")
    if args.limit is not None:
        frames = frames[: args.limit]
    shard = shard_sequence(frames, rank, world_size)

    if args.output_dir and args.split != "all":
        output_dir = Path(args.output_dir).resolve()
    else:
        work_dir = Path(args.work_dir).resolve()
        output_dir = work_dir / "features" / ("train" if split == "train" else "val")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[extract] split={split} rank={rank} local_rank={local_rank} device={device_name} "
        f"shard={len(shard)}/{len(frames)} output_dir={output_dir}",
        flush=True,
    )

    current_frame_token: str | None = None
    start_time = time.time()
    written = 0
    skipped = 0
    for frame_record in tqdm(shard, ncols=80, disable=rank != 0):
        current_frame_token = str(frame_record["frame_token"])
        output_path = output_dir / f"{current_frame_token}.pt"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        torch.save(extractor.extract(current_frame_token), output_path)
        written += 1

    elapsed = time.time() - start_time
    print(f"[extract] split={split} rank={rank} wrote={written} skipped={skipped} elapsed={elapsed:.1f}s", flush=True)


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    device_name = resolve_local_rank_device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BEV extraction.")
    primary_error: Exception | None = None
    try:
        resolved = resolve_dataset_version_paths(args.dataset_version, prepared_dir=args.prepared_dir, work_dir=args.work_dir)
        prepared_dir = Path(resolved["prepared_dir"]).resolve()
        work_dir = Path(resolved["work_dir"]).resolve()
        args.prepared_dir = str(prepared_dir)
        args.work_dir = str(work_dir)
        manifest_path = prepared_dir / "split_manifest.json"
        ensure(manifest_path.is_file(), f"Prepared manifest not found: {manifest_path}")
        manifest = load_json(manifest_path)
        ensure(
            str(manifest.get("dataset_version", "v5")) == str(resolved["dataset_version"]),
            f"Prepared data version mismatch: expected {resolved['dataset_version']}, found {manifest.get('dataset_version')}",
        )
        extractor = OpenPCDetBEVExtractor(
            openpcdet_root=Path(args.openpcdet_root).resolve(),
            config_path=Path(args.bevfusion_config).resolve(),
            checkpoint_path=Path(args.bevfusion_ckpt).resolve(),
            info_pkl=Path(args.info_pkl).resolve(),
            prepared_dir=prepared_dir,
            device=device_name,
            feature_key=args.feature_key,
            rank=rank,
        )
        splits = ("train", "val") if args.split == "all" else (args.split,)
        for split in splits:
            extract_split(
                args=args,
                split=split,
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                device_name=device_name,
                extractor=extractor,
            )
        if rank == 0:
            dump_json(
                work_dir / "features" / "feature_manifest.json",
                {
                    "dataset_version": str(resolved["dataset_version"]),
                    "prepared_dir": str(prepared_dir),
                    "feature_key": args.feature_key,
                },
            )
    except Exception as exc:
        primary_error = exc
        print(f"[extract] rank={rank} local_rank={local_rank} failed with {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
    finally:
        cleanup_distributed(suppress_errors=primary_error is not None)


if __name__ == "__main__":
    main()
