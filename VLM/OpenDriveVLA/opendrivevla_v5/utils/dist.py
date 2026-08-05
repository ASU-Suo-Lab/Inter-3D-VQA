from __future__ import annotations

import os
import sys
from typing import Any, List, Sequence, Tuple

import torch
import torch.distributed as dist


def resolve_local_rank_device(requested_device: str) -> str:
    if requested_device.startswith("cuda") and requested_device == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return f"cuda:{local_rank}"
    return requested_device


def init_distributed() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
    return rank, world_size, local_rank


def synchronize_distributed(label: str) -> None:
    if not dist.is_initialized():
        return
    try:
        dist.barrier()
    except Exception as exc:
        raise RuntimeError(
            f"Distributed synchronization failed during {label}; another rank likely exited earlier."
        ) from exc


def cleanup_distributed(suppress_errors: bool = False) -> None:
    if not dist.is_initialized():
        return
    try:
        dist.destroy_process_group()
    except Exception as exc:
        if suppress_errors:
            print(f"[dist] cleanup warning: {exc}", file=sys.stderr, flush=True)
            return
        raise


def shard_sequence(items: Sequence[Any], rank: int, world_size: int) -> List[Any]:
    if world_size <= 1:
        return list(items)
    return list(items[rank::world_size])


def is_rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0
