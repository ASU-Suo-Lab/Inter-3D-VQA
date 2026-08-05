from __future__ import annotations

import os
from typing import Any, List, Sequence

import torch
import torch.distributed as dist


def resolve_local_rank_device(requested_device: str) -> str:
    if requested_device == "cuda":
        return f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
    return requested_device


def init_distributed() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed(suppress_errors: bool = True) -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    try:
        dist.destroy_process_group()
    except Exception:
        if not suppress_errors:
            raise


def synchronize_distributed(label: str) -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    try:
        dist.barrier()
    except Exception as exc:
        raise RuntimeError(f"Distributed synchronization failed during {label}.") from exc


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def shard_sequence(items: Sequence[Any], rank: int, world_size: int) -> List[Any]:
    if world_size <= 1:
        return list(items)
    return list(items[rank::world_size])
