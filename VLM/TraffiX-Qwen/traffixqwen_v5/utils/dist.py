from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist


def init_distributed() -> tuple[int, int, int]:
    if "WORLD_SIZE" not in os.environ:
        return 0, 1, 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for distributed TraffiX-Qwen V5 execution.")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    try:
        dist.destroy_process_group()
    except Exception as exc:  # pragma: no cover
        print(f"[dist] cleanup warning: {exc}", file=sys.stderr)

