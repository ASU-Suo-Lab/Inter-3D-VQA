from __future__ import annotations

from typing import Sequence, TypeVar


T = TypeVar("T")


def shard_sequence(items: Sequence[T], rank: int, world_size: int) -> list[T]:
    return list(items[rank::world_size])
