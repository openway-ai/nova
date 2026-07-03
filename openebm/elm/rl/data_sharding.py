"""Rank/worker sharding helpers for RL iterable datasets."""

from __future__ import annotations

from typing import List

import numpy as np


def build_rank_worker_indices(
    num_items: int,
    *,
    rank: int,
    world_size: int,
    worker_id: int,
    num_workers: int,
    seed: int,
) -> List[int]:
    """Return one deterministic, non-overlapping shard for a rank/worker.

    All ranks/workers use the same shuffled global order for a given seed, then
    take a stride slice by global shard id. If there are more shards than items,
    duplicates are unavoidable; assign one deterministic item so no worker spins
    forever on an empty iterable.
    """
    if num_items <= 0:
        return []

    safe_world_size = max(1, int(world_size))
    safe_num_workers = max(1, int(num_workers))
    safe_rank = max(0, int(rank))
    safe_worker = max(0, int(worker_id))

    indices = list(range(num_items))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(indices)

    num_shards = safe_world_size * safe_num_workers
    shard_id = safe_rank * safe_num_workers + safe_worker
    shard = indices[shard_id::num_shards]
    if not shard:
        shard = [indices[shard_id % num_items]]
    return shard
