"""Bootstrap for multi-node runs under srun: gloo process group (TCP) for exchanging ids and barriers.
Rank layout: world = 2N; ranks [0, N) on node A (inference), [N, 2N) on node B (store); rank i pairs with i + N.
MASTER_ADDR must be node A's hostname (the launcher sets it)."""
import datetime
import os

import torch
import torch.distributed as dist

GROUP = None   # collectives below use this group (None = world); P2 sets it to the inference ranks


def init():
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1)))
    local = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0)))
    os.environ.setdefault("MASTER_PORT", "29511")
    if world > 1:
        dist.init_process_group("gloo", rank=rank, world_size=world, timeout=datetime.timedelta(minutes=15))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    return rank, world, local


def subgroup(ranks):
    """Every rank must call this (torch.distributed.new_group is collective)."""
    return dist.new_group(ranks=list(ranks)) if dist.is_initialized() else None


def bcast(obj, src=0):
    box = [obj]
    if dist.is_initialized():
        dist.broadcast_object_list(box, src=src, group=GROUP)
    return box[0]


def barrier(group="default"):
    if dist.is_initialized():
        dist.barrier(group=GROUP if group == "default" else group)


def gather(obj):
    if not dist.is_initialized():
        return [obj]
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out
