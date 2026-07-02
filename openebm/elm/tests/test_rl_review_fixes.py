import torch

from openebm.elm.rl.data_sharding import build_rank_worker_indices
from openebm.elm.rl.optimizer_utils import make_skip_missing_grad_muon_adamw


def test_rank_worker_indices_are_disjoint_and_cover_epoch():
    num_items = 37
    world_size = 4
    num_workers = 2
    shards = []
    for rank in range(world_size):
        for worker_id in range(num_workers):
            shard = build_rank_worker_indices(
                num_items,
                rank=rank,
                world_size=world_size,
                worker_id=worker_id,
                num_workers=num_workers,
                seed=123,
            )
            shards.append(shard)

    flat = [idx for shard in shards for idx in shard]
    assert sorted(flat) == list(range(num_items))
    assert len(flat) == len(set(flat))


def test_muon_missing_grad_group_is_skipped_not_zero_filled():
    class FakeMuonAdamW:
        def __init__(self, param_groups):
            self.param_groups = param_groups
            self.muon_steps = 0
            self.adamw_steps = 0

        def _step_adamw(self, group):
            self.adamw_steps += 1

        def _step_muon(self, group):
            self.muon_steps += 1

    Wrapped = make_skip_missing_grad_muon_adamw(FakeMuonAdamW)
    p_missing = torch.nn.Parameter(torch.ones(2, 2))
    p_present = torch.nn.Parameter(torch.ones(2, 2))
    p_present.grad = torch.ones_like(p_present)

    opt = Wrapped([
        {"kind": "muon", "params": [p_missing, p_present]},
        {"kind": "adamw", "params": [p_present]},
    ])
    opt.step()

    assert p_missing.grad is None
    assert opt.muon_steps == 0
    assert opt.adamw_steps == 1
