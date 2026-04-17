"""
End-to-end tests comparing the SDPA Attention implementation
(ar_ebt_time_embed.py) against the original superdiagonal implementation
(ar_ebt_time_embed_cleanup.py).

Single-GPU tests (pytest):
  1. forward pass numerical consistency
  2. backward pass (gradient) numerical consistency
  3. latency benchmark across sequence lengths
  4. peak memory benchmark across sequence lengths

Distributed tests (torchrun):
  5. DDP forward/backward consistency across 2 GPUs
  6. DDP latency & memory benchmark

Run:
  cd nova/ebt
  source ../../.venv/ebt/bin/activate

  # Single-GPU correctness + benchmarks
  python test/test_sdpa_attention.py

  # Distributed (2 GPUs)
  torchrun --standalone --nproc_per_node=2 test/test_sdpa_attention.py

  # Via helper script (both modes)
  bash runs/test_sdpa.sh
"""

import sys, os, gc, time
import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ar_ebt_time_embed_cleanup import EBTTimeConcat as EBTOrig
from ar_ebt_time_embed import EBTTimeConcat as EBTSdpa
from utils import EBTModelArgs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_MCMC_STEPS = 4


def make_model_pair(dim=128, n_layers=2, n_heads=4, seq_len=32, seed=42, device=None):
    """Return two EBTTimeConcat instances with identical weights."""
    if device is None:
        device = DEVICE
    args = EBTModelArgs(
        dim=dim, n_layers=n_layers, n_heads=n_heads,
        max_seq_len=seq_len * 2, vocab_size=256,
    )
    torch.manual_seed(seed)
    orig = EBTOrig(args, MAX_MCMC_STEPS).to(device)
    sdpa = EBTSdpa(args, MAX_MCMC_STEPS).to(device)
    sdpa.load_state_dict(orig.state_dict())
    return orig, sdpa


def make_embeddings(batch=2, seq_len=16, dim=128, seed=0, device=None):
    """Random embedding [B, 2*(S-1), D] with requires_grad=True."""
    if device is None:
        device = DEVICE
    torch.manual_seed(seed)
    return torch.randn(batch, 2 * (seq_len - 1), dim,
                       device=device, requires_grad=True)


# ---------------------------------------------------------------------------
# 1. Forward pass numerical consistency (single-GPU)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seq_len", [64, 128, 256, 512, 1024])
@pytest.mark.parametrize("mcmc_step", [2, 4])
def test_forward_consistency(seq_len, mcmc_step):
    orig, sdpa = make_model_pair(seq_len=seq_len)
    emb_orig = make_embeddings(seq_len=seq_len)
    emb_sdpa = emb_orig.detach().clone().requires_grad_(True)

    orig.eval(); sdpa.eval()
    with torch.no_grad():
        out_orig = orig(emb_orig, start_pos=0, mcmc_step=mcmc_step)
        out_sdpa = sdpa(emb_sdpa, start_pos=0, mcmc_step=mcmc_step)

    assert out_orig.shape == out_sdpa.shape
    max_err = (out_orig - out_sdpa).abs().max().item()
    print(f"\n[forward] seq={seq_len} step={mcmc_step}  max_err={max_err:.2e}")
    assert max_err < 1e-4, f"Forward max error {max_err:.2e} exceeds 1e-4"


# ---------------------------------------------------------------------------
# 2. Backward pass (gradient) numerical consistency (single-GPU)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seq_len", [64, 128, 256, 512, 1024])
def test_backward_consistency(seq_len):
    orig, sdpa = make_model_pair(seq_len=seq_len)
    emb_orig = make_embeddings(seq_len=seq_len)
    emb_sdpa = emb_orig.detach().clone().requires_grad_(True)

    orig.train(); sdpa.train()
    orig(emb_orig, start_pos=0, mcmc_step=0).sum().backward()
    sdpa(emb_sdpa, start_pos=0, mcmc_step=0).sum().backward()

    max_err  = (emb_orig.grad - emb_sdpa.grad).abs().max().item()
    wq_orig  = list(orig.layers[0].attention.wq.parameters())[0].grad
    wq_sdpa  = list(sdpa.layers[0].attention.wq.parameters())[0].grad
    wq_err   = (wq_orig - wq_sdpa).abs().max().item()
    print(f"\n[backward] seq={seq_len}  input_grad_err={max_err:.2e}  wq_grad_err={wq_err:.2e}")

    assert max_err < 1e-4, f"Input grad max error {max_err:.2e} exceeds 1e-4"
    assert wq_err  < 1e-4, f"wq.grad max error {wq_err:.2e} exceeds 1e-4"


# ---------------------------------------------------------------------------
# 3 & 4. Single-GPU latency & memory benchmarks
# ---------------------------------------------------------------------------

def _gpu_time_ms(fn, warmup=5, repeats=20):
    if DEVICE == "cpu":
        for _ in range(warmup): fn()
        t0 = time.perf_counter()
        for _ in range(repeats): fn()
        return (time.perf_counter() - t0) / repeats * 1000

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start.record(); fn(); end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def _peak_mem_mb(fn, device=None):
    if device is None:
        device = DEVICE
    if not str(device).startswith("cuda"):
        return float("nan")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 1024 ** 2


def run_benchmarks(device=None, prefix=""):
    if device is None:
        device = DEVICE
    SEQ_LENS = [64, 128, 256, 512, 1024]
    BATCH, DIM, N_HEADS, N_LAYERS = 4, 512, 8, 4

    print(f"\n{'=' * 72}")
    print(f"BENCHMARK{prefix}: SDPA vs Original — latency (ms) and peak HBM (MB)")
    print(f"{'=' * 72}")
    print(f"{'seq_len':>8}  {'orig_ms':>9}  {'sdpa_ms':>9}  {'speedup':>8}  "
          f"{'orig_MB':>9}  {'sdpa_MB':>9}  {'mem_save':>9}")
    print("-" * 72)

    for seq_len in SEQ_LENS:
        args = EBTModelArgs(dim=DIM, n_layers=N_LAYERS, n_heads=N_HEADS,
                            max_seq_len=seq_len * 2, vocab_size=256)
        torch.manual_seed(0)
        orig = EBTOrig(args, MAX_MCMC_STEPS).to(device).train()
        sdpa = EBTSdpa(args, MAX_MCMC_STEPS).to(device).train()
        sdpa.load_state_dict(orig.state_dict())

        def _fwd_bwd(model):
            emb = torch.randn(BATCH, 2 * (seq_len - 1), DIM,
                              device=device, requires_grad=True)
            model(emb, start_pos=0, mcmc_step=0).sum().backward()

        orig_ms = _gpu_time_ms(lambda: _fwd_bwd(orig))
        sdpa_ms = _gpu_time_ms(lambda: _fwd_bwd(sdpa))
        speedup = orig_ms / sdpa_ms if sdpa_ms > 0 else float("inf")

        orig_mb = _peak_mem_mb(lambda: _fwd_bwd(orig), device)
        sdpa_mb = _peak_mem_mb(lambda: _fwd_bwd(sdpa), device)
        mem_save = orig_mb / sdpa_mb if sdpa_mb > 0 else float("inf")

        print(f"{seq_len:>8}  {orig_ms:>9.2f}  {sdpa_ms:>9.2f}  {speedup:>8.2f}x  "
              f"{orig_mb:>9.1f}  {sdpa_mb:>9.1f}  {mem_save:>8.2f}x")

        del orig, sdpa
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 5 & 6. Distributed tests (torchrun)
# ---------------------------------------------------------------------------

def _dist_print(rank, *args, **kwargs):
    """Print only from rank 0."""
    if rank == 0:
        print(*args, **kwargs)


def run_distributed():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    _dist_print(rank, f"\n{'=' * 72}")
    _dist_print(rank, f"DISTRIBUTED CORRECTNESS  (world_size={world_size})")
    _dist_print(rank, f"{'=' * 72}")
    _dist_print(rank, f"{'seq_len':>8}  {'fwd_err':>10}  {'bwd_err':>10}  "
                      f"{'wq_grad_err':>12}  {'result':>6}")
    _dist_print(rank, "-" * 72)

    all_passed = True

    for seq_len in [64, 128, 256, 512, 1024]:
        orig, sdpa = make_model_pair(seq_len=seq_len, device=device)

        # Wrap with DDP — gradient all-reduce is exercised on backward
        orig_ddp = DDP(orig, device_ids=[local_rank])
        sdpa_ddp = DDP(sdpa, device_ids=[local_rank])

        # Same input on every rank (fixed seed ensures reproducibility)
        emb_orig = make_embeddings(seq_len=seq_len, device=device)
        emb_sdpa = emb_orig.detach().clone().requires_grad_(True)

        # Forward
        orig_ddp.train(); sdpa_ddp.train()
        out_orig = orig_ddp(emb_orig, start_pos=0, mcmc_step=0)
        out_sdpa = sdpa_ddp(emb_sdpa, start_pos=0, mcmc_step=0)
        fwd_err  = (out_orig - out_sdpa).abs().max().item()

        # Backward (DDP all-reduces gradients across ranks)
        out_orig.sum().backward()
        out_sdpa.sum().backward()
        bwd_err = (emb_orig.grad - emb_sdpa.grad).abs().max().item()

        wq_orig = list(orig_ddp.module.layers[0].attention.wq.parameters())[0].grad
        wq_sdpa = list(sdpa_ddp.module.layers[0].attention.wq.parameters())[0].grad
        wq_err  = (wq_orig - wq_sdpa).abs().max().item()

        passed = fwd_err < 1e-4 and bwd_err < 1e-4 and wq_err < 1e-4
        if not passed:
            all_passed = False

        # Gather max error across all ranks so rank 0 can report global result
        err_tensor = torch.tensor([fwd_err, bwd_err, wq_err], device=device)
        dist.all_reduce(err_tensor, op=dist.ReduceOp.MAX)
        g_fwd, g_bwd, g_wq = err_tensor.tolist()

        _dist_print(rank,
            f"{seq_len:>8}  {g_fwd:>10.2e}  {g_bwd:>10.2e}  "
            f"{g_wq:>12.2e}  {'PASS' if g_fwd < 1e-4 and g_bwd < 1e-4 and g_wq < 1e-4 else 'FAIL':>6}"
        )

    dist.barrier()
    _dist_print(rank,
        "\nAll distributed correctness tests PASSED ✓"
        if all_passed else "\nSOME DISTRIBUTED TESTS FAILED ✗"
    )

    # -- Distributed benchmark -----------------------------------------------
    _dist_print(rank, f"\n{'=' * 72}")
    _dist_print(rank, f"DISTRIBUTED BENCHMARK  (world_size={world_size}, reporting rank-0 device)")
    _dist_print(rank, f"{'=' * 72}")
    _dist_print(rank, f"{'seq_len':>8}  {'orig_ms':>9}  {'sdpa_ms':>9}  {'speedup':>8}  "
                      f"{'orig_MB':>9}  {'sdpa_MB':>9}  {'mem_save':>9}")
    _dist_print(rank, "-" * 72)

    SEQ_LENS = [64, 128, 256, 512, 1024]
    BATCH, DIM, N_HEADS, N_LAYERS = 4, 512, 8, 4

    for seq_len in SEQ_LENS:
        args = EBTModelArgs(dim=DIM, n_layers=N_LAYERS, n_heads=N_HEADS,
                            max_seq_len=seq_len * 2, vocab_size=256)
        torch.manual_seed(0)
        orig = EBTOrig(args, MAX_MCMC_STEPS).to(device).train()
        sdpa = EBTSdpa(args, MAX_MCMC_STEPS).to(device).train()
        sdpa.load_state_dict(orig.state_dict())
        orig_ddp = DDP(orig, device_ids=[local_rank])
        sdpa_ddp = DDP(sdpa, device_ids=[local_rank])

        def _fwd_bwd(model):
            emb = torch.randn(BATCH, 2 * (seq_len - 1), DIM,
                              device=device, requires_grad=True)
            dist.barrier()  # sync before timing
            model(emb, start_pos=0, mcmc_step=0).sum().backward()

        orig_ms = _gpu_time_ms(lambda: _fwd_bwd(orig_ddp))
        sdpa_ms = _gpu_time_ms(lambda: _fwd_bwd(sdpa_ddp))
        speedup = orig_ms / sdpa_ms if sdpa_ms > 0 else float("inf")

        orig_mb = _peak_mem_mb(lambda: _fwd_bwd(orig_ddp), device)
        sdpa_mb = _peak_mem_mb(lambda: _fwd_bwd(sdpa_ddp), device)
        mem_save = orig_mb / sdpa_mb if sdpa_mb > 0 else float("inf")

        _dist_print(rank,
            f"{seq_len:>8}  {orig_ms:>9.2f}  {sdpa_ms:>9.2f}  {speedup:>8.2f}x  "
            f"{orig_mb:>9.1f}  {sdpa_mb:>9.1f}  {mem_save:>8.2f}x"
        )

        del orig, sdpa, orig_ddp, sdpa_ddp
        gc.collect()
        torch.cuda.empty_cache()

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        # torchrun with multiple GPUs → distributed correctness + benchmark
        run_distributed()
    else:
        # torchrun --nproc_per_node=1  OR  plain python:
        # run pytest correctness tests then single-GPU benchmarks
        import pytest as _pytest
        print(f"\nRunning single-GPU correctness tests on {DEVICE}...")
        ret = _pytest.main([__file__, "-v", "--tb=short"])
        if ret == 0:
            run_benchmarks()
