"""CPU W4A16 prefill experts (moe/wna16_cpu_prefill.py) against a dequantized torch reference that
rounds to bf16 at the GPU path's store points."""
from __future__ import annotations

import pytest
import torch

from freetoken.moe.wna16_cpu_prefill import cpu_supported

pytestmark = pytest.mark.skipif(not cpu_supported(), reason="needs AVX-512 BF16")

H, I, E = 512, 192, 6


def _banks(g, inter, down_groups, experts=E):
    def ints(*shape):
        return torch.randint(-(2**31), 2**31 - 1, shape, generator=g, dtype=torch.int64).to(torch.int32)

    def scales(*shape):
        return (torch.randn(shape, generator=g) * 0.02).to(torch.float16)

    return {
        "gate_up": ints(experts, H // 8, 2 * inter),
        "gate_up_zero": ints(experts, H // 128, 2 * inter // 8),
        "gate_up_scale": scales(experts, H // 128, 2 * inter),
        "down": ints(experts, inter // 8, H),
        "down_zero": ints(experts, down_groups, H // 8),
        "down_scale": scales(experts, down_groups, H),
    }


def _nibbles(words, axis):
    """[..., W, ...] int32 -> codes with axis W expanded 8x (code j of word w -> 8w + j)."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    w = words.unsqueeze(axis + 1) if axis >= 0 else words.unsqueeze(axis)
    shape = [1] * w.dim()
    shape[axis + 1 if axis >= 0 else axis] = 8
    codes = (w >> shifts.view(shape)) & 15
    out_shape = list(words.shape)
    out_shape[axis] *= 8
    return codes.reshape(out_shape).float()


def _dequant(qw, qz, sc, k_off):
    codes = _nibbles(qw, 0)                                     # [K, N]
    K, N = codes.shape
    g = (torch.arange(K) + k_off) // 128
    zeros = _nibbles(qz, 1)[g] + 1                              # [K, N]
    return (codes - zeros) * sc.float()[g]


def _reference(banks, x, ids, w, k_off):
    bf = lambda t: t.to(torch.bfloat16).float()  # noqa: E731
    y = torch.zeros(x.shape[0], H)
    for t in range(x.shape[0]):
        for k in range(ids.shape[1]):
            e = int(ids[t, k])
            if e < 0:
                continue
            wgu = _dequant(banks["gate_up"][e], banks["gate_up_zero"][e], banks["gate_up_scale"][e], 0)
            wdn = _dequant(banks["down"][e], banks["down_zero"][e], banks["down_scale"][e], k_off)
            gu = bf(x[t].float() @ wgu)
            inter = gu.shape[0] // 2
            gate, up = gu[:inter], gu[inter:]
            h = bf(gate / (1 + torch.exp(-gate)) * up)
            y[t] += bf(float(w[t, k]) * (h @ wdn))
    return y.to(torch.bfloat16)


@pytest.mark.parametrize("inter,down_groups,k_off", [(I, 2, 0), (I, 2, 64)])
def test_matches_reference(inter, down_groups, k_off):
    from freetoken.moe.wna16_cpu_prefill import Wna16CpuPrefill

    g = torch.Generator().manual_seed(0)
    banks = _banks(g, inter, down_groups)
    T, top_k = 9, 3
    x = (torch.randn(T, H, generator=g) * 0.5).to(torch.bfloat16)
    ids = torch.randint(0, E, (T, top_k), generator=g, dtype=torch.int32)
    ids[:, 0] = 2                     # one expert serves every token (> one 4-token tile)
    ids[3] = -1                       # a token with no CPU routes -> zero row
    ids[5, 1] = -1                    # a skipped route
    w = torch.rand(T, top_k, generator=g)
    ex = Wna16CpuPrefill(hidden=H, inter=inter, down_groups=down_groups, k_off=k_off, num_experts=E,
                         top_k=top_k, max_tokens=16, threads=3, cpus=[])
    ex.set_layer(0, banks)
    y = torch.empty(T, H, dtype=torch.bfloat16)
    n = ex.forward(0, x, ids, w, y)
    assert n == len(set(ids[ids >= 0].tolist()))
    ref = _reference(banks, x, ids, w, k_off)
    assert torch.all(y[3] == 0)
    err = (y.float() - ref.float()).abs().max() / ref.float().abs().max()
    assert err < 2e-2, err


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hybrid_routed_prefill_matches_full_gpu_moe():
    """OffloadMoELayer._prefill_cpu_hybrid (hits + fetched misses on the GPU, the other misses on
    the CPU) against the plain grouped GEMM over every expert, across calls that hit, fetch,
    evict and spill to the CPU in a cache smaller than the two layers, with a slow fetch."""
    from types import SimpleNamespace

    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.layers.quantization.moe.base import BankSpec
    from freetoken.moe.fused_wna16 import fused_experts_wna16
    from freetoken.moe.offload_cache import OffloadMoeCache
    from freetoken.moe.wna16_cpu_prefill import Wna16CpuPrefill

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    E, L, top_k, inter = 16, 2, 4, 128
    g = torch.Generator().manual_seed(0)
    per_layer = [_banks(g, inter, 1, E) for _ in range(L)]
    layout = {role: BankSpec(tuple(t.shape[1:]), t.dtype) for role, t in per_layer[0].items()}
    cache = OffloadMoeCache(num_layers=L, num_experts=E, cache_size=20, device=torch.device("cuda"),
                            layout=layout)
    cache.set_bank_sources({role: [b[role].pin_memory() for b in per_layer] for role in layout})
    cpu = Wna16CpuPrefill(hidden=H, inter=inter, down_groups=1, k_off=0, num_experts=E, top_k=top_k,
                          max_tokens=64, threads=2, cpus=[])
    for layer, b in enumerate(per_layer):
        cpu.set_layer(layer, {role: cache.bank_sources[role][layer] for role in layout})
    cache.cpu_prefill, cache.cpu_prefill_fetch = cpu, 0.5
    # a slow PCIe fetch: any GEMM that reads a just-fetched slot before the copy lands sees stale rows
    copy_missing = cache.copy_missing
    cache.copy_missing = lambda: (torch.cuda._sleep(50_000_000), copy_missing())

    for step, (layer, T) in enumerate([(0, 9), (0, 20), (1, 33), (0, 5), (1, 12), (0, 40)]):
        stub = SimpleNamespace(layer_id=layer, activation="silu", apply_router_weight_on_input=False,
                               alpha=1.0, limit=None)
        x = (torch.randn(T, H, generator=g) * 0.5).to(torch.bfloat16).cuda()
        ids = torch.stack([torch.randperm(E, generator=g)[:top_k] for _ in range(T)]).to(torch.int32).cuda()
        w = torch.rand(T, top_k, generator=g).cuda()
        out = OffloadMoELayer._prefill_cpu_hybrid(stub, cache, x, w, ids)
        full = [cache.bank_sources[role][layer].cuda() for role in layout]
        ref = fused_experts_wna16(x, *full, w, ids, E, "silu", False, 1.0, float("inf"))
        err = (out.float() - ref.float()).abs().max() / ref.float().abs().max()
        assert err < 2e-2, (step, err)
