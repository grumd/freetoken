"""CPU NVFP4 prefill experts (moe/wna16_cpu_prefill.Nvfp4CpuPrefill, kernel/csrc/cpu_moe/nvfp4_cpu.h)
against exact NVFP4 math: weights dequantized in fp32, gate/up in fp32, the activation rounded
to bf16 as the executor stores it, the routed sum in fp32."""
from __future__ import annotations

import pytest
import torch

from freetoken.moe.wna16_cpu_prefill import cpu_supported

pytestmark = pytest.mark.skipif(not cpu_supported(), reason="needs AVX-512 BF16")

H, I, E, TOPK = 512, 192, 6, 4


def _banks(g):
    u8 = lambda *s: torch.randint(0, 256, s, generator=g, dtype=torch.uint8)  # noqa: E731
    sc = lambda *s: torch.randint(60, 124, s, generator=g, dtype=torch.uint8).view(torch.float8_e4m3fn)  # noqa: E731
    gl = lambda *s: torch.rand(s, generator=g) * 1e-4 + 2.5e-5  # noqa: E731
    return {
        "gate_up": u8(E, 2 * I, H // 2), "gate_up_scale": sc(E, 2 * I, H // 16), "gate_up_global": gl(E, 2 * I),
        "down": u8(E, H, I // 2), "down_scale": sc(E, H, I // 16), "down_global": gl(E, H),
    }


def _reference(b, x, ids, w):
    from freetoken.models.checkpoint_index import dequantize_nvfp4

    deq = lambda p, s, g: dequantize_nvfp4(p, s, None).float() * g[:, None]  # noqa: E731
    y = torch.zeros(x.shape[0], H)
    for t in range(x.shape[0]):
        for k in range(ids.shape[1]):
            e = int(ids[t, k])
            if e < 0:
                continue
            h = deq(b["gate_up"][e], b["gate_up_scale"][e], b["gate_up_global"][e]) @ x[t].float()
            a = (torch.nn.functional.silu(h[:I]) * h[I:]).to(torch.bfloat16).float()
            y[t] += w[t, k] * (deq(b["down"][e], b["down_scale"][e], b["down_global"][e]) @ a)
    return y


@pytest.mark.parametrize("T", [1, 5, 23])
def test_nvfp4_cpu_prefill_matches_exact_math(T):
    from freetoken.moe.wna16_cpu_prefill import Nvfp4CpuPrefill

    g = torch.Generator().manual_seed(T)
    b = _banks(g)
    ex = Nvfp4CpuPrefill(hidden=H, inter=I, num_experts=E, top_k=TOPK, max_tokens=32, threads=3, cpus=[])
    ex.set_layer(0, b)
    x = (torch.randn(T, H, generator=g) * 0.5).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(T)]).to(torch.int32)
    ids[0, 1] = -1  # a route the GPU computes
    w = torch.rand(T, TOPK, generator=g)
    y = torch.empty(T, H)
    n = ex.forward(0, x, ids, w, y)
    assert n == len(set(ids[ids >= 0].tolist()))
    ref = _reference(b, x, ids, w)
    rel = ((y - ref).pow(2).mean() / ref.pow(2).mean()).sqrt().item()
    assert rel < 2e-3, rel
