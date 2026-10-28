# moe/fused_nvfp4_tile.py: the RTX 3080-tuned NVFP4 expert kernels against exact NVFP4 math
# (weights dequantized in fp32, SiLU(gate) * up in fp32, the activation rounded to bf16 as the
# kernels store it).
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

H, I, TOPK = 512, 128, 4


def _banks(S, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ru8 = lambda *s: torch.randint(0, 256, s, dtype=torch.uint8, device="cuda", generator=g)  # noqa: E731
    sc = lambda *s: torch.randint(80, 118, s, dtype=torch.uint8, device="cuda", generator=g).view(torch.float8_e4m3fn)  # noqa: E731
    gl = lambda *s: (torch.rand(s, device="cuda", generator=g) * 1e-4 + 2.5e-5).float()  # noqa: E731
    return (ru8(S, 2 * I, H // 2), sc(S, 2 * I, H // 16), gl(S, 2 * I), ru8(S, H, I // 2), sc(S, H, I // 16), gl(S, H))


def _w(packed, scale, glob):
    from freetoken.models.checkpoint_index import dequantize_nvfp4

    return dequantize_nvfp4(packed.cpu(), scale.cpu(), None).float().cuda() * glob[:, None]


def _reference(x, bk, tw, ids):
    gup, gus, gug, dnp, dns, dng = bk
    out = torch.zeros(x.shape[0], H, device="cuda")
    for t in range(x.shape[0]):
        for k in range(ids.shape[1]):
            e = int(ids[t, k])
            h = _w(gup[e], gus[e], gug[e]) @ x[t].float()
            a = (torch.nn.functional.silu(h[:I]) * h[I:]).to(torch.bfloat16).float()
            out[t] += tw[t, k] * (_w(dnp[e], dns[e], dng[e]) @ a)
    return out


def _rel(a, b):
    return ((a.float() - b).pow(2).mean() / b.pow(2).mean()).sqrt().item()


def test_decode_matches_exact_math():
    from freetoken.moe.fused_nvfp4_tile import decode_supported, nvfp4_decode_silu

    bk = _banks(12)
    x = (torch.randn(2, H, device="cuda") * 0.5).to(torch.bfloat16)
    tw = torch.softmax(torch.randn(2, TOPK, device="cuda"), -1)
    ids = torch.stack([torch.randperm(12, device="cuda")[:TOPK] for _ in range(2)]).to(torch.int32)
    assert decode_supported(bk[0], bk[3])
    assert _rel(nvfp4_decode_silu(x, *bk, tw, ids), _reference(x, bk, tw, ids)) < 4e-3


@pytest.mark.parametrize("in_flight", [0, 1, 3, 8])
def test_decode_split_around_the_copy_is_bit_identical(in_flight):
    """Resident routes before the join, copied-in routes after it: the same bits as one pass,
    whichever routes the copy is still filling (none, some, all)."""
    from freetoken.moe.fused_nvfp4_tile import nvfp4_decode_silu

    bk = _banks(16, seed=3)
    x = (torch.randn(2, H, device="cuda") * 0.5).to(torch.bfloat16)
    tw = torch.softmax(torch.randn(2, TOPK, device="cuda"), -1)
    ids = torch.randperm(16, device="cuda")[: 2 * TOPK].view(2, TOPK).to(torch.int32)
    copy_slots = torch.full((16,), -1, dtype=torch.int32, device="cuda")
    copy_slots[:in_flight] = ids.view(-1)[torch.randperm(2 * TOPK, device="cuda")[:in_flight]]
    copy_slots[in_flight : in_flight + 2] = 15  # stale entries past num_copy are never read
    num_copy = torch.tensor([in_flight], dtype=torch.int64, device="cuda")
    joins = []
    got = nvfp4_decode_silu(x, *bk, tw, ids, copy=(copy_slots, num_copy), join=lambda: joins.append(1))
    torch.testing.assert_close(got, nvfp4_decode_silu(x, *bk, tw, ids), rtol=0, atol=0)
    assert joins == [1]


@pytest.mark.parametrize("fuse_act", [True, False])
@pytest.mark.parametrize("nk_tile", [True, False])
@pytest.mark.parametrize("routed", [False, True])
def test_prefill_matches_exact_math(fuse_act, nk_tile, routed, monkeypatch):
    from freetoken.moe import fused_nvfp4_tile as ft

    monkeypatch.setattr(ft, "_NK_TILE", nk_tile)
    E, M = 16, 40
    bk = _banks(E, 1)
    x = (torch.randn(M, H, device="cuda") * 0.5).to(torch.bfloat16)
    tw, ids = torch.topk(torch.softmax(torch.randn(M, E, device="cuda"), -1), TOPK, dim=-1)
    tw, ids = tw / tw.sum(-1, keepdim=True), ids.to(torch.int32)
    ref = _reference(x, bk, tw, ids)
    if routed:  # slot-cache view: compact ids + a map onto the bank rows
        expert_map, compact = torch.unique(ids, return_inverse=True)
        out = ft.fused_experts_nvfp4_tile(x, *bk, tw, compact.to(torch.int32), expert_map.numel(),
                                          expert_map=expert_map, fuse_act=fuse_act)
    else:
        out = ft.fused_experts_nvfp4_tile(x, *bk, tw, ids, E, fuse_act=fuse_act)
    assert _rel(out, ref) < (4e-3 if fuse_act else 8e-3)


@pytest.mark.parametrize("copy", [False, True])
def test_decode_with_cpu_routes(copy):
    """Routes with slot -1 belong to the CPU: the GPU skips them (their slot is never read) and adds
    the CPU's fp32 partial before the one rounding, after cpu_wait()."""
    from freetoken.moe.fused_nvfp4_tile import nvfp4_decode_silu

    bk = _banks(12, seed=5)
    x = (torch.randn(2, H, device="cuda") * 0.5).to(torch.bfloat16)
    tw = torch.softmax(torch.randn(2, TOPK, device="cuda"), -1)
    ids = torch.stack([torch.randperm(12, device="cuda")[:TOPK] for _ in range(2)]).to(torch.int32)
    on_cpu = torch.zeros_like(ids, dtype=torch.bool)
    on_cpu[0, 1] = on_cpu[1, 0] = on_cpu[1, 3] = True
    cpu_part = _reference(x, bk, tw.masked_fill(~on_cpu, 0), ids)
    slots = ids.masked_fill(on_cpu, -1)
    waits = []
    kw = {}
    if copy:
        copy_slots = torch.full((16,), -1, dtype=torch.int32, device="cuda")
        copy_slots[0] = ids[0, 0]
        kw = dict(copy=(copy_slots, torch.tensor([1], dtype=torch.int64, device="cuda")), join=lambda: None)
    got = nvfp4_decode_silu(x, *bk, tw, slots, cpu_y=cpu_part.cpu().pin_memory(),
                            cpu_wait=lambda: waits.append(1), **kw)
    assert waits == [1]
    assert _rel(got, _reference(x, bk, tw, ids)) < 4e-3
    # GPU part alone (CPU partial 0) == the GPU kernel over only the GPU routes
    alone = nvfp4_decode_silu(x, *bk, tw, slots, cpu_y=torch.zeros(2, H, device="cuda"), cpu_wait=lambda: None)
    torch.testing.assert_close(alone, nvfp4_decode_silu(x, *bk, tw.masked_fill(on_cpu, 0), ids), rtol=0, atol=0)
