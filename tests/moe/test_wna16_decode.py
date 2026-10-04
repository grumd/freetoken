# Copyright (c) 2026 FreeToken contributors
# moe/fused_wna16.py decode path (route-parallel INT4 GEMM, split-K) and prefill path
# (tile-dequant grouped GEMM, plus the legacy per-nibble kernel) against a torch dequant
# reference of the whole routed MoE.
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

G = 128


def _pack(codes: torch.Tensor) -> torch.Tensor:
    """[..., R*8, C] int4 codes -> [..., R, C] int32 (code j of a word -> row 8w+j)."""
    *lead, rows, cols = codes.shape
    c = codes.view(*lead, rows // 8, 8, cols).to(torch.int64)
    word = sum(c[..., j, :] << (4 * j) for j in range(8))
    return torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)  # two's complement


def _expert_bank(S, N, K, dev):
    codes = torch.randint(0, 16, (S, K, N), device=dev)
    zeros = torch.randint(0, 15, (S, K // G, N), device=dev)  # stored zero-point - 1
    scales = (torch.rand(S, K // G, N, device=dev) * 0.02 + 0.001).half()
    qw = _pack(codes)
    qz = _pack(zeros.transpose(1, 2)).transpose(1, 2).contiguous()  # [S, K//G, N//8]
    w = (codes - (zeros + 1).repeat_interleave(G, dim=1)).float() * scales.float().repeat_interleave(G, dim=1)
    return qw, qz, scales, w.transpose(1, 2)  # dequant [S, N, K]


@pytest.mark.parametrize("M", [1, 4])
@pytest.mark.parametrize("split", [None, 1])
@pytest.mark.parametrize("v2", [True, False])
def test_decode_matches_dequant_reference(M, split, v2, monkeypatch):
    """Both decode paths: the fused-epilogue one (kernel/triton/wna16_decode.py, default)
    and the per-op split-K one it replaces (FREETOKEN_WNA16_DECODE_V2=0)."""
    from freetoken.moe import fused_wna16

    monkeypatch.setattr(fused_wna16, "_DECODE_V2", v2)
    if split == 1:  # pin the pre-split tiles to cover the SPLIT_K=1 path too
        monkeypatch.setattr(fused_wna16, "_decode_config", lambda r, n, k: dict(
            BLOCK_SIZE_N=64, BLOCK_SIZE_KW=8, num_warps=1, num_stages=2, SPLIT_K=1))
    torch.manual_seed(0)
    dev = "cuda"
    S, H, I, TOPK = 16, 512, 256, 4
    gu_qw, gu_qz, gu_sc, gu_w = _expert_bank(S, 2 * I, H, dev)
    dn_qw, dn_qz, dn_sc, dn_w = _expert_bank(S, H, I, dev)
    x = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(S, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
    tw = torch.softmax(torch.randn(M, TOPK, device=dev), dim=-1)

    out = fused_wna16.fused_experts_decode_wna16(
        x.clone(), gu_qw, gu_qz, gu_sc, dn_qw, dn_qz, dn_sc, tw, ids
    ).float()

    ref = torch.zeros(M, H, device=dev)
    for m in range(M):
        for j in range(TOPK):
            e = int(ids[m, j])
            h = gu_w[e] @ x[m].float()
            act = torch.nn.functional.silu(h[:I]) * h[I:]
            ref[m] += tw[m, j] * (dn_w[e] @ act.to(torch.bfloat16).float())
    rel = (out - ref).abs().max() / ref.abs().max()
    assert rel.item() < 2e-2, rel.item()


@pytest.mark.parametrize("M", [32, 300, 4096])
@pytest.mark.parametrize("legacy,precise", [(False, False), (True, False), (False, True)])
def test_prefill_matches_dequant_reference(M, legacy, precise, monkeypatch):
    from freetoken.moe import fused_wna16

    monkeypatch.setenv("FREETOKEN_WNA16_PREFILL_LEGACY", "1" if legacy else "")
    monkeypatch.setenv("FREETOKEN_WNA16_PREFILL_PRECISE", "1" if precise else "")
    torch.manual_seed(0)
    dev = "cuda"
    S, H, I, TOPK = 16, 512, 384, 4  # I=384: the rank-1 band of Flash-Next at TP=2
    gu_qw, gu_qz, gu_sc, gu_w = _expert_bank(S, 2 * I, H, dev)
    dn_qw, dn_qz, dn_sc, dn_w = _expert_bank(S, H, I, dev)
    x = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(S, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
    tw = torch.softmax(torch.randn(M, TOPK, device=dev), dim=-1)

    out = fused_wna16.fused_experts_wna16(
        x.clone(), gu_qw, gu_qz, gu_sc, dn_qw, dn_qz, dn_sc, tw, ids, S
    ).float()

    h = torch.einsum("snk,mk->msn", gu_w, x.float())  # [M, S, 2I]
    act = (torch.nn.functional.silu(h[..., :I]) * h[..., I:]).to(torch.bfloat16).float()
    y = torch.einsum("snk,msk->msn", dn_w, act)  # [M, S, H]
    ref = (y.gather(1, ids.long()[..., None].expand(-1, -1, H)) * tw[..., None]).sum(1)
    rel = (out - ref).abs().max() / ref.abs().max()
    assert rel.item() < 2e-2, rel.item()


@pytest.mark.parametrize("M", [32, 300])
def test_prefill_expert_map_matches_slot_ids(M):
    """Routed-only prefill over the slot cache aligns on compact ids and maps blocks back
    to their slot rows; the result equals aligning on the raw slot ids directly."""
    from freetoken.moe import fused_wna16

    torch.manual_seed(0)
    dev = "cuda"
    S, H, I, TOPK = 48, 512, 384, 4
    gu_qw, gu_qz, gu_sc, _ = _expert_bank(S, 2 * I, H, dev)
    dn_qw, dn_qz, dn_sc, _ = _expert_bank(S, H, I, dev)
    banks = (gu_qw, gu_qz, gu_sc, dn_qw, dn_qz, dn_sc)
    x = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
    resident = torch.randperm(S, device=dev)[:12]  # the slots this layer's experts occupy
    ids = torch.stack([resident[torch.randperm(12, device=dev)[:TOPK]] for _ in range(M)]).to(torch.int32)
    tw = torch.softmax(torch.randn(M, TOPK, device=dev), dim=-1)

    direct = fused_wna16.fused_experts_wna16(x.clone(), *banks, tw, ids, S)
    expert_map, compact = torch.unique(ids, return_inverse=True)
    mapped = fused_wna16.fused_experts_wna16(
        x.clone(), *banks, tw, compact.to(torch.int32), expert_map.numel(), expert_map=expert_map
    )
    torch.testing.assert_close(mapped, direct, rtol=0, atol=0)


@pytest.mark.parametrize("prefill,precise", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("v2", [True, False])
def test_balanced_tp_bands_sum_to_full(prefill, precise, v2, monkeypatch):
    """FREETOKEN_WNA16_BALANCED_TP: two mid-group bands (rank 1 starts 64 rows into a
    group) sliced like the HF loader, each run with its own down K offset, sum to the
    unsplit MoE."""
    from freetoken.models import wna16_banks
    from freetoken.moe import fused_wna16

    monkeypatch.setenv("FREETOKEN_WNA16_BALANCED_TP", "1")
    monkeypatch.setenv("FREETOKEN_WNA16_PREFILL_PRECISE", "1" if precise else "")
    monkeypatch.setattr(fused_wna16, "_DECODE_V2", v2)
    torch.manual_seed(0)
    dev = "cuda"
    S, H, I, TOPK, TP = 16, 512, 384, 4, 2  # bands of 192: [0,192) and [192,384)
    M = 300 if prefill else 4
    gu_qw, gu_qz, gu_sc, gu_w = _expert_bank(S, 2 * I, H, dev)
    dn_qw, dn_qz, dn_sc, dn_w = _expert_bank(S, H, I, dev)
    x = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(S, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
    tw = torch.softmax(torch.randn(M, TOPK, device=dev), dim=-1)

    spec = type("Spec", (), {"group_size": G})()
    out = torch.zeros(M, H, device=dev)
    for rank in range(TP):
        lo, hi, g_lo, g_hi = wna16_banks.wna16_tp_bands(I, TP, rank)
        assert (hi - lo, g_hi - g_lo) == (192, 2)
        sl = lambda role, t: wna16_banks._tp_slice_piece(spec, role, t, lo, hi)  # noqa: E731
        # gate_up is fused [gate | up] on N: slice each half like the loader does per piece
        gate_up = [torch.cat([sl(r, t[..., :n]), sl(r, t[..., n:])], dim=-1).contiguous()
                   for r, t, n in (("gate", gu_qw, I), ("gate_zero", gu_qz, I // 8), ("gate_scale", gu_sc, I))]
        down = [sl(r, t.transpose(0, 1)).transpose(0, 1).contiguous()
                for r, t in (("down", dn_qw), ("down_zero", dn_qz), ("down_scale", dn_sc))]
        monkeypatch.setattr("freetoken.distributed.get_tp_info",
                            lambda r=rank: type("TP", (), {"rank": r, "size": TP})())
        k_off = wna16_banks.wna16_down_k_off(hi - lo)
        assert k_off == (lo % G)
        fn = fused_wna16.fused_experts_wna16 if prefill else fused_wna16.fused_experts_decode_wna16
        extra = (S,) if prefill else ()
        out += fn(x.clone(), *gate_up, *down, tw, ids, *extra, down_k_off=k_off).float()

    h = torch.einsum("snk,mk->msn", gu_w, x.float())
    act = (torch.nn.functional.silu(h[..., :I]) * h[..., I:]).to(torch.bfloat16).float()
    y = torch.einsum("snk,msk->msn", dn_w, act)
    ref = (y.gather(1, ids.long()[..., None].expand(-1, -1, H)) * tw[..., None]).sum(1)
    rel = (out - ref).abs().max() / ref.abs().max()
    assert rel.item() < 2e-2, rel.item()


@pytest.mark.parametrize("M", [32, 300, 4096])
def test_prefill_precise_rounds_only_the_activation_and_outputs(M, monkeypatch):
    """FREETOKEN_WNA16_PREFILL_PRECISE (exact dequant, fused fp32 gate/up + SiLU) equals a reference
    that rounds to bf16 only where that path still stores bf16 -- the activation, each route's
    output and the sum -- and lands well below the default path's error against exact W4 math."""
    from freetoken.moe import fused_wna16

    torch.manual_seed(0)
    dev = "cuda"
    S, H, I, TOPK = 16, 512, 384, 4
    gu_qw, gu_qz, gu_sc, gu_w = _expert_bank(S, 2 * I, H, dev)
    dn_qw, dn_qz, dn_sc, dn_w = _expert_bank(S, H, I, dev)
    banks = (gu_qw, gu_qz, gu_sc, dn_qw, dn_qz, dn_sc)
    x = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(S, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
    tw = torch.softmax(torch.randn(M, TOPK, device=dev), dim=-1)

    def run(precise):
        monkeypatch.setenv("FREETOKEN_WNA16_PREFILL_PRECISE", "1" if precise else "")
        return fused_wna16.fused_experts_wna16(x.clone(), *banks, tw, ids, S).double()

    bf = lambda t: t.to(torch.bfloat16).double()  # noqa: E731
    gu = torch.einsum("snk,mk->msn", gu_w.double(), x.double())
    act = torch.nn.functional.silu(gu[..., :I]) * gu[..., I:]
    pick = ids.long()[..., None].expand(-1, -1, H)
    w = tw.double()[..., None]
    exact = (torch.einsum("snk,msk->msn", dn_w.double(), act).gather(1, pick) * w).sum(1)
    stored = bf((bf(torch.einsum("snk,msk->msn", dn_w.double(), bf(act)).gather(1, pick) * w)).sum(1))
    rel = lambda a, b: ((a - b).norm() / b.norm()).item()  # noqa: E731
    default, precise = run(False), run(True)
    assert rel(precise, stored) < 5e-4, rel(precise, stored)
    assert rel(precise, exact) < 0.7 * rel(default, exact), (rel(precise, exact), rel(default, exact))
