# Copyright (c) 2026 FreeToken contributors
# kernel/triton/gemv.py: the single-token bf16 GEMV the unquantized linear kernel runs at
# M=1, and its three-matrix variant, against an fp32 reference.
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# every tile rule branch: wide K, K <= 512 (wide and narrow N), K % 1024 == 0, the rest,
# plus ragged N and K
SHAPES = [(336, 10240), (10240, 320), (2560, 320), (2560, 3072), (8240, 2560), (513, 2560), (100, 1000)]


@pytest.mark.parametrize("N,K", SHAPES)
def test_gemv_matches_fp32_reference(N, K):
    from freetoken.kernel.triton.gemv import gemv

    torch.manual_seed(0)
    x = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    ref = F.linear(x.float(), w.float())
    out = gemv(x, w)
    assert out.shape == (1, N) and out.dtype == torch.bfloat16
    rel = (out.float() - ref).abs().max() / ref.abs().max()
    assert rel.item() < 1e-2, rel.item()


def test_gemv_reads_strided_rows():
    """A row-sliced weight view (stride(0) > K) is read in place."""
    from freetoken.kernel.triton.gemv import gemv

    x = torch.randn(1, 2560, device="cuda", dtype=torch.bfloat16)
    big = torch.randn(512, 4096, device="cuda", dtype=torch.bfloat16)
    w = big[:, :2560]
    torch.testing.assert_close(gemv(x, w), gemv(x, w.contiguous()), rtol=0, atol=0)


def test_gemv3_equals_gemv_over_concat():
    from freetoken.kernel.triton.gemv import gemv, gemv3

    torch.manual_seed(0)
    x = torch.randn(1, 2560, device="cuda", dtype=torch.bfloat16)
    ws = [torch.randn(n, 2560, device="cuda", dtype=torch.bfloat16) * 0.02 for n in (512, 640, 1)]
    torch.testing.assert_close(gemv3(x, *ws), gemv(x, torch.cat(ws)), rtol=0, atol=0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gemv3_split_router_first_is_bit_identical(seed):
    """Qwen4ExpMoE's early-router decode at the q38 TP=2 shapes: the router rows alone, then gemv3
    over the rest with an empty middle matrix, both into the front's row, reproduce the one-launch
    gemv3 bit for bit."""
    from freetoken.kernel.triton.gemv import gemv, gemv3

    torch.manual_seed(seed)
    x = torch.randn(1, 2560, device="cuda", dtype=torch.bfloat16)
    r, gu, g = (torch.randn(n, 2560, device="cuda", dtype=torch.bfloat16) * 0.02 for n in (512, 640, 1))
    full = gemv3(x, r, gu, g)
    front = torch.empty_like(full)
    gemv(x, r, out=front[:, :512])
    gemv3(x, gu, gu[:0], g, out=front[:, 512:])
    torch.testing.assert_close(front, full, rtol=0, atol=0)


def test_linear_kernel_takes_gemv_only_for_one_bf16_row(monkeypatch):
    from freetoken.layers.quantization.linear import unquantized

    calls = []
    from freetoken.kernel.triton import gemv as gemv_mod

    real = gemv_mod.gemv
    monkeypatch.setattr(gemv_mod, "gemv", lambda x, w: calls.append(x.shape) or real(x, w))
    layer = type("L", (), {})()
    layer.weight = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    layer.bias = None
    k = unquantized.TorchLinearKernel()
    for m in (1, 3):
        x = torch.randn(m, 256, device="cuda", dtype=torch.bfloat16)
        out = k.apply(layer, x)
        torch.testing.assert_close(out.float(), F.linear(x.float(), layer.weight.float()), rtol=2e-2, atol=2e-2)
    assert calls == [(1, 256)]


@pytest.mark.parametrize(
    "N,K", [(8240, 2560), (2560, 3072), (640, 2560), (2560, 320), (100, 1024), (336, 10240), (1280, 2560)]
)
def test_gemv_mxfp8_matches_dequantized_reference(N, K):
    from freetoken.kernel.triton.gemv import gemv_mxfp8

    torch.manual_seed(0)
    q = (torch.randint(0, 256, (N, K), device="cuda", dtype=torch.uint8) & 0xF7).view(torch.float8_e4m3fn)
    codes = torch.randint(110, 125, (N, K // 32), device="cuda", dtype=torch.uint8)
    w = (q.float().view(N, -1, 32) * torch.exp2(codes.float() - 127).unsqueeze(-1)).view(N, K)
    x = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
    ref = F.linear(x.float(), w)
    out = gemv_mxfp8(x, q, codes)
    rel = ((out.float() - ref).pow(2).mean() / ref.pow(2).mean()).sqrt()
    assert rel.item() < 4e-3, rel.item()


def test_mxfp8_encode_is_exact_for_mxfp8_weights_and_refuses_others():
    from freetoken.layers.quantization.linear.unquantized import mxfp8_encode_exact

    torch.manual_seed(0)
    q = (torch.randint(0, 256, (300, 512), device="cuda", dtype=torch.uint8) & 0xF7).view(torch.float8_e4m3fn)
    codes = torch.randint(100, 130, (300, 16), device="cuda", dtype=torch.uint8)
    w = (q.float().view(300, -1, 32) * torch.exp2(codes.float() - 127).unsqueeze(-1)).view(300, 512).to(torch.bfloat16)
    w[7] = 0  # all-zero blocks encode too
    enc = mxfp8_encode_exact(w, rows=128)
    assert enc is not None
    q2, c2 = enc
    back = (q2.float().view(300, -1, 32) * torch.exp2(c2.float() - 127).unsqueeze(-1)).view(300, 512).to(torch.bfloat16)
    assert torch.equal(back, w)
    assert mxfp8_encode_exact(torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)) is None


def _dequant(q, codes):
    N, K = q.shape
    return (q.float().view(N, -1, 32) * torch.exp2(codes.float() - 127).unsqueeze(-1)).view(N, K)


def test_mxfp8_encode_lossy_rounds_to_nearest_within_half_an_e4m3_step():
    """Lossy MXFP8 (--dense-mxfp8-groups): the no-clip power-of-two block scale, then e4m3
    round-to-nearest: each weight moves by at most half a step of its block's grid."""
    from freetoken.layers.quantization.linear.unquantized import mxfp8_encode

    torch.manual_seed(0)
    w = (torch.randn(300, 512, device="cuda") * torch.logspace(-6, 1, 300, device="cuda")[:, None]).to(torch.bfloat16)
    w[5, :64] = 0
    assert mxfp8_encode(w, exact=True) is None
    q, codes = mxfp8_encode(w, exact=False, rows=128)
    back = _dequant(q, codes)
    k = (codes.float() - 127).repeat_interleave(32, dim=1)
    # normals: 3 mantissa bits -> half a step is 2^-4 relative; subnormals: 2^(k-9) apart
    bound = torch.maximum(w.float().abs() * 2.0**-4, torch.exp2(k - 10))
    assert bool(((back - w.float()).abs() <= bound).all())
    assert back.abs().view(300, -1, 32).amax(-1).le(448 * torch.exp2(codes.float() - 127)).all()
    assert torch.equal(back[5, :64], torch.zeros(64, device="cuda"))
    rel = ((back - w.float()).pow(2).mean() / w.float().pow(2).mean()).sqrt().item()
    assert 0.01 < rel < 0.04, rel


def test_mxfp8_linear_dequant_slices_match_one_transient(monkeypatch):
    """Past the GEMV cap a large MXFP8 weight (an lm_head) is dequantized in row slices."""
    from freetoken.kernel.triton import mxfp8_linear as mod

    torch.manual_seed(0)
    q = (torch.randint(0, 256, (1000, 256), device="cuda", dtype=torch.uint8) & 0xF7).view(torch.float8_e4m3fn)
    codes = torch.randint(110, 125, (1000, 8), device="cuda", dtype=torch.uint8)
    x = torch.randn(300, 256, device="cuda", dtype=torch.bfloat16)
    whole = mod.mxfp8_linear(x, q, codes)
    monkeypatch.setattr(mod, "_DEQUANT_ELEMS", 256 * 96)
    sliced = mod.mxfp8_linear(x, q, codes)
    ref = F.linear(x.float(), _dequant(q, codes))

    def rel(a, b):
        return ((a.float() - b.float()).pow(2).mean() / b.float().pow(2).mean()).sqrt().item()

    assert rel(sliced, ref) < 4e-3 and rel(sliced, whole) < 4e-3


@pytest.mark.parametrize("seed", [0, 1])
def test_hc_up_gate_mix_mxfp8_matches_unfused_chain(seed):
    """The fused MXFP8 hc mix against hc_silu -> gemv_mxfp8 -> hc_gate_mix and the fp32 math."""
    from freetoken.kernel.triton.gemv import gemv_mxfp8
    from freetoken.kernel.triton.hc import hc_gate_mix, hc_silu, hc_up_gate_mix_mxfp8

    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randint(0, 256, (10240, 320), device="cuda", dtype=torch.uint8, generator=gen) & 0xF7).view(
        torch.float8_e4m3fn
    )
    codes = torch.randint(110, 120, (10240, 10), device="cuda", dtype=torch.uint8, generator=gen)
    lora = torch.randn(1, 336, device="cuda", dtype=torch.bfloat16, generator=gen)[:, :320] * 3
    rn = torch.randn(1, 10240, device="cuda", dtype=torch.bfloat16, generator=gen)
    fused = hc_up_gate_mix_mxfp8(lora, q, codes, rn, 4)
    chain = hc_gate_mix(rn, gemv_mxfp8(hc_silu(lora, 4), q, codes), 4)
    # same math, the fp32 block partials summed in another order
    torch.testing.assert_close(fused, chain, rtol=1e-2, atol=1e-3)
    a = F.silu(lora.float() / 4).to(torch.bfloat16).float()
    gate = (a @ _dequant(q, codes).t()).view(4, 2560)
    ref = (torch.sigmoid(gate) * rn.float().view(4, 2560)).mean(0, keepdim=True)
    torch.testing.assert_close(fused.float(), ref, rtol=2e-2, atol=2e-2)
