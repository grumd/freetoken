# Copyright (c) 2026 FreeToken contributors
# moe/nvfp4_quant.py: load-time RTN NVFP4 of bf16 experts (--mtp-experts-fp4).
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="NVFP4 kernels need CUDA")


def test_dequant_reproduces_the_rounded_weights():
    from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4
    from freetoken.moe.nvfp4_quant import quantize_nvfp4

    torch.manual_seed(0)
    w = (torch.randn(5, 64, 256, device="cuda") * 0.02).to(torch.bfloat16)
    w[1, 3] = 0  # an all-zero row keeps a finite global and decodes to zeros
    packed, scale, glob = quantize_nvfp4(w, experts_per_chunk=2)
    back = dequant_nvfp4(packed, scale, glob, torch.arange(5, dtype=torch.int32, device="cuda"),
                         dtype=torch.float32)
    assert torch.all(back[1, 3] == 0)
    err = (back - w.float()).norm() / w.float().norm()
    assert err < 0.12, err  # E2M1 with per-16 e4m3 scales: ~9% relative RMS on gaussian rows
    # every block's largest magnitude is on the grid's top point, so it survives nearly exactly
    blk = w.float().view(5, 64, 16, 16).abs().amax(-1)
    bblk = back.view(5, 64, 16, 16).abs().amax(-1)
    assert torch.allclose(bblk, blk, rtol=0.07, atol=1e-6)


def test_decode_gemv_on_quantized_banks_matches_the_dequantized_reference():
    from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4
    from freetoken.moe.fused_nvfp4_tile import nvfp4_decode_silu
    from freetoken.moe.nvfp4_quant import quantize_nvfp4

    torch.manual_seed(1)
    E, H, I, K = 16, 512, 128, 4
    gu = (torch.randn(E, 2 * I, H, device="cuda") * 0.03).to(torch.bfloat16)
    dn = (torch.randn(E, H, I, device="cuda") * 0.03).to(torch.bfloat16)
    banks = (*quantize_nvfp4(gu), *quantize_nvfp4(dn))
    x = torch.randn(2, H, device="cuda").to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device="cuda")[:K] for _ in range(2)]).to(torch.int32)
    tw = torch.softmax(torch.randn(2, K, device="cuda"), -1)
    out = nvfp4_decode_silu(x, *banks, tw, ids)
    slots = torch.arange(E, dtype=torch.int32, device="cuda")
    gu_d = dequant_nvfp4(*banks[:3], slots, dtype=torch.float32)
    dn_d = dequant_nvfp4(*banks[3:], slots, dtype=torch.float32)
    ref = torch.zeros(2, H, device="cuda")
    for t in range(2):
        for j in range(K):
            e = int(ids[t, j])
            h = gu_d[e] @ x[t].float()
            a = (torch.nn.functional.silu(h[:I]) * h[I:]).to(torch.bfloat16).float()
            ref[t] += tw[t, j] * (dn_d[e] @ a)
    assert torch.allclose(out.float(), ref, rtol=2e-2, atol=2e-3), (out.float() - ref).abs().max()
