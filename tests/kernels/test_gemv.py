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
