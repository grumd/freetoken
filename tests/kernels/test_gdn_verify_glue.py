# Copyright (c) 2026 FreeToken contributors
# kernel/triton/gdn_verify_glue.py: the fused MTP verify staging and commit conv-window copies
# equal the torch ops they replace, bit for bit, on row-strided inputs.
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("n,t,C,HV", [(1, 4, 5120, 24), (3, 3, 1000, 7)])
def test_stage_and_commit_match_torch(n, t, C, HV):
    from freetoken.kernel.triton.gdn_verify_glue import commit_conv, stage_verify

    torch.manual_seed(0)
    km1, slots_total = 3, 6
    proj = torch.randn(n * t, C + 2 * HV + 9, device="cuda", dtype=torch.bfloat16)  # fused projection
    conv_in, a, b = proj[:, :C], proj[:, C : C + HV], proj[:, C + HV + 3 : C + 2 * HV + 3]
    conv_state = torch.randn(slots_total, C, km1, device="cuda", dtype=torch.bfloat16)
    slots = torch.tensor([4, 0, 2][:n], device="cuda", dtype=torch.int32)
    win = torch.zeros(n + 1, C, km1 + t, device="cuda", dtype=torch.bfloat16)
    mixed = torch.zeros(n + 1, t, C, device="cuda", dtype=torch.bfloat16)
    abuf = torch.zeros(n + 1, t, HV, device="cuda", dtype=torch.bfloat16)
    bbuf = torch.zeros_like(abuf)
    stage_verify(conv_in, a, b, conv_state, slots, win, mixed[:n], abuf, bbuf, n, t)

    raw = conv_in.reshape(n, t, -1)
    ref_win = torch.cat([conv_state[slots.long()], raw.transpose(1, 2)], dim=2)
    assert torch.equal(win[:n], ref_win) and not win[n:].any()
    assert torch.equal(mixed[:n], raw) and not mixed[n:].any()
    assert torch.equal(abuf[:n], a.reshape(n, t, -1)) and torch.equal(bbuf[:n], b.reshape(n, t, -1))

    lens = torch.tensor([1, t, 2][:n], device="cuda", dtype=torch.int32)
    ref = conv_state.clone()
    cols = lens.long().view(n, 1, 1) + torch.arange(km1, device="cuda").view(1, 1, km1)
    ref.index_copy_(0, slots.long(), win[:n].gather(2, cols.expand(n, C, km1)))
    commit_conv(win[:n], lens, slots, conv_state, n)
    assert torch.equal(conv_state, ref)
