# Copyright (c) 2026 FreeToken contributors
# kernel/fla/fused_sigmoid_gating_recurrent.py: the value-tile width only splits the work, so
# decode, MTP verify and commit stay bit-identical to the vendored 32-wide tile.
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("mode", ["decode", "verify", "commit"])
def test_block_v_is_bit_identical(mode):
    from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update

    torch.manual_seed(0)
    n, t, Hk, Hv, K = 2, (1 if mode == "decode" else 4), 4, 12, 128
    q, k = (torch.randn(n, t, Hk, K, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    v = torch.randn(n, t, Hv, K, device="cuda", dtype=torch.bfloat16)
    a, b = (torch.randn(n, t, Hv, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    A_log, dt = torch.randn(Hv, device="cuda"), torch.randn(Hv, device="cuda")
    states = torch.randn(5, Hv, K, K, device="cuda") * 0.05
    idx = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    extra = {"verify": {"disable_state_update": True},
             "commit": {"num_steps": torch.tensor([2, 4], device="cuda", dtype=torch.int32)}}.get(mode, {})
    outs = []
    for bv in (32, 8):
        st = states.clone()
        o = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log, a=a, dt_bias=dt, softplus_beta=1.0, softplus_threshold=20.0,
            q=q, k=k, v=v, b=b, initial_state_source=st, initial_state_indices=idx,
            scale=K ** -0.5, use_qk_l2norm_in_kernel=True, block_v=bv, **extra,
        )
        outs.append((o, st))
    assert torch.equal(outs[0][0], outs[1][0])
    assert torch.equal(outs[0][1], outs[1][1])
    if mode == "verify":
        assert torch.equal(outs[1][1], states)
    else:
        assert not torch.equal(outs[1][1][idx.long()], states[idx.long()])
